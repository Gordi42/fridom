"""P3a: price the one-path upwind spellings on the E3 composed tendency.

Reuses the parent microbench harness (common.py, e3.py) unchanged;
identical methodology to P2a (block_until_ready, 5 warmups, median of
40, fori_loop T=50 per-iter probe, memory/cost/HLO capture).

Two schemes, two/three spellings each (all in the E3 recon signature
``recon(bp, a, n, vel)`` -> face value, so they drop straight into
``E.tendency_composed``):

  UPWIND5 (linear, no divides)
    u5_baseline    both biased rows computed, selected by sign (today)
    u5_selected    per-tap select of the union window, ONE left row
    u5_dissipation flux/v face value = c_sym.U - sign(v)*c_diss.U

  WENO5 (nonlinear weights, eps = 1e-6 as in P2a/e3)
    w5_baseline       both biased WENO recons, selected by sign (today)
    w5_selected       per-tap select, ONE left WENO recon
    w5_selected_f32w  selected-input with the weights in f32 (standard
                      per-candidate spelling)

Usage: JAX_PLATFORMS=cuda python p3a_micro.py [--n 256]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)  # common.py, e3.py

import e3 as E  # noqa: E402
from common import analyze, check_device, loop_periter, time_median  # noqa: E402

RES = os.path.join(HERE, "results")
HLO = os.path.join(RES, "hlo")
os.makedirs(HLO, exist_ok=True)

_RNG = np.random.default_rng(1234)

F32 = jnp.float32
F64 = jnp.float64

# ================================================================
#  Order-5 static rows (union window U = cells off -2..+3, len 6)
# ================================================================
#  Left biased row reads U[0..4], right reads U[1..5].
U5_LEFT = (2.0, -13.0, 47.0, 27.0, -3.0)          # /60, on U[0..4]
#  Dissipation split (linear upwind), exact:
#    flux = v*(c_sym.U) - |v|*(c_diss.U)
C_SYM = (1.0 / 60, -2.0 / 15, 37.0 / 60, 37.0 / 60, -2.0 / 15, 1.0 / 60)
C_DISS = (-1.0 / 60, 1.0 / 12, -1.0 / 6, 1.0 / 6, -1.0 / 12, 1.0 / 60)

EPS = 1e-6           # e3/P2a micro eps (NOT the real 1e-10)
C13 = 13.0 / 12.0


def _union(bp, a, n):
    """The 6 union-window cells U0..U5 (cell offsets -2..+3)."""
    return [E.face_win(bp, a, off, n) for off in (-2, -1, 0, 1, 2, 3)]


def _row5(cells):
    """U5_LEFT . cells (5 taps)."""
    c = U5_LEFT
    return (c[0] * cells[0] + c[1] * cells[1] + c[2] * cells[2]
            + c[3] * cells[3] + c[4] * cells[4]) / 60.0


# ================================================================
#  Upwind5 spellings
# ================================================================
def recon_u5_baseline(bp, a, n, vel):
    """Both biased rows, select by sign (today's spelling)."""
    U = _union(bp, a, n)
    left = _row5(U[0:5])
    right = _row5(U[5:0:-1])          # U5,U4,U3,U2,U1 -> mirror = right
    return jnp.where(vel > 0.0, left, right)


def recon_u5_selected(bp, a, n, vel):
    """Per-tap union select, ONE left row."""
    U = _union(bp, a, n)
    pos = vel > 0.0
    taps = [jnp.where(pos, U[i], U[5 - i]) for i in range(5)]
    return _row5(taps)


def recon_u5_dissipation(bp, a, n, vel):
    """face value = c_sym.U - sign(v)*c_diss.U (flux -> v*sym - |v|*diss)."""
    U = _union(bp, a, n)
    sym = sum(C_SYM[i] * U[i] for i in range(6))
    diss = sum(C_DISS[i] * U[i] for i in range(6))
    s = jnp.where(vel > 0.0, 1.0, -1.0)     # tie (v=0) -> right branch
    return sym - s * diss


# ================================================================
#  WENO5 spellings (weights over 5 cells; eps=1e-6, r=3)
# ================================================================
def _weno_left_cells(cells, weight_dtype=None):
    """Left WENO-JS from the 5 window cells [c0..c4] -> face value.

    weight_dtype=F32 runs the nonlinear-weight sub-computation in f32
    (standard per-candidate spelling) with f64 candidates.
    """
    c0, c1, c2, c3, c4 = cells
    # candidates (f64)
    p0 = (2.0 * c0 - 7.0 * c1 + 11.0 * c2) / 6.0
    p1 = (-c1 + 5.0 * c2 + 2.0 * c3) / 6.0
    p2 = (2.0 * c2 + 5.0 * c3 - c4) / 6.0
    if weight_dtype is F32:
        d0, d1, d2, d3, d4 = (c0.astype(F32), c1.astype(F32),
                              c2.astype(F32), c3.astype(F32),
                              c4.astype(F32))
        beta0 = C13 * (d0 - 2.0 * d1 + d2) ** 2 \
            + 0.25 * (d0 - 4.0 * d1 + 3.0 * d2) ** 2
        beta1 = C13 * (d1 - 2.0 * d2 + d3) ** 2 \
            + 0.25 * (d1 - d3) ** 2
        beta2 = C13 * (d2 - 2.0 * d3 + d4) ** 2 \
            + 0.25 * (3.0 * d2 - 4.0 * d3 + d4) ** 2
        eps = F32(EPS)
        a0 = F32(0.1) / (eps + beta0) ** 2
        a1 = F32(0.6) / (eps + beta1) ** 2
        a2 = F32(0.3) / (eps + beta2) ** 2
        inv = F32(1.0) / (a0 + a1 + a2)
        w0 = (a0 * inv).astype(F64)
        w1 = (a1 * inv).astype(F64)
        w2 = (a2 * inv).astype(F64)
        return w0 * p0 + w1 * p1 + w2 * p2
    beta0 = C13 * (c0 - 2.0 * c1 + c2) ** 2 \
        + 0.25 * (c0 - 4.0 * c1 + 3.0 * c2) ** 2
    beta1 = C13 * (c1 - 2.0 * c2 + c3) ** 2 \
        + 0.25 * (c1 - c3) ** 2
    beta2 = C13 * (c2 - 2.0 * c3 + c4) ** 2 \
        + 0.25 * (3.0 * c2 - 4.0 * c3 + c4) ** 2
    a0 = 0.1 / (EPS + beta0) ** 2
    a1 = 0.6 / (EPS + beta1) ** 2
    a2 = 0.3 / (EPS + beta2) ** 2
    asum = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / asum


def recon_w5_baseline(bp, a, n, vel):
    """Both biased WENO recons, select by sign (today's spelling)."""
    U = _union(bp, a, n)
    left = _weno_left_cells(U[0:5])
    right = _weno_left_cells(U[5:0:-1])       # mirror -> right recon
    return jnp.where(vel > 0.0, left, right)


def recon_w5_selected(bp, a, n, vel):
    """Per-tap union select, ONE left WENO recon."""
    U = _union(bp, a, n)
    pos = vel > 0.0
    taps = [jnp.where(pos, U[i], U[5 - i]) for i in range(5)]
    return _weno_left_cells(taps)


def recon_w5_selected_f32w(bp, a, n, vel):
    """Selected-input with the nonlinear weights in f32."""
    U = _union(bp, a, n)
    pos = vel > 0.0
    taps = [jnp.where(pos, U[i], U[5 - i]) for i in range(5)]
    return _weno_left_cells(taps, weight_dtype=F32)


RECON = {
    "u5_baseline": recon_u5_baseline,
    "u5_selected": recon_u5_selected,
    "u5_dissipation": recon_u5_dissipation,
    "w5_baseline": recon_w5_baseline,
    "w5_selected": recon_w5_selected,
    "w5_selected_f32w": recon_w5_selected_f32w,
}

UPWIND = ("u5_baseline", "u5_selected", "u5_dissipation")
WENO = ("w5_baseline", "w5_selected", "w5_selected_f32w")


# ================================================================
#  HLO divide counting (split by result dtype)
# ================================================================
def count_divides(hlo: str) -> dict:
    out = {"total": 0, "f64": 0, "f32": 0, "other": 0}
    for m in re.finditer(r"(f64|f32|[a-z0-9]+)\[[^\]]*\][^ ]*\s+divide\(",
                         hlo):
        out["total"] += 1
        t = m.group(1)
        out[t if t in ("f64", "f32") else "other"] += 1
    return out


def make(shape, dt):
    a = _RNG.standard_normal(shape).astype(
        np.float64 if dt == jnp.float64 else np.float32)
    return jax.device_put(jnp.asarray(a, dtype=dt))


def save(name, obj):
    path = os.path.join(RES, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"wrote {path}")


def tendency_fn(recon, n):
    def f(u, v, w, b):
        return E.tendency_composed(u, v, w, b, recon, n)
    return f


def run_variant(name, recon, fields, n):
    check_device()
    f = tendency_fn(recon, n)
    compiled, metrics, hlo = analyze(f, fields)
    t = time_median(compiled, *fields, n_timed=40)
    metrics["ms_single"] = t * 1e3
    metrics["gbs_single"] = metrics["ideal_bytes"] / t / 1e9
    metrics["divides"] = count_divides(hlo)
    metrics["name"] = name
    p = os.path.join(HLO, f"p3a_{name}_n{n}.txt")
    with open(p, "w") as fh:
        fh.write(hlo)
    metrics["hlo_path"] = p
    return metrics


def loop_probe(name, recon, bp, n):
    H = E.H

    def apply_once(b):
        tend = E.tendency_composed(b, b, b, b, recon, n)
        return jnp.pad(tend, H, mode="constant")

    lp = loop_periter(apply_once, bp, T=50)
    lp["name"] = name
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    n = args.n
    print(f"P3a micro n={n}")

    pad = n + 2 * E.H
    up, vp, wp, bp = (make((pad, pad, pad), jnp.float64) for _ in range(4))
    fields = (up, vp, wp, bp)

    results = []
    for name in (*UPWIND, *WENO):
        m = run_variant(name, RECON[name], fields, n)
        results.append(m)
        print(f"  {name:18s}: {m['ms_single']:.3f} ms  "
              f"flops={m['flops'] / 1e9:.2f}G  gbs={m['gbs_single']:.0f}  "
              f"div={m['divides']['total']} "
              f"(f64={m['divides']['f64']} f32={m['divides']['f32']})  "
              f"temp={m['temp_bytes'] / 1e6:.0f}MB")

    loops = []
    for name in (*UPWIND, *WENO):
        lp = loop_probe(name, RECON[name], bp, n)
        loops.append(lp)
        print(f"  loop {name:18s}: periter {lp['periter_ms']:.3f} ms")

    # numerical validation vs each scheme's own both-then-select base
    valid = {}
    for base_name, group in (("u5_baseline", UPWIND),
                             ("w5_baseline", WENO)):
        base = np.asarray(jax.jit(tendency_fn(RECON[base_name], n))(*fields))
        denom = np.maximum(np.abs(base), 1e-30)
        for name in group:
            if name == base_name:
                continue
            out = np.asarray(jax.jit(tendency_fn(RECON[name], n))(*fields))
            valid[name] = {
                "max_abs": float(np.max(np.abs(out - base))),
                "max_rel": float(np.max(np.abs(out - base) / denom)),
                "finite": bool(np.all(np.isfinite(out))),
            }
            print(f"  validate {name:18s}: "
                  f"max_abs={valid[name]['max_abs']:.2e} "
                  f"max_rel={valid[name]['max_rel']:.2e}")

    save(f"p3a_tendency_n{n}",
         {"variants": results, "loops": loops, "validation": valid})


if __name__ == "__main__":
    main()
