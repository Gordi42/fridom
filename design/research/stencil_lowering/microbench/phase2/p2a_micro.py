"""P2a: price divide-reduction spellings on the E3 weno5 tendency.

Reuses the parent microbench harness (common.py, e3.py) unchanged;
identical methodology (block_until_ready, 5 warmups, median of 40,
fori_loop T=50 per-iter probe, memory/cost/HLO capture).

Usage: JAX_PLATFORMS=cuda python p2a_micro.py [--n 256]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)  # common.py, e3.py
sys.path.insert(0, HERE)    # weno_variants.py

import e3 as E  # noqa: E402
from common import (  # noqa: E402
    analyze, check_device, loop_periter, time_median,
)
import weno_variants as W  # noqa: E402

RES = os.path.join(HERE, "results")
HLO = os.path.join(RES, "hlo")
os.makedirs(HLO, exist_ok=True)

_RNG = np.random.default_rng(1234)


def make(shape, dt):
    a = _RNG.standard_normal(shape).astype(
        np.float64 if dt == jnp.float64 else np.float32)
    return jax.device_put(jnp.asarray(a, dtype=dt))


def save(name, obj):
    path = os.path.join(RES, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"wrote {path}")


# ================================================================
#  HLO divide counting
# ================================================================
def count_divides(hlo: str) -> dict:
    """Count divide( ops in optimized HLO, split by result dtype."""
    out = {"total": 0, "f64": 0, "f32": 0, "other": 0}
    for m in re.finditer(r"(f64|f32|[a-z0-9]+)\[[^\]]*\][^ ]*\s+divide\(",
                         hlo):
        out["total"] += 1
        t = m.group(1)
        out[t if t in ("f64", "f32") else "other"] += 1
    out["total_all"] = len(re.findall(r"\bdivide\(", hlo))
    return out


# ================================================================
#  Item 0: raw divide vs multiply throughput reference
# ================================================================
def opcost(n: int) -> dict:
    check_device()
    dt64, dt32 = jnp.float64, jnp.float32
    x64, y64 = make((n, n, n), dt64), 1.5 + jnp.abs(make((n, n, n), dt64))
    x32, y32 = x64.astype(dt32), y64.astype(dt32)
    nbytes64 = x64.size * 8
    nbytes32 = x64.size * 4

    # (a) single elementwise op: bandwidth-bound (3 arrays touched)
    div = jax.jit(lambda a, b: a / b)
    mul = jax.jit(lambda a, b: a * b)
    div32 = jax.jit(lambda a, b: a / b)
    res = {"n": n}
    for tag, f, a, b, nb in (("div_f64", div, x64, y64, nbytes64),
                             ("mul_f64", mul, x64, y64, nbytes64),
                             ("div_f32", div32, x32, y32, nbytes32)):
        c = jax.jit(f).lower(a, b).compile()
        t = time_median(c, a, b)
        res[f"single_{tag}_ms"] = t * 1e3
        res[f"single_{tag}_gbs"] = 3 * nb / t / 1e9

    # (b) compute-bound chain: K ops UNROLLED in registers within one
    #     fused kernel (each element loaded once, K ops, stored once) so
    #     the arithmetic, not HBM, sets the time.  div divides by a
    #     distinct (d + c_k) each step (no common reciprocal -> genuine
    #     divides); mul multiplies by (1/d + c_k), one setup reciprocal
    #     amortized over K (charged to the mul side, so the div/mul
    #     ratio is a conservative lower bound).  Both stay bounded and
    #     finite (per-step factor < 1).  K = 32 stays inside the register
    #     file; K >= 64 spills the f32 Newton-Raphson divide sequence to
    #     local memory and inflates the f32 divide time ~4x (measured).
    K = 32

    def chain(op, a, b):
        def f(x, y):
            d = jnp.abs(y) + 2.0
            r = (1.0 / d).astype(x.dtype)  # 1 setup divide
            acc = x
            for k in range(K):
                c = x.dtype.type((k + 1) * 1e-3)
                acc = (acc + c) / (d + c) if op == "div" \
                    else (acc + c) * (r + c)
            return acc
        c = jax.jit(f).lower(a, b).compile()
        return time_median(c, a, b)

    for tag, a, b in (("f64", x64, y64), ("f32", x32, y32)):
        td = chain("div", a, b)
        tm = chain("mul", a, b)
        res[f"chain_div_{tag}_ms"] = td * 1e3
        res[f"chain_mul_{tag}_ms"] = tm * 1e3
        res[f"chain_div_perop_ns_{tag}"] = td / K * 1e9
        res[f"chain_mul_perop_ns_{tag}"] = tm / K * 1e9
        res[f"chain_div_over_mul_{tag}"] = td / tm
    res["K"] = K
    return res


# ================================================================
#  Item 1-5: E3 weno5 tendency variants
# ================================================================
def tendency_fn(recon, n):
    def f(u, v, w, b):
        return E.tendency_composed(u, v, w, b, recon, n)
    return f


def run_variant(name, recon, fields, n, save_hlo=True):
    """analyze + time single-call + loop probe; return metrics."""
    check_device()
    f = tendency_fn(recon, n)
    compiled, metrics, hlo = analyze(f, fields)
    t = time_median(compiled, *fields, n_timed=40)
    metrics["ms_single"] = t * 1e3
    metrics["gbs_single"] = metrics["ideal_bytes"] / t / 1e9
    metrics["divides"] = count_divides(hlo)
    metrics["name"] = name
    # loop probe: shape-preserving? tendency shrinks by 1 vs padded input.
    # Reuse the prior E4 style: no loop for tendency (shape changes).
    if save_hlo:
        p = os.path.join(HLO, f"p2a_{name}_n{n}.txt")
        with open(p, "w") as fh:
            fh.write(hlo)
        metrics["hlo_path"] = p
    return metrics


def loop_probe(name, recon, x0_pad, n):
    """fori_loop T=50 per-iter probe on the composed tendency.

    The tendency maps padded (N+2H)^3 -> interior N^3, so it is not
    shape-preserving.  We wrap it: re-pad the interior back with zeros
    each iter (constant halo) so the loop body is shape-stable.  The
    per-iter number then includes a cheap pad but isolates the kernel
    ratio between variants (all variants share the pad).
    """
    H = E.H

    def apply_once(bpad):
        u = v = w = bpad
        tend = E.tendency_composed(u, v, w, bpad, recon, n)  # N^3
        return jnp.pad(tend, H, mode="constant")  # back to (N+2H)^3

    lp = loop_periter(apply_once, x0_pad, T=50)
    lp["name"] = name
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--skip-opcost", action="store_true")
    args = ap.parse_args()
    n = args.n
    print(f"P2a micro n={n}")

    if not args.skip_opcost:
        oc = opcost(n)
        save(f"p2a_opcost_n{n}", oc)
        print(f"  single div/mul f64: {oc['single_div_f64_ms']:.3f} / "
              f"{oc['single_mul_f64_ms']:.3f} ms (bandwidth-bound)")
        print(f"  chain perop f64 div/mul: "
              f"{oc['chain_div_perop_ns_f64']:.3f} / "
              f"{oc['chain_mul_perop_ns_f64']:.3f} ns  "
              f"(div/mul = {oc['chain_div_over_mul_f64']:.2f}x)")
        print(f"  chain perop f32 div/mul: "
              f"{oc['chain_div_perop_ns_f32']:.3f} / "
              f"{oc['chain_mul_perop_ns_f32']:.3f} ns  "
              f"(div/mul = {oc['chain_div_over_mul_f32']:.2f}x)")

    # E3 fields (f64) and a parallel f32 copy for the full-f32 ceiling
    pad = n + 2 * E.H
    up, vp, wp, bp = (make((pad, pad, pad), jnp.float64) for _ in range(4))
    fields = (up, vp, wp, bp)

    results = []
    # variants over f64 fields
    for name in ("baseline", "singlediv", "f32w", "combined"):
        m = run_variant(name, W.RECON[name], fields, n)
        results.append(m)
        print(f"  {name:10s}: {m['ms_single']:.3f} ms  "
              f"flops={m['flops']/1e9:.2f}G  gbs={m['gbs_single']:.0f}  "
              f"divides={m['divides']['total']} "
              f"(f64={m['divides']['f64']} f32={m['divides']['f32']})  "
              f"temp={m['temp_bytes']/1e6:.0f}MB")

    # full-f32 ceiling: baseline spelling on f32 fields
    f32fields = tuple(x.astype(jnp.float32) for x in fields)
    m = run_variant("full_f32", W.RECON["baseline"], f32fields, n)
    results.append(m)
    print(f"  {'full_f32':10s}: {m['ms_single']:.3f} ms  "
          f"divides={m['divides']['total']} "
          f"(f64={m['divides']['f64']} f32={m['divides']['f32']})")

    # loop probes (shape-stable wrapper)
    loops = []
    for name in ("baseline", "singlediv", "f32w", "combined"):
        lp = loop_probe(name, W.RECON[name], bp, n)
        loops.append(lp)
        print(f"  loop {name:10s}: periter {lp['periter_ms']:.3f} ms")

    # ------------------------------------------------------------
    #  Validation: max rel/abs diff of the full tendency vs baseline
    # ------------------------------------------------------------
    base = jax.jit(tendency_fn(W.RECON["baseline"], n))(*fields)
    base = np.asarray(base)
    denom = np.maximum(np.abs(base), 1e-30)
    valid = {}
    for name in ("singlediv", "f32w", "combined"):
        out = np.asarray(jax.jit(tendency_fn(W.RECON[name], n))(*fields))
        rel = np.abs(out - base) / denom
        valid[name] = {
            "max_abs": float(np.max(np.abs(out - base))),
            "max_rel": float(np.max(rel)),
            "finite": bool(np.all(np.isfinite(out))),
        }
        print(f"  validate {name:10s}: max_abs={valid[name]['max_abs']:.2e} "
              f"max_rel={valid[name]['max_rel']:.2e}")

    # also validate the reconstruction weights alone (smooth field)
    save(f"p2a_tendency_n{n}",
         {"variants": results, "loops": loops, "validation": valid})


if __name__ == "__main__":
    main()
