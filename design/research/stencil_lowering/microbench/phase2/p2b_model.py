"""P2b: real nonhydro2 model A/B for WENO divide-reduction spellings.

Matched Oceananigans-comparison config: periodic 256^3, domain
10000x10000x100, f0 = 1e-4, N^2 = (50 f0)^2, dt = 20 s, AB3, smooth
jet IC, WENOAdvection(order=5), chunk_size = 50.

The kernel A/B monkeypatches ``fridom.nonhydro2.modules.advection.
weno_reconstruct`` (the name bound in the advection module's namespace
at import; the periodic ``_biased_kernel`` resolves it as a global at
trace time).  Timing uses a FRESH process per variant (jit caches);
correctness runs all variants in one process, clearing the framework's
AOT chunk cache between them (its key is the static AssemblyRecord, not
the traced kernel, so it would otherwise reuse the stale executable).

Usage:
  JAX_PLATFORMS=cuda python p2b_model.py time --variant baseline --n 256
  JAX_PLATFORMS=cuda python p2b_model.py correct --n 256
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

# --- worktree fridom must shadow the editable install -----------
WT_SRC = "/work/uo0780/u301533/fridom/fridom-dev/.claude/worktrees/" \
         "perf+stencil-lowering/src"
sys.path.insert(0, WT_SRC)

import fridom as fr  # noqa: E402
import fridom.nonhydro2 as nh  # noqa: E402
from fridom.model import model as MM  # noqa: E402
from fridom.nonhydro2.modules import advection as ADV  # noqa: E402
from fridom.spatial.operators import weno as WENO  # noqa: E402
from fridom.spatial.operators import fallback as FALLBACK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
HLO = os.path.join(RES, "hlo")
os.makedirs(HLO, exist_ok=True)

# reuse the real static tables + window slicer + smoothness helpers
weno_tables = WENO.weno_tables
_window_views = WENO._window_views
_weighted_sum = WENO._weighted_sum
WENO_EPS = WENO.WENO_EPS
F32 = jnp.float32
F64 = jnp.float64

_PATCH_CALLS = {"n": 0}  # trace-time call counter (patch liveness)


def _assert_worktree():
    if WT_SRC not in (fr.__file__ or ""):
        msg = f"fridom is NOT the worktree copy: {fr.__file__}"
        raise RuntimeError(msg)
    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        msg = (f"expected single gpu, got {jax.default_backend()} "
               f"x{jax.device_count()}")
        raise RuntimeError(msg)


# ================================================================
#  Patched reconstruction kernels (match the real semantics)
# ================================================================
def _betas_and_candidates(windows, tables):
    """Per-candidate (beta, q) with NO division (mirrors _alpha_...)."""
    r = len(tables.optimal)
    betas, cands = [], []
    for m, offset in enumerate(tables.offsets):
        cells = windows[offset:offset + r]
        cands.append(_weighted_sum(cells, tables.coeffs[m]))
        beta = None
        for scale, diff in zip(tables.beta_scale, tables.beta_rows[m],
                               strict=True):
            square = _weighted_sum(cells, diff) ** 2
            term = square if scale == 1.0 else scale * square
            beta = term if beta is None else beta + term
        betas.append(beta)
    return betas, cands


def weno_reconstruct_singlediv(arr, axis, order=5, bias="left"):
    """Single f64 divide (exact algebra vs the standard spelling)."""
    _PATCH_CALLS["n"] += 1
    tables = weno_tables(order, bias)
    windows = _window_views(arr, axis, tables.size)
    r = len(tables.optimal)
    betas, q = _betas_and_candidates(windows, tables)
    s = [(b + WENO_EPS) ** 2 for b in betas]
    d = tables.optimal
    num = []
    for m in range(r):
        prod = None
        for k in range(r):
            if k == m:
                continue
            prod = s[k] if prod is None else prod * s[k]
        num.append(d[m] * prod)
    total = num[0]
    combined = num[0] * q[0]
    for m in range(1, r):
        total = total + num[m]
        combined = combined + num[m] * q[m]
    return combined / total


def weno_reconstruct_f32w(arr, axis, order=5, bias="left"):
    """Nonlinear weights in f32 (STANDARD per-candidate divide); f64
    candidates.

    Standard spelling on purpose: the single-divide PRODUCT form
    (n_m = d_m * prod s_k) scales like beta^4 and blows f32's ~1e+-38
    range for the real eps = 1e-10 on smooth data (s = (beta+eps)^2 ~
    1e-20 -> prod ~ 1e-40 -> reciprocal overflows -> NaN).  The
    per-candidate form alpha_m = d_m/(beta_m+eps)^2 peaks near 1e19
    (beta -> 0), safely in f32.
    """
    _PATCH_CALLS["n"] += 1
    tables = weno_tables(order, bias)
    windows = _window_views(arr, axis, tables.size)
    r = len(tables.optimal)
    # f64 candidates
    q = [_weighted_sum(windows[o:o + r], tables.coeffs[m])
         for m, o in enumerate(tables.offsets)]
    # f32 weight sub-computation (beta, per-candidate alpha, normalize)
    eps = F32(WENO_EPS)
    alpha = []
    for m, o in enumerate(tables.offsets):
        cells32 = tuple(c.astype(F32) for c in windows[o:o + r])
        beta = None
        for scale, diff in zip(tables.beta_scale, tables.beta_rows[m],
                               strict=True):
            square = _weighted_sum(cells32, diff) ** 2
            term = square if scale == 1.0 else F32(scale) * square
            beta = term if beta is None else beta + term
        alpha.append(F32(tables.optimal[m]) / (beta + eps) ** 2)
    asum = alpha[0]
    for m in range(1, r):
        asum = asum + alpha[m]
    inv = F32(1.0) / asum
    w = [(alpha[m] * inv).astype(F64) for m in range(r)]
    out = w[0] * q[0]
    for m in range(1, r):
        out = out + w[m] * q[m]
    return out


PATCHES = {
    "singlediv": weno_reconstruct_singlediv,
    "f32w": weno_reconstruct_f32w,
}


def apply_patch(variant):
    """Bind the patched kernel into every namespace holding it."""
    fn = PATCHES[variant]
    ADV.weno_reconstruct = fn
    WENO.weno_reconstruct = fn
    FALLBACK.weno_reconstruct = fn


# ================================================================
#  Model construction (matched config)
# ================================================================
def build_model(n, advection, dt=20.0):
    """Periodic n^3 nonhydro2 with a smooth jet IC."""
    lx = ly = 10000.0
    lz = 100.0
    f0 = 1e-4
    n2 = (50.0 * f0) ** 2
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, lx), periodic=True,
                                        name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, ly), periodic=True,
                                        name="y")
    mz = fr.spatial.meshes.IntervalMesh(n, (0.0, lz), periodic=True,
                                        name="z")
    grid = fr.spatial.Grid((mx, my, mz))
    model = nh.Model(
        grid=grid, dt=dt,
        coriolis=nh.FPlaneCoriolis(f0=f0),
        stratification=nh.ConstantStratification(n2=n2),
        advection=advection,
        chunk_size=50)
    kx, ky, kz = 2 * np.pi / lx, 2 * np.pi / ly, 2 * np.pi / lz
    xc = (np.arange(n) + 0.5) * (lx / n)
    yc = (np.arange(n) + 0.5) * (ly / n)
    zc = (np.arange(n) + 0.5) * (lz / n)
    x, y, z = np.meshgrid(xc, yc, zc, indexing="ij")
    # amplitude only sets how long the unbalanced jet stays finite; the
    # per-step KERNEL cost (what we time) is amplitude-independent.  A
    # gentle jet keeps every variant finite across the timed window.
    u0 = float(os.environ.get("P2B_U0", "0.2"))
    b0 = float(os.environ.get("P2B_B0", "1e-4"))
    model.set_fields(
        u=u0 * np.sin(kx * x) * np.cos(ky * y),
        v=0.3 * u0 * np.cos(kx * x),
        b=b0 * np.cos(kz * z))
    return model


def weno5(order=5):
    return nh.WENOAdvection(order=order)


# ================================================================
#  Compiled-chunk introspection (patch liveness + temp bytes)
# ================================================================
def _chunk_exe(n_steps=50):
    """The compiled length-n_steps chunk executable (or None)."""
    for key, exe in MM._CHUNK_EXECUTABLES.items():
        if key[1] == n_steps:
            return exe
    return None


def count_divides(hlo):
    out = {"total": 0, "f64": 0, "f32": 0, "c128": 0, "c64": 0}
    for m in re.finditer(
            r"(f64|f32|c128|c64|[a-z0-9]+)\[[^\]]*\][^ ]*\s+divide\(", hlo):
        out["total"] += 1
        t = m.group(1)
        out[t] = out.get(t, 0) + 1
    return out


def chunk_stats(n_steps=50):
    exe = _chunk_exe(n_steps)
    if exe is None:
        return {}
    stats = {}
    try:
        hlo = exe.as_text()
        stats["divides"] = count_divides(hlo)
    except Exception as e:  # noqa: BLE001
        stats["divides_err"] = str(e)
    try:
        mem = exe.memory_analysis()
        stats["temp_bytes"] = int(mem.temp_size_in_bytes)
        stats["peak_bytes"] = int(mem.peak_memory_in_bytes)
    except Exception as e:  # noqa: BLE001
        stats["mem_err"] = str(e)
    return stats


# ================================================================
#  Timing
# ================================================================
def _block(model):
    jax.block_until_ready(jax.tree_util.tree_leaves(model.state))


def _finite(model):
    return all(bool(np.all(np.isfinite(np.asarray(leaf))))
               for leaf in jax.tree_util.tree_leaves(model.state)
               if isinstance(leaf, jax.Array))


def time_variant(variant, n, n_chunks=6, steps=50):
    _assert_worktree()
    if variant in PATCHES:
        apply_patch(variant)
    adv = nh.CenteredAdvection() if variant == "centered" else weno5()
    model = build_model(n, adv)
    _block(model)

    # first chunk: compile + warmup (discarded)
    t0 = perf_counter()
    model.advance(steps)
    _block(model)
    first_s = perf_counter() - t0

    stats = chunk_stats(steps)
    per_step_ms = []
    for _ in range(n_chunks):
        t0 = perf_counter()
        model.advance(steps)
        _block(model)
        per_step_ms.append((perf_counter() - t0) / steps * 1e3)

    result = {
        "variant": variant, "n": n, "steps": steps,
        "n_timed_chunks": n_chunks,
        "first_chunk_compile_s": first_s,
        "ms_per_step_median": statistics.median(per_step_ms),
        "ms_per_step_min": min(per_step_ms),
        "ms_per_step_all": per_step_ms,
        "finite": _finite(model),
        "patch_trace_calls": _PATCH_CALLS["n"],
        "chunk": stats,
        "fridom_file": fr.__file__,
    }
    return result


# ================================================================
#  Correctness (single process, cache cleared between variants)
# ================================================================
def _state_arrays(model):
    out = {}
    for name in ("u", "v", "w", "b"):
        try:
            out[name] = np.asarray(model.state[name]._data)
        except Exception:  # noqa: BLE001
            pass
    return out


def _reset_kernels():
    """Restore the pristine kernel and drop the AOT chunk cache."""
    ADV.weno_reconstruct = WENO_ORIG
    WENO.weno_reconstruct = WENO_ORIG
    FALLBACK.weno_reconstruct = WENO_ORIG
    MM._CHUNK_EXECUTABLES.clear()
    MM._CHUNK_COMPILE_LOG.clear()


WENO_ORIG = WENO.weno_reconstruct


def correctness(n, steps=20):
    _assert_worktree()
    # pristine baseline
    _reset_kernels()
    base = build_model(n, weno5())
    base.advance(steps)
    _block(base)
    base_state = _state_arrays(base)
    del base

    out = {"n": n, "steps": steps, "variants": {}}
    for variant in ("singlediv", "f32w"):
        _reset_kernels()
        _PATCH_CALLS["n"] = 0
        apply_patch(variant)
        m = build_model(n, weno5())
        m.advance(steps)
        _block(m)
        st = _state_arrays(m)
        diffs = {}
        for name, arr in st.items():
            b = base_state[name]
            denom = np.maximum(np.abs(b), 1e-30)
            diffs[name] = {
                "max_abs": float(np.max(np.abs(arr - b))),
                "max_rel": float(np.max(np.abs(arr - b) / denom)),
                "finite": bool(np.all(np.isfinite(arr))),
            }
        out["variants"][variant] = {
            "diffs": diffs,
            "patch_trace_calls": _PATCH_CALLS["n"],
            "chunk_divides": chunk_stats(steps).get("divides"),
        }
        del m
    # record the pristine baseline's chunk divide profile too
    _reset_kernels()
    b2 = build_model(n, weno5())
    b2.advance(steps)
    _block(b2)
    out["baseline_chunk_divides"] = chunk_stats(steps).get("divides")
    return out


def save(name, obj):
    path = os.path.join(RES, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["time", "correct"])
    ap.add_argument("--variant",
                    choices=["baseline", "singlediv", "f32w", "centered"],
                    default="baseline")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--chunks", type=int, default=6)
    args = ap.parse_args()

    if args.mode == "time":
        r = time_variant(args.variant, args.n, n_chunks=args.chunks)
        save(f"p2b_time_{args.variant}_n{args.n}", r)
        print(f"[{args.variant} n{args.n}] "
              f"ms/step median={r['ms_per_step_median']:.3f} "
              f"min={r['ms_per_step_min']:.3f}  finite={r['finite']}  "
              f"trace_calls={r['patch_trace_calls']}  "
              f"divides={r['chunk'].get('divides')}  "
              f"temp={r['chunk'].get('temp_bytes', 0)/1e6:.0f}MB")
    else:
        r = correctness(args.n)
        save(f"p2b_correct_n{args.n}", r)
        print(f"baseline chunk divides: {r['baseline_chunk_divides']}")
        for v, d in r["variants"].items():
            print(f"[{v}] trace_calls={d['patch_trace_calls']} "
                  f"divides={d['chunk_divides']}")
            for name, dd in d["diffs"].items():
                print(f"   {name}: max_abs={dd['max_abs']:.2e} "
                      f"max_rel={dd['max_rel']:.2e} finite={dd['finite']}")


if __name__ == "__main__":
    main()
