"""P3b (RTX 3060 adaptation): real nonhydro2 model A/B for one-path
upwind spellings, run against the editable dev install (not a worktree).

Adapted from design/research/stencil_lowering/microbench/phase3/p3b_model.py
for THIS machine (single consumer Ampere GPU, f64-slow).  Changes vs the
DKRZ original:
  * no hardcoded WT_SRC sys.path insertion — the editable install of the
    repo (branch dev) is imported directly.
  * ``_assert_single_gpu`` keeps the single-GPU assertion, drops the
    worktree-path check.
  * RES/HLO output dirs live under this scratchpad, never in design/.

Semantic note (dev): ``WENOAdvection`` now OVERRIDES ``_face_value``
(advection.py:2984) via ``_SelectedFaceReconstruction``.  The harness
monkeypatches ``UpwindAdvection._face_value``; WENO's override shadows it,
so the ``w5_selected`` / ``w5_selected_f32w`` patch variants are DEAD on
dev (patch_trace_calls == 0, diffs ~0 vs the already-selected baseline).
Only ``w5_baseline`` (kind=None, the shipped production selected-input
path) is meaningful for WENO here.  The ``u5_*`` variants patch the linear
path, which is still both-then-select on dev, so they DO take effect.

Usage:
  JAX_PLATFORMS=cuda python p3b_model.py time --variant u5_baseline --n 128
  JAX_PLATFORMS=cuda python p3b_model.py correct --n 64
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model import model as MM
# advection modules were rehomed nonhydro2.modules -> model.modules
# mid-session (dev commit 9b530109 "refactor/advection-rehome", HY-D5);
# the classes are still re-exported as nh.UpwindAdvection etc.
from fridom.model.modules import advection as ADV
from fridom.model.modules.advection import _linear_row
from fridom.spatial.operators.base import (
    _ensure_valid,
    _finalize,
    _required_halo,
    resolve_codomain,
)
from fridom.spatial.operators.reconstruct import (
    apply_fv_staggered,
)
from fridom.spatial.operators.weno import (
    WENO_EPS,
    _weighted_sum,
    _window_views,
    weno_tables,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
HLO = os.path.join(RES, "hlo")
os.makedirs(HLO, exist_ok=True)

F32 = jnp.float32
F64 = jnp.float64

_PATCH_CALLS = {"n": 0}  # trace-time call counter (patch liveness)
_ORIG_FACE_VALUE = ADV.UpwindAdvection._face_value


def _assert_single_gpu():
    # P3B_ALLOW_CPU=1 permits a CPU-only correctness dry-run (numerical
    # parity is platform-independent at f64); timed runs never set it.
    if os.environ.get("P3B_ALLOW_CPU") == "1":
        return
    if jax.default_backend() != "gpu" or jax.device_count() != 1:
        raise RuntimeError(
            f"expected single gpu, got {jax.default_backend()} "
            f"x{jax.device_count()}")


# ================================================================
#  Left WENO reconstruction from pre-sliced tap cells
# ================================================================
def _weno_left_from_taps(cells, tables):
    """Left WENO-JS from ``order`` tap cells (f64 weights)."""
    r = len(tables.optimal)
    alphas, cands = [], []
    for m, offset in enumerate(tables.offsets):
        cm = cells[offset:offset + r]
        cands.append(_weighted_sum(cm, tables.coeffs[m]))
        beta = None
        for scale, diff in zip(tables.beta_scale, tables.beta_rows[m],
                               strict=True):
            square = _weighted_sum(cm, diff) ** 2
            term = square if scale == 1.0 else scale * square
            beta = term if beta is None else beta + term
        alphas.append(tables.optimal[m] / (beta + WENO_EPS) ** 2)
    total = alphas[0]
    combined = alphas[0] * cands[0]
    for a, c in zip(alphas[1:], cands[1:], strict=True):
        total = total + a
        combined = combined + a * c
    return combined / total


def _weno_left_from_taps_f32(cells, tables):
    """Left WENO-JS from tap cells with the weights in f32 (standard
    per-candidate spelling; f64 candidates)."""
    r = len(tables.optimal)
    cands = [_weighted_sum(cells[o:o + r], tables.coeffs[m])
             for m, o in enumerate(tables.offsets)]
    eps = F32(WENO_EPS)
    alpha = []
    for m, o in enumerate(tables.offsets):
        cm32 = tuple(c.astype(F32) for c in cells[o:o + r])
        beta = None
        for scale, diff in zip(tables.beta_scale, tables.beta_rows[m],
                               strict=True):
            square = _weighted_sum(cm32, diff) ** 2
            term = square if scale == 1.0 else F32(scale) * square
            beta = term if beta is None else beta + term
        alpha.append(F32(tables.optimal[m]) / (beta + eps) ** 2)
    asum = alpha[0]
    for m in range(1, r):
        asum = asum + alpha[m]
    inv = F32(1.0) / asum
    w = [(alpha[m] * inv).astype(F64) for m in range(r)]
    out = w[0] * cands[0]
    for m in range(1, r):
        out = out + w[m] * cands[m]
    return out


def _dissipation_rows(order):
    """(c_sym, c_diss) over the order+1 union window (exact split)."""
    left = _linear_row(order, "left")
    right = _linear_row(order, "right")
    lu = (*left, 0.0)
    ru = (0.0, *right)
    c_sym = tuple((a + b) / 2 for a, b in zip(lu, ru))
    c_diss = tuple((b - a) / 2 for a, b in zip(lu, ru))
    return c_sym, c_diss


def _slice_axis(arr, ax, lo, hi):
    idx = [slice(None)] * arr.ndim
    idx[ax] = slice(lo, hi)
    return arr[tuple(idx)]


# ================================================================
#  Patched _face_value (bound method)
# ================================================================
def make_patched_face_value(kind):
    def _patched(self, q, v_face, axis, flux_space):
        # halo-negotiation pass hands HaloTracer operands (no _data);
        # the union path's per-side reach (3) equals the real
        # reconstruction's, so the halo demand is identical -> delegate
        # the trace to the original for a faithful (unchanged) width.
        if getattr(q, "_trace_apply", None) is not None:
            return _ORIG_FACE_VALUE(self, q, v_face, axis, flux_space)
        _PATCH_CALLS["n"] += 1
        order = self._order
        weighting = self._weighting
        u_size = order + 1
        bare = q.function_space.bare
        domain = bare.factor(axis)
        # union alignment == the LEFT reconstruction's: biased_offset
        # (order//2) plus the dual-direction cell-frame shift (1 on
        # Right->Center velocity self-advection, 0 on Center->Right)
        m0 = order // 2 + ADV._wall_shift(domain)
        left_op = self._left[axis]            # bound recon op (codomain)
        v_data = v_face._data

        if kind == "dissipation":
            c_sym, c_diss = _dissipation_rows(order)
            s_full = jnp.where(v_data > 0.0, 1.0, -1.0)  # tie -> right

            def kernel(storage, ax):
                wins = _window_views(storage, ax, u_size)
                length = wins[0].shape[ax]
                s = _slice_axis(s_full, ax, m0, m0 + length)
                sym = _weighted_sum(wins, c_sym)
                diss = _weighted_sum(wins, c_diss)
                return sym - s * diss
        else:
            pos_full = v_data > 0.0
            tables = weno_tables(order, "left")
            lin_row = _linear_row(order, "left")

            def kernel(storage, ax):
                wins = _window_views(storage, ax, u_size)  # U0..U_order
                length = wins[0].shape[ax]
                pos = _slice_axis(pos_full, ax, m0, m0 + length)
                taps = [jnp.where(pos, wins[i], wins[order - i])
                        for i in range(order)]
                if weighting == "weno":
                    if kind == "selected_f32w":
                        return _weno_left_from_taps_f32(taps, tables)
                    return _weno_left_from_taps(taps, tables)
                return _weighted_sum(tuple(taps), lin_row)

        # replicate the operator __call__ template so the operand is
        # halo-synced (union reach 3 == left_op requirement) and the
        # operand layout is re-attached to the bare-codomain result
        codomain = resolve_codomain(left_op, q.function_space)
        q = _ensure_valid(q, _required_halo(left_op, q.function_space))
        result = apply_fv_staggered(left_op, q, axis, u_size, kernel,
                                    metadata=q.metadata, align=m0)
        result = _finalize(q, result, codomain)
        return result.retag(flux_space)

    return _patched


# variant name -> (scheme, patch kind or None)
_VARIANTS = {
    "centered": ("centered", None),
    "u5_baseline": ("upwind5", None),
    "u5_selected": ("upwind5", "selected"),
    "u5_dissipation": ("upwind5", "dissipation"),
    "w5_baseline": ("weno5", None),
    "w5_selected": ("weno5", "selected"),
    "w5_selected_f32w": ("weno5", "selected_f32w"),
}


def _advection(scheme):
    if scheme == "centered":
        return nh.CenteredAdvection()
    if scheme == "upwind5":
        return nh.UpwindAdvection(order=5)
    return nh.WENOAdvection(order=5)


def apply_patch(kind):
    ADV.UpwindAdvection._face_value = make_patched_face_value(kind)


def reset_patch():
    ADV.UpwindAdvection._face_value = _ORIG_FACE_VALUE
    MM._CHUNK_EXECUTABLES.clear()
    MM._CHUNK_COMPILE_LOG.clear()


# ================================================================
#  Model construction (matched config; identical to P2b)
# ================================================================
def build_model(n, advection, dt=20.0):
    lx = ly = 10000.0
    lz = 100.0
    f0 = 1e-4
    n2 = (50.0 * f0) ** 2
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, lx), periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, ly), periodic=True, name="y")
    mz = fr.spatial.meshes.IntervalMesh(n, (0.0, lz), periodic=True, name="z")
    grid = fr.spatial.Grid((mx, my, mz))
    # family="nodal": the A100 reference config predates dev's FV
    # default (nh.Model auto-flips a fully-periodic grid to family="fv"
    # -> CellAvg tracers, on which the biased reconstruction currently
    # fails to assemble).  FV is bitwise-parity with nodal per the
    # design record, so nodal reproduces the A100 arithmetic exactly
    # and is the working full-step path.
    model = nh.Model(
        grid=grid, dt=dt,
        coriolis=nh.FPlaneCoriolis(f0=f0),
        stratification=nh.ConstantStratification(n2=n2),
        advection=advection,
        family="nodal",
        chunk_size=50)
    kx, ky, kz = 2 * np.pi / lx, 2 * np.pi / ly, 2 * np.pi / lz
    xc = (np.arange(n) + 0.5) * (lx / n)
    yc = (np.arange(n) + 0.5) * (ly / n)
    zc = (np.arange(n) + 0.5) * (lz / n)
    x, y, z = np.meshgrid(xc, yc, zc, indexing="ij")
    u0 = float(os.environ.get("P3B_U0", "0.2"))
    b0 = float(os.environ.get("P3B_B0", "1e-4"))
    model.set_fields(
        u=u0 * np.sin(kx * x) * np.cos(ky * y),
        v=0.3 * u0 * np.cos(kx * x),
        b=b0 * np.cos(kz * z))
    return model


# ================================================================
#  Chunk introspection (temp bytes + divide profile)
# ================================================================
def _chunk_exe(n_steps=50):
    for key, exe in MM._CHUNK_EXECUTABLES.items():
        if key[1] == n_steps:
            return exe
    return None


def count_divides(hlo):
    out = {"total": 0, "f64": 0, "f32": 0}
    for m in re.finditer(
            r"(f64|f32|[a-z0-9]+)\[[^\]]*\][^ ]*\s+divide\(", hlo):
        out["total"] += 1
        t = m.group(1)
        out[t] = out.get(t, 0) + 1
    return out


def chunk_stats(n_steps=50, save_hlo_as=None):
    exe = _chunk_exe(n_steps)
    if exe is None:
        return {}
    stats = {}
    try:
        hlo = exe.as_text()
        stats["divides"] = count_divides(hlo)
        if save_hlo_as is not None:
            p = os.path.join(HLO, f"{save_hlo_as}.txt")
            with open(p, "w") as fh:
                fh.write(hlo)
            stats["hlo_path"] = p
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
#  Timing / correctness
# ================================================================
def _block(model):
    jax.block_until_ready(jax.tree_util.tree_leaves(model.state))


def _finite(model):
    return all(bool(np.all(np.isfinite(np.asarray(leaf))))
               for leaf in jax.tree_util.tree_leaves(model.state)
               if isinstance(leaf, jax.Array))


def time_variant(variant, n, n_chunks=6, steps=50):
    _assert_single_gpu()
    scheme, kind = _VARIANTS[variant]
    if kind is not None:
        apply_patch(kind)
    model = build_model(n, _advection(scheme))
    _block(model)

    t0 = perf_counter()
    model.advance(steps)          # compile + warmup chunk (discarded)
    _block(model)
    first_s = perf_counter() - t0

    stats = chunk_stats(steps, save_hlo_as=f"p3b_{variant}_n{n}")
    per_step_ms = []
    for _ in range(n_chunks):
        t0 = perf_counter()
        model.advance(steps)
        _block(model)
        per_step_ms.append((perf_counter() - t0) / steps * 1e3)

    return {
        "variant": variant, "scheme": scheme, "n": n, "steps": steps,
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


def _state_arrays(model):
    out = {}
    for name in ("u", "v", "w", "b"):
        try:
            out[name] = np.asarray(model.state[name]._data)
        except Exception:  # noqa: BLE001
            pass
    return out


def correctness(n, steps=20):
    _assert_single_gpu()
    out = {"n": n, "steps": steps, "variants": {}}
    for scheme, base_name, spellings in (
            ("upwind5", "u5_baseline",
             (("u5_selected", "selected"),
              ("u5_dissipation", "dissipation"))),
            ("weno5", "w5_baseline",
             (("w5_selected", "selected"),
              ("w5_selected_f32w", "selected_f32w")))):
        reset_patch()
        base = build_model(n, _advection(scheme))
        base.advance(steps)
        _block(base)
        base_state = _state_arrays(base)
        out["variants"][base_name] = {
            "chunk_divides": chunk_stats(steps).get("divides"),
            "temp_bytes": chunk_stats(steps).get("temp_bytes"),
        }
        del base
        for variant, kind in spellings:
            reset_patch()
            _PATCH_CALLS["n"] = 0
            apply_patch(kind)
            m = build_model(n, _advection(scheme))
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
    reset_patch()
    return out


def save(name, obj):
    path = os.path.join(RES, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["time", "correct"])
    ap.add_argument("--variant", choices=list(_VARIANTS), default="w5_baseline")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--chunks", type=int, default=6)
    args = ap.parse_args()

    if args.mode == "time":
        r = time_variant(args.variant, args.n, n_chunks=args.chunks)
        save(f"p3b_time_{args.variant}_n{args.n}", r)
        c = r["chunk"]
        print(f"[{args.variant} n{args.n}] "
              f"ms/step median={r['ms_per_step_median']:.3f} "
              f"min={r['ms_per_step_min']:.3f}  finite={r['finite']}  "
              f"trace_calls={r['patch_trace_calls']}  "
              f"divides={c.get('divides')}  "
              f"temp={c.get('temp_bytes', 0) / 1e6:.0f}MB")
    else:
        r = correctness(args.n)
        save(f"p3b_correct_n{args.n}", r)
        for v, d in r["variants"].items():
            if "diffs" not in d:
                print(f"[{v}] (baseline) divides={d['chunk_divides']} "
                      f"temp={ (d.get('temp_bytes') or 0)/1e6:.0f}MB")
                continue
            print(f"[{v}] trace_calls={d['patch_trace_calls']} "
                  f"divides={d['chunk_divides']}")
            for name, dd in d["diffs"].items():
                print(f"   {name}: max_abs={dd['max_abs']:.2e} "
                      f"max_rel={dd['max_rel']:.2e} finite={dd['finite']}")


if __name__ == "__main__":
    main()
