"""Stage-1 micro: Pallas(Triton) vs XLA composed for upwind5 flux-div.

Target (one stencil axis, storage frame, padded N=n+8 arrays q,v):
  face f (between cell f and f+1):
    Lrec(f) = sum_k cL[k] q[f-2+k]   (cells f-2..f+2)
    Rrec(f) = sum_k cR[k] q[f-1+k]   (cells f-1..f+3)
    F(f)    = where(v[f]>0, Lrec, Rrec)       (tie -> right)
    flux(f) = v[f]*F(f)
  output d(i) = flux(i) - flux(i-1)  over interior cells [4 : N-4] (n cells)

Both the XLA reference and the Pallas kernels compute this exact algebra
so the correctness gate isolates Triton f64 FMA reordering only.
"""
from __future__ import annotations

import argparse
import statistics
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

jax.config.update("jax_enable_x64", True)
# This jaxlib defaults jax_pallas_use_mosaic_gpu=True (Hopper-only); force
# the Triton backend on this Ampere (sm_86) card.
jax.config.update("jax_pallas_use_mosaic_gpu", False)
_TRITON = pltriton.CompilerParams()

# ---- exact upwind5 rows (fractions -> identical f64 both sides) ----
cL = (1.0 / 30, -13.0 / 60, 47.0 / 60, 9.0 / 20, -1.0 / 20)
cR = tuple(reversed(cL))                       # right == left reversed
# dissipation split over the 6-cell union window (cells f-2..f+3)
_lu = (*cL, 0.0)
_ru = (0.0, *cR)
c_sym = tuple((a + b) / 2 for a, b in zip(_lu, _ru))
c_diss = tuple((b - a) / 2 for a, b in zip(_lu, _ru))


def _sl(a, ax, lo, hi):
    idx = [slice(None)] * a.ndim
    idx[ax] = slice(lo, hi)
    return a[tuple(idx)]


# ================================================================
#  XLA composed reference (one jit)
# ================================================================
@partial(jax.jit, static_argnums=(2,))
def xla_ref(q, v, axis):
    n_ax = q.shape[axis]
    # flux over faces f in [3, N-4)  (length N-7)
    lo, hi = 3, n_ax - 4
    length = hi - lo                             # N-7
    vf = _sl(v, axis, lo, hi)
    lrec = None
    for k, w in enumerate(cL):
        term = w * _sl(q, axis, lo - 2 + k, lo - 2 + k + length)
        lrec = term if lrec is None else lrec + term
    rrec = None
    for k, w in enumerate(cR):
        term = w * _sl(q, axis, lo - 1 + k, lo - 1 + k + length)
        rrec = term if rrec is None else rrec + term
    face = jnp.where(vf > 0.0, lrec, rrec)
    flux = vf * face
    return _sl(flux, axis, 1, length) - _sl(flux, axis, 0, length - 1)


@partial(jax.jit, static_argnums=(2,))
def xla_ref_diss(q, v, axis):
    """Dissipation-form XLA reference (no where)."""
    n_ax = q.shape[axis]
    lo, hi = 3, n_ax - 4
    length = hi - lo
    vf = _sl(v, axis, lo, hi)
    sym = None
    dis = None
    for k in range(6):
        seg = _sl(q, axis, lo - 2 + k, lo - 2 + k + length)
        s = c_sym[k] * seg
        d = c_diss[k] * seg
        sym = s if sym is None else sym + s
        dis = d if dis is None else dis + d
    flux = vf * sym - jnp.abs(vf) * dis
    return _sl(flux, axis, 1, length) - _sl(flux, axis, 0, length - 1)


# ================================================================
#  Pallas kernels
# ================================================================
def _tile(ref, sax, s_start, a0, a1, ob, tb0, tb1):
    """Load a pow2 (tb0,tb1,ob) tile with the stencil window on axis sax."""
    if sax == 2:
        return ref[pl.ds(a0, tb0), pl.ds(a1, tb1), pl.ds(s_start, ob)]
    return ref[pl.ds(s_start, ob), pl.ds(a0, tb0), pl.ds(a1, tb1)]


def _shift_body(q_ref, v_ref, o_ref, sax, ob, tb0, tb1, form):
    """Fixed-offset pow2-tile stencil (no non-pow2 slice anywhere).

    Every array is (tb0,tb1,ob) [sax=2] or (ob,tb0,tb1) [sax=0], all
    dims pow2.  q read at rel offsets -3..3, v at 0 and -1.  Flux is
    computed at BOTH faces of each output cell (the OB+1-face single
    pass is non-pow2), so the reconstruction runs twice per cell.
    """
    b_s = pl.program_id(0)
    b_0 = pl.program_id(1)
    b_1 = pl.program_id(2)
    c0 = 4 + b_s * ob
    a0 = b_0 * tb0
    a1 = b_1 * tb1
    q = {r: _tile(q_ref, sax, c0 + r, a0, a1, ob, tb0, tb1)
         for r in range(-3, 4)}
    v_i = _tile(v_ref, sax, c0, a0, a1, ob, tb0, tb1)
    v_h = _tile(v_ref, sax, c0 - 1, a0, a1, ob, tb0, tb1)
    if form == "both":
        # face i: left cells i-2..i+2, right i-1..i+3
        fi_l = (cL[0] * q[-2] + cL[1] * q[-1] + cL[2] * q[0]
                + cL[3] * q[1] + cL[4] * q[2])
        fi_r = (cR[0] * q[-1] + cR[1] * q[0] + cR[2] * q[1]
                + cR[3] * q[2] + cR[4] * q[3])
        flux_i = v_i * jnp.where(v_i > 0.0, fi_l, fi_r)
        # face i-1: left cells i-3..i+1, right i-2..i+2
        fh_l = (cL[0] * q[-3] + cL[1] * q[-2] + cL[2] * q[-1]
                + cL[3] * q[0] + cL[4] * q[1])
        fh_r = (cR[0] * q[-2] + cR[1] * q[-1] + cR[2] * q[0]
                + cR[3] * q[1] + cR[4] * q[2])
        flux_h = v_h * jnp.where(v_h > 0.0, fh_l, fh_r)
    else:  # dissipation
        sym_i = (c_sym[0] * q[-2] + c_sym[1] * q[-1] + c_sym[2] * q[0]
                 + c_sym[3] * q[1] + c_sym[4] * q[2] + c_sym[5] * q[3])
        dis_i = (c_diss[0] * q[-2] + c_diss[1] * q[-1] + c_diss[2] * q[0]
                 + c_diss[3] * q[1] + c_diss[4] * q[2] + c_diss[5] * q[3])
        flux_i = v_i * sym_i - jnp.abs(v_i) * dis_i
        sym_h = (c_sym[0] * q[-3] + c_sym[1] * q[-2] + c_sym[2] * q[-1]
                 + c_sym[3] * q[0] + c_sym[4] * q[1] + c_sym[5] * q[2])
        dis_h = (c_diss[0] * q[-3] + c_diss[1] * q[-2] + c_diss[2] * q[-1]
                 + c_diss[3] * q[0] + c_diss[4] * q[1] + c_diss[5] * q[2])
        flux_h = v_h * sym_h - jnp.abs(v_h) * dis_h
    o_ref[...] = flux_i - flux_h


def _pallas_specs(sax, n, tb0, tb1, ob):
    big = n + 8
    if sax == 2:
        out_shape = (big, big, n)
        out_block = (tb0, tb1, ob)
        out_map = lambda s, i, j: (i, j, s)  # noqa: E731
    else:
        out_shape = (n, big, big)
        out_block = (ob, tb0, tb1)
        out_map = lambda s, i, j: (s, i, j)  # noqa: E731
    grid = (n // ob, big // tb0, big // tb1)
    full = pl.BlockSpec((big, big, big), lambda *_: (0, 0, 0))
    return grid, full, out_block, out_map, out_shape


def make_pallas(sax, n, tb0, tb1, ob, form="both", interpret=False):
    """Build a jitted Pallas call for stencil axis sax on N=n+8 arrays."""
    grid, full, out_block, out_map, out_shape = _pallas_specs(
        sax, n, tb0, tb1, ob)

    def kernel(q_ref, v_ref, o_ref):
        _shift_body(q_ref, v_ref, o_ref, sax, ob, tb0, tb1, form)

    call = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[full, full],
        out_specs=pl.BlockSpec(out_block, out_map),
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float64),
        compiler_params=None if interpret else _TRITON,
        interpret=interpret,
    )
    return jax.jit(call)


# ================================================================
#  Timing
# ================================================================
def bench(fn, args, iters=50, warmup=5):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    best = []
    for _ in range(6):
        t0 = perf_counter()
        for _ in range(iters):
            r = fn(*args)
        jax.block_until_ready(r)
        best.append((perf_counter() - t0) / iters * 1e3)
    return statistics.median(best), min(best)


def ideal_bytes(n):
    big = n + 8
    return (big ** 3 * 8) * 2 + (big * big * n) * 8   # q + v read, out write


def make_inputs(n, seed=0):
    big = n + 8
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((big, big, big)))
    v = jnp.asarray(rng.standard_normal((big, big, big)))
    return q, v


def correctness_interp(n=32):
    """Interpret-mode (CPU) correctness vs XLA ref for both axes/forms."""
    q, v = make_inputs(n, seed=3)
    out = {}
    for sax in (2, 0):
        ref = np.asarray(xla_ref(q, v, sax))
        for form, ob, tb in (("both", 16, 1), ("both", 32, 1),
                             ("both", 16, 2), ("dissipation", 16, 1)):
            k = make_pallas(sax, n, tb, tb, ob, form, interpret=True)
            got = np.asarray(k(q, v))
            out[(sax, form, ob, tb)] = float(np.max(np.abs(got - ref)))
    return out


def numpy_oracle(q, v, axis):
    """Independent triple-loop oracle to validate xla_ref itself."""
    q = np.asarray(q)
    v = np.asarray(v)
    big = q.shape[axis]
    q = np.moveaxis(q, axis, 0)
    v = np.moveaxis(v, axis, 0)
    n = big - 8
    out = np.zeros((n, *q.shape[1:]))
    cLn = np.array(cL)
    cRn = np.array(cR)

    def flux(f):
        lrec = sum(cLn[k] * q[f - 2 + k] for k in range(5))
        rrec = sum(cRn[k] * q[f - 1 + k] for k in range(5))
        face = np.where(v[f] > 0.0, lrec, rrec)
        return v[f] * face

    for j, i in enumerate(range(4, big - 4)):
        out[j] = flux(i) - flux(i - 1)
    return np.moveaxis(out, 0, axis)


# ================================================================
#  GPU sweep
# ================================================================
# (label, form, OB, tb0, tb1)  — all dims must be powers of 2 (Triton)
CONFIGS = [
    ("ob64",        "both", 64,   1, 1),
    ("ob128",       "both", 128,  1, 1),
    ("ob256",       "both", 256,  1, 1),
    ("ob64_tb2x1",  "both", 64,   2, 1),
    ("ob64_tb1x2",  "both", 64,   1, 2),
    ("ob64_tb2x2",  "both", 64,   2, 2),
    ("ob64_tb4x2",  "both", 64,   4, 2),
    ("ob64_tb1x8",  "both", 64,   1, 8),
    ("ob64_tb8x8",  "both", 64,   8, 8),
    ("ob32_tb8x8",  "both", 32,   8, 8),
    ("ob64_diss",   "dissipation", 64, 1, 1),
    ("ob64_tb8x8_diss", "dissipation", 64, 8, 8),
]


def run_time(n, iters):
    import json
    q, v = make_inputs(n, seed=7)
    rows = []
    for sax in (2, 0):
        ref = xla_ref(q, v, sax)
        ref_np = np.asarray(ref)
        med, mn = bench(xla_ref, (q, v, sax), iters=iters)
        ib = ideal_bytes(n)
        rows.append(dict(kernel="xla_ref", form="both", sax=sax, n=n,
                         bs=None, tb0=1, tb1=1, ms_med=med, ms_min=mn,
                         gbs_med=ib / (med * 1e-3) / 1e9,
                         gbs_min=ib / (mn * 1e-3) / 1e9, max_abs=0.0))
        med2, mn2 = bench(xla_ref_diss, (q, v, sax), iters=iters)
        d_np = np.asarray(xla_ref_diss(q, v, sax))
        rows.append(dict(kernel="xla_ref_diss", form="diss", sax=sax, n=n,
                         bs=None, tb0=1, tb1=1, ms_med=med2, ms_min=mn2,
                         gbs_med=ib / (med2 * 1e-3) / 1e9,
                         gbs_min=ib / (mn2 * 1e-3) / 1e9,
                         max_abs=float(np.max(np.abs(d_np - ref_np)))))
        for label, form, ob, tb0, tb1 in CONFIGS:
            if n % ob:
                continue
            try:
                k = make_pallas(sax, n, tb0, tb1, ob, form)
                got = k(q, v)
                jax.block_until_ready(got)
                ma = float(np.max(np.abs(np.asarray(got) - ref_np)))
                med, mn = bench(k, (q, v), iters=iters)
                rows.append(dict(kernel=f"pallas_{label}", form=form,
                                 sax=sax, n=n, bs=ob, tb0=tb0, tb1=tb1,
                                 ms_med=med, ms_min=mn,
                                 gbs_med=ib / (med * 1e-3) / 1e9,
                                 gbs_min=ib / (mn * 1e-3) / 1e9,
                                 max_abs=ma))
            except Exception as e:  # noqa: BLE001
                rows.append(dict(kernel=f"pallas_{label}", form=form,
                                 sax=sax, n=n, bs=ob, tb0=tb0, tb1=tb1,
                                 error=repr(e)[:300]))
    with open(f"stage1_n{n}.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    # print table
    print(f"\n=== n={n}  backend={jax.default_backend()} "
          f"ideal_bytes={ideal_bytes(n)/1e6:.1f}MB ===")
    print(f"{'kernel':<22}{'ax':>3}{'bs':>5}{'tb':>6}"
          f"{'ms_med':>9}{'ms_min':>9}{'GB/s':>8}{'max_abs':>11}")
    for r in rows:
        if "error" in r:
            print(f"{r['kernel']:<22}{r['sax']:>3}{str(r['bs']):>5}"
                  f"  ERROR {r['error'][:60]}")
            continue
        tb = f"{r['tb0']}x{r['tb1']}"
        print(f"{r['kernel']:<22}{r['sax']:>3}{str(r['bs']):>5}{tb:>6}"
              f"{r['ms_med']:>9.3f}{r['ms_min']:>9.3f}"
              f"{r['gbs_med']:>8.1f}{r['max_abs']:>11.2e}")
    return rows


def run_smoke(n):
    """Compile + single run each config on the active backend."""
    q, v = make_inputs(n, seed=5)
    ref = np.asarray(xla_ref(q, v, 2))
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    for sax in (2, 0):
        r2 = np.asarray(xla_ref(q, v, sax))
        for label, form, ob, tb0, tb1 in CONFIGS:
            if n % ob:
                print(f"ax{sax} {label:<18} SKIP (ob {ob} !| n {n})")
                continue
            try:
                k = make_pallas(sax, n, tb0, tb1, ob, form)
                got = np.asarray(k(q, v))
                ma = float(np.max(np.abs(got - r2)))
                print(f"ax{sax} {label:<18} OK max_abs={ma:.2e}")
            except Exception as e:  # noqa: BLE001
                print(f"ax{sax} {label:<18} FAIL {repr(e)[:160]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oracle", "interp", "time", "smoke"])
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if args.mode == "smoke":
        run_smoke(args.n)
    elif args.mode == "time":
        run_time(args.n, args.iters)
    elif args.mode == "oracle":
        for n in (16, 24):
            q, v = make_inputs(n, seed=1)
            for sax in (2, 0):
                r = np.asarray(xla_ref(q, v, sax))
                o = numpy_oracle(q, v, sax)
                print(f"n={n} ax{sax} ref-vs-oracle max_abs="
                      f"{np.max(np.abs(r - o)):.2e} shape={r.shape}")
    elif args.mode == "interp":
        res = correctness_interp(n=24)
        for k, val in res.items():
            print(f"{k}: max_abs={val:.2e}")
