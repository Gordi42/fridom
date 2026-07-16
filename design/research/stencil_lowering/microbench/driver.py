"""Driver: one subcommand per experiment, fresh process per XLA_FLAGS.

Usage: JAX_PLATFORMS=cuda python driver.py <cmd> [--n N] [--dtype f64|f32]
Writes results/<cmd>.json and results/hlo/*.txt.  Never imports fridom.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import stencils as S
import e3 as E
from common import (
    analyze, check_device, dispatch_floor, loop_periter, measure_roofline,
    run_case, time_median,
)

HERE = os.path.dirname(os.path.abspath(__file__))  # noqa: PTH100,PTH120
RES = os.path.join(HERE, "results")  # noqa: PTH118
HLO = os.path.join(RES, "hlo")  # noqa: PTH118
os.makedirs(HLO, exist_ok=True)  # noqa: PTH103

_RNG = np.random.default_rng(1234)


def dtype_of(name):
    return jnp.float64 if name == "f64" else jnp.float32


def make(shape, dt):
    a = _RNG.standard_normal(shape).astype(
        np.float64 if dt == jnp.float64 else np.float32)
    return jax.device_put(jnp.asarray(a, dtype=dt))


def save(cmd, results):
    path = os.path.join(RES, f"{cmd}.json")  # noqa: PTH118
    with open(path, "w") as fh:  # noqa: PTH123
        json.dump(results, fh, indent=2)
    print(f"wrote {path} ({len(results)} cases)")


def periter_case(name, apply_once, x0, ideal_bytes):
    lp = loop_periter(apply_once, x0)
    lp["name"] = name
    lp["variant"] = "loop"
    s = lp["periter_ms"] / 1e3
    lp["periter_gbs"] = ideal_bytes / s / 1e9
    return lp


# ================================================================
#  roofline + dispatch
# ================================================================


def cmd_roofline(args):
    check_device()
    out = {"dispatch_us": dispatch_floor() * 1e6, "rooflines": []}
    for n in (256, 512):
        out["rooflines"].append(measure_roofline(n, np.float64))
    out["rooflines"].append(measure_roofline(256, np.float32))
    save("roofline", out)
    for r in out["rooflines"]:
        print(f"  n={r['n']} {r['dtype']}: identity {r['identity_gbs']:.0f} "
              f"GB/s, triad {r['triad_gbs']:.0f} GB/s")
    print(f"  dispatch floor {out['dispatch_us']:.1f} us")


# ================================================================
#  E1
# ================================================================

E1_SPELL_PADDED = {
    "a_valid": S.valid_sum,
    "d_conv": S.conv_stencil,
    "e_stack": S.stack_tensordot,
    "f_taploop": S.tap_loop,
}
E1_SPELL_NPAD = {
    "b_roll": S.roll_sum,
    "c_padinside": S.pad_inside,
}


def cmd_e1(args):
    check_device()
    dt = dtype_of(args.dtype)
    n = args.n
    results = []
    ks = [2, 3, 5, 7] if not args.spot else [5]
    axes = [0, 2]
    for axis in axes:
        for k in ks:
            w = S.w_for(k)
            # padded input
            pshape = [n, n, n]
            pshape[axis] = n + (k - 1)
            fp = make(tuple(pshape), dt)
            fn = make((n, n, n), dt)  # size-N input for b/c
            for sname, fnc in E1_SPELL_PADDED.items():
                if args.spot and sname not in ("a_valid", "d_conv"):
                    continue
                name = f"e1_{sname}_k{k}_ax{axis}_n{n}"
                def f(x, _f=fnc, _k=k, _a=axis, _w=w):
                    return _f(x, _k, _a, _w)
                results.append(run_case(name, f, (fp,), hlo_dir=HLO))
            for sname, fnc in E1_SPELL_NPAD.items():
                if args.spot and sname not in ("b_roll",):
                    continue
                name = f"e1_{sname}_k{k}_ax{axis}_n{n}"
                def f(x, _f=fnc, _k=k, _a=axis, _w=w):
                    return _f(x, _k, _a, _w)
                results.append(run_case(name, f, (fn,), hlo_dir=HLO))
    # per-iter loop probes at k=5 (256 only)
    if not args.spot and n == 256:
        k = 5
        for axis in axes:
            w = S.w_for(k)
            x0 = make((n, n, n), dt)
            ideal = 2 * x0.nbytes
            loops = {
                "a_valid": lambda x, _w=w: S.valid_preserve(x, 5, axis, _w),
                "b_roll": lambda x, _w=w: S.roll_sum(x, 5, axis, _w),
                "d_conv": lambda x, _w=w: S.conv_preserve(x, 5, axis, _w),
                "e_stack": lambda x, _w=w: S.stack_preserve(x, 5, axis, _w),
                "f_taploop": lambda x, _w=w: S.taploop_preserve(x, 5, axis, _w),
            }
            for sname, ap in loops.items():
                name = f"e1_{sname}_k5_ax{axis}_n{n}_loop"
                results.append(periter_case(name, ap, x0, ideal))
    save(f"e1{'_512' if n == 512 else ''}"
         f"{'_' + args.dtype if args.dtype != 'f64' else ''}", results)


# ================================================================
#  E2
# ================================================================


def cmd_e2(args):
    check_device()
    dt = dtype_of(args.dtype)
    n = args.n
    results = []
    # second derivative axis 0 and 2
    for axis in (0, 2):
        pshape = [n, n, n]
        pshape[axis] = n + 2
        fp = make(tuple(pshape), dt)
        cases = {
            "sd_composed": S.second_deriv_composed,
            "sd_direct": S.second_deriv_direct,
            "sd_barrier": S.second_deriv_composed_barrier,
        }
        for sname, fnc in cases.items():
            name = f"e2_{sname}_ax{axis}_n{n}"
            def f(x, _f=fnc, _a=axis):
                return _f(x, _a)
            results.append(run_case(name, f, (fp,), hlo_dir=HLO))
    # laplacian (padded by 1)
    fp1 = make((n + 2, n + 2, n + 2), dt)
    for sname, fnc in {"lap_composed": S.laplacian_composed,
                       "lap_fused": S.laplacian_fused,
                       "lap_barrier": S.laplacian_composed_barrier}.items():
        results.append(run_case(f"e2_{sname}_n{n}", fnc, (fp1,), hlo_dir=HLO))
    # biharmonic (padded by 2)
    fp2 = make((n + 4, n + 4, n + 4), dt)
    for sname, fnc in {"bih_composed": S.biharmonic_composed,
                       "bih_direct": S.biharmonic_direct,
                       "bih_barrier": S.biharmonic_composed_barrier}.items():
        results.append(run_case(f"e2_{sname}_n{n}", fnc, (fp2,), hlo_dir=HLO))
    # per-iter loops
    if n == 256:
        x0 = make((n, n, n), dt)
        ideal = 2 * x0.nbytes
        for sname, ap in {"lap_fused": S.lap_preserve,
                          "lap_composed": S.lap_composed_preserve,
                          "bih_composed": S.biharm_preserve,
                          "bih_direct": S.biharm_direct_preserve}.items():
            results.append(periter_case(f"e2_{sname}_n{n}_loop", ap, x0,
                                        ideal))
    save("e2", results)


# ================================================================
#  E3 / E3-mof / E4
# ================================================================


def _e3_fields(n, dt):
    pad = n + 2 * E.H
    return tuple(make((pad, pad, pad), dt) for _ in range(4))


def _run_tendency(results, tag, recon_name, recon, up, vp, wp, bp, n,
                  spellings):
    for sname, fnc in spellings.items():
        name = f"{tag}_{recon_name}_{sname}_n{n}"
        def f(u, v, w, b, _f=fnc, _r=recon, _n=n):
            return _f(u, v, w, b, _r, _n)
        results.append(run_case(name, f, (up, vp, wp, bp), hlo_dir=HLO))


def cmd_e3(args):
    check_device()
    dt = dtype_of(args.dtype)
    n = args.n
    results = []
    up, vp, wp, bp = _e3_fields(n, dt)
    spellings = {
        "composed": E.tendency_composed,
        "handfused": E.tendency_handfused,
        "barrier": E.tendency_barrier,
    }
    recon_set = E.RECON if not args.spot else {"weno5": E.recon_weno5}
    for rname, recon in recon_set.items():
        _run_tendency(results, "e3", rname, recon, up, vp, wp, bp, n,
                      spellings if not args.spot else
                      {"composed": E.tendency_composed})
    tag = "e3" + ("_512" if n == 512 else "")
    save(tag, results)


def cmd_e3mof(args):
    """E3 composed for upwind5/weno5 under multi_output_fusion disabled."""
    check_device()
    dt = dtype_of(args.dtype)
    n = args.n
    results = []
    up, vp, wp, bp = _e3_fields(n, dt)
    for rname in ("upwind5", "weno5"):
        recon = E.RECON[rname]
        _run_tendency(results, "e3mof", rname, recon, up, vp, wp, bp, n,
                      {"composed": E.tendency_composed})
    results.append({"note": "XLA_FLAGS",
                    "xla_flags": os.environ.get("XLA_FLAGS", "")})
    save("e3mof", results)


def cmd_e4(args):
    check_device()
    n = args.n
    results = []
    # f64 weno composed already in e3; here: single-axis, hoist, f32.
    up, vp, wp, bp = _e3_fields(n, jnp.float64)

    def f_full(u, v, w, b):
        return E.tendency_composed(u, v, w, b, E.recon_weno5, n)

    def f_axis0(u, v, w, b):
        return E.tendency_weno_axis0(u, v, w, b, n)

    def f_hoist(u, v, w, b):
        return E.tendency_weno_hoist(u, v, w, b, n)

    results.append(run_case(f"e4_weno_full_f64_n{n}", f_full,
                            (up, vp, wp, bp), hlo_dir=HLO))
    results.append(run_case(f"e4_weno_axis0_f64_n{n}", f_axis0,
                            (up, vp, wp, bp), hlo_dir=HLO))
    results.append(run_case(f"e4_weno_hoist_f64_n{n}", f_hoist,
                            (up, vp, wp, bp), hlo_dir=HLO))

    # f32 spot-check
    up2, vp2, wp2, bp2 = _e3_fields(n, jnp.float32)
    results.append(run_case(f"e4_weno_full_f32_n{n}", f_full,
                            (up2, vp2, wp2, bp2), hlo_dir=HLO))
    save("e4", results)


CMDS = {
    "roofline": cmd_roofline, "e1": cmd_e1, "e2": cmd_e2,
    "e3": cmd_e3, "e3mof": cmd_e3mof, "e4": cmd_e4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--dtype", default="f64", choices=["f64", "f32"])
    ap.add_argument("--spot", action="store_true")
    args = ap.parse_args()
    print(f"cmd={args.cmd} n={args.n} dtype={args.dtype} spot={args.spot} "
          f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
    CMDS[args.cmd](args)


if __name__ == "__main__":
    main()
