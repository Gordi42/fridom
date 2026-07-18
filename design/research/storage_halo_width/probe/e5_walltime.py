"""E5: indicative CPU wall-time per step, baseline vs forced-3.

upwind5, n^3, chunk_size=1. Warm up, then time single steps and report
the median. INDICATIVE ONLY: login-node CPU, one process per variant.
Run once per variant (fresh process): `python e5_walltime.py <n> base`
or `python e5_walltime.py <n> 3`.
"""
from __future__ import annotations

import contextlib
import statistics
import sys
import time

import jax

import _common as C
from _force_halo import force_halo

import fridom.nonhydro2 as nh


def main(n, cap, warmup=3, reps=25):
    build = (force_halo(cap) if cap is not None
             else contextlib.nullcontext())
    with build:
        model = nh.Model(
            grid=C.make_grid(n), dt=0.001, advection=C.upwind5(),
            family="nodal", chunk_size=1)
        C.seed_state(model)
        w = C.widths(model)
        # warmup: compile the length-1 chunk
        model.run(steps=warmup, progress=False)
        jax.block_until_ready(model.state["u"].storage)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model.run(steps=1, progress=False)
            jax.block_until_ready(model.state["u"].storage)
            times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    lo, hi = min(times), max(times)
    print(f"widths={w}  median/step={med * 1e3:.2f} ms  "
          f"(min={lo * 1e3:.2f}, max={hi * 1e3:.2f}, reps={reps})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    variant = sys.argv[2] if len(sys.argv) > 2 else "base"
    cap = None if variant == "base" else int(variant)
    label = "baseline (natural width)" if cap is None else f"forced-{cap}"
    print(f"=== E5 wall-time, upwind5 n={n}^3, {label} "
          f"[login-node CPU, INDICATIVE] ===")
    main(n, cap)
