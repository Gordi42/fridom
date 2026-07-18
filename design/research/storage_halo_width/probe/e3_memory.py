"""E3: compiled-step memory for baseline vs forced-narrow storage.

upwind5 at n^3, chunk_size=1. Reports the compiled chunk's
``memory_analysis()`` (temp/argument/output bytes) plus the exact
per-field storage-shape byte accounting and the predicted byte ratio
(n+6)^3 / (n+8)^3.
"""
from __future__ import annotations

import contextlib
import sys

import numpy as np

import _common as C
from _force_halo import force_halo

import fridom.nonhydro2 as nh
from fridom.model.model import _CHUNK_COMPILE_LOG, _CHUNK_EXECUTABLES


def _mem_attrs(mem):
    if mem is None:
        return "no memory_analysis"
    keys = ("temp_size_in_bytes", "argument_size_in_bytes",
            "output_size_in_bytes", "generated_code_size_in_bytes")
    return {k: getattr(mem, k, None) for k in keys}


def build_and_compile(n, cap):
    _CHUNK_COMPILE_LOG.clear()
    _CHUNK_EXECUTABLES.clear()
    build = (force_halo(cap) if cap is not None
             else contextlib.nullcontext())
    with build:
        model = nh.Model(
            grid=C.make_grid(n), dt=0.001, advection=C.upwind5(),
            family="nodal", chunk_size=1)
        C.seed_state(model)
        shapes = C.storage_shapes(model)
        w = C.widths(model)
        model.run(steps=1, progress=False)
    logs = [(k[1], _mem_attrs(m)) for k, (s, m) in
            _CHUNK_COMPILE_LOG.items()]
    return w, shapes, logs


def field_bytes(shapes):
    total = 0
    per = {}
    for c, shp in shapes.items():
        b = int(np.prod(shp)) * 8  # float64
        per[c] = (shp, b)
        total += b
    return per, total


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    print(f"=== E3 memory, upwind5, n={n}^3, chunk_size=1 ===\n")
    for label, cap in (("baseline (width 4)", None),
                       ("forced-3", 3)):
        w, shapes, logs = build_and_compile(n, cap)
        per, total = field_bytes(shapes)
        print(f"--- {label}: widths={w} ---")
        for c, (shp, b) in per.items():
            print(f"  {c}: storage shape {shp}  {b:,} bytes")
        print(f"  sum state storage bytes: {total:,}")
        for nn, mem in logs:
            print(f"  compiled chunk n={nn}: {mem}")
        print()
    pred = (n + 6) ** 3 / (n + 8) ** 3
    print(f"predicted storage byte ratio (n+6)^3/(n+8)^3 = "
          f"{pred:.4f}  -> {100 * (1 - pred):.2f}% fewer bytes")
