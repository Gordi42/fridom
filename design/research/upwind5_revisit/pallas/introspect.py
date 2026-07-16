"""Characterize the XLA composed micro: temp bytes, fusion count, HLO.

Answers whether the single-axis micro is representative of the real
step's flux materialization (the ~12ms 'extra HBM bytes' lever at n192)
or only its fusion-efficiency lever.
"""
from __future__ import annotations

import re

import jax
import numpy as np

import stage1 as S

jax.config.update("jax_enable_x64", True)


def analyze(n):
    big = n + 8
    q = jax.numpy.asarray(np.random.default_rng(0).standard_normal(
        (big, big, big)))
    v = jax.numpy.asarray(np.random.default_rng(1).standard_normal(
        (big, big, big)))
    print(f"\n=== n={n} backend={jax.default_backend()} ===")
    for name, fn in (("xla_ref_ax2", lambda: S.xla_ref(q, v, 2)),
                     ("xla_ref_ax0", lambda: S.xla_ref(q, v, 0))):
        low = jax.jit(fn).lower()
        comp = low.compile()
        txt = comp.as_text()
        nfus = len(re.findall(r"fusion", txt))
        try:
            mem = comp.memory_analysis()
            temp = mem.temp_size_in_bytes
            peak = mem.peak_memory_in_bytes
        except Exception as e:  # noqa: BLE001
            temp = peak = f"err {e}"
        print(f"{name:<14} fusion_words={nfus:<4} "
              f"temp={temp if isinstance(temp,str) else temp/1e6:.1f}MB "
              f"peak={peak if isinstance(peak,str) else peak/1e6:.1f}MB")

    # a Pallas kernel: temp bytes (should be ~output only)
    k = S.make_pallas(2, n, 1, 1, 64, "both")
    comp = jax.jit(lambda: k(q, v)).lower().compile()
    try:
        mem = comp.memory_analysis()
        print(f"pallas_ob64_ax2 temp={mem.temp_size_in_bytes/1e6:.1f}MB "
              f"peak={mem.peak_memory_in_bytes/1e6:.1f}MB")
    except Exception as e:  # noqa: BLE001
        print("pallas mem err", e)


if __name__ == "__main__":
    for n in (128, 192):
        analyze(n)
