#!/usr/bin/env python
"""Minimal repro: XLA:GPU distributed-FFT emits complex64 twiddles for a
complex128 transform, and the HLO verifier rejects the mixed multiply.

Context
-------
This is the *actual* root cause of the "channel eigenmodes broken on real
multi-GPU" fault catalogued in
``design/research/multidevice_test_faults.md`` (item 1, GPU mechanism).
The record attributed the ``c64[] multiply c128[]`` verifier error to jax's
FFT *normalization* constant (``jax.numpy.fft`` ``_fft_norm``'s
``1/prod(s)``). **That attribution is refuted here:** the fault reproduces
with ``norm=None`` (no jax normalization at all), so the offending
``complex64`` scalar is not the norm constant.

The SPMD-partitioned HLO shows the true source. When XLA:GPU/GSPMD lowers
an FFT whose transform axis is *sharded* across devices, it emits a
distributed Cooley-Tukey decomposition (a ``while`` loop named
``fft_collective_permute_body`` doing local FFTs + ``collective-permute`` +
``all-to-all``). The per-stage **twiddle factors** in that decomposition
are synthesized at ``complex64`` regardless of the operand precision::

    %constant.1 = c64[] constant((0, -1.57079637))        # -i*pi/2
    %get-tuple-element.4 = c128[] get-tuple-element(...)   # c128 data
    %multiply = c64[] multiply(%constant.1, %get-tuple-element.4)   # <-- bug
    %exponential = c64[] exponential(%multiply.1)          # exp(twiddle)
    ...
    %constant.4 = c64[] constant((0, -0.392699093))        # -i*2*pi/16

The HLO verifier then rejects ``multiply c64[] c128[]``::

    INVALID_ARGUMENT: during context [hlo verifier]: Binary op multiply
    with different element types: c64[] and c128[].

This is an upstream XLA:GPU fault in the same family as jax#39100, and it
is **not** covered by ``--xla_disable_hlo_passes=multi_output_fusion``
(that flag is active in every run below and does not help).

Trigger conditions (measured 2026-07-17, DKRZ node, 4x A100-SXM4-80GB)
--------------------------------------------------------------------
* jax 0.10.2, jaxlib 0.10.2, CUDA backend, ``jax_enable_x64=True``.
* >= 2 devices; a jit'd complex128 FFT whose **transform axis is
  sharded**, feeding a downstream consumer (an einsum/dot) that keeps the
  transform axis sharded (so GSPMD chooses the distributed-FFT lowering
  rather than all-gathering the axis).
* An *isolated* sharded FFT does **not** reproduce (GSPMD all-gathers the
  axis and does a single local cuFFT); the contraction is what forces the
  distributed decomposition. Sharding a **non**-transform axis does
  **not** reproduce (the transform axis is then local -> local cuFFT).

Run
---
    XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion \
    JAX_PLATFORMS=cuda python repro_distributed_fft_c64_twiddle.py

Expected: the [BUG] trials raise the c64/c128 verifier error; the [OK]
trials (norm out of the picture, or the transform axis local) compile and
run. To see the offending HLO, add
``--xla_dump_to=/tmp/fftdump`` and grep the
``*.after_spmd_partitioner.txt`` for ``c64``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

N = 16
DEVS = jax.devices()
MESH = Mesh(np.array(DEVS), ("d",))
RNG = np.random.default_rng(0)


def _shard(x: np.ndarray, spec: tuple) -> jax.Array:
    return jax.device_put(x, NamedSharding(MESH, P(*spec)))


def _run(label: str, fn, *args) -> None:
    try:
        jax.jit(fn)(*args).block_until_ready()
        print(f"[OK ] {label}")
    except Exception as exc:  # noqa: BLE001
        line = next((ln for ln in str(exc).splitlines()
                     if "element types" in ln), str(exc).splitlines()[0])
        print(f"[BUG] {label}\n      {line.strip()[:120]}")


def main() -> None:
    print(f"devices: {len(DEVS)} ({DEVS[0].platform})")
    if len(DEVS) < 2:
        print("needs >= 2 devices; nothing to reproduce on 1 device")
        return

    x3 = _shard(RNG.standard_normal((N, N, N)), ("d", None, None))
    q4 = jnp.asarray(RNG.standard_normal((N, N, N, N)))

    def fft_then_einsum(a: jax.Array, q: jax.Array, *, norm) -> jax.Array:
        c = jnp.fft.fft(a.astype(jnp.complex128), axis=0, norm=norm)
        return jnp.einsum("...dj,...d->...j", q, jnp.moveaxis(c, 1, -1))

    # (1) the fault, with jax's forward normalization
    _run("fft(sharded axis 0, norm='forward') + einsum",
         lambda a, q: fft_then_einsum(a, q, norm="forward"), x3, q4)

    # (2) the fault persists with norm=None -> the c64 scalar is NOT the
    #     jax FFT-norm constant; it is the distributed-FFT twiddle
    _run("fft(sharded axis 0, norm=None) + einsum  [refutes fft-norm]",
         lambda a, q: fft_then_einsum(a, q, norm=None), x3, q4)

    # (3) transform axis LOCAL (shard a non-transform axis): no
    #     distributed FFT, no c64 twiddles -> compiles and runs
    x3b = _shard(RNG.standard_normal((N, N, N)), (None, "d", None))
    _run("fft(local axis 0, norm='forward') + einsum  [axis 1 sharded]",
         lambda a, q: fft_then_einsum(a, q, norm="forward"), x3b, q4)


if __name__ == "__main__":
    main()
