#!/usr/bin/env python
"""Minimal repro: jax.numpy.linalg.eigh heap-corrupts on a large batch.

Context
-------
This is the *actual* root cause of the "channel eigenmodes segfault on
forced-CPU multi-device" fault catalogued in
``design/research/multidevice_test_faults.md`` (item 1, CPU mechanism).
The record attributed the exit-139 crash to jax 0.10.2 lowering the
``sort`` primitive, because the faulthandler backtrace pointed into the
argsort comparator lowering (``_sort_lower`` / ``_canonicalize_float_for_sort``
/ ``shaped_abstractify``). That attribution is wrong: the crash is a
**heap corruption inside the batched eigendecomposition** (OpenBLAS, the
LAPACK backend jaxlib loads on CPU -- ``libscipy_openblas``), and the
sort comparator lowering is merely the next allocator to touch the
already-corrupted heap. Remove the sort entirely and the eigh alone
still crashes.

Trigger conditions (measured 2026-07-17, DKRZ A100 login/compute node)
---------------------------------------------------------------------
* jax 0.10.2, jaxlib 0.10.2, CPU backend, ``jax_enable_x64=True``.
* A 256-logical-core host. The bundled ``libscipy_openblas`` spins up its
  default thread pool (min(cores, build-time NUM_THREADS)); the machine's
  256 cores overrun a per-thread work buffer.
* ``jnp.linalg.eigh`` on a batch of >= ~144 small (63x63) symmetric
  matrices. Batch <= 128 is fine, 144 crashes deterministically. A single
  63x63 eigh is fine. (fridom hits exactly B = 16*9 = 144: the channel
  eigenbasis solves one 63x63 generalized-Hermitian pencil per horizontal
  Fourier mode of a 16^3 walled-y channel.)

Manifestation is non-deterministic in *form* (same corruption, different
symptom): single-process runs usually abort with glibc
``corrupted size vs. prev_size`` / ``BLAS : Bad memory unallocation``
(SIGABRT, exit 134); runs under
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` usually SIGSEGV
(exit 139) somewhere inside the following op's lowering -- which is how
the sort took the blame.

Run
---
    # crashes (exit 134 or 139), no sort, no sharding, ONE device:
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        python repro_batched_eigh_heap_corruption.py

    # mitigation -- capping the OpenBLAS pool avoids the overrun:
    OPENBLAS_NUM_THREADS=32 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        python repro_batched_eigh_heap_corruption.py   # prints OK
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

BATCH = 144  # 16 * 9 -- one 63x63 pencil per channel Fourier mode
DIM = 63

a = jax.random.normal(jax.random.PRNGKey(0), (BATCH, DIM, DIM),
                      dtype=jnp.float64)
a = (a + jnp.swapaxes(a, -1, -2)) / 2  # symmetric
w, v = jnp.linalg.eigh(a)              # <-- heap corruption here
w.block_until_ready()
print("OK: batch=%d dim=%d (no crash on this host)" % (BATCH, DIM))
