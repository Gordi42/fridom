# Batched `jnp.linalg.eigh` heap-corrupts on many-core CPUs (jaxlib `_lapack` over-subscribes OpenBLAS threads)

**DO NOT FILE without the repo owner's go-ahead.** Draft for jax-ml/jax.

## Summary

On a many-core CPU host, `jnp.linalg.eigh` on a moderately large *batch*
of small symmetric matrices corrupts the process heap and crashes
(SIGABRT `corrupted size vs. prev_size`, or SIGSEGV). A single matrix is
fine; the batched path is not. The corruption happens inside jaxlib's own
CPU LAPACK kernel (`jaxlib/cpu/_lapack.so`,
`jax::EigenvalueDecompositionSymmetric::Kernel`), which fans the batch
across the XLA/Eigen intra-op threadpool while each worker also calls a
multithreaded OpenBLAS `?syevd`. The nested oversubscription overruns
OpenBLAS's precompiled per-thread metadata array
(`OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary
array for thread metadata`), and the workspace `free` inside the kernel
then trips glibc's heap checks.

Capping `OPENBLAS_NUM_THREADS` to a value comfortably below the machine's
core count (e.g. 32 on a 256-thread box) avoids it.

## Versions

- jax 0.10.2, jaxlib 0.10.2 (pip/uv wheels)
- Python 3.12.11, CPU backend, `jax_enable_x64=True`
- Linux glibc 2.28 (`Linux-4.18.0-...el8`, x86_64)
- **2× AMD EPYC 7763 64-Core (128 physical / 256 logical cores)**
- BLAS/LAPACK backend loaded at runtime: `libscipy_openblas-5f890258.so`
  (the scipy-bundled OpenBLAS), driven through jaxlib's `_lapack.so` FFI.

## Reproducer (no sharding, one device)

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

a = jax.random.normal(jax.random.PRNGKey(0), (144, 63, 63), dtype=jnp.float64)
a = (a + jnp.swapaxes(a, -1, -2)) / 2   # symmetric
w, v = jnp.linalg.eigh(a)               # <-- heap corruption
w.block_until_ready()
print("OK")
```

```
JAX_PLATFORMS=cpu python -X faulthandler repro.py
```

- Batch **<= 128** of `63x63`: runs fine.
- Batch **144** of `63x63`: **crashes deterministically** (4/4 runs).
- A single `63x63` (unbatched): fine.
- `OPENBLAS_NUM_THREADS=32 JAX_PLATFORMS=cpu python repro.py`: prints `OK`
  (2/2). `OPENBLAS_NUM_THREADS=64`: still crashes.

(The exact batch/dim threshold is machine-specific — it is the point at
which `batch_workers * openblas_threads_per_call` exceeds OpenBLAS's
precompiled `NUM_THREADS`. `144` is what our workload hits: one `63x63`
generalized-Hermitian pencil per Fourier mode of a `16^3` grid.)

## Observed

`gdb` backtrace of the aborting worker thread:

```
OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary
array for thread metadata.

Thread 282 "tf_XLAEigen" received signal SIGABRT, Aborted.
#3  malloc_printerr ()                                  from libc.so.6
#4  unlink_chunk.isra ()                                from libc.so.6
#5  _int_free ()                                        from libc.so.6
#6  jax::EigenvalueDecompositionSymmetric<(xla::ffi::DataType)12, int>
        ::Kernel(...)::{lambda(long, long)#1}::operator()(long, long)
                                            from jaxlib/cpu/_lapack.so
#7  xla::ffi::ThreadPool::Schedule<jax::ParallelBatchMap(...)::$_0>(...)
                                            from jaxlib/cpu/_lapack.so
#8  Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)
                                            from jaxlib/libjax_common.so
#9  ...  tsl::thread::EigenEnvironment::CreateThread ...
#10 tsl::(anonymous namespace)::PThread::ThreadFn(void*)
#11 start_thread ()                                from libpthread.so.0
```

Frames #6–#8 show the mechanism: jaxlib's `_lapack` `ParallelBatchMap`
schedules the per-matrix eighs across the Eigen intra-op threadpool
(up to core-count workers), and each worker's `?syevd` call into OpenBLAS
also spins up OpenBLAS threads. On a 256-thread host that product exceeds
OpenBLAS's precompiled thread-metadata capacity; the fallback path
("adding auxiliary array") corrupts adjacent heap chunks, and the
per-matrix workspace `free` inside the kernel lambda (frame #6) faults.

Because it is heap corruption, the crash **site wanders** across runs:
sometimes it aborts here in the LAPACK kernel's `free`; sometimes it
surfaces later — e.g. under
`XLA_FLAGS=--xla_force_host_platform_device_count=4` it usually SIGSEGVs
inside the *next* op's lowering (we first misattributed it to the
`sort`/`argsort` comparator lowering — `_sort_lower` →
`_canonicalize_float_for_sort` → `shaped_abstractify` — because that op
followed the eigh and was the next code to walk the poisoned heap).

## Expected

Batched `jnp.linalg.eigh` should complete or raise a Python-level error,
never corrupt the heap. Concretely, jaxlib's `ParallelBatchMap` eigh
kernel should not create nested Eigen-threadpool × OpenBLAS-threadpool
oversubscription that exceeds OpenBLAS's precompiled `NUM_THREADS`
(e.g. serialize the inner BLAS calls, cap the inner OpenBLAS thread count
around the parallel batch, or size the batch parallelism against the
available BLAS thread budget).

## Workaround

Set `OPENBLAS_NUM_THREADS` (and/or `OMP_NUM_THREADS`) to a value well
below the host core count before importing jax, e.g. `OPENBLAS_NUM_THREADS=8`
for a single-process CPU run.

Note: this workaround does **not** cover a run under
`XLA_FLAGS=--xla_force_host_platform_device_count=N`, where each forced
device drives the eigh concurrently and the corruption reappears even at
low per-pool caps.

## Notes / open questions

- Is `_lapack.so`'s `ParallelBatchMap` intended to run with a
  single-threaded inner BLAS? If so, the OpenBLAS thread count is leaking
  through from the environment.
- The scipy-bundled `libscipy_openblas` here is precompiled with a
  `NUM_THREADS` below this host's 256; a jaxlib-bundled OpenBLAS built for
  more threads would raise, not remove, the ceiling — the nested
  oversubscription is the root issue.
