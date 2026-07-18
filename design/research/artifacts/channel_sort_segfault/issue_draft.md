# Batched `jnp.linalg.eigh` corrupts the heap on many-core CPUs (jaxlib `_lapack` oversubscribes OpenBLAS threads)

**Filed 2026-07-18 as [jax-ml/jax#39292](https://github.com/jax-ml/jax/issues/39292).**

### Description

`jnp.linalg.eigh` on a batch of small symmetric float64 matrices
crashes with heap corruption on a many-core CPU host (SIGABRT with
`corrupted size vs. prev_size`, or SIGSEGV). jaxlib's CPU LAPACK
kernel fans the batch across the XLA/Eigen intra-op threadpool while
each worker calls a multithreaded OpenBLAS `?syevd`. The nested thread
counts exceed OpenBLAS's precompiled `NUM_THREADS` (`OpenBLAS warning:
precompiled NUM_THREADS exceeded, adding auxiliary array for thread
metadata`) and the heap gets corrupted.

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

run as

```
JAX_PLATFORMS=cpu python -X faulthandler repro.py
```

On a 256-logical-core host (2x AMD EPYC 7763)

- a batch of 128 or fewer `63x63` matrices runs fine, 144 crashes
  (4/4 runs)
- a single `63x63` is fine
- `OPENBLAS_NUM_THREADS=32` prints `OK`, 64 still crashes

The threshold scales with the core count. On a 128-core host batch
144 passes and batch 1024 crashes, on jax 0.10.2 with SIGABRT and on
0.11.0 with SIGSEGV.

`gdb` backtrace of the aborting thread

```
OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary
array for thread metadata.

Thread 282 "tf_XLAEigen" received signal SIGABRT, Aborted.
#5  _int_free ()                                        from libc.so.6
#6  jax::EigenvalueDecompositionSymmetric<(xla::ffi::DataType)12, int>
        ::Kernel(...)::{lambda(long, long)#1}::operator()(long, long)
                                            from jaxlib/cpu/_lapack.so
#7  xla::ffi::ThreadPool::Schedule<jax::ParallelBatchMap(...)::$_0>(...)
                                            from jaxlib/cpu/_lapack.so
#8  Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)
                                            from jaxlib/libjax_common.so
```

I would expect the batched eigh to complete or raise a Python-level
error, not corrupt the heap. The workaround is to set
`OPENBLAS_NUM_THREADS` well below the host core count before importing
jax.

### System info

- jax 0.10.2 and 0.11.0, matching jaxlib (pip/uv wheels), both crash
  (last checked 2026-07-18)
- Python 3.12.11, CPU backend, `jax_enable_x64=True`
- Linux glibc 2.28, x86_64
- The runtime BLAS/LAPACK is `libscipy_openblas-5f890258.so` (scipy's
  bundled OpenBLAS), driven through jaxlib's `_lapack.so` FFI.
