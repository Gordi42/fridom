# XLA SPMD-partitioned FFT emits complex64 twiddle factors for a complex128 transform (HLO verifier rejects the mixed multiply)

**Filed 2026-07-18 as [jax-ml/jax#39291](https://github.com/jax-ml/jax/issues/39291).**

### Description

A jitted `complex128` FFT whose transform axis is sharded fails to
compile when the FFT result feeds a consumer that keeps that axis
sharded. XLA then lowers the FFT through its distributed Cooley-Tukey
decomposition, whose twiddle-factor constants come out as `complex64`
against the `complex128` data, and the HLO verifier aborts compilation.

```
INVALID_ARGUMENT: during context [hlo verifier]: Binary op multiply with
different element types: c64[] and c128[].
, for instruction %multiply = c64[] multiply(%constant.1, %get-tuple-element.4),
metadata={op_name="jit(fft)/fft"}
```

I hit this in an ocean model. The reproducer below is distilled to
pure jax and fails on jax 0.10.2 and 0.11.0, on CUDA and on the CPU
backend with forced host devices, so the fault sits in the
backend-agnostic SPMD partitioning of the FFT.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

n = 16
mesh = Mesh(np.array(jax.devices()), ("d",))
sh = lambda x, s: jax.device_put(x, NamedSharding(mesh, P(*s)))

x = sh(np.random.default_rng(0).standard_normal((n, n, n)), ("d", None, None))
q = jnp.asarray(np.random.default_rng(1).standard_normal((n, n, n, n)))

def f(a, q):
    c = jnp.fft.fft(a.astype(jnp.complex128), axis=0)   # norm=None; axis 0 sharded
    return jnp.einsum("...dj,...d->...j", q, jnp.moveaxis(c, 1, -1))

jax.jit(f)(x, q).block_until_ready()   # <-- c64/c128 HLO verifier error
```

run as

```
JAX_PLATFORMS=cuda python repro.py
# or without any GPU
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu python repro.py
```

- Transform axis (`axis 0`) sharded plus the einsum crashes (4/4 runs,
  with `norm=None` and `norm="forward"`).
- A non-transform axis sharded (`P(None, "d", None)`) is fine (local
  FFT, no twiddles).
- The FFT alone, transform axis sharded but no einsum, is fine (GSPMD
  all-gathers the axis).

The partitioned HLO (`*.after_spmd_partitioner.txt`) shows the mixed
dtypes directly

```
%constant.1 = c64[] constant((0, -1.57079637))              # -i*pi/2
%get-tuple-element.4 = c128[] get-tuple-element(%param.1), index=2
%multiply = c64[] multiply(%constant.1, %get-tuple-element.4)   # <-- rejected
%exponential = c64[] exponential(%multiply.1)                  # exp(twiddle)
%broadcast = c128[16,16,4] broadcast(%exponential)
```

I would expect the twiddle constants to be synthesized at the
operand's element type. As a workaround we reshard so the transform
axis is device-local before the FFT, which avoids the distributed
decomposition entirely.

### System info

- jax 0.10.2 and 0.11.0, matching jaxlib (pip/uv wheels), both
  affected (last checked 2026-07-18)
- Python 3.12.11, `jax_enable_x64=True`
- GPU runs on 4x NVIDIA A100-SXM4-80GB, single process
  (single-controller GSPMD)
