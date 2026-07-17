# XLA:GPU distributed FFT emits complex64 twiddle factors for a complex128 transform (HLO verifier rejects the mixed multiply)

**DO NOT FILE without the repo owner's go-ahead.** Draft for jax-ml/jax.

## Summary

On a multi-device GPU (single-controller GSPMD), a jit'd `complex128` FFT
whose **transform axis is sharded** — and whose result feeds a downstream
consumer that keeps that axis sharded — makes XLA lower the FFT through its
**distributed (collective) FFT** decomposition. The twiddle-factor
constants of that decomposition are synthesized at **`complex64`** while
the operand and the cuFFT custom-call output are `complex128`, so the HLO
verifier rejects the mixed-precision multiply:

```
INVALID_ARGUMENT: during context [hlo verifier]: Binary op multiply with
different element types: c64[] and c128[].
, for instruction %multiply = c64[] multiply(%constant.1, %get-tuple-element.4),
metadata={op_name="jit(fft)/fft"}
```

The transform never runs; compilation aborts in the verifier.

It is **not** a jax-side normalization issue: it reproduces with
`norm=None` (no `1/prod(s)` scaling anywhere). It is **not** covered by
`--xla_disable_hlo_passes=multi_output_fusion` (that flag is active in
every run below). An *isolated* sharded FFT compiles fine (GSPMD
all-gathers the axis and does one local cuFFT); the downstream contraction
is what forces the distributed decomposition where the c64 twiddles appear.

## Versions

- jax 0.10.2, jaxlib 0.10.2 (pip/uv wheels)
- Python 3.12.11, **CUDA backend**, `jax_enable_x64=True`
- 4x NVIDIA A100-SXM4-80GB, single process (single-controller GSPMD)
- `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion` (does not help)

## Reproducer (no host gather, single process, >= 2 GPUs)

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

```
XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion \
JAX_PLATFORMS=cuda python repro.py
```

- Sharding the **transform axis** (`axis 0`) + the einsum: **crashes**
  (4/4 runs), with `norm="forward"` and with `norm=None`.
- Sharding a **non-transform axis** (`P(None, "d", None)`): **OK** — the
  transform axis is local, so XLA does a local cuFFT (no twiddles).
- The FFT **alone** (no einsum), sharded on the transform axis: **OK** —
  GSPMD all-gathers the axis.

## Observed HLO (`*.after_spmd_partitioner.txt`, `--xla_dump_to`)

The FFT lowers to a distributed Cooley-Tukey `while` loop; its twiddle
constants are `c64` while the data is `c128`:

```
%fft_collective_permute_body (... c128[16,16,4], c128[], c128[], u32[]) -> (...) {
  %constant.1 = c64[] constant((0, -1.57079637))              # -i*pi/2
  %get-tuple-element.4 = c128[] get-tuple-element(%param.1), index=2
  %multiply = c64[] multiply(%constant.1, %get-tuple-element.4)   # <-- c64 * c128
  %get-tuple-element.5 = c128[] get-tuple-element(%param.1), index=3
  %multiply.1 = c64[] multiply(%multiply, %get-tuple-element.5)
  %exponential = c64[] exponential(%multiply.1)                  # exp(twiddle)
  %broadcast = c128[16,16,4] broadcast(%exponential)
  %multiply.2 = c128[16,16,4] multiply(%broadcast, %get-tuple-element.3)
  %add = c128[16,16,4] add(%multiply.2, %get-tuple-element.2)
  %collective-permute.1 = c128[16,16,4] collective-permute(%get-tuple-element.3)
  ...
}
...
%constant.4 = c64[] constant((0, -0.392699093))                 # -i*2*pi/16
%broadcast.2 = c128[16,16,4] broadcast(%constant.4)
%fft.3 = c128[16,16,4] fft(%slice), fft_type=FFT, fft_length={4}   # local chunk
%all-to-all = c128[16,16,4] all-to-all(%dot)
```

The twiddle angles are `-i*pi/2` and `-i*2*pi/16` — the distributed-FFT
phase factors `exp(-2*pi*i*k/N)` — emitted at `complex64` while every
data buffer they multiply is `complex128`.

## Expected

The distributed-FFT decomposition should synthesize its twiddle-factor
constants at the operand's element type (`complex128` here), so
`multiply`/`exponential` stay `c128 x c128`. A `complex128` FFT must never
be lowered with `complex64` twiddles — the result would silently lose
precision even if the verifier did not reject it.

## Workaround

Keep the FFT's transform axis **device-local** (do not shard it): e.g.
`jax.lax.with_sharding_constraint` the operand to a sharding that
replicates the transform axis before the FFT, or transpose so the
transform runs over a local axis (a slab/pencil FFT that reshards to make
each transform axis local in turn). Both give one local cuFFT per shard
and avoid the distributed decomposition entirely (bit-for-bit identical to
the single-device result in our tests).

## Notes / open questions

- Same family as jax#39100 (an XLA:GPU/GSPMD FFT lowering fault), but a
  distinct instruction: the twiddle-constant dtype, not fusion.
- Does the emitter derive the twiddle dtype from a hard-coded
  `complex64`/`float32` default rather than the transform's element type?
- `--xla_disable_hlo_passes=multi_output_fusion` does **not** mask it; we
  did not find an `--xla_disable_hlo_passes` value that does (the twiddles
  are emitted during SPMD partitioning of the FFT itself).
