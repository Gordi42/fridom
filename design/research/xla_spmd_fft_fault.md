---
status: frozen
date: 2026-07-12
---

# XLA SPMD-FFT fault — a jitted FFT along a sharded axis

Frozen investigation record. Opened by the coordinate-systems work
([`../plans/done/coordinate_systems_plan.md`](../plans/done/coordinate_systems_plan.md)
§8): under forced-4 devices, mapped pressure solves with a tall
bounded column failed to compile, and two validation gates
(`tests/validation/test_moving_geometry.py:143`, `:181`) are
`single_device`-marked because of it. The fault is **not** in fridom
and was not introduced by that work.

## 1. The fault

Under multiple devices, a **jitted** FFT along a **sharded** axis
fails at dispatch. XLA:CPU layout assignment hands the `fft` HLO a
column-major operand layout, and the CPU FFT thunk RET_CHECKs:

```
INTERNAL: RET_CHECK failure
  (external/xla/xla/backends/cpu/runtime/fft_thunk.cc:168)
  LayoutUtil::IsMonotonicWithDim0Major(input_shape_.layout())
```

Platform: jax / jaxlib **0.10.2**, CPU backend, x64 on or off.

## 2. Reproducer (pure jax, no fridom)

```python
import jax, jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType

mesh = jax.make_mesh((2,), ("d",), axis_types=(AxisType.Auto,))
x = jax.device_put(jnp.ones((8, 4)),
                   NamedSharding(mesh, P("d", None)))

@jax.jit
def solve(x):                      # fridom's spectral-solve shape
    k = jnp.ones(x.shape, dtype=complex)
    return jnp.fft.ifft(jnp.fft.fft(x, axis=0) * k, axis=0).real

jax.block_until_ready(solve(x))    # RET_CHECK
```

Run with `XLA_FLAGS=--xla_force_host_platform_device_count=2`.

> **Block on the result.** Dispatch is async: without
> `block_until_ready` (or a host read) the failure does not surface
> and every variant looks like it passes. This burned the first pass
> of this investigation.

## 3. Condition matrix (measured)

`AxisType.Auto` mesh — the mode `fridom.spatial.decomposition` uses
(`decomposition/tensor.py:176`). Shape `(8, 4)`, sharded on axis 0.

| Variant | 1 device | 2 devices | 4 devices |
|---|---|---|---|
| `fft -> * array -> ifft`, FFT on the **sharded** axis | OK | **FAIL** | **FAIL** |
| same, **no jit** | OK | OK | OK |
| same, FFT on an **unsharded** axis | OK | OK | OK |
| same, **replicated** input (`P(None, None)`) | OK | OK | OK |
| same, `with_sharding_constraint` to replicated *inside* jit | OK | OK | OK |
| `fft` alone (no inverse) on the sharded axis | OK | OK | OK |
| bare `ifft(fft(x))`, no op between | OK | OK | OK |

Independent of dtype (float32 / float64 / complex) and of shape
(4x2, 8x4, 16x16, 32x8, 64x128 all behave identically).

**Necessary conditions:** jit, >= 2 devices, and an `fft` **chained
into** a second FFT HLO along the axis that carries the shards. A
single FFT is fine.

**What does *not* generalize:** whether an intervening op triggers or
avoids the fault. Under an `AxisType.Auto` mesh a bare `ifft(fft(x))`
passes and multiplication by an array fails; under the
`AxisType.Explicit` mesh `jax.make_mesh` builds by default, the bare
chain **fails** and a scalar multiply passes. Scalar `* 2.0` and
`sin()` between the transforms also pass under Auto. This is
layout-assignment fragility, not a crisp rule — the only reliable
predictor is "two chained FFTs on the sharded axis, under jit".

The fridom-relevant pattern — `fft -> multiply by the symbol ->
ifft` — fails in **every** configuration tested.

## 4. Workaround

The FFT operand must not be sharded along the transformed axis. Three
forms work (all re-measured 2026-07-12 under forced 4 devices, with
`block_until_ready` so the async dispatch cannot hide the fault):

- a genuinely **replicated input** array;
- **`with_sharding_constraint(x, replicated)` inside** the jitted
  function — verified in both the `(8, 4)`/axis-0 and `(8, 16)`/axis-1
  configurations, with the constraint on the input alone and on the
  input plus the coefficient-space intermediate;
- **sharding a different axis** than the transformed one — the
  fridom-side mitigation of section 5.

(An earlier draft of this note recorded the
`with_sharding_constraint` route as failing; that was a measurement
error and is corrected here.)

## 5. Consequence for fridom (the actionable part)

The fridom-level *shape* dependence seen during C3/C4 (8x8 fine,
8x16 fails) is **not** part of the XLA bug — the bug is
shape-independent. It is fridom's layout choice interacting with it.

Two code facts:

- `decomposition.negotiate` shards the **first GHOST-capable factor
  whose cell count divides the device count and whose per-shard
  extent covers `min_local_size` and the halo**
  (`spatial/decomposition/decomposition.py:519`). It is *not* the
  longest axis. So which axis ends up sharded depends on the cell
  counts, the halo and the device count: shrink the horizontal
  relative to the column and the shards migrate onto the column.
- The transform planner has **no reshard stages**:
  "on one device all axes are local, so the plan carries no reshard
  stages; the multi-device planner extends this object"
  (`spatial/operators/transform.py:240`). So a `Transform` runs the
  FFT on whatever axis the data happens to be sharded along — there
  is no pencil transpose.

Together: whenever the negotiated shard axis coincides with a
transformed axis, fridom emits exactly the failing HLO. The
bounded-column solves hit it because the column is the axis that
becomes shardable when the horizontal is small (and the bounded-axis
trig transforms are FFT-backed).

**Recommended mitigation, independent of any upstream fix:** the
spectral path must not run a transform on a distributed axis. Either

- the decomposition avoids sharding an axis that carries a transform
  (a layout constraint the transform's `requirements` can declare —
  the `layout="local"` demand the one-sided closures already use), or
- the transform planner gains its **reshard (pencil-transpose)
  stages** — the extension `transform.py` explicitly designs for —
  and redistributes to a layout where the transformed axis is local.

The second is the real fix and is already the designed-for path; the
first is a cheap guard that makes the failure a taught error instead
of an XLA RET_CHECK.

## 6. Prior art

- **jax-ml/jax#22484** — same RET_CHECK. Closed 2025-07-22 as
  completed **with no fix**: the check merely moved from
  `ir_emitter.cc` to the thunk runtime.
- **openxla/xla#44541** — the adjacent c64/c128 SPMD-FFT verifier
  failure. Open; reported for CUDA, reproduces on CPU.
- **openxla/xla#24** — umbrella issue: SPMD `fft` support.
- A new jax issue is being filed by the owner.

**Distinct from** the owner's existing **jax-ml/jax#39100**
(GPU-only multi-output-fusion buffer aliasing -> silent wrong
results). Different backend, different mechanism; that one is silent,
this one is loud.

## 7. Not verified here

The GPU/CUDA behavior: this machine exposes one CPU device, so the
matrix above is CPU-only. openxla/xla#44541 reports the adjacent
failure on CUDA, so a GPU check is worth doing before claiming the
mitigation covers every backend.
