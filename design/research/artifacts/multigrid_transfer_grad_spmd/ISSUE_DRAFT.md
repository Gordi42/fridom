# SPMD partitioner emits a malformed concatenate for reverse-mode through `jax.linear_transpose` of `jnp.roll` on a fully sharded axis

## What happens

I have a function that applies the linear transpose of a two neighbour periodic
stencil, `roll(c, 1) + roll(c, -1)`, and I take `jax.grad` of a scalar loss
through it. When the rolled axis is sharded down to one element per device the
compile fails after spmd partitioning with an HLO verifier error.

```
INTERNAL: during context [hlo verifier]: Expected instruction to have shape equal to f64[6], actual shape is f64[4]:
Failed after spmd-partitioning
```

The forward pass compiles and runs fine. Only the reverse-mode compile fails.

Looking at the dumped HLO after spmd partitioning, the offending instruction is
a `concatenate` produced by the transpose of `jnp.roll`. It concatenates two
`f64[3,1]` slices along dimension 0, which should give `f64[6,1]`, but the
partitioner stamped its result shape as `f64[4,1]`. The second operand is the
same slice the sibling roll's transpose already uses, so the two transposed
rolls appear to get their partitioned operands cross wired. A single roll is
fine. Two rolls in the stencil are needed to trigger it.

## Minimal reproducer

This is CPU only, on four forced host devices, no accelerator needed.

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

jax.config.update("jax_enable_x64", True)

mesh = Mesh(np.array(jax.devices()), ("d",))
sharded = NamedSharding(mesh, P("d"))

# a size-4 axis over 4 devices is one element per device
x = jax.device_put(np.arange(4.0), sharded)

def stencil(c):
    return jnp.roll(c, 1) + jnp.roll(c, -1)

def loss(x):
    (cot,) = jax.linear_transpose(stencil, jax.ShapeDtypeStruct((4,), x.dtype))(x)
    return jnp.sum(cot ** 2)

print("forward runs:", float(jax.jit(loss)(x)))                       # 40.0
jax.jit(jax.grad(loss), out_shardings=sharded)(x)                     # crashes
```

## What I expect

The reverse-mode compile should succeed and give the gradient the single device
run gives. On one device the same loss and gradient are well defined and finite.

## What I have narrowed down

- One roll in the stencil compiles. Two rolls are needed.
- The rolled axis has to be sharded at exactly one element per device. Two
  elements per device (size 8 on 4 devices) compiles. A replicated input
  compiles.
- The `jax.linear_transpose` matters. Taking `jax.grad` of the plain stencil
  `sum(stencil(x) ** 2)` compiles. It is the grad through the linear transpose
  that fails, so the transposed roll ends up inside the forward graph of the
  differentiated computation.
- The forward only compile of the same linear transpose is fine. The failure
  needs both the transposed roll in the forward and the outer reverse-mode
  transpose.
- Same direction rolls fail the same way (`roll(c, 1) + roll(c, 2)`), so it is
  not about opposite directions.
- float32 fails the same way, so it is not a float64 issue.

A workaround that produces silently wrong gradients is worth flagging. Wrapping
the transposed function's argument or the transpose output in
`with_sharding_constraint(..., replicated)` makes the compile succeed but returns
a gradient that disagrees with the single device answer. Constraining the array
that flows into the linear transpose to replicated before the transpose does
give the correct gradient, but that path forces a gather.

## Versions

- jax 0.10.2 and jax 0.11.0, both reproduce identically
- jaxlib matches the jax version in both cases
- CPU backend, four forced host devices
- Linux x86_64, Python 3.12
</content>
