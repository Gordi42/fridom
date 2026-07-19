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
    # the linear transpose of a two neighbour periodic stencil
    (cot,) = jax.linear_transpose(stencil, jax.ShapeDtypeStruct((4,), x.dtype))(x)
    return jnp.sum(cot ** 2)

print("forward runs:", float(jax.jit(loss)(x)))
print("grad builds:", np.asarray(jax.jit(jax.grad(loss), out_shardings=sharded)(x)))
