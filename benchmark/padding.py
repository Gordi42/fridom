# disable jax pre memory allocation
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jaxdecomp
from functools import partial

# ================================================================
#  Initialize JAX and print the rank and size
# ================================================================

jax.distributed.initialize()

rank = jax.process_index()
size = jax.process_count()
devices = jax.devices()

if rank == 0:
    print(f"Rank: {rank}, Size: {size}")
    print(f"Devices: {devices}")

# inspect the memory usage using nvidia-smi
def print_memory_usage():
    if rank == 0:
        os.system("nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv")

print_memory_usage()

# ================================================================
#  Create a device mesh and sharding
# ================================================================
p_dims = (len(devices), 1)
devices = mesh_utils.create_device_mesh(p_dims, devices=devices)
mesh = Mesh(devices, axis_names=('x', 'y'))
pencil = P('x', 'y', None)
sharding = NamedSharding(mesh, pencil)


# ================================================================
#  Create a sharded array and print the memory usage
# ================================================================
@partial(jax.jit, out_shardings=sharding)
def create_array():
    return jnp.zeros((1024, 1024, 1024))

arr = create_array().block_until_ready()
print_memory_usage()

jax.distributed.shutdown()

# ================================================================
#  Fourier transform
# ================================================================
arr_hat = jaxdecomp.pfft3d(arr).block_until_ready()
print_memory_usage()

# ================================================================
#  Pad the array and print the memory usage
# ================================================================
# pad the array
padding = ((16, 16), (16, 16), (16, 16))
@partial(
    shard_map, mesh=mesh, in_specs=pencil, out_specs=pencil)
def pad(arr):
    return jnp.pad(arr, padding)

# arr_padded = jnp.pad(arr, padding)
arr_padded = pad(arr).block_until_ready()
print_memory_usage()




jax.distributed.shutdown()