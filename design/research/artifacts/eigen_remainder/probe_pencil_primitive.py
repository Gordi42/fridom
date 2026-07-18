"""Task 2: pencil primitive probe -- two per-axis all_to_alls inside
ONE shard_map over a 2-D device mesh. Standalone jax, no fridom."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

P, Q = 2, 2  # mesh axes "p" (x) and "q" (y)
devices = np.array(jax.devices()[:P * Q]).reshape(P, Q)
mesh = jax.sharding.Mesh(devices, ("p", "q"))
print("mesh:", mesh, "axis_names:", mesh.axis_names, "shape:", mesh.shape)

nx, ny, nz = 16, 16, 8
rng = np.random.default_rng(0)
u = jnp.asarray(rng.standard_normal((nx, ny, nz))).astype(jnp.complex128)

P_spec = jax.sharding.PartitionSpec  # noqa

# input sharded: x by "p", y by "q", z local
in_spec = P_spec("p", "q", None)

def forward(block):
    # block: [nx/P, ny/Q, nz], x sharded by p, y sharded by q.
    # all_to_all over ONLY "p": split z (a LOCAL axis), gather x.
    # concat_axis=0 (x), split_axis=2 (z) -> x becomes local, z sharded by p.
    c = jax.lax.all_to_all(block, "p", split_axis=2, concat_axis=0,
                           tiled=True)
    # now x is fully local within each p-group -> fft along x
    c = jnp.fft.fft(c, axis=0, norm="forward")
    # all_to_all over ONLY "q": split z (still has a q-shardable extent?)
    # z is now sharded by p (size nz/P), still local wrt q -> split z, gather y
    c = jax.lax.all_to_all(c, "q", split_axis=2, concat_axis=1,
                           tiled=True)
    c = jnp.fft.fft(c, axis=1, norm="forward")
    return c

def backward(c):
    c = jnp.fft.ifft(c, axis=1, norm="forward")
    c = jax.lax.all_to_all(c, "q", split_axis=1, concat_axis=2, tiled=True)
    c = jnp.fft.ifft(c, axis=0, norm="forward")
    c = jax.lax.all_to_all(c, "p", split_axis=0, concat_axis=2, tiled=True)
    return c

# forward coefficient sharding after forward():
#   x local (gathered on p then fft), y local (gathered on q then fft),
#   z sharded by both p and q -> spec ("p"?..). Actually z carries both
#   splits: first split by p, then that per-p z-shard split by q. So the
#   coefficient array is (nx, ny, nz) with z sharded across BOTH axes.
coeff_spec = P_spec(None, None, ("p", "q"))

@jax.jit
def run_forward(u):
    return jax.shard_map(forward, mesh=mesh, in_specs=in_spec,
                         out_specs=coeff_spec)(u)

@jax.jit
def run_roundtrip(u):
    def body(block):
        return backward(forward(block))
    return jax.shard_map(body, mesh=mesh, in_specs=in_spec,
                         out_specs=in_spec)(u)

u_dist = jax.device_put(u, jax.sharding.NamedSharding(mesh, in_spec))

# round-trip gate
rt = run_roundtrip(u_dist)
rt_err = float(jnp.max(jnp.abs(rt - u)))
print("\nROUND-TRIP max err:", rt_err, "PASS" if rt_err < 1e-13 else "FAIL")

# forward coefficients vs replicated fftn reference (x then y only)
coeff = run_forward(u_dist)
coeff_gathered = np.asarray(jax.device_get(coeff))
ref = np.asarray(jnp.fft.fft(jnp.fft.fft(u, axis=0, norm="forward"),
                             axis=1, norm="forward"))
fwd_err = float(np.max(np.abs(coeff_gathered - ref)))
print("FORWARD vs fftn ref max err:", fwd_err,
      "PASS" if fwd_err < 1e-13 else "FAIL")

# HLO collectives by substring
lowered = jax.jit(run_roundtrip).lower(u_dist)
hlo = lowered.compile().as_text()
for sub in ("all-to-all", "all-gather", "all-reduce", "collective-permute"):
    print(f"HLO contains {sub!r}:", sub in hlo)
