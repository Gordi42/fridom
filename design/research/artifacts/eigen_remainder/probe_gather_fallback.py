"""Task 3: universal gather fallback probe -- gather the sharded
periodic axis inside a shard_map, run the single-device contraction
per device on the gathered block, slice the local shard back out.
Standalone jax, 1-D mesh of 4 forced-CPU devices."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

Pn = 4
mesh = jax.sharding.Mesh(np.array(jax.devices()[:Pn]), ("devices",))
PS = jax.sharding.PartitionSpec

nx, nz = 16, 8         # x periodic (sharded), z bounded (local)
kx = nx // 2 + 1       # rfft half spectrum along x
J = nz                 # column count (one mode column per bounded DOF)

rng = np.random.default_rng(1)
u = jnp.asarray(rng.standard_normal((nx, nz)))
# replicated basis / weights / metric (dynamic apply-time args)
q = jnp.asarray(rng.standard_normal((kx, nz, J))
                + 1j * rng.standard_normal((kx, nz, J)))
w = jnp.asarray(rng.standard_normal((kx, J)) + 1j * rng.standard_normal((kx, J)))
metric = jnp.asarray(rng.standard_normal((nz,)))

def contract_single(block, q, w, metric):
    """The existing single-device contraction math, on a full x block."""
    c = jnp.fft.rfft(block, axis=0, norm="forward")          # (kx, nz)
    amp = jnp.einsum("kdj,d,kd->kj", jnp.conj(q), metric, c)  # (kx, J)
    out = jnp.einsum("kdj,kj->kd", q, w * amp)                # (kx, nz)
    return jnp.fft.irfft(out, n=nx, axis=0, norm="forward").real

# reference: fully replicated single device
ref = contract_single(u, q, w, metric)

# gather fallback: x sharded, gather inside shard_map, contract, slice
in_spec = PS("devices", None)
def gather_body(block, q, w, metric):
    full = jax.lax.all_gather(block, "devices", axis=0, tiled=True)  # (nx, nz)
    res = contract_single(full, q, w, metric)                       # (nx, nz)
    # slice the local x shard back out
    idx = jax.lax.axis_index("devices")
    per = nx // Pn
    return jax.lax.dynamic_slice_in_dim(res, idx * per, per, axis=0)

@jax.jit
def run(u, q, w, metric):
    return jax.shard_map(
        gather_body, mesh=mesh,
        in_specs=(in_spec, PS(), PS(), PS()), out_specs=in_spec)(
            u, q, w, metric)

u_dist = jax.device_put(u, jax.sharding.NamedSharding(mesh, in_spec))
out = run(u_dist, q, w, metric)
out_g = np.asarray(jax.device_get(out))
err = float(np.max(np.abs(out_g - np.asarray(ref))))
print("GATHER-FALLBACK vs single-device ref max err:", err,
      "PASS" if err < 1e-13 else "FAIL")

hlo = jax.jit(run).lower(u_dist, q, w, metric).compile().as_text()
for sub in ("all-to-all", "all-gather", "all-reduce", "collective-permute"):
    print(f"HLO contains {sub!r}:", sub in hlo)
