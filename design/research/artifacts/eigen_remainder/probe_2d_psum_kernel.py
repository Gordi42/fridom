"""Task 2: standalone pure-jax psum kernel for the 2-D-channel contraction.

Geometry: real u[n_x, n_z]; periodic x SHARDED over 1-D mesh "s" (P=4);
bounded z local. The bounded axis is used as the transpose partner, so the
per-plane column contraction (sum over d = z) becomes a distributed psum
over the bounded-axis shards. Reference = the same math single-device.
"""
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from functools import partial

P_DEV = 4
mesh = Mesh(np.array(jax.devices()[:P_DEV]), axis_names=("s",))
print("device_count:", jax.device_count(), " mesh:", mesh.shape)


def ceil_mult(n, s):
    return -(-n // s) * s


def tail_pad(a, axis, count):
    pads = [(0, 0)] * a.ndim
    pads[axis] = (0, count)
    return jnp.pad(a, pads)


def build_kernel(n_x, n_z, J, Pn=P_DEV):
    pad_x = ceil_mult(n_x, Pn)
    pad_z = ceil_mult(n_z, Pn)
    n_kx = n_x // 2 + 1

    def body(u, q, w, metric):
        # u local: (pad_x/P, n_z) real
        c = tail_pad(u, 1, pad_z - n_z)                    # (pad_x/P, pad_z)
        c = jax.lax.all_to_all(c, "s", split_axis=1, concat_axis=0,
                               tiled=True)                 # (pad_x, pad_z/P)
        if pad_x != n_x:
            c = c[:n_x]                                    # unpad outer x
        c = jnp.fft.rfft(c, axis=0, norm="forward")        # (n_kx, pad_z/P)
        # contraction over local d (=z shard); psum over "s"
        amp_p = jnp.einsum("kdj,d,kd->kj", jnp.conj(q), metric, c)
        amp = jax.lax.psum(amp_p, "s")                     # (n_kx, J) replic.
        out = jnp.einsum("kdj,kj->kd", q, w * amp)         # (n_kx, pad_z/P)
        r = jnp.fft.irfft(out, n=n_x, axis=0, norm="forward").real
        if pad_x != n_x:
            r = tail_pad(r, 0, pad_x - n_x)                # (pad_x, pad_z/P)
        r = jax.lax.all_to_all(r, "s", split_axis=0, concat_axis=1,
                               tiled=True)                 # (pad_x/P, pad_z)
        return r[:, :n_z]                                  # (pad_x/P, n_z)

    fn = jax.jit(jax.shard_map(
        body, mesh=mesh,
        in_specs=(P("s", None), P(None, "s", None), P(None, None), P("s")),
        out_specs=P("s", None)))
    return fn, pad_x, pad_z, n_kx


def run_variant(n_x, n_z, J, seed):
    fn, pad_x, pad_z, n_kx = build_kernel(n_x, n_z, J)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n_x, n_z))
    q_true = (rng.standard_normal((n_kx, n_z, J))
              + 1j * rng.standard_normal((n_kx, n_z, J)))
    metric_true = rng.uniform(0.5, 1.5, n_z)
    w = (rng.standard_normal((n_kx, J))
         + 1j * rng.standard_normal((n_kx, J)))

    # reference (single device)
    C = np.fft.rfft(u, axis=0, norm="forward")
    amp = np.einsum("kdj,d,kd->kj", np.conj(q_true), metric_true, C)
    out = np.einsum("kdj,kj->kd", q_true, w * amp)
    ref = np.fft.irfft(out, n=n_x, axis=0, norm="forward").real

    # padded operands for the shard_map (zero pad on the d=z axis)
    u_pad = np.pad(u, ((0, pad_x - n_x), (0, 0)))
    q_pad = np.pad(q_true, ((0, 0), (0, pad_z - n_z), (0, 0)))
    m_pad = np.pad(metric_true, (0, pad_z - n_z))

    u_j = jax.device_put(jnp.asarray(u_pad),
                         jax.sharding.NamedSharding(mesh, P("s", None)))
    res = np.asarray(fn(u_j, jnp.asarray(q_pad), jnp.asarray(w),
                        jnp.asarray(m_pad)))
    res = res[:n_x]  # unpad outer x
    diff = float(np.abs(res - ref).max())
    is_real = not np.iscomplexobj(res)
    return diff, is_real, fn, (u_j, q_pad, w, m_pad)


variants = [
    ("divisible          n_x=16 n_z=8  J=8", 16, 8, 8, 1),
    ("indiv. bounded z    n_x=16 n_z=7  J=8", 16, 7, 8, 2),
    ("indiv. sharded x    n_x=18 n_z=7  J=8", 18, 7, 8, 3),
]
print("\n=== shape variants (gate max|diff| < 1e-13) ===")
last = None
for label, nx, nz, J, seed in variants:
    diff, is_real, fn, args = run_variant(nx, nz, J, seed)
    ok = "PASS" if diff < 1e-13 else "FAIL"
    print(f"  {label}: max|diff|={diff:.3e} real={is_real} [{ok}]")
    last = (fn, args)

# ---- HLO collective summary (use the divisible variant) ----
fn, args = build_kernel(16, 8, 8)[0], None
diff, is_real, fn, args = run_variant(16, 8, 8, 1)
hlo = fn.lower(*args).compile().as_text()
print("\n=== HLO collective substrings (compiled) ===")
for sub in ("all-to-all", "all-reduce", "all-gather", "collective-permute"):
    print(f"  present: {sub:20s} -> {sub in hlo}")
