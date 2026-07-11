"""
Multi-device invariance of a full PDE run (task 1.7).

The 2-D advection-diffusion run over N steps must produce a
bitwise-equal gathered state on one device and on every available
device (genuinely sharded under the forced-4 suite), including one
mid-run ``f.reshard`` round trip and one transform round trip.

The stepping is deliberately **eager** (op-by-op, the plain Python
loop): that is the bitwise device-count-invariance contract the
operator layer guarantees. Wrapping the whole RK4 step in
``jax.jit`` compiles *different* XLA programs for the sharded and
unsharded meshes, whose fusion differs in rounding — measured
max |diff| ~4.4e-16 (about 2 ulp) after a few steps on 16^2.
Recorded as a wave-5 finding: whole-tendency jit is device-count
invariant only to rounding, not bitwise.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr

N = 16
N_STEPS = 3  # per half, around the mid-run round trips


def bitwise(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


@pytest.fixture
def device_check(forced_devices):
    if forced_devices is not None:
        # fail (not skip) when the forcing did not take effect
        assert jax.device_count() == forced_devices


def run_advection_diffusion(device_ids):
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), name="y")
    grid = fr.spatial.Grid((mx, my), device_ids=device_ids)
    u0 = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x)
        * jnp.cos(2 * jnp.pi * y))
    center = u0.function_space.bare
    t = grid.dispatch.resolve("transform", center)
    cx, cy, nu = 1.0, 0.5, 0.02
    dt = 0.2 * mx.dx

    def tendency(u):
        adv = (cx * u.diff("x").to(center)
               + cy * u.diff("y").to(center))
        dif = u.diff("x").diff("x") + u.diff("y").diff("y")
        return -1.0 * adv + nu * dif

    def step(u):
        # eager RK4: every operator application runs op-by-op, so
        # both device counts execute the identical kernel programs
        k1 = tendency(u)
        k2 = tendency(u + (dt / 2) * k1)
        k3 = tendency(u + (dt / 2) * k2)
        k4 = tendency(u + dt * k3)
        return u + (dt / 6) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    u = u0
    for _ in range(N_STEPS):
        u = step(u)
    # mid-run reshard round trip (explicit layout change and back)
    default = u.function_space.layout
    pencil = grid.decomposition.layout_for(("x",))
    u = u.reshard(pencil)
    assert u.function_space.layout == pencil
    u = u.reshard(default)
    # mid-run transform round trip
    u = t.backward(t.forward(u))
    for _ in range(N_STEPS):
        u = step(u)
    gathered = grid.decomposition.gather(u._data, u.function_space)
    return u, gathered


@pytest.mark.usefixtures("device_check")
def test_advection_diffusion_is_device_count_invariant():
    u_many, g_many = run_advection_diffusion(None)
    u_one, g_one = run_advection_diffusion((0,))
    # bitwise-equal true-shape data and gathered global state
    assert bitwise(u_many.data, u_one.data)
    assert bitwise(g_many, g_one)
    assert bitwise(g_many, u_one.data)
    # the run did something (guards a trivially-zero comparison)
    assert float(jnp.abs(u_one.data).max()) > 0.1
