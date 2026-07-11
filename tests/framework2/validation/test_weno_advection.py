"""
WENO advection of a step (task 1.7).

Periodic transport of a square wave in FV flux form with the
upwind-biased ``WenoReconstruction``: total variation stays bounded
(no oscillations), the transport speed is correct, and the same run
with the default ``LinearReconstruction`` (centered two-point mean)
oscillates strongly — the ENO advantage.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework2 as fr

N = 64
SPEED = 1.0
FINAL_TIME = 0.25


def ssprk3(q, dt, tendency):
    q1 = q + dt * tendency(q)
    q2 = 0.75 * q + 0.25 * (q1 + dt * tendency(q1))
    return (1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dt * tendency(q2))


def total_variation(data):
    closed = jnp.concatenate([data, data[:1]])
    return float(jnp.abs(jnp.diff(closed)).sum())


@pytest.fixture
def setup():
    mx = fr.grid.meshes.IntervalMesh(N, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    # WENO5 needs halo 3: renegotiate before creating fields
    grid.negotiate(halo=fr.grid.decomposition.HaloSpec({"x": 3}))
    q0 = grid.create_field(
        mx.cell_avg,
        init=lambda x: jnp.where((x > 0.25) & (x < 0.75), 1.0, 0.0))
    return grid, mx, q0


def march(grid, mx, q0, reconstruct):
    flux_diff = grid.dispatch.resolve("flux_diff", mx.right)["x"]

    def tendency(q):
        return -1.0 * flux_diff(SPEED * reconstruct(q))

    dt = 0.4 * mx.dx / SPEED
    n_steps = round(FINAL_TIME / dt)
    dt = FINAL_TIME / n_steps

    @jax.jit
    def step(q):
        return ssprk3(q, dt, tendency)

    q = q0
    for _ in range(n_steps):
        q = step(q)
    return q


def measured_shift(q0, q):
    """Measure the translation from the first-mode phase."""
    spec0 = np.fft.fft(np.asarray(q0.data))
    spec = np.fft.fft(np.asarray(q.data))
    phase = np.angle(spec[1] / spec0[1])
    return -phase / (2 * np.pi)


def exact_translation(grid, mx):
    x = grid.evaluation_nodes(mx.cell_avg).data
    xs = (x - SPEED * FINAL_TIME) % 1.0
    return jnp.where((xs > 0.25) & (xs < 0.75), 1.0, 0.0)


def test_weno_step_transport_is_non_oscillatory(setup):
    grid, mx, q0 = setup
    # upwind for c > 0 is the left-biased instance
    weno = fr.grid.operators.WenoReconstruction(5, bias="left")["x"]
    q = march(grid, mx, q0, weno)
    # TV bounded: no oscillations beyond a tiny WENO5 tolerance
    # (measured TV growth ~1.1e-5; WENO is ENO, not strictly TVD)
    assert total_variation(q.data) < total_variation(q0.data) + 1e-4
    assert float(q.data.max()) < 1.0 + 1e-4
    assert float(q.data.min()) > -1e-4
    # correct transport speed (measured shift error ~1e-6)
    assert abs(measured_shift(q0, q) - SPEED * FINAL_TIME) < mx.dx


def test_linear_reconstruction_oscillates_weno_does_not(setup):
    grid, mx, q0 = setup
    weno = fr.grid.operators.WenoReconstruction(5, bias="left")["x"]
    q_weno = march(grid, mx, q0, weno)
    # the seeded default: centered linear reconstruction via f.to
    q_lin = march(grid, mx, q0, lambda q: q.to(mx.right))
    # measured: TV(linear) ~ 9.5 vs TV(weno) ~ 2.00001, overshoot
    # 0.28 vs 1.2e-6
    assert total_variation(q_lin.data) > 4.0
    assert float(q_lin.data.max()) > 1.1
    exact = exact_translation(grid, mx)
    err_weno = float(jnp.abs(q_weno.data - exact).mean())
    err_lin = float(jnp.abs(q_lin.data - exact).mean())
    assert err_weno < 0.5 * err_lin
