"""
Linear advection through the user-facing grid surface (task 1.7).

Hand-rolled periodic advection: FD centered/upwind tendencies with
manual RK4 stepping under a plain Python loop, exact-translation
comparison after one period, measured convergence orders across
resolutions, and the same run FV-style (``CellAvg`` +
reconstruct/flux_diff) with exact per-step conservation of the
integral.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr


def rk4(u, dt, tendency):
    k1 = tendency(u)
    k2 = tendency(u + (dt / 2) * k1)
    k3 = tendency(u + (dt / 2) * k2)
    k4 = tendency(u + dt * k3)
    return u + (dt / 6) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def one_period(u0, dt, tendency, period=1.0):
    """March one period with a jitted RK4 step in a Python loop."""
    n_steps = round(period / dt)
    dt = period / n_steps

    @jax.jit
    def step(u):
        return rk4(u, dt, tendency)

    u = u0
    for _ in range(n_steps):
        u = step(u)
    return u


# ================================================================
#  Centered FD advection (1-D): translation + convergence order
# ================================================================
def centered_error_1d(n):
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), name="x")
    grid = fr.spatial.Grid((mx,))
    u0 = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    c = 1.0

    def tendency(u):
        return -c * u.diff("x").to(mx.center)

    u = one_period(u0, 0.4 * mx.dx / c, tendency)
    # after one period the exact solution is the initial condition
    return float(jnp.abs(u.data - u0.data).max())


def test_centered_fd_advection_is_second_order_1d():
    errs = [centered_error_1d(n) for n in (16, 32, 64)]
    orders = [np.log2(a / b)
              for a, b in itertools.pairwise(errs)]
    # measured: errs ~ (0.159, 0.040, 0.010), orders ~ (1.99, 1.99)
    assert errs[0] > errs[1] > errs[2]
    assert all(order > 1.9 for order in orders)
    assert errs[-1] < 0.02


# ================================================================
#  Upwind FD advection (1-D): artificial-diffusion form
# ================================================================
def upwind_error_1d(n):
    # first-order upwind for c > 0 via the exact identity
    # (u_i - u_{i-1})/dx = centered - (dx/2) * second difference;
    # iteration 1 has no biased nodal stencils, so this is the
    # user-facing spelling of upwinding on nodal fields
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), name="x")
    grid = fr.spatial.Grid((mx,))
    u0 = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    c = 1.0
    dx = mx.dx

    def tendency(u):
        centered = u.diff("x").to(mx.center)
        d2 = u.diff("x").diff("x")
        return -c * (centered - (dx / 2) * d2)

    u = one_period(u0, 0.4 * mx.dx / c, tendency)
    err = float(jnp.abs(u.data - u0.data).max())
    peak = float(jnp.abs(u.data).max())
    return err, peak


def test_upwind_fd_advection_is_first_order_and_dissipative():
    (err_c, peak_c), (err_f, peak_f) = (
        upwind_error_1d(32), upwind_error_1d(64))
    order = np.log2(err_c / err_f)
    # measured: errs ~ (0.460, 0.265), order ~ 0.79 (pre-asymptotic
    # first order); the scheme is stable and strictly dissipative
    assert 0.6 < order < 1.2
    assert peak_c < 1.0
    assert peak_f < 1.0
    # upwind is much more diffusive than centered at equal n
    assert err_f > centered_error_1d(64)


# ================================================================
#  Centered FD advection (2-D): diagonal translation
# ================================================================
def centered_error_2d(n):
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), name="y")
    grid = fr.spatial.Grid((mx, my))
    u0 = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x)
        * jnp.cos(2 * jnp.pi * y))
    center = u0.function_space.bare

    def tendency(u):
        return -1.0 * (u.diff("x").to(center)
                       + u.diff("y").to(center))

    u = one_period(u0, 0.2 * mx.dx, tendency)
    return float(jnp.abs(u.data - u0.data).max())


def test_centered_fd_advection_is_second_order_2d():
    err_c, err_f = centered_error_2d(16), centered_error_2d(32)
    order = np.log2(err_c / err_f)
    assert err_f < err_c
    assert order > 1.9


# ================================================================
#  FV-style run: CellAvg + reconstruct/flux_diff, conservation
# ================================================================
def fv_setup(n):
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), name="x")
    grid = fr.spatial.Grid((mx,))
    q0 = grid.create_field(
        mx.cell_avg,
        init=lambda x: 1.0 + 0.5 * jnp.sin(2 * jnp.pi * x))
    c = 1.0
    flux_diff = grid.dispatch.resolve("flux_diff", mx.right)["x"]

    def tendency(q):
        # reconstruct to the faces, flux, exact flux difference
        return -1.0 * flux_diff(c * q.to(mx.right))

    return grid, mx, q0, tendency


def test_fv_advection_translation_and_order():
    def error(n):
        _, mx, q0, tendency = fv_setup(n)
        q = one_period(q0, 0.4 * mx.dx, tendency)
        return float(jnp.abs(q.data - q0.data).max())

    err_c, err_f = error(16), error(32)
    assert err_f < err_c
    assert np.log2(err_c / err_f) > 1.7


def test_fv_advection_conserves_the_integral_each_step():
    _, mx, q0, tendency = fv_setup(32)
    dt = 0.4 * mx.dx

    @jax.jit
    def step(q):
        return rk4(q, dt, tendency)

    total0 = float(q0.integrate("x").data.ravel()[0])
    q = q0
    drift = 0.0
    for _ in range(25):
        q = step(q)
        total = float(q.integrate("x").data.ravel()[0])
        drift = max(drift, abs(total - total0))
    # discrete Gauss: the flux difference telescopes exactly
    assert drift < 1e-14


def test_fv_advection_2d_conserves_the_integral():
    n = 16
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), name="y")
    grid = fr.spatial.Grid((mx, my))
    q0 = grid.create_field(
        mx.cell_avg * my.cell_avg,
        init=lambda x, y: 1.0
        + 0.25 * jnp.sin(2 * jnp.pi * (x + y)))
    fdx = grid.dispatch.resolve("flux_diff", mx.right)["x"]
    fdy = grid.dispatch.resolve("flux_diff", my.right)["y"]
    space = q0.function_space.bare

    def tendency(q):
        flux_x = 1.0 * q.to(space.replace(x=mx.right))
        flux_y = 0.5 * q.to(space.replace(y=my.right))
        return -1.0 * (fdx(flux_x) + fdy(flux_y))

    dt = 0.2 * mx.dx

    @jax.jit
    def step(q):
        return rk4(q, dt, tendency)

    total0 = float(q0.integrate().data.ravel()[0])
    q = q0
    for _ in range(10):
        q = step(q)
    total = float(q.integrate().data.ravel()[0])
    assert abs(total - total0) < 1e-14
