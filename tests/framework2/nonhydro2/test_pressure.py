"""Numerical guard for the S2 pressure-solve reframe.

The ``SpectralPressureSolver`` inverts the ``dsqr``-weighted discrete
Laplacian by delegating to ``fr``'s ``SpectralSolve`` (the realized-map
composition ``backward @ inverse @ forward``, S2). The bitwise-identity
of that composition to the retired imperative body is now pinned in the
framework's own ``test_spectral_solve``; here we pin the *physics*: the
solve drives the discrete divergence to machine zero and lands in the
mean-free gauge.

The walled sections (C7) pin the bounded-axis path: the solve runs on
the Neumann-tagged sibling of the divergence space (the DCT-II /
Cosine-II pressure parity at rigid lids), reproduces manufactured
discrete eigenfunctions to machine precision, projects a hand-made
staggered velocity divergence-free, and keeps the mean gauge — while
the periodic path stays retag-free.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.composed import (
    Divergence,
    Gradient,
    Laplacian,
)
from fridom.framework2.grid.spaces.nodal import NodeSet
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver

N = 8
NY = 6
LZ = 1.0


def make_grid(n=N, length=2 * np.pi):
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")))


def make_walled_grid(n=N, ny=NY, lz=LZ):
    # x, y periodic; z bounded (rigid lids)
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, 2 * np.pi), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, lz), periodic=False, name="z")
    return Grid((mx, my, mz)), (mx, my, mz)


def _khat(k, dx):
    """Return the staggered-difference magnitude 2 sin(k dx/2)/dx."""
    return 2.0 * np.sin(k * dx / 2.0) / dx


# ================================================================
#  Periodic grid (regression: the walled seam must not change it)
# ================================================================
def test_pressure_solve_drives_divergence_to_zero():
    # solving lap(p) = div and reapplying the weighted Laplacian must
    # recover div to machine zero -- i.e. div(u* - grad p) == 0, the
    # incompressibility constraint the projection enforces
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    dsqr = jnp.asarray(0.25)
    solver = SpectralPressureSolver(
        grid, div.function_space, vertical="z")
    p = solver.solve(div, dsqr=dsqr)
    weighted_lap = Laplacian(metric={"z": 1.0 / dsqr}).expand(
        div.function_space.bare, grid).scalar()
    residual = weighted_lap(p) - div
    maxdiff = float(jnp.abs(residual.data).max())
    assert maxdiff < 1e-12


def test_pressure_solve_is_mean_free():
    # the k = 0 nullspace is regularized to the mean-free gauge
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    solver = SpectralPressureSolver(
        grid, div.function_space, vertical="z")
    p = solver.solve(div, dsqr=jnp.asarray(1.0))
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-13


def test_periodic_solver_never_retags(monkeypatch):
    # on a fully periodic grid the Neumann sibling IS the space itself
    # (interned identity): the solve path must not touch retag at all
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    solver = SpectralPressureSolver(
        grid, div.function_space, vertical="z")
    assert solver._solve_space is div.function_space.bare

    def forbidden(self, target):  # noqa: ARG001
        raise AssertionError("retag must not run on a periodic grid")

    monkeypatch.setattr(ScalarField, "retag", forbidden)
    p = solver.solve(div, dsqr=jnp.asarray(0.25))
    assert p.function_space.bare is div.function_space.bare


# ================================================================
#  Walled grid: manufactured discrete Poisson solutions
# ================================================================
@pytest.mark.parametrize(
    ("kx", "ky", "m"),
    [
        pytest.param(1, 0, 3, id="kx1-m3"),
        pytest.param(2, 1, 1, id="kx2-ky1-m1"),
        pytest.param(0, 0, 5, id="pure-vertical"),
        pytest.param(1, 2, 0, id="barotropic-m0"),
        pytest.param(3, 2, N - 1, id="top-vertical-mode"),
    ],
)
def test_walled_manufactured_poisson_solution(kx, ky, m):
    # p_exact = cos(kx x + ky y) cos(m pi z / Lz) is a discrete
    # eigenfunction of the walled Div @ Diag @ Grad chain; feeding
    # rhs = lambda * p_exact through the solver must reproduce
    # p_exact to machine precision, with the exact trig eigenvalue
    # k_hat_z = 2 sin(pi m dz / (2 Lz)) / dz on the vertical axis
    grid, (mx, my, mz) = make_walled_grid()
    dsqr = jnp.asarray(0.25)
    dx, dy, dz = 2 * np.pi / N, 2 * np.pi / NY, LZ / N
    lam = -(_khat(kx, dx) ** 2 + _khat(ky, dy) ** 2
            + _khat(m * np.pi / LZ, dz) ** 2 / float(dsqr))
    space = mx.center * my.center * mz.center  # BC-free, as div comes

    def make_init(kx, ky, m):
        def init(x, y, z):
            return (jnp.cos(kx * x + ky * y)
                    * jnp.cos(m * np.pi * z / LZ))
        return init

    p_exact = grid.create_field(space, init=make_init(kx, ky, m))
    rhs = p_exact * lam
    solver = SpectralPressureSolver(
        grid, rhs.function_space, vertical="z")
    p = solver.solve(rhs, dsqr=dsqr)
    # the solution comes back on the caller's BC-free space
    assert p.function_space.bare is space
    assert float(jnp.abs(p.data - p_exact.data).max()) < 1e-12


def test_walled_solver_runs_on_the_neumann_sibling():
    grid, (mx, my, mz) = make_walled_grid()
    space = mx.center * my.center * mz.center
    solver = SpectralPressureSolver(grid, space, vertical="z")
    tagged = space.replace(
        z=mz.nodal(NodeSet.CENTER, bc=BC.NEUMANN))
    assert solver._solve_space is tagged
    assert solver._solve_space is not space


# ================================================================
#  Walled grid: the projection acceptance gate (no model)
# ================================================================
def test_walled_projection_drives_divergence_to_zero():
    # mirror core.py's _project on a hand-made staggered state:
    # u, v on the periodic faces, w on Inner(DIRICHLET) (the rigid-lid
    # wall-normal velocity), random interior data; after subtracting
    # the weighted pressure gradient the discrete divergence must be
    # machine zero relative to the original divergence
    grid, (mx, my, mz) = make_walled_grid()
    dsqr = jnp.asarray(0.25)
    u = grid.random.normal(mx.right * my.center * mz.center, seed=1)
    v = grid.random.normal(mx.center * my.right * mz.center, seed=2)
    w = grid.random.normal(
        mx.center * my.center
        * mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET), seed=3)
    vel = VectorField({"u": u, "v": v, "w": w})
    div = Divergence()(vel)
    solver = SpectralPressureSolver(
        div.grid, div.function_space, vertical="z")
    p = solver.solve(div, dsqr=dsqr)
    grad = Gradient()(p)
    # the vertical gradient lands on the BC-free Inner sibling;
    # adopt w's Dirichlet tag before the subtraction
    dp_dz = grad["z"].retag(w.function_space.bare)
    vel_new = VectorField({
        "u": u - grad["x"],
        "v": v - grad["y"],
        "w": w - dp_dz / dsqr,
    })
    residual = Divergence()(vel_new)
    rel = (float(jnp.abs(residual.data).max())
           / float(jnp.abs(div.data).max()))
    assert rel < 1e-12


# ================================================================
#  Walled grid: the mean gauge (the (0, 0, 0) structural zero)
# ================================================================
def test_walled_solve_lands_in_the_mean_free_gauge():
    grid, (mx, my, mz) = make_walled_grid()
    space = mx.center * my.center * mz.center
    rhs = grid.random.normal(space, seed=4)
    solver = SpectralPressureSolver(grid, space, vertical="z")
    p = solver.solve(rhs, dsqr=jnp.asarray(1.0))
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-13


def test_walled_constant_rhs_is_annihilated():
    # the constant mode is the operator's nullspace: the inverse
    # symbol is regularized to zero there, so a constant rhs returns
    # (numerically) zero pressure instead of an amplified blow-up
    grid, (mx, my, mz) = make_walled_grid()
    space = mx.center * my.center * mz.center
    rhs = grid.create_field(
        space, init=lambda x, y, z: jnp.ones_like(x))  # noqa: ARG005
    solver = SpectralPressureSolver(grid, space, vertical="z")
    p = solver.solve(rhs, dsqr=jnp.asarray(1.0))
    assert float(jnp.abs(p.data).max()) < 1e-13
