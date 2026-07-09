"""Numerical guard for the S2 pressure-solve reframe.

The ``SpectralPressureSolver`` inverts the ``dsqr``-weighted discrete
Laplacian by delegating to ``fr``'s ``SpectralSolve`` (the realized-map
composition ``backward @ inverse @ forward``, S2). The bitwise-identity
of that composition to the retired imperative body is now pinned in the
framework's own ``test_spectral_solve``; here we pin the *physics*: the
solve drives the discrete divergence to machine zero and lands in the
mean-free gauge.
"""
import jax.numpy as jnp
import numpy as np

from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.composed import Laplacian
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver

N = 8


def make_grid(n=N, length=2 * np.pi):
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")))


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
