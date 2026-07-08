"""Bitwise-equivalence guard for the S2 pressure-solve reframe.

The ``SpectralPressureSolver`` inverts the ``dsqr``-weighted discrete
Laplacian through the realized-map composition
``backward @ weighted_lap.inverse() @ forward`` (S2). These tests pin
that the composition is **bitwise-identical** (``maxdiff == 0``) to the
pre-refactor imperative body — ``transform.backward(inverse(
transform.forward(div)))`` — for the ``dsqr``-weighted symbol, on every
mode, so the pressure projection is unchanged.
"""
import jax.numpy as jnp
import numpy as np

from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver

N = 8


def make_grid(n=N, length=2 * np.pi):
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")))


def test_pressure_solve_is_bitwise_equal_to_imperative():
    # the composite ``backward @ inverse @ forward`` must reproduce the
    # pre-refactor imperative transform round-trip exactly (maxdiff == 0)
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    dsqr = jnp.asarray(0.25)
    solver = SpectralPressureSolver(
        grid, div.function_space, vertical="z")
    p_new = solver.solve(div, dsqr=dsqr)
    # reconstruct the retired imperative body on the same weighted symbol
    laplace = solver._laplacian_symbol(dsqr)
    inverse = laplace.inverse()
    transform = solver._transform
    imperative = transform.backward(inverse(transform.forward(div)))
    maxdiff = float(jnp.abs(p_new.data - imperative.data).max())
    assert maxdiff == 0.0


def test_pressure_solve_is_mean_free():
    # the k = 0 nullspace is regularized to the mean-free gauge
    grid = make_grid()
    div = grid.create_field(
        init=lambda x, y, z: jnp.sin(x) * jnp.cos(2 * y) * jnp.cos(z))
    solver = SpectralPressureSolver(
        grid, div.function_space, vertical="z")
    p = solver.solve(div, dsqr=jnp.asarray(1.0))
    assert float(jnp.abs(p.mean().data.ravel()[0])) < 1e-13
