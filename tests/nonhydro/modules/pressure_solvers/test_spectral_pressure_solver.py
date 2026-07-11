"""Tests for the spectral pressure solver of the nonhydrostatic model."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr
import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi


# ================================================================
#  Tests
# ================================================================
def test_cartesian_grid_solves_discrete_poisson_equation():
    # the pressure must satisfy the discrete poisson equation
    # laplacian(p) = div, where the laplacian is built from the same
    # finite differences that the model uses
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=1.0, stratification_n2=4.0).setup()
    solver = nh.modules.pressure_solvers.SpectralPressureSolver()
    solver.setup(mset=mset)

    mz = fr.ModelState(mset)
    x, _y, z = grid.x_mesh
    mz.z_diag.div.arr = jnp.sin(2 * x) * jnp.sin(z)
    mz.z_diag.div.sync()

    mz = solver.update(mz=mz)

    p = mz.z_diag.p.sync()
    diff = grid.diff_module
    laplacian = None
    for axis in range(3):
        gradient = diff.diff(p, axis=axis).sync()
        second_derivative = diff.diff(gradient, axis=axis)
        if laplacian is None:
            laplacian = second_derivative
        else:
            laplacian = laplacian + second_derivative

    div = mz.z_diag.div
    assert (laplacian - div).norm_l2() / div.norm_l2() < 1e-12


def test_unsupported_grid_raises():
    # the framework cartesian grid is not a nonhydro grid
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid=grid).setup()
    solver = nh.modules.pressure_solvers.SpectralPressureSolver()

    with pytest.raises(ValueError, match="does not support this grid type"):
        solver.setup(mset=mset)


def test_info():
    solver = nh.modules.pressure_solvers.SpectralPressureSolver()
    assert solver.info["Solver"] == "Spectral"
