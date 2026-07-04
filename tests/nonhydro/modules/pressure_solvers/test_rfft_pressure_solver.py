"""Tests for the RFFT pressure solver of the nonhydrostatic model."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

import fridom.framework as fr
import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi


# ================================================================
#  Helpers
# ================================================================
def make_mset(shape=(16, 16, 16), periodic_bounds=None):
    kwargs = {}
    if periodic_bounds is not None:
        kwargs["periodic_bounds"] = periodic_bounds
    grid = nh.grid.cartesian.Grid(
        shape=shape, domain_size=(2*PI,)*3, **kwargs)
    mset = nh.ModelSettings(grid, f0=1.0, stratification_n2=4.0)
    return mset.setup()


def check_poisson_solution(mset):
    # for div = sin(2x) sin(z), the solution of the discrete poisson
    # equation is p = -div / (k_hat²(2) + k_hat²(1))
    solver = mset.tendencies.pressure_solver
    grid = mset.grid

    mz = fr.ModelState(mset)
    x, _y, z = grid.x_mesh
    mz.z_diag.div.arr = jnp.sin(2 * x) * jnp.sin(z)
    mz.z_diag.div.sync()
    mz = solver.update(mz=mz)

    dx = grid.dx[0]
    dz = grid.dx[2]
    k2 = ((2 - 2 * jnp.cos(2 * dx)) / dx**2
          + (2 - 2 * jnp.cos(1 * dz)) / dz**2)
    expected = -mz.z_diag.div / k2
    return float((mz.z_diag.p - expected).norm_l2() / expected.norm_l2())


# ================================================================
#  Tests
# ================================================================
def test_is_default_pressure_solver():
    mset = make_mset()
    solver = mset.tendencies.pressure_solver
    assert isinstance(solver,
                      nh.modules.pressure_solvers.RFFTPressureSolver)
    assert solver.info["Solver"] == "Spectral (RFFTN)"


def test_solves_poisson_equation():
    mset = make_mset()
    solver = mset.tendencies.pressure_solver
    assert solver.rfft_axis == 2
    assert not solver.multiple_gpus
    assert check_poisson_solution(mset) < 1e-12


def test_mixed_boundaries():
    # the non-periodic axis is transformed with a cosine transform
    mset = make_mset(periodic_bounds=(True, False, True))
    solver = mset.tendencies.pressure_solver
    assert solver.dct_axes == {1}
    assert solver.fft_axes == {0, 2}
    assert check_poisson_solution(mset) < 1e-12


def test_fully_nonperiodic():
    # without periodic axes there is no rfft axis and the default
    # transform functions (pure cosine transforms) are used;
    # cos(x) cos(z) is a discrete eigenfunction of the cosine transform
    mset = make_mset(periodic_bounds=(False, False, False))
    solver = mset.tendencies.pressure_solver
    assert solver.rfft_axis is None
    assert solver.dct_axes == {0, 1, 2}

    grid = mset.grid
    mz = fr.ModelState(mset)
    x, _y, z = grid.x_mesh
    mz.z_diag.div.arr = jnp.cos(x) * jnp.cos(z)
    mz.z_diag.div.sync()
    mz = solver.update(mz=mz)

    dx = grid.dx[0]
    dz = grid.dx[2]
    k2 = ((2 - 2 * jnp.cos(dx)) / dx**2
          + (2 - 2 * jnp.cos(dz)) / dz**2)
    expected = -mz.z_diag.div / k2
    error = (mz.z_diag.p - expected).norm_l2() / expected.norm_l2()
    assert error < 1e-12


def test_single_point_axes_are_not_transformed():
    mset = make_mset(shape=(16, 1, 16))
    solver = mset.tendencies.pressure_solver
    assert solver.fft_axes == {0, 2}
    assert check_poisson_solution(mset) < 1e-12


@pytest.mark.parametrize("periodic_bounds", [
    pytest.param((True, True, True), id="periodic"),
    pytest.param((True, False, True), id="mixed"),
])
def test_multi_gpu_transform_functions(periodic_bounds):
    # the multi-gpu code path must produce the same solution; on a
    # single device the parallel transform wrappers are pass-throughs
    mset = make_mset(periodic_bounds=periodic_bounds)
    solver = mset.tendencies.pressure_solver
    solver.multiple_gpus = True
    solver._determine_rfft_axis()
    solver._setup_k_squared_inv()
    solver._setup_transform_functions()

    assert solver.rfft_axis == 2
    assert check_poisson_solution(mset) < 1e-12

    # transforming only a subset of the axes roundtrips exactly
    field = fr.ScalarField(mset, name="f")
    x, _y, z = mset.grid.x_mesh
    field.arr = jnp.sin(2 * x) * jnp.sin(z)
    field.sync()

    for axes in ({0}, {2}):
        transformed = solver.forward_transform(field.arr, axes=axes)
        restored = solver.backward_transform(transformed, axes=axes).real
        assert jnp.abs(restored - field.arr).max() < 1e-12


def test_multiple_gpus_detection():
    mset = make_mset()
    solver = mset.tendencies.pressure_solver
    assert not solver.multiple_gpus

    fake_decomp = MagicMock(spec=fr.domain_decomposition.JaxDecomposition)
    fake_decomp.n_devices = 4
    original = mset.grid._domain_decomp
    try:
        mset.grid._domain_decomp = fake_decomp
        solver._check_multiple_gpus()
    finally:
        mset.grid._domain_decomp = original

    assert solver.multiple_gpus


def test_wrong_grid_raises():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid=grid).setup()
    solver = nh.modules.pressure_solvers.RFFTPressureSolver()

    with pytest.raises(TypeError, match="only supports cartesian grids"):
        solver.setup(mset=mset)
