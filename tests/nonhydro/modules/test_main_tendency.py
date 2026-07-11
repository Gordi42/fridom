"""Tests for the main tendency container of the nonhydrostatic model."""

import jax.numpy as jnp
import pytest

import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid(shape=(8, 8, 8), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=1.0, stratification_n2=4.0)
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
def test_module_order(mset):
    # the pressure solver and pressure gradient are always the last
    # two modules
    tendencies = mset.tendencies
    module_list = tendencies.module_list
    assert module_list[-2] is tendencies.pressure_solver
    assert module_list[-1] is tendencies.pressure_gradient_tendency


def test_additional_modules_are_inserted_before_the_solver(mset):
    tendencies = mset.tendencies
    module = nh.modules.closures.HarmonicFriction(ah=1.0, av=1.0)
    tendencies.add_module(module)

    module_list = tendencies.module_list
    assert module in module_list
    assert module_list.index(module) < module_list.index(
        tendencies.pressure_solver)
    assert module.is_setup


@pytest.mark.parametrize(("name", "class_path"), [
    pytest.param("linear_tendency", "LinearTendency",
                 id="linear-tendency"),
    pytest.param("advection", "advection.CenteredAdvection",
                 id="advection"),
    pytest.param("tendency_divergence", "TendencyDivergence",
                 id="tendency-divergence"),
    pytest.param("pressure_solver",
                 "pressure_solvers.SpectralPressureSolver",
                 id="pressure-solver"),
    pytest.param("pressure_gradient_tendency", "PressureGradientTendency",
                 id="pressure-gradient"),
])
def test_module_setters(mset, name, class_path):
    tendencies = mset.tendencies
    obj = nh.modules
    for part in class_path.split("."):
        obj = getattr(obj, part)
    new_module = obj()

    setattr(tendencies, name, new_module)

    assert getattr(tendencies, name) is new_module
    assert new_module in tendencies.module_list
    # replacing a module on a set up container also sets up the module
    assert new_module.is_setup


def test_advection_getter_raises_without_module(mset):
    tendencies = mset.tendencies
    tendencies._advection = None
    with pytest.raises(ValueError, match="available after the setup"):
        _ = tendencies.advection
