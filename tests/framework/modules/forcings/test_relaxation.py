"""Tests for the relaxation module."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr

# ================================================================
#  Constants
# ================================================================
TAU = 2.0
TARGET = 1.5
INITIAL = 0.5


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(16,), domain_size=(16.0,))
    mset = fr.ModelSettingsBase(grid=grid)

    def _state_constructor() -> fr.VectorField:
        var = fr.ScalarField(mset, name="var")
        return fr.VectorField(mset, field_list=[var])

    mset.state_constructor = _state_constructor
    return mset.setup()


def domain_function(mesh):
    return mesh[0] > 8.0


def apply_relaxation(mset, target):
    relaxation = fr.modules.forcings.Relaxation(
        tau=TAU, field_name="var", target=target,
        domain_function=domain_function)
    relaxation.setup(mset=mset)

    mz = fr.ModelState(mset)
    x = mset.grid.x_mesh[0]
    mz.z["var"].arr = INITIAL * jnp.ones_like(x)
    return relaxation, mz, relaxation.update(mz=mz)


# ================================================================
#  Tests
# ================================================================
def test_relaxation_tendency(mset):
    # dz = (target - z) / tau inside the domain, zero outside
    _relaxation, _mz, result = apply_relaxation(mset, TARGET)

    x = mset.grid.x_mesh[0]
    expected = jnp.where(domain_function([x]), (TARGET - INITIAL) / TAU, 0.0)
    assert jnp.array_equal(result.dz["var"].arr, expected)


def test_target_as_scalar_field(mset):
    target = fr.ScalarField(mset, name="target")
    target.arr = TARGET * jnp.ones_like(target.arr)

    relaxation, _mz, result = apply_relaxation(mset, target)

    # the target field is stored as a plain array
    assert not isinstance(relaxation.target, fr.ScalarField)
    x = mset.grid.x_mesh[0]
    expected = jnp.where(domain_function([x]), (TARGET - INITIAL) / TAU, 0.0)
    assert jnp.array_equal(result.dz["var"].arr, expected)


def test_relaxed_field_converges_to_target(mset):
    # explicit euler steps converge towards the target value
    relaxation, mz, _result = apply_relaxation(mset, TARGET)

    dt = 0.5
    for _ in range(100):
        mz.dz["var"].arr = jnp.zeros_like(mz.dz["var"].arr)
        mz = relaxation.update(mz=mz)
        mz.z["var"].arr = mz.z["var"].arr + dt * mz.dz["var"].arr

    x = mset.grid.x_mesh[0]
    inside = domain_function([x])
    values = mz.z["var"].arr
    assert jnp.abs(jnp.where(inside, values - TARGET, 0.0)).max() < 1e-8
    assert jnp.abs(jnp.where(inside, 0.0, values - INITIAL)).max() == 0
