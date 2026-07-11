"""Tests for the diffusion closures of the shallow water model."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr
import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
K = 2.0
COEFF = 0.3

CLOSURE_PARAMS = [
    pytest.param("HarmonicMixing", "kh", "p", 1, id="harmonic-mixing"),
    pytest.param("HarmonicFriction", "ah", "u", 1, id="harmonic-friction"),
    pytest.param("BiharmonicMixing", "kh", "p", 2, id="biharmonic-mixing"),
    pytest.param("BiharmonicFriction", "ah", "u", 2,
                 id="biharmonic-friction"),
]


# ================================================================
#  Helpers
# ================================================================
def setup_closure(class_name):
    closure = getattr(sw.modules.closures, class_name)(COEFF)
    grid = sw.grid.cartesian.Grid(shape=(31, 31), domain_size=(2*PI, 2*PI))
    mset = sw.ModelSettings(grid, f0=0.0)
    mset.tendencies.add_module(closure)
    mset.setup()
    return mset, closure


def eigval2(mset, axis):
    # eigenvalue of the discrete second derivative for sin(k x)
    dx = mset.grid.dx[axis]
    return (2 - 2 * jnp.cos(K * dx)) / dx**2


def apply_closure(closure, mset, target, axis):
    z = sw.State(mset)
    if target == "p":
        # the pressure field is not flagged for mixing by default
        z.p.flags = {"ENABLE_MIXING": True}
    z.fields[target].arr = jnp.sin(K * mset.grid.x_mesh[axis])
    z.sync()
    mz = fr.ModelState(mset)
    mz.z = z
    return z, closure.update(mz=mz)


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize(("class_name", "attr", "target", "power"),
                         CLOSURE_PARAMS)
def test_single_mode_decay(class_name, attr, target, power):  # noqa: ARG001
    # single fourier modes are discrete eigenfunctions:
    # df = -coeff * k_eff^(2 power) f
    mset, closure = setup_closure(class_name)

    for axis in (0, 1):
        f, mz = apply_closure(closure, mset, target, axis)
        f = f.fields[target]

        expected = -COEFF * eigval2(mset, axis)**power * f
        error = (mz.dz[target] - expected).norm_l2() / expected.norm_l2()
        assert error < 1e-12

        # only the flagged target field is diffused
        for name, df in mz.dz.fields.items():
            if name != target:
                assert jnp.abs(df.arr).max() == 0


@pytest.mark.parametrize(("class_name", "attr", "target", "power"),
                         CLOSURE_PARAMS)
def test_coefficient_property(class_name, attr, target, power):
    mset, closure = setup_closure(class_name)

    # the getter roundtrips the constructor argument
    assert getattr(closure, attr) == COEFF

    # the setter roundtrips and rescales the tendency
    setattr(closure, attr, 2 * COEFF)
    assert getattr(closure, attr) == 2 * COEFF

    f, mz = apply_closure(closure, mset, target, axis=0)
    f = f.fields[target]

    expected = -2 * COEFF * eigval2(mset, 0)**power * f
    error = (mz.dz[target] - expected).norm_l2() / expected.norm_l2()
    assert error < 1e-12
