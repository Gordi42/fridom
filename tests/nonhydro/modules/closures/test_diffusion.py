"""Tests for the diffusion closures of the nonhydrostatic model."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr
import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
K = 2.0
COEFF_H = 0.3
COEFF_V = 0.1

CLOSURE_PARAMS = [
    pytest.param("HarmonicMixing", ("kh", "kv"), "b", 1,
                 id="harmonic-mixing"),
    pytest.param("HarmonicFriction", ("ah", "av"), "u", 1,
                 id="harmonic-friction"),
    pytest.param("BiharmonicMixing", ("kh", "kv"), "b", 2,
                 id="biharmonic-mixing"),
    pytest.param("BiharmonicFriction", ("ah", "av"), "u", 2,
                 id="biharmonic-friction"),
]


# ================================================================
#  Helpers
# ================================================================
def setup_closure(class_name):
    closure = getattr(nh.modules.closures, class_name)(COEFF_H, COEFF_V)
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=0.0, stratification_n2=0.0)
    mset.tendencies.add_module(closure)
    mset.setup()
    return mset, closure


def eigval2(mset, axis):
    # eigenvalue of the discrete second derivative for sin(k x)
    dx = mset.grid.dx[axis]
    return (2 - 2 * jnp.cos(K * dx)) / dx**2


def apply_closure(closure, mset, target, axis):
    z = nh.State(mset)
    z.fields[target].arr = jnp.sin(K * mset.grid.x_mesh[axis])
    z.sync()
    mz = fr.ModelState(mset)
    mz.z = z
    return z, closure.update(mz=mz)


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize(("class_name", "attrs", "target", "power"),
                         CLOSURE_PARAMS)
def test_single_mode_decay(class_name, attrs, target, power):  # noqa: ARG001
    # single fourier modes are discrete eigenfunctions:
    # df = -coeff * k_eff^(2 power) f
    mset, closure = setup_closure(class_name)

    for axis, coeff in ((0, COEFF_H), (2, COEFF_V)):
        f, mz = apply_closure(closure, mset, target, axis)
        f = f.fields[target]

        expected = -coeff * eigval2(mset, axis)**power * f
        error = (mz.dz[target] - expected).norm_l2() / expected.norm_l2()
        assert error < 1e-12

        # only the flagged target field is diffused
        for name, df in mz.dz.fields.items():
            if name != target:
                assert jnp.abs(df.arr).max() == 0


@pytest.mark.parametrize(("class_name", "attrs", "target", "power"),
                         CLOSURE_PARAMS)
def test_coefficient_properties(class_name, attrs, target, power):
    mset, closure = setup_closure(class_name)
    attr_h, attr_v = attrs

    # the getters roundtrip the constructor arguments
    assert getattr(closure, attr_h) == COEFF_H
    assert getattr(closure, attr_v) == COEFF_V

    # the setters roundtrip and rescale the tendency
    setattr(closure, attr_h, 2 * COEFF_H)
    setattr(closure, attr_v, 3 * COEFF_V)
    assert getattr(closure, attr_h) == 2 * COEFF_H
    assert getattr(closure, attr_v) == 3 * COEFF_V

    for axis, coeff in ((0, 2 * COEFF_H), (2, 3 * COEFF_V)):
        f, mz = apply_closure(closure, mset, target, axis)
        f = f.fields[target]

        expected = -coeff * eigval2(mset, axis)**power * f
        error = (mz.dz[target] - expected).norm_l2() / expected.norm_l2()
        assert error < 1e-12
