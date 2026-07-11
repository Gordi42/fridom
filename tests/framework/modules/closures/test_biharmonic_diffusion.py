"""Tests for the biharmonic diffusion module."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
K = 2.0
KAPPA = 0.4


# ================================================================
#  Helpers
# ================================================================
def make_mset(closure=None):
    grid = fr.grid.cartesian.Grid(shape=(31, 31), domain_size=(2*PI, 2*PI))
    mset = fr.ModelSettingsBase(grid=grid)

    def _state_constructor() -> fr.VectorField:
        tracer = fr.ScalarField(mset, name="tracer",
                                flags={"ENABLE_MIXING": True})
        return fr.VectorField(mset, field_list=[tracer])

    mset.state_constructor = _state_constructor
    if closure is not None:
        mset.tendencies.add_module(closure)
    mset.setup()
    return mset


def make_closure(coefficients):
    return fr.modules.closures.BiharmonicDiffusion(
        field_flags=["ENABLE_MIXING"], diffusion_coefficients=coefficients)


def eigval4(mset, axis):
    # eigenvalue of the discrete fourth derivative for sin(k x)
    dx = mset.grid.dx[axis]
    return ((2 - 2 * jnp.cos(K * dx)) / dx**2) ** 2


def single_mode_tendency(closure, mset):
    x, _y = mset.grid.x_mesh
    mz = fr.ModelState(mset)
    f = mz.z["tracer"] + jnp.sin(K * x)
    mz.z["tracer"] += f
    mz.z.sync()
    mz = closure.update(mz=mz)
    return f, mz.dz["tracer"]


def relative_error(field, expected):
    return float((field - expected).norm_l2() / expected.norm_l2())


# ================================================================
#  Tests
# ================================================================
def test_required_halo():
    # the biharmonic operator applies two successive derivatives in each
    # direction, hence it requires two halo points
    closure = make_closure([KAPPA, KAPPA])
    mset = make_mset(closure)
    assert closure.required_halo == 2
    assert mset.grid.halo == 2


def test_single_mode_damping():
    # a single fourier mode is a discrete eigenfunction:
    # df = -kappa k_eff^4 f
    closure = make_closure([KAPPA, KAPPA])
    mset = make_mset(closure)

    f, df = single_mode_tendency(closure, mset)

    expected = -KAPPA * eigval4(mset, 0) * f
    assert relative_error(df, expected) < 1e-13


def test_negative_coefficient_antidiffuses():
    # negative coefficients invert the sign of the operator; a zero
    # coefficient does not contribute to the sign
    closure = make_closure([-KAPPA, 0.0])
    mset = make_mset(closure)

    f, df = single_mode_tendency(closure, mset)

    expected = KAPPA * eigval4(mset, 0) * f
    assert relative_error(df, expected) < 1e-13


def test_mixed_signs_raise():
    with pytest.raises(ValueError, match="mixed signs"):
        make_closure([KAPPA, -KAPPA])


def test_coefficients_roundtrip():
    # the getter returns the coefficients as they were set, not the
    # internally stored square roots
    closure = make_closure([4.0, 9.0])
    assert closure.diffusion_coefficients == [4.0, 9.0]

    closure.diffusion_coefficients = [16.0, 25.0]
    assert closure.diffusion_coefficients == [16.0, 25.0]


def test_scalar_field_coefficient():
    # a constant scalar-field coefficient must reproduce the scalar case
    closure = make_closure([KAPPA, KAPPA])
    mset = make_mset(closure)

    coeff = fr.ScalarField(mset, name="kappa") + KAPPA
    closure.diffusion_coefficients = [coeff, coeff]
    assert isinstance(closure.diffusion_coefficients[0], fr.ScalarField)

    f, df = single_mode_tendency(closure, mset)

    expected = -KAPPA * eigval4(mset, 0) * f
    assert relative_error(df, expected) < 1e-13
