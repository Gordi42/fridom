"""Tests for the harmonic diffusion module."""

import jax.numpy as jnp

import fridom.framework as fr

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
K = 2.0
KH = 0.4
KV = 0.1


# ================================================================
#  Helpers
# ================================================================
def make_mset(closure=None):
    grid = fr.grid.cartesian.Grid(shape=(31, 31), domain_size=(2*PI, 2*PI))
    mset = fr.ModelSettingsBase(grid=grid)

    def _state_constructor() -> fr.VectorField:
        tracer = fr.ScalarField(mset, name="tracer",
                                flags={"ENABLE_MIXING": True})
        passive = fr.ScalarField(mset, name="passive")
        return fr.VectorField(mset, field_list=[tracer, passive])

    mset.state_constructor = _state_constructor
    if closure is not None:
        mset.tendencies.add_module(closure)
    mset.setup()
    return mset


def eigval2(mset, axis):
    # eigenvalue of the discrete second derivative for sin(k x)
    dx = mset.grid.dx[axis]
    return (2 - 2 * jnp.cos(K * dx)) / dx**2


def relative_error(field, expected):
    return float((field - expected).norm_l2() / expected.norm_l2())


# ================================================================
#  Tests
# ================================================================
def test_anisotropic_single_mode_decay():
    # a single fourier mode along each axis decays with the discrete
    # eigenvalue of the corresponding diffusion coefficient
    closure = fr.modules.closures.HarmonicDiffusion(
        field_flags=["ENABLE_MIXING"], diffusion_coefficients=[KH, KV])
    mset = make_mset(closure)
    x, y = mset.grid.x_mesh

    mz = fr.ModelState(mset)
    fx = mz.z["tracer"] + jnp.sin(K * x)
    fy = mz.z["tracer"] + jnp.sin(K * y)
    mz.z["tracer"] += fx + fy
    mz.z.sync()

    mz = closure.update(mz=mz)

    expected = (-KH * eigval2(mset, 0) * fx
                - KV * eigval2(mset, 1) * fy)
    assert relative_error(mz.dz["tracer"], expected) < 1e-13


def test_fields_without_flag_are_skipped():
    closure = fr.modules.closures.HarmonicDiffusion(
        field_flags=["ENABLE_MIXING"], diffusion_coefficients=[KH, KV])
    mset = make_mset(closure)
    x, _y = mset.grid.x_mesh

    mz = fr.ModelState(mset)
    mz.z["tracer"].arr = jnp.sin(K * x)
    mz.z["passive"].arr = jnp.sin(K * x)
    mz.z.sync()

    mz = closure.update(mz=mz)

    assert jnp.abs(mz.dz["tracer"].arr).max() > 0
    assert jnp.abs(mz.dz["passive"].arr).max() == 0


def test_scalar_field_coefficient():
    # a constant scalar-field coefficient is interpolated to the
    # position of the gradient and must reproduce the scalar case
    mset = make_mset()
    coeff = fr.ScalarField(mset, name="kappa") + KH
    closure = fr.modules.closures.HarmonicDiffusion(
        field_flags=["ENABLE_MIXING"], diffusion_coefficients=[coeff, KV])
    closure.setup(mset=mset)
    x, _y = mset.grid.x_mesh

    mz = fr.ModelState(mset)
    f = mz.z["tracer"] + jnp.sin(K * x)
    mz.z["tracer"] += f
    mz.z.sync()

    mz = closure.update(mz=mz)

    expected = -KH * eigval2(mset, 0) * f
    assert relative_error(mz.dz["tracer"], expected) < 1e-13


def test_properties():
    closure = fr.modules.closures.HarmonicDiffusion(
        field_flags=["ENABLE_MIXING"], diffusion_coefficients=[KH, KV])
    assert closure.field_flags == ["ENABLE_MIXING"]
    assert closure.diffusion_coefficients == [KH, KV]

    closure.field_flags = ["ENABLE_FRICTION"]
    assert closure.field_flags == ["ENABLE_FRICTION"]
