"""The energy metric ``M``: apply, inner product, norm, from_model.

Covers the Phase-A surface (``notes/framework2/projection_eigenmode_
roadmap.md``): the Hermitian, positive-definite energy inner product in
both the physical/nodal (``grid.measure`` quadrature) and coefficient-
space (Parseval) reductions, the ``½⟨z,z⟩_M == ∫(ekin+epot)`` identity
against the bound diagnostics, and ``from_model`` on both concrete
models plus its structural gates.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.fourier import Fourier
from fridom.framework2.model.energy import EnergyMetric, _read_scalar
from fridom.framework2.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.framework2.modules.coriolis import BetaPlaneCoriolis
from fridom.nonhydro2.diagnostics import ekin as nh_ekin
from fridom.nonhydro2.diagnostics import epot as nh_epot
from fridom.shallowwater2.diagnostics import ekin as sw_ekin
from fridom.shallowwater2.diagnostics import epot as sw_epot

DT = 0.02


# ================================================================
#  Grids / states
# ================================================================
def nh_grid(n=8, length=2 * np.pi):
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")))


def sw_grid(n=16):
    return Grid((
        IntervalMesh(n, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(n, (0.0, 1.0), periodic=True, name="y")))


def collocated_nh_state(grid):
    """Return a nonhydro State with every component on the centre."""
    def mk(fn, name):
        return grid.create_field(init=fn, name=name)
    return nh.State({
        "u": mk(lambda x, y, z: np.sin(x) * np.cos(y) * np.cos(z), "u"),
        "v": mk(lambda x, y, z: 0.3 * np.cos(x) * np.sin(y) * np.cos(z),
                "v"),
        "w": mk(lambda x, y, z: 0.2 * np.cos(x) * np.cos(y) * np.sin(z),
                "w"),
        "b": mk(lambda x, y, z: 0.5 * np.sin(x) * np.sin(y) * np.sin(z),
                "b"),
        "p": grid.create_field(name="p")})


def collocated_sw_state(grid):
    """Return a shallow-water State with every component on the centre."""
    two_pi = 2 * np.pi

    def mk(fn, name):
        return grid.create_field(init=fn, name=name)
    return sw.State({
        "u": mk(lambda x, y: np.sin(two_pi * x) * np.cos(two_pi * y),
                "u"),
        "v": mk(lambda x, y: 0.4 * np.cos(two_pi * x) * np.sin(two_pi * y),
                "v"),
        "p": mk(lambda x, y: 0.7 * np.cos(two_pi * (x + y)), "p")})


def sw_model(grid=None, *, csqr=4.0, f0=1.0):
    if grid is None:
        grid = sw_grid()
    return sw.Model(
        grid=grid, csqr=csqr, rossby_number=0.2,
        coriolis=sw.modules.FPlaneCoriolis(f0=f0),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


# --- full-complex spectral helpers (Parseval-exact) --------------
def _transform(grid):
    center = fr.Collocated().resolve(grid)
    return grid.dispatch.resolve("transform", center)


def full_complex(grid, transform, fn, name):
    """Return a full (complex) spectrum of a physical field."""
    real = grid.create_field(
        fr.Collocated().resolve(grid), init=fn, name=name)
    return real, transform.forward(real.as_complex())


def random_coeff_state(template, seed):
    """Return random complex coefficients on ``template``'s space."""
    rng = np.random.default_rng(seed)

    def draw(name):
        shape = template.data.shape
        data = (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape))
        return template.with_data(jnp.asarray(data)).with_metadata(
            name=name)
    return sw.State({c: draw(c) for c in ("u", "v", "p")})


# ================================================================
#  from_model: weights + structural gates
# ================================================================
def test_from_model_nonhydro_weights():
    model = nh.Model(grid=nh_grid(), dt=DT, advection=False)
    metric = EnergyMetric.from_model(model)
    assert metric.component_names == ("u", "v", "w", "b")
    # dsqr default 1.0, n2 default 1.0 -> all-unit weights
    assert dict(metric.weights) == {
        "u": 1.0, "v": 1.0, "w": 1.0, "b": 1.0}


def test_from_model_shallowwater_weights():
    metric = EnergyMetric.from_model(sw_model(csqr=4.0))
    assert metric.component_names == ("u", "v", "p")
    assert metric.weights["p"] == pytest.approx(0.25)


def test_from_model_is_fr_exported():
    assert fr.EnergyMetric is EnergyMetric


def test_from_model_rejects_beta_plane():
    grid = nh_grid()
    bp = nh.Model(grid=grid, dt=DT, advection=False,
                  coriolis=BetaPlaneCoriolis(f0=1.0, beta=0.5))
    with pytest.raises(ValueError, match="coriolis"):
        EnergyMetric.from_model(bp)


def test_from_model_freezes_ramp_at_time():
    # a Ramp-valued dsqr must be frozen at at_time (constancy snapshot)
    ramp = fr.Ramp(0.0, 1.0, period=1.0)
    params = {CORIOLIS_F0: 1.0, "nonhydro.dsqr": ramp,
              STRATIFICATION_N2: 1.0}
    model = SimpleNamespace(parameters=params)
    metric = EnergyMetric.from_model(model, at_time=1.0)
    # the ramp reaches its target 1.0 at t=1
    assert metric.weights["w"] == pytest.approx(1.0)


def test_from_model_rejects_zero_stratification():
    params = {CORIOLIS_F0: 1.0, "nonhydro.dsqr": 1.0,
              STRATIFICATION_N2: 0.0}
    with pytest.raises(ValueError, match="1/N"):
        EnergyMetric.from_model(SimpleNamespace(parameters=params))


def test_from_model_rejects_zero_phase_speed():
    params = {CORIOLIS_F0: 1.0, "shallowwater.csqr": 0.0}
    with pytest.raises(ValueError, match="1/c"):
        EnergyMetric.from_model(SimpleNamespace(parameters=params))


def test_from_model_rejects_unknown_energy():
    params = {CORIOLIS_F0: 1.0}
    with pytest.raises(ValueError, match="unrecognized"):
        EnergyMetric.from_model(SimpleNamespace(parameters=params))


def test_from_model_missing_stratification_scalar():
    params = {CORIOLIS_F0: 1.0, "nonhydro.dsqr": 1.0}
    with pytest.raises(ValueError, match="constant"):
        EnergyMetric.from_model(SimpleNamespace(parameters=params))


def test_read_scalar_rejects_non_scalar():
    with pytest.raises(ValueError, match="constant scalar"):
        _read_scalar({"x": object()}, "x", 0.0)


def test_metric_needs_a_component():
    with pytest.raises(ValueError, match="at least one"):
        EnergyMetric({})


# ================================================================
#  apply: M z
# ================================================================
def test_apply_scales_each_component():
    metric = EnergyMetric({"u": 1.0, "v": 2.0, "p": 0.5})
    grid = sw_grid()
    z = collocated_sw_state(grid)
    mz = metric.apply(z)
    for name, weight in (("u", 1.0), ("v", 2.0), ("p", 0.5)):
        np.testing.assert_allclose(
            np.asarray(mz[name].data),
            weight * np.asarray(z[name].data))
    # the callable spelling M(z) agrees
    np.testing.assert_allclose(
        np.asarray(metric(z)["v"].data), np.asarray(mz["v"].data))


def test_apply_ignores_unweighted_components():
    metric = EnergyMetric({"u": 3.0})
    z = collocated_sw_state(sw_grid())
    mz = metric.apply(z)
    # v and p pass through unchanged
    assert np.asarray(mz["p"].data) is not None
    np.testing.assert_allclose(
        np.asarray(mz["v"].data), np.asarray(z["v"].data))


# ================================================================
#  Physical inner product: identity vs the diagnostics
# ================================================================
def test_physical_identity_nonhydro():
    model = nh.Model(grid=nh_grid(), dt=DT, advection=False)
    metric = EnergyMetric.from_model(model)
    z = collocated_nh_state(model.grid)
    params = model.parameters
    lhs = 0.5 * float(np.real(metric.inner(z, z)))
    rhs = float((nh_ekin(z, params)
                 + nh_epot(z, params)).integrate().data.sum())
    assert lhs == pytest.approx(rhs)
    assert rhs > 0.0


def test_physical_identity_shallowwater():
    model = sw_model(csqr=4.0)
    metric = EnergyMetric.from_model(model)
    z = collocated_sw_state(model.grid)
    params = model.parameters
    lhs = 0.5 * float(np.real(metric.inner(z, z)))
    rhs = float((sw_ekin(z, params)
                 + sw_epot(z, params)).integrate().data.sum())
    assert lhs == pytest.approx(rhs)
    assert rhs > 0.0


def test_physical_norm_is_positive_definite():
    metric = EnergyMetric.from_model(sw_model())
    z = collocated_sw_state(sw_grid())
    assert float(metric.norm(z)) > 0.0
    assert float(metric.norm(z * 0.0)) == 0.0


def test_physical_inner_is_hermitian():
    metric = EnergyMetric.from_model(sw_model(csqr=4.0))
    grid = sw_grid()
    a = collocated_sw_state(grid)
    b = a.map(lambda f: f.as_complex() * (1.0 + 2.0j))
    ab = metric.inner(a, b)
    ba = metric.inner(b, a)
    assert complex(ab) == pytest.approx(complex(np.conj(ba)))


# ================================================================
#  Spectral (Parseval) inner product
# ================================================================
def test_parseval_matches_physical():
    grid = sw_grid()
    transform = _transform(grid)
    metric = EnergyMetric.from_model(sw_model(grid=grid, csqr=4.0))
    fns = {
        "u": lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y),
        "v": lambda x, y: 0.5 * np.sin(4 * np.pi * y) * np.cos(
            2 * np.pi * x),
        "p": lambda x, y: 0.3 * np.cos(2 * np.pi * (x + y))}
    phys, spec = {}, {}
    for name, fn in fns.items():
        real, coeff = full_complex(grid, transform, fn, name)
        phys[name] = real
        spec[name] = coeff
    phys_state = sw.State(phys)
    spec_state = sw.State(spec)
    phys_inner = float(np.real(metric.inner(phys_state, phys_state)))
    spec_inner = float(np.real(metric.inner(spec_state, spec_state)))
    assert spec_inner == pytest.approx(phys_inner)


def test_spectral_inner_hermitian_and_positive():
    grid = sw_grid()
    transform = _transform(grid)
    metric = EnergyMetric.from_model(sw_model(grid=grid, csqr=4.0))
    _, template = full_complex(
        grid, transform,
        lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y), "u")
    a = random_coeff_state(template, seed=1)
    b = random_coeff_state(template, seed=2)
    ab = metric.inner(a, b)
    ba = metric.inner(b, a)
    assert complex(ab) == pytest.approx(complex(np.conj(ba)))
    # positive-definite on a non-degenerate random state
    assert float(metric.norm(a)) > 0.0


def test_spectral_inner_is_linear():
    grid = sw_grid()
    transform = _transform(grid)
    metric = EnergyMetric.from_model(sw_model(grid=grid, csqr=4.0))
    _, template = full_complex(
        grid, transform,
        lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y), "u")
    a = random_coeff_state(template, seed=3)
    b = random_coeff_state(template, seed=4)
    c = random_coeff_state(template, seed=5)
    lhs = metric.inner(a, (b * 2.0) + c)
    rhs = 2.0 * metric.inner(a, b) + metric.inner(a, c)
    assert complex(lhs) == pytest.approx(complex(rhs))


def test_inner_rejects_mixed_space():
    grid = sw_grid()
    metric = EnergyMetric({"u": 1.0})
    # transform only the x axis -> a mixed coefficient/physical space
    fourier_x = Fourier(grid, axes=("x",))
    real = grid.create_field(
        fr.Collocated().resolve(grid),
        init=lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y),
        name="u")
    mixed = fourier_x.forward(real)
    state = sw.State({"u": mixed, "v": mixed, "p": mixed})
    with pytest.raises(NotImplementedError, match="mixed"):
        metric.inner(state, state)
