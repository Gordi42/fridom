"""The energy metric ``M``: apply, inner product, norm, from_model.

Covers the Phase-A surface (``design/plans/active/projection_eigenmode_
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

import fridom as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.model.energy import (
    EnergyMetric,
    _read_scalar,
)
from fridom.model.modules.coriolis import BetaPlaneCoriolis
from fridom.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.nonhydro2.diagnostics import ekin as nh_ekin
from fridom.nonhydro2.diagnostics import epot as nh_epot
from fridom.shallowwater2.diagnostics import ekin as sw_ekin
from fridom.shallowwater2.diagnostics import epot as sw_epot
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.fourier import Fourier

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
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


# --- full-complex spectral helpers (Parseval-exact) --------------
def _transform(grid):
    center = fr.spatial.Collocated().resolve(grid)
    return grid.dispatch.resolve("transform", center)


def full_complex(grid, transform, fn, name):
    """Return a full (complex) spectrum of a physical field."""
    real = grid.create_field(
        fr.spatial.Collocated().resolve(grid), init=fn, name=name)
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
    assert fr.model.EnergyMetric is EnergyMetric


def test_from_model_rejects_beta_plane():
    grid = nh_grid()
    bp = nh.Model(grid=grid, dt=DT, advection=False,
                  coriolis=BetaPlaneCoriolis(f0=1.0, beta=0.5))
    with pytest.raises(ValueError, match="coriolis"):
        EnergyMetric.from_model(bp)


def test_from_model_beta_plane_without_the_coriolis_gate():
    # the weights never involve f (rotation does no work), so a
    # consumer that tolerates a varying f — the dense-column channel
    # probe — opts out of the constancy gate and still reads csqr
    params = {"shallowwater.csqr": 4.0}
    metric = EnergyMetric.from_model(
        SimpleNamespace(parameters=params),
        require_constant_coriolis=False)
    assert metric.weights["p"] == pytest.approx(0.25)


def test_from_model_freezes_ramp_at_time():
    # a Ramp-valued dsqr must be frozen at at_time (constancy snapshot)
    ramp = fr.model.Ramp(0.0, 1.0, period=1.0)
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


def test_read_scalar_rejects_a_missing_name():
    with pytest.raises(ValueError, match="does not provide"):
        _read_scalar({}, "x", 0.0)


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


def _walled_sw_grid(n=8):
    return Grid((
        IntervalMesh(n, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")))


def varying_sw_model(csqr_fn, grid=None):
    """Build a walled channel with a varying csqr(y) profile."""
    return sw.Model(
        grid=_walled_sw_grid() if grid is None else grid,
        csqr=csqr_fn, rossby_number=0.2, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def varying_nh_model(n2_fn):
    """Build a walled-y channel with a varying N^2(y) profile."""
    grid = Grid((
        IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name="z")))
    return nh.Model(
        grid=grid, dt=DT, advection=False, dsqr=2.0,
        stratification=nh.MeridionalStratification(n2=n2_fn))


def csqr_tanh(y):
    return 1.0 + 0.5 * jnp.tanh(4.0 * (y - 0.5))


# ================================================================
#  Varying (profile-valued) weights
# ================================================================
def test_from_model_varying_shallowwater_assembles_field_weights():
    # absent scalar provide + present csqr profile field -> the
    # varying metric diag(c^2, c^2, 1), field weights on u and v
    model = varying_sw_model(csqr_tanh)
    assert "shallowwater.csqr" not in model.parameters
    metric = EnergyMetric.from_model(
        model, require_constant_coriolis=False,
        allow_field_weights=True)
    assert isinstance(metric.weights["u"], ScalarField)
    assert metric.weights["v"] is metric.weights["u"]
    assert metric.weights["p"] == 1.0


def test_from_model_varying_nonhydro_assembles_the_reciprocal():
    model = varying_nh_model(lambda y: 1.0 + 2.0 * y * y)
    assert fr.model.params.STRATIFICATION_N2 not in model.parameters
    metric = EnergyMetric.from_model(
        model, allow_field_weights=True)
    inv_n2 = metric.weights["b"]
    assert isinstance(inv_n2, ScalarField)
    n2 = model.state["n2"]
    np.testing.assert_allclose(
        np.asarray(inv_n2.data), 1.0 / np.asarray(n2.data))
    assert metric.weights["w"] == pytest.approx(2.0)


@pytest.mark.parametrize("build", [
    pytest.param(lambda: varying_sw_model(csqr_tanh), id="sw-csqr"),
    pytest.param(lambda: varying_nh_model(lambda y: 1.0 + y * y),
                 id="nh-n2"),
])
def test_from_model_varying_without_opt_in_is_a_taught_error(build):
    # the beta-style gate extended to the metric coefficients: a
    # varying csqr/n2 breaks translation invariance, so the default
    # (translation-invariant consumers) rejects with the channel hint
    with pytest.raises(ValueError, match="channel"):
        EnergyMetric.from_model(
            build(), require_constant_coriolis=False)


def test_apply_samples_a_field_weight_per_component():
    # the field weight is sampled on the component's own node set
    # through .to — u gets the broadcast centre values, v the
    # interpolated face values (the tendency's own flux sampling)
    model = varying_sw_model(csqr_tanh)
    metric = EnergyMetric.from_model(
        model, require_constant_coriolis=False,
        allow_field_weights=True)
    z = sw.State({c: model.state[c] for c in ("u", "v", "p")})
    rng = np.random.default_rng(3)
    z = sw.State({
        c: z[c].with_data(jnp.asarray(rng.standard_normal(
            np.asarray(z[c].data).shape)))
        for c in ("u", "v", "p")})
    mz = metric.apply(z)
    csqr = model.state["csqr"]
    for c in ("u", "v"):
        expect = (csqr.to(z[c]) * z[c]).data
        np.testing.assert_allclose(np.asarray(mz[c].data),
                                   np.asarray(expect))
    np.testing.assert_allclose(np.asarray(mz["p"].data),
                               np.asarray(z["p"].data))


def test_varying_constant_profile_inner_is_the_scaled_constant():
    # csqr(y) = c0 through the varying path: diag(c^2, c^2, 1) is
    # exactly c0^2 times the constant path's diag(1, 1, 1/c^2)
    c0 = 4.0
    grid = _walled_sw_grid()
    varying = varying_sw_model(lambda y: c0 + 0.0 * y, grid=grid)
    metric_v = EnergyMetric.from_model(
        varying, require_constant_coriolis=False,
        allow_field_weights=True)
    metric_c = EnergyMetric({"u": 1.0, "v": 1.0, "p": 1.0 / c0})
    rng = np.random.default_rng(9)
    z = sw.State({
        c: varying.state[c].with_data(jnp.asarray(
            rng.standard_normal(
                np.asarray(varying.state[c].data).shape)))
        for c in ("u", "v", "p")})
    lhs = complex(metric_v.inner(z, z))
    rhs = c0 * complex(metric_c.inner(z, z))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_from_model_ignores_absent_or_empty_state():
    # the varying detection degrades gracefully on models exposing
    # no csqr field: an empty or None state falls through to the
    # unrecognized-energy error, never an attribute crash
    for state in (None, {}):
        probe = SimpleNamespace(parameters={CORIOLIS_F0: 1.0},
                                state=state)
        with pytest.raises(ValueError, match="unrecognized"):
            EnergyMetric.from_model(probe)


def test_spectral_inner_rejects_a_field_weight():
    grid = sw_grid()
    transform = _transform(grid)
    csqr = grid.create_field(
        fr.spatial.Profile().resolve(grid), data=jnp.full((1, 1), 4.0),
        name="csqr")
    metric = EnergyMetric({"u": csqr})
    _, template = full_complex(
        grid, transform,
        lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y),
        "u")
    state = random_coeff_state(template, seed=6)
    with pytest.raises(NotImplementedError, match="Parseval"):
        metric.inner(state, state)


def test_inner_rejects_mixed_space():
    grid = sw_grid()
    metric = EnergyMetric({"u": 1.0})
    # transform only the x axis -> a mixed coefficient/physical space
    fourier_x = Fourier(grid, axes=("x",))
    real = grid.create_field(
        fr.spatial.Collocated().resolve(grid),
        init=lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y),
        name="u")
    mixed = fourier_x.forward(real)
    state = sw.State({"u": mixed, "v": mixed, "p": mixed})
    with pytest.raises(NotImplementedError, match="mixed"):
        metric.inner(state, state)
