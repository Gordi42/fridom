"""The energy metric ``M``: apply, inner product, norm, from_model.

Covers the Phase-A surface (``design/plans/active/projection_eigenmode_
roadmap.md``): the Hermitian, positive-definite energy inner product in
both the physical/nodal (``grid.measure`` quadrature) and coefficient-
space (Parseval) reductions, the ``½⟨z,z⟩_M == ∫(ekin+epot)`` identity
against the bound diagnostics, and ``from_model`` on both concrete
models plus its structural gates.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.model.energy import (
    EnergyMetric,
    StateSourcedWeight,
    _read_scalar,
    _reciprocal,
)
from fridom.model.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)
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
        for name in ("x", "y", "z")), device_ids=(0,))


def sw_grid(n=16):
    return Grid((
        IntervalMesh(n, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(n, (0.0, 1.0), periodic=True, name="y")), device_ids=(0,))


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
    # rotation is opt-in: name the f0 = 1 f-plane the old implicit
    # coriolis=None default installed (EnergyMetric.from_model needs
    # the constant coriolis.f0 provide as its diagonalizability gate)
    model = nh.Model(grid=nh_grid(), dt=DT, advection=False,
                     coriolis=FPlaneCoriolis(f0=1.0))
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
    model = nh.Model(grid=nh_grid(), dt=DT, advection=False,
                     coriolis=FPlaneCoriolis(f0=1.0))
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
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")),
        device_ids=(0,))


def varying_sw_model(csqr_fn, grid=None):
    """Build a walled channel with a varying csqr(y) profile."""
    return sw.Model(
        grid=_walled_sw_grid() if grid is None else grid,
        csqr=csqr_fn, rossby_number=0.2, advection=False,
        coriolis=FPlaneCoriolis(f0=1.0, metric_weight="csqr"),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


def varying_nh_model(n2_fn):
    """Build a walled-y channel with a varying N^2(y) profile."""
    grid = Grid((
        IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name="z")),
        device_ids=(0,))
    return nh.Model(
        grid=grid, dt=DT, advection=False, dsqr=2.0,
        coriolis=FPlaneCoriolis(f0=1.0),
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


# ================================================================
#  Hydrostatic weights: diag(1, 1, 1/N^2, 1/c^2) on (u, v, b, ps)
# ================================================================
def hydro_grid(nx=4, nz=8, depth=1.0):
    return Grid((
        IntervalMesh(nx, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(nx, (0.0, 1.0), periodic=True, name="y"),
        IntervalMesh(nz, (0.0, depth), periodic=False, name="z")),
        device_ids=(0,))


def test_from_model_hydrostatic_weights():
    model = hy.Model(
        grid=hydro_grid(), dt=DT, csqr=10.0, advection=False,
        coriolis=hy.FPlaneCoriolis(f0=1.0),
        stratification=hy.ConstantStratification(n2=4.0),
        time_stepper=fr.model.time_steppers.CNAB2(DT))
    metric = EnergyMetric.from_model(model)
    assert metric.component_names == ("u", "v", "b", "ps")
    assert dict(metric.weights) == {
        "u": 1.0, "v": 1.0, "b": pytest.approx(0.25),
        "ps": pytest.approx(0.1)}


def test_from_model_hydrostatic_rejects_zero_phase_speed():
    params = {CORIOLIS_F0: 1.0, "hydrostatic.csqr": 0.0,
              STRATIFICATION_N2: 1.0}
    with pytest.raises(ValueError, match="1/c"):
        EnergyMetric.from_model(SimpleNamespace(parameters=params))


def test_from_model_hydrostatic_rejects_zero_stratification():
    params = {CORIOLIS_F0: 1.0, "hydrostatic.csqr": 10.0,
              STRATIFICATION_N2: 0.0}
    with pytest.raises(ValueError, match="1/N"):
        EnergyMetric.from_model(SimpleNamespace(parameters=params))


# ================================================================
#  Hydrostatic ps depth weight H/c^2 (flat, stretched-z, terrain)
# ================================================================
def _hydro_model(grid, *, csqr=3.0, n2=2.0):
    return hy.Model(
        grid=grid, dt=DT, csqr=csqr, advection=False, coriolis=None,
        stratification=hy.ConstantStratification(n2=n2),
        free_surface=hy.ExplicitFreeSurface(),
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=3))


def _terrain_hydro_grid(nx=8, nz=6, a=0.2):
    def depth(x, y):
        return 1.0 + a * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    return Grid((
        IntervalMesh(nx, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(nx, (0.0, 1.0), periodic=True, name="y"),
        IntervalMesh(nz, (-1.0, 0.0), periodic=False, name="z")),
        mapping=fr.spatial.CoordinateMapping(
            maps={"zp": lambda z, H: z * H}, params={"H": depth}),
        device_ids=(0,))


def _barotropic_bilinear_skew(metric, model):
    """(<X,LY>_M + <Y,LX>_M)/(|.|+|.|) for random barotropic states."""
    def _state(seed):
        rng = np.random.default_rng(seed)
        model.set_fields(
            u=rng.standard_normal(model.state["u"].shape),
            v=rng.standard_normal(model.state["v"].shape),
            b=np.zeros(model.state["b"].shape),
            ps=rng.standard_normal(model.state["ps"].shape))
        return model.state, model.tendency(model.state)
    x, lx = _state(101)
    y, ly = _state(202)
    xy = complex(metric.inner(x, ly))
    yx = complex(metric.inner(y, lx))
    return (xy + yx).real / (abs(xy) + abs(yx))


@pytest.mark.parametrize("depth", [1.0, 2.0, 3.0])
def test_hydrostatic_ps_weight_is_depth_over_csqr(depth):
    # the ps weight carries the physical column depth H/c^2 (a scalar
    # on a flat grid); depth != 1 was silently wrong before this weight
    model = _hydro_model(hydro_grid(depth=depth), csqr=3.0)
    metric = EnergyMetric.from_model(model, require_constant_coriolis=False)
    assert metric.weights["ps"] == pytest.approx(depth / 3.0)


@pytest.mark.parametrize("depth", [1.0, 2.0])
def test_hydrostatic_barotropic_energy_is_skew_on_a_deep_grid(depth):
    # barotropic energy conservation: with the H/c^2 ps weight the
    # bilinear skew is machine-zero at any depth (it was O(1) at depth 2
    # with the depth-blind 1/c^2 weight)
    model = _hydro_model(hydro_grid(depth=depth))
    metric = EnergyMetric.from_model(model, require_constant_coriolis=False)
    assert abs(_barotropic_bilinear_skew(metric, model)) < 1e-12


def test_hydrostatic_terrain_ps_weight_is_a_field():
    # on a terrain-following grid H(x, y) varies horizontally, so the ps
    # weight is field-valued and enters only with allow_field_weights
    model = _hydro_model(_terrain_hydro_grid())
    metric = EnergyMetric.from_model(
        model, require_constant_coriolis=False, allow_field_weights=True)
    assert isinstance(metric.weights["ps"], ScalarField)


def test_hydrostatic_terrain_ps_weight_needs_the_opt_in():
    model = _hydro_model(_terrain_hydro_grid())
    with pytest.raises(ValueError, match="varies with horizontal"):
        EnergyMetric.from_model(model, require_constant_coriolis=False)


def test_hydrostatic_terrain_barotropic_energy_is_skew():
    # the probe-proven exact barotropic physical skewness expressed
    # through the public metric: with the H(x, y)/c^2 field weight the
    # terrain barotropic subsystem conserves energy to round-off
    model = _hydro_model(_terrain_hydro_grid())
    metric = EnergyMetric.from_model(
        model, require_constant_coriolis=False, allow_field_weights=True)
    assert abs(_barotropic_bilinear_skew(metric, model)) < 1e-12


def test_hydrostatic_ps_weight_on_a_walled_channel():
    # a walled horizontal axis adds a second bounded axis; the depth
    # axis is read off ps's own ConstantSpace factor, not "the bounded
    # axis", so the channel still assembles with the H/c^2 weight
    grid = Grid((
        IntervalMesh(4, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(4, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(6, (0.0, 2.0), periodic=False, name="z")),
        device_ids=(0,))
    model = _hydro_model(grid, csqr=4.0)
    metric = EnergyMetric.from_model(model, require_constant_coriolis=False)
    assert metric.weights["ps"] == pytest.approx(2.0 / 4.0)


# ================================================================
#  State-sourced (time-dependent) field weights (TDF-D10)
# ================================================================
_CHANNEL = {"require_constant_coriolis": False, "allow_field_weights": True}


def _affine_csqr_law(c0=1.0, s=0.5):
    """c^2(y, t) = c0 + s*t + 0.1*y — a time_dependent ProfileFunction."""
    return fr.model.ProfileFunction(
        lambda y, t, c0, s: c0 + s * t + 0.1 * y, params=(c0, s))


def tracking_sw_model(grid=None, *, order=3, dt=5e-3):
    """Build a walled sw channel whose csqr is a time_dependent field."""
    return sw.Model(
        grid=_walled_sw_grid() if grid is None else grid,
        csqr=_affine_csqr_law(), rossby_number=0.2, advection=False,
        coriolis=None,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=order))


def _seed_sw(model, seed):
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in ("u", "v", "p")})


def test_static_profile_is_snapshot_agnostic():
    # a static csqr(y) profile is NOT time_dependent, so snapshot=False
    # bakes the field exactly like snapshot=True — weights and numbers
    # bit-identical, no state-sourced descriptor
    model = varying_sw_model(csqr_tanh)
    live = EnergyMetric.from_model(model, snapshot=False, **_CHANNEL)
    frozen = EnergyMetric.from_model(model, snapshot=True, **_CHANNEL)
    assert isinstance(live.weights["u"], ScalarField)
    assert isinstance(frozen.weights["u"], ScalarField)
    np.testing.assert_array_equal(
        np.asarray(live.weights["u"].data),
        np.asarray(frozen.weights["u"].data))
    _seed_sw(model, 7)
    z = sw.State({c: model.state[c] for c in ("u", "v", "p")})
    assert complex(live.inner(z, z)) == complex(frozen.inner(z, z))
    assert float(live.norm(z)) == float(frozen.norm(z))


def test_state_sourced_metric_tracks_stage_time():
    # one state-sourced metric, built once, matches a per-time baked
    # oracle at two distinct stage times — it tracks csqr off the state
    model = tracking_sw_model()
    _seed_sw(model, 3)
    metric = EnergyMetric.from_model(model, **_CHANNEL)  # snapshot=False
    assert isinstance(metric.weights["u"], StateSourcedWeight)
    assert metric.weights["u"].field == "csqr"
    assert metric.weights["v"] is metric.weights["u"]
    assert metric.weights["p"] == 1.0

    # apply resolves the descriptor off the operand's own csqr
    mz = metric.apply(model.state)
    csqr0 = model.state["csqr"]
    np.testing.assert_allclose(
        np.asarray(mz["u"].data),
        np.asarray((csqr0.to(model.state["u"]) * model.state["u"]).data))

    # sample 1
    model.advance(2)
    oracle1 = EnergyMetric.from_model(model, snapshot=True, **_CHANNEL)
    m1 = float(metric.norm(model.state))
    i1 = complex(metric.inner(model.state, model.state))
    assert m1 == pytest.approx(float(oracle1.norm(model.state)), rel=1e-12)
    assert i1 == pytest.approx(
        complex(oracle1.inner(model.state, model.state)), rel=1e-12)

    # sample 2 (later): same metric, a fresh per-time oracle
    model.advance(3)
    oracle2 = EnergyMetric.from_model(model, snapshot=True, **_CHANNEL)
    m2 = float(metric.norm(model.state))
    assert m2 == pytest.approx(float(oracle2.norm(model.state)), rel=1e-12)

    # the two samples genuinely differ (csqr and state both evolved)
    assert abs(m1 - m2) > 1e-9


def test_state_sourced_apply_missing_source_is_taught():
    # a bare (u, v, p) bundle carries the weighted u/v but not the csqr
    # source: a taught error naming the snapshot spelling
    model = tracking_sw_model()
    _seed_sw(model, 1)
    metric = EnergyMetric.from_model(model, **_CHANNEL)
    bundle = sw.State({c: model.state[c] for c in ("u", "v", "p")})
    with pytest.raises(ValueError, match="snapshot=True"):
        metric.apply(bundle)


def test_snapshot_true_freezes_the_time_dependent_weight():
    # snapshot=True reproduces the old t=0-baked behaviour: a plain
    # field weight, frozen — it does NOT track the stage time
    model = tracking_sw_model()
    _seed_sw(model, 5)
    frozen = EnergyMetric.from_model(model, snapshot=True, **_CHANNEL)
    assert isinstance(frozen.weights["u"], ScalarField)
    np.testing.assert_array_equal(
        np.asarray(frozen.weights["u"].data),
        np.asarray(model.state["csqr"].data))

    # at t=0 the state-sourced metric agrees with the frozen bake
    live = EnergyMetric.from_model(model, **_CHANNEL)
    z0 = sw.State({c: model.state[c] for c in ("u", "v", "p", "csqr")})
    assert float(live.norm(z0)) == pytest.approx(float(frozen.norm(z0)))

    # advance: the frozen metric keeps the t=0 csqr while the live one
    # tracks, so they now disagree (the staleness snapshot=True opts into)
    model.advance(4)
    assert float(frozen.norm(model.state)) != pytest.approx(
        float(live.norm(model.state)), rel=1e-9)


def _affine_n2_law(n0=1.0, s=0.5):
    """N^2(y, t) = n0 + s*t + 0.1*y — a time_dependent ProfileFunction."""
    return fr.model.ProfileFunction(
        lambda y, t, n0, s: n0 + s * t + 0.1 * y, params=(n0, s))


def tracking_nh_model(*, order=3, dt=5e-3):
    """Build a walled-y nonhydro channel whose N^2 is a time_dependent law."""
    grid = Grid((
        IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(8, (0.0, 2 * np.pi), periodic=True, name="z")),
        device_ids=(0,))
    return nh.Model(
        grid=grid, advection=False, dsqr=2.0,
        coriolis=FPlaneCoriolis(f0=1.0),
        stratification=nh.MeridionalStratification(n2=_affine_n2_law()),
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=order))


def test_state_sourced_nonhydro_reciprocal_tracks_stage_time():
    # the TDF-D10 nonhydro 1/N^2 reciprocal path, reachable since the
    # n2(y, t) law landed (2026-07-19): a real law-n2 model yields a
    # state-sourced 1/n2 b-weight that tracks the state's stage time
    # against a per-time snapshot=True oracle at two distinct times
    model = tracking_nh_model()
    metric = EnergyMetric.from_model(model, allow_field_weights=True)
    assert isinstance(metric.weights["b"], StateSourcedWeight)
    assert metric.weights["b"].field == "n2"
    assert metric.weights["w"] == pytest.approx(2.0)

    rng = np.random.default_rng(3)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in ("u", "v", "w", "b")})

    # sample 1: the single tracking metric matches a fresh per-time oracle
    model.advance(2)
    oracle1 = EnergyMetric.from_model(
        model, snapshot=True, allow_field_weights=True)
    m1 = float(metric.norm(model.state))
    assert m1 == pytest.approx(float(oracle1.norm(model.state)), rel=1e-12)

    # sample 2 (later): same metric, a fresh per-time oracle
    model.advance(3)
    oracle2 = EnergyMetric.from_model(
        model, snapshot=True, allow_field_weights=True)
    m2 = float(metric.norm(model.state))
    assert m2 == pytest.approx(float(oracle2.norm(model.state)), rel=1e-12)

    # the two samples genuinely differ (N^2 and the state both evolved)
    assert abs(m1 - m2) > 1e-9


def test_state_sourced_norm_is_differentiable():
    # the reciprocal weight 1/N^2 is a genuine divide (no seal); grad
    # through the state-sourced norm must be finite (TDF-D8 spirit). The
    # descriptor is built by hand here; the real law-n2 model path is
    # covered by test_state_sourced_nonhydro_reciprocal_tracks_stage_time.
    model = varying_nh_model(lambda y: 1.0 + 2.0 * y * y)
    metric = EnergyMetric({
        "u": 1.0, "v": 1.0, "w": 2.0,
        "b": StateSourcedWeight("n2", _reciprocal)})
    # seed nonzero fields so the energy is strictly positive (the norm's
    # sqrt has an infinite derivative at zero energy, unrelated to 1/N^2)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in ("u", "v", "w", "b")})
    base = {c: jnp.asarray(np.asarray(model.state[c].data))
            for c in ("u", "v", "w", "b")}

    def loss(scale):
        state = model.state.replace(**{
            c: model.state[c].with_data(scale * base[c])
            for c in ("u", "v", "w", "b")})
        return metric.norm(state)

    g = jax.grad(loss)(1.3)
    assert np.isfinite(float(g))
