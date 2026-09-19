r"""
Continents on the spherical hydrostatic model (chart + immersed).

The thin-shell chart arm composed with the immersed (cut-cell)
machinery in its **full-cell staircase** form
(``ImmersedDomain(order=None)``): a land/sea mask and z-level
bathymetry on the ``(lon, lat, z)`` sphere. The open-area fractions
weight the *area-weighted* transports (``alpha h_j U_i``) and the wet
fraction divides next to ``sqrt_g``, so the two compose without a new
code path. Gates (measured on CPU, float64):

- **All-wet mask == no mask**: tendencies agree to 1 ULP (measured
  2.2e-16 relative on ``u``/``v`` — the immersed momentum takes the
  exact full-3-D surface closure instead of the slice form — and
  bitwise on ``b``/``ps``); a 50-step run agrees to 5e-16.
- **Closed basins**: the wet sqrt(g)-weighted tracer content and the
  ``ps`` volume are conserved to rounding (measured 3.6e-17 / 5.8e-17
  over 300 steps of a stirred flow around a continent, an island and a
  seamount); a constant tracer stays constant exactly.
- **Rest state over bathymetry**: a stratified ocean at rest over the
  staircase stays at rest to rounding (measured max |u| 8e-17 after
  100 steps) — z-levels carry no pressure-gradient error.
- **Dry DOFs stay dead** (exactly zero).
- **Temperature + salinity + nonlinear EOS** (``hy.TemperatureSalinity``
  with the Roquet EOS and the pairwise convective adjustment) composes
  with the chart + immersed arm: a resting ``T(z)``, ``S(z)`` ocean
  over the staircase stays at rest to rounding and a stirred run keeps
  a uniform salinity uniform.
- **Split-explicit free surface** around continents: ``ps`` volume
  conserved to rounding (measured -1.8e-16), rest state to rounding.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.immersed_domain import ImmersedDomain

IM = fr.spatial.meshes.IntervalMesh
CenteredAdvection = fr.model.modules.CenteredAdvection
HOR = ("lon", "lat")
LAT_MAX = float(np.deg2rad(80.0))
NAMES = ("u", "v", "b", "ps")


def all_wet(lon, lat, z):
    return 1.0 + 0.0 * (lon + lat + z)


def world(lon, lat, z):
    """Return a continent, an island and a seamount (z-level staircase)."""
    depth = 1.0 - 0.6 * jnp.exp(-((lon - 4.0) ** 2 + lat ** 2) / 0.3)
    land = (jnp.abs(lat) < 0.7) & (lon > 1.0) & (lon < 2.0)
    land = land | ((lat > 0.9) & (lon > 4.5) & (lon < 5.5))
    return jnp.where(land | (z < -depth), 0.0, 1.0)


def build(indicator, *, surface_flux=None, buoyancy=None, split=0,
          dt=2e-3, modules_extra=()):
    grid = fr.spatial.spherical.Grid(
        (32, 16), radius=1.0, lat_extent=(-LAT_MAX, LAT_MAX),
        vertical=IM(6, (-1.0, 0.0), periodic=False, name="z"),
        immersed=(None if indicator is None
                  else ImmersedDomain(indicator)))
    return hy.Model(
        grid=grid, core=hy.Core(gravity=2.0, horizontal=HOR),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=fr.model.modules.RotationCoriolis((0.0, 0.0, 2.0)),
        buoyancy=hy.BuoyancyTracer() if buoyancy is None else buoyancy,
        free_surface=(
            hy.SplitExplicitFreeSurface(substeps=split, horizontal=HOR)
            if split else hy.ExplicitFreeSurface(horizontal=HOR)),
        advection=CenteredAdvection(surface_flux=surface_flux),
        modules_extra=modules_extra)


def seed(model, seed_value=0):
    rng = np.random.default_rng(seed_value)
    model.set_fields(**{
        k: 0.1 * rng.standard_normal(model.state[k].shape)
        for k in NAMES})


def content(field, theta=None):
    """Return the (wet) sqrt(g)-weighted sum of a field."""
    data = np.asarray(field.data)
    if theta is not None:
        data = data * np.asarray(theta.data)
    sqrt_g = np.asarray(field.grid.metric(
        field.function_space.bare, "sqrt_g").data)
    return float((sqrt_g * data).sum())


def test_all_wet_mask_reproduces_the_unmasked_sphere():
    masked, plain = build(all_wet), build(None)
    seed(masked)
    seed(plain)
    with jax.disable_jit():
        dm = masked.tendency(masked.state)
        dp = plain.tendency(plain.state)
    for name in ("b", "ps"):
        assert np.array_equal(np.asarray(dm[name].data),
                              np.asarray(dp[name].data))
    for name in ("u", "v"):
        a, b = np.asarray(dm[name].data), np.asarray(dp[name].data)
        assert np.abs(a - b).max() < 4e-16 * np.abs(b).max()
    masked.advance(50)
    plain.advance(50)
    for name in NAMES:
        a = np.asarray(masked.state[name].data)
        b = np.asarray(plain.state[name].data)
        assert np.abs(a - b).max() < 1e-13 * np.abs(b).max()


def test_closed_basins_conserve_tracer_and_volume_to_rounding():
    model = build(world, surface_flux=False)
    seed(model, 3)
    model.advance(1)  # MaskState has zeroed the dry DOFs
    theta = model.grid.immersed.fraction(
        model.state["b"].function_space)
    wet_fraction = float(np.asarray(theta.data).mean())
    assert 0.8 < wet_fraction < 0.9  # continents and a seamount exist
    b0 = content(model.state["b"], theta)
    b_scale = content(abs(model.state["b"]), theta)
    p0 = content(model.state["ps"])
    p_scale = content(abs(model.state["ps"]))
    model.advance(300)
    assert not model.panicked
    assert (abs(content(model.state["b"], theta) - b0)
            < 1e-13 * b_scale)
    assert abs(content(model.state["ps"]) - p0) < 1e-13 * p_scale


def test_constant_tracer_stays_constant_around_continents():
    model = build(world)
    seed(model, 3)
    model.set_fields(b=0.37 * np.ones(model.state["b"].shape))
    model.advance(300)
    theta = np.asarray(model.grid.immersed.fraction(
        model.state["b"].function_space).data)
    wet = np.asarray(model.state["b"].data)[theta > 0]
    assert np.abs(wet - 0.37).max() < 1e-13


def test_resting_stratified_ocean_over_bathymetry_stays_at_rest():
    model = build(world)
    z = np.asarray(model.grid.evaluation_nodes(
        model.state["b"].function_space, "z").data)
    model.set_fields(b=np.exp(2 * z) * np.ones(model.state["b"].shape))
    model.advance(100)
    for name in ("u", "v", "w", "ps"):
        assert np.abs(np.asarray(model.state[name].data)).max() < 1e-14


@pytest.mark.parametrize("name", ["u", "v", "b"])
def test_dry_dofs_stay_dead(name):
    model = build(world)
    seed(model, 5)
    model.advance(50)
    theta = np.asarray(model.grid.immersed.fraction(
        model.state[name].function_space).data)
    dry = np.asarray(model.state[name].data)[theta == 0]
    assert dry.size > 0
    assert np.abs(dry).max() == 0.0


# ================================================================
#  Temperature + salinity + nonlinear EOS on the masked sphere
# ================================================================
def _ts_model():
    return build(world, buoyancy=hy.TemperatureSalinity(
        hy.RoquetEOS(), convective_adjustment=2))


def test_resting_temperature_salinity_ocean_stays_at_rest():
    model = _ts_model()
    model.set_fields(
        T=lambda lon, lat, z: 2.0 + 20.0 * jnp.exp(3 * z) + 0.0 * (lon + lat),
        S=lambda lon, lat, z: 35.0 - 0.5 * jnp.exp(3 * z) + 0.0 * (lon + lat))
    model.advance(100)
    assert not model.panicked
    for name in ("u", "v", "w", "ps"):
        assert np.abs(np.asarray(model.state[name].data)).max() < 1e-13


def test_stirred_temperature_salinity_run_keeps_uniform_salinity():
    model = _ts_model()
    rng = np.random.default_rng(9)
    model.set_fields(
        u=0.1 * rng.standard_normal(model.state["u"].shape),
        v=0.1 * rng.standard_normal(model.state["v"].shape),
        T=lambda lon, lat, z: (2.0 + 20.0 * jnp.exp(3 * z)
                               * jnp.cos(lat) ** 2 + 0.0 * lon),
        S=35.0 * np.ones(model.state["S"].shape))
    model.advance(200)
    assert not model.panicked
    theta = np.asarray(model.grid.immersed.fraction(
        model.state["S"].function_space).data)
    wet = np.asarray(model.state["S"].data)[theta > 0]
    assert np.abs(wet - 35.0).max() < 1e-11
    temperature = np.asarray(model.state["T"].data)[theta > 0]
    assert np.isfinite(temperature).all()


# ================================================================
#  Split-explicit free surface around continents
# ================================================================
def test_split_explicit_conserves_volume_around_continents():
    model = build(world, split=20, dt=1e-2)
    seed(model, 1)
    model.advance(1)
    before = content(model.state["ps"])
    scale = content(abs(model.state["ps"]))
    model.advance(100)
    assert not model.panicked
    # measured -1.8e-16
    assert abs(content(model.state["ps"]) - before) < 1e-13 * scale


def test_split_explicit_rest_state_over_bathymetry():
    model = build(world, split=20, dt=1e-2)
    z = np.asarray(model.grid.evaluation_nodes(
        model.state["b"].function_space, "z").data)
    model.set_fields(b=np.exp(2 * z) * np.ones(model.state["b"].shape))
    model.advance(50)
    for name in ("u", "v", "w", "ps"):
        assert np.abs(np.asarray(model.state[name].data)).max() < 1e-13


# ================================================================
#  Biharmonic closures around continents (chart + immersed staircase)
# ================================================================
def test_biharmonic_closures_around_continents():
    # the scale-selective closures of an eddying global run: on the
    # sphere WITH continents the iterated Laplace-Beltrami pass keeps
    # dry DOFs dead, conserves the wet sqrt(g)-weighted tracer content
    # and dissipates variance (closure tendency isolated by differencing
    # against the closure-free model on the same state)
    closures = [
        fr.model.closures.BiharmonicDiffusion(kappa=2e-4, kappa_v=0.0),
        fr.model.closures.BiharmonicFriction(nu=1e-4, nu_v=0.0)]
    with_c = build(world, modules_extra=closures)
    without = build(world)
    seed(with_c, 3)
    seed(without, 3)
    full = with_c.tendency(with_c.state)
    base = without.tendency(without.state)
    immersed = with_c.grid.immersed
    for name in ("u", "v", "b"):
        field = with_c.state[name]
        space = field.function_space
        theta = np.asarray(immersed.fraction(space).data)
        weight = theta * np.asarray(
            with_c.grid.metric(space.bare, "sqrt_g").data)
        # two grids: difference the raw arrays
        closure = np.asarray(full[name].data) - np.asarray(base[name].data)
        assert np.abs(closure).max() > 0.0
        assert np.abs(closure[theta == 0.0]).max() == 0.0
        rate = float((weight * np.asarray(field.data) * closure).sum())
        assert rate < 0.0, name
        if name == "b":
            drift = float((weight * closure).sum())
            scale = float((weight * np.abs(closure)).sum())
            assert abs(drift) < 1e-13 * scale
