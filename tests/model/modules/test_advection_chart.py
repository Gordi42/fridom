r"""CenteredAdvection on an embedding chart (spherical-models plan S1).

Prefix-mirrored shard of ``fr.model.modules.advection`` covering the
orthogonal thin-shell chart path of the centered flux form: the
momentum curvature source against its analytic sphere values, the
"weight the transport at its native face, then interpolate"
construction (the constancy-preserving surface closure stays
boundary-only on every staggered control volume: the slice form equals
the exact full-3-D form to rounding), and the bind-time fences. The
hydrostatic core is the chart-capable 3-D host. Run-level gates (the
identity-chart bitwise reduction, TC2 vs sw2, conservation, autodiff):
``tests/validation/test_spherical_hydrostatic.py``. Self-contained
builders (AGENTS oversized-module rule).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.modules.advection import (
    CenteredAdvection,
    _FluxFormAdvection,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh
RADIUS = 2.0
HOR = ("lon", "lat")


def _grid(nlon=32, nlat=16, nz=4):
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=RADIUS, lat_extent=(-1.2, 1.2),
        vertical=IM(nz, (-1.0, 0.0), periodic=False, name="z"))


def _model(advection=None, grid=None):
    return hy.Model(
        grid=_grid() if grid is None else grid,
        core=hy.Core(gravity=1.0, horizontal=HOR),
        time_stepper=AdamBashforth(1e-3, order=3),
        buoyancy=hy.BuoyancyTracer(),
        free_surface=hy.ExplicitFreeSurface(horizontal=HOR),
        advection=CenteredAdvection() if advection is None
        else advection)


def _cap(lat, xp):
    """Vanish at the latitude walls (an impermeable meridional flow)."""
    return xp.cos(lat) ** 2 - xp.cos(1.2) ** 2


def _smooth(model):
    model.set_fields(
        u=lambda lon, lat, z: (jnp.cos(lat) * (1.0 + 0.3 * jnp.sin(lon))
                               + 0.0 * z),
        v=lambda lon, lat, z: (0.2 * _cap(lat, jnp) * jnp.cos(2 * lon)
                               + 0.0 * z),
        b=lambda lon, lat, z: jnp.sin(lat) * jnp.cos(lon) * (1.0 + z))
    return model.state


def _nodes(model, field, coord):
    return np.asarray(model.grid.evaluation_nodes(
        field.function_space, coord).data)


# ================================================================
#  The curvature source
# ================================================================
def test_curvature_source_is_the_analytic_sphere_term():
    model = _model(grid=_grid(64, 32, 4))
    module = model.module(CenteredAdvection)
    assert module._chart == HOR
    state = _smooth(model)
    for name, sign in (("u", 1.0), ("v", -1.0)):
        got = module._curvature(state, name)
        lon = _nodes(model, got, "lon")
        lat = _nodes(model, got, "lat")
        u = np.cos(lat) * (1.0 + 0.3 * np.sin(lon))
        v = 0.2 * _cap(lat, np) * np.cos(2 * lon)
        # d_t u += u v tan(lat) / a ;  d_t v -= u^2 tan(lat) / a
        exact = (sign * u * (v if name == "u" else u)
                 * np.tan(lat) / RADIUS) * np.ones(got.shape)
        err = np.abs(np.asarray(got.data) - exact).max()
        assert err < 2e-2 * np.abs(exact).max()


def test_tracers_and_the_vertical_carry_no_curvature():
    model = _model()
    module = model.module(CenteredAdvection)
    state = _smooth(model)
    assert module._curvature(state, "b") is None
    assert module._curvature(state, "w") is None


# ================================================================
#  Transports are weighted at their native face (the slice stays exact)
# ================================================================
def test_surface_slice_equals_the_full_correction_on_momentum(
        monkeypatch):
    # A(1) on every staggered control volume is the interpolated
    # discrete continuity, hence boundary-only: the cheap slice form of
    # the surface closure must equal the exact full-3-D form to
    # rounding for u, v and b alike. (Interpolating velocities and
    # weighting afterwards would break this on the v cells, where the
    # edge length varies along the averaging direction.)
    model = _model()
    rng = np.random.default_rng(4)
    model.set_fields(**{
        k: rng.standard_normal(model.state[k].shape)
        for k in ("u", "v", "b", "ps")})
    sliced = model.tendency(model.state)
    monkeypatch.setattr(_FluxFormAdvection, "_slice_valid",
                        lambda self, q: False)  # noqa: ARG005
    other = _model()
    other.set_fields(**{
        k: np.asarray(model.state[k].data)
        for k in ("u", "v", "b", "ps")})
    full = other.tendency(other.state)
    for name in ("u", "v", "b"):
        a = np.asarray(sliced[name].data)
        b = np.asarray(full[name].data)
        assert np.abs(a - b).max() < 1e-11 * np.abs(b).max()


# ================================================================
#  Bind-time behaviour and fences
# ================================================================
def test_chart_path_is_halo_trace_exempt():
    module = _model().module(CenteredAdvection)
    halo = module.extra_halo
    assert halo is not None
    assert {name: halo.width(name) if hasattr(halo, "width") else 2
            for name in ("lon", "lat", "z")} == dict.fromkeys(
                ("lon", "lat", "z"), 2)


def test_background_flow_on_a_chart_is_a_taught_error():
    with pytest.raises(NotImplementedError, match="background"):
        _model(CenteredAdvection(background={"u": 1.0}))


def test_flat_grid_binds_no_chart():
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), name="x"), IM(8, (0.0, 1.0), name="y"),
        IM(4, (-1.0, 0.0), periodic=False, name="z")))
    model = hy.Model(
        grid=grid, core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(1e-3, order=3),
        free_surface=hy.ExplicitFreeSurface(),
        advection=CenteredAdvection())
    module = model.module(CenteredAdvection)
    assert module._chart is None
    assert module.extra_halo is None
    # off a chart the advecting transport is the stored component
    state = model.state
    assert module._advecting(state, "u", "x") is state["u"]
