r"""hy.Core on an embedding chart (the thin-shell arm, plan S2).

Prefix-mirrored shard of ``hy.modules.core`` covering the orthogonal
thin-shell chart arm: the area-weighted metric continuity
``(1/sqrt_g)[d_1(h_2 u) + d_2(h_1 v)] + d_z w == 0`` (machine-exact
fundamental theorem), the physical pressure gradient ``d_i p / h_i``,
and the bind-time taught errors. The run-level gates (TC2, rest
states, conservation, the identity-chart bitwise reduction, autodiff)
live in ``tests/validation/test_spherical_hydrostatic.py``.
Self-contained builders (AGENTS oversized-module rule).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh
RADIUS = 2.0
HOR = ("lon", "lat")


def _grid(nlon=16, nlat=8, nz=4):
    return fr.spatial.spherical.Grid(
        (nlon, nlat), radius=RADIUS, lat_extent=(-1.2, 1.2),
        vertical=IM(nz, (-1.0, 0.0), periodic=False, name="z"))


def _model(grid=None, **core_kwargs):
    core_kwargs.setdefault("horizontal", HOR)
    return hy.Model(
        grid=_grid() if grid is None else grid,
        core=hy.Core(gravity=1.0, **core_kwargs),
        time_stepper=AdamBashforth(1e-3, order=3),
        buoyancy=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(horizontal=HOR))


def _random_state(model, seed=0):
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: rng.standard_normal(model.state[k].shape)
        for k in ("u", "v", "b")})
    return model.state


def test_chart_continuity_is_the_metric_divergence_to_machine_zero():
    model = _model()
    core = model.module(hy.Core)
    assert core._chart == HOR
    state = _random_state(model)
    u, v = state["u"], state["v"]
    w = core._diagnose_w(state, None)["w"]
    grid = model.grid
    # h_lat u = a u on the u faces, h_lon v = a cos(lat) v on the v faces
    lat_v = grid.evaluation_nodes(v.function_space, "lat")
    cos_v = lat_v.with_data(jnp.cos(lat_v.data))
    div = ((RADIUS * u).diff("lon")
           + (RADIUS * (v * cos_v)).diff("lat"))
    lat_c = np.asarray(grid.evaluation_nodes(
        div.function_space, "lat").data)
    div_h = np.asarray(div.data) / (RADIUS ** 2 * np.cos(lat_c))
    residual = np.asarray(w.diff("z").data) + div_h
    assert np.abs(residual).max() < 1e-12 * np.abs(div_h).max()
    # the bottom seed: no flow through the flat bottom
    assert float(jnp.abs(w.data[..., 0]).max()) == 0.0


def test_chart_pressure_gradient_is_the_physical_gradient():
    # b = cos(lat) uniform in z: p_hyd = cos(lat) * (z - 0) column sum;
    # the v tendency is -(1/a) d_lat p_hyd, the u tendency exactly 0
    model = _model(_grid(16, 32, 4))
    core = model.module(hy.Core)
    model.set_fields(b=lambda lon, lat, z: jnp.cos(lat) + 0.0 * (lon + z))
    state = model.state
    p_hyd = core._diagnose_p_hyd(state, None)["p_hyd"]
    tend = core.pressure_gradient(state.replace(p_hyd=p_hyd), None)
    # zonally uniform p_hyd: no zonal force. Exactly zero on one device;
    # on several the vertical column sums round per shard, so the
    # zonal difference is rounding-level (measured 1.1e-16), not 0.0
    assert (np.abs(np.asarray(tend["u"].data)).max()
            < 1e-13 * np.abs(np.asarray(tend["v"].data)).max())
    grid = model.grid
    space = tend["v"].function_space
    lat = np.asarray(grid.evaluation_nodes(space, "lat").data)
    z = np.asarray(grid.evaluation_nodes(space, "z").data)
    # p_hyd(z) = -int_z^0 b dz' = cos(lat) * z  ->  -(1/a) d_lat p
    exact = np.sin(lat) * z / RADIUS * np.ones(space.shape)
    err = np.abs(np.asarray(tend["v"].data) - exact).max()
    assert err < 2e-3 * np.abs(exact).max()


def test_swapped_horizontal_names_are_a_taught_error():
    grid = _grid()
    with pytest.raises(ValueError, match="does not match the grid's "
                                         "chart coordinates"):
        hy.Model(
            grid=grid,
            core=hy.Core(gravity=1.0, horizontal=("lat", "lon")),
            time_stepper=AdamBashforth(1e-3, order=3),
            free_surface=hy.ExplicitFreeSurface(horizontal=HOR))


def test_fv_family_on_a_chart_is_a_taught_error():
    with pytest.raises(NotImplementedError, match="family='fv'"):
        _model(family="fv")


def test_flat_core_carries_no_chart():
    grid = fr.spatial.Grid((
        IM(8, (0.0, 1.0), name="x"), IM(8, (0.0, 1.0), name="y"),
        IM(4, (-1.0, 0.0), periodic=False, name="z")))
    model = hy.Model(
        grid=grid, core=hy.Core(gravity=1.0),
        time_stepper=AdamBashforth(1e-3, order=3),
        free_surface=hy.ExplicitFreeSurface())
    core = model.module(hy.Core)
    assert core._chart is None
    assert core.extra_halo is None
