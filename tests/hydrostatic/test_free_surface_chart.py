r"""hy.ExplicitFreeSurface on an embedding chart (plan S2).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
orthogonal thin-shell chart arm of the explicit free surfaces: the
area-weighted barotropic transport divergence closed by ``sqrt_g``
(volume-conserving to rounding), its exact adjointness to the physical
surface-pressure gradient under the ``sqrt_g`` measure, and the
bind-time taught errors (the implicit variant stays refused on a chart
until the chart Helmholtz of plan S3), and the **split-explicit**
subcycle on the chart — an explicit 2-D pair that needs no elliptic
operator (volume conservation, the identity-chart reduction). Self-contained
builders (AGENTS oversized-module rule).
"""
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

IM = fr.spatial.meshes.IntervalMesh
HOR = ("lon", "lat")
GRAVITY = 3.0
DEPTH, NZ = 1.5, 4


def _grid():
    return fr.spatial.spherical.Grid(
        (16, 8), radius=2.0, lat_extent=(-1.2, 1.2),
        vertical=IM(NZ, (-DEPTH, 0.0), periodic=False, name="z"))


def _model(free_surface=None):
    if free_surface is None:
        free_surface = hy.ExplicitFreeSurface(horizontal=HOR)
    return hy.Model(
        grid=_grid(), core=hy.Core(gravity=GRAVITY, horizontal=HOR),
        time_stepper=AdamBashforth(1e-3, order=3),
        free_surface=free_surface)


def _weighted_sum(field, other=None):
    """Return the sqrt(g) dV-weighted sum of ``field`` (* ``other``).

    The meshes are uniform, so the constant ``dlon dlat`` factor drops
    out of every comparison; a 3-D field carries its ``dz``, the 2-D
    (z-reduced) ``ps`` cell does not.
    """
    weight = np.asarray(field.grid.metric(
        field.function_space.bare, "sqrt_g").data)
    data = np.asarray(field.data)
    if other is not None:
        data = data * np.asarray(other.data)
    dz = DEPTH / NZ if data.shape[-1] > 1 else 1.0
    return float((weight * data).sum() * dz)


def _random(model, seed=0):
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        k: rng.standard_normal(model.state[k].shape)
        for k in ("u", "v", "ps")})
    return model.state


def test_chart_gravity_term_conserves_volume_to_rounding():
    model = _model()
    module = model.module(hy.ExplicitFreeSurface)
    assert module._chart == HOR
    state = _random(model)
    div = module._transport_div(state)
    assert abs(_weighted_sum(div)) < 1e-13 * _weighted_sum(abs(div))


def test_chart_gravity_pair_is_skew_adjoint_under_the_area_measure():
    # <u, -grad ps> + <ps, -T*> = 0 under sqrt(g) dlon dlat dz: the
    # area-weighted divergence is the exact adjoint of the physical
    # gradient (the energy pairing of the barotropic gravity wave)
    model = _model()
    module = model.module(hy.ExplicitFreeSurface)
    state = _random(model, seed=2)
    grad = module.pressure_gradient(state, None)
    div = module._transport_div(state)
    work = (_weighted_sum(grad["u"], state["u"])
            + _weighted_sum(grad["v"], state["v"]))
    # T* is the z-integrated divergence on the 2-D ps cell: its pairing
    # carries no dz weight
    pairing = -_weighted_sum(div, state["ps"])
    scale = abs(work) + abs(pairing)
    assert abs(work + pairing) < 1e-12 * scale


def test_implicit_variant_stays_refused_on_a_chart():
    # the implicit barotropic solve needs the chart Helmholtz (plan S3)
    with pytest.raises(NotImplementedError,
                       match="curvilinear / spherical"):
        _model(hy.ImplicitFreeSurface(horizontal=HOR))


# ================================================================
#  The split-explicit subcycle on the chart (no elliptic operator)
# ================================================================
def _split(substeps=8):
    return hy.SplitExplicitFreeSurface(substeps=substeps, horizontal=HOR)


def test_split_explicit_binds_the_chart_with_a_two_cell_halo():
    module = _model(_split()).module(hy.SplitExplicitFreeSurface)
    assert module._chart == HOR
    halo = module.extra_halo
    assert halo == type(halo)(dict.fromkeys(HOR, 2))


def test_split_explicit_subcycle_conserves_volume_on_the_sphere():
    model = _model(_split())
    _random(model, seed=4)
    model.set_fields(ps=0.1 * np.asarray(model.state["ps"].data))
    before = _weighted_sum(model.state["ps"])
    scale = _weighted_sum(abs(model.state["ps"]))
    model.advance(40)
    assert not model.panicked
    # measured 3e-17: the substep divergence is the area-weighted
    # metric divergence, whose sqrt(g)-weighted sum telescopes
    assert abs(_weighted_sum(model.state["ps"]) - before) < 1e-13 * scale


def test_identity_chart_split_explicit_run_is_bitwise_flat():
    def meshes():
        return (IM(8, (0.0, 1.0), name="x"),
                IM(8, (0.0, 1.0), periodic=False, name="y"),
                IM(NZ, (-DEPTH, 0.0), periodic=False, name="z"))

    def build(grid):
        return hy.Model(
            grid=grid, core=hy.Core(gravity=GRAVITY),
            time_stepper=AdamBashforth(5e-3, order=3),
            buoyancy=hy.ConstantStratification(n2=1.0),
            free_surface=hy.SplitExplicitFreeSurface(substeps=8))

    chart = build(fr.spatial.Grid(
        meshes(), mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x, y, 0.0 * x)},
            orthogonal=True)))
    flat = build(fr.spatial.Grid(meshes()))
    rng = np.random.default_rng(12)
    fields = {k: 0.1 * rng.standard_normal(flat.state[k].shape)
              for k in ("u", "v", "b", "ps")}
    chart.set_fields(**fields)
    flat.set_fields(**fields)
    chart.advance(20)
    flat.advance(20)
    # measured exactly equal; a few ULP is the honest jitted bound (two
    # structurally different programs may fuse differently)
    for name in ("u", "v", "b", "ps", "U", "V"):
        assert np.allclose(np.asarray(chart.state[name].data),
                           np.asarray(flat.state[name].data),
                           rtol=1e-12, atol=1e-14)


def test_horizontal_names_must_match_the_chart():
    with pytest.raises(ValueError, match="does not match the grid's "
                                         "chart coordinates"):
        _model(hy.ExplicitFreeSurface(horizontal=("lat", "lon")))


def test_nondimensional_variant_is_refused_on_a_chart():
    with pytest.raises(NotImplementedError, match="froude_number"):
        hy.Model(
            grid=_grid(), core=hy.Core(horizontal=HOR),
            scaling=fr.scaling.ExternalWave(),
            time_stepper=AdamBashforth(1e-3, order=3),
            free_surface=hy.ExplicitFreeSurface(
                froude_number=0.5, horizontal=HOR))
