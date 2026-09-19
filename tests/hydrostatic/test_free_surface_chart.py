r"""hy.ExplicitFreeSurface on an embedding chart (plan S2).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
orthogonal thin-shell chart arm of the explicit free surface: the
area-weighted barotropic transport divergence closed by ``sqrt_g``
(volume-conserving to rounding), its exact adjointness to the physical
surface-pressure gradient under the ``sqrt_g`` measure, and the
bind-time taught errors (the implicit / split-explicit variants stay
refused on a chart until the chart Helmholtz of plan S3). Self-contained
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


@pytest.mark.parametrize("variant", [
    pytest.param(lambda: hy.ImplicitFreeSurface(horizontal=HOR),
                 id="implicit"),
    pytest.param(lambda: hy.SplitExplicitFreeSurface(
        substeps=4, horizontal=HOR), id="split"),
])
def test_elliptic_variants_stay_refused_on_a_chart(variant):
    with pytest.raises(NotImplementedError,
                       match="curvilinear / spherical"):
        _model(variant())


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
