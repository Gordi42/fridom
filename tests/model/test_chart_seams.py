"""Tests for fridom.model.chart_seams (the shared chart seam helpers)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.chart_seams import (
    chart_gradient,
    edge_scale,
    scale_factor,
    sealed_metric_divide,
    thin_shell_chart,
    to_contravariant,
    to_physical_tendency,
    volume_scale,
)
from fridom.spatial.bc import BC

IM = fr.spatial.meshes.IntervalMesh
RADIUS = 3.0


def _shell():
    return fr.spatial.spherical.Grid(
        (16, 8), radius=RADIUS, lat_extent=(-1.2, 1.2),
        vertical=IM(4, (-10.0, 0.0), periodic=False, name="z"))


def _field(grid, pattern, seed=0):
    space = pattern.resolve(grid)
    rng = np.random.default_rng(seed)
    return grid.create_field(
        space, data=jnp.asarray(rng.standard_normal(space.shape)))


def _lat(field):
    return np.asarray(field.grid.evaluation_nodes(
        field.function_space, "lat").data)


# ================================================================
#  thin_shell_chart: discovery and taught errors
# ================================================================
def test_no_chart_is_none():
    grid = fr.spatial.Grid((IM(4, (0.0, 1.0), name="x"),
                            IM(4, (0.0, 1.0), name="y")))
    assert thin_shell_chart(grid, "consumer") is None


def test_sphere_shell_reports_the_chart_pair():
    assert thin_shell_chart(_shell(), "consumer") == ("lon", "lat")


def test_non_orthogonal_chart_is_refused():
    grid = fr.spatial.Grid(
        (IM(4, (0.0, 1.0), name="x"), IM(4, (0.0, 1.0), name="y")),
        mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x + 0.4 * y, y, 0.0 * x)}))
    with pytest.raises(NotImplementedError, match="orthogonal"):
        thin_shell_chart(grid, "consumer")


def test_three_coordinate_chart_is_refused():
    grid = fr.spatial.Grid(
        (IM(4, (0.0, 1.0), name="x"), IM(4, (0.0, 1.0), name="y"),
         IM(4, (0.0, 1.0), name="z")),
        mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y, z: (x, y, z)}, orthogonal=True))
    with pytest.raises(NotImplementedError, match="exactly two"):
        thin_shell_chart(grid, "consumer")


# ================================================================
#  Geometry factors on the sphere
# ================================================================
def test_scale_factors_edges_and_area_are_the_analytic_sphere():
    grid = _shell()
    u = _field(grid, fr.spatial.Staggered("lon"))
    v = _field(grid, fr.spatial.Staggered(
        "lat", wall_bc={"lat": BC.DIRICHLET}))
    chart = ("lon", "lat")
    cos_u = np.cos(_lat(u))
    # a zonal flux crosses a meridian segment of length a dlat
    assert np.allclose(np.asarray(edge_scale(u, "lon", chart).data),
                       RADIUS + 0.0 * cos_u)
    # a meridional flux crosses a parallel segment a cos(lat) dlon
    assert np.allclose(np.asarray(edge_scale(v, "lat", chart).data),
                       RADIUS * np.cos(_lat(v)))
    assert edge_scale(u, "z", chart) is None
    assert np.allclose(np.asarray(scale_factor(u, "lon").data),
                       RADIUS * cos_u)
    assert np.allclose(np.asarray(volume_scale(u).data),
                       RADIUS ** 2 * cos_u)
    # thin shell: nothing depends on the vertical
    assert volume_scale(u).data.shape[-1] == 1


def test_chart_gradient_is_the_physical_gradient():
    grid = _shell()
    p = grid.create_field(
        fr.spatial.Collocated().resolve(grid),
        init=lambda lon, lat, z: jnp.sin(lat) + 0.0 * (lon + z))
    got = chart_gradient(p, "lat")
    lat = _lat(got)
    assert np.allclose(np.asarray(got.data),
                       np.cos(lat) / RADIUS + 0.0 * np.asarray(got.data),
                       atol=2e-2)
    # the zonal gradient of a zonally uniform field is exactly zero
    assert np.abs(np.asarray(chart_gradient(p, "lon").data)).max() == 0.0


def test_identity_chart_gradient_is_bitwise_the_diff():
    meshes = (IM(8, (0.0, 1.0), name="x"),
              IM(8, (0.0, 1.0), periodic=False, name="y"))
    grid = fr.spatial.Grid(meshes, mapping=fr.spatial.CoordinateMapping(
        chart={"X": lambda x, y: (x, y, 0.0 * x)}, orthogonal=True))
    p = _field(grid, fr.spatial.Collocated())
    for axis in ("x", "y"):
        assert np.array_equal(np.asarray(chart_gradient(p, axis).data),
                              np.asarray(p.diff(axis).data))


# ================================================================
#  The seams: round trip and the sealed divide
# ================================================================
def test_seam_round_trip_is_the_identity():
    grid = _shell()
    u = _field(grid, fr.spatial.Staggered("lon"))
    back = to_physical_tendency(to_contravariant(u, "lon"), "lon")
    assert np.allclose(np.asarray(back.data), np.asarray(u.data),
                       rtol=1e-14)


def test_sealed_divide_is_finite_forward_and_reverse_in_the_padding():
    grid = _shell()
    u = _field(grid, fr.spatial.Staggered("lon"))
    area = volume_scale(u)
    # the never-valid padding of a chart metric is an exact zero: the
    # bare quotient would plant inf forward and NaN in reverse
    assert area.storage.shape != area.data.shape
    assert float(jnp.min(area.storage)) == 0.0

    def loss(storage):
        return jnp.sum(sealed_metric_divide(
            u.with_storage(storage), area).storage ** 2)

    quotient = sealed_metric_divide(u, area)
    assert bool(jnp.all(jnp.isfinite(quotient.storage)))
    assert np.allclose(np.asarray(quotient.data),
                       np.asarray(u.data) / np.asarray(area.data))
    grad = jax.grad(loss)(u.storage)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_chart_on_top_of_a_mapped_column_is_refused():
    # chart + terrain(sigma) is a later composition (plan section 6)
    grid = fr.spatial.Grid(
        (IM(4, (0.0, 1.0), name="x"), IM(4, (0.0, 1.0), name="y"),
         IM(4, (-1.0, 0.0), periodic=False, name="z")),
        mapping=fr.spatial.CoordinateMapping(
            chart={"X": lambda x, y: (x, y, 0.0 * x)}, orthogonal=True,
            maps={"zp": lambda z, H: z * H},
            params={"H": lambda x, y: 1.0 + 0.1 * jnp.sin(x) + 0.0 * y}))
    with pytest.raises(NotImplementedError, match="maps= column"):
        thin_shell_chart(grid, "consumer")
