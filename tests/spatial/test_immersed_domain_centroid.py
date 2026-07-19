"""Wet-centroid offset for fridom.spatial.immersed_domain (PB-D1).

Covers ``ImmersedDomain.centroid_offset`` -- the first-vertical-moment
quadrature feeding the hydrostatic partial-bottom pressure-gradient
correction: analytic exactness on a linear indicator, the lateral-cut /
full / dry / collocation zeros, the sign discriminant, memoization, the
chart taught error, and the cell-space validation. Self-contained
builders per the AGENTS oversized-module rule.
"""
import numpy as np
import pytest

from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh
from fridom.spatial.spaces.tensor_product import TensorProductSpace


# ================================================================
#  Self-contained builders (no cross-file test imports)
# ================================================================
def _grid_xz(nz, init, *, nx=4, order=4, min_fraction=0.0):
    """Build an (x periodic, z bounded) immersed grid and cell space."""
    mx = IntervalMesh(nx, (0.0, 1.0), periodic=True, name="x")
    mz = IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z")
    dom = ImmersedDomain(init, order=order, min_fraction=min_fraction)
    grid = Grid((mx, mz), immersed=dom)
    return grid, TensorProductSpace.of(mx.center, mz.center)


def _z_centers(nz, extent=(0.0, 1.0)):
    """Cell-centre z coordinates of a uniform mesh."""
    dz = (extent[1] - extent[0]) / nz
    return extent[0] + dz * (np.arange(nz) + 0.5), dz


# ================================================================
#  Analytic exactness on a linear indicator (bottom cut, PB-D1)
# ================================================================
def test_linear_indicator_offset_is_analytic_exact():
    # chi = (z - a) / w linear within every cell (a below, a + w above
    # the domain, so no clip kink): the first moment is a degree-2
    # integrand, exact at order >= 2. Closed form:
    #   delta = <z chi>/<chi> - z_c = dz^2 / (12 (z_c - a)).
    nz, a, w = 4, -0.2, 2.0
    grid, cell = _grid_xz(nz, lambda x, z: (z - a) / w)  # noqa: ARG005
    delta = np.asarray(grid.immersed.centroid_offset(cell, "z").data)
    z_c, dz = _z_centers(nz)
    analytic = dz**2 / (12.0 * (z_c - a))
    assert np.allclose(delta[0], analytic, atol=1e-13)
    # constant along x (the cut is x-independent)
    assert np.allclose(delta, delta[0][None, :], atol=1e-15)


def test_offset_high_order_matches_low_order_on_polynomial():
    # a linear indicator is exact at any order >= 2, so order 2 == 8
    g2, c2 = _grid_xz(5, lambda x, z: 0.7 * z + 0.1, order=2)  # noqa: ARG005
    g8, c8 = _grid_xz(5, lambda x, z: 0.7 * z + 0.1, order=8)  # noqa: ARG005
    d2 = np.asarray(g2.immersed.centroid_offset(c2, "z").data)
    d8 = np.asarray(g8.immersed.centroid_offset(c8, "z").data)
    assert np.allclose(d2, d8, atol=1e-13)


# ================================================================
#  The lateral cut leaves the vertical centroid put (PB-D1)
# ================================================================
def test_lateral_cut_offset_is_exactly_zero():
    # chi depends on x only (a vertical side wall): the z-moment of a
    # z-constant chi vanishes identically -- the volume fraction alone
    # cannot tell this from a bottom cut, only the first moment can.
    grid, cell = _grid_xz(
        6, lambda x, z: (x > 0.4).astype(float) + 0.0 * z, nx=6)
    delta = np.asarray(grid.immersed.centroid_offset(cell, "z").data)
    assert np.abs(delta).max() == 0.0


# ================================================================
#  Full / dry / collocation cells are exactly zero
# ================================================================
def test_all_wet_offset_is_exactly_zero():
    grid, cell = _grid_xz(4, lambda x, z: x * 0.0 + z * 0.0 + 1.0)
    assert np.abs(
        np.asarray(grid.immersed.centroid_offset(cell, "z").data)
    ).max() == 0.0


def test_all_dry_offset_is_exactly_zero():
    grid, cell = _grid_xz(4, lambda x, z: x * 0.0 + z * 0.0)
    assert np.abs(
        np.asarray(grid.immersed.centroid_offset(cell, "z").data)
    ).max() == 0.0


def test_collocation_order_offset_is_exactly_zero():
    # order=None is the staircase (theta in {0, 1}) -- no partial cells,
    # so the offset is identically zero (the G2 no-op).
    mx = IntervalMesh(4, (0.0, 1.0), periodic=True, name="x")
    mz = IntervalMesh(6, (0.0, 1.0), periodic=False, name="z")
    dom = ImmersedDomain(lambda x, z: (z > 0.42).astype(float),  # noqa: ARG005
                         order=None)
    grid = Grid((mx, mz), immersed=dom)
    cell = TensorProductSpace.of(mx.center, mz.center)
    assert np.abs(
        np.asarray(grid.immersed.centroid_offset(cell, "z").data)
    ).max() == 0.0


# ================================================================
#  Sign discriminant: bottom cut positive, top cut negative
# ================================================================
def test_bottom_cut_positive_top_cut_negative():
    # wet region ABOVE the cut (increasing chi) -> centroid above centre
    # (delta > 0); wet region BELOW (decreasing chi) -> delta < 0.
    gb, cb = _grid_xz(6, lambda x, z: (z > 0.55).astype(float) + 0.0 * x)
    gt, ct = _grid_xz(6, lambda x, z: (z < 0.55).astype(float) + 0.0 * x)
    db = np.asarray(gb.immersed.centroid_offset(cb, "z").data)
    dt = np.asarray(gt.immersed.centroid_offset(ct, "z").data)
    assert db.max() > 1e-6      # a bottom cut lifts the centroid
    assert dt.min() < -1e-6     # a top cut lowers it


# ================================================================
#  Stretched vertical mesh (separable, not a chart) runs
# ================================================================
def test_stretched_offset_is_finite_and_lateral_zero():
    mx = IntervalMesh(4, (0.0, 1.0), periodic=True, name="x")
    mz = MappedIntervalMesh(6, (0.0, 1.0), lambda s: s**2,
                            periodic=False, name="z")
    dom = ImmersedDomain(lambda x, z: (z > 0.3).astype(float) + 0.0 * x,
                         order=6, min_fraction=0.0)
    grid = Grid((mx, mz), immersed=dom)
    assert grid.mapping is None  # a separable stretch is not a chart
    cell = TensorProductSpace.of(mx.center, mz.center)
    delta = np.asarray(grid.immersed.centroid_offset(cell, "z").data)
    assert np.isfinite(delta).all()
    assert np.allclose(delta, delta[0][None, :], atol=1e-15)  # x-uniform


# ================================================================
#  Memoization (concrete cache hit, like fraction)
# ================================================================
def test_offset_is_memoized_concrete():
    grid, cell = _grid_xz(5, lambda x, z: (z + 0.2) / 2.0)  # noqa: ARG005
    a = grid.immersed.centroid_offset(cell, "z")
    b = grid.immersed.centroid_offset(cell, "z")
    assert a is not b                       # a fresh field wrapper
    assert np.array_equal(np.asarray(a.data), np.asarray(b.data))
    # the concrete-only cache is populated (a hit on the second call)
    assert any(key[1] == "centroid_offset"
               for key in grid.immersed._cache)


# ================================================================
#  Chart taught error (PB-D3)
# ================================================================
def test_chart_offset_is_taught_error():
    ms = IntervalMesh(4, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(maps={"zp": lambda sigma: sigma**2})
    dom = ImmersedDomain(lambda sigma: (sigma > 0.4).astype(float),
                         order=4, min_fraction=0.0)
    grid = Grid((ms,), mapping=mapping, immersed=dom)
    with pytest.raises(NotImplementedError, match="chart"):
        grid.immersed.centroid_offset(ms.center, "sigma")


# ================================================================
#  Validation: cell-centred space and a resolved moment axis
# ================================================================
def test_offset_rejects_non_cell_space():
    mx = IntervalMesh(4, (0.0, 1.0), periodic=True, name="x")
    mz = IntervalMesh(6, (0.0, 1.0), periodic=False, name="z")
    dom = ImmersedDomain(lambda x, z: (z > 0.4).astype(float),  # noqa: ARG005
                         order=4)
    grid = Grid((mx, mz), immersed=dom)
    # a face-family factor (mx.right) is not cell-positioned
    with pytest.raises(ValueError, match="cell"):
        grid.immersed.centroid_offset(
            TensorProductSpace.of(mx.right, mz.center), "z")


def test_offset_rejects_unresolved_moment_axis():
    grid, cell = _grid_xz(4, lambda x, z: (z > 0.4).astype(float))  # noqa: ARG005
    with pytest.raises(ValueError, match="not a coordinate"):
        grid.immersed.centroid_offset(cell, "q")
