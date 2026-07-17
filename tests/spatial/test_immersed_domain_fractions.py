"""Genuine partial-cell fractions for fridom.spatial.immersed_domain.

Covers the I0 opt-in (order=, min-transfer, small-cell floor,
explicit-fraction passthrough) layered on the collocation staircase
tested in ``test_immersed_domain.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain, Slip
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh


# ================================================================
#  Self-contained builders (no cross-file test imports)
# ================================================================
def _grid_1d(n=5, extent=(0.0, 1.0), **dom_kw):
    """Build a bounded 1D grid with an immersed domain."""
    mesh = IntervalMesh(n, extent, periodic=False, name="x")
    init = dom_kw.pop("init")
    dom = ImmersedDomain(init, **dom_kw)
    return Grid((mesh,), immersed=dom), mesh


def _col(field):
    """First x-column of a 2D field as a float array."""
    return np.asarray(field.data[:, 0])


# ================================================================
#  Quadrature exactness (smooth polynomial fraction)
# ================================================================
def test_quadrature_linear_fraction_is_exact():
    # a genuine LINEAR local fraction theta = 1 - x: the per-cell
    # average equals the cell-center value (midpoint-exact), so the
    # tensor Gauss-Legendre quadrature reproduces it to machine zero
    # at any q >= 2 (exact for polynomials up to degree 2q - 1).
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, order=2,
                          min_fraction=0.0)
    frac = grid.immersed.fraction(mesh.center)
    centers = (np.arange(5) + 0.5) / 5.0
    assert np.allclose(np.asarray(frac.data), 1.0 - centers,
                       atol=1e-12)


def test_quadrature_high_q_matches_low_q_on_polynomial():
    # a polynomial below the exactness degree is q-independent
    grid2, mesh2 = _grid_1d(6, init=lambda x: 1.0 - x, order=2,
                            min_fraction=0.0)
    grid8, mesh8 = _grid_1d(6, init=lambda x: 1.0 - x, order=8,
                            min_fraction=0.0)
    assert np.allclose(
        np.asarray(grid2.immersed.fraction(mesh2.center).data),
        np.asarray(grid8.immersed.fraction(mesh8.center).data),
        atol=1e-12)


def test_quadrature_hard_boundary_converges_in_q():
    # the hard half-plane x < 0.35 cuts cell 1 ([0.2, 0.4)) at
    # fraction 0.75; a discontinuous integrand converges slowly under
    # Gauss-Legendre, so the test pins convergence (high q closer than
    # low q, and within a modest tolerance), not machine agreement.
    true = 0.75

    def err(q):
        grid, mesh = _grid_1d(
            5, init=lambda x: (x < 0.35).astype(float), order=q,
            min_fraction=0.0)
        val = float(np.asarray(grid.immersed.fraction(mesh.center)
                               .data)[1])
        return abs(val - true)

    assert err(64) < err(2)
    assert err(128) < 0.02


# ================================================================
#  Min-transfer on faces (partial neighbours)
# ================================================================
def test_face_fraction_is_min_of_neighbours():
    # theta = 1 - x on 5 cells -> [0.9, 0.7, 0.5, 0.3, 0.1]; each
    # inner x-face takes the min of its two adjacent cells (the
    # further-x, smaller value), the geometric hFacW rule.
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, order=2,
                          min_fraction=0.0)
    face = grid.immersed.fraction(mesh.inner)
    assert np.allclose(np.asarray(face.data), [0.7, 0.5, 0.3, 0.1],
                       atol=1e-12)


def test_face_fraction_dry_exterior_bounded():
    # the outer node set carries both domain-boundary faces; the
    # exterior is dry (min against 0), so both ends are 0.
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, order=2,
                          min_fraction=0.0)
    outer = grid.immersed.fraction(mesh.outer)
    vals = np.asarray(outer.data)
    assert vals[0] == 0.0
    assert vals[-1] == 0.0
    # interior faces are still the neighbour minima
    assert np.allclose(vals[1:-1], [0.7, 0.5, 0.3, 0.1], atol=1e-12)


def test_fraction_min_transfer_is_slip_independent():
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, order=2,
                          min_fraction=0.0, slip=Slip.FREE_SLIP)
    face = grid.immersed.fraction(mesh.inner)
    # free-slip does not loosen the fraction transfer (masks only)
    assert np.allclose(np.asarray(face.data), [0.7, 0.5, 0.3, 0.1],
                       atol=1e-12)


# ================================================================
#  Two-dimensional partial cells (x-partials that vary with y)
# ================================================================
def test_2d_fraction_has_x_partials_varying_with_y():
    # theta(x, y) = 1 - x - 0.5 x y is bilinear and interior on
    # x in [0, 0.5], so midpoint-exact: every cell is a genuine
    # x-partial whose value depends on y (a sloping boundary B(y)).
    mx = IntervalMesh(4, (0.0, 0.5), periodic=False, name="x")
    my = IntervalMesh(3, (0.0, 1.0), periodic=False, name="y")
    dom = ImmersedDomain(lambda x, y: 1.0 - x - 0.5 * x * y,
                         order=2, min_fraction=0.0)
    grid = Grid((mx, my), immersed=dom)
    frac = grid.immersed.fraction(mx.center * my.center)
    xc = (np.arange(4) + 0.5) / 4.0 * 0.5
    yc = (np.arange(3) + 0.5) / 3.0
    analytic = 1.0 - xc[:, None] - 0.5 * xc[:, None] * yc[None, :]
    assert np.allclose(np.asarray(frac.data), analytic, atol=1e-12)
    # the x-partials genuinely differ between the first and last y-row
    assert not np.allclose(np.asarray(frac.data)[:, 0],
                           np.asarray(frac.data)[:, -1])


def test_2d_x_face_fraction_min_transfer():
    mx = IntervalMesh(4, (0.0, 0.5), periodic=False, name="x")
    my = IntervalMesh(3, (0.0, 1.0), periodic=False, name="y")
    dom = ImmersedDomain(lambda x, y: 1.0 - x - 0.5 * x * y,
                         order=2, min_fraction=0.0)
    grid = Grid((mx, my), immersed=dom)
    cell = np.asarray(
        grid.immersed.fraction(mx.center * my.center).data)
    inner = np.asarray(
        grid.immersed.fraction(mx.inner * my.center).data)
    # each x-inner face is the min of the two adjacent x-cells
    assert np.allclose(inner, np.minimum(cell[:-1], cell[1:]),
                       atol=1e-12)


# ================================================================
#  Small-cell floor (IP-D3)
# ================================================================
def _explicit_cells(grid, mesh, values):
    """Return a cell-center fraction field carrying `values`."""
    return grid.create_field(mesh.center,
                             data=jnp.asarray(values, dtype=float))


def test_floor_semantics():
    # below min/2 -> 0; in [min/2, min) -> min; >= min and exact
    # {0, 1} untouched. min_fraction = 0.1 -> min/2 = 0.05.
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, min_fraction=0.1)
    cells = _explicit_cells(grid, mesh, [1.0, 0.5, 0.09, 0.04, 0.0])
    frac = grid.immersed.fraction(mesh.center, fraction=cells)
    assert np.allclose(np.asarray(frac.data),
                       [1.0, 0.5, 0.1, 0.0, 0.0], atol=1e-12)


def test_floor_disabled_by_zero_min_fraction():
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, min_fraction=0.0)
    cells = _explicit_cells(grid, mesh, [1.0, 0.5, 0.09, 0.04, 0.0])
    frac = grid.immersed.fraction(mesh.center, fraction=cells)
    assert np.allclose(np.asarray(frac.data),
                       [1.0, 0.5, 0.09, 0.04, 0.0], atol=1e-12)


def test_floor_untouched_boolean_fractions():
    # a {0, 1} staircase never trips the floor (the default path)
    grid, mesh = _grid_1d(5, init=lambda x: (x < 0.55).astype(float),
                          min_fraction=0.1)
    frac = grid.immersed.fraction(mesh.center)
    assert set(np.unique(np.asarray(frac.data))) <= {0.0, 1.0}
    assert np.array_equal(np.asarray(frac.data).astype(int),
                          [1, 1, 1, 0, 0])


def test_floor_applies_to_quadrature_source():
    # a genuine small partial cell is lifted to the floor. theta =
    # 0.06 x-independent? build a cell whose average lands in the
    # floor band via explicit data is covered above; here a linear
    # ramp gives a small last-cell average lifted to min_fraction.
    grid, mesh = _grid_1d(10, init=lambda x: (1.0 - x) * 0.1,
                          order=2, min_fraction=0.1)
    # theta_center = 0.1 * (1 - center); last centers give <0.1 values
    frac = np.asarray(grid.immersed.fraction(mesh.center).data)
    centers = (np.arange(10) + 0.5) / 10.0
    raw = 0.1 * (1.0 - centers)
    for r, f in zip(raw, frac, strict=True):
        if r < 0.05:
            assert f == 0.0
        elif r < 0.1:
            assert f == pytest.approx(0.1)
        else:
            assert f == pytest.approx(r)


# ================================================================
#  Explicit-fraction passthrough (non-boolean survives)
# ================================================================
def test_explicit_non_boolean_fraction_passes_through():
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, min_fraction=0.0)
    cells = _explicit_cells(grid, mesh, [0.3, 0.7, 0.5, 0.9, 0.2])
    frac = grid.immersed.fraction(mesh.center, fraction=cells)
    assert np.allclose(np.asarray(frac.data),
                       [0.3, 0.7, 0.5, 0.9, 0.2], atol=1e-12)


def test_explicit_fraction_is_clipped():
    grid, mesh = _grid_1d(3, init=lambda x: 1.0 - x, min_fraction=0.0)
    cells = _explicit_cells(grid, mesh, [1.4, -0.2, 0.5])
    frac = grid.immersed.fraction(mesh.center, fraction=cells)
    assert np.allclose(np.asarray(frac.data), [1.0, 0.0, 0.5],
                       atol=1e-12)


def test_explicit_fraction_mask_is_theta_positive():
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, min_fraction=0.0)
    cells = _explicit_cells(grid, mesh, [0.3, 0.0, 0.5, 0.0, 0.9])
    mask = grid.immersed.mask(mesh.center, fraction=cells)
    assert mask.dtype == jnp.bool_
    assert np.array_equal(np.asarray(mask.data).astype(int),
                          [1, 0, 1, 0, 1])


# ================================================================
#  mask = theta > 0, both slip rules, over genuine fractions
# ================================================================
def test_mask_is_theta_positive_both_slips():
    # theta = 1 - x on 5 cells is positive in every cell, so the cell
    # mask is all-wet; the interior x-faces are wet under both rules,
    # but the dry exterior differs (no-slip closes the boundary
    # faces, free-slip keeps a single-wet-neighbour face open).
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, order=2,
                          min_fraction=0.0)
    no_slip = grid.immersed.mask(mesh.outer, slip=Slip.NO_SLIP)
    free = grid.immersed.mask(mesh.outer, slip=Slip.FREE_SLIP)
    assert np.array_equal(np.asarray(no_slip.data).astype(int),
                          [0, 1, 1, 1, 1, 0])
    assert np.array_equal(np.asarray(free.data).astype(int),
                          [1, 1, 1, 1, 1, 1])


def test_mask_kills_only_fully_dry_cells():
    # theta with a genuine dry cell (0.0) and partial cells; the cell
    # mask is theta > 0, so the partial cells stay wet.
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, min_fraction=0.0)
    cells = _explicit_cells(grid, mesh, [0.8, 0.4, 0.0, 0.2, 0.0])
    mask = grid.immersed.mask(mesh.center, fraction=cells)
    assert np.array_equal(np.asarray(mask.data).astype(int),
                          [1, 1, 0, 1, 0])


# ================================================================
#  Stretched-mesh quadrature (coordinate_map honoured)
# ================================================================
def _square(s):
    """Monotone stretch s -> s**2 (bunched near x = 0)."""
    return s ** 2


def test_stretched_mesh_quadrature_uses_physical_edges():
    # cell edges at s**2 for s = 0, 1/4, ..., 1 -> unequal widths; a
    # linear fraction 1 - x averages to 1 - (edge_l + edge_r)/2 per
    # cell (midpoint-exact on each cell's own physical span).
    mesh = MappedIntervalMesh(4, (0.0, 1.0), _square, name="x")
    dom = ImmersedDomain(lambda x: 1.0 - x, order=2, min_fraction=0.0)
    grid = Grid((mesh,), immersed=dom)
    frac = grid.immersed.fraction(mesh.center)
    edges = (np.arange(5) / 4.0) ** 2
    analytic = 1.0 - 0.5 * (edges[:-1] + edges[1:])
    assert np.allclose(np.asarray(frac.data), analytic, atol=1e-12)


def test_stretched_quadrature_differs_from_uniform():
    mesh = MappedIntervalMesh(4, (0.0, 1.0), _square, name="x")
    umesh = IntervalMesh(4, (0.0, 1.0), periodic=False, name="x")
    dom = ImmersedDomain(lambda x: 1.0 - x, order=2, min_fraction=0.0)
    udom = ImmersedDomain(lambda x: 1.0 - x, order=2, min_fraction=0.0)
    stretched = Grid((mesh,), immersed=dom).immersed.fraction(
        mesh.center)
    uniform = Grid((umesh,), immersed=udom).immersed.fraction(
        umesh.center)
    assert not np.allclose(np.asarray(stretched.data),
                           np.asarray(uniform.data))


# ================================================================
#  order = None / 1 parity (bitwise iteration-1 staircase)
# ================================================================
def test_order_none_and_one_alias_the_collocation_staircase():
    def indicator(x):
        return (x < 0.35).astype(float)

    none = Grid((IntervalMesh(5, (0.0, 1.0), periodic=False,
                              name="x"),),
                immersed=ImmersedDomain(indicator))
    one = Grid((IntervalMesh(5, (0.0, 1.0), periodic=False,
                             name="x"),),
               immersed=ImmersedDomain(indicator, order=1))
    m = IntervalMesh(5, (0.0, 1.0), periodic=False, name="x")
    fn = none.immersed.fraction(none.factors[0].center)
    fo = one.immersed.fraction(one.factors[0].center)
    assert np.array_equal(np.asarray(fn.data), np.asarray(fo.data))
    assert set(np.unique(np.asarray(fn.data))) <= {0.0, 1.0}
    del m


def test_order_none_fraction_matches_documented_column():
    # the iteration-1 example geometry (x < 0.5 periodic, y < 1.0
    # bounded): the default fraction reproduces the staircase column.
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")
    dom = ImmersedDomain(lambda x, y: (x < 0.5) & (y < 1.0))
    grid = Grid((mx, my), immersed=dom)
    frac = grid.immersed.fraction(mx.center * my.center)
    assert np.array_equal(_col(frac).astype(int),
                          [1, 1, 1, 1, 0, 0, 0, 0])


# ================================================================
#  Constructor validation
# ================================================================
@pytest.mark.parametrize("order", [0, -1, 1.5, True])
def test_order_validation_errors(order):
    with pytest.raises(ValueError, match="order="):
        ImmersedDomain(lambda x: x < 0.5, order=order)


@pytest.mark.parametrize("mf", [-0.1, 1.0, 1.5, True])
def test_min_fraction_validation_errors(mf):
    with pytest.raises(ValueError, match="min_fraction="):
        ImmersedDomain(lambda x: x < 0.5, min_fraction=mf)


def test_order_and_min_fraction_properties():
    dom = ImmersedDomain(lambda x: x < 0.5, order=6, min_fraction=0.2)
    assert dom.order == 6
    assert dom.min_fraction == 0.2
    default = ImmersedDomain(lambda x: x < 0.5)
    assert default.order is None
    assert default.min_fraction == 0.1


# ================================================================
#  Memoization (concrete cache; new field object per call)
# ================================================================
def test_fraction_is_memoized_but_returns_new_fields():
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, order=2,
                          min_fraction=0.0)
    space = mesh.center
    a = grid.immersed.fraction(space)
    b = grid.immersed.fraction(space)
    assert a is not b  # derive-on-demand contract stands
    assert np.array_equal(np.asarray(a.data), np.asarray(b.data))


# ================================================================
#  Trace-time materialization and device-count invariance
# ================================================================
def test_fraction_materializes_under_jit():
    grid, mesh = _grid_1d(5, init=lambda x: 1.0 - x, order=4,
                          min_fraction=0.0)
    space = mesh.inner
    eager = grid.immersed.fraction(space)

    @jax.jit
    def derive():
        return grid.immersed.fraction(space).data

    assert np.allclose(np.asarray(derive()), np.asarray(eager.data),
                       atol=1e-12)


def _partial_grid(device_ids):
    mesh = IntervalMesh(16, (0.0, 1.0), periodic=False, name="x")
    dom = ImmersedDomain(lambda x: 1.0 - x, order=4, min_fraction=0.0)
    return Grid((mesh,), immersed=dom, device_ids=device_ids)


def test_fractions_are_device_count_invariant(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, one = _partial_grid(None), _partial_grid((0,))
    for factory in ("center", "inner"):
        f_many = many.immersed.fraction(
            getattr(many.factors[0], factory))
        f_one = one.immersed.fraction(
            getattr(one.factors[0], factory))
        assert np.allclose(np.asarray(f_many.data),
                           np.asarray(f_one.data), atol=1e-12)
