"""Tests for fridom.framework2.grid.immersed_domain."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.immersed_domain import ImmersedDomain, Slip
from fridom.framework2.grid.meshes.chebyshev import ChebyshevMesh
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.spaces.nodal import NodeSet


def indicator(x, y):
    """Wet region: x < 0.5 (periodic axis) and y < 1.0 (bounded)."""
    return (x < 0.5) & (y < 1.0)


@pytest.fixture
def mx():
    # periodic, 8 cells, dx = 1/8: cells 0..3 wet under x < 0.5
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    # bounded, 6 cells, dy = 1/3: cells 0..2 wet under y < 1.0
    return IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def dom():
    return ImmersedDomain(indicator)


@pytest.fixture
def grid(mx, my, dom):
    return Grid((mx, my), immersed=dom)


def column(field):
    """First x-column of a 2D mask/fraction as an int array."""
    return np.asarray(field.data[:, 0]).astype(int)


def row(field):
    """First y-row of a 2D mask/fraction as an int array."""
    return np.asarray(field.data[0, :]).astype(int)


# ================================================================
#  Construction and attachment
# ================================================================
def test_slip_default_and_override_constructor():
    assert ImmersedDomain(indicator).slip is Slip.NO_SLIP
    dom = ImmersedDomain(indicator, slip=Slip.FREE_SLIP)
    assert dom.slip is Slip.FREE_SLIP


def test_constructor_validation():
    with pytest.raises(TypeError, match="must be a callable"):
        ImmersedDomain(3.14)
    with pytest.raises(TypeError, match="must be a Slip member"):
        ImmersedDomain(indicator, slip="no_slip")


def test_identity_semantics(dom):
    alias = dom
    assert dom == alias
    assert dom != ImmersedDomain(indicator)
    assert hash(dom) == id(dom)


def test_unbound_descriptor_raises(mx, my):
    unbound = ImmersedDomain(indicator)
    mesh_x, mesh_y = mx, my
    Grid((mesh_x, mesh_y))  # a grid exists, but is not attached
    with pytest.raises(RuntimeError, match="not attached to a grid"):
        unbound.mask(mesh_x.center * mesh_y.center)


def test_grid_attachment_and_default(grid, dom, mx, my):
    assert grid.immersed is dom
    assert Grid((mx, my)).immersed is None


def test_with_immersed(mx, my):
    grid = Grid((mx, my))
    dom = ImmersedDomain(indicator)
    assert grid.with_immersed(dom) is grid
    assert grid.immersed is dom


def test_with_immersed_after_freeze_raises(mx, my):
    grid = Grid((mx, my))
    grid.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        grid.with_immersed(ImmersedDomain(indicator))


def test_rebinding_to_another_grid_raises(mx, my, dom):
    Grid((mx, my), immersed=dom)
    other = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),
                  IntervalMesh(6, (0.0, 2.0), periodic=False,
                               name="y")))
    with pytest.raises(ValueError,
                       match="already attached to another grid"):
        other.with_immersed(dom)


def test_reattaching_to_same_grid_is_idempotent(grid, dom):
    assert grid.with_immersed(dom) is grid
    assert grid.immersed is dom


# ================================================================
#  Cell-positioned masks (the declared datum)
# ================================================================
def test_center_mask_values_and_space(grid, dom, mx, my):
    space = mx.center * my.center
    mask = grid.immersed.mask(space)
    assert mask.dtype == jnp.bool_
    assert mask.function_space.bare is space
    assert mask.shape == (8, 6)
    assert np.array_equal(column(mask), [1, 1, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(row(mask), [1, 1, 1, 0, 0, 0])
    assert dom.mask(space) is not mask  # derive-on-demand, no cache


def test_cell_avg_mask_matches_center(grid, mx, my):
    center = grid.immersed.mask(mx.center * my.center)
    cell_avg = grid.immersed.mask(mx.cell_avg * my.cell_avg)
    assert cell_avg.shape == (8, 6)
    assert np.array_equal(np.asarray(cell_avg.data),
                          np.asarray(center.data))


def test_boolean_threshold_of_declared_fraction(mx, my):
    # non-boolean declarations are thresholded at 0.5 (iteration-1
    # boolean subset)
    dom = ImmersedDomain(
        lambda x, y: jnp.where((x < 0.5) & (y < 1.0), 0.8, 0.3))
    grid = Grid((mx, my), immersed=dom)
    mask = grid.immersed.mask(mx.center * my.center)
    assert np.array_equal(column(mask), [1, 1, 1, 1, 0, 0, 0, 0])


# ================================================================
#  Staggered masks (slip combination rule)
# ================================================================
def test_right_mask_periodic_no_slip(grid, mx, my):
    mask = grid.immersed.mask(mx.right * my.center)
    # face i sits between cells i and i+1 (wrapping): the wet-dry
    # face at i=3 and the wrap face at i=7 are killed
    assert np.array_equal(column(mask), [1, 1, 1, 0, 0, 0, 0, 0])


def test_right_mask_periodic_free_slip(grid, mx, my):
    mask = grid.immersed.mask(mx.right * my.center,
                              slip=Slip.FREE_SLIP)
    assert np.array_equal(column(mask), [1, 1, 1, 1, 0, 0, 0, 1])


def test_left_mask_periodic_no_slip(grid, mx, my):
    mask = grid.immersed.mask(mx.left * my.center)
    # face i sits between cells i-1 and i (wrapping)
    assert np.array_equal(column(mask), [0, 1, 1, 1, 0, 0, 0, 0])


def test_outer_mask_bounded(grid, mx, my):
    no_slip = grid.immersed.mask(mx.center * my.outer)
    assert no_slip.shape == (8, 7)
    # the exterior is dry: no-slip closes both domain-boundary faces
    assert np.array_equal(row(no_slip), [0, 1, 1, 0, 0, 0, 0])
    free = grid.immersed.mask(mx.center * my.outer,
                              slip=Slip.FREE_SLIP)
    assert np.array_equal(row(free), [1, 1, 1, 1, 0, 0, 0])


def test_left_mask_bounded(grid, mx, my):
    mask = grid.immersed.mask(mx.center * my.left)
    assert mask.shape == (8, 6)
    # face i sits between cells i-1 and i; the exterior is dry
    assert np.array_equal(row(mask), [0, 1, 1, 0, 0, 0])


def test_left_bc_constrained_boundary_dof_dropped(grid, mx, my):
    space = mx.center * my.nodal(NodeSet.LEFT, bc=BC.DIRICHLET)
    mask = grid.immersed.mask(space)
    # the Dirichlet-constrained left boundary face is dropped
    assert mask.shape == (8, 5)
    assert np.array_equal(row(mask), [1, 1, 0, 0, 0])


def test_inner_mask_bounded(grid, mx, my):
    mask = grid.immersed.mask(mx.center * my.inner)
    assert mask.shape == (8, 5)
    assert np.array_equal(row(mask), [1, 1, 0, 0, 0])


def test_face_avg_masks(grid, mx, my):
    # FaceAvg sits at right faces (periodic) / inner faces (bounded)
    mask = grid.immersed.mask(mx.face_avg * my.face_avg)
    assert mask.shape == (8, 5)
    assert np.array_equal(column(mask), [1, 1, 1, 0, 0, 0, 0, 0])
    assert np.array_equal(row(mask), [1, 1, 0, 0, 0])


def test_bc_constrained_boundary_dofs_dropped(grid, mx, my):
    space = mx.center * my.nodal(NodeSet.RIGHT, bc=BC.DIRICHLET)
    mask = grid.immersed.mask(space)
    # the Dirichlet-constrained right boundary face is dropped,
    # exactly like the space shape drops it
    assert mask.shape == (8, 5)
    assert np.array_equal(row(mask), [1, 1, 0, 0, 0])


def test_constructor_slip_is_the_default_rule(mx, my):
    dom = ImmersedDomain(indicator, slip=Slip.FREE_SLIP)
    grid = Grid((mx, my), immersed=dom)
    free = grid.immersed.mask(mx.right * my.center)
    assert np.array_equal(column(free), [1, 1, 1, 1, 0, 0, 0, 1])
    override = grid.immersed.mask(mx.right * my.center,
                                  slip=Slip.NO_SLIP)
    assert np.array_equal(column(override), [1, 1, 1, 0, 0, 0, 0, 0])


def test_invalid_slip_override_raises(grid, mx, my):
    with pytest.raises(TypeError, match="must be a Slip member"):
        grid.immersed.mask(mx.center * my.center, slip=0)


# ================================================================
#  Fractions (iteration-1 {0, 1} subset, slip-independent)
# ================================================================
def test_fraction_values_and_dtype(grid, mx, my):
    fraction = grid.immersed.fraction(mx.center * my.center)
    assert jnp.issubdtype(fraction.dtype, jnp.floating)
    values = np.unique(np.asarray(fraction.data))
    assert set(values) <= {0.0, 1.0}
    assert np.array_equal(column(fraction), [1, 1, 1, 1, 0, 0, 0, 0])


def test_fraction_transfer_is_slip_independent(mx, my):
    # the staircase transfer is geometric (AND), even when the
    # descriptor's mask rule is free-slip
    dom = ImmersedDomain(indicator, slip=Slip.FREE_SLIP)
    grid = Grid((mx, my), immersed=dom)
    fraction = grid.immersed.fraction(mx.right * my.center)
    assert np.array_equal(column(fraction), [1, 1, 1, 0, 0, 0, 0, 0])


# ================================================================
#  Explicit-data overloads (module-owned geometry)
# ================================================================
def test_explicit_fraction_overload(grid, mx, my):
    declared = grid.immersed.mask(mx.right * my.center)
    cells = grid.create_field(
        init=lambda x, y: ((x < 0.5) & (y < 1.0)).astype(float))
    explicit = grid.immersed.mask(mx.right * my.center,
                                  fraction=cells)
    assert np.array_equal(np.asarray(explicit.data),
                          np.asarray(declared.data))


def test_explicit_fraction_broadcasts_constant_factors(grid, mx, my):
    # a fraction constant along y still derives the full mask
    cells = grid.create_field(
        mx.center * my.constant,
        data=(jnp.arange(8) < 4).astype(float).reshape(8, 1))
    mask = grid.immersed.mask(mx.center * my.center, fraction=cells)
    assert np.array_equal(column(mask), [1, 1, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(row(mask), [1, 1, 1, 1, 1, 1])


def test_explicit_fraction_on_wrong_family_raises(grid, mx, my):
    staggered = grid.create_field(mx.right * my.center)
    with pytest.raises(ValueError, match="cell family"):
        grid.immersed.mask(mx.center * my.center, fraction=staggered)


def test_explicit_fraction_on_other_grid_raises(grid, mx, my):
    other = Grid((mx, my))
    foreign = other.create_field(
        init=lambda x, y: ((x < 0.5) & (y < 1.0)).astype(float))
    with pytest.raises(ValueError, match="different grid"):
        grid.immersed.mask(mx.center * my.center, fraction=foreign)


# ================================================================
#  Error behavior
# ================================================================
def test_coefficient_space_raises(grid, mx, my):
    space = mx.fourier(origin=mx.center) * my.center
    with pytest.raises(ValueError, match="coefficient-space"):
        grid.immersed.mask(space)


def test_constant_factor_raises(grid, mx, my):
    with pytest.raises(ValueError, match="constant factors"):
        grid.immersed.mask(mx.center * my.constant)


def test_partial_space_raises(grid, mx):
    with pytest.raises(ValueError, match="every grid coordinate"):
        grid.immersed.mask(mx.center)


def test_wrong_indicator_signature_raises(mx, my):
    dom = ImmersedDomain(lambda x: x < 0.5)
    grid = Grid((mx, my), immersed=dom)
    with pytest.raises(TypeError, match="exactly the grid"):
        grid.immersed.mask(mx.center * my.center)


def test_transition_is_designed_for(grid, mx, my):
    with pytest.raises(NotImplementedError, match="designed-for"):
        grid.immersed.transition(mx.center * my.center)


def test_non_interval_mesh_raises():
    mz = ChebyshevMesh(8, (0.0, 1.0), name="z")
    dom = ImmersedDomain(lambda z: z < 0.5)
    grid = Grid((mz,), immersed=dom)
    with pytest.raises(NotImplementedError, match="IntervalMesh"):
        grid.immersed.mask(mz.outer)


# ================================================================
#  One-dimensional grids and trace-time materialization
# ================================================================
def test_lone_factor_space_on_1d_grid():
    mesh = IntervalMesh(8, (0.0, 1.0), name="x")
    dom = ImmersedDomain(lambda x: x < 0.5)
    grid = Grid((mesh,), immersed=dom)
    mask = grid.immersed.mask(mesh.center)
    assert mask.shape == (8,)
    assert np.array_equal(np.asarray(mask.data).astype(int),
                          [1, 1, 1, 1, 0, 0, 0, 0])


def test_mask_materializes_under_jit(grid, mx, my):
    space = mx.right * my.center
    eager = grid.immersed.mask(space)

    @jax.jit
    def derive():
        return grid.immersed.mask(space).data

    assert np.array_equal(np.asarray(derive()),
                          np.asarray(eager.data))


# ================================================================
#  Device-count invariance (genuine under the forced-4 suite)
# ================================================================
def build_grid(device_ids):
    mesh_x = IntervalMesh(16, (0.0, 1.0), name="x")
    mesh_y = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
    dom = ImmersedDomain(
        lambda x, y: (x - 0.5) ** 2 + (y - 1.0) ** 2 < 0.16)
    return Grid((mesh_x, mesh_y), immersed=dom,
                device_ids=device_ids)


def test_masks_are_device_count_invariant(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, one = build_grid(None), build_grid((0,))
    for factory in ("center", "right"):
        space_many = (getattr(many.factors[0], factory)
                      * many.factors[1].center)
        space_one = (getattr(one.factors[0], factory)
                     * one.factors[1].center)
        m_many = many.immersed.mask(space_many)
        m_one = one.immersed.mask(space_one)
        assert np.array_equal(np.asarray(m_many.data),
                              np.asarray(m_one.data))
