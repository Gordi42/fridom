"""Tests for TraceSpace and the Side vocabulary (spaces/trace.py)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.spaces.tensor_product import join, join_factor
from fridom.spatial.spaces.trace import Side, TraceSpace

N = 8


@pytest.fixture
def mz():
    # bounded 1D mesh: the axis a trace is taken along
    return IntervalMesh(N, (0.0, 3.0), periodic=False, name="z")


@pytest.fixture
def grid():
    # x/y periodic horizontals + bounded z (the traced axis)
    mx = IntervalMesh(N, (0.0, 1.0), name="x")
    my = IntervalMesh(N, (0.0, 2.0), name="y")
    mz = IntervalMesh(4, (0.0, 3.0), periodic=False, name="z")
    return Grid((mx, my, mz))


# ================================================================
#  Defining attributes
# ================================================================
def test_shape_is_one(mz):
    assert mz.trace(NodeSet.CENTER, Side.HIGH).shape == (1,)


def test_is_not_constant(mz):
    # the whole point: a trace must NOT broadcast into the interior
    assert mz.trace(NodeSet.CENTER, Side.HIGH).is_constant is False


def test_collapses_axis_true(mz):
    # storage/locality role: collapsed like a constant, but not constant
    assert mz.trace(NodeSet.CENTER, Side.HIGH).collapses_axis is True


def test_collapses_axis_of_constant_and_nodal(mz):
    assert mz.constant.collapses_axis is True
    assert mz.center.collapses_axis is False


def test_is_bc_free(mz):
    assert mz.trace(NodeSet.CENTER, Side.HIGH).bc.is_free


def test_not_a_nodal_space(mz):
    # must never enter isinstance(x, NodalSpace) branches
    trace = mz.trace(NodeSet.OUTER, Side.HIGH)
    assert not isinstance(trace, NodalSpace)
    assert isinstance(trace, TraceSpace)


def test_static_descriptors(mz):
    trace = mz.trace(NodeSet.OUTER, Side.LOW)
    assert trace.parent_node_set is NodeSet.OUTER
    assert trace.side is Side.LOW
    assert trace.depth == 0
    assert trace.mesh is mz


# ================================================================
#  Interning identity
# ================================================================
def test_interning_same_locator(mz):
    assert (mz.trace(NodeSet.CENTER, Side.HIGH)
            is mz.trace(NodeSet.CENTER, Side.HIGH))


def test_interning_distinguishes_side(mz):
    assert (mz.trace(NodeSet.CENTER, Side.LOW)
            is not mz.trace(NodeSet.CENTER, Side.HIGH))


def test_interning_distinguishes_node_set(mz):
    assert (mz.trace(NodeSet.CENTER, Side.HIGH)
            is not mz.trace(NodeSet.OUTER, Side.HIGH))


def test_interning_is_per_mesh():
    mz1 = IntervalMesh(N, (0.0, 3.0), periodic=False, name="z")
    mz2 = IntervalMesh(N, (0.0, 3.0), periodic=False, name="z")
    assert (mz1.trace(NodeSet.CENTER, Side.HIGH)
            is not mz2.trace(NodeSet.CENTER, Side.HIGH))


def test_scalar_variant_interns(mz):
    trace = mz.trace(NodeSet.CENTER, Side.HIGH)
    comp = trace.as_complex()
    assert comp is trace.as_complex()
    assert comp is not trace
    assert comp.as_real() is trace
    assert comp.shape == (1,)
    assert isinstance(comp, TraceSpace)
    assert comp.side is Side.HIGH


def test_repr(mz):
    trace = mz.trace(NodeSet.CENTER, Side.HIGH)
    assert repr(trace) == "Trace(z, CENTER, side=HIGH)"


# ================================================================
#  Factory rejections (taught errors)
# ================================================================
def test_rejects_points(mz):
    with pytest.raises(ValueError, match="POINTS is the PointMesh"):
        mz.trace(NodeSet.POINTS, Side.HIGH)


def test_rejects_non_node_set(mz):
    with pytest.raises(TypeError, match="node_set must be a NodeSet"):
        mz.trace("center", Side.HIGH)


def test_rejects_non_side(mz):
    with pytest.raises(TypeError, match="side must be a Side member"):
        mz.trace(NodeSet.CENTER, "high")


def test_rejects_nonzero_depth(mz):
    with pytest.raises(NotImplementedError,
                       match="interior fixed-depth traces"):
        mz.trace(NodeSet.CENTER, Side.HIGH, depth=1)


def test_rejects_periodic_mesh():
    mx = IntervalMesh(N, (0.0, 1.0), name="x")  # periodic
    with pytest.raises(ValueError, match="need a bounded mesh"):
        mx.trace(NodeSet.CENTER, Side.HIGH)


def test_chebyshev_trace_and_local_traits():
    # ChebyshevMesh inherits the structured 1D trace factory (and its
    # node-set restriction: only Outer/Lobatto); its decomposition
    # traits classify the trace factor as collapsed
    cheb = ChebyshevMesh(N, (0.0, 1.0), name="z")
    with pytest.raises(ValueError, match="no CENTER spaces"):
        cheb.trace(NodeSet.CENTER, Side.HIGH)
    trace = cheb.trace(NodeSet.OUTER, Side.HIGH)
    assert isinstance(trace, TraceSpace)
    assert cheb.decomposition_traits(trace) == cheb.decomposition_traits(
        cheb.constant)


# ================================================================
#  The strict algebra: the design win (loud rejections)
# ================================================================
def test_join_factor_identical_trace_joins(mz):
    trace = mz.trace(NodeSet.CENTER, Side.HIGH)
    assert join_factor(trace, trace) is trace


def test_join_factor_trace_and_full_rejects(mz):
    trace = mz.trace(NodeSet.CENTER, Side.HIGH)
    assert join_factor(trace, mz.center) is None
    assert join_factor(mz.center, trace) is None


def test_join_factor_different_side_trace_rejects(mz):
    low = mz.trace(NodeSet.CENTER, Side.LOW)
    high = mz.trace(NodeSet.CENTER, Side.HIGH)
    assert join_factor(low, high) is None


def test_join_factor_trace_and_constant_rejects(mz):
    # a Constant is NEVER implicitly relocated to the boundary row
    trace = mz.trace(NodeSet.CENTER, Side.HIGH)
    assert join_factor(trace, mz.constant) is None
    assert join_factor(mz.constant, trace) is None


def test_field_trace_times_full_raises(grid):
    mx, my, mz = grid.factors
    tspace = mx.center * my.center * mz.trace(NodeSet.CENTER, Side.HIGH)
    trace = grid.create_field(tspace, data=jnp.ones((N, N, 1)))
    full = grid.create_field(mx.center * my.center * mz.center)
    with pytest.raises(SpaceMismatchError, match="cannot combine"):
        _ = trace + full


def test_field_trace_times_profile_raises(grid):
    # Trace x Constant-z Profile: the wind-stress mistake, loud
    mx, my, mz = grid.factors
    tspace = mx.center * my.center * mz.trace(NodeSet.CENTER, Side.HIGH)
    trace = grid.create_field(tspace, data=jnp.ones((N, N, 1)))
    profile = grid.create_field(mx.center * my.center * mz.constant)
    with pytest.raises(SpaceMismatchError, match="cannot combine"):
        _ = trace * profile


def test_field_trace_times_trace_works(grid):
    # identical-Trace elementwise arithmetic resolves on the Trace
    # product (Role B: _resolve_product skips the collapsed factor)
    mx, my, mz = grid.factors
    tspace = mx.center * my.center * mz.trace(NodeSet.CENTER, Side.HIGH)
    data = jnp.arange(N * N, dtype=float).reshape(N, N, 1)
    a = grid.create_field(tspace, data=data)
    b = grid.create_field(tspace, data=data + 1.0)
    product = a * b
    assert product.function_space.bare is tspace
    assert np.allclose(np.asarray(product.data),
                       np.asarray(data) * (np.asarray(data) + 1.0))
    total = a + b
    assert total.function_space.bare is tspace


def test_join_reports_the_trace_mismatch(grid):
    # the strict algebra names the offending trace factor
    mx, my, mz = grid.factors
    tspace = mx.center * my.center * mz.trace(NodeSet.CENTER, Side.HIGH)
    full = mx.center * my.center * mz.center
    with pytest.raises(SpaceMismatchError, match="Trace"):
        join(tspace, full, operation="+")


# ================================================================
#  Storage / decomposition: shards like a Profile, no z halo
# ================================================================
def test_trace_axis_has_no_halo_and_is_length_one(grid):
    mx, my, mz = grid.factors
    tspace = (mx.center * my.center
              * mz.trace(NodeSet.CENTER, Side.HIGH))
    profile = mx.center * my.center * mz.constant
    decomp = grid.decomposition
    trace_storage = decomp.storage_shape(grid._laid_out(tspace))
    profile_storage = decomp.storage_shape(grid._laid_out(profile))
    # the z (collapsed) axis is length 1 in storage, exactly like a
    # ConstantSpace Profile — no halo, no stagger padding
    assert trace_storage[2] == 1
    assert trace_storage[2] == profile_storage[2]


def test_trace_shards_like_a_profile(grid):
    mx, my, mz = grid.factors
    tspace = grid._laid_out(
        mx.center * my.center * mz.trace(NodeSet.CENTER, Side.HIGH))
    profile = grid._laid_out(mx.center * my.center * mz.constant)
    decomp = grid.decomposition
    assert decomp.sharding(tspace).spec == decomp.sharding(profile).spec


def test_export_squeezes_the_trace_axis(grid):
    mx, my, mz = grid.factors
    tspace = mx.center * my.center * mz.trace(NodeSet.CENTER, Side.HIGH)
    field = grid.create_field(tspace, data=jnp.ones((N, N, 1)))
    # the collapsed z axis is squeezed out of the xarray export
    assert list(field.xr.dims) == ["x", "y"]


# ================================================================
#  Role D: measure on a trace factor is a taught error
# ================================================================
def test_measure_on_trace_factor_raises(grid):
    mx, my, mz = grid.factors
    tspace = mx.center * my.center * mz.trace(NodeSet.CENTER, Side.HIGH)
    with pytest.raises(ValueError,
                       match="trace carries no per-factor measure"):
        grid.measure(tspace, name="z")


# ================================================================
#  Multi-device: the trace factor is replicated (forced-4 suite)
# ================================================================
@pytest.mark.multi_device
def test_trace_factor_replicated_multi_device():
    devices = jax.device_count()
    mx = IntervalMesh(16, (0.0, 1.0), name="x")  # periodic, sharded
    my = IntervalMesh(8, (0.0, 2.0), name="y")   # periodic
    mz = IntervalMesh(4, (0.0, 3.0), periodic=False, name="z")
    grid = Grid((mx, my, mz))
    decomp = grid.decomposition
    assert decomp.device_count == devices
    tspace = mx.center * my.center * mz.trace(NodeSet.CENTER, Side.HIGH)
    profile = mx.center * my.center * mz.constant
    trace_laid = grid._laid_out(tspace)
    profile_laid = grid._laid_out(profile)
    # x is the sharded axis; y and the collapsed z are replicated,
    # bitwise-identical sharding to the ConstantSpace Profile
    spec = decomp.sharding(trace_laid).spec
    assert spec[0] == "devices"
    assert spec[2] is None
    assert spec == decomp.sharding(profile_laid).spec
    # the trace z axis stays length 1 in storage, no per-shard block
    assert decomp.storage_shape(trace_laid)[2] == 1
    field = grid.create_field(tspace, data=jnp.ones((16, 8, 1)))
    assert len(field._data.sharding.device_set) == devices
