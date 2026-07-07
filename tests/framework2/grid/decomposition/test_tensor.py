"""Tests for the single-device TensorDecomposition."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.spaces.nodal import NodeSet

NAMES = ("x", "y")


@dataclass(frozen=True)
class StandInMesh:

    """Minimal mesh stand-in (topology only, for the halo fill)."""

    periodic: bool = True


_PERIODIC_MESH = StandInMesh()


@dataclass(frozen=True)
class StandInSpace:

    """Minimal frozen stand-in implementing the SpaceLike protocol."""

    shape: tuple
    names: tuple
    layout: object = None
    mesh: object = _PERIODIC_MESH

    @property
    def factors(self):
        return (self,)

    def factor(self, name):
        if name in self.names:
            return self
        raise KeyError(name)


def make_decomp(halo=None, layouts=None, names=NAMES):
    halo = HaloSpec.zero(names) if halo is None else halo
    layouts = (Layout({}),) if layouts is None else layouts
    return TensorDecomposition(
        meshes=(object(), object()),
        names=names,
        halo=halo,
        layouts=layouts,
    )


@pytest.fixture
def space():
    return StandInSpace(shape=(8, 5), names=NAMES)


# ================================================================
#  Construction and negotiated structure
# ================================================================

def test_negotiated_structure_properties():
    halo = HaloSpec({"x": 1, "y": 0})
    layouts = (Layout({}), Layout({"x": "px"}))
    decomp = make_decomp(halo=halo, layouts=layouts)
    assert decomp.halo == halo
    assert decomp.default_layout == Layout({})
    assert decomp.layouts == layouts


def test_one_device_mesh_regardless_of_device_count(forced_devices):
    if forced_devices is not None:
        # the forced-devices suite must actually see the devices
        assert jax.device_count() == forced_devices
    decomp = make_decomp()
    assert decomp._device_mesh.size == 1


def test_layouts_sharing_a_device_axis_share_one_mesh_axis():
    # two pencils over the same device axis (the main/alt pattern)
    layouts = (Layout({"x": "px"}), Layout({"y": "px"}))
    decomp = make_decomp(layouts=layouts)
    assert decomp._device_mesh.axis_names == ("px",)
    assert decomp._device_mesh.size == 1


def test_empty_layouts_rejected():
    with pytest.raises(ValueError, match="at least one Layout"):
        make_decomp(layouts=())


def test_layout_with_unknown_name_rejected():
    with pytest.raises(ValueError, match="unknown coordinate name"):
        make_decomp(layouts=(Layout({"z": "pz"}),))


def test_duplicate_device_ids_rejected():
    with pytest.raises(ValueError, match="duplicate device ids"):
        TensorDecomposition(
            meshes=(object(),),
            names=("x",),
            halo=HaloSpec.zero(("x",)),
            layouts=(Layout({}),),
            device_ids=(0, 0),
        )


# ================================================================
#  Shapes and shardings
# ================================================================

def test_sharding_default_layout_is_replicated(space):
    decomp = make_decomp()
    sharding = decomp.sharding(space)
    assert isinstance(sharding, jax.sharding.NamedSharding)
    assert sharding.mesh.size == 1
    assert sharding.spec == jax.sharding.PartitionSpec(None, None)


def test_sharding_with_pencil_layout(space):
    layouts = (Layout({}), Layout({"x": "px"}))
    decomp = make_decomp(layouts=layouts)
    sharding = decomp.sharding(space, Layout({"x": "px"}))
    assert sharding.spec == jax.sharding.PartitionSpec("px", None)
    assert "px" in sharding.mesh.shape


def test_space_layout_used_when_set():
    layouts = (Layout({}), Layout({"x": "px"}))
    decomp = make_decomp(layouts=layouts)
    space = StandInSpace(
        shape=(8, 5), names=NAMES, layout=Layout({"x": "px"}))
    assert decomp.sharding(space).spec == jax.sharding.PartitionSpec(
        "px", None)


def test_foreign_layout_rejected(space):
    decomp = make_decomp()
    with pytest.raises(ValueError, match="vocabulary"):
        decomp.sharding(space, Layout({"x": "px"}))


def test_storage_shape_equals_true_shape_for_zero_halo(space):
    decomp = make_decomp()
    assert decomp.storage_shape(space) == space.shape


def test_storage_shape_adds_halo_per_name(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    assert decomp.storage_shape(space) == (8 + 4, 5 + 2)


def test_names_outside_halo_spec_carry_width_zero():
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    # a coefficient-style space whose name the negotiation never saw
    space = StandInSpace(shape=(8, 5), names=("x", "ky"))
    assert decomp.storage_shape(space) == (12, 5)


def test_local_slice_is_full_true_extent(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    assert decomp.local_slice(space) == (slice(0, 8), slice(0, 5))


# ================================================================
#  Storage construction and views
# ================================================================

def test_zeros_is_storage_shaped_sharded_and_zero(space):
    decomp = make_decomp(halo=HaloSpec({"x": 1, "y": 0}))
    arr = decomp.zeros(space)
    assert arr.shape == decomp.storage_shape(space)
    assert arr.sharding == decomp.sharding(space)
    assert bool(jnp.all(arr == 0))


def test_pad_produces_storage_shape(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    arr = jnp.arange(40.0).reshape(space.shape)
    padded = decomp.pad(arr, space)
    assert padded.shape == decomp.storage_shape(space)
    assert padded.sharding == decomp.sharding(space)
    # the true extent sits behind the leading halo offsets
    assert bool(jnp.all(padded[2:10, 1:6] == arr))


def test_pad_unpad_roundtrip(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    arr = jnp.arange(40.0).reshape(space.shape)
    assert bool(jnp.all(decomp.unpad(decomp.pad(arr, space), space)
                        == arr))


def test_pad_rejects_non_true_shape(space):
    decomp = make_decomp()
    with pytest.raises(ValueError, match="true-shape"):
        decomp.pad(jnp.zeros((3, 3)), space)


def test_unpad_rejects_non_storage_shape(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    with pytest.raises(ValueError, match="storage-shaped"):
        decomp.unpad(jnp.zeros(space.shape), space)


def test_zeros_pad_sync_unpad_preserves_values(space):
    decomp = make_decomp(halo=HaloSpec({"x": 1, "y": 1}))
    zeros = decomp.unpad(decomp.zeros(space), space)
    assert bool(jnp.all(zeros == 0))
    arr = jnp.arange(40.0).reshape(space.shape)
    storage = decomp.sync(decomp.pad(arr, space), space)
    assert bool(jnp.all(decomp.unpad(storage, space) == arr))


# ================================================================
#  Data movement
# ================================================================

def test_sync_zero_halo_is_identity_on_one_device(space):
    decomp = make_decomp()
    storage = decomp.zeros(space)
    # sanctioned static branch: no exchange and (with all widths 0)
    # no fill either — the array is returned unchanged
    assert decomp.sync(storage, space) is storage


def test_sync_fills_periodic_ghosts_locally(space):
    decomp = make_decomp(halo=HaloSpec({"x": 1, "y": 0}))
    arr = jnp.arange(40.0).reshape(space.shape)
    storage = decomp.sync(decomp.pad(arr, space), space)
    # single device: no exchange, but the ghost slots are wrapped
    assert jnp.array_equal(storage[0], arr[-1])
    assert jnp.array_equal(storage[-1], arr[0])
    assert jnp.array_equal(decomp.unpad(storage, space), arr)


def test_sync_rejects_inhomogeneous_fills(space):
    decomp = make_decomp()
    storage = decomp.zeros(space)
    with pytest.raises(NotImplementedError, match="ghost fill"):
        decomp.sync(storage, space, fills={"x": jnp.zeros(5)})


def test_layout_for_returns_first_matching_layout():
    layouts = (Layout({"x": "px"}), Layout({}))
    decomp = make_decomp(layouts=layouts)
    assert decomp.layout_for(("y",)) == Layout({"x": "px"})
    assert decomp.layout_for(("x",)) == Layout({})
    assert decomp.layout_for(("x", "y")) == Layout({})


def test_layout_for_without_match_raises():
    decomp = make_decomp(layouts=(Layout({"x": "px"}),))
    with pytest.raises(ValueError, match="device-local"):
        decomp.layout_for(("x",))


def test_redistribute_between_negotiated_layouts(space):
    layouts = (Layout({}), Layout({"x": "px"}))
    decomp = make_decomp(layouts=layouts)
    arr = decomp.pad(jnp.arange(40.0).reshape(space.shape), space)
    moved = decomp.redistribute(arr, space, layouts[0], layouts[1])
    assert bool(jnp.all(moved == arr))
    assert moved.sharding == decomp.sharding(space, layouts[1])


def test_redistribute_rejects_non_storage_shape(space):
    layouts = (Layout({}), Layout({"x": "px"}))
    decomp = make_decomp(halo=HaloSpec({"x": 1, "y": 0}),
                         layouts=layouts)
    with pytest.raises(ValueError, match="storage-shaped"):
        decomp.redistribute(
            jnp.zeros(space.shape), space, layouts[0], layouts[1])


def test_redistribute_rejects_foreign_layout(space):
    decomp = make_decomp()
    arr = decomp.zeros(space)
    with pytest.raises(ValueError, match="vocabulary"):
        decomp.redistribute(
            arr, space, decomp.default_layout, Layout({"y": "py"}))


def test_gather_returns_global_true_shape(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    arr = jnp.arange(40.0).reshape(space.shape)
    gathered = decomp.gather(decomp.pad(arr, space), space)
    assert gathered.shape == space.shape
    assert bool(jnp.all(gathered == arr))


# ================================================================
#  Single-device BC-structured halo fill (real mesh spaces)
# ================================================================
def _mesh_decomp(mesh, width):
    name = mesh.names[0]
    return TensorDecomposition(
        meshes=(mesh,), names=(name,),
        halo=HaloSpec({name: width}), layouts=(Layout({}),))


@pytest.fixture
def bounded():
    return IntervalMesh(4, (0.0, 1.0), periodic=False, name="y")


def _filled(decomp, space, values):
    return decomp.sync(decomp.pad(jnp.asarray(values), space), space)


def test_periodic_wrap_fill_real_space():
    mesh = IntervalMesh(4, (0.0, 1.0), name="x")
    decomp = _mesh_decomp(mesh, 2)
    out = _filled(decomp, mesh.center, [1.0, 2.0, 3.0, 4.0])
    assert jnp.array_equal(
        out, jnp.array([3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0]))


def test_bc_free_fill_is_linear_extrapolation(bounded):
    decomp = _mesh_decomp(bounded, 1)
    out = _filled(decomp, bounded.center, [1.0, 2.0, 3.0, 4.0])
    # one-sided linear extrapolation, never blanket zeros
    assert jnp.array_equal(
        out, jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


def test_dirichlet_fill_is_the_odd_extension(bounded):
    space = bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    decomp = _mesh_decomp(bounded, 2)
    out = _filled(decomp, space, [1.0, 2.0, 3.0, 4.0])
    assert jnp.array_equal(
        out,
        jnp.array([-2.0, -1.0, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0]))


def test_neumann_fill_is_the_even_extension(bounded):
    space = bounded.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    decomp = _mesh_decomp(bounded, 1)
    out = _filled(decomp, space, [1.0, 2.0, 3.0, 4.0])
    assert jnp.array_equal(
        out, jnp.array([1.0, 1.0, 2.0, 3.0, 4.0, 4.0]))


def test_dirichlet_fill_on_face_lattice_zeroes_the_boundary(bounded):
    space = bounded.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    decomp = _mesh_decomp(bounded, 2)
    out = _filled(decomp, space, [1.0, 2.0, 3.0])
    # ghost slot 1 IS the boundary (0); deeper slots odd-reflect
    assert jnp.array_equal(
        out, jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0, 0.0, -3.0]))


def test_bc_free_fill_on_inner_extrapolates_boundary_faces(bounded):
    decomp = _mesh_decomp(bounded, 1)
    out = _filled(decomp, bounded.inner, [1.0, 2.0, 3.0])
    assert jnp.array_equal(
        out, jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_face_avg_fill_uses_the_vacant_boundary_geometry(bounded):
    decomp = _mesh_decomp(bounded, 1)
    out = _filled(decomp, bounded.face_avg, [1.0, 2.0, 3.0])
    assert jnp.array_equal(
        out, jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_cell_avg_fill_uses_the_offset_geometry(bounded):
    decomp = _mesh_decomp(bounded, 1)
    out = _filled(decomp, bounded.cell_avg, [1.0, 2.0, 3.0, 4.0])
    assert jnp.array_equal(
        out, jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))


def test_neumann_fill_on_vacant_boundary_not_grounded(bounded):
    # Inner Neumann: the boundary lattice node is a ghost slot, not
    # a DOF — the even extension carries no datum for it
    space = bounded.nodal(NodeSet.INNER, bc=BC.NEUMANN)
    decomp = _mesh_decomp(bounded, 1)
    with pytest.raises(NotImplementedError, match="Neumann"):
        _filled(decomp, space, [1.0, 2.0, 3.0])


def test_neumann_outer_keeps_nodes_and_mirrors_about_them(bounded):
    # Neumann never drops the boundary DOF (owner decision
    # 2026-07-07): the even extension reflects about the boundary
    # node, which is excluded from the reflection
    space = bounded.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    assert space.shape == (5,)  # all n + 1 nodes kept
    decomp = _mesh_decomp(bounded, 2)
    out = _filled(decomp, space, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert jnp.array_equal(
        out,
        jnp.array([3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]))


def test_neumann_outer_fill_is_fd_consistent_at_the_wall(bounded):
    # symmetric analytic profile cos(pi x) on [0, 1]: derivative
    # zero at both walls, so the centered FD through the fill must
    # vanish at the boundary nodes and stay second-order accurate
    # one node in
    space = bounded.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    decomp = _mesh_decomp(bounded, 1)
    dx = 0.25
    x = jnp.linspace(0.0, 1.0, 5)
    out = _filled(decomp, space, jnp.cos(jnp.pi * x))
    # centered difference at the wall nodes (storage index 1 and 5)
    left_slope = (out[2] - out[0]) / (2 * dx)
    right_slope = (out[6] - out[4]) / (2 * dx)
    assert left_slope == 0.0
    assert right_slope == 0.0
    # one node in: matches -pi sin(pi dx) to second order
    slope_in = (out[3] - out[1]) / (2 * dx)
    exact = -jnp.pi * jnp.sin(jnp.pi * dx)
    assert jnp.abs(slope_in - exact) < 0.5 * dx**2 * jnp.pi**3


def test_dirichlet_outer_drops_and_fills_like_inner(bounded):
    space = bounded.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    assert space.shape == (3,)  # boundary DOFs dropped
    decomp = _mesh_decomp(bounded, 1)
    out = _filled(decomp, space, [1.0, 2.0, 3.0])
    assert jnp.array_equal(
        out, jnp.array([0.0, 1.0, 2.0, 3.0, 0.0]))


def test_bc_free_fill_needs_two_dofs():
    mesh = IntervalMesh(1, (0.0, 1.0), periodic=False, name="y")
    decomp = _mesh_decomp(mesh, 1)
    with pytest.raises(NotImplementedError, match="two DOFs"):
        _filled(decomp, mesh.center, [1.0])


def test_bounded_fill_deeper_than_the_axis_raises(bounded):
    space = bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    decomp = _mesh_decomp(bounded, 5)
    with pytest.raises(NotImplementedError, match="deeper"):
        _filled(decomp, space, [1.0, 2.0, 3.0, 4.0])


def test_periodic_wrap_wider_than_the_axis_raises():
    mesh = IntervalMesh(2, (0.0, 1.0), name="x")
    decomp = _mesh_decomp(mesh, 3)
    with pytest.raises(NotImplementedError, match="wider"):
        _filled(decomp, mesh.center, [1.0, 2.0])


def test_coefficient_factors_carry_no_halo_storage():
    mesh = IntervalMesh(8, (0.0, 1.0), name="x")
    decomp = _mesh_decomp(mesh, 2)
    space = mesh.fourier(origin=mesh.center)
    assert decomp.storage_shape(space) == space.shape
    assert decomp.storage_shape(mesh.center) == (12,)


def test_constant_factors_carry_no_halo_storage():
    mesh = IntervalMesh(8, (0.0, 1.0), name="x")
    decomp = _mesh_decomp(mesh, 2)
    assert decomp.storage_shape(mesh.constant) == (1,)
