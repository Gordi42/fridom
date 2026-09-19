"""Tests for the single-device TensorDecomposition."""
import collections
import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import (
    TensorDecomposition,
    _fill_axis,
    _write_axis,
)
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.nodal import NodeSet

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
    def collapses_axis(self):
        # a full stand-in factor: not a collapsed (constant/trace) axis
        return False

    @property
    def is_flat(self):
        # not a periodic single-cell axis: no halo elision
        return False

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
    assert decomp.device_count == 1


# ================================================================
#  The multi-slice fallback of the device mesh
# ================================================================
_MULTI_SLICE = "jax.make_mesh does not support multi-slice topologies"


def _refuse(message):
    def make_mesh(*args, **kwargs):  # noqa: ARG001
        raise ValueError(message)
    return make_mesh


@pytest.mark.multi_device
def test_multi_slice_refusal_falls_back_to_the_plain_mesh(monkeypatch):
    # a multi-node GPU run: every node is its own slice and
    # jax.make_mesh refuses the device set. The 1-D mesh is then built
    # directly over the selected devices IN THE ORDER GIVEN
    ids = tuple(reversed(range(jax.device_count())))
    monkeypatch.setattr(jax, "make_mesh", _refuse(_MULTI_SLICE))
    mesh = IntervalMesh(8, (0.0, 1.0), periodic=True, name="x")
    decomp = TensorDecomposition(
        meshes=(mesh,), names=("x",), halo=HaloSpec({"x": 1}),
        layouts=(Layout({"x": "devices"}),), device_ids=ids)
    device_mesh = decomp.device_mesh
    assert device_mesh.axis_names == ("devices",)
    assert device_mesh.axis_types == (jax.sharding.AxisType.Auto,)
    assert device_mesh.devices.shape == (len(ids),)
    assert ([d.id for d in device_mesh.devices.ravel()]
            == [jax.devices()[i].id for i in ids])
    # and the decomposition built on it is a working one
    true = jnp.arange(1.0, 9.0)
    storage = decomp.sync(decomp.pad(true, mesh.center), mesh.center)
    assert np.array_equal(
        np.asarray(decomp.unpad(storage, mesh.center)), np.asarray(true))


@pytest.mark.multi_device
def test_the_fallback_mesh_equals_the_make_mesh_one(monkeypatch):
    # on a device set make_mesh accepts, the fallback spelling is the
    # same mesh: the fallback changes nothing but the refusal
    ids = tuple(range(jax.device_count()))
    kwargs = {"meshes": (IntervalMesh(8, (0.0, 1.0), name="x"),),
              "names": ("x",), "halo": HaloSpec({"x": 1}),
              "layouts": (Layout({"x": "devices"}),), "device_ids": ids}
    regular = TensorDecomposition(**kwargs).device_mesh
    monkeypatch.setattr(jax, "make_mesh", _refuse(_MULTI_SLICE))
    assert TensorDecomposition(**kwargs).device_mesh == regular


def test_any_other_make_mesh_failure_propagates(monkeypatch):
    # only the multi-slice refusal is caught, on any device count
    monkeypatch.setattr(jax, "make_mesh", _refuse("some other refusal"))
    ids = tuple(range(jax.device_count()))
    with pytest.raises(ValueError, match="some other refusal"):
        TensorDecomposition(
            meshes=(IntervalMesh(8, (0.0, 1.0), name="x"),),
            names=("x",), halo=HaloSpec({"x": 1}),
            layouts=(Layout({"x": "devices"}),), device_ids=ids)


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


def test_zeros_honors_dtype(space):
    decomp = make_decomp(halo=HaloSpec({"x": 1, "y": 0}))
    assert decomp.zeros(space).dtype == jnp.zeros(()).dtype
    arr = decomp.zeros(space, dtype=jnp.complex128)
    assert arr.dtype == jnp.complex128
    assert arr.shape == decomp.storage_shape(space)


def test_zeros_is_traceable(space):
    # staged out, zeros is the whole-extent constant (a tracer cannot
    # be committed block by block)
    decomp = make_decomp(halo=HaloSpec({"x": 1, "y": 0}))
    arr = jax.jit(lambda: decomp.zeros(space, dtype=jnp.float32))()
    assert arr.shape == decomp.storage_shape(space)
    assert arr.dtype == jnp.float32
    assert bool(jnp.all(arr == 0))


def test_assemble_on_one_device_is_pad_of_the_whole_piece(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    arr = jnp.arange(40.0).reshape(space.shape)
    boxes = []

    def piece(box):
        boxes.append(box)
        return arr[box]

    stored = decomp.assemble(space, piece)
    # unblocked geometry: exactly one whole-extent piece
    assert boxes == [(slice(0, 8), slice(0, 5))]
    assert stored.sharding == decomp.sharding(space)
    assert np.array_equal(np.asarray(stored),
                          np.asarray(decomp.pad(arr, space)))


def test_assemble_is_traceable(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    arr = jnp.arange(40.0).reshape(space.shape)
    stored = jax.jit(
        lambda a: decomp.assemble(space, lambda box: 2.0 * a[box]))(arr)
    assert np.array_equal(np.asarray(stored),
                          np.asarray(decomp.pad(2.0 * arr, space)))


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


def test_even_shape_equals_true_shape_single_device(space):
    # single device: nothing is blocked, so the padded-even frame is
    # the true frame
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    assert decomp.even_shape(space) == space.shape


def test_unpad_even_equals_unpad_single_device(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    arr = jnp.arange(40.0).reshape(space.shape)
    storage = decomp.pad(arr, space)
    assert bool(jnp.all(decomp.unpad_even(storage, space) == arr))


def test_pad_even_roundtrips_single_device(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    arr = jnp.arange(40.0).reshape(space.shape)
    even = decomp.unpad_even(decomp.pad(arr, space), space)
    storage = decomp.pad_even(even, space)
    assert storage.shape == decomp.storage_shape(space)
    assert bool(jnp.all(decomp.unpad(storage, space) == arr))


def test_unpad_even_rejects_non_storage_shape(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    with pytest.raises(ValueError, match="storage-shaped"):
        decomp.unpad_even(jnp.zeros(space.shape), space)


def test_pad_even_rejects_non_even_shape(space):
    decomp = make_decomp(halo=HaloSpec({"x": 2, "y": 1}))
    with pytest.raises(ValueError, match="padded-even"):
        decomp.pad_even(jnp.zeros((3, 3)), space)


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
#  The halo fill's spelling (a performance contract, not a value one)
# ================================================================
def _opcodes(text: str) -> collections.Counter:
    """Count HLO opcodes (an instruction is `%name = <shape> op(..)`)."""
    return collections.Counter(
        re.findall(r"= \S+ ([a-z][a-z0-9-]*)\(", text))


@pytest.mark.parametrize("periodic", [True, False], ids=["wrap", "bc"])
def test_sync_spells_the_halo_fill_as_one_index_map(periodic):
    # The ghost fill is an INDEX MAP: one gather (plus a sign flip on
    # an odd extension), because every filled slot reads exactly one
    # slot of the same array. Every spelling computes the same values,
    # so no value test can see the difference -- but the spelling
    # decides what the fill costs, and both alternatives lose:
    #
    #   * `concatenate` rebuilds the array. XLA makes it a fusion ROOT,
    #     so an unabsorbed fill materializes a full O(field) buffer and
    #     every synced field round-trips through HBM.
    #   * `dynamic_update_slice` writes in place, but XLA ABSORBS the
    #     write into the consuming kernel rather than materializing it,
    #     and a chain of 2 * ndim DUS is an index-conditional read of an
    #     index-conditional read: a wide stencil consumer re-evaluates
    #     that nest at every offset it reads (+4% on the advective
    #     nonhydro step, where two fusions swallowed 42 of them).
    #
    # The map is absorbed like the DUS and costs one indexed load to
    # re-derive like the concatenate. This guards the spelling on both
    # sides -- it bites on a concatenate rebuild AND on a DUS chain --
    # standalone and, the case that actually matters, fused into a
    # consumer that reads across the halo.
    width, n = 2, 4096
    mesh = IntervalMesh(n, (0.0, 1.0), periodic=periodic, name="x")
    space = (mesh.center if periodic
             else mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))
    decomp = _mesh_decomp(mesh, width)
    storage = decomp.pad(jnp.zeros(n), space)

    standalone = jax.jit(
        lambda s: decomp.sync(s, space), donate_argnums=0,
    ).lower(storage).compile().as_text()

    def stencil(s):
        """Read across the filled halo (the case that actually matters)."""
        s = decomp.sync(s, space)
        return (s[2 * width:] + s[:-2 * width]).sum()

    absorbed = jax.jit(stencil).lower(storage).compile().as_text()

    for text in (standalone, absorbed):
        ops = _opcodes(text)
        assert ops["gather"] == 1
        assert not ops["concatenate"]
        assert not ops["dynamic-update-slice"]


@pytest.mark.parametrize("periodic", [True, False], ids=["wrap", "bc"])
def test_materialized_sync_writes_the_ghosts_in_place(periodic):
    # sync(materialize=True) is the CALLER's claim that the operand is
    # dead after the sync (the model's carry seal): the fill is then
    # spelled as in-place DUS ghost writes behind an
    # optimization_barrier. The barrier keeps consumers from absorbing
    # the write chain (an absorbed chain is re-derived per consumer
    # offset -- the 8c940666 advective regression) and the dead
    # operand lets XLA's in-place emitter patch O(halo) bytes onto the
    # buffer instead of copying O(field). Values are identical to the
    # default index-map spelling by construction; this guards the
    # spelling and the memory bound.
    width, n = 2, 4096
    mesh = IntervalMesh(n, (0.0, 1.0), periodic=periodic, name="x")
    space = (mesh.center if periodic
             else mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))
    decomp = _mesh_decomp(mesh, width)
    storage = decomp.pad(jnp.arange(1.0, n + 1.0), space)

    assert jnp.array_equal(
        decomp.sync(storage, space, materialize=True),
        decomp.sync(storage, space))

    lowered = jax.jit(
        lambda s: decomp.sync(s, space, materialize=True),
        donate_argnums=0,
    ).lower(storage)
    # the barrier is consumed during optimization (it exists to block
    # fusion, then disappears), so it is guarded on the lowered module
    assert "optimization_barrier" in lowered.as_text()
    compiled = lowered.compile()
    ops = _opcodes(compiled.as_text())
    assert ops["dynamic-update-slice"] >= 2  # one write per side
    assert not ops["gather"]
    # (no concatenate assertion: the bounded fill legitimately
    # concatenates the O(halo) ghost slab; the memory bound below is
    # what forbids an O(field) rebuild)
    field_bytes = storage.size * storage.dtype.itemsize
    memory = compiled.memory_analysis()
    assert memory.temp_size_in_bytes < field_bytes // 8


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


def test_bc_free_walled_sides_are_left_untouched(bounded):
    # R1 (boundary_plan.md): a BC-free bounded side defines no
    # exterior values — the sync fills nothing there (no invented
    # extrapolation, no blanket zeros: the slots simply stay)
    decomp = _mesh_decomp(bounded, 1)
    padded = decomp.pad(jnp.asarray([1.0, 2.0, 3.0, 4.0]),
                        bounded.center)
    out = decomp.sync(padded, bounded.center)
    assert jnp.array_equal(out, padded)


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


def test_bc_free_inner_walled_sides_are_left_untouched(bounded):
    decomp = _mesh_decomp(bounded, 1)
    padded = decomp.pad(jnp.asarray([1.0, 2.0, 3.0]), bounded.inner)
    out = decomp.sync(padded, bounded.inner)
    assert jnp.array_equal(out, padded)


def test_average_spaces_are_bc_free_and_skip_walled_sides(bounded):
    # average spaces carry no BC structure, so their walled sides
    # are never filled (R1) — the FV boundary story closes through
    # declared physics (FluxDifference INNER) or graded Fallback,
    # never through a storage-layer fill
    decomp = _mesh_decomp(bounded, 1)
    for space, values in ((bounded.face_avg, [1.0, 2.0, 3.0]),
                          (bounded.cell_avg, [1.0, 2.0, 3.0, 4.0])):
        padded = decomp.pad(jnp.asarray(values), space)
        out = decomp.sync(padded, space)
        assert jnp.array_equal(out, padded)


def test_neumann_fill_on_vacant_boundary_not_grounded(bounded):
    # Inner Neumann: the boundary lattice node is a ghost slot, not
    # a DOF — the even extension carries no datum for it
    space = bounded.nodal(NodeSet.INNER, bc=BC.NEUMANN)
    decomp = _mesh_decomp(bounded, 1)
    with pytest.raises(NotImplementedError, match="Neumann"):
        _filled(decomp, space, [1.0, 2.0, 3.0])


def test_robin_fill_points_at_the_data_path(bounded):
    # Robin fills are data-parameterized (alpha, g dynamic) and
    # arrive with the ('ghost_fill', space) path (stage 2e); until
    # then a sync on a Robin space is a loud, guiding error
    space = bounded.nodal(NodeSet.CENTER, bc=BC.ROBIN)
    decomp = _mesh_decomp(bounded, 1)
    with pytest.raises(NotImplementedError, match="ghost_fill"):
        _filled(decomp, space, [1.0, 2.0, 3.0, 4.0])


def test_robin_keeps_boundary_dofs_like_neumann(bounded):
    # Robin constrains a derivative combination, not a nodal value:
    # it never drops a DOF (unlike Dirichlet)
    assert bounded.nodal(NodeSet.OUTER, bc=BC.ROBIN).shape == (5,)
    assert bounded.nodal(NodeSet.CENTER, bc=BC.ROBIN).shape == (4,)


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


def test_bc_free_single_cell_axis_syncs_untouched():
    # no extrapolation means no minimum-DOF demand: a one-cell
    # BC-free bounded axis syncs (and fills nothing)
    mesh = IntervalMesh(1, (0.0, 1.0), periodic=False, name="y")
    decomp = _mesh_decomp(mesh, 1)
    padded = decomp.pad(jnp.asarray([1.0]), mesh.center)
    out = decomp.sync(padded, mesh.center)
    assert jnp.array_equal(out, padded)


def test_bounded_fill_deeper_than_the_axis_raises(bounded):
    space = bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    decomp = _mesh_decomp(bounded, 5)
    with pytest.raises(NotImplementedError, match="deeper"):
        _filled(decomp, space, [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("materialize", [False, True])
def test_periodic_wrap_wider_than_the_axis_tiles(materialize):
    # a halo wider than the axis needs more than one wrap: the fill
    # tiles the true region instead of refusing. Reachable through a
    # decomposition for 2 <= n < width -- at n = 1 the axis is flat
    # and its halo is elided before any fill runs, so that leg is
    # covered at the fill-function level below
    mesh2 = IntervalMesh(2, (0.0, 1.0), name="x")
    decomp2 = _mesh_decomp(mesh2, 3)
    padded2 = decomp2.pad(jnp.asarray([1.0, 2.0]), mesh2.center)
    out2 = decomp2.sync(padded2, mesh2.center, materialize=materialize)
    assert jnp.array_equal(
        out2, jnp.array([2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]))

    mesh3 = IntervalMesh(3, (0.0, 1.0), name="x")
    decomp3 = _mesh_decomp(mesh3, 4)
    padded3 = decomp3.pad(jnp.asarray([1.0, 2.0, 3.0]), mesh3.center)
    out3 = decomp3.sync(padded3, mesh3.center, materialize=materialize)
    assert jnp.array_equal(
        out3, jnp.array([3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
                         1.0, 2.0, 3.0, 1.0]))


@pytest.mark.parametrize("fill", [_fill_axis, _write_axis])
def test_wide_wrap_of_a_single_dof_axis_tiles(fill):
    # the n = 1 leg of the tiling branch: a flat axis no longer
    # reaches it through a decomposition (its halo is elided), so
    # the fill functions are exercised directly -- the coverage the
    # thin-axis fix bought must not evaporate with the elision
    mesh = IntervalMesh(1, (0.0, 1.0), name="x")
    storage = jnp.asarray([0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0])
    out = fill(storage, 0, 1, 3, mesh.center)
    assert jnp.array_equal(out, jnp.full((7,), 7.0))


@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_periodic_wrap_matches_the_single_wrap_form(n, materialize):
    # the modular fill is a strict generalization: wherever a single
    # wrap suffices it must reproduce the slice form exactly
    mesh = IntervalMesh(n, (0.0, 1.0), name="x")
    values = jnp.arange(1.0, n + 1.0)
    for width in range(1, n + 1):
        decomp = _mesh_decomp(mesh, width)
        out = decomp.sync(decomp.pad(values, mesh.center), mesh.center,
                          materialize=materialize)
        expect = jnp.concatenate(
            [values[n - width:], values, values[:width]])
        assert jnp.array_equal(out, expect), (n, width)


@pytest.mark.parametrize("materialize", [False, True])
def test_periodic_axis_without_dofs_raises(materialize):
    # the modular wrap is undefined without a DOF to wrap onto (numpy
    # would warn and yield 0 rather than raise), so both spellings
    # refuse up front
    space = StandInSpace(shape=(0,), names=("x",),
                         mesh=_PERIODIC_MESH)
    decomp = TensorDecomposition(
        meshes=(_PERIODIC_MESH,), names=("x",),
        halo=HaloSpec({"x": 2}), layouts=(Layout({}),))
    with pytest.raises(ValueError, match="at least one DOF"):
        decomp.sync(jnp.zeros((4,)), space, materialize=materialize)


def test_materialized_sync_mirrors_the_map_edge_cases(bounded):
    # the write spelling honors the same contract as the map: R1
    # (BC-free sides stay untouched)
    decomp = _mesh_decomp(bounded, 1)
    padded = decomp.pad(jnp.asarray([1.0, 2.0, 3.0, 4.0]),
                        bounded.center)
    assert jnp.array_equal(
        decomp.sync(padded, bounded.center, materialize=True), padded)


def test_materialized_sync_mirrors_the_map_on_a_zero_dof_factor():
    # the edge case the 4-DOF Center above cannot reach: a walled
    # one-cell axis empties the Dirichlet face lattice, and the vacant
    # ghost slot then needs a *shape*, not a DOF. The write spelling
    # used to read a DOF for it and refuse where its twin filled zeros.
    # The whole (space, bc, n_cells) matrix lives in
    # test_tensor_flat_axis.py; this keeps the twin claim next to it.
    mesh = IntervalMesh(1, (0.0, 1.0), periodic=False, name="y")
    space = mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    assert space.shape == (0,)
    decomp = _mesh_decomp(mesh, 1)
    padded = decomp.pad(jnp.zeros((0,)), space)
    written = decomp.sync(padded, space, materialize=True)
    assert jnp.array_equal(written,
                           decomp.sync(padded, space, materialize=False))
    assert jnp.array_equal(written, jnp.zeros((2,)))


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


# ================================================================
#  Gather-free output: shard_writes / chunk_hint
# ================================================================
# These build a decomposition that shards ``x`` over *all* available
# devices: a single shard in the default suite (device-count agnostic)
# and a genuine 1-D sharding under the forced-devices suite
# (XLA_FLAGS=--xla_force_host_platform_device_count=4), where the
# padding / stagger-reserve / ghost stripping is actually exercised.
def _sharded(n_cells, *, periodic=False, width=1):
    """Build a decomposition sharding ``x`` over every device."""
    mesh = IntervalMesh(n_cells, (0.0, 1.0), periodic=periodic,
                        name="x")
    decomp = TensorDecomposition(
        meshes=(mesh,), names=("x",), halo=HaloSpec({"x": width}),
        layouts=(Layout({"x": "devices"}),),
        device_ids=tuple(range(jax.device_count())))
    return mesh, decomp


# every space family and divisibility class on P = 4: n_cells 8 is
# divisible (center uniform, outer the +1 stagger surplus, inner/
# face_avg a mild deficit); 7 and 10 are non-divisible (mild cell
# padding, several inner/face_avg spaces empty their last shard).
_SPACE_CASES = [
    pytest.param(False, "center", id="bounded-center"),
    pytest.param(False, "outer", id="bounded-outer"),
    pytest.param(False, "inner", id="bounded-inner"),
    pytest.param(False, "face_avg", id="bounded-face_avg"),
    pytest.param(True, "center", id="periodic-center"),
    pytest.param(True, "face_avg", id="periodic-face_avg"),
]
_N_CELLS = [pytest.param(7, id="ncells7"),
            pytest.param(8, id="ncells8"),
            pytest.param(10, id="ncells10")]


@pytest.mark.parametrize("n_cells", _N_CELLS)
@pytest.mark.parametrize(("periodic", "attr"), _SPACE_CASES)
def test_shard_writes_tile_by_tile_equals_gather(
        n_cells, periodic, attr, forced_devices):
    # the core contract: writing every locally-owned tile into a
    # true-shape buffer reproduces gather() exactly, every true index
    # is covered exactly once, and the tiles are host numpy.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    mesh, decomp = _sharded(n_cells, periodic=periodic)
    space = getattr(mesh, attr)
    arr = jnp.arange(1.0, space.shape[0] + 1.0)
    storage = decomp.pad(arr, space)
    gathered = np.asarray(decomp.gather(storage, space))
    out = np.zeros(space.shape)
    counter = np.zeros(space.shape, dtype=int)
    for target, values in decomp.shard_writes(storage, space):
        assert isinstance(values, np.ndarray)  # host copy, not device
        out[target] = values
        counter[target] += 1
    assert np.array_equal(out, gathered)
    assert np.array_equal(counter, np.ones(space.shape, dtype=int))


@pytest.mark.parametrize("n_cells", _N_CELLS)
@pytest.mark.parametrize(("periodic", "attr"), _SPACE_CASES)
def test_chunk_hint_is_cells_on_blocked_axis(
        n_cells, periodic, attr, forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    mesh, decomp = _sharded(n_cells, periodic=periodic)
    space = getattr(mesh, attr)
    devices = jax.device_count()
    # blocked axis -> ceil(n_cells / devices) cells; unblocked (single
    # device) -> the full true extent n
    expected = -(-n_cells // devices) if devices > 1 else space.shape[0]
    assert decomp.chunk_hint(space) == (expected,)


def test_shard_writes_dedupes_replicated_factor(forced_devices):
    # a constant factor is replicated on every device: its storage has
    # one true DOF held on all shards (replica_id 0..P-1), so the
    # replica-0 skip must collapse it to a single tile covering the DOF
    # exactly once.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    _mesh, decomp = _sharded(8)
    const = decomp._meshes[0].constant
    arr = decomp.zeros(const)
    writes = decomp.shard_writes(arr, const)
    assert len(writes) == 1
    counter = np.zeros(const.shape, dtype=int)
    for target, _values in writes:
        counter[target] += 1
    assert np.array_equal(counter, np.ones(const.shape, dtype=int))


def test_shard_writes_2d_blocked_and_replicated_axis(forced_devices):
    # blocked x (device axis) composed with an unblocked, replicated y:
    # the per-axis source/target composition must still tile to gather,
    # and chunk_hint is cells on x, full n on y.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")
    decomp = TensorDecomposition(
        meshes=(mx, my), names=("x", "y"),
        halo=HaloSpec({"x": 1, "y": 1}),
        layouts=(Layout({"x": "devices"}),),
        device_ids=tuple(range(jax.device_count())))
    space = mx.center * my.outer
    arr = jnp.arange(
        float(space.shape[0] * space.shape[1])).reshape(space.shape)
    storage = decomp.pad(arr, space)
    gathered = np.asarray(decomp.gather(storage, space))
    out = np.zeros(space.shape)
    counter = np.zeros(space.shape, dtype=int)
    for target, values in decomp.shard_writes(storage, space):
        out[target] = values
        counter[target] += 1
    assert np.array_equal(out, gathered)
    assert np.array_equal(counter, np.ones(space.shape, dtype=int))
    devices = jax.device_count()
    x_chunk = -(-8 // devices) if devices > 1 else 8
    assert decomp.chunk_hint(space) == (x_chunk, space.shape[1])


# ================================================================
#  Gather-free output: the fully-replicated / misaligned frames
# ================================================================
# non-divisible cell counts so the padded-even storage genuinely pads
# (block > cells) and a replicated array must tile EVERY block, not just
# block 0 (the fully-replicated output regression).
_N_CELLS_PADDED = [pytest.param(7, id="ncells7"),
                   pytest.param(10, id="ncells10")]


@pytest.mark.parametrize("n_cells", _N_CELLS_PADDED)
@pytest.mark.parametrize(("periodic", "attr"), _SPACE_CASES)
def test_shard_writes_replicated_array_tiles_every_block(
        n_cells, periodic, attr, forced_devices):
    # regression: a fully replicated storage array (PartitionSpec()) has
    # a single replica-0 shard whose window spans the whole blocked axis,
    # so shard_writes must walk EVERY storage block to reproduce gather.
    # Before the fix only block 0's window was written (3/4 of the domain
    # came back silently zero). On one device this degenerates to the
    # unblocked single-tile path; under the forced-4 suite it exercises
    # the replicated multi-block tiling.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    mesh, decomp = _sharded(n_cells, periodic=periodic)
    space = getattr(mesh, attr)
    arr = jnp.arange(1.0, space.shape[0] + 1.0)
    storage = decomp.pad(arr, space)
    replicated = jax.device_put(
        storage,
        jax.sharding.NamedSharding(
            decomp.device_mesh, jax.sharding.PartitionSpec()))
    gathered = np.asarray(decomp.gather(storage, space))
    out = np.zeros(space.shape)
    counter = np.zeros(space.shape, dtype=int)
    writes = decomp.shard_writes(replicated, space)
    for target, values in writes:
        assert isinstance(values, np.ndarray)  # host copy, not device
        out[target] = values
        counter[target] += 1
    assert np.array_equal(out, gathered)
    assert np.array_equal(counter, np.ones(space.shape, dtype=int))
    if jax.device_count() > 1:
        # the regression signal: the one replica-0 shard yielded a write
        # per storage block along the blocked axis, not a single tile.
        replica0 = [shard for shard in replicated.addressable_shards
                    if shard.replica_id == 0]
        assert len(replica0) == 1
        assert len(writes) > 1


def test_shard_writes_misaligned_window_raises(forced_devices):
    # a shard window that does not fall on the storage-block grid is not
    # in this space's storage frame and must raise. The 4-device storage
    # has block = cells + 1 + 2 * width = 3 + 1 + 2 = 6 and total = 4 * 6
    # = 24; resharding onto a 3-device subset gives 24 / 3 = 8 per shard,
    # and 8 % 6 == 2, so the first window [0, 8) is off the block grid.
    if jax.device_count() < 4:
        pytest.skip("requires >= 4 jax devices for a 3-device subset")
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    mesh, decomp = _sharded(10, periodic=False)
    space = mesh.center
    arr = jnp.arange(1.0, space.shape[0] + 1.0)
    storage = decomp.pad(arr, space)
    subset = jax.sharding.Mesh(np.array(jax.devices()[:3]), ("d",))
    misaligned = jax.device_put(
        storage,
        jax.sharding.NamedSharding(
            subset, jax.sharding.PartitionSpec("d")))
    with pytest.raises(ValueError, match="not aligned with the"):
        decomp.shard_writes(misaligned, space)


def test_shard_writes_and_chunk_hint_honor_explicit_layout():
    mesh, decomp = _sharded(8)
    space = mesh.center
    storage = decomp.pad(jnp.arange(1.0, 9.0), space)
    # an explicit default layout resolves identically to None
    default = decomp.shard_writes(storage, space)
    explicit = decomp.shard_writes(storage, space, decomp.default_layout)
    assert len(default) == len(explicit)
    for (ta, va), (tb, vb) in zip(default, explicit, strict=True):
        assert ta == tb
        assert np.array_equal(va, vb)
    assert decomp.chunk_hint(space) == decomp.chunk_hint(
        space, decomp.default_layout)


def test_shard_writes_and_chunk_hint_reject_foreign_layout():
    mesh, decomp = _sharded(8)
    space = mesh.center
    storage = decomp.pad(jnp.arange(1.0, 9.0), space)
    foreign = Layout({"x": "other"})
    with pytest.raises(ValueError, match="vocabulary"):
        decomp.shard_writes(storage, space, foreign)
    with pytest.raises(ValueError, match="vocabulary"):
        decomp.chunk_hint(space, foreign)


# ================================================================
#  patch_physical_ends: the co-array seam
# ================================================================
def _last_block_start(n_cells):
    """Return the global index of the last block's first true cell.

    The patch callbacks below read their block's FIRST true slot, so
    the high physical end (which lives in the last block) sees cell 0
    on one device and the last shard's first cell on several.
    """
    devices = jax.device_count()
    return (devices - 1) * -(-n_cells // devices)


def test_patch_physical_ends_passes_co_arrays_to_the_callback():
    # the geometry seam of the non-uniform (stretched-mesh) graded
    # rungs: extra input-frame storage arrays reach the callback as
    # trailing block arguments, so a rung's window indices address
    # BLOCK-LOCAL data (a closed-over array would stay global under
    # ``shard_map``). Runs on any device count: the high end reads the
    # LAST block's first slot, which is exactly the block-local claim
    # (the operator-level sharded gate lives in
    # tests/spatial/operators/test_weno_nonuniform.py).
    mesh, decomp = _sharded(8, width=1)
    inner, cell_avg = mesh.inner, mesh.cell_avg
    out_arr = decomp.zeros(inner)
    in_arr = decomp.pad(jnp.arange(1.0, 9.0), cell_avg)
    widths = decomp.pad(jnp.arange(10.0, 90.0, 10.0), cell_avg)
    seen = []

    def patch(in_block, out_block, side, width_in, _t_in,
              width_out, t_out, *co_blocks):
        seen.append((side, len(co_blocks)))
        assert co_blocks[0].shape == in_block.shape
        value = jax.lax.dynamic_slice_in_dim(
            co_blocks[0], width_in, 1, 0)
        slot = width_out if side == 0 else width_out + t_out - 1
        return jax.lax.dynamic_update_slice_in_dim(
            out_block, value, slot, 0)

    patched = decomp.patch_physical_ends(
        out_arr, in_arr, inner, cell_avg, "x", patch,
        co_arrays=(widths,))
    assert seen == [(0, 1), (1, 1)]
    true = np.asarray(decomp.unpad(patched, inner))
    assert true[0] == 10.0        # the co-array's first true slot
    assert true[-1] == 10.0 * (_last_block_start(8) + 1)


def test_patch_physical_ends_without_co_arrays_is_unchanged():
    mesh, decomp = _sharded(8, width=1)
    inner, cell_avg = mesh.inner, mesh.cell_avg
    out_arr = decomp.zeros(inner)
    in_arr = decomp.pad(jnp.arange(1.0, 9.0), cell_avg)

    def patch(in_block, out_block, side, width_in, _t_in,
              width_out, t_out, *co_blocks):
        assert co_blocks == ()
        value = jax.lax.dynamic_slice_in_dim(in_block, width_in, 1, 0)
        slot = width_out if side == 0 else width_out + t_out - 1
        return jax.lax.dynamic_update_slice_in_dim(
            out_block, value, slot, 0)

    patched = decomp.patch_physical_ends(
        out_arr, in_arr, inner, cell_avg, "x", patch)
    true = np.asarray(decomp.unpad(patched, inner))
    assert true[0] == 1.0
    assert true[-1] == 1.0 + _last_block_start(8)
