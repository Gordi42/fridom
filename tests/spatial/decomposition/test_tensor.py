"""Tests for the single-device TensorDecomposition."""
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
