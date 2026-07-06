"""Tests for the single-device TensorDecomposition."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.tensor import (
    TensorDecomposition,
)

NAMES = ("x", "y")


@dataclass(frozen=True)
class StandInSpace:

    """Minimal frozen stand-in implementing the SpaceLike protocol."""

    shape: tuple
    names: tuple
    layout: object = None

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


def test_multiple_devices_not_implemented():
    with pytest.raises(NotImplementedError, match="Wave 3"):
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

def test_sync_skips_exchange_on_one_device(space):
    decomp = make_decomp(halo=HaloSpec({"x": 1, "y": 0}))
    storage = decomp.zeros(space)
    # sanctioned static branch: no exchange, the array is returned
    assert decomp.sync(storage, space) is storage


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
