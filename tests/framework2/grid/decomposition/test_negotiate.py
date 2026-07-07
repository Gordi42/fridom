"""Tests for the negotiation entry point (decomposition.negotiate)."""
import jax
import pytest

from fridom.framework2.grid.decomposition.decomposition import (
    _registry_halo,
    _shardable_names,
    negotiate,
)
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


# ================================================================
#  Halo sources: explicit > traced > registry maximum
# ================================================================
def test_explicit_halo_wins_and_missing_names_are_zero(grid):
    decomp = negotiate(grid, grid.dispatch, halo=HaloSpec({"x": 4}),
                       device_ids=(0,))
    assert decomp.halo["x"] == 4
    assert decomp.halo["y"] == 0


def test_registry_maximum_is_the_provisional_default(grid):
    decomp = negotiate(grid, grid.dispatch, device_ids=(0,))
    # FiniteDifference/LinearInterp declare halo 1 on both meshes
    assert decomp.halo["x"] == 1
    assert decomp.halo["y"] == 1


def test_traced_tendency_overrides_the_registry_maximum(grid):
    space = grid.create_field().function_space

    def tendency(state):
        return state.diff("y")

    decomp = negotiate(grid, grid.dispatch, state_spaces=(space,),
                       tendency=tendency, device_ids=(0,))
    assert decomp.halo["x"] == 0
    assert decomp.halo["y"] == 1


def test_tendency_without_state_spaces_raises(grid):
    with pytest.raises(ValueError, match="state_spaces"):
        negotiate(grid, grid.dispatch, tendency=lambda state: state)


def test_registry_halo_scopes_to_state_space_meshes(grid, my):
    # scoping to a y-only state silences the x-mesh demands
    spec = _registry_halo(("x", "y"), grid.dispatch,
                          state_spaces=(my.center,))
    assert spec["x"] == 0
    assert spec["y"] == 1


def test_registry_halo_without_items_surface_is_zero():
    spec = _registry_halo(("x",), object())
    assert spec["x"] == 0


# ================================================================
#  Layout selection (device-count independent pieces)
# ================================================================
def test_single_device_negotiation_is_the_trivial_layout(grid):
    decomp = negotiate(grid, grid.dispatch, device_ids=(0,))
    assert isinstance(decomp, TensorDecomposition)
    assert decomp.layouts == (Layout({}),)


def test_shardable_names_respect_divisibility(mx, my):
    halo = HaloSpec({"x": 1, "y": 1})
    # 8 cells over 4 devices: 2 cells/shard >= halo + 1
    assert _shardable_names((mx, my), halo, 4) == ("x", "y")
    # 8 cells over 3 devices do not divide
    assert _shardable_names((mx, my), halo, 3) == ()


def test_shardable_names_respect_the_halo_constraint(mx):
    # 2 cells/shard must cover halo + 1 (short staggered last shard)
    assert _shardable_names((mx,), HaloSpec({"x": 2}), 4) == ()
    assert _shardable_names((mx,), HaloSpec({"x": 1}), 4) == ("x",)


def test_shardable_names_skip_meshes_without_ghost_traits():
    class NoTraitsMesh:
        n_cells = 8
        names = ("q",)

    assert _shardable_names((NoTraitsMesh(),),
                            HaloSpec({"q": 0}), 4) == ()


def test_explicit_devices_with_nothing_shardable_raise():
    mesh = IntervalMesh(5, (0.0, 1.0), name="x")  # 5 % 4 != 0
    grid = Grid((mesh,))
    if jax.device_count() < 2:
        pytest.skip("needs several devices to request them")
    ids = tuple(range(jax.device_count()))
    with pytest.raises(ValueError, match="GHOST-shardable"):
        negotiate(grid, grid.dispatch, device_ids=ids)


def test_auto_selection_falls_back_to_one_device():
    mesh = IntervalMesh(5, (0.0, 1.0), name="x")  # never divisible
    grid = Grid((mesh,))
    assert grid.decomposition.layouts == (Layout({}),)
    assert grid.decomposition._device_mesh.size == 1


# ================================================================
#  Multi-device negotiation (forced-devices suite)
# ================================================================
@pytest.mark.multi_device
def test_default_layout_shards_the_first_ghost_factor(grid):
    decomp = grid.decomposition
    default = decomp.default_layout
    assert dict(default.device_axes) == {"x": "devices"}
    # pencil for the y factor plus the replicated fallback
    assert Layout({"y": "devices"}) in decomp.layouts
    assert decomp.layouts[-1] == Layout({})
    assert decomp._device_mesh.size == jax.device_count()


@pytest.mark.multi_device
def test_layout_for_reaches_a_pencil(grid):
    decomp = grid.decomposition
    pencil = decomp.layout_for(("x",))
    assert pencil.is_local("x")
