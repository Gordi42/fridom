"""Tests for name-keyed state spaces in trace_halo."""
import pytest

from fridom.framework2.grid.decomposition.decomposition import (
    negotiate,
)
from fridom.framework2.grid.decomposition.halo import (
    HaloTracer,
    VectorTracer,
    trace_halo,
)
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


@pytest.fixture
def space(mx, my):
    return TensorProductSpace.of(mx.center, my.center)


def widths(spec):
    return dict(spec.widths)


# ================================================================
#  Name-keyed mapping form
# ================================================================
def test_mapping_components_are_name_addressed(grid, space):
    def tendency(state):
        assert isinstance(state, VectorTracer)
        assert state.component_names == ("u", "v")
        assert "u" in state
        assert "c0" not in state
        return state["u"].diff("x"), state["v"].diff("y")

    spec = trace_halo(tendency, {"u": space, "v": space},
                      grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 1}


def test_mapping_names_key_replace_and_errors(grid, space):
    def tendency(state):
        with pytest.raises(KeyError, match="w"):
            state.replace(w=state["u"])
        with pytest.raises(KeyError, match="c0"):
            _ = state["c0"]
        return state.replace(u=state["v"]).map(
            lambda f: f.diff("y"))

    spec = trace_halo(tendency, {"u": space, "v": space},
                      grid.dispatch)
    assert widths(spec) == {"x": 0, "y": 1}


def test_single_entry_mapping_stays_name_addressed(grid, space):
    def tendency(state):
        assert isinstance(state, VectorTracer)
        return state["u"].diff("x")

    spec = trace_halo(tendency, {"u": space}, grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 0}


def test_empty_mapping_raises(grid):
    with pytest.raises(ValueError, match="at least one"):
        trace_halo(lambda state: state, {}, grid.dispatch)


# ================================================================
#  Positional sequence form (backwards compatibility)
# ================================================================
def test_positional_sequence_keeps_anonymous_naming(grid, space):
    def tendency(state):
        assert isinstance(state, VectorTracer)
        assert state.component_names == ("c0", "c1")
        return state["c0"].diff("x"), state["c1"].diff("y")

    spec = trace_halo(tendency, (space, space), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 1}


def test_lone_positional_space_is_a_bare_tracer(grid, space):
    def tendency(f):
        assert isinstance(f, HaloTracer)
        return f.diff("x")

    spec = trace_halo(tendency, (space,), grid.dispatch)
    assert widths(spec) == {"x": 1, "y": 0}


# ================================================================
#  Negotiation plumbing accepts the mapping form
# ================================================================
def test_negotiate_traces_a_name_keyed_mapping(grid, space):
    def tendency(state):
        return state["u"].diff("x"), state["v"].diff("y")

    decomp = negotiate(grid, grid.dispatch,
                       state_spaces={"u": space, "v": space},
                       tendency=tendency, device_ids=(0,))
    assert decomp.halo["x"] == 1
    assert decomp.halo["y"] == 1


def test_negotiate_scopes_the_registry_halo_by_mapping(grid, my):
    # y-only mapping values silence the x-mesh demands (the y width
    # is 2: the FV-derivative chain is the widest seeded entry)
    decomp = negotiate(grid, grid.dispatch,
                       state_spaces={"v": my.center},
                       device_ids=(0,))
    assert decomp.halo["x"] == 0
    assert decomp.halo["y"] == 2
