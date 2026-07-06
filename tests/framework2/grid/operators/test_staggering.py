"""Tests for fridom.framework2.grid.operators.staggering."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.meshes.chebyshev import ChebyshevMesh
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.staggering import (
    first_node_offset,
    uniform_spacing,
)
from fridom.framework2.grid.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


def test_first_node_offsets(mx, my):
    assert first_node_offset(mx.center) == 0.5
    assert first_node_offset(mx.left) == 0.0
    assert first_node_offset(mx.right) == 1.0
    assert first_node_offset(my.outer) == 0.0
    assert first_node_offset(my.inner) == 1.0


def test_first_node_offset_rejects_non_nodal(mx):
    with pytest.raises(SpaceMismatchError, match="nodal"):
        first_node_offset(mx.cell_avg)


def test_first_node_offset_rejects_bc_structured(my):
    space = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    with pytest.raises(SpaceMismatchError, match="BC-free"):
        first_node_offset(space)


def test_uniform_spacing(mx, my):
    assert uniform_spacing(mx.center) == 0.125
    assert uniform_spacing(my.inner) == 0.25


def test_uniform_spacing_needs_a_uniform_mesh():
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="s")
    with pytest.raises(NotImplementedError, match=r"grid\.measure"):
        uniform_spacing(cheb.outer)


def test_axis_missing_from_the_halo_spec_counts_as_width_zero(mx):
    # a decomposition whose halo spec never saw the axis: the
    # kernel sees width 0 and reports the storage as too small

    decomp = TensorDecomposition(
        meshes=(mx,), names=("x",), halo=HaloSpec({}),
        layouts=(Layout({}),))

    class GridStandIn:
        decomposition = decomp

        def sync(self, field):
            return field

    class FieldStandIn:
        def __init__(self, grid, function_space, data, metadata=None):
            self.grid = grid
            self.function_space = function_space
            self._data = data
            self.metadata = metadata

    field = FieldStandIn(GridStandIn(), mx.center, jnp.arange(8.0))
    with pytest.raises(ValueError, match="halo width 0"):
        FiniteDifference()["x"](field)
