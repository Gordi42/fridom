"""Tests for fridom.spatial.operators.staggering."""
import jax
import jax.numpy as jnp
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.staggering import (
    first_node_offset,
    mapped_factor,
    mapped_mesh,
    mapped_order_hint,
    uniform_spacing,
)
from fridom.spatial.spaces.nodal import NodeSet


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


def test_first_node_offset_accepts_dof_preserving_bc_tags(my):
    # Center/Inner carry no boundary members and Neumann keeps the
    # boundary DOF: the offsets of the BC-free table apply unchanged
    assert first_node_offset(
        my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)) == 0.5
    assert first_node_offset(
        my.nodal(NodeSet.CENTER, bc=BC.NEUMANN)) == 0.5
    assert first_node_offset(
        my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)) == 1.0
    assert first_node_offset(
        my.nodal(NodeSet.OUTER, bc=BC.NEUMANN)) == 0.0


def test_first_node_offset_rejects_dirichlet_dropped_dofs(my):
    # Dirichlet on a member node set drops the boundary value DOF,
    # breaking the window-alignment lattice
    with pytest.raises(SpaceMismatchError, match="boundary DOF"):
        first_node_offset(my.nodal(NodeSet.OUTER, bc=BC.DIRICHLET))
    with pytest.raises(SpaceMismatchError, match="boundary DOF"):
        first_node_offset(my.nodal(NodeSet.RIGHT, bc=BC.DIRICHLET))
    # a Dirichlet component on the non-member side drops nothing
    assert first_node_offset(
        my.nodal(NodeSet.RIGHT, bc=(BC.DIRICHLET, BC.NONE))) == 1.0


def test_uniform_spacing(mx, my):
    assert uniform_spacing(mx.center) == 0.125
    assert uniform_spacing(my.inner) == 0.25


def test_uniform_spacing_needs_a_uniform_mesh():
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="s")
    with pytest.raises(NotImplementedError, match=r"grid\.measure"):
        uniform_spacing(cheb.outer)


def test_mapped_factor_routes_on_the_coordinate_map(mx):
    mapped = MappedIntervalMesh(
        8, (0.0, 1.0), lambda s: s**2 / 2 + s / 2, name="v")
    assert mapped_factor(mapped.center) is True
    assert mapped_factor(mx.center) is False
    # the mesh-level spelling (the bind guards of the model modules)
    assert mapped_mesh(mapped) is True
    assert mapped_mesh(mx) is False
    # a mapped mesh deliberately has no scalar dx
    with pytest.raises(NotImplementedError, match=r"grid\.measure"):
        uniform_spacing(mapped.center)


def test_mapped_order_hint_names_the_stencil_and_the_reason():
    hint = mapped_order_hint("the biased face reconstructions")
    assert "the biased face reconstructions" in hint
    assert "uniform-offset" in hint
    assert "silently drop to 2nd order" in hint


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
        def __init__(self, grid, function_space, data, metadata=None,
                     halo_valid=None):
            self.grid = grid
            self.function_space = function_space
            self._data = data
            self.metadata = metadata
            self.halo_valid = (
                HaloSpec.zero(tuple(function_space.names))
                if halo_valid is None else halo_valid)

    field = FieldStandIn(GridStandIn(), mx.center, jnp.arange(8.0))
    with pytest.raises(ValueError, match="halo width 0"):
        FiniteDifference()["x"](field)


def test_bounded_mapped_divide_vjp_is_sealed():
    # reverse-mode AD through a bounded mapped-mesh diff: the codomain
    # measure's zero ghost slots must not turn the boundary rows' VJP
    # into NaN (the double-where seal in divide_by_codomain_measure)
    mesh = MappedIntervalMesh(
        8, (0.0, 1.0), lambda s: s**2 / 2 + s / 2, name="v")
    grid = Grid((mesh,))
    f = grid.random.normal(mesh.center, seed=7)

    def loss(theta):
        return jnp.sum((f * theta).diff("v").data ** 2)

    grad = jax.grad(loss)(jnp.asarray(1.0))
    assert bool(jnp.isfinite(grad))
