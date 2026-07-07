"""Tests for fridom.framework2.grid.operators.reconstruct."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import EigenbasisError
from fridom.framework2.grid.operators.reconstruct import (
    LinearReconstruction,
    fv_node_offset,
)
from fridom.framework2.grid.operators.registry import OperatorRegistry
from fridom.framework2.grid.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def recon():
    return LinearReconstruction()


# ================================================================
#  Construction and static surface
# ================================================================
def test_dispatch_kind(recon):
    assert recon.dispatch_kind == "reconstruct"


def test_target_knob(recon):
    assert recon.target is None
    assert LinearReconstruction(
        target=NodeSet.OUTER).target is NodeSet.OUTER
    with pytest.raises(TypeError, match="NodeSet member"):
        LinearReconstruction(target="outer")


def test_requirements(recon, mx):
    assert recon.requirements(mx.cell_avg).halo == 1
    assert recon.requirements(mx.cell_avg).layout == "any"


def test_eigenvalues_designed_for(recon, mx):
    with pytest.raises(EigenbasisError):
        recon.eigenvalues(None, mx.cell_avg)


# ================================================================
#  FV node offsets (window-alignment calculus)
# ================================================================
def test_fv_node_offsets(mx, my):
    assert fv_node_offset(mx.cell_avg) == 0.5
    assert fv_node_offset(mx.face_avg) == 1.0
    assert fv_node_offset(my.face_avg) == 1.0
    assert fv_node_offset(mx.center) == 0.5  # nodal delegation


# ================================================================
#  Per-factor signatures (codomain table)
# ================================================================
def test_codomain_periodic(recon, mx):
    assert recon.codomain(mx.cell_avg) is mx.right
    assert recon.codomain(mx.right) is mx.cell_avg
    assert recon.codomain(mx.center) is mx.face_avg
    assert recon.codomain(mx.face_avg) is mx.center


def test_codomain_bounded(recon, my):
    assert recon.codomain(my.cell_avg) is my.inner
    assert recon.codomain(my.outer) is my.cell_avg
    assert recon.codomain(my.inner) is my.cell_avg
    assert recon.codomain(my.center) is my.face_avg
    assert recon.codomain(my.face_avg) is my.center


def test_codomain_outer_variant(my, mx):
    outer = LinearReconstruction(target=NodeSet.OUTER)
    assert outer.codomain(my.cell_avg) is my.outer
    with pytest.raises(SpaceMismatchError, match="target="):
        outer.codomain(mx.cell_avg)  # periodic mesh
    with pytest.raises(SpaceMismatchError, match="target="):
        outer.codomain(my.center)  # not CellAvg


def test_codomain_preserves_scalars(recon, mx):
    assert recon.codomain(mx.cell_avg.as_complex()) is (
        mx.right.as_complex())


def test_codomain_rejects_unlisted_node_sets(recon, mx):
    with pytest.raises(SpaceMismatchError,
                       match="no reconstruct signature"):
        recon.codomain(mx.left)


def test_codomain_rejects_bc_structured_spaces(recon, my):
    space = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    with pytest.raises(SpaceMismatchError, match="BC-free"):
        recon.codomain(space)


# ================================================================
#  Application (two-point means over halo-extended storage)
# ================================================================
def test_periodic_cell_avg_to_right_converges(recon):
    errors = []
    for n in (16, 32):
        mesh = IntervalMesh(n, (0.0, 1.0), name="x")
        grid = Grid((mesh,))
        f = grid.create_field(
            mesh.cell_avg, init=lambda x: jnp.sin(2 * jnp.pi * x))
        g = recon["x"](f)
        assert g.function_space.bare is mesh.right
        x = grid.evaluation_nodes(mesh.right).data
        errors.append(
            jnp.abs(g.data - jnp.sin(2 * jnp.pi * x)).max())
    assert errors[0] / errors[1] > 3.0  # second order


def test_right_to_cell_avg_is_exact_on_linears(recon, mx):
    # the cell average of a linear equals the mean of its face
    # values; the first cell consumes the periodic wrap ghost
    grid = Grid((mx,))
    f = grid.create_field(mx.right, data=jnp.arange(8.0))
    g = recon["x"](f)
    assert g.function_space.bare is mx.cell_avg
    expected = jnp.concatenate(
        [jnp.array([(7.0 + 0.0) / 2]),
         (jnp.arange(7.0) + jnp.arange(1.0, 8.0)) / 2])
    assert jnp.allclose(g.data, expected)


def test_bounded_outer_to_cell_avg_is_exact_on_linears(recon, my):
    grid = Grid((my,))
    f = grid.create_field(my.outer, init=lambda y: 3.0 * y - 1.0)
    g = recon["y"](f)
    assert g.function_space.bare is my.cell_avg
    y_c = grid.evaluation_nodes(my.cell_avg).data
    assert jnp.allclose(g.data, 3.0 * y_c - 1.0)


def test_bounded_center_to_face_avg_is_exact_on_linears(recon, my):
    grid = Grid((my,))
    f = grid.create_field(my.center, init=lambda y: 2.0 * y + 1.0)
    g = recon["y"](f)
    assert g.function_space.bare is my.face_avg
    y_f = grid.evaluation_nodes(my.face_avg).data
    assert jnp.allclose(g.data, 2.0 * y_f + 1.0)


def test_outer_variant_extrapolates_the_boundary_faces(my):
    # linear cell averages extrapolate exactly onto the walls
    grid = Grid((my,))
    outer = LinearReconstruction(target=NodeSet.OUTER)
    f = grid.create_field(my.cell_avg, init=lambda y: 4.0 * y)
    g = outer["y"](f)
    assert g.function_space.bare is my.outer
    y_o = grid.evaluation_nodes(my.outer).data
    assert jnp.allclose(g.data, 4.0 * y_o)


def test_metadata_is_kept(recon, mx):
    grid = Grid((mx,))
    f = grid.create_field(mx.cell_avg, name="q", units="kg")
    assert recon["x"](f).name == "q"  # same quantity


def test_reconstruct_needs_the_negotiated_halo(recon, mx):
    # pinned to one device: the halo-0 reach check is asserted in
    # the single-shard storage frame (blocked frames carry a stagger
    # slot that masks it -- flagged for the wave-3 integration pass)
    bare = Grid((mx,), dispatch=OperatorRegistry({}),
                device_ids=(0,))  # halo 0
    f = bare.create_field(mx.cell_avg)
    with pytest.raises(ValueError, match="halo width 0"):
        recon["x"](f)


# ================================================================
#  Registry rows and the .to sugar
# ================================================================
def test_to_reconstructs_average_sources(mx):
    grid = Grid((mx,))
    f = grid.create_field(mx.cell_avg, data=jnp.arange(8.0))
    g = f.to(mx.right)
    assert g.function_space.bare is mx.right


def test_to_averages_nodal_sources(mx):
    # nodal -> average resolves the seeded ("average", ...) rows
    grid = Grid((mx,))
    f = grid.create_field(mx.right, data=jnp.arange(8.0))
    assert f.to(mx.cell_avg).function_space.bare is mx.cell_avg
    h = grid.create_field(mx.center, data=jnp.arange(8.0))
    assert h.to(mx.face_avg).function_space.bare is mx.face_avg


def test_to_rejects_dual_family_transfer(mx):
    # CellAvg -> FaceAvg needs the designed-for dual transfer; the
    # registered reconstruct codomain (Right) does not match
    grid = Grid((mx,))
    f = grid.create_field(mx.cell_avg)
    with pytest.raises(SpaceMismatchError, match="lands on"):
        f.to(mx.face_avg)
