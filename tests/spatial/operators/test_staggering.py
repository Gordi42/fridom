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
    divide_by_codomain_measure,
    exterior_reach,
    first_node_offset,
    footprint_reach,
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


# ================================================================
#  divide_by_codomain_measure: reverse-mode seal (AGENTS.md diff
#  policy). A bounded stretched axis's measure has exactly-zero
#  ghost slots; the raw storage-frame quotient is a masked
#  singularity that NaNs jax.grad through the difference.
# ================================================================
def _wavy(s):
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


def _stretched_diff_loss(space_attr, periodic):
    def loss(c):
        mesh = MappedIntervalMesh(8, (0.0, 1.0), _wavy,
                                  periodic=periodic, name="v")
        grid = Grid((mesh,))
        space = getattr(mesh, space_attr)
        f = grid.create_field(
            space, init=lambda v: jnp.sin(2.0 * jnp.pi * v)) * c
        return jnp.sum(f.diff("v").data ** 2)
    return loss


@pytest.mark.parametrize("space_attr", ["cell_avg", "center"])
def test_stretched_diff_grad_is_finite_and_matches_fd(space_attr):
    # jax.grad through the mapped-mesh diff (the codomain-measure
    # divide) on a BOUNDED stretched axis is finite and matches a
    # central FD: the FV (flux_diff) row and the nodal (finite_
    # difference) row both route through divide_by_codomain_measure.
    loss = _stretched_diff_loss(space_attr, periodic=False)
    c0 = 1.3
    grad = float(jax.grad(loss)(c0))
    assert bool(jnp.isfinite(grad))
    assert grad != 0.0
    h = 1e-4
    fd = float((loss(c0 + h) - loss(c0 - h)) / (2.0 * h))
    assert abs(grad - fd) <= 1e-4 * abs(fd)


def test_stretched_diff_forward_mode_jvp_is_finite():
    # the seal is a double-jnp.where (never a custom_vjp), so forward
    # mode is untouched: one jax.jvp through the mapped diff stays
    # finite (policy: never foreclose jvp).
    loss = _stretched_diff_loss("center", periodic=False)
    primal, tangent = jax.jvp(loss, (1.3,), (1.0,))
    assert bool(jnp.isfinite(primal))
    assert bool(jnp.isfinite(tangent))


def test_divide_by_codomain_measure_is_bitwise_on_true_cells():
    # the seal is bitwise-transparent on every valid cell: the guarded
    # quotient equals the raw ``result / measure`` on the true region.
    mesh = MappedIntervalMesh(8, (0.0, 1.0), _wavy, periodic=False,
                              name="v")
    grid = Grid((mesh,))
    result = grid.create_field(
        mesh.cell_avg, init=lambda v: jnp.cos(2.0 * jnp.pi * v) + 2.0)
    out = divide_by_codomain_measure(result, result, "v")
    query = result.function_space.with_layout(
        result.function_space.layout)
    measure = grid.sync(grid.measure(query, name="v"))
    assert jnp.array_equal(out.data, result.data / measure.data)


def test_divide_by_codomain_measure_bitwise_whole_array_periodic():
    # on a periodic mapped axis the measure's wrap fill is strictly
    # positive, so ``bad`` is empty and the WHOLE storage array (ghost
    # slots included) is bitwise identical to the raw quotient.
    mesh = MappedIntervalMesh(8, (0.0, 1.0), _wavy, periodic=True,
                              name="v")
    grid = Grid((mesh,))
    result = grid.create_field(
        mesh.cell_avg, init=lambda v: jnp.sin(2.0 * jnp.pi * v) + 3.0)
    out = divide_by_codomain_measure(result, result, "v")
    query = result.function_space.with_layout(
        result.function_space.layout)
    measure = grid.sync(grid.measure(query, name="v"))
    assert jnp.array_equal(out._data, result._data / measure._data)


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


# ================================================================
#  Per-shard footprint reach vs boundary exterior reach: the
#  fix/walled-shard-halo-validity regression
# ================================================================
def test_footprint_reach_is_the_pure_stencil_span():
    # a 2-point stencil aligned at m0=0 reads its own slot and the one
    # above -> footprint (0, 1); m0=1 reads its own slot and the one
    # below -> (1, 0). No global codomain/domain length term enters.
    assert footprint_reach(2, 0) == (0, 1)
    assert footprint_reach(2, 1) == (1, 0)
    # a 4-point kernel aligned at m0=1 spans one below, two above
    assert footprint_reach(4, 1) == (1, 2)


def test_footprint_and_exterior_reach_differ_on_a_shrinking_wall(my):
    # THE regression pin (fix/walled-shard-halo-validity): a staggered
    # first difference Center(bounded) -> Inner shrinks the codomain
    # (n_out = n_in - 1), so its boundary EXTERIOR reach cancels to
    # (0, 0) -- no exterior/wall slot is read, which is why the diff is
    # legal on a BC-free bounded side. But the PER-SHARD stencil
    # footprint is (0, 1): every interior shard boundary still reads a
    # neighbor slot. The requirements must publish the footprint, or a
    # sharded walled operand's inter-shard halo is never synced before
    # the flux difference (silent wrong physics). The two are distinct.
    op = FiniteDifference(order=2)["y"]
    center = my.center
    codomain = op.codomain(center)
    assert exterior_reach(center, codomain, 2) == (0, 0)
    assert footprint_reach(2, 0) == (0, 1)
    # the operator publishes the footprint (a would-be-skipped sync now
    # fires), not the cancelled exterior reach
    assert op.requirements(center).reach == (0, 1)
    assert op.requirements(center).reach != (0, 0)


def test_periodic_reach_is_unchanged_by_the_footprint_fix(mx):
    # on a periodic axis n_out == n_in, so the footprint and the
    # boundary exterior reach coincide -- the fix leaves the periodic
    # path (and its two-sided storage optimization) bit-for-bit intact
    op = FiniteDifference(order=2)["x"]
    center = mx.center
    ext = exterior_reach(center, op.codomain(center), 2)
    assert ext != (0, 0)
    assert op.requirements(center).reach == ext
