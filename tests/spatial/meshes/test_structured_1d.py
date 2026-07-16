"""Tests for StructuredMesh1D (spatial/meshes/structured_1d.py)."""
import pytest

from fridom.spatial.bc import BC, BCStructure
from fridom.spatial.decomposition.traits import HaloStrategy
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import (
    AverageSpace,
    CellAvg,
    FaceAvg,
)
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.nodal import (
    Center,
    Inner,
    Left,
    NodeSet,
    Outer,
    Right,
)

N = 8


@pytest.fixture
def periodic():
    return IntervalMesh(N, (0, 1), name="x")


@pytest.fixture
def bounded():
    return IntervalMesh(N, (0, 1), periodic=False, name="x")


# ================================================================
#  Construction validation
# ================================================================
def test_shape_must_be_positive_int():
    with pytest.raises(TypeError, match="integer cell count"):
        IntervalMesh(8.0, (0, 1), name="x")
    with pytest.raises(TypeError, match="integer cell count"):
        IntervalMesh(True, (0, 1), name="x")  # noqa: FBT003
    with pytest.raises(ValueError, match="positive cell count"):
        IntervalMesh(0, (0, 1), name="x")


def test_extent_must_increase():
    with pytest.raises(ValueError, match="increasing"):
        IntervalMesh(8, (1, 0), name="x")
    with pytest.raises(ValueError, match="increasing"):
        IntervalMesh(8, (0, 0, 1), name="x")


def test_geometry_descriptors(periodic):
    assert periodic.n_cells == N
    assert periodic.extent == (0, 1)
    assert periodic.periodic is True
    assert periodic.dim == 1


# ================================================================
#  Nodal factories
# ================================================================
def test_sugar_properties_are_the_nodal_factory(bounded):
    assert bounded.center is bounded.nodal(NodeSet.CENTER)
    assert bounded.left is bounded.nodal(NodeSet.LEFT)
    assert bounded.right is bounded.nodal(NodeSet.RIGHT)
    assert bounded.outer is bounded.nodal(NodeSet.OUTER)
    assert bounded.inner is bounded.nodal(NodeSet.INNER)


def test_nodal_classes(bounded):
    assert type(bounded.center) is Center
    assert type(bounded.left) is Left
    assert type(bounded.right) is Right
    assert type(bounded.outer) is Outer
    assert type(bounded.inner) is Inner


def test_outer_inner_raise_on_periodic(periodic):
    with pytest.raises(ValueError, match="bounded"):
        _ = periodic.outer
    with pytest.raises(ValueError, match="bounded"):
        _ = periodic.inner


def test_left_right_exist_on_both_topologies(periodic, bounded):
    assert periodic.left.shape == (N,)
    assert periodic.right.shape == (N,)
    assert bounded.left.shape == (N,)
    assert bounded.right.shape == (N,)


def test_nodal_rejects_points(bounded):
    with pytest.raises(ValueError, match="PointMesh"):
        bounded.nodal(NodeSet.POINTS)


def test_nodal_rejects_non_node_set(bounded):
    with pytest.raises(TypeError, match="NodeSet"):
        bounded.nodal("center")


def test_value_equal_factory_calls_return_the_same_object(bounded):
    # interning: the strict-algebra equality check is `is`
    spellings = [
        bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET),
        bounded.nodal(NodeSet.CENTER, bc=(BC.DIRICHLET, BC.DIRICHLET)),
        bounded.nodal(NodeSet.CENTER,
                      bc=BCStructure((BC.DIRICHLET, BC.DIRICHLET))),
    ]
    assert spellings[0] is spellings[1]
    assert spellings[1] is spellings[2]


def test_bc_variants_are_distinct_spaces(bounded):
    free = bounded.center
    dirichlet = bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    assert free is not dirichlet
    assert free != dirichlet


def test_bc_wrong_length_rejected(bounded):
    with pytest.raises(ValueError, match="2 boundary components"):
        bounded.nodal(NodeSet.CENTER, bc=(BC.DIRICHLET,))


def test_bc_on_periodic_mesh_rejected(periodic):
    with pytest.raises(ValueError, match="no boundary"):
        periodic.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)


# ================================================================
#  Average factories
# ================================================================
def test_average_factories(periodic, bounded):
    assert type(periodic.cell_avg) is CellAvg
    assert type(bounded.face_avg) is FaceAvg
    assert isinstance(periodic.cell_avg, AverageSpace)
    assert periodic.cell_avg is periodic.cell_avg
    assert bounded.face_avg is bounded.face_avg


def test_averages_are_not_nodal(periodic):
    assert periodic.cell_avg is not periodic.center
    assert periodic.cell_avg != periodic.center


def test_average_general_factory_and_sugar(periodic, bounded):
    # the general average(kind, bc=...) factory; the BC-free properties
    # are sugar for it (interned identity)
    assert periodic.cell_avg is periodic.average(CellAvg)
    assert bounded.face_avg is bounded.average(FaceAvg)


def test_average_bc_tagged_cell_avg(bounded):
    # a Neumann-tagged CellAvg keeps shape (n,) (averages have no
    # boundary DOF) and interns per BC structure (F4)
    neu = bounded.average(CellAvg, bc=BC.NEUMANN)
    assert type(neu) is CellAvg
    assert neu.shape == bounded.cell_avg.shape
    assert neu is not bounded.cell_avg
    assert all(c is BC.NEUMANN for c in neu.bc.components)
    assert neu is bounded.average(CellAvg, bc=BC.NEUMANN)  # interned


def test_average_rejects_non_average_kind(bounded):
    with pytest.raises(TypeError, match="AverageSpace subclass"):
        bounded.average(object)


def test_face_avg_is_untaggable(bounded):
    # FV-D2: the dual-cell average family is a dead-end -- a non-NONE
    # bc on FaceAvg is a taught error
    with pytest.raises(ValueError, match="FaceAvg is untaggable"):
        bounded.average(FaceAvg, bc=BC.NEUMANN)


def test_average_periodic_rejects_bc(periodic):
    with pytest.raises(ValueError, match="no boundary to constrain"):
        periodic.average(CellAvg, bc=BC.NEUMANN)


# ================================================================
#  Coefficient factories
# ================================================================
def test_fourier_needs_a_periodic_mesh(bounded):
    with pytest.raises(ValueError, match="periodic"):
        bounded.fourier(origin=bounded.center)


def test_sine_cosine_need_a_bounded_mesh(periodic):
    with pytest.raises(ValueError, match="bounded"):
        periodic.sine(origin=periodic.center)
    with pytest.raises(ValueError, match="bounded"):
        periodic.cosine(origin=periodic.center)


def test_fourier_is_interned_per_origin(periodic):
    assert (periodic.fourier(origin=periodic.center)
            is periodic.fourier(origin=periodic.center))
    assert (periodic.fourier(origin=periodic.center)
            is not periodic.fourier(origin=periodic.right))
    assert type(periodic.fourier(origin=periodic.center)) is FourierSpace


def test_origin_must_live_on_the_mesh(periodic):
    other = IntervalMesh(N, (0, 1), name="x")
    with pytest.raises(ValueError, match="lives on"):
        periodic.fourier(origin=other.center)


def test_origin_must_be_nodal_or_average(periodic):
    with pytest.raises(TypeError, match="nodal or average"):
        periodic.fourier(origin=periodic.constant)
    with pytest.raises(TypeError, match="nodal or average"):
        periodic.fourier(origin=periodic.fourier(origin=periodic.center))


def test_laid_out_origin_rejected(periodic):
    laid_out = periodic.center.with_layout("L0")
    with pytest.raises(ValueError, match="bare"):
        periodic.fourier(origin=laid_out)


def test_origin_is_a_required_keyword(periodic):
    with pytest.raises(TypeError):
        periodic.fourier()


def test_sine_requires_dirichlet_origin(bounded):
    with pytest.raises(ValueError, match="Dirichlet"):
        bounded.sine(origin=bounded.center)
    origin = bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    assert type(bounded.sine(origin=origin)) is SineSpace
    assert bounded.sine(origin=origin) is bounded.sine(origin=origin)


def test_cosine_requires_neumann_origin(bounded):
    with pytest.raises(ValueError, match="Neumann"):
        bounded.cosine(origin=bounded.center)
    with pytest.raises(ValueError, match="Neumann"):
        bounded.cosine(
            origin=bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))
    origin = bounded.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    assert type(bounded.cosine(origin=origin)) is CosineSpace


def test_fourier_of_average_origin_is_distinct(periodic):
    # the sinc(k dx / 2) factor: origin=CellAvg keys a new space
    of_center = periodic.fourier(origin=periodic.center)
    of_cell_avg = periodic.fourier(origin=periodic.cell_avg)
    assert of_center is not of_cell_avg


def test_galerkin_is_designed_for(bounded):
    with pytest.raises(NotImplementedError, match="designed-for"):
        bounded.galerkin(bc=BC.DIRICHLET)


# ================================================================
#  Decomposition traits
# ================================================================
def test_traits_nodal_and_average_are_ghost_first(periodic):
    for space in (periodic.center, periodic.cell_avg):
        traits = periodic.decomposition_traits(space)
        assert traits.strategies == (
            HaloStrategy.GHOST, HaloStrategy.TRANSPOSE)
        assert traits.min_local_size == 1


def test_traits_coefficient_is_local_first(periodic):
    traits = periodic.decomposition_traits(
        periodic.fourier(origin=periodic.center))
    assert traits.strategies == (
        HaloStrategy.LOCAL, HaloStrategy.TRANSPOSE)


def test_traits_constant_is_local_only(periodic):
    traits = periodic.decomposition_traits(periodic.constant)
    assert traits.strategies == (HaloStrategy.LOCAL,)


def test_traits_reject_foreign_spaces(periodic, bounded):
    with pytest.raises(ValueError, match="lives on"):
        periodic.decomposition_traits(bounded.center)


# ================================================================
#  Misc
# ================================================================
def test_scalars_default_real(bounded):
    assert bounded.center.scalars is Scalars.REAL
    assert bounded.cell_avg.scalars is Scalars.REAL


def test_repr(periodic, bounded):
    assert repr(periodic) == (
        "IntervalMesh(x: n=8, extent=(0, 1), periodic)")
    assert repr(bounded) == (
        "IntervalMesh(x: n=8, extent=(0, 1), bounded)")
