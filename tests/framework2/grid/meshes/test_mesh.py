"""Tests for the Mesh ABC (framework2/grid/meshes/mesh.py)."""
import pytest

from fridom.framework2.grid.decomposition.traits import (
    HaloStrategy,
    MeshDecompositionTraits,
)
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.meshes.mesh import Mesh
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.constant import ConstantSpace


class TwoNameMesh(Mesh):

    """Minimal concrete mesh to exercise the base class directly."""

    @property
    def dim(self):
        return 2

    @property
    def boundary(self):
        raise NotImplementedError

    @property
    def _n_boundary_components(self):
        return 0

    def decomposition_traits(self, space):
        self._check_owned(space)
        return MeshDecompositionTraits((HaloStrategy.LOCAL,))


# ================================================================
#  Identity semantics
# ================================================================
def test_meshes_are_not_interned_by_value():
    # a square domain needs two distinct factors
    a = IntervalMesh(8, (0, 1), name="x")
    b = IntervalMesh(8, (0, 1), name="x")
    assert a is not b
    assert a != b
    assert hash(a) != hash(b)


def test_identity_eq_is_explicit():
    mesh = IntervalMesh(8, (0, 1), name="x")
    assert mesh == mesh  # noqa: PLR0124 — identity semantics
    assert (mesh == "x") is False
    assert mesh != IntervalMesh(8, (0, 1), name="x")


# ================================================================
#  Coordinate names
# ================================================================
def test_names_fixed_at_construction():
    mesh = IntervalMesh(8, (0, 1), name="x")
    assert mesh.names == ("x",)
    assert TwoNameMesh(("lon", "lat")).names == ("lon", "lat")


def test_empty_names_rejected():
    with pytest.raises(ValueError, match="at least one"):
        TwoNameMesh(())


def test_non_string_names_rejected():
    with pytest.raises(TypeError, match="non-empty strings"):
        TwoNameMesh(("x", 3))
    with pytest.raises(TypeError, match="non-empty strings"):
        IntervalMesh(8, (0, 1), name="")


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        TwoNameMesh(("x", "x"))


# ================================================================
#  The universal constant factory
# ================================================================
def test_constant_is_interned():
    mesh = IntervalMesh(8, (0, 1), name="x")
    assert mesh.constant is mesh.constant
    assert isinstance(mesh.constant, ConstantSpace)
    assert mesh.constant.mesh is mesh


def test_constant_is_per_mesh():
    a = IntervalMesh(8, (0, 1), name="x")
    b = IntervalMesh(8, (0, 1), name="x")
    assert a.constant is not b.constant


def test_constant_is_universal():
    # every mesh type has it, even the 0D boundary mesh
    mesh = IntervalMesh(8, (0, 1), periodic=False, name="x")
    assert mesh.boundary.constant is mesh.boundary.constant


def test_constant_scalar_variants_are_interned():
    mesh = IntervalMesh(8, (0, 1), name="x")
    complex_constant = mesh.constant.as_complex()
    assert complex_constant is not mesh.constant
    assert complex_constant is mesh.constant.as_complex()
    assert complex_constant.scalars is Scalars.COMPLEX
    assert complex_constant.as_real() is mesh.constant


# ================================================================
#  Traits seam and repr
# ================================================================
def test_decomposition_traits_rejects_foreign_spaces():
    a = TwoNameMesh(("lon", "lat"))
    b = IntervalMesh(8, (0, 1), name="x")
    with pytest.raises(ValueError, match="lives on"):
        a.decomposition_traits(b.constant)


def test_generic_repr():
    assert repr(TwoNameMesh(("lon", "lat"))) == "TwoNameMesh(lon, lat)"
