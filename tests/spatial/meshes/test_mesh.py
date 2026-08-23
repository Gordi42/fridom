"""Tests for the Mesh ABC (spatial/meshes/mesh.py)."""
import pytest

from fridom.spatial.decomposition.traits import (
    HaloStrategy,
    MeshDecompositionTraits,
)
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mesh import Mesh
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.constant import ConstantSpace


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


# ================================================================
#  Flat-axis predicate (halo elision)
# ================================================================
def test_base_mesh_is_never_flat():
    # a mesh with no periodic 1D topology carries no flat axis: the
    # base answer is False, and StructuredMesh1D overrides it
    assert TwoNameMesh(("lon", "lat")).is_flat is False
    assert IntervalMesh(8, (0, 1), name="x").boundary.is_flat is False


# ================================================================
#  Taught attribute misses (.centers is the wrong word, not a
#  missing alias)
# ================================================================
def test_plural_node_set_names_the_singular_and_says_why():
    mesh = IntervalMesh(8, (0, 1), name="x")
    with pytest.raises(AttributeError) as excinfo:
        mesh.centers  # noqa: B018 — the miss IS the assertion
    message = str(excinfo.value)
    assert "'IntervalMesh' object has no attribute 'centers'" in message
    assert "did you mean 'center'?" in message
    # and the reason a .centers alias is refused rather than added
    assert "one function space" in message
    assert "grid.evaluation_nodes(space, name)" in message
    # the singular really is a space object, not coordinate values
    assert mesh.center is mesh.center


@pytest.mark.parametrize(("plural", "singular"), [
    pytest.param("lefts", "left", id="left"),
    pytest.param("rights", "right", id="right"),
    pytest.param("outers", "outer", id="outer"),
    pytest.param("inners", "inner", id="inner"),
    pytest.param("constants", "constant", id="constant"),
])
def test_the_whole_node_set_family_is_covered(plural, singular):
    mesh = IntervalMesh(8, (0, 1), periodic=False, name="x")
    with pytest.raises(AttributeError, match=f"did you mean '{singular}'"):
        getattr(mesh, plural)


def test_a_plural_outside_the_node_sets_hints_without_the_paragraph():
    with pytest.raises(AttributeError) as excinfo:
        TwoNameMesh(("lon", "lat")).dims  # noqa: B018 — the miss IS the assertion
    message = str(excinfo.value)
    assert "did you mean 'dim'?" in message
    assert "node-set factories" not in message


def test_a_singular_miss_hints_at_the_plural():
    # the mirror image: IntervalMesh takes name= at construction, so
    # mesh.name is the natural (wrong) reach for mesh.names
    with pytest.raises(AttributeError, match="did you mean 'names'"):
        IntervalMesh(8, (0, 1), name="x").name  # noqa: B018 — the miss IS the assertion


def test_an_unrelated_miss_stays_a_bare_attribute_error():
    mesh = IntervalMesh(8, (0, 1), name="x")
    with pytest.raises(AttributeError) as excinfo:
        mesh.bogus  # noqa: B018 — the miss IS the assertion
    assert str(excinfo.value) == (
        "'IntervalMesh' object has no attribute 'bogus'")


def test_private_and_dunder_probes_get_the_bare_miss():
    # copy/pickle/jax probe dunders by getattr; a hinted message
    # there would be noise, and hasattr must stay cheap and False
    mesh = IntervalMesh(8, (0, 1), name="x")
    with pytest.raises(AttributeError) as excinfo:
        mesh.__deepcopy__  # noqa: B018 — the miss IS the assertion
    assert "did you mean" not in str(excinfo.value)
    assert not hasattr(mesh, "_no_such_slot")


def test_the_getattr_default_idiom_still_works():
    # src probes meshes with getattr(mesh, "periodic", True); the
    # taught miss must not turn that into a raise
    mesh = TwoNameMesh(("lon", "lat"))
    assert getattr(mesh, "periodic", True) is True
    assert getattr(mesh, "dims", None) is None
