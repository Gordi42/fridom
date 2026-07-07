"""Tests for fridom.framework2.grid.grid."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.errors import GridMismatchError
from fridom.framework2.grid.fields.metadata import FieldMetadata
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.chebyshev import ChebyshevMesh
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.meshes.point import PointMesh
from fridom.framework2.grid.operators.base import (
    OperatorRequirements,
)
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.operators.registry import (
    DispatchError,
    OperatorRegistry,
)
from fridom.framework2.grid.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(4, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


@pytest.fixture
def grid1d(mx):
    return Grid((mx,))


# ================================================================
#  Construction and identity
# ================================================================
def test_structure(grid, mx, my):
    assert grid.factors == (mx, my)
    assert grid.names == ("x", "y")


def test_empty_meshes_rejected():
    with pytest.raises(ValueError, match="at least one mesh"):
        Grid(())


def test_duplicate_names_rejected(mx):
    other = IntervalMesh(4, (0.0, 2.0), name="x")
    with pytest.raises(ValueError, match="duplicate coordinate"):
        Grid((mx, other))


def test_identity_equality_and_hash(mx, my):
    a = Grid((mx, my))
    b = Grid((mx, my))
    assert a == a  # noqa: PLR0124 — identity semantics under test
    assert a != b
    assert a != "grid"
    assert hash(a) == id(a)


def test_dispatch_defaults_to_seeded_registry_and_is_settable(mx):
    registry = Grid((mx,)).dispatch
    assert isinstance(registry, OperatorRegistry)
    assert isinstance(registry.resolve("diff", mx.center),
                      FiniteDifference)
    assert isinstance(registry.resolve("interpolate", mx.center),
                      LinearInterp)
    custom = object()
    assert Grid((mx,), dispatch=custom).dispatch is custom


def test_decomposition_is_single_device_provisional_halo(
        grid, mx, my):
    # provisional negotiation: halo = per-operator max over the
    # seeded registry; the widest entry is the two-factor
    # FV-derivative chain (reconstruct + flux_diff, width 1 each)
    dec = grid.decomposition
    assert dec.halo["x"] == 2
    assert dec.halo["y"] == 2
    space = mx.center * my.center
    assert dec.storage_shape(space) == (8 + 4, 4 + 4)
    assert dec.default_layout.device_axes == ()


def test_duck_registry_without_items_gets_zero_halo(mx):
    grid = Grid((mx,), dispatch=object())
    assert grid.decomposition.halo["x"] == 0


def test_seeded_registry_covers_the_default_rows(grid, mx, my):
    registry = grid.dispatch
    fd = registry.resolve("diff", mx.center)
    assert fd is registry.resolve("diff", mx.right)
    assert fd is registry.resolve("diff", my.outer)
    assert fd is registry.resolve("diff", my.inner)
    multiply = registry.resolve("multiply", mx.center)
    assert multiply is registry.resolve("multiply", mx.cell_avg)
    assert multiply is registry.resolve(
        "multiply", mx.center.as_complex())
    registry.resolve("divide", my.face_avg)
    registry.resolve("power", mx.cell_avg)
    registry.resolve("abs", mx.center)
    with pytest.raises(DispatchError, match="abs"):
        registry.resolve("abs", mx.cell_avg)  # nodal-only by design
    with pytest.raises(DispatchError, match="diff"):
        registry.resolve("diff", mx.fourier(origin=mx.center))
    with pytest.raises(DispatchError, match="diff"):
        registry.resolve("diff", my.right)  # no bounded Right row


# ================================================================
#  Lifecycle stubs (Wave 3)
# ================================================================
def test_lifecycle_stubs_raise(grid):
    with pytest.raises(NotImplementedError, match="registry"):
        grid.merge_overrides({})
    with pytest.raises(NotImplementedError, match="Wave 3"):
        grid.negotiate()
    with pytest.raises(NotImplementedError, match="Wave 3"):
        grid.freeze()


def test_sync_is_identity_on_one_device(grid):
    f = grid.create_field(data=jnp.arange(32.0).reshape(8, 4),
                          name="f")
    synced = grid.sync(f)
    assert jnp.array_equal(synced.data, f.data)
    assert synced.function_space is f.function_space
    assert synced.metadata == f.metadata


def test_sync_boundary_data_not_implemented(grid):
    f = grid.create_field()
    with pytest.raises(NotImplementedError, match="ghost fill"):
        grid.sync(f, boundary_data={})


# ================================================================
#  create_field — spaces and layout
# ================================================================
def test_default_space_is_all_center_laid_out(grid, mx, my):
    f = grid.create_field()
    assert f.function_space.bare is mx.center * my.center
    assert f.function_space.layout is grid.decomposition.default_layout
    assert jnp.array_equal(f.data, jnp.zeros((8, 4)))


def test_bare_space_gets_default_layout(grid, mx, my):
    f = grid.create_field(mx.right * my.center)
    assert f.function_space.layout is grid.decomposition.default_layout


def test_laid_out_space_is_honored(grid, mx, my):
    space = (mx.center * my.center).with_layout(
        grid.decomposition.default_layout)
    f = grid.create_field(space)
    assert f.function_space is space


def test_lone_factor_space(grid1d, mx):
    f = grid1d.create_field(mx.center)
    assert f.function_space.bare is mx.center
    assert f.shape == (8,)


def test_foreign_mesh_space_raises(grid):
    foreign = IntervalMesh(8, (0.0, 1.0), name="z")
    with pytest.raises(GridMismatchError, match="not a factor"):
        grid.create_field(foreign.center)


def test_refined_mesh_spaces_are_adopted(grid1d, mx):
    fine = mx.refined(2)
    f = grid1d.create_field(fine.center)
    assert f.shape == (16,)


def test_default_space_needs_center_family():
    points = PointMesh(((0.0,), (1.0,)), name="p")
    grid = Grid((points,))
    with pytest.raises(TypeError, match="no Center space"):
        grid.create_field()


# ================================================================
#  create_field — argument validation and metadata sugar
# ================================================================
def test_init_and_data_are_exclusive(grid):
    with pytest.raises(ValueError, match="mutually exclusive"):
        grid.create_field(init=lambda x, y: x + y,
                          data=jnp.zeros((8, 4)))


def test_metadata_and_sugar_are_exclusive(grid):
    with pytest.raises(ValueError, match="mutually exclusive"):
        grid.create_field(metadata=FieldMetadata(), name="f")


def test_metadata_sugar(grid):
    f = grid.create_field(name="u", units="m/s")
    assert f.metadata == FieldMetadata.create(name="u", units="m/s")
    g = grid.create_field(units="K")
    assert g.metadata.units == "K"
    assert g.metadata.name == "unnamed"


def test_full_metadata_record(grid):
    md = FieldMetadata.create(name="b", nc_attrs={"axis": "Z"})
    assert grid.create_field(metadata=md).metadata == md


# ================================================================
#  create_field — data path
# ================================================================
def test_data_shape_validated(grid):
    with pytest.raises(ValueError, match="true shape"):
        grid.create_field(data=jnp.zeros((4, 8)))


def test_data_dtype_coerced_to_space_dtype(grid):
    f = grid.create_field(data=jnp.zeros((8, 4), dtype=jnp.int32))
    assert f.dtype == jnp.float64


def test_complex_data_into_real_space_raises(grid):
    with pytest.raises(ValueError, match="demoted"):
        grid.create_field(data=jnp.zeros((8, 4), dtype=complex))


def test_complex_data_into_complex_space(grid, mx, my):
    space = (mx.center * my.center).as_complex()
    f = grid.create_field(space, data=jnp.full((8, 4), 1.0 + 1.0j))
    assert f.dtype == jnp.complex128


def test_data_into_fourier_space_projects_hermitian(grid1d, mx):
    space = mx.fourier(origin=mx.center)
    data = jnp.full(space.shape, 1.0 + 1.0j)
    f = grid1d.create_field(space, data=data)
    assert f.dtype == jnp.complex128
    assert f.data[0] == 1.0 + 0.0j
    assert f.data[-1] == 1.0 + 0.0j  # Nyquist (n = 8 even)
    assert jnp.array_equal(f.data[1:-1], data[1:-1])


# ================================================================
#  create_field — init path (collocation discretization)
# ================================================================
def test_init_matches_analytic_function(grid):
    f = grid.create_field(init=lambda x, y: x**2 + 3.0 * y)
    x = (jnp.arange(8) + 0.5) * 0.125
    y = (jnp.arange(4) + 0.5) * 0.5
    expected = x[:, None] ** 2 + 3.0 * y[None, :]
    assert jnp.allclose(f.data, expected)


def test_init_matches_by_keyword_not_order(grid):
    f = grid.create_field(init=lambda y, x: x + 10.0 * y)
    x = (jnp.arange(8) + 0.5) * 0.125
    y = (jnp.arange(4) + 0.5) * 0.5
    assert jnp.allclose(f.data, x[:, None] + 10.0 * y[None, :])


def test_init_on_staggered_space_samples_staggered_nodes(
        grid, mx, my):
    f = grid.create_field(mx.right * my.center,
                          init=lambda x, y: x + 0.0 * y)
    x = (jnp.arange(8) + 1.0) * 0.125
    assert jnp.allclose(f.data, jnp.broadcast_to(x[:, None], (8, 4)))


def test_init_with_constant_factor_omits_its_name(grid, mx, my):
    f = grid.create_field(mx.constant * my.center,
                          init=lambda y: 2.0 * y)
    y = (jnp.arange(4) + 0.5) * 0.5
    assert jnp.allclose(f.data, (2.0 * y)[None, :])
    assert f.shape == (1, 4)


def test_init_signature_validated(grid):
    with pytest.raises(TypeError, match="exactly"):
        grid.create_field(init=lambda x: x)
    with pytest.raises(TypeError, match="exactly"):
        grid.create_field(init=lambda x, y, z: x + y + z)


def test_init_on_coefficient_space_not_wired_yet(grid1d, mx):
    with pytest.raises(NotImplementedError, match="transform"):
        grid1d.create_field(mx.fourier(origin=mx.center),
                            init=lambda x: x)


# ================================================================
#  evaluation_nodes
# ================================================================
def test_nodes_center(grid1d, mx):
    nodes = grid1d.evaluation_nodes(mx.center)
    assert nodes.function_space.bare is mx.center
    assert jnp.allclose(nodes.data,
                        (jnp.arange(8) + 0.5) * 0.125)
    assert nodes.name == "x"


@pytest.mark.parametrize(("attr", "expected"), [
    pytest.param("left", jnp.arange(8) * 0.125, id="left"),
    pytest.param("right", (jnp.arange(8) + 1) * 0.125, id="right"),
    pytest.param("cell_avg", (jnp.arange(8) + 0.5) * 0.125,
                 id="cell_avg-midpoints"),
    pytest.param("face_avg", (jnp.arange(8) + 1) * 0.125,
                 id="face_avg-periodic-faces"),
])
def test_nodes_periodic_families(grid1d, mx, attr, expected):
    nodes = grid1d.evaluation_nodes(getattr(mx, attr))
    assert jnp.allclose(nodes.data, expected)


@pytest.mark.parametrize(("attr", "expected"), [
    pytest.param("outer", jnp.arange(5) * 0.5, id="outer"),
    pytest.param("inner", (jnp.arange(3) + 1) * 0.5, id="inner"),
    pytest.param("face_avg", (jnp.arange(3) + 1) * 0.5,
                 id="face_avg-bounded-inner-faces"),
])
def test_nodes_bounded_families(my, attr, expected):
    grid = Grid((my,))
    nodes = grid.evaluation_nodes(getattr(my, attr))
    assert jnp.allclose(nodes.data, expected)


def test_nodes_drop_bc_constrained_boundary_dofs(my):
    grid = Grid((my,))
    space = my.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    nodes = grid.evaluation_nodes(space)
    assert space.shape == (3,)
    assert jnp.allclose(nodes.data, (jnp.arange(3) + 1) * 0.5)


def test_nodes_on_product_replace_other_factors_by_constant(
        grid, mx, my):
    space = mx.center * my.center
    x = grid.evaluation_nodes(space, "x")
    assert x.shape == (8, 1)
    assert x.function_space.factor("y") is my.constant
    assert x.function_space.factor("x") is mx.center
    y = grid.evaluation_nodes(space, "y")
    assert y.shape == (1, 4)
    assert y.function_space.factor("x") is mx.constant


def test_nodes_broadcast_under_the_strict_algebra(grid):
    f = grid.create_field(init=lambda x, y: x + 0.0 * y)
    x = grid.evaluation_nodes(f.function_space, "x")
    h = f - x
    assert jnp.allclose(h.data, jnp.zeros((8, 4)))


def test_nodes_name_required_when_ambiguous(grid, mx, my):
    with pytest.raises(ValueError, match="ambiguous"):
        grid.evaluation_nodes(mx.center * my.center)


def test_nodes_unambiguous_with_constant_factor(grid, mx, my):
    nodes = grid.evaluation_nodes(mx.constant * my.center)
    assert nodes.name == "y"
    assert nodes.shape == (1, 4)


def test_nodes_unknown_name_raises(grid, mx, my):
    with pytest.raises(KeyError, match="no factor"):
        grid.evaluation_nodes(mx.center * my.center, "z")


def test_nodes_constant_factor_has_none(grid1d, mx):
    with pytest.raises(ValueError, match="constant"):
        grid1d.evaluation_nodes(mx.constant, "x")


def test_nodes_coefficient_factor_raises(grid1d, mx):
    with pytest.raises(ValueError, match="wavenumbers"):
        grid1d.evaluation_nodes(mx.fourier(origin=mx.center))


def test_nodes_on_non_interval_mesh_not_implemented():
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="s")
    grid = Grid((cheb,))
    with pytest.raises(NotImplementedError, match="IntervalMesh"):
        grid.evaluation_nodes(cheb.outer)


def test_wavenumbers_not_wired_yet(grid1d, mx):
    with pytest.raises(NotImplementedError, match="transform"):
        grid1d.wavenumbers(mx.fourier(origin=mx.center))


def test_registry_halo_derivation_skips_unusable_entries(mx):
    class ForeignSpace:
        names = ("z",)

    class WideOp:
        def requirements(self, space):  # noqa: ARG002
            return OperatorRequirements(halo=3)

    class DuckRegistry:
        def items(self):
            yield "kind_only", object()          # no space to size on
            yield ("diff", ForeignSpace()), WideOp()  # foreign name
            yield ("multiply", mx.center), object()   # no requirements

    grid = Grid((mx,), dispatch=DuckRegistry())
    assert grid.decomposition.halo["x"] == 0
