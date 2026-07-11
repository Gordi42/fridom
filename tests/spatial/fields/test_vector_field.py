"""Tests for fridom.spatial.fields.vector_field."""
import jax
import jax.numpy as jnp
import pytest

from fridom.spatial.errors import (
    GridMismatchError,
    MissingComponentError,
    SpaceMismatchError,
)
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(4, (0.0, 1.0), name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


@pytest.fixture
def u(grid, mx, my):
    return grid.create_field(mx.right * my.center, name="u",
                             units="m/s")


@pytest.fixture
def v(grid, mx, my):
    return grid.create_field(mx.center * my.right, name="v",
                             units="m/s")


@pytest.fixture
def vec(u, v):
    return VectorField([u, v])


# ================================================================
#  Construction
# ================================================================
def test_iterable_input_takes_metadata_names(vec):
    assert vec.component_names == ("u", "v")


def test_mapping_input_takes_the_keys(u, v):
    vec = VectorField({"zonal": u, "meridional": v})
    assert vec.component_names == ("zonal", "meridional")
    assert vec["zonal"] is u


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        VectorField([])


def test_duplicate_names_are_rejected(u):
    with pytest.raises(ValueError, match="duplicate"):
        VectorField([u, u.with_metadata(units="1")])


def test_duplicate_unnamed_gets_the_naming_hint(grid):
    a = grid.create_field()
    b = grid.create_field()
    with pytest.raises(ValueError, match="name the fields"):
        VectorField([a, b])


def test_differing_grids_are_rejected(mx, my, u):
    other = Grid((mx, my))
    w = other.create_field(name="w")
    with pytest.raises(GridMismatchError, match="identical grid"):
        VectorField([u, w])


def test_component_spaces_are_unconstrained(vec, mx, my):
    # the whole point of 2.4: staggered components coexist
    assert vec["u"].function_space.bare is mx.right * my.center
    assert vec["v"].function_space.bare is mx.center * my.right


# ================================================================
#  Access surface
# ================================================================
def test_access(vec, u, v, grid):
    assert vec.grid is grid
    assert len(vec) == 2
    assert "u" in vec
    assert "w" not in vec
    assert vec[0] is u
    assert vec["v"] is v
    assert tuple(vec) == (u, v)
    assert dict(vec.components) == {"u": u, "v": v}
    with pytest.raises(KeyError, match="no component"):
        vec["w"]
    with pytest.raises(TypeError, match="name or position"):
        vec[1.5]


def test_getitem_missing_raises_missing_component(vec):
    with pytest.raises(MissingComponentError, match="no component"):
        vec["w"]
    # subclasses KeyError: existing handlers keep working
    with pytest.raises(KeyError):
        vec["w"]


def test_require_returns_present_component(vec, u):
    assert vec.require("u", hint="pass the zonal velocity") is u


def test_require_missing_raises_with_hint(vec):
    with pytest.raises(MissingComponentError) as exc:
        vec.require("w", hint="pass the vertical velocity w")
    msg = str(exc.value)
    assert "w" in msg
    assert "pass the vertical velocity w" in msg
    assert "u" in msg
    assert "v" in msg


def test_select_subsets_and_reorders(vec, u, v):
    sub = vec.select("v", "u")
    assert sub.component_names == ("v", "u")
    assert sub["u"] is u
    assert sub["v"] is v
    single = vec.select("u")
    assert single.component_names == ("u",)
    assert single["u"] is u


def test_select_rejects_unknown_names(vec):
    with pytest.raises(MissingComponentError,
                       match="no components named"):
        vec.select("u", "w")


def test_0d_array_acts_as_scalar(vec):
    s = jnp.asarray(2.0)  # traced ctx.params leaf shape
    assert jnp.allclose((s * vec)["u"].data, 2.0 * vec["u"].data)
    assert jnp.allclose((vec * s)["v"].data, vec["v"].data * 2.0)
    assert jnp.allclose((vec + s)["u"].data, vec["u"].data + 2.0)
    assert jnp.allclose((s - vec)["u"].data, 2.0 - vec["u"].data)
    assert jnp.allclose((vec / s)["u"].data, vec["u"].data / 2.0)
    assert jnp.allclose((vec ** jnp.asarray(2.0))["u"].data,
                        vec["u"].data ** 2)


def test_no_truth_value(vec):
    with pytest.raises(TypeError, match="truth value"):
        bool(vec)


def test_equality_is_identity(vec, u, v):
    same = vec
    assert vec == same
    assert vec != VectorField([u, v])  # equal content, new object
    assert hash(vec) == id(vec)


def test_repr_names_the_components(vec):
    assert "u" in repr(vec)
    assert "Right" in repr(vec)


# ================================================================
#  Functional surface
# ================================================================
def test_map_keeps_names_and_order(vec):
    doubled = vec.map(lambda f: f * 2.0)
    assert doubled.component_names == ("u", "v")
    assert jnp.allclose(doubled["u"].data, 2.0 * vec["u"].data)


def test_map_allows_space_changes(vec, mx, my):
    moved = vec.map(lambda f: f.to(mx.center * my.center))
    assert moved.component_names == ("u", "v")
    assert moved["u"].function_space.bare is mx.center * my.center


def test_replace(vec, u):
    shifted = vec.replace(u=u + 1.0)
    assert shifted["v"] is vec["v"]
    assert jnp.allclose(shifted["u"].data, u.data + 1.0)
    with pytest.raises(KeyError, match="no components named"):
        vec.replace(w=u)


# ================================================================
#  Componentwise arithmetic
# ================================================================
def test_vector_vector_arithmetic(vec):
    total = vec + vec
    assert jnp.allclose(total["u"].data, 2.0 * vec["u"].data)
    diff = vec - vec
    assert jnp.allclose(diff["v"].data, 0.0)


def test_name_tuples_must_match(vec, u, v):
    renamed = VectorField({"v": v, "u": u})  # order differs
    with pytest.raises(ValueError, match="identical component-name"):
        vec + renamed


def test_scalar_arithmetic(vec):
    assert jnp.allclose((vec + 1.0)["u"].data, vec["u"].data + 1.0)
    assert jnp.allclose((2.0 + vec)["u"].data, vec["u"].data + 2.0)
    assert jnp.allclose((1.0 - vec)["u"].data, 1.0 - vec["u"].data)
    assert jnp.allclose((-vec)["u"].data, -vec["u"].data)
    assert (+vec) is vec
    assert jnp.allclose((vec * 3.0)["u"].data, 3.0 * vec["u"].data)
    assert jnp.allclose((vec / 2.0)["u"].data, vec["u"].data / 2.0)


def test_field_broadcast_product(vec, grid, mx, my):
    # one ScalarField broadcast against every component; each pair
    # follows the join, so the coefficient must live on constant
    # factors wherever the components stagger
    f_cor = grid.create_field(mx.constant * my.constant,
                              data=jnp.full((1, 1), 2.0))
    scaled = f_cor * vec
    assert scaled.component_names == ("u", "v")
    assert scaled["u"].function_space.bare is mx.right * my.center
    assert jnp.allclose(scaled["u"].data, 2.0 * vec["u"].data)
    scaled_r = vec * f_cor
    assert jnp.allclose(scaled_r["v"].data, scaled["v"].data)


def test_componentwise_product_follows_the_join(vec, mx, my):
    # same-space pairs multiply; incompatible pairs raise per
    # component (the ScalarField strict-algebra rule)
    squared = vec * vec
    assert jnp.allclose(squared["u"].data, vec["u"].data ** 2)
    moved = vec.map(lambda f: f.to(mx.center * my.center))
    with pytest.raises(SpaceMismatchError):
        _ = vec * moved


def test_power_and_division(vec):
    u2 = (vec + 2.0) ** 2
    assert jnp.allclose(u2["u"].data, (vec["u"].data + 2.0) ** 2)
    quot = vec / (vec + 1.0)
    assert jnp.allclose(quot["u"].data,
                        vec["u"].data / (vec["u"].data + 1.0))


def test_unsupported_operands_raise_type_errors(vec):
    for operation in (
        lambda: vec + "no",
        lambda: "no" + vec,
        lambda: vec - "no",
        lambda: "no" - vec,
        lambda: vec * "no",
        lambda: "no" * vec,
        lambda: vec / "no",
        lambda: vec ** "no",
    ):
        with pytest.raises(TypeError):
            operation()


# ================================================================
#  Metadata preservation (the scan-stability rule)
# ================================================================
def test_componentwise_ops_preserve_metadata(vec):
    stepped = vec + 0.5 * vec.map(lambda f: f * 2.0)
    assert stepped["u"].name == "u"
    assert stepped["u"].metadata.units == "m/s"
    assert (-vec)["v"].name == "v"
    assert (vec * 2.0)["u"].name == "u"


def test_tree_structure_is_scan_stable(vec):
    dz = vec.map(lambda f: f * 0.1)
    stepped = vec + 0.5 * dz
    assert (jax.tree_util.tree_structure(stepped)
            == jax.tree_util.tree_structure(vec))


def test_renaming_changes_the_treedef(vec, u, v):
    renamed = VectorField({"a": u, "b": v})
    assert (jax.tree_util.tree_structure(renamed)
            != jax.tree_util.tree_structure(vec))


# ================================================================
#  Pytree round trips
# ================================================================
def test_flatten_unflatten_round_trip(vec):
    leaves, treedef = jax.tree_util.tree_flatten(vec)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, VectorField)
    assert rebuilt.component_names == vec.component_names
    assert jnp.allclose(rebuilt["u"].data, vec["u"].data)


def test_jit_and_scan_round_trips(vec):
    @jax.jit
    def step(z):
        return z + 0.1 * z

    out = step(vec)
    assert out.component_names == ("u", "v")

    def body(carry, _):
        return carry + 0.1 * carry, None

    final, _ = jax.lax.scan(body, vec, None, length=3)
    assert final.component_names == ("u", "v")
    assert final["u"].name == "u"


# ================================================================
#  Diagnostics
# ================================================================
def test_has_nan(vec, u):
    assert not bool(vec.has_nan())
    poisoned = vec.replace(
        u=u.with_data(u.data.at[0, 0].set(jnp.nan)))
    assert bool(poisoned.has_nan())


def test_block_until_ready(vec):
    assert vec.block_until_ready() is vec


def test_xr_is_the_export_entry_point(vec):
    # thin forwarder to the export module (label rules are tested in
    # tests/spatial/test_export.py)
    ds = vec.xr
    assert sorted(ds.data_vars) == ["u", "v"]
