"""Treedef metadata-insensitivity through the jaxify annotation aux.

Covers phase2 grid follow-up item 7 (fields.md 2026-07-08 amendment;
phase1_findings finding 2): ``FieldMetadata`` stays in the static aux
but is exempt from aux equality, so treedefs, ``lax.scan`` carries,
and jit caching are metadata-insensitive — while the old-framework
``jaxify`` behavior (no opt-in) is unchanged.
"""
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from fridom.framework.utils import jaxify
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


# ================================================================
#  Treedef stability across metadata changes
# ================================================================
def test_treedef_is_metadata_insensitive(grid):
    a = grid.create_field(name="a")
    b = a.with_metadata(name="b", units="m/s")
    assert (jax.tree_util.tree_structure(a)
            == jax.tree_util.tree_structure(b))


def test_treedef_still_changes_when_space_changes(grid, mx, my):
    # the exemption is metadata-only, never structural
    a = grid.create_field(mx.center * my.center)
    b = grid.create_field(mx.right * my.center)
    assert (jax.tree_util.tree_structure(a)
            != jax.tree_util.tree_structure(b))


def test_metadata_survives_flatten_unflatten(grid):
    # annotation-exempt from equality, but carried in the aux data
    a = grid.create_field(name="a", units="K")
    leaves, treedef = jax.tree_util.tree_flatten(a)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.name == "a"
    assert back.metadata.units == "K"


def test_vector_treedef_is_component_metadata_insensitive(grid):
    u = grid.create_field(name="u", units="m/s")
    v = grid.create_field(name="v", units="m/s")
    vec = VectorField([u, v])
    other = VectorField({"u": u.with_metadata(units="1"), "v": v})
    assert (jax.tree_util.tree_structure(vec)
            == jax.tree_util.tree_structure(other))
    # component *names* stay structural: re-keying changes the treedef
    renamed = VectorField({"a": u, "b": v})
    assert (jax.tree_util.tree_structure(renamed)
            != jax.tree_util.tree_structure(vec))


# ================================================================
#  scan carries and jit caching
# ================================================================
def test_scan_carry_survives_a_metadata_change(grid):
    # phase1_findings finding 2: bare arithmetic resets metadata, so
    # a named-field carry used to change its treedef and scan raised
    a = grid.create_field(name="a", data=jnp.ones((8, 4)))

    def body(carry, _):
        return carry + 1.0, None

    final, _ = jax.lax.scan(body, a, None, length=3)
    assert jnp.allclose(final.data, 4.0)


def test_scan_carry_survives_vector_add_with_reset_metadata(grid):
    u = grid.create_field(name="u", data=jnp.ones((8, 4)))
    v = grid.create_field(name="v", data=jnp.ones((8, 4)))
    vec = VectorField([u, v])

    def body(carry, _):
        return carry.add(u=carry["u"] * 0.5), None

    final, _ = jax.lax.scan(body, vec, None, length=2)
    assert final.component_names == ("u", "v")
    assert jnp.allclose(final["u"].data, 2.25)
    assert jnp.allclose(final["v"].data, 1.0)


def test_jit_does_not_retrace_on_a_metadata_change(
        grid, compile_counter):
    a = grid.create_field(name="a", data=jnp.ones((8, 4)))

    @jax.jit
    def step(x):
        return x + 1.0

    step(a).block_until_ready()  # compile once
    compile_counter.reset()
    out = step(a.with_metadata(name="b", units="K"))
    out.block_until_ready()
    assert compile_counter.count == 0


def test_jit_returns_trace_time_metadata(grid):
    # documented consequence of the amendment: host code wanting
    # authoritative names reads container keys, never round-tripped
    # field metadata
    a = grid.create_field(name="a", data=jnp.ones((8, 4)))

    @jax.jit
    def identity(x):
        return x

    identity(a)  # trace with the "a" annotation
    out = identity(a.with_metadata(name="b"))  # cache hit
    assert out.name == "a"


# ================================================================
#  Old-framework jaxify regression (no opt-in, behavior unchanged)
# ================================================================
@partial(jaxify, dynamic=("arr",))
class _Legacy:

    """Minimal jaxified class without the annotation opt-in."""

    def __init__(self, arr, tag):
        self.arr = arr
        self.tag = tag


def test_legacy_jaxify_declares_no_annotation_category():
    assert _Legacy.annotation_jax_attrs == frozenset()


def test_legacy_static_attrs_still_enter_treedef_equality():
    a = _Legacy(jnp.ones(8), tag="a")
    b = _Legacy(jnp.ones(8), tag="b")
    assert (jax.tree_util.tree_structure(a)
            != jax.tree_util.tree_structure(b))


def test_legacy_equivalent_statics_still_share_the_treedef():
    # the structural (by value) equality of jaxified aux is untouched
    a = _Legacy(jnp.ones(8), tag="a")
    c = _Legacy(jnp.zeros(8), tag="a")
    assert (jax.tree_util.tree_structure(a)
            == jax.tree_util.tree_structure(c))


# ================================================================
#  The annotation category at the jaxify level (mechanism unit test)
# ================================================================
@partial(jaxify, dynamic=("arr",), annotation=("label",))
class _Annotated:

    """Minimal jaxified class with an annotation attribute."""

    def __init__(self, arr, power, label):
        self.arr = arr
        self.power = power
        self.label = label


def test_annotation_attr_is_exempt_from_treedef_equality():
    a = _Annotated(jnp.ones(8), power=2, label="a")
    b = _Annotated(jnp.ones(8), power=2, label="b")
    assert (jax.tree_util.tree_structure(a)
            == jax.tree_util.tree_structure(b))


def test_non_annotation_statics_still_compare():
    a = _Annotated(jnp.ones(8), power=2, label="a")
    b = _Annotated(jnp.ones(8), power=3, label="a")
    assert (jax.tree_util.tree_structure(a)
            != jax.tree_util.tree_structure(b))


def test_annotation_attr_survives_the_round_trip():
    a = _Annotated(jnp.ones(8), power=2, label="a")
    leaves, treedef = jax.tree_util.tree_flatten(a)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.label == "a"
    assert back.power == 2
