"""Tests for fridom.spatial.fields.vector_field."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.decomposition.halo import HaloSpec
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
#  set (the coerced functional update)
# ================================================================
def test_set_accepts_a_matching_field(vec, u):
    out = vec.set(u=u + 2.0)
    assert jnp.allclose(out["u"].data, u.data + 2.0)
    assert out["v"] is vec["v"]


def test_set_rejects_a_space_mismatched_field(vec, v):
    with pytest.raises(ValueError, match="declared component space"):
        vec.set(u=v)


def test_set_accepts_a_coordinate_callable(vec, grid, mx, my):
    out = vec.set(u=lambda x, y: jnp.sin(x) + 0.0 * y)
    direct = grid.create_field(mx.right * my.center,
                               init=lambda x, y: jnp.sin(x) + 0.0 * y)
    assert jnp.allclose(out["u"].data, direct.data)
    # the callable samples each component's OWN space
    out2 = vec.set(v=lambda x, y: jnp.cos(y) + 0.0 * x)
    direct2 = grid.create_field(mx.center * my.right,
                                init=lambda x, y: jnp.cos(y) + 0.0 * x)
    assert jnp.allclose(out2["v"].data, direct2.data)


def test_set_callable_with_wrong_names_teaches(vec):
    with pytest.raises(TypeError, match="name exactly"):
        vec.set(u=lambda x: x)


def test_set_accepts_a_scalar_fill(vec):
    out = vec.set(u=2.5)
    data = np.asarray(out["u"].data)
    assert (data == 2.5).all()


def test_set_is_device_count_invariant():
    # every re-home flavor (callable, scalar fill, host array, device
    # array) is born sharded on a multi-device grid and must equal the
    # one-device result bit for bit, walled staggered spaces included
    def build(device_ids):
        mx = IntervalMesh(16, (0.0, 1.0), periodic=False, name="x")
        my = IntervalMesh(4, (0.0, 1.0), name="y")
        grid = Grid((mx, my), device_ids=device_ids)
        return VectorField({
            "u": grid.create_field(mx.inner * my.center),
            "o": grid.create_field(mx.outer * my.center)})

    host = np.random.default_rng(0).standard_normal((15, 4))
    for value in (lambda x, y: jnp.sin(3.0 * x) * jnp.cos(y), 2.5,
                  host, host.astype(np.float32), jnp.asarray(host)):
        many, one = (build(ids).set(u=value) for ids in (None, (0,)))
        assert many["u"].dtype == one["u"].dtype
        assert np.array_equal(np.asarray(many["u"].data),
                              np.asarray(one["u"].data))
    many, one = (
        build(ids).set(o=lambda x, y: jnp.exp(x) + y)
        for ids in (None, (0,)))
    assert np.array_equal(np.asarray(many["o"].data),
                          np.asarray(one["o"].data))


def test_set_accepts_a_true_shape_array(vec, u):
    out = vec.set(u=np.full(u.shape, 3.0))
    assert (np.asarray(out["u"].data) == 3.0).all()
    with pytest.raises(ValueError, match="true shape"):
        vec.set(u=np.zeros((3, 3)))


def test_set_unknown_component_teaches(vec):
    with pytest.raises(MissingComponentError):
        vec.set(w=1.0)


def test_set_keeps_incumbent_metadata(vec, grid, mx, my):
    other = grid.create_field(mx.right * my.center, name="bogus",
                              units="K")
    out = vec.set(u=other)
    assert out["u"].name == "u"
    assert out["u"].metadata.units == "m/s"


def test_set_preserves_the_subclass_and_treedef(vec):
    class Tagged(VectorField):
        pass

    tagged = Tagged(dict(vec.components))
    out = tagged.set(u=1.0)
    assert type(out) is Tagged
    assert (jax.tree_util.tree_structure(out)
            == jax.tree_util.tree_structure(tagged))


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


def test_numpy_array_operand_raises_instead_of_degrading(vec):
    # __array_ufunc__ = None: numpy defers instead of silently
    # returning an object ndarray of per-element containers
    arr = np.ones((8, 4))
    for op in (
        lambda: vec + arr, lambda: arr + vec,
        lambda: vec - arr, lambda: arr - vec,
        lambda: vec * arr, lambda: arr * vec,
    ):
        with pytest.raises(TypeError):
            op()
    with pytest.raises(TypeError, match="does not support ufuncs"):
        np.sqrt(vec)


def test_numpy_scalars_still_act_as_scalar_operands(vec):
    for s in (np.float64(2.5), np.array(2.5)):
        assert jnp.allclose((vec + s)["u"].data, vec["u"].data + 2.5)
        assert jnp.allclose((s + vec)["u"].data, vec["u"].data + 2.5)
        assert jnp.allclose((vec * s)["u"].data, 2.5 * vec["u"].data)
        assert jnp.allclose((s * vec)["u"].data, 2.5 * vec["u"].data)


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


@pytest.mark.parametrize("valid", [(0, 0), (1, 2), (3, 3)])
def test_metadata_reattachment_preserves_result_halo_claims(valid):
    mesh = IntervalMesh(32, (0., 1.), name="x")
    grid = Grid((mesh,))
    grid.negotiate(state_spaces=(mesh.center,), halo=HaloSpec({"x": 3}))
    source = grid.create_field(mesh.center, name="original",
                               init=lambda x: jnp.sin(7 * x))
    full = grid.sync(source)
    result = type(full)(grid, full.function_space, full.storage,
                         full.metadata.cleared(),
                         halo_valid=HaloSpec({"x": valid}))
    vector = VectorField({"component": source})
    for output in (vector.replace(component=result),
                   vector.map(lambda _: result)):
        assert output["component"].metadata is source.metadata
        assert output["component"].halo_valid == result.halo_valid
        np.testing.assert_array_equal(output["component"].storage,
                                      result.storage)
    # Arithmetic must keep the intersection computed by ScalarField,
    # rather than the original component's larger claim or zero claims.
    carried = VectorField({"component": full})
    incoming = VectorField({"component": result})
    for output in (carried.add(component=result), carried + incoming,
                   carried * incoming):
        assert output["component"].halo_valid == result.halo_valid
        assert output["component"].metadata is full.metadata


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


# ================================================================
#  Taught attribute misses (.fields is not aliased, it is taught)
# ================================================================
def test_fields_names_the_three_public_spellings(vec):
    # the natural reach: the storage slot really is _fields
    with pytest.raises(AttributeError) as excinfo:
        vec.fields  # noqa: B018 — the miss IS the assertion
    message = str(excinfo.value)
    assert "'VectorField' object has no attribute 'fields'" in message
    assert ".components" in message
    assert ".component_names" in message
    assert "tuple(vec)" in message
    # and all three really do work
    assert dict(vec.components) == {"u": vec["u"], "v": vec["v"]}
    assert vec.component_names == ("u", "v")
    assert tuple(vec) == (vec["u"], vec["v"])


def test_names_points_at_component_names(vec):
    with pytest.raises(AttributeError, match="component_names"):
        vec.names  # noqa: B018 — the miss IS the assertion


def test_an_unrelated_miss_stays_a_bare_attribute_error(vec):
    with pytest.raises(AttributeError) as excinfo:
        vec.w  # noqa: B018 — the miss IS the assertion
    assert str(excinfo.value) == (
        "'VectorField' object has no attribute 'w'")


def test_private_probes_get_the_bare_miss(vec):
    # copy/pickle/jax probe dunders; a hinted message there is noise
    with pytest.raises(AttributeError) as excinfo:
        vec.__deepcopy__  # noqa: B018 — the miss IS the assertion
    assert "did you mean" not in str(excinfo.value)
    assert not hasattr(vec, "_fields_")
    # and the private slot itself is still readable (the storage seam)
    assert vec._fields == (vec["u"], vec["v"])


def test_the_taught_miss_survives_a_pytree_round_trip(vec):
    # tree_unflatten builds via object.__new__; __getattr__ touches no
    # instance state, so it must not recurse on the half-built object
    leaves, treedef = jax.tree_util.tree_flatten(vec)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.component_names == ("u", "v")
    with pytest.raises(AttributeError, match=r"\.components"):
        rebuilt.fields  # noqa: B018 — the miss IS the assertion
