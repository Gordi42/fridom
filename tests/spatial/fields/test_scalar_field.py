"""Tests for fridom.spatial.fields.scalar_field."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.errors import (
    ImmutableStateError as ModelImmutableStateError,
)
from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.errors import (
    GridMismatchError,
    ImmutableStateError,
    SpaceMismatchError,
)
from fridom.spatial.fields.metadata import FieldMetadata
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.registry import OperatorRegistry
from fridom.spatial.scalars import Scalars, Variance
from fridom.spatial.spaces.nodal import NodeSet


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


@pytest.fixture
def f(grid):
    data = jnp.arange(32.0).reshape(8, 4) + 1.0
    return grid.create_field(data=data, name="f")


@pytest.fixture
def g(grid):
    data = jnp.linspace(-1.0, 1.0, 32).reshape(8, 4)
    return grid.create_field(data=data, name="g")


# ================================================================
#  Construction, properties, storage contract
# ================================================================
def test_properties(grid, mx, my, f):
    assert f.grid is grid
    assert f.function_space.bare is mx.center * my.center
    assert f.shape == (8, 4)
    assert f.dtype == jnp.float64
    assert f.name == "f"
    assert f.metadata.name == "f"


def test_data_roundtrips_through_pad_unpad(grid):
    data = jnp.arange(32.0).reshape(8, 4)
    f = grid.create_field(data=data)
    assert jnp.array_equal(f.data, data)
    g = f.with_data(2.0 * data)
    assert jnp.array_equal(g.data, 2.0 * data)


def test_with_data_keeps_grid_space_metadata(f):
    g = f.with_data(jnp.zeros((8, 4)))
    assert g.grid is f.grid
    assert g.function_space is f.function_space
    assert g.metadata == f.metadata


def test_with_data_rejects_wrong_shape(f):
    with pytest.raises(ValueError, match="true-shape"):
        f.with_data(jnp.zeros((4, 8)))


def test_data_assignment_raises_a_taught_immutable_state_error(f):
    with pytest.raises(ImmutableStateError, match="with_data"):
        f.data = f.data * 2.0


def test_data_inplace_op_raises_immutable_state_error(f):
    with pytest.raises(ImmutableStateError):
        f.data += 1.0


def test_immutable_state_error_re_exported_from_model():
    # the class moved to the grid cluster; model.errors re-exports it
    assert ModelImmutableStateError is ImmutableStateError


def test_storage_is_the_padded_frame_no_copy(f):
    assert f.storage is f._data
    assert all(s >= t for s, t
               in zip(f.storage.shape, f.data.shape, strict=True))


def test_with_storage_round_trips_without_pad(f):
    g = f.with_storage(f.storage)
    assert g.grid is f.grid
    assert g.function_space is f.function_space
    assert g.metadata == f.metadata
    assert g.storage is f.storage
    assert jnp.array_equal(g.data, f.data)


def test_with_storage_claims_zero_ghost_validity(grid, f):
    # the canonical with_data state: claims are dropped, so the
    # first ghost-consuming application re-syncs — always sound
    synced = grid.sync(f)
    g = synced.with_storage(synced.storage)
    assert g.halo_valid == synced.halo_valid.zero(
        tuple(f.function_space.names))
    assert jnp.array_equal(g.data, synced.data)


def test_with_metadata(f):
    g = f.with_metadata(name="q", units="m/s")
    assert g.name == "q"
    assert g.metadata.units == "m/s"
    assert jnp.array_equal(g.data, f.data)
    assert f.name == "f"


def test_trusting_constructor_takes_storage_shaped_data(grid, mx, my):
    space = (mx.center * my.center).with_layout(
        grid.decomposition.default_layout)
    stored = grid.decomposition.zeros(space)
    f = ScalarField(grid, space, stored)
    assert f.metadata == FieldMetadata()
    assert jnp.array_equal(f.data, jnp.zeros((8, 4)))


def test_repr(f):
    text = repr(f)
    assert "'f'" in text
    assert "(8, 4)" in text


# ================================================================
#  Identity, truthiness, comparisons
# ================================================================
def test_equality_is_identity(grid):
    data = jnp.ones((8, 4))
    a = grid.create_field(data=data)
    b = grid.create_field(data=data)
    assert a == a  # noqa: PLR0124 — identity semantics under test
    assert (a == b) is False
    assert a != b


def test_bool_raises(f):
    with pytest.raises(TypeError, match="no truth value"):
        bool(f)


def test_no_ordering_comparisons(f, g):
    with pytest.raises(TypeError):
        _ = f < g


# ================================================================
#  Linear arithmetic and the strict algebra
# ================================================================
def test_add_same_space(f, g):
    h = f + g
    assert jnp.array_equal(h.data, f.data + g.data)
    assert h.function_space is f.function_space
    assert h.metadata == FieldMetadata()  # default-metadata rule


def test_sub_same_space(f, g):
    h = f - g
    assert jnp.array_equal(h.data, f.data - g.data)


def test_cross_space_add_raises_with_factor_diff(grid, mx, my, f):
    other = grid.create_field(mx.right * my.center)
    with pytest.raises(SpaceMismatchError,
                       match=r"x: .*Right.* vs .*Center.*") as exc:
        _ = other + f
    assert exc.value.mismatched_names == ("x",)
    assert "y" in str(exc.value)  # the agreeing factor is reported
    assert ".to(" in str(exc.value)


def test_cross_grid_raises(mx, my, f):
    other_grid = Grid((mx, my))
    other = other_grid.create_field(data=jnp.ones((8, 4)))
    with pytest.raises(GridMismatchError, match="different grids"):
        _ = f + other


def test_factor_count_mismatch_raises(grid, mx, f):
    lean = grid.create_field(mx.center)
    with pytest.raises(SpaceMismatchError, match="meshes differ"):
        _ = f + lean


def test_constant_broadcast(grid, mx, my, f):
    profile = grid.create_field(
        mx.constant * my.center, data=jnp.arange(4.0).reshape(1, 4))
    h = f + profile
    assert h.function_space.bare is mx.center * my.center
    expected = f.data + jnp.arange(4.0)[None, :]
    assert jnp.array_equal(h.data, expected)


def test_real_complex_promotion(grid, mx, my, f):
    space = (mx.center * my.center).as_complex()
    z = grid.create_field(space, data=jnp.full((8, 4), 1.0 + 2.0j))
    h = f + z
    assert h.function_space.bare is space
    assert h.dtype == jnp.complex128
    assert jnp.array_equal(h.data, f.data + z.data)


def test_scalar_add_sub(f):
    assert jnp.array_equal((f + 2.0).data, f.data + 2.0)
    assert jnp.array_equal((3 + f).data, f.data + 3.0)
    assert jnp.array_equal((f - 1.0).data, f.data - 1.0)
    assert jnp.array_equal((1.0 - f).data, 1.0 - f.data)


def test_complex_scalar_add_promotes_space(f):
    h = f + 1.0j
    assert h.function_space.scalars is Scalars.COMPLEX
    assert h.dtype == jnp.complex128


def test_scalar_add_on_coefficient_space_raises(grid1d, mx):
    fhat = grid1d.create_field(mx.fourier(origin=mx.center))
    with pytest.raises(KeyError, match="zero-mode"):
        _ = fhat + 1.0


def test_coefficient_space_linear_ops_are_elementwise(grid1d, mx):
    space = mx.fourier(origin=mx.center)
    a = grid1d.random.normal(space, seed=0)
    b = grid1d.random.normal(space, seed=1)
    h = a + b
    assert h.function_space is a.function_space
    assert jnp.array_equal(h.data, a.data + b.data)


def test_neg_pos(f):
    assert jnp.array_equal((-f).data, -f.data)
    assert (-f).metadata == FieldMetadata()
    assert +f is f


def test_unsupported_operand_types(f):
    with pytest.raises(TypeError):
        _ = f + "nope"
    with pytest.raises(TypeError):
        _ = "nope" + f
    with pytest.raises(TypeError):
        _ = f - "nope"
    with pytest.raises(TypeError):
        _ = "nope" - f
    with pytest.raises(TypeError):
        _ = f * "nope"
    with pytest.raises(TypeError):
        _ = "nope" * f
    with pytest.raises(TypeError):
        _ = f / "nope"
    with pytest.raises(TypeError):
        _ = "nope" / f


def test_hash_is_identity_based(f):
    assert hash(f) == id(f)


# ================================================================
#  0-d array as a scalar operand (R1)
# ================================================================
def test_0d_array_acts_as_scalar(f):
    s = jnp.asarray(3.0)  # traced ctx.params leaf shape
    assert jnp.array_equal((s * f).data, 3.0 * f.data)
    assert jnp.array_equal((f * s).data, f.data * 3.0)
    assert jnp.array_equal((f + s).data, f.data + 3.0)
    assert jnp.array_equal((s + f).data, 3.0 + f.data)
    assert jnp.array_equal((f - s).data, f.data - 3.0)
    assert jnp.array_equal((s - f).data, 3.0 - f.data)
    assert jnp.allclose((f / s).data, f.data / 3.0)
    assert jnp.allclose((s / f).data, 3.0 / f.data)
    assert jnp.allclose((f ** jnp.asarray(2.0)).data, f.data ** 2)


def test_0d_array_matches_float_result(f):
    s = jnp.asarray(2.5)
    assert jnp.array_equal((s * f).data, (2.5 * f).data)
    assert jnp.array_equal((f + s).data, (f + 2.5).data)
    assert jnp.allclose((f / s).data, (f / 2.5).data)


def test_1d_array_is_not_a_scalar(f):
    arr = jnp.ones(4)  # a field-shaped array is not a scalar
    with pytest.raises(TypeError):
        _ = f * arr
    with pytest.raises(TypeError):
        _ = f + arr
    with pytest.raises(TypeError):
        _ = f / arr
    with pytest.raises(TypeError):
        _ = f ** arr


# ================================================================
#  Products (dispatch seam and iteration-1 fallback)
# ================================================================
def test_mul_same_space_elementwise_fallback(f, g):
    h = f * g
    assert jnp.array_equal(h.data, f.data * g.data)
    assert h.function_space is f.function_space
    assert h.metadata == FieldMetadata()


def test_mul_on_average_space(grid1d, mx):
    a = grid1d.create_field(mx.cell_avg, data=jnp.arange(8.0))
    b = grid1d.create_field(mx.cell_avg, data=jnp.ones(8) * 2.0)
    assert jnp.array_equal((a * b).data, jnp.arange(8.0) * 2.0)


def test_mul_cross_space_raises(grid, mx, my, f):
    other = grid.create_field(mx.right * my.center)
    with pytest.raises(SpaceMismatchError):
        _ = f * other


def test_mul_on_coefficient_space_raises(grid1d, mx):
    space = mx.fourier(origin=mx.center)
    a = grid1d.random.normal(space, seed=0)
    b = grid1d.random.normal(space, seed=1)
    with pytest.raises(KeyError, match="multiply"):
        _ = a * b


def test_div(f, g):
    h = f / (g + 2.0)
    assert jnp.allclose(h.data, f.data / (g.data + 2.0))


def test_div_on_coefficient_space_raises(grid1d, mx):
    space = mx.fourier(origin=mx.center)
    a = grid1d.random.normal(space, seed=0)
    with pytest.raises(KeyError, match="divide"):
        _ = a / a


def test_scalar_mul_div(f):
    assert jnp.array_equal((f * 2.0).data, f.data * 2.0)
    assert jnp.array_equal((3 * f).data, 3.0 * f.data)
    assert jnp.array_equal((f / 2.0).data, f.data / 2.0)
    assert jnp.allclose((2.0 / f).data, 2.0 / f.data)


def test_scalar_scaling_is_legal_on_coefficient_spaces(grid1d, mx):
    a = grid1d.random.normal(mx.fourier(origin=mx.center), seed=0)
    assert jnp.array_equal((2.0 * a).data, 2.0 * a.data)
    assert jnp.array_equal((a / 2.0).data, a.data / 2.0)


def test_complex_scalar_mul_promotes(f):
    h = f * 1.0j
    assert h.function_space.scalars is Scalars.COMPLEX
    assert jnp.array_equal(h.data, f.data * 1.0j)


def test_complex_scalar_on_half_spectrum_raises(grid1d, mx):
    a = grid1d.random.normal(mx.fourier(origin=mx.center), seed=0)
    with pytest.raises(NotImplementedError, match="Hermitian"):
        _ = a * 1.0j


def test_complex_scalar_rtruediv_promotes(f):
    h = 1.0j / f
    assert h.function_space.scalars is Scalars.COMPLEX
    assert jnp.allclose(h.data, 1.0j / f.data)


def test_rtruediv_on_coefficient_space_raises(grid1d, mx):
    a = grid1d.random.normal(mx.fourier(origin=mx.center), seed=0)
    with pytest.raises(KeyError, match="divide"):
        _ = 1.0 / a


def test_pow(f):
    assert jnp.allclose((f ** 2).data, f.data ** 2)
    assert jnp.allclose((f ** 0.5).data, f.data ** 0.5)


def test_pow_on_coefficient_space_raises(grid1d, mx):
    a = grid1d.random.normal(mx.fourier(origin=mx.center), seed=0)
    with pytest.raises(KeyError, match="power"):
        _ = a ** 2


def test_pow_bad_exponent(f):
    with pytest.raises(TypeError):
        _ = f ** "2"


def test_abs(mx, my, g):
    h = abs(g)
    assert jnp.array_equal(h.data, jnp.abs(g.data))
    assert h.function_space.bare is mx.center * my.center


def test_abs_of_complex_field_lands_on_real_space(grid, mx, my):
    space = (mx.center * my.center).as_complex()
    z = grid.create_field(space, data=jnp.full((8, 4), 3.0 + 4.0j))
    h = abs(z)
    assert h.function_space.bare is mx.center * my.center
    assert jnp.allclose(h.data, jnp.full((8, 4), 5.0))


def test_abs_on_lone_complex_factor(grid1d, mx):
    z = grid1d.create_field(mx.center.as_complex(),
                            data=jnp.full(8, 3.0 + 4.0j))
    h = abs(z)
    assert h.function_space.bare is mx.center
    assert jnp.allclose(h.data, jnp.full(8, 5.0))


def test_abs_on_average_spaces(grid1d, mx):
    # abs is pointwise, so it is well-defined on cell/face averages at
    # the same order (seeded on nodal + average, not nodal only)
    data = jnp.arange(8.0) - 4.0
    a = grid1d.create_field(mx.cell_avg, data=data)
    h = abs(a)
    assert h.function_space.bare is mx.cell_avg
    assert jnp.array_equal(h.data, jnp.abs(data))


def test_constant_into_coefficient_lift_raises(grid1d, mx):
    space = mx.fourier(origin=mx.center)
    a = grid1d.random.normal(space, seed=0)
    c = grid1d.create_field(mx.constant, data=jnp.ones(1))
    with pytest.raises(KeyError, match="broadcast"):
        _ = a + c


def test_half_spectrum_promotion_lift_raises(grid1d, mx):
    real_space = mx.fourier(origin=mx.center)
    full_space = mx.fourier(origin=mx.center.as_complex())
    a = grid1d.random.normal(real_space, seed=0)
    b = grid1d.random.normal(full_space, seed=1)
    with pytest.raises(NotImplementedError, match="Hermitian"):
        _ = a + b


# ================================================================
#  Registry dispatch seam
# ================================================================
class _RecordingRegistry:

    """Duck-typed OperatorRegistry standing in for the merge."""

    def __init__(self):
        self.calls = []
        self.operands = []

        def op(a, b):
            self.operands.append((a, b))
            return a.with_data(a.data * b.data + 1.0)

        self._op = op

    def resolve(self, kind, space):
        self.calls.append((kind, space))
        return self._op


class _EmptyRegistry:

    """Registry with no entries: resolve always raises KeyError."""

    def resolve(self, kind, space):
        raise KeyError(f"no ({kind!r}, {space!r}) entry")


def test_mul_routes_through_grid_dispatch(mx, my):
    registry = _RecordingRegistry()
    grid = Grid((mx, my), dispatch=registry)
    f = grid.create_field(data=jnp.ones((8, 4)) * 2.0)
    g = grid.create_field(data=jnp.ones((8, 4)) * 3.0)
    h = f * g
    # the fallback would give 6.0; the registry op gives 7.0
    assert jnp.array_equal(h.data, jnp.full((8, 4), 7.0))
    assert registry.calls == [
        ("multiply", (mx.center * my.center))]


def test_dispatch_receives_lifted_operands(mx, my):
    registry = _RecordingRegistry()
    grid = Grid((mx, my), dispatch=registry)
    f = grid.create_field(data=jnp.ones((8, 4)))
    c = grid.create_field(
        mx.constant * my.center, data=jnp.arange(4.0).reshape(1, 4))
    _ = f * c
    a, b = registry.operands[0]
    assert a.function_space is b.function_space
    assert b.function_space.bare is mx.center * my.center
    assert jnp.array_equal(
        b.data, jnp.broadcast_to(jnp.arange(4.0), (8, 4)))


def test_dispatch_resolution_error_propagates(mx, my):
    grid = Grid((mx, my), dispatch=_EmptyRegistry())
    f = grid.create_field(data=jnp.ones((8, 4)))
    with pytest.raises(KeyError, match="no \\('multiply'"):
        _ = f * f


# ================================================================
#  Scalars (Körper) surface
# ================================================================
def test_real_is_identity_on_real_fields(f):
    assert f.real is f


def test_real_is_identity_on_real_origin_fourier(grid1d, mx):
    a = grid1d.random.normal(mx.fourier(origin=mx.center), seed=0)
    assert a.real is a
    assert a.conj() is a


def test_imag_of_real_field_is_zero_same_space(f):
    h = f.imag
    assert h.function_space is f.function_space
    assert jnp.array_equal(h.data, jnp.zeros((8, 4)))
    assert h.metadata == f.metadata  # same-quantity rule


def test_conj_is_identity_on_real_fields(f):
    assert f.conj() is f


def test_real_imag_conj_on_complex_nodal(grid, mx, my):
    space = (mx.center * my.center).as_complex()
    z = grid.create_field(
        space, data=jnp.full((8, 4), 1.0 + 2.0j), name="z")
    assert z.real.function_space.bare is mx.center * my.center
    assert jnp.array_equal(z.real.data, jnp.full((8, 4), 1.0))
    assert jnp.array_equal(z.imag.data, jnp.full((8, 4), 2.0))
    assert jnp.array_equal(z.conj().data, jnp.full((8, 4), 1.0 - 2.0j))
    assert z.conj().function_space is z.function_space
    assert z.real.name == "z"  # same-quantity ops keep metadata


def test_real_imag_conj_on_complex_coefficients_raise(grid1d, mx):
    space = mx.fourier(origin=mx.center.as_complex())
    z = grid1d.random.normal(space, seed=0)
    with pytest.raises(NotImplementedError, match="conjugate"):
        _ = z.real
    with pytest.raises(NotImplementedError, match="conjugate"):
        _ = z.imag
    with pytest.raises(NotImplementedError, match="conjugate"):
        _ = z.conj()


def test_as_complex(f):
    z = f.as_complex()
    assert z.function_space.scalars is Scalars.COMPLEX
    assert z.dtype == jnp.complex128
    assert jnp.array_equal(z.data.real, f.data)
    assert z.as_complex() is z
    assert z.metadata == f.metadata


def test_as_complex_on_half_spectrum_raises(grid1d, mx):
    a = grid1d.random.normal(mx.fourier(origin=mx.center), seed=0)
    with pytest.raises(NotImplementedError, match="Hermitian"):
        a.as_complex()


# ================================================================
#  Dispatch sugar (thin forwarders over the seeded registry)
# ================================================================
def test_to_identity_returns_self(f, mx):
    assert f.to(f) is f
    assert f.to(f.function_space) is f
    assert f.to(f.function_space.bare) is f
    assert f.to(mx.center) is f  # single-factor shorthand


def test_to_interpolates_through_the_registry(f, mx, my):
    g = f.to(mx.right)
    assert g.function_space.bare is mx.right * my.center
    # periodic two-point mean (Center -> Right), wrap at the seam
    expected = 0.5 * (f.data + jnp.roll(f.data, -1, axis=0))
    assert jnp.allclose(g.data, expected)
    assert g.metadata == f.metadata  # same-quantity rule


def test_to_field_target_and_round_trip_space(f, mx):
    g = f.to(mx.right)
    back = g.to(f)
    assert back.function_space is f.function_space


def test_to_routes_outer_to_inner_through_the_restrict_kind(grid, mx, my):
    # the bounded Outer -> Inner conversion is the exact restriction
    # (drops the two boundary faces), a distinct kind from the
    # Outer -> Center interpolate: .to reads "restrict" off the family
    # matrix and the seeded row selects the interior faces.
    w = grid.create_field(mx.center * my.outer,
                          init=lambda x, y: y**2 + 0.0 * x)
    r = w.to(mx.center * my.inner)
    assert r.function_space.bare is mx.center * my.inner
    assert jnp.allclose(r.data, w.data[:, 1:-1])
    # Outer -> Center stays the (distinct) two-point interpolate
    assert w.to(my.center).function_space.bare is mx.center * my.center


def test_to_single_factor_shorthand_on_lone_factor(grid1d, mx):
    a = grid1d.create_field(mx.center)
    assert a.to(mx.center) is a
    assert a.to(mx.right).function_space.bare is mx.right


def test_to_transform_target_raises_space_error(grid1d, mx):
    a = grid1d.create_field(mx.center)
    with pytest.raises(SpaceMismatchError, match="transform"):
        a.to(mx.fourier(origin=mx.center))


def test_to_unregistered_kind_raises_dispatch_error(mx):
    bare = Grid((mx,), dispatch=OperatorRegistry({}))
    a = bare.create_field(mx.cell_avg)
    with pytest.raises(KeyError, match="reconstruct"):
        a.to(mx.right)  # empty registry: no reconstruct rows


def test_diff_forwards_to_the_seeded_verb(grid1d, mx):
    a = grid1d.create_field(mx.center)
    d = a.diff("x")
    assert d.function_space.bare is mx.right
    assert d.metadata == FieldMetadata()  # new quantity


def test_xr_is_the_export_entry_point(f):
    # thin forwarder to the export module (label rules are tested in
    # tests/spatial/test_export.py)
    da = f.xr
    assert da.name == "f"
    assert da.dims == ("x", "y")
    assert jnp.array_equal(jnp.asarray(da.values), f.data)


# ================================================================
#  Grid accessor forwarders (R16a)
# ================================================================
def test_nodes_forwards_to_grid(f, grid):
    field = f.nodes("x")
    ref = grid.evaluation_nodes(f.function_space, "x")
    assert field.function_space is ref.function_space
    assert jnp.array_equal(field.data, ref.data)


def test_measure_forwards_to_grid(f, grid):
    field = f.measure("y")
    ref = grid.measure(f.function_space, "y")
    assert field.function_space is ref.function_space
    assert jnp.array_equal(field.data, ref.data)


def test_wavenumbers_forwards_to_grid(grid1d, mx):
    a = grid1d.random.normal(mx.fourier(origin=mx.center), seed=0)
    field = a.wavenumbers()  # name omitted: lone factor
    ref = grid1d.wavenumbers(a.function_space)
    assert field.function_space is ref.function_space
    assert jnp.array_equal(field.data, ref.data)


def test_reshard_forwards_to_the_movement_operator(f):
    # matching layout: identity elision (no operator application)
    assert f.reshard(f.function_space.layout) is f
    # a foreign layout is outside the closed negotiated vocabulary
    with pytest.raises(ValueError, match="vocabulary"):
        f.reshard(None)


# ================================================================
#  Diagnostics
# ================================================================
def test_has_nan(f):
    assert not bool(f.has_nan())
    bad = f.with_data(f.data.at[2, 1].set(jnp.nan))
    assert bool(bad.has_nan())


def test_block_until_ready_returns_self(f):
    assert f.block_until_ready() is f


# ================================================================
#  Pytree behavior
# ================================================================
def test_pytree_roundtrip_preserves_statics(grid, f):
    leaves, treedef = jax.tree_util.tree_flatten(f)
    assert len(leaves) == 1
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.grid is grid
    assert back.function_space is f.function_space
    assert back.metadata == f.metadata
    assert jnp.array_equal(back.data, f.data)


def test_treedef_stable_across_same_space_fields(grid):
    a = grid.create_field(data=jnp.ones((8, 4)), name="a")
    b = grid.create_field(data=jnp.zeros((8, 4)), name="a")
    assert (jax.tree_util.tree_structure(a)
            == jax.tree_util.tree_structure(b))


def test_treedef_changes_when_space_changes(grid, mx, my):
    a = grid.create_field(mx.center * my.center)
    b = grid.create_field(mx.right * my.center)
    assert (jax.tree_util.tree_structure(a)
            != jax.tree_util.tree_structure(b))


def test_treedef_is_metadata_insensitive(grid):
    # the annotation-exempt equality (fields.md amendment,
    # 2026-07-08): metadata is carried in the aux but exempt from
    # treedef equality, so scan carries and jit caches survive
    # metadata changes
    a = grid.create_field(name="a")
    b = a.with_metadata(name="b")
    assert (jax.tree_util.tree_structure(a)
            == jax.tree_util.tree_structure(b))


def test_jit_function_over_fields(grid, f, g):
    @jax.jit
    def step(a, b):
        return a + 0.5 * b

    out = step(f, g)
    assert isinstance(out, ScalarField)
    assert out.grid is grid
    assert out.function_space is f.function_space
    assert jnp.allclose(out.data, f.data + 0.5 * g.data)


def test_arithmetic_traces_once_across_same_shape_calls(
        grid, compile_counter):
    a = grid.create_field(data=jnp.ones((8, 4)))
    b = grid.create_field(data=jnp.full((8, 4), 2.0))
    c = grid.create_field(data=jnp.full((8, 4), 3.0))

    @jax.jit
    def tendency(u, v):
        return u * v + u - 0.5 * v

    tendency(a, b).block_until_ready()  # compile once
    compile_counter.reset()
    tendency(b, c).block_until_ready()
    tendency(c, a).block_until_ready()
    assert compile_counter.count == 0


def test_to_rejects_differing_coordinate_names(grid1d, mx):
    mz = IntervalMesh(8, (0.0, 1.0), name="z")
    a = grid1d.create_field(mx.center)
    with pytest.raises(SpaceMismatchError, match="names differ"):
        a.to(mz.center)


def test_to_codomain_disagreement_raises():
    my = IntervalMesh(4, (0.0, 2.0), periodic=False, name="y")
    grid = Grid((my,))
    a = grid.create_field(my.center)
    # the registered interpolate default lands on Inner, not Outer
    with pytest.raises(SpaceMismatchError, match="lands on"):
        a.to(my.outer)


def test_to_nodal_to_average_resolves_the_average_rows(grid1d, mx):
    a = grid1d.create_field(mx.right)
    # Right -> CellAvg is the shifted evaluate-to-average ("average")
    assert a.to(mx.cell_avg).function_space.bare is mx.cell_avg
    # Center -> FaceAvg is the shifted dual "average" kind
    b = grid1d.create_field(mx.center)
    assert b.to(mx.face_avg).function_space.bare is mx.face_avg


def test_to_colocated_average_nodal_is_the_deconvolution(grid1d, mx):
    # Center <-> CellAvg is co-located (both at the cell midpoint): the
    # 2nd-order deconvolution (G3), a distinct kind from the shifted
    # "average"/"reconstruct" rows, resolving through .to in both
    # directions with the data unchanged (an identity retag)
    b = grid1d.create_field(mx.center, data=jnp.arange(8.0))
    to_avg = b.to(mx.cell_avg)
    assert to_avg.function_space.bare is mx.cell_avg
    assert jnp.array_equal(to_avg.data, b.data)
    p = grid1d.create_field(mx.cell_avg, data=jnp.arange(8.0))
    to_nod = p.to(mx.center)
    assert to_nod.function_space.bare is mx.center
    assert jnp.array_equal(to_nod.data, p.data)
    # round trip is exact
    assert jnp.array_equal(p.to(mx.center).to(mx.cell_avg).data, p.data)


def test_to_between_coefficient_origins_is_unregistered(grid1d, mx):
    a = grid1d.random.normal(mx.fourier(origin=mx.center), seed=0)
    with pytest.raises(KeyError, match="interpolate"):
        a.to(mx.fourier(origin=mx.right))


def test_to_from_constant_factor_broadcasts(grid, mx, my):
    # GAP A fix: `.to` from a ConstantSpace factor is the sanctioned
    # constant broadcast (rules 3.3), equal to the implicit lift in a
    # product, and materializes the full field.
    profile = grid.create_field(mx.constant * my.center,
                                init=lambda y: 1.0 + y)
    lifted = profile.to(mx.center * my.center)
    assert lifted.function_space.bare is (mx.center * my.center)
    full = grid.create_field(
        mx.center * my.center,
        init=lambda x, y: 1.0 + y)  # noqa: ARG005 — init(**coords) by name
    assert jnp.allclose(lifted.data, full.data)


# ================================================================
#  BC-sibling retag seam (C6)
# ================================================================
@pytest.fixture
def walled():
    mz = IntervalMesh(8, (0.0, 1.0), periodic=False, name="z")
    return Grid((mz,)), mz


def test_retag_identity_returns_self(f, my):
    assert f.retag(f) is f
    assert f.retag(f.function_space) is f
    assert f.retag(my.center) is f  # single-factor shorthand


def test_retag_between_bc_siblings_preserves_data(mx, my, f):
    tagged = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    g = f.retag(tagged)
    assert g.function_space.bare is mx.center * tagged
    assert g.grid is f.grid
    assert g.metadata == f.metadata  # same-quantity rule
    assert jnp.array_equal(g.data, f.data)


def test_retag_between_average_bc_siblings(mx, my, grid):
    # F4: average factors now carry retaggable BC tags -- a CellAvg and
    # its Neumann sibling agree on class/mesh/shape/scalars, so
    # div.retag(neumann_sibling) in the walled FV pressure solve is a
    # pure BC-tag swap (data preserved)
    from fridom.spatial.spaces.average import CellAvg  # noqa: PLC0415
    space = mx.cell_avg * my.cell_avg
    p = grid.create_field(space, data=jnp.arange(32.0).reshape(8, 4))
    tagged = space.replace(y=my.average(CellAvg, bc=BC.NEUMANN))
    g = p.retag(tagged)
    assert g.function_space.bare is tagged
    assert jnp.array_equal(g.data, p.data)
    assert jnp.array_equal(g.retag(p.function_space).data, p.data)


def test_retag_full_product_target_and_round_trip(mx, my, f):
    tagged = mx.center * my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    g = f.retag(tagged)
    assert g.function_space.bare is tagged
    back = g.retag(f.function_space)
    assert back.function_space is f.function_space
    assert jnp.array_equal(back.data, f.data)


def test_retag_rejects_different_node_set(my, f):
    # Left free matches Center free in mesh, shape, and scalars but
    # not in node-set class: not a BC sibling
    with pytest.raises(SpaceMismatchError,
                       match="beyond their BC tags"):
        f.retag(my.left)


def test_retag_rejects_different_shape(grid, my):
    h = grid.create_field(grid.factors[0].center * my.outer)
    # Outer(DIRICHLET) drops the two boundary DOFs: same node-set
    # class, different shape
    with pytest.raises(SpaceMismatchError,
                       match="beyond their BC tags"):
        h.retag(my.nodal(NodeSet.OUTER, bc=BC.DIRICHLET))


def test_retag_rejects_different_mesh(grid1d, mx):
    other = IntervalMesh(8, (0.0, 1.0), name="x")
    a = grid1d.create_field(mx.center)
    with pytest.raises(SpaceMismatchError,
                       match="beyond their BC tags"):
        a.retag(other.center)


def test_retag_rejects_different_scalars(my, f):
    tagged = my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET).as_complex()
    with pytest.raises(SpaceMismatchError,
                       match="beyond their BC tags"):
        f.retag(tagged)


def test_retag_rejects_differing_coordinate_names(grid1d, mx):
    mz = IntervalMesh(8, (0.0, 1.0), name="z")
    a = grid1d.create_field(mx.center)
    with pytest.raises(SpaceMismatchError, match="names differ"):
        a.retag(mz.center)


def test_retag_resets_validity_on_retagged_axes_only(grid, my, f):
    synced = grid.sync(f)
    assert synced.halo_valid["x"] > 0  # meaningful preservation
    g = synced.retag(my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))
    assert g.halo_valid["y"] == 0  # the ghost policy changed
    assert g.halo_valid["x"] == synced.halo_valid["x"]


def test_to_adopts_bc_sibling_of_registered_codomain(walled):
    # b.to(w-space) on a walled grid: the registered bounded
    # interpolation lands on the BC-free Inner sibling (nodal
    # operator outputs are BC-free; owner decision) and the
    # requested Inner(DIRICHLET) tag is adopted via retag. This
    # BC-free-source path is the one the walled model exercises
    # (C8 owner decision: b stays BC-free in declarations).
    grid, mz = walled
    b = grid.create_field(mz.center, init=lambda z: z * (1.0 - z))
    w_space = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    w = b.to(w_space)
    assert w.function_space.bare is w_space
    free = b.to(mz.inner)
    assert jnp.array_equal(w.data, free.data)


def test_to_accepts_a_bc_tagged_source(walled):
    # Center(DIRICHLET) -> Inner(DIRICHLET): unblocked by the
    # bounded-staggering relaxation (nodal interp accepts BC-tagged
    # domains); the BC-free codomain adopts the requested tag
    grid, mz = walled
    tagged = mz.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    b = grid.create_field(tagged, init=lambda z: z * (1.0 - z))
    w_space = mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    w = b.to(w_space)
    assert w.function_space.bare is w_space
    free = grid.create_field(
        mz.center, init=lambda z: z * (1.0 - z)).to(mz.inner)
    assert jnp.array_equal(w.data, free.data)


def test_to_field_target_adopts_the_sibling_tag(walled):
    # the state["b"].to(state["w"]) shape of the seam
    grid, mz = walled
    b = grid.create_field(mz.center, init=lambda z: z)
    w = grid.create_field(mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    out = b.to(w)
    assert out.function_space is w.function_space
    assert jnp.array_equal(out.data, b.to(mz.inner).data)


def test_to_non_sibling_codomain_disagreement_still_raises(walled):
    # Center -> Outer: the registered operator lands on Inner, and
    # Outer free is not a BC sibling of Inner free
    grid, mz = walled
    b = grid.create_field(mz.center)
    with pytest.raises(SpaceMismatchError, match="lands on"):
        b.to(mz.outer)


def test_real_on_lone_complex_factor(grid1d, mx):
    z = grid1d.create_field(mx.center.as_complex(),
                            data=jnp.full(8, 1.0 + 2.0j))
    h = z.real
    assert h.function_space.bare is mx.center
    assert jnp.array_equal(h.data, jnp.full(8, 1.0))


def test_to_nodal_to_constant_has_no_conversion(grid1d, mx):
    a = grid1d.create_field(mx.center)
    with pytest.raises(SpaceMismatchError,
                       match=r"no \.to conversion"):
        a.to(mx.constant)


# ================================================================
#  Variance tagging (stage C2: a pure space-identity claim)
# ================================================================
def test_with_variance_retags_the_space_only(f):
    cov = f.with_variance(Variance.COVARIANT)
    assert cov.function_space.variance is Variance.COVARIANT
    assert cov.function_space.bare.with_variance(None) is (
        f.function_space.bare)
    assert cov.function_space.layout == f.function_space.layout
    assert cov._data is f._data  # no copy, no re-store
    assert cov.halo_valid == f.halo_valid
    assert cov.metadata is f.metadata


def test_with_variance_is_idempotent(f):
    cov = f.with_variance(Variance.COVARIANT)
    assert cov.with_variance(Variance.COVARIANT) is cov
    assert f.with_variance(None) is f


def test_tagged_arithmetic_keeps_the_claim(f, g):
    cov = f.with_variance(Variance.COVARIANT)
    total = cov + cov
    assert total.function_space.variance is Variance.COVARIANT
    scaled = cov * g  # untagged operand adopts the claim
    assert scaled.function_space.variance is Variance.COVARIANT


def test_variance_mixing_raises(f):
    cov = f.with_variance(Variance.COVARIANT)
    con = f.with_variance(Variance.CONTRAVARIANT)
    with pytest.raises(SpaceMismatchError, match="variance mixing"):
        _ = cov + con


def test_diff_preserves_the_variance_claim(grid):
    field = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) + 0 * y)
    cov = field.with_variance(Variance.COVARIANT)
    d = cov.diff("x")
    assert d.function_space.variance is Variance.COVARIANT
    assert jnp.allclose(d.data, field.diff("x").data)


def test_complex_promotion_keeps_the_variance(f):
    cov = f.with_variance(Variance.COVARIANT)
    assert cov.as_complex().function_space.variance is (
        Variance.COVARIANT)


# ================================================================
#  item(): the host read of a fully reduced field
# ================================================================
def test_item_returns_the_single_value(grid):
    field = grid.create_field(init=lambda x, y: 1.0 + 0.0 * x * y)
    total = field.integrate()
    assert total.data.size == 1
    value = total.item()
    assert isinstance(value, float)
    assert value == pytest.approx(2.0)  # the domain volume 1 x 2


def test_item_of_a_complex_field_is_complex(f):
    value = f.as_complex().integrate().item()
    assert isinstance(value, complex)
    assert value.imag == 0.0


def test_item_needs_a_one_dof_field(f):
    with pytest.raises(ValueError, match="one-DOF field"):
        f.item()


# ================================================================
#  mean() on a maps= terrain grid: the physical (Jacobian-weighted)
#  divisor (the physical-integral-default flip)
# ================================================================
def _terrain_grid_2d(n):
    # zp = sigma * H(x): a single-base terrain column (sigma the base
    # axis, x the parameter axis the column Jacobian H varies over)
    mx = IntervalMesh(n, (0.0, 2.0 * jnp.pi), name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    return Grid((mx, ms), mapping=mapping)


def _raw(field, *names):
    result = field
    for name in (names or field.function_space.bare.names):
        result = Integral()[name](result)
    return result


def test_terrain_partial_mean_is_the_physical_column_depth_mean():
    # mean("sigma") divides the physical column integral by the
    # per-column physical depth H(x) = int J dsigma -- a FIELD over x
    grid = _terrain_grid_2d(16)
    space = grid.factors[0].center * grid.factors[1].center
    f = grid.create_field(
        space, init=lambda x, sigma: jnp.cos(sigma) + 0.3 * jnp.sin(x))
    mz = f.mean("sigma")
    jac = grid.metric(space.bare, "dzp_dsigma")
    num = _raw(f * jac, "sigma")
    den = _raw(jac, "sigma")
    expected = num.with_data(num.data / den.data)
    assert jnp.allclose(mz.data, expected.data, atol=1e-13)
    # the result is a per-column FIELD over x, not a scalar (partial
    # mean lands on Constant(sigma) but keeps the x factor)
    assert "x" in mz.function_space.bare.names
    assert mz.data.size > 1
    # (for zp = sigma * H the column Jacobian H(x) is constant along
    # sigma and factors out, so this base-axis mean equals the plain
    # sigma-mean numerically -- the full mean below shows the flip's
    # genuine numeric effect, where H's x-variation does not cancel)


def test_terrain_full_mean_is_the_physical_volume_mean():
    grid = _terrain_grid_2d(16)
    space = grid.factors[0].center * grid.factors[1].center
    f = grid.create_field(
        space, init=lambda x, sigma: 2.0 + jnp.cos(sigma) + jnp.sin(x))
    mean = f.mean().item()
    jac = grid.metric(space.bare, "dzp_dsigma")
    physical_int = float(_raw(f * jac).item())
    physical_vol = float(_raw(jac).item())
    assert mean == pytest.approx(physical_int / physical_vol, rel=1e-12)
    # genuinely different from the plain computational full mean: H(x)'s
    # x-variation weights the average and does not cancel
    ones = grid.create_field(space, data=jnp.ones(space.shape))
    comp = float(_raw(f).item()) / float(_raw(ones).item())
    assert abs(mean - comp) > 1e-3


def test_flat_mean_is_bitwise_the_computational_average(grid):
    # off a mapped grid the mean keeps the plain computational divisor,
    # bitwise the historical path
    field = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) + 0.5 * y)
    space = field.function_space.bare
    total = 1.0
    for name in space.names:
        total = total * float(grid.measure(space, name=name).data.sum())
    expected = _raw(field).data / total
    assert jnp.array_equal(field.mean().data, expected)


def test_terrain_mean_is_differentiable():
    # the double-`where` divisor guard keeps grad through mean("sigma")
    # finite and matches a central finite difference (diff. policy)
    grid = _terrain_grid_2d(8)
    space = grid.factors[0].center * grid.factors[1].center

    def loss(data):
        f = grid.create_field(space, data=data)
        return (f.mean("sigma").data ** 2).sum()

    rng = jnp.asarray(
        np.random.default_rng(0).standard_normal(space.shape))
    g = jax.grad(loss)(rng)
    assert bool(jnp.all(jnp.isfinite(g)))
    eps = 1e-6
    idx = (2, 3)
    plus = rng.at[idx].add(eps)
    minus = rng.at[idx].add(-eps)
    fd = float((loss(plus) - loss(minus)) / (2 * eps))
    assert float(g[idx]) == pytest.approx(fd, rel=1e-4, abs=1e-7)
