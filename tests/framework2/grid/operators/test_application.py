"""Tests for the shared application path (stand-in field objects)."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.errors import (
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.framework2.grid.operators.base import (
    BinaryOperator,
    Dispatched,
    Identity,
    Zero,
)
from fridom.framework2.grid.operators.registry import (
    DispatchError,
    OperatorRegistry,
)


# ================================================================
#  Unary template: codomain, layout, sync
# ================================================================
def test_apply_1d(mx, field_1d, stagger):
    result = stagger(field_1d)
    assert result.function_space is mx.right
    assert jnp.array_equal(result.data, field_1d.data + 1.0)


def test_every_application_returns_a_synced_field(
        grid, field_1d, stagger):
    result = stagger(field_1d)
    assert grid.sync_log[-1] is result


def test_apply_bound_on_product(mx, my, field_2d, stagger):
    result = stagger["y"](field_2d)
    assert result.function_space is (mx.center * my.right)
    assert jnp.array_equal(result.data, field_2d.data + 1.0)


def test_apply_unbound_on_product_raises(field_2d, stagger):
    with pytest.raises(ValueError, match="ambiguous"):
        stagger(field_2d)


def test_apply_unsupported_domain_raises(
        grid, mx, field_cls, stagger):
    field = field_cls(grid, mx.cell_avg, jnp.zeros(8))
    with pytest.raises(SpaceMismatchError, match="unsupported"):
        stagger(field)


def test_layout_is_reattached(grid, mx, my, field_cls, stagger):
    layout = Layout({})
    space = (mx.center * my.center).with_layout(layout)
    field = field_cls(grid, space, jnp.zeros((8, 4)))
    result = stagger["x"](field)
    expected = (mx.right * my.center).with_layout(layout)
    assert result.function_space is expected
    assert result.function_space.layout is layout


def test_constant_factor_application_is_identity(
        grid, mx, my, field_cls, stagger):
    space = mx.center * my.constant
    field = field_cls(grid, space, jnp.zeros((8, 1)))
    result = stagger["y"](field)
    assert result.function_space is space
    assert jnp.array_equal(result.data, field.data)


def test_kernel_returning_wrong_space_is_caught(
        field_1d, stagger_cls):
    class Broken(stagger_cls):

        def _apply_factor(self, f, axis):  # noqa: ARG002
            return f  # claims Center -> Right but keeps the space

    with pytest.raises(SpaceMismatchError, match="kernel returned"):
        Broken()(field_1d)


# ================================================================
#  Algebra objects through the application path
# ================================================================
def test_separable_chain_applies_right_to_left(
        mx, field_1d, stagger_cls):
    a, b = stagger_cls(shift=1.0), stagger_cls(shift=10.0)
    result = (a @ b)(field_1d)
    assert result.function_space is mx.center
    assert jnp.array_equal(result.data, field_1d.data + 11.0)


def test_bound_separable_chain_on_product(
        mx, my, field_2d, stagger_cls):
    a = stagger_cls()
    result = (a @ a)["y"](field_2d)
    assert result.function_space is (mx.center * my.center)
    assert jnp.array_equal(result.data, field_2d.data + 2.0)


def test_whole_space_composite_applies_factorwise(
        grid, mx, my, field_2d, stagger_cls):
    a, b = stagger_cls(shift=1.0), stagger_cls(shift=10.0)
    result = (a["x"] @ b["y"])(field_2d)
    assert result.function_space is (mx.right * my.right)
    assert jnp.array_equal(result.data, field_2d.data + 11.0)
    # inner applications sync too (iteration-1 contract)
    assert len(grid.sync_log) == 3


def test_identity_application(field_1d):
    assert Identity()(field_1d) is field_1d


def test_zero_application(field_1d):
    result = Zero()(field_1d)
    assert result.function_space is field_1d.function_space
    assert jnp.array_equal(result.data, jnp.zeros_like(field_1d.data))


def test_operator_sum_application(mx, field_1d, stagger_cls):
    a, b = stagger_cls(shift=1.0), stagger_cls(shift=10.0)
    result = (a + b)(field_1d)
    assert result.function_space is mx.right
    assert jnp.array_equal(
        result.data, 2 * field_1d.data + 11.0)


def test_operator_sum_validates_the_signature(
        field_1d, stagger, keep_cls):
    with pytest.raises(SpaceMismatchError, match="share a signature"):
        (stagger + keep_cls())(field_1d)


def test_scaled_operator_application(mx, field_1d, stagger):
    result = (2.0 * stagger)(field_1d)
    assert result.function_space is mx.right
    assert jnp.array_equal(result.data, 2.0 * (field_1d.data + 1.0))


# ================================================================
#  Binary template
# ================================================================
class Pointwise(BinaryOperator):

    """Strict same-space pointwise product (test operator)."""

    def codomain(self, domain_a, domain_b):
        if domain_a is domain_b:
            return domain_a
        raise SpaceMismatchError(
            f"pointwise operands differ: {domain_a!r} vs "
            f"{domain_b!r}", left=domain_a, right=domain_b)

    def _apply(self, f, g):
        return f.with_data(f.data * g.data)


def test_binary_application(grid, mx, field_1d):
    result = Pointwise()(field_1d, field_1d)
    assert result.function_space is mx.center
    assert jnp.array_equal(result.data, field_1d.data ** 2)
    assert grid.sync_log[-1] is result


def test_binary_space_mismatch_raises(grid, mx, field_1d, field_cls):
    other = field_cls(grid, mx.right, jnp.zeros(8))
    with pytest.raises(SpaceMismatchError, match="differ"):
        Pointwise()(field_1d, other)


def test_binary_grid_mismatch_raises(
        grid_cls, mx, field_1d, field_cls):
    other = field_cls(grid_cls(), mx.center, jnp.zeros(8))
    with pytest.raises(GridMismatchError, match="same grid"):
        Pointwise()(field_1d, other)


def test_binary_layout_mismatch_raises(
        grid, mx, field_1d, field_cls):
    laid = field_cls(grid, mx.center.with_layout(Layout({})),
                     jnp.zeros(8))
    with pytest.raises(SpaceMismatchError, match="layouts differ"):
        Pointwise()(field_1d, laid)


def test_strict_binary_rejects_extra_operands(field_1d):
    with pytest.raises(TypeError):
        Pointwise()(field_1d, field_1d, field_1d)


# ================================================================
#  Dispatched as the standalone user verb
# ================================================================
@pytest.fixture
def dispatch_grid(grid, mx, my, stagger):
    grid.dispatch = OperatorRegistry({
        ("diff", mx.center): stagger,
        ("diff", my.center): stagger,
    })
    return grid


@pytest.mark.usefixtures("dispatch_grid")
def test_dispatched_verb_on_1d_field(mx, field_1d):
    result = Dispatched("diff")(field_1d)
    assert result.function_space is mx.right
    assert jnp.array_equal(result.data, field_1d.data + 1.0)


@pytest.mark.usefixtures("dispatch_grid")
def test_dispatched_verb_bound_on_product(mx, my, field_2d):
    result = Dispatched("diff")["y"](field_2d)
    assert result.function_space is (mx.center * my.right)


@pytest.mark.usefixtures("dispatch_grid")
def test_dispatched_verb_unbound_on_product_raises(field_2d):
    with pytest.raises(ValueError, match="bind explicitly"):
        Dispatched("diff")(field_2d)


@pytest.mark.usefixtures("dispatch_grid")
def test_dispatched_unknown_kind_raises(field_1d):
    with pytest.raises(DispatchError, match="no operator registered"):
        Dispatched("nope")(field_1d)


def test_dispatched_codomain_is_unresolved(mx):
    with pytest.raises(DispatchError, match="unresolved"):
        Dispatched("diff").codomain(mx.center)


def test_unresolved_hole_in_chain_fails_at_application(
        field_1d, stagger):
    chain = stagger @ Dispatched("reconstruct")
    with pytest.raises(DispatchError, match="unresolved"):
        chain(field_1d)
