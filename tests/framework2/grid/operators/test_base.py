"""Tests for the Operator base hierarchy (base.py, Wave 2)."""
import dataclasses

import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    BinaryOperator,
    Block,
    EigenbasisError,
    Operator,
    OperatorRequirements,
    SeparableOperator,
    UnaryOperator,
)


# ================================================================
#  OperatorRequirements
# ================================================================
def test_requirements_defaults():
    req = OperatorRequirements()
    assert req.halo == 0
    assert req.layout == "any"
    assert req.collective is False


def test_requirements_frozen():
    req = OperatorRequirements(halo=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.halo = 3


def test_requirements_value_equality():
    assert OperatorRequirements(halo=1) == OperatorRequirements(halo=1)
    assert OperatorRequirements(halo=1) != OperatorRequirements(halo=2)


# ================================================================
#  EigenbasisError and the optional-capability default
# ================================================================
def test_eigenbasis_error_is_type_error():
    assert issubclass(EigenbasisError, TypeError)


def test_base_eigenvalues_raises(mx, stagger):
    with pytest.raises(EigenbasisError, match="no eigenvalues"):
        stagger.eigenvalues(None, mx.center)


# ================================================================
#  ABC structure
# ================================================================
@pytest.mark.parametrize("cls", [
    pytest.param(Operator, id="Operator"),
    pytest.param(UnaryOperator, id="UnaryOperator"),
    pytest.param(BinaryOperator, id="BinaryOperator"),
    pytest.param(SeparableOperator, id="SeparableOperator"),
])
def test_abcs_not_instantiable(cls):
    with pytest.raises(TypeError):
        cls()


def test_dispatch_kind_default_and_override(stagger, whole_cls):
    assert Operator.dispatch_kind is None
    assert whole_cls().dispatch_kind is None
    assert stagger.dispatch_kind == "diff"


def test_default_requirements(mx, whole_cls):
    assert whole_cls().requirements(mx.center) == OperatorRequirements()


# ================================================================
#  Identity hashing (cluster invariant)
# ================================================================
def test_identity_equality_and_hash(stagger_cls):
    a, b = stagger_cls(), stagger_cls()
    assert a == a  # noqa: PLR0124 — tests __eq__ identity
    assert a != b  # structurally equal but distinct instances
    assert hash(a) == id(a)


# ================================================================
#  Base dunder failure modes
# ================================================================
def test_fixed_signature_getitem_raises(whole_cls):
    with pytest.raises(TypeError, match="fixed signature"):
        whole_cls()["x"]


def test_tuple_on_the_left_of_matmul_raises(stagger):
    with pytest.raises(TypeError, match="no meaning"):
        (stagger, stagger) @ stagger


def test_scalar_on_the_left_of_matmul_raises(stagger):
    with pytest.raises(TypeError, match="must be an operator"):
        2 @ stagger


def test_matmul_tuple_operand_designed_for(stagger):
    with pytest.raises(NotImplementedError, match="designed-for"):
        stagger @ (stagger, stagger)


def test_operator_times_operator_unsupported(stagger, keep_cls):
    with pytest.raises(TypeError):
        stagger * keep_cls()


def test_pow_argument_validation(stagger):
    with pytest.raises(TypeError):
        stagger ** 1.5
    with pytest.raises(TypeError):
        stagger ** True
    with pytest.raises(ValueError, match="non-negative"):
        stagger ** -1


def test_bind_requires_a_string_axis(stagger):
    with pytest.raises(TypeError, match="coordinate name"):
        stagger[0]


# ================================================================
#  Separable axis binding surface
# ================================================================
def test_bound_axis_default_is_none(stagger):
    assert stagger.bound_axis is None
    assert stagger.unbound is stagger


def test_binding_sets_axis_and_keeps_original(stagger):
    bound = stagger["x"]
    assert bound.bound_axis == "x"
    assert bound is not stagger
    assert bound.unbound is stagger
    assert stagger.bound_axis is None


def test_rebinding_raises(stagger):
    with pytest.raises(TypeError, match="already bound"):
        stagger["x"]["y"]


def test_bound_operator_keeps_static_parameters(stagger_cls):
    op = stagger_cls(halo=3)
    assert op["x"].halo == 3


# ================================================================
#  Block is a declared, designed-for stub
# ================================================================
def test_block_is_designed_for(whole_cls):
    with pytest.raises(NotImplementedError, match="designed-for"):
        Block([[whole_cls()]])


# ================================================================
#  Codomain resolvers reject unsupported domains
# ================================================================
def test_unsupported_domain_raises_space_mismatch(mx, stagger):
    with pytest.raises(SpaceMismatchError, match="unsupported"):
        stagger.codomain(mx.cell_avg)
