"""Tests for the state-transform algebra (base + nodes + Identity/Shift).

Covers composition (right-to-left, associativity, Identity elision),
the pointwise sum and scalar scaling, ``complement`` (idempotent-
gated), the power operator, the eager signature checks, and the
``T(s) == call_with_info(s)[0]`` info law per node.
"""
import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.model.time_dependent import Ramp
from fridom.framework2.transforms.algebra import (
    Compose,
    Power,
    Scaled,
    Sum,
)
from fridom.framework2.transforms.errors import SignatureMismatchError
from fridom.framework2.transforms.identity import Identity
from fridom.framework2.transforms.info import TransformInfo
from fridom.framework2.transforms.shift import Shift
from fridom.framework2.transforms.signature import StateSignature

from .conftest import KeepFirst, Scale, build_state, make_cross


def _equal(a, b):
    """Whether two states are bitwise-equal componentwise."""
    return a.component_names == b.component_names and all(
        jnp.array_equal(a[name].data, b[name].data)
        for name in a.component_names)


# ================================================================
#  Composition
# ================================================================
def test_compose_is_right_to_left(state, sig):
    double = Scale(sig, 2.0)
    triple = Scale(sig, 3.0)
    composed = double @ triple
    assert _equal(composed(state), double(triple(state)))


def test_compose_flattens_and_reports_parts(sig):
    a, b, c = Scale(sig, 2.0), Scale(sig, 3.0), Scale(sig, 5.0)
    composed = (a @ b) @ c
    assert isinstance(composed, Compose)
    assert composed.parts == (a, b, c)


def test_composition_is_associative(state, sig):
    a, b, c = Scale(sig, 2.0), Scale(sig, 3.0), Scale(sig, 5.0)
    left = (a @ b) @ c
    right = a @ (b @ c)
    assert _equal(left(state), right(state))
    assert left.parts == right.parts


def test_identity_is_elided_in_chains(sig):
    a = Scale(sig, 2.0)
    assert a @ Identity() is a
    assert Identity() @ a is a
    assert isinstance(Identity() @ Identity(), Identity)


def test_compose_signature_mismatch(state, other_grid):
    sig = StateSignature.of_prognostic(state)
    other = StateSignature.of_prognostic(build_state(other_grid))
    a, b = Scale(sig, 2.0), Scale(other, 3.0)
    with pytest.raises(SignatureMismatchError, match="compose"):
        _ = a @ b


# ================================================================
#  Sum, scaling, negation, subtraction
# ================================================================
def test_sum_is_pointwise(state, sig):
    a, b = Scale(sig, 2.0), Scale(sig, 3.0)
    summed = a + b
    assert isinstance(summed, Sum)
    assert _equal(summed(state), a(state) + b(state))


def test_sum_flattens(sig):
    a, b, c = Scale(sig, 2.0), Scale(sig, 3.0), Scale(sig, 5.0)
    assert ((a + b) + c).parts == (a, b, c)


def test_scalar_scaling(state, sig):
    a = Scale(sig, 2.0)
    scaled = 3.0 * a
    assert isinstance(scaled, Scaled)
    assert _equal(scaled(state), 3.0 * a(state))
    assert _equal((a * 3.0)(state), 3.0 * a(state))


def test_negation_and_subtraction(state, sig):
    a, b = Scale(sig, 2.0), Scale(sig, 3.0)
    assert _equal((-a)(state), (-1.0) * a(state))
    assert _equal((a - b)(state), a(state) - b(state))


def test_sum_signature_mismatch(state, other_grid):
    sig = StateSignature.of_prognostic(state)
    other = StateSignature.of_prognostic(build_state(other_grid))
    with pytest.raises(SignatureMismatchError, match="sum"):
        _ = Scale(sig, 2.0) + Scale(other, 3.0)


def test_ramp_coefficient_rejected(sig):
    with pytest.raises(TypeError, match="Ramp"):
        _ = Ramp(0.0, 1.0, period=1.0) * Scale(sig, 2.0)


def test_non_transform_operands_return_notimplemented(sig):
    a = Scale(sig, 2.0)
    assert a.__matmul__(5) is NotImplemented
    assert a.__add__(5) is NotImplemented
    assert a.__mul__("x") is NotImplemented
    assert a.__pow__(1.5) is NotImplemented


# ================================================================
#  Power
# ================================================================
def test_power_zero_is_identity(sig):
    a = Scale(sig, 2.0)
    p0 = a ** 0
    assert isinstance(p0, Identity)
    assert p0.domain == sig


def test_power_one_is_self(sig):
    a = Scale(sig, 2.0)
    assert (a ** 1) is a


def test_power_n_iterates(state, sig):
    a = Scale(sig, 2.0)
    p3 = a ** 3
    assert isinstance(p3, Power)
    assert p3.n == 3
    assert _equal(p3(state), a(a(a(state))))


def test_power_negative_raises(sig):
    with pytest.raises(ValueError, match="inverse"):
        _ = Scale(sig, 2.0) ** -1


def test_power_requires_endo(state, other_grid):
    sig = StateSignature.of_prognostic(state)
    other = StateSignature.of_prognostic(build_state(other_grid))
    with pytest.raises(SignatureMismatchError, match="endo"):
        _ = make_cross(sig, other) ** 2


# ================================================================
#  complement -- gated on the idempotent flag
# ================================================================
def test_complement_of_idempotent(state, sig):
    keep = KeepFirst(sig)
    residual = keep.complement
    # I - P applied: s - P(s)
    expected = state - keep(state)
    assert _equal(residual(state), expected)


def test_complement_rejects_non_idempotent(sig):
    with pytest.raises(TypeError, match="idempotent"):
        _ = Scale(sig, 2.0).complement


# ================================================================
#  The info law: T(s) == call_with_info(s)[0] bitwise
# ================================================================
@pytest.fixture
def nodes(sig):
    a, b = Scale(sig, 2.0), Scale(sig, 3.0)
    keep = KeepFirst(sig)
    return {
        "identity": Identity(),
        "scale": a,
        "compose": a @ b,
        "sum": a + b,
        "scaled": 4.0 * a,
        "power": a ** 3,
        "complement": keep.complement,
    }


def test_info_law_bitwise(nodes, state):
    for name, transform in nodes.items():
        out = transform(state)
        with_info, info = transform.call_with_info(state)
        assert _equal(out, with_info), name
        assert isinstance(info, TransformInfo)


def test_info_tree_mirrors_composition(sig, state):
    a, b = Scale(sig, 2.0), Scale(sig, 3.0)
    _, info = (a @ b).call_with_info(state)
    assert len(info.children) == 2
    labels = [label for label, _ in info.children]
    assert labels[0].startswith("0:")
    assert labels[1].startswith("1:")
    assert isinstance(info[0], TransformInfo)
    assert isinstance(info["1:Scale"], TransformInfo)


def test_info_getitem_unknown_label_raises(sig, state):
    _, info = (Scale(sig, 2.0) + Scale(sig, 3.0)).call_with_info(state)
    with pytest.raises(KeyError, match="no child"):
        _ = info["nope"]


# ================================================================
#  Identity / Shift specifics
# ================================================================
def test_identity_passthrough(state):
    ident = Identity()
    assert _equal(ident(state), state)
    assert ident.idempotent
    assert ident.traceable


def test_identity_repr(sig):
    assert repr(Identity()) == "Identity()"
    assert "StateSignature" in repr(Identity(sig))


def test_shift_adds_state0(state, make_state):
    offset = make_state(u_shift=1.0, v_shift=1.0)
    shifted = Shift(offset)
    out = shifted(state)
    assert _equal(out, state.add(u=offset["u"], v=offset["v"]))
    assert shifted.traceable
    assert not shifted.idempotent


def test_shift_repr_and_state0(state):
    shift = Shift(state)
    assert "Shift" in repr(shift)
    assert shift.state0 is state


# ================================================================
#  Tier-1 trees are jit-able (frozen pytrees)
# ================================================================
def test_tier1_tree_is_jittable(state, sig):
    transform = Scale(sig, 2.0) @ Scale(sig, 3.0)

    @jax.jit
    def apply(s):
        return transform(s)

    out = apply(state)
    assert _equal(out, transform(state))


def test_cost_sums_structurally(sig):
    a, b = Scale(sig, 2.0), Scale(sig, 3.0)
    # all Tier-1: zero model steps, sums structurally
    assert (a @ b).cost().model_steps == 0
    assert (a + b).cost().model_steps == 0
    assert (a ** 3).cost().model_steps == 0


def test_node_reprs_and_properties(sig):
    a, b = Scale(sig, 2.0), Scale(sig, 3.0)
    assert "@" in repr(a @ b)
    assert "+" in repr(a + b)
    scaled = 2.0 * a
    assert scaled.coefficient == 2.0
    assert scaled.inner is a
    assert "*" in repr(scaled)
    assert scaled.cost().model_steps == 0
    power = a ** 2
    assert power.n == 2
    assert power.inner is a
    assert "**" in repr(power)


def test_base_default_repr_and_bad_subtraction(sig):
    cross = make_cross(sig, sig)  # no __repr__ override -> base default
    assert repr(cross) == "Cross()"
    assert cross.__sub__(5) is NotImplemented


def test_node_codomains(sig, state):
    a, b = Scale(sig, 2.0), Scale(sig, 3.0)
    assert (a @ b).codomain == sig
    assert (a + b).codomain == sig
    assert (a ** 2).codomain == sig
    assert Shift(state).codomain == sig


def test_sum_of_polymorphic_identities_is_polymorphic(state):
    poly = Identity() + Identity()
    assert isinstance(poly, Sum)
    assert poly.domain is None
    assert poly.codomain is None
    assert _equal(poly(state), state + state)
