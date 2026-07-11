"""Tests for relative_l2 and assert_idempotent."""
import pytest

from fridom.framework2.transforms.norms import assert_idempotent, relative_l2

from .conftest import KeepFirst, Scale


def test_relative_l2_of_identical_states_is_zero(state):
    assert relative_l2(state, state) == pytest.approx(0.0)


def test_relative_l2_of_zero_states_is_zero(state):
    zero = state * 0.0
    assert relative_l2(zero, zero) == 0.0


def test_relative_l2_is_positive_for_distinct(state, make_state):
    other = make_state(u_shift=5.0, v_shift=5.0)
    assert relative_l2(state, other) > 0.0


def test_relative_l2_symmetric(state, make_state):
    other = make_state(u_shift=2.0)
    assert relative_l2(state, other) == pytest.approx(
        relative_l2(other, state))


def test_assert_idempotent_passes_on_projector(state, sig):
    assert_idempotent(KeepFirst(sig), state)  # no raise


def test_assert_idempotent_fails_on_non_idempotent(state, sig):
    with pytest.raises(AssertionError, match="not idempotent"):
        assert_idempotent(Scale(sig, 2.0), state)


def test_assert_idempotent_custom_tol(state, sig):
    # a huge tolerance masks the non-idempotency
    assert_idempotent(Scale(sig, 2.0), state, tol=10.0)
