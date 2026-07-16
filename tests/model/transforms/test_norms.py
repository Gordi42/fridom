"""Tests for relative_l2, relative_imbalance and assert_idempotent."""
import numpy as np
import pytest

from fridom.model.energy import EnergyMetric
from fridom.model.transforms.norms import (
    _l2_norm,
    assert_idempotent,
    relative_imbalance,
    relative_l2,
)

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


# ================================================================
#  relative_imbalance  (eta(z) = ||(I - P) z|| / ||z||, §10.9)
# ================================================================
def _l2(state):
    return float(_l2_norm(state))


def test_relative_imbalance_equals_the_residual_fraction(state, sig):
    # KeepFirst zeroes every component but the first; the residual is
    # exactly those components, so eta = ||residual|| / ||z||.
    proj = KeepFirst(sig)
    eta = relative_imbalance(state, proj)
    residual = state - proj(state)
    assert eta == pytest.approx(_l2(residual) / _l2(state))
    # the dropped 'v' is the whole residual on this two-component state
    assert np.array_equal(np.asarray(residual["u"].data),
                          np.zeros_like(np.asarray(residual["u"].data)))
    assert np.array_equal(np.asarray(residual["v"].data),
                          np.asarray(state["v"].data))


def test_relative_imbalance_is_zero_on_an_already_projected_state(
        state, sig):
    # a projected state has no residual under its own projection
    proj = KeepFirst(sig)
    assert relative_imbalance(proj(state), proj) == pytest.approx(
        0.0, abs=1e-12)


def test_relative_imbalance_zero_norm_input_returns_zero(state, sig):
    zero = state * 0.0
    assert relative_imbalance(zero, KeepFirst(sig)) == 0.0


def test_relative_imbalance_full_projection_is_zero(state, sig):
    # a component-preserving idempotent projection (Scale-by-1 style)
    # leaves no residual: use the Identity-like KeepFirst on a state
    # whose non-first components already vanish
    z = state.replace(v=state["v"] * 0.0)
    assert relative_imbalance(z, KeepFirst(sig)) == pytest.approx(
        0.0, abs=1e-12)


def test_relative_imbalance_metric_reweights_the_norm(state, sig):
    # a metric with a heavier 'v' weight raises the imbalance (the
    # residual is the 'v' component); the accepted interface is any
    # object exposing norm(state).
    proj = KeepFirst(sig)
    metric = EnergyMetric({"u": 1.0, "v": 4.0})
    eta = relative_imbalance(state, proj, metric=metric)
    residual = state - proj(state)
    assert eta == pytest.approx(
        float(metric.norm(residual)) / float(metric.norm(state)))
    # heavier weight on the residual component => larger than plain l2
    assert eta > relative_imbalance(state, proj)


class _DuckMetric:

    """A minimal norm-provider (the documented metric interface)."""

    def norm(self, state):
        return _l2_norm(state)


def test_relative_imbalance_accepts_any_norm_provider(state, sig):
    # metric= is duck-typed on .norm(state); a provider that reproduces
    # the l2 norm gives the default result exactly
    proj = KeepFirst(sig)
    assert relative_imbalance(state, proj, metric=_DuckMetric()) == (
        pytest.approx(relative_imbalance(state, proj)))
