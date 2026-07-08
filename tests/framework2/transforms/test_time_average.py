"""Tests for the Tier-2 ``TimeAverage`` preset (wave 7 B).

A linear inertial oscillation is a pure wave mode: averaging over an
inertial period removes it. The tests exercise the wave removal, the
nested descending-period plan, the ``fr.terms.linear`` parity delta
(Smagorinsky/advection dropped), the period=None inertial default, and
the Tier-2 trace guard.
"""
import jax
import pytest

from fridom.framework2.model import term_predicates as terms
from fridom.framework2.transforms.errors import TraceError
from fridom.framework2.transforms.norms import _l2_norm
from fridom.framework2.transforms.time_average import TimeAverage

from .conftest import (
    F0,
    Coriolis,
    F0Provider,
    NonlinearU,
    make_model,
)

PERIOD = 2 * 3.141592653589793 / F0


# ================================================================
#  Killing a pure wave mode
# ================================================================
def test_time_average_kills_a_pure_wave_mode(toy_model, toy_state):
    before = float(_l2_norm(toy_state))
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=2)
    after = float(_l2_norm(ta(toy_state)))
    # the rotating signal is averaged away over the inertial period
    assert after < before / 100


def test_second_pass_does_not_regrow_the_wave(toy_model, toy_state):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1)
    once = ta(toy_state)
    twice = ta(once)
    # idempotent-ish: the near-balanced state does not grow back
    assert float(_l2_norm(twice)) <= float(_l2_norm(once)) * 2


# ================================================================
#  The period plan (descending vs flat)
# ================================================================
def test_equidistant_gives_descending_periods(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=2)
    # linspace(P/2, P, 3)[1:][::-1] -> descending, longest first
    assert ta.n_steps[0] > ta.n_steps[1]


def test_non_equidistant_gives_equal_periods(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=3,
                     equidistant=False)
    assert len(set(ta.n_steps)) == 1
    assert len(ta.n_steps) == 3


def test_backward_forward_runs_both_legs(toy_model, toy_state):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1,
                     backward_forward=True)
    out = ta(toy_state)
    # both legs executed -> twice the per-pass steps in the cost
    assert ta.cost().model_steps == 2 * sum(ta.n_steps)
    assert float(_l2_norm(out)) < float(_l2_norm(toy_state))


# ================================================================
#  The fr.terms.linear parity delta (drops the nonlinear term)
# ================================================================
def test_default_filter_drops_the_nonlinear_term():
    model = make_model(modules=(Coriolis(), NonlinearU()))
    ta = TimeAverage(model, period=PERIOD, n_ave=1)
    keys = {e.key for e in
            ta._forward._artifacts.schedule.kind_entries(None)}
    assert keys == {"Coriolis/cor"}  # NonlinearU/adv dropped


def test_explicit_filter_is_honored():
    model = make_model(modules=(Coriolis(), NonlinearU()))
    # an explicit filter that KEEPS the nonlinear term (unlike the
    # linear default) — both terms survive
    ta = TimeAverage(
        model, period=PERIOD, n_ave=1,
        filter=terms.named("Coriolis/cor", "NonlinearU/adv"))
    keys = {e.key for e in
            ta._forward._artifacts.schedule.kind_entries(None)}
    assert keys == {"Coriolis/cor", "NonlinearU/adv"}


# ================================================================
#  period=None -> the inertial period
# ================================================================
def test_period_none_reads_the_inertial_period(toy_model):
    ta_default = TimeAverage(toy_model, n_ave=1)
    model = make_model()
    ta_explicit = TimeAverage(model, period=PERIOD, n_ave=1)
    assert ta_default.n_steps == ta_explicit.n_steps


def test_period_none_without_coriolis_errors():
    model = make_model(modules=(Coriolis(),))  # no F0Provider
    with pytest.raises(ValueError, match=r"no 'coriolis\.f0'"):
        TimeAverage(model, n_ave=1)


def test_period_none_with_zero_f0_errors():
    model = make_model(modules=(Coriolis(), F0Provider(f0=0.0)))
    with pytest.raises(ValueError, match=r"coriolis\.f0 is zero"):
        TimeAverage(model, n_ave=1)


# ================================================================
#  Construction guards
# ================================================================
def test_non_positive_period_errors(toy_model):
    with pytest.raises(ValueError, match="period must be positive"):
        TimeAverage(toy_model, period=0.0)


def test_invalid_n_ave_errors(toy_model):
    with pytest.raises(ValueError, match="n_ave must be a positive"):
        TimeAverage(toy_model, period=PERIOD, n_ave=0)


# ================================================================
#  Structure, cost, repr, trace guard
# ================================================================
def test_time_average_is_endo(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1)
    assert ta.domain == ta.codomain


def test_cost_is_the_total_step_count(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=2)
    assert ta.cost().model_steps == sum(ta.n_steps)


def test_repr_reports_the_plan(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1)
    assert "TimeAverage(n_steps=" in repr(ta)


def test_trace_guard_raises_on_a_tracer(toy_model, toy_state):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1)
    assert ta.traceable is False
    with pytest.raises(TraceError, match="Tier-2"):
        jax.jit(ta)(toy_state)
