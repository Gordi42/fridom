"""Tests for the Tier-2 ``Propagator`` preset (wave 7 B).

A small RK4 inertial-oscillation model exercises the run-as-transform
building block: the normative reset/set_state/advance call, the
forward-backward reversibility on the linear model, the cost/repr
step reporting, and the Tier-2 trace guard.
"""
import jax
import jax.numpy as jnp
import pytest

from fridom.model.transforms.errors import TraceError
from fridom.model.transforms.norms import relative_l2
from fridom.model.transforms.propagator import Propagator

from .conftest import RossbyProvider


# ================================================================
#  The normative Tier-2 call (§10.3 law 1)
# ================================================================
def test_propagator_round_trips_advance(toy_model, toy_state):
    prop = Propagator(toy_model, steps=10)
    out = prop(toy_state)
    # the normative spelling, run by hand on the SAME internal model
    model = prop.model
    model.reset()
    model.set_state(toy_state)
    model.advance(10)
    for name in ("u", "v"):
        assert jnp.allclose(out[name].data, model.state[name].data)


def test_propagator_reads_back_prognostic_only(toy_model, toy_state):
    out = Propagator(toy_model, steps=5)(toy_state)
    assert out.component_names == ("u", "v")


def test_propagator_zero_steps_is_identity(toy_model, toy_state):
    out = Propagator(toy_model, steps=0)(toy_state)
    for name in ("u", "v"):
        assert jnp.allclose(out[name].data, toy_state[name].data)


# ================================================================
#  Forward ∘ backward ≈ Identity on the linear model
# ================================================================
def test_forward_then_backward_is_identity(toy_model, toy_state):
    forward = Propagator(toy_model, steps=60)
    backward = Propagator(forward.model, steps=60, backward=True)
    recovered = backward(forward(toy_state))
    assert relative_l2(recovered, toy_state) < 1e-4


def test_backward_flips_the_internal_dt(toy_model):
    backward = Propagator(toy_model, steps=3, backward=True)
    assert backward.is_backward
    assert float(backward.model.parameters["stepper.dt"]) < 0.0


def test_forward_keeps_a_positive_internal_dt(toy_model):
    forward = Propagator(toy_model, steps=3)
    assert not forward.is_backward
    assert float(forward.model.parameters["stepper.dt"]) > 0.0


# ================================================================
#  extra_modules passthrough (the envelope-delivery seam)
# ================================================================
def test_extra_modules_reach_the_internal_variant(toy_model):
    prop = Propagator(toy_model, steps=2,
                      extra_modules=(RossbyProvider(0.3),))
    bound = prop.model.parameters["scaling.rossby"]
    assert float(bound) == pytest.approx(0.3)
    # the passed model never grows the provider (§10.3 law 3)
    assert "scaling.rossby" not in toy_model.parameters


# ================================================================
#  Steps / runlen resolution
# ================================================================
def test_runlen_resolves_to_a_step_count(toy_model):
    # dt = 2e-3; runlen 0.02 -> ceil(0.02/2e-3) = 10 steps
    prop = Propagator(toy_model, runlen=0.02)
    assert prop.steps == 10


def test_steps_and_runlen_are_mutually_exclusive(toy_model):
    with pytest.raises(ValueError, match="exactly one"):
        Propagator(toy_model, steps=3, runlen=0.02)


def test_neither_steps_nor_runlen_errors(toy_model):
    with pytest.raises(ValueError, match="exactly one"):
        Propagator(toy_model)


def test_negative_steps_errors(toy_model):
    with pytest.raises(ValueError, match="non-negative int"):
        Propagator(toy_model, steps=-1)


def test_non_positive_runlen_errors(toy_model):
    with pytest.raises(ValueError, match="positive duration"):
        Propagator(toy_model, runlen=0.0)


# ================================================================
#  Structure, cost, repr
# ================================================================
def test_propagator_is_endo(toy_model):
    prop = Propagator(toy_model, steps=4)
    assert prop.domain == prop.codomain


def test_cost_reports_internal_steps(toy_model):
    prop = Propagator(toy_model, steps=17)
    assert prop.cost().model_steps == 17
    assert not prop.cost().upper_bound


def test_repr_reports_steps_and_direction(toy_model):
    assert repr(Propagator(toy_model, steps=7, backward=True)) == (
        "Propagator(steps=7, backward=True)")


def test_info_reports_model_steps_and_clock(toy_model, toy_state):
    prop = Propagator(toy_model, steps=8)
    _, info = prop.call_with_info(toy_state)
    assert info.model_steps == 8
    # RK4 dt = 2e-3, 8 steps -> t = 0.016
    assert info.elapsed_model_time == pytest.approx(0.016)
    assert info.extra["backward"] is False


# ================================================================
#  The Tier-2 trace guard
# ================================================================
def test_propagator_is_not_traceable(toy_model):
    assert Propagator(toy_model, steps=2).traceable is False


def test_trace_guard_raises_on_a_tracer(toy_model, toy_state):
    prop = Propagator(toy_model, steps=2)
    with pytest.raises(TraceError, match="Tier-2"):
        jax.jit(prop)(toy_state)
