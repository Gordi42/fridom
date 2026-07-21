"""Tests for the Tier-2 ``OptimalBalance`` preset (wave 7 B).

The ramped fixed-point cycle ``forward @ base @ backward`` with an
injected (test-double) base projection. The machinery under test: the
FixedPoint iteration converges, the base-coordinate invariant holds
exactly (the exchange is a projector by construction), the two ramped
legs use ``Ramp.reversed()`` / a flipped dt, and the Tier-2 trace
guard fires. No dependence on the wave-7 C projections.
"""
import jax
import pytest

from fridom.model import term_predicates as terms
from fridom.model.time_dependent import Ramp
from fridom.model.transforms.errors import TraceError
from fridom.model.transforms.norms import relative_l2
from fridom.model.transforms.optimal_balance import OptimalBalance
from fridom.model.transforms.propagator import Propagator
from fridom.model.transforms.signature import StateSignature

from .conftest import (
    Coriolis,
    KeepFirst,
    NonlinearU,
    RossbyProvider,
    make_model,
    set_wave_ic,
)

RAMP = 0.04  # dt = 2e-3 -> 20 ramp steps per leg


def _base(model):
    """Return an idempotent base projector double (keeps u, zeros v)."""
    return KeepFirst(StateSignature.of_prognostic(model))


# ================================================================
#  Convergence of the fixed-point iteration
# ================================================================
def test_optimal_balance_converges(toy_model, toy_state):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP,
                        max_it=5, tol=1e-10)
    _, info = ob.call_with_info(toy_state)
    assert info.iterations >= 1
    assert len(info.errors) >= 1
    assert info.errors[-1] <= info.errors[0]
    assert info.errors[-1] < 1e-3


def test_stopped_by_is_reported(toy_model, toy_state):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP,
                        max_it=5, tol=1e-10)
    _, info = ob.call_with_info(toy_state)
    assert info.extra["stopped_by"] in ("tol", "max_it", "divergence")


# ================================================================
#  The base-coordinate invariant (exact by construction)
# ================================================================
def test_base_coordinate_is_preserved(toy_model, toy_state):
    base = _base(toy_model)
    ob = OptimalBalance(toy_model, base, ramp_period=RAMP, max_it=4,
                        tol=1e-12, update_base_point=False)
    result = ob(toy_state)
    # z_new = rc - base(rc) + base(z0)  ->  base(z_new) == base(z0)
    assert relative_l2(base(result), base(toy_state)) < 1e-8


def test_update_base_point_true_runs(toy_model, toy_state):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP,
                        max_it=3, update_base_point=True)
    result = ob(toy_state)
    assert result.component_names == ("u", "v")


# ================================================================
#  The ramped legs (forward @ base @ backward, Ramp.reversed())
# ================================================================
def test_ramp_cycle_is_forward_base_backward(toy_model):
    base = _base(toy_model)
    ob = OptimalBalance(toy_model, base, ramp_period=RAMP)
    parts = ob.ramp_cycle.parts
    assert parts[0] is ob.forward
    assert parts[1] is base
    assert parts[2] is ob.backward


def test_legs_have_opposite_dt_signs(toy_model):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP)
    assert not ob.forward.is_backward
    assert ob.backward.is_backward
    assert float(ob.forward.model.parameters["stepper.dt"]) > 0.0
    assert float(ob.backward.model.parameters["stepper.dt"]) < 0.0


def test_ramp_steps_from_period(toy_model):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP)
    assert ob.forward.steps == 20  # round(0.04 / 2e-3)
    assert ob.backward.steps == 20


# ================================================================
#  The scaling.nonlinearity ramp branch (Ramp up / Ramp.reversed() down)
# ================================================================
@pytest.mark.parametrize("nominal", [1.0, 0.1])
def test_rossby_model_applies_the_ramp(nominal):
    model = make_model(modules=(Coriolis(), RossbyProvider(nominal)))
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=2)
    # the forward leg's scaling.nonlinearity ramps from 0 to the MODEL's
    # nominal value (user parameter choices are preserved); the
    # backward leg's is its time-domain reversal (spanning [-T, 0])
    fwd_rossby = ob.forward.model.parameters["scaling.nonlinearity"]
    bwd_rossby = ob.backward.model.parameters["scaling.nonlinearity"]
    assert float(fwd_rossby.at_time(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert float(fwd_rossby.at_time(RAMP)) == pytest.approx(
        nominal, abs=1e-9)
    # reversed leg: nonlinear (nominal) at clock 0, linear (0) at -T
    assert float(bwd_rossby.at_time(0.0)) == pytest.approx(
        nominal, abs=1e-9)
    assert float(bwd_rossby.at_time(-RAMP)) == pytest.approx(0.0, abs=1e-9)


def test_time_dependent_rossby_is_rejected():
    model = make_model(modules=(Coriolis(), RossbyProvider()))
    ramped = model.variant(
        updates={"scaling.nonlinearity": Ramp(0.0, 1.0, period=1.0)})
    with pytest.raises(TypeError, match=r"constant 'scaling\.nonlinearity'"):
        OptimalBalance(ramped, _base(ramped), ramp_period=RAMP)


def test_rossby_model_balances():
    model = make_model(modules=(Coriolis(), RossbyProvider()))
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=2)
    result = ob(set_wave_ic(model))
    assert result.component_names == ("u", "v")


# ================================================================
#  backward_filter threading
# ================================================================
def test_backward_filter_is_threaded():
    model = make_model(modules=(Coriolis(), NonlinearU()))
    ob = OptimalBalance(
        model, _base(model), ramp_period=RAMP,
        filter=None, backward_filter=terms.linear)
    bwd_keys = {e.key for e in
                ob.backward.model._artifacts.schedule.kind_entries(None)}
    assert bwd_keys == {"Coriolis/cor"}  # nonlinear dropped backward


# ================================================================
#  Cost, repr, structure, trace guard
# ================================================================
def test_cost_reports_internal_steps(toy_model):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP,
                        max_it=3)
    cost = ob.cost()
    assert cost.model_steps == 3 * (20 + 20)
    assert cost.upper_bound


def test_info_reports_executed_model_steps(toy_model, toy_state):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP,
                        max_it=2, tol=0.0)
    _, info = ob.call_with_info(toy_state)
    # tol=0 forces exactly max_it iterations, each 20 + 20 steps
    assert info.model_steps == 2 * (20 + 20)


def test_optimal_balance_is_endo(toy_model):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP)
    assert ob.domain == ob.codomain


def test_forward_is_a_propagator(toy_model):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP)
    assert isinstance(ob.forward, Propagator)
    assert isinstance(ob.backward, Propagator)


def test_base_property_exposes_the_injection(toy_model):
    base = _base(toy_model)
    ob = OptimalBalance(toy_model, base, ramp_period=RAMP)
    assert ob.base is base


def test_repr_reports_ramp_and_iterations(toy_model):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP,
                        max_it=4)
    text = repr(ob)
    assert "OptimalBalance(" in text
    assert "ramp_steps=20" in text
    assert "max_it=4" in text


def test_trace_guard_raises_on_a_tracer(toy_model, toy_state):
    ob = OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP)
    assert ob.traceable is False
    with pytest.raises(TraceError, match="Tier-2"):
        jax.jit(ob)(toy_state)
