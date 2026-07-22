"""Tests for the Tier-2 ``OptimalBalance`` preset (wave 7 B).

The ramped fixed-point cycle ``forward @ base @ backward`` with an
injected (test-double) base projection. OB's forward leg is
``AdiabaticRamping(model, envelope=True, ...)`` (§C): it ramps the
**nonlinear terms as a whole** through ``"ramping.envelope"`` and
never touches a scaling parameter — so OB balances dimensional
models (no ``scaling.nonlinearity`` bound) and accepts Ramp-valued
scaling. The machinery under test: the FixedPoint iteration
converges, the base-coordinate invariant holds exactly (the exchange
is a projector by construction), the two ramped legs carry the
envelope Ramp / its reversal and a flipped dt, a purely linear model
is refused (the taught empty-match error), and the Tier-2 trace
guard fires. No dependence on the wave-7 C projections.

The shared conftest toy (Coriolis + F0Provider) is purely linear, so
this file builds its own model with the conftest ``NonlinearU``
term (weak: 0.1*u^2) — OB needs something to envelope.
"""
from functools import partial

import jax
import jax.numpy as jnp
import pytest

import fridom as fr
from fridom.framework.utils import dtype_real, jaxify
from fridom.model import params
from fridom.model import term_predicates as terms
from fridom.model.errors import AssemblyError
from fridom.model.model import Model as FrModel
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.time_dependent import Ramp
from fridom.model.time_steppers.runge_kutta import (
    ExplicitRungeKutta,
    tableaus,
)
from fridom.model.transforms.errors import TraceError
from fridom.model.transforms.norms import relative_l2
from fridom.model.transforms.optimal_balance import OptimalBalance
from fridom.model.transforms.propagator import Propagator
from fridom.model.transforms.signature import StateSignature

from .conftest import (
    Coriolis,
    F0Provider,
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


def nonlinear_model():
    """Return the toy rotation model plus the weak nonlinear term."""
    return make_model(modules=(Coriolis(), NonlinearU(),
                               F0Provider()))


@pytest.fixture
def model():
    return nonlinear_model()


@pytest.fixture
def state(model):
    return set_wave_ic(model)


# ================================================================
#  Convergence of the fixed-point iteration
# ================================================================
def test_optimal_balance_converges(model, state):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=5, tol=1e-10)
    _, info = ob.call_with_info(state)
    assert info.iterations >= 1
    assert len(info.errors) >= 1
    assert info.errors[-1] <= info.errors[0]
    assert info.errors[-1] < 1e-3


def test_stopped_by_is_reported(model, state):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=5, tol=1e-10)
    _, info = ob.call_with_info(state)
    assert info.extra["stopped_by"] in ("tol", "max_it", "divergence")


def test_balances_without_any_scaling_parameter(model, state):
    # the §C headline: a DIMENSIONAL model (no scaling bound)
    # is genuinely balanced — the envelope ramps the terms themselves
    assert params.SCALING_NONLINEARITY not in model.parameters
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=4, tol=1e-12)
    _, info = ob.call_with_info(state)
    assert info.errors[-1] <= info.errors[0]
    assert info.errors[-1] < 1e-3


# ================================================================
#  The base-coordinate invariant (exact by construction)
# ================================================================
def test_base_coordinate_is_preserved(model, state):
    base = _base(model)
    ob = OptimalBalance(model, base, ramp_period=RAMP, max_it=4,
                        tol=1e-12, update_base_point=False)
    result = ob(state)
    # z_new = rc - base(rc) + base(z0)  ->  base(z_new) == base(z0)
    assert relative_l2(base(result), base(state)) < 1e-8


def test_update_base_point_true_runs(model, state):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=3, update_base_point=True)
    result = ob(state)
    assert result.component_names == ("u", "v")


# ================================================================
#  The ramped legs (forward @ base @ backward, Ramp.reversed())
# ================================================================
def test_ramp_cycle_is_forward_base_backward(model):
    base = _base(model)
    ob = OptimalBalance(model, base, ramp_period=RAMP)
    parts = ob.ramp_cycle.parts
    assert parts[0] is ob.forward
    assert parts[1] is base
    assert parts[2] is ob.backward


def test_legs_have_opposite_dt_signs(model):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP)
    assert not ob.forward.is_backward
    assert ob.backward.is_backward
    assert float(ob.forward.model.parameters["stepper.dt"]) > 0.0
    assert float(ob.backward.model.parameters["stepper.dt"]) < 0.0


def test_ramp_steps_from_period(model):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP)
    assert ob.forward.steps == 20  # round(0.04 / 2e-3)
    assert ob.backward.steps == 20


# ================================================================
#  The envelope ramp (rho up / Ramp.reversed() down)
# ================================================================
def test_legs_carry_the_envelope_ramp(model):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=2)
    # the forward leg's ramping.envelope ramps 0 -> 1; the backward
    # leg's is its time-domain reversal (spanning [-T, 0])
    fwd_rho = ob.forward.model.parameters["ramping.envelope"]
    bwd_rho = ob.backward.model.parameters["ramping.envelope"]
    assert isinstance(fwd_rho, Ramp)
    assert float(fwd_rho.at_time(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert float(fwd_rho.at_time(RAMP)) == pytest.approx(1.0,
                                                         abs=1e-9)
    # reversed leg: nonlinear (1) at clock 0, linear (0) at -T
    assert float(bwd_rho.at_time(0.0)) == pytest.approx(1.0, abs=1e-9)
    assert float(bwd_rho.at_time(-RAMP)) == pytest.approx(0.0,
                                                          abs=1e-9)


def test_forward_leg_marks_the_nonlinear_term_enveloped(model):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP)
    flags = {e.key: e.enveloped for e in
             ob.forward.model._artifacts.schedule.kind_entries(None)}
    assert flags == {"Coriolis/cor": False, "NonlinearU/adv": True}


# ================================================================
#  Scaling parameters are never touched (§C acceptance)
# ================================================================
@pytest.mark.parametrize("nominal", [1.0, 0.1])
def test_constant_scaling_stays_constant_through_the_legs(nominal):
    model = make_model(modules=(Coriolis(), NonlinearU(),
                                RossbyProvider(nominal)))
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=2)
    for leg in (ob.forward, ob.backward):
        eps = leg.model.parameters[params.SCALING_NONLINEARITY]
        assert not isinstance(eps, Ramp)
        assert float(eps) == pytest.approx(nominal)


def test_time_dependent_scaling_is_accepted():
    # replaces the old Ramp-eps TypeError: OB no longer floats eps as
    # a ramp target, so a Ramp-valued scaling parameter is legal and
    # rides the legs untouched — the envelope carries the ramp
    model = make_model(modules=(Coriolis(), NonlinearU(),
                                RossbyProvider()))
    eps_ramp = Ramp(0.0, 1.0, period=1.0)
    ramped = model.variant(
        updates={params.SCALING_NONLINEARITY: eps_ramp})
    ob = OptimalBalance(ramped, _base(ramped), ramp_period=RAMP)
    for leg in (ob.forward, ob.backward):
        eps = leg.model.parameters[params.SCALING_NONLINEARITY]
        assert isinstance(eps, Ramp)
        assert float(eps.at_time(1.0)) == pytest.approx(1.0)
        assert isinstance(
            leg.model.parameters["ramping.envelope"], Ramp)


def test_scaled_model_balances():
    model = make_model(modules=(Coriolis(), NonlinearU(),
                                RossbyProvider()))
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=2)
    result = ob(set_wave_ic(model))
    assert result.component_names == ("u", "v")


# ================================================================
#  A purely linear model is refused (the taught empty-match error)
# ================================================================
def test_purely_linear_model_is_refused(toy_model):
    # was: a silent no-ramp cycle; now the envelope's empty-match
    # refusal surfaces at OB construction
    with pytest.raises(AssemblyError, match="matches no collected"):
        OptimalBalance(toy_model, _base(toy_model), ramp_period=RAMP)


# ================================================================
#  backward_filter threading
# ================================================================
def test_backward_filter_is_threaded():
    model = make_model(modules=(Coriolis(), NonlinearU()))
    # the linear-only backward leg keeps an inert envelope: the
    # empty-match refusal downgrades to a warning under the filter
    with pytest.warns(UserWarning, match="matches no collected"):
        ob = OptimalBalance(
            model, _base(model), ramp_period=RAMP,
            filter=None, backward_filter=terms.linear)
    bwd_keys = {e.key for e in
                ob.backward.model._artifacts.schedule.kind_entries(None)}
    assert bwd_keys == {"Coriolis/cor"}  # nonlinear dropped backward


# ================================================================
#  Cost, repr, structure, trace guard
# ================================================================
def test_cost_reports_internal_steps(model):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=3)
    cost = ob.cost()
    assert cost.model_steps == 3 * (20 + 20)
    assert cost.upper_bound


def test_info_reports_executed_model_steps(model, state):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=2, tol=0.0)
    _, info = ob.call_with_info(state)
    # tol=0 forces exactly max_it iterations, each 20 + 20 steps
    assert info.model_steps == 2 * (20 + 20)


def test_optimal_balance_is_endo(model):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP)
    assert ob.domain == ob.codomain


def test_forward_is_a_propagator(model):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP)
    assert isinstance(ob.forward, Propagator)
    assert isinstance(ob.backward, Propagator)


def test_base_property_exposes_the_injection(model):
    base = _base(model)
    ob = OptimalBalance(model, base, ramp_period=RAMP)
    assert ob.base is base


def test_repr_reports_ramp_and_iterations(model):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP,
                        max_it=4)
    text = repr(ob)
    assert "OptimalBalance(" in text
    assert "ramp_steps=20" in text
    assert "max_it=4" in text


def test_trace_guard_raises_on_a_tracer(model, state):
    ob = OptimalBalance(model, _base(model), ramp_period=RAMP)
    assert ob.traceable is False
    with pytest.raises(TraceError, match="Tier-2"):
        jax.jit(ob)(state)


# ================================================================
#  Mechanism-scaled (nondimensional) models balance too (§C)
# ================================================================
def test_mechanism_scaled_model_builds_and_ramps_the_envelope():
    # replaces the interim alias-row guard (deleted by §C): under a
    # mechanism scaling the epsilon row aliases the mechanism
    # module's own nonlinearity leaf — OB no longer ramps that row,
    # so the model is accepted; the envelope carries the ramp and
    # epsilon stays the constant regime number through both legs
    @partial(jaxify, dynamic=("froude_number",))
    class MechProvider(Module):
        scaling_mechanism = "gravity_wave"
        nonlinearity_attr = "froude_number"
        scaling_variant = "nondimensional"
        field_declarations = ()
        parameter_declarations = (
            ParameterDeclaration("toy.froude", attr="froude_number",
                                 units="1"),)

        def __init__(self, froude_number=0.2):
            self.froude_number = jnp.asarray(froude_number,
                                             dtype=dtype_real())

    plain = make_model()
    nondim = FrModel(
        grid=plain.grid,
        modules=(Coriolis(), NonlinearU(), MechProvider()),
        time_stepper=ExplicitRungeKutta(2e-3, tableau=tableaus.RK4),
        scaling=fr.scaling.GravityWave())
    ob = OptimalBalance(nondim, _base(nondim), ramp_period=RAMP)
    for leg in (ob.forward, ob.backward):
        rho = leg.model.parameters["ramping.envelope"]
        assert isinstance(rho, Ramp)
        eps = leg.model.parameters[params.SCALING_NONLINEARITY]
        assert not isinstance(eps, Ramp)
        assert float(eps) == pytest.approx(0.2)
    fwd_rho = ob.forward.model.parameters["ramping.envelope"]
    assert float(fwd_rho.at_time(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert float(fwd_rho.at_time(RAMP)) == pytest.approx(1.0,
                                                         abs=1e-9)
