"""Tests for the Tier-2 ``AdiabaticProjection`` preset (R5).

The paper's appendix-B adiabatic projector (08 §10.9, AR-D8):
``P_adiab = leg @ P_ref @ leg.backward`` — ramp the *linearized* system
adiabatically to a reference configuration where a spectral projector
exists, project there, ramp back. The physics gates run on a small
linearized shallow-water beta channel; the in-tree oracle is the direct
labeled-eigenmode slow (``VorticalProjection``) projection at the target
beta, which the channel engine resolves at any beta.

Parameters (``f0=1``, ``c^2=1``, ``beta=2`` — well below the
``2 k^2 ~ 79`` Rossby/Kelvin classification edge, ``k >= 2 pi``;
``dt=1e-2`` inside the AB3 stability limit; ``tau in {0.5, 1, 2}``
spans the adiabatic onset at small step counts). The 8x8 grid gives a
finite leakage floor (plan risk 5.1), so the gates assert a monotone
trend and a conservative margin, not convergence to zero.

Measured tolerances (this grid, ``tau in {0.5, 1, 2}``, ``dt=1e-2``):
  * (i) oracle ``relative_l2(P_adiab(z), P_direct(z))``:
    6.5e-2 > 4.0e-2 > 2.6e-2 (net reduction ~0.40x);
  * (ii) idempotency ``relative_l2(P(P(z)), P(z))``:
    5.2e-6 > 4.1e-6 > 1.6e-6 (documented tol 1e-4);
  * (iii) slow-state deviation, backward-forward: 1.4e-5 > 9.3e-6 >
    2.6e-6 (decreasing); forward-forward: 3.7e-2 < 6.1e-2 < 1.1e-1
    (GROWING, ratio to bf 2.6e3 .. 4.3e4 — the AR-D8 phase argument);
  * (iv) OB(base=P_adiab) errors ~ 1.0 -> 1.9e-2 -> 9.9e-4;
  * (v) dt halving at ``tau=1``: error ratio ~1.00 (ramp-limited).
"""
import jax
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.framework.utils import jaxify
from fridom.model import term_predicates as terms
from fridom.model.closures.base import ClosureBase
from fridom.model.declarations import FieldDeclaration
from fridom.model.errors import (
    IrreversibleTermError,
    LinearOperatorGapError,
)
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.terms import term
from fridom.model.time_steppers.runge_kutta import (
    ExplicitRungeKutta,
    tableaus,
)
from fridom.model.transforms.adiabatic_projection import AdiabaticProjection
from fridom.model.transforms.adiabatic_ramping import AdiabaticRamping
from fridom.model.transforms.errors import TraceError
from fridom.model.transforms.identity import Identity
from fridom.model.transforms.norms import assert_idempotent, relative_l2
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

LIN = terms.linear
F0, CSQR, BETA = 1.0, 1.0, 2.0
COMPS = ("u", "v", "p")
TAUS = (0.5, 1.0, 2.0)
DT = 1e-2
#: documented leg-dependent idempotency / phase-neutrality tolerance
IDEM_TOL = 1e-4


# ================================================================
#  Builders (the small linear/nonlinear beta channel)
# ================================================================
def _channel_model(dt, *, advection):
    """Build a small sw2 beta channel; nonlinear iff ``advection``."""
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")
    return sw.Model(
        grid=Grid((mx, my)), csqr=CSQR, rossby_number=0.2,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=F0, beta=BETA),
        advection=advection,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))


def _rand_state(model, seed):
    """Fill a channel model with random prognostics; return the state."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in COMPS})
    return sw.State({c: model.state[c] for c in COMPS})


def _up_leg(model, tau, *, term_filter=LIN):
    """Return the up (reference -> target) beta-ramp leg (linear)."""
    return AdiabaticRamping(
        model, ramps={"coriolis.beta": (0.0, BETA)},
        ramp_period=tau, curve="exp", term_filter=term_filter)


# ================================================================
#  Shared fixtures
# ================================================================
@pytest.fixture(scope="module")
def linear_channel():
    """Build the linear channel + reference/target projectors + probe.

    The reference is a beta=0 variant of the target, so both eigenmode
    projections and the ramped legs carry the same grid identity (the
    signatures compose).
    """
    target = _channel_model(DT, advection=False)
    reference = target.variant(updates={"coriolis.beta": 0.0})
    return {
        "target": target,
        "reference": reference,
        "P_ref": sw.transforms.VorticalProjection(sw.eigenbasis(reference)),
        "P_tgt": sw.transforms.VorticalProjection(sw.eigenbasis(target)),
        "z": _rand_state(target, seed=1),
    }


@pytest.fixture(scope="module")
def oracle_sweep(linear_channel):
    """P_adiab(z) vs P_direct(z) and idempotency residuals, per tau."""
    ch = linear_channel
    z = ch["z"]
    p_direct = ch["P_tgt"](z)
    rows = {}
    for tau in TAUS:
        proj = AdiabaticProjection(_up_leg(ch["target"], tau), ch["P_ref"])
        once = proj(z)
        twice = proj(once)
        rows[tau] = {
            "oracle_err": relative_l2(once, p_direct),
            "idem": relative_l2(twice, once),
        }
    return rows


@pytest.fixture(scope="module")
def phase_neutrality(linear_channel):
    """Backward-forward (P_adiab) vs forward-forward on a slow state."""
    ch = linear_channel
    z_ref_slow = ch["P_ref"](_rand_state(ch["reference"], seed=7))
    rows = {}
    for tau in TAUS:
        leg = _up_leg(ch["target"], tau)
        # the adiabatic image of a reference-slow state: slow at the
        # target end (AR-D8: build it as leg(P_ref(z_ref_slow)))
        z_slow = leg(z_ref_slow)
        p_adiab = AdiabaticProjection(leg, ch["P_ref"])
        # the forward-forward variant, built explicitly (leg.down is the
        # forward target->reference leg, not the backward retrace)
        forward_forward = leg @ ch["P_ref"] @ leg.down
        rows[tau] = {
            "dev_bf": relative_l2(p_adiab(z_slow), z_slow),
            "dev_ff": relative_l2(forward_forward(z_slow), z_slow),
        }
    return rows


# ================================================================
#  Gate (i): the ORACLE — P_adiab -> direct slow projection with tau
# ================================================================
def test_oracle_error_decreases_with_tau(oracle_sweep):
    # P_adiab matches the direct labeled-eigenmode slow projection at
    # the target beta better as the ramp lengthens (the adiabatic limit)
    errs = [oracle_sweep[tau]["oracle_err"] for tau in TAUS]
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[-1] < 0.6 * errs[0], errs   # net reduction (measured ~0.40)
    assert errs[0] < 0.2, errs              # away from the classification edge


# ================================================================
#  Gate (ii): APPROXIMATE IDEMPOTENCY
# ================================================================
def test_idempotency_residual_is_small_and_decreases(oracle_sweep):
    resid = [oracle_sweep[tau]["idem"] for tau in TAUS]
    assert all(r < IDEM_TOL for r in resid), resid   # measured ~5e-6
    assert resid[0] > resid[1] > resid[2], resid     # more idempotent with tau


def test_assert_idempotent_passes_with_documented_tolerance(linear_channel):
    # P_adiab is idempotent only up to diabatic leakage, so the check
    # uses the documented leg-dependent tolerance (measured ~1.6e-6 at
    # tau=2 << 1e-4); the default tol=1e-9 would (correctly) fail
    ch = linear_channel
    proj = AdiabaticProjection(_up_leg(ch["target"], TAUS[-1]), ch["P_ref"])
    assert_idempotent(proj, ch["z"], tol=IDEM_TOL)
    with pytest.raises(AssertionError, match="not idempotent"):
        assert_idempotent(proj, ch["z"])   # default 1e-9 is too tight


# ================================================================
#  Gate (iii): PHASE NEUTRALITY (the corrected AR-D8 law)
# ================================================================
def test_backward_forward_is_phase_neutral_on_a_slow_state(phase_neutrality):
    # a state already in the adiabatic slow subspace at the target end
    # is (nearly) fixed by P_adiab, more so as tau grows
    dev = [phase_neutrality[tau]["dev_bf"] for tau in TAUS]
    assert all(d < 1e-3 for d in dev), dev   # measured ~1e-5
    assert dev[0] > dev[1] > dev[2], dev      # decreasing in tau


def test_forward_forward_is_not_phase_neutral(phase_neutrality):
    # AR-D8 (corrected): a forward-forward cycle (leg @ P_ref @ leg.down)
    # advances the slow phase by ~2 tau, so it is NOT a projection — its
    # deviation is orders of magnitude larger than backward-forward and
    # GROWS with tau (pins the correction against regression)
    for tau in TAUS:
        row = phase_neutrality[tau]
        # measured ratio 2.6e3 .. 4.3e4
        assert row["dev_ff"] > 100 * row["dev_bf"], (tau, row)
    dev_ff = [phase_neutrality[tau]["dev_ff"] for tau in TAUS]
    assert dev_ff[-1] > dev_ff[0], dev_ff     # phase grows with tau


# ================================================================
#  Gate (iv): OB INTEGRATION (appendix-B smoke)
# ================================================================
@pytest.fixture(scope="module")
def ob_info():
    """OptimalBalance on the NONLINEAR channel with base=P_adiab."""
    model = _channel_model(DT, advection=True)
    reference = model.variant(updates={"coriolis.beta": 0.0})
    p_ref = sw.transforms.VorticalProjection(sw.eigenbasis(reference))
    p_adiab = AdiabaticProjection(_up_leg(model, 0.3), p_ref)
    ob = fr.model.OptimalBalance(
        model, base_projection=p_adiab, ramp_period=0.3, max_it=3)
    _, info = ob.call_with_info(_rand_state(model, seed=2))
    return info


def test_ob_with_adiabatic_base_projection_converges(ob_info):
    # OB runs with the adiabatically-obtained slow projector and its
    # fixed-point errors decrease over iterations (measured
    # ~1.0 -> 1.9e-2 -> 9.9e-4)
    errors = ob_info.errors
    assert len(errors) >= 2, errors
    assert errors[-1] < errors[0], errors
    assert min(errors[1:]) < 0.1 * errors[0], errors
    assert ob_info.model_steps > 0


# ================================================================
#  Gate (v): dt HALVING — the leakage is ramp-limited, not stepper-limited
# ================================================================
@pytest.fixture(scope="module")
def dt_halving():
    """Oracle error at a fixed tau for dt and dt/2 (fresh channels)."""
    tau = 1.0
    errs = {}
    for dt in (DT, DT / 2):
        model = _channel_model(dt, advection=False)
        reference = model.variant(updates={"coriolis.beta": 0.0})
        p_ref = sw.transforms.VorticalProjection(sw.eigenbasis(reference))
        p_tgt = sw.transforms.VorticalProjection(sw.eigenbasis(model))
        z = _rand_state(model, seed=1)
        proj = AdiabaticProjection(_up_leg(model, tau), p_ref)
        errs[dt] = relative_l2(proj(z), p_tgt(z))
    return errs


def test_error_is_ramp_limited_not_stepper_limited(dt_halving):
    # halving dt at fixed tau barely moves the oracle error: the leakage
    # is set by the finite ramp, not the time stepper (measured ~1.00)
    ratio = dt_halving[DT / 2] / dt_halving[DT]
    assert 0.5 < ratio < 2.0, (dt_halving, ratio)


# ================================================================
#  Declared structure, cost, repr, complement
# ================================================================
def test_declared_structure(linear_channel):
    ch = linear_channel
    proj = AdiabaticProjection(_up_leg(ch["target"], 1.0), ch["P_ref"])
    assert proj.idempotent is True
    assert proj.traceable is False
    assert proj.domain == proj.codomain   # endo
    assert proj.domain is not None


def test_cost_is_two_leg_integrations(linear_channel):
    ch = linear_channel
    leg = _up_leg(ch["target"], 1.0)
    proj = AdiabaticProjection(leg, ch["P_ref"])
    # away backward + return forward; the projector is Tier-1 (free)
    assert proj.cost().model_steps == 2 * leg.steps
    assert proj.cost().model_steps == proj.cycle.cost().model_steps


def test_exposes_the_legs_and_cycle(linear_channel):
    ch = linear_channel
    leg = _up_leg(ch["target"], 1.0)
    proj = AdiabaticProjection(leg, ch["P_ref"])
    assert proj.leg is leg
    assert proj.reference_projection is ch["P_ref"]
    assert proj.backward_leg.is_backward
    assert not leg.is_backward


def test_repr_reports_the_legs(linear_channel):
    ch = linear_channel
    proj = AdiabaticProjection(_up_leg(ch["target"], 1.0), ch["P_ref"])
    text = repr(proj)
    assert "AdiabaticProjection(" in text
    assert "steps=" in text


def test_complement_partitions_the_state(linear_channel):
    # idempotent=True enables the complement (I - P), the imbalance
    # residual; P + (I - P) reconstructs the state
    ch = linear_channel
    proj = AdiabaticProjection(_up_leg(ch["target"], 1.0), ch["P_ref"])
    residual = proj.complement
    z = ch["z"]
    reconstructed = proj(z) + residual(z)
    assert relative_l2(reconstructed, z) < 1e-9


# ================================================================
#  Constructor guards
# ================================================================
def test_rejects_a_non_ramping_leg(linear_channel):
    ch = linear_channel
    with pytest.raises(TypeError, match="BUILT AdiabaticRamping"):
        AdiabaticProjection(ch["P_ref"], ch["P_ref"])


def test_rejects_a_non_transform_reference(linear_channel):
    ch = linear_channel
    with pytest.raises(TypeError, match="StateTransform reference"):
        AdiabaticProjection(_up_leg(ch["target"], 1.0), object())


def test_rejects_a_model_with_a_linear_operator_gap():
    # route B carries the rotation in a nonlinear term, so L has no
    # rotation: require_linear_operator refuses it (the ramp would
    # integrate a different system than L describes)
    grid = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),
                 IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")))
    route_b = sw.Model(
        grid=grid, csqr=CSQR, rossby_number=0.2,
        coriolis=sw.modules.NonlinearBetaPlaneCoriolis(f0=F0, beta=BETA),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=3))
    leg = AdiabaticRamping(route_b, ramps={}, ramp_period=1.0)
    with pytest.raises(LinearOperatorGapError, match="AdiabaticProjection"):
        AdiabaticProjection(leg, Identity())


def test_trace_guard_raises_under_jit(linear_channel):
    ch = linear_channel
    proj = AdiabaticProjection(_up_leg(ch["target"], 1.0), ch["P_ref"])
    assert proj.traceable is False
    with pytest.raises(TraceError, match="Tier-2"):
        jax.jit(proj)(ch["z"])


# ================================================================
#  AR-D6: the backward guard fires at AdiabaticProjection construction
# ================================================================
@jaxify
class _ToyRotation(Module):

    """Linear f-plane rotation with a static f0 (linear=True)."""

    field_declarations = (
        FieldDeclaration("u", space=Collocated()),
        FieldDeclaration("v", space=Collocated()),
    )

    @term(name="cor", advances=("u", "v"), linear=True)
    def cor(self, state, _ctx):
        return {"u": state["v"] * 4.0, "v": state["u"] * (-4.0)}


@jaxify
class _ToyLinearDrag(ClosureBase):

    """A LINEAR dissipative closure (survives the ``linear`` filter)."""

    default_targets = None
    field_declarations = ()

    @term(name="drag", advances=("u",), linear=True)
    def drag(self, state, _ctx):
        return {"u": state["u"] * (-0.1)}


def _toy_model():
    """Build a linear inertial oscillator carrying a linear closure."""
    grid = Grid((IntervalMesh(8, (0.0, 1.0), periodic=True, name="x"),))
    return Model(
        grid=grid, modules=(_ToyRotation(), _ToyLinearDrag(fields=("u",))),
        time_stepper=ExplicitRungeKutta(2e-3, tableau=tableaus.RK4),
        name="toy")


def test_backward_guard_fires_on_a_surviving_linear_closure():
    # a linearized model can retain a LINEAR closure; the linear filter
    # keeps it, and constructing AdiabaticProjection builds leg.backward
    # eagerly, firing AR-D6 at construction
    leg = AdiabaticRamping(
        _toy_model(), ramps={}, ramp_period=0.04, term_filter=LIN)
    with pytest.raises(IrreversibleTermError, match="_ToyLinearDrag/drag"):
        AdiabaticProjection(leg, Identity())


def test_term_filter_drops_the_closure_and_builds():
    # the taught fix (the term_filter idiom): drop the closure from the
    # legs, and the projector constructs
    leg = AdiabaticRamping(
        _toy_model(), ramps={}, ramp_period=0.04,
        term_filter=LIN & ~terms.owned_by(ClosureBase))
    proj = AdiabaticProjection(leg, Identity())
    assert proj.cost().model_steps == 2 * leg.steps
