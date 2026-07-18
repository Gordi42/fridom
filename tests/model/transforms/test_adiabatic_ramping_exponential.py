"""Stretched-exponential leakage of an endpoint-smooth adiabatic ramp.

A prefix-mirrored shard of ``test_adiabatic_ramping`` (keeps the
``test_adiabatic_ramping*`` glob mapping to
``src/fridom/model/transforms/adiabatic_ramping.py``). It pins the one
quantitative *physical* law the base-surface unit tests do not: the
diabatic leakage of a single adiabatic ramp leg decays as a
**stretched exponential** ``eta(tau) ~ exp(-c * sqrt(tau))`` in the
ramp period ``tau`` — the Gevrey-2 (``curve="exp"``) rate the adiabatic
theorem buys from a ramp whose endpoint time-derivatives all vanish
(plan ``design/plans/active/adiabatic_ramping.md`` §1 AR-D1/§6, Masur &
Oliver 2020 §3.4). The gate is a regression guard: if a future change
loses ramp endpoint-smoothness the ``exp`` leg collapses toward the
merely algebraic ``linear``-ramp rate, and every assertion here fails.

The diagnostic is a **single UP leg**, on purpose. The two-leg
``AdiabaticProjection`` cycle (``leg @ P_ref @ leg.backward``) cannot
show the deep regime: its projection error floors at its own
idempotency residual — an O(dt) + multistep warm-up asymmetry
reversibility artifact that never falls below ~1e-6 on this grid — so
its ``eta`` saturates long before the stretched-exponential decay does.
The single up leg carries only the diabatic leakage and reaches ~5e-8
at ``tau=40``, four orders below that floor (see
``test_adiabatic_projection`` gate (ii) for the cycle floor itself).

Recipe (16x16 linearized sw2 channel, x-periodic / y-walled, csqr=1,
FPlaneCoriolis f0 ramped 0 -> 1, advection off, AB3, dt=0.15/16, seed
1). The reference (f0=0) slow state is carried adiabatically to the
target f0=1; ``eta`` is its relative imbalance against the DISCRETE
target vortical eigenbasis. Investigation-measured ``eta`` (validated
robust to seed and grid), for reference:

  * exp:    2.4e-3, 7.5e-5, 4.4e-6, 4.7e-8   (tau = 5, 10, 20, 40)
  * linear: 3.2e-3, 4.6e-4, 4.1e-4, 3.4e-4

giving a log(eta_exp)-vs-sqrt(tau) slope ~ -2.57 (R^2 ~ 0.990), a
deep/shallow ratio ~2e-5 (exp) vs ~0.11 (linear), and an exp-vs-linear
separation ~1.4e-4 at tau=40. Whole shard ~3.3 s wall on CPU (one
module-scope sweep shared by every assertion).
"""
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

F0 = 1.0
CSQR = 1.0
DT = 0.15 / 16
COMPS = ("u", "v", "p")
TAUS = (5.0, 10.0, 20.0, 40.0)
LIN = fr.model.term_predicates.linear


# ================================================================
#  Builders (self-contained: no cross-test-file imports)
# ================================================================
def _channel(f0):
    """Build the 16x16 linearized sw2 channel at the given f0."""
    mx = fr.spatial.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(
        16, (0.0, 1.0), periodic=False, name="y")
    return sw.Model(
        grid=fr.spatial.Grid((mx, my), device_ids=(0,)),
        csqr=CSQR, rossby_number=0.2,
        coriolis=sw.modules.FPlaneCoriolis(f0=f0), advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=3))


def _l2(state):
    """Volume-weighted l2 norm of a state (duplicated norms._l2_norm)."""
    total = 0.0
    for field in state:
        total += float(np.sum(np.real(
            (field.conj() * field).integrate().data)))
    return np.sqrt(total)


# ================================================================
#  The shared sweep: one build, both curves x every tau
# ================================================================
@pytest.fixture(scope="module")
def leakage_sweep():
    """Ramp the reference slow state to the target; measure eta(tau).

    Built once on ONE shared grid (the reference is an f0=0 variant of
    the target) so the target eigenbasis projection and the ramped legs
    carry the same grid identity. ``eta`` is scale-invariant (a ratio),
    and the linear leg is a linear operator, so the reference slow
    state needs no normalization for the measured eta — it is
    normalized only to make the reference-imbalance sanity check read
    as a unit state.
    """
    target = _channel(F0)
    reference = target.variant(updates={"coriolis.f0": 0.0})
    p_ref = sw.transforms.VorticalProjection(sw.eigenbasis(reference))
    p_tgt = sw.transforms.VorticalProjection(sw.eigenbasis(target))

    # normalized slow state of the reference (f0=0): purely slow there
    rng = np.random.default_rng(1)
    reference.set_fields(**{
        c: rng.standard_normal(np.asarray(reference.state[c].data).shape)
        for c in COMPS})
    z0 = p_ref(sw.State({c: reference.state[c] for c in COMPS}))
    z0 = z0 * (1.0 / _l2(z0))

    etas = {}
    for curve in ("exp", "linear"):
        etas[curve] = [
            fr.model.transforms.relative_imbalance(
                fr.model.transforms.AdiabaticRamping(
                    target, ramps={"coriolis.f0": (0.0, F0)},
                    ramp_period=tau, curve=curve, term_filter=LIN)(z0),
                p_tgt)
            for tau in TAUS]
    return {
        "exp": etas["exp"], "linear": etas["linear"],
        "reference_imbalance": fr.model.transforms.relative_imbalance(
            z0, p_ref)}


# ================================================================
#  Sanity: the initial state is (numerically) purely slow
# ================================================================
def test_reference_state_is_numerically_slow(leakage_sweep):
    # z0 lives in the reference (f0=0) vortical subspace to rounding
    # (measured ~1e-15), so all leakage seen downstream is diabatic
    assert leakage_sweep["reference_imbalance"] < 1e-10


# ================================================================
#  Gate 1: exp leg leaks as a stretched exponential in sqrt(tau)
# ================================================================
def test_exp_leakage_is_stretched_exponential_in_sqrt_tau(leakage_sweep):
    # exp(-c*sqrt(tau)): log(eta) is linear in sqrt(tau) with a clearly
    # negative slope. A merely algebraic (endpoint-nonsmooth) ramp would
    # flatten the slope and spoil the fit. Measured slope ~ -2.57,
    # R^2 ~ 0.990.
    sqrt_tau = np.sqrt(np.array(TAUS))
    log_eta = np.log(np.array(leakage_sweep["exp"]))
    basis = np.vstack([sqrt_tau, np.ones_like(sqrt_tau)]).T
    (slope, intercept), *_ = np.linalg.lstsq(basis, log_eta, rcond=None)
    predicted = basis @ np.array([slope, intercept])
    ss_res = float(np.sum((log_eta - predicted) ** 2))
    ss_tot = float(np.sum((log_eta - log_eta.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    assert slope < -1.5, (slope, leakage_sweep["exp"])
    assert r_squared > 0.98, (r_squared, leakage_sweep["exp"])


# ================================================================
#  Gate 2: exp deepens with tau; linear saturates at its floor
# ================================================================
def test_exp_deepens_while_linear_saturates(leakage_sweep):
    exp = leakage_sweep["exp"]
    lin = leakage_sweep["linear"]
    # the exp leg drops >1000x from the shallowest to the deepest ramp
    # (measured ~2e-5x): the adiabatic regime is genuinely reached
    assert exp[-1] < 1e-3 * exp[0], exp
    # the linear (endpoint-nonsmooth) leg barely moves — it floors at
    # its algebraic O(1/tau)-ish rate (measured deep/shallow ~0.11x)
    assert lin[-1] > 0.05 * lin[0], lin


# ================================================================
#  Gate 3: in the deep regime exp beats linear by orders of magnitude
# ================================================================
def test_exp_beats_linear_in_the_deep_regime(leakage_sweep):
    # at tau=40 the endpoint-smooth ramp leaks >100x less than the
    # linear ramp (measured ~1.4e-4x): the separation IS the smoothness
    # payoff this regression pins
    exp = leakage_sweep["exp"]
    lin = leakage_sweep["linear"]
    assert exp[-1] < 1e-2 * lin[-1], (exp[-1], lin[-1])
