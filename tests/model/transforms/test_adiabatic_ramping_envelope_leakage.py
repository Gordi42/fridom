"""Stretched-exponential leakage of the term-envelope ramp cycle.

A prefix-mirrored shard of ``test_adiabatic_ramping`` pinning the
adiabatic-theorem rate on the §C **envelope** path, mirroring the
parameter-ramp gate of ``test_adiabatic_ramping_exponential``. The
diagnostic is the round trip ``down @ up`` with ``envelope=True``:
rho ramps 0 -> 1 -> 0, so both endpoints are exactly the linear
model and the vortical subspace of its eigenbasis is the exact slow
manifold at both ends — the balanced O(eps) correction of the
mid-ramp nonlinear model cancels out of the measured imbalance, and
what is left is pure diabatic leakage. For the Gevrey-2 ``exp``
curve it decays as the stretched exponential
``eta(tau) ~ exp(-c*sqrt(tau))``; the endpoint-nonsmooth ``linear``
curve floors at its algebraic rate.

Recipe (16x16 sw2 channel, x-periodic / y-walled, csqr=1, f0=1,
advection on, rossby 0.2, AB3, dt=0.15/16, seed 1, slow state
normalized to amplitude 0.1 — weak effective nonlinearity, the
adiabatic regime). Measured ``eta`` (relative imbalance vs the
target vortical eigenbasis after ``down @ up``), for reference:

  * exp:    5.4e-5, 2.8e-6, 1.5e-7, 7.0e-9   (tau = 5, 10, 20, 40)
  * linear: 2.0e-4, 6.5e-5, 4.2e-5, 2.0e-5

giving a log(eta_exp)-vs-sqrt(tau) slope ~ -2.15 (R^2 ~ 0.98), an
exp deep/shallow ratio ~1.3e-4 vs ~0.10 for linear, and an
exp-vs-linear separation ~3.4e-4 at tau=40. The physics gates below
are relaxed against those measurements (regression guards, not
tight fits).
"""
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw

F0 = 1.0
CSQR = 1.0
DT = 0.15 / 16
AMP = 0.1
COMPS = ("u", "v", "p")
TAUS = (5.0, 10.0, 20.0, 40.0)


# ================================================================
#  Builders (self-contained: no cross-test-file imports)
# ================================================================
def _channel():
    """Build the 16x16 sw2 channel with advection on.

    Today-parity spelling of the retired (csqr=CSQR,
    rossby_number=0.2, f0=F0) channel: GravityWave scaling with
    Fr = 0.2 and the rotation as Ro = Fr / f0.
    """
    mx = fr.spatial.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    my = fr.spatial.meshes.IntervalMesh(
        16, (0.0, 1.0), periodic=False, name="y")
    return sw.Model(
        grid=fr.spatial.Grid((mx, my), device_ids=(0,)),
        scaling=fr.scaling.GravityWave(),
        core=sw.Core(froude_number=0.2, depth=CSQR),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=0.2 / F0),
        advection=True,
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
    """Cycle rho 0 -> 1 -> 0 over the slow state; measure eta(tau).

    One shared model/grid: the eigenbasis is the linear operator L of
    the target itself (the envelope never deforms L), so the same
    vortical projection defines the slow subspace at rho = 0 on both
    ends of the cycle.
    """
    target = _channel()
    p = sw.transforms.VorticalProjection(sw.eigenbasis(target))

    rng = np.random.default_rng(1)
    target.set_fields(**{
        c: rng.standard_normal(np.asarray(target.state[c].data).shape)
        for c in COMPS})
    z0 = p(sw.State({c: target.state[c] for c in COMPS}))
    z0 = z0 * (AMP / _l2(z0))

    etas = {}
    for curve in ("exp", "linear"):
        etas[curve] = []
        for tau in TAUS:
            up = fr.model.transforms.AdiabaticRamping(
                target, envelope=True, ramp_period=tau, curve=curve)
            etas[curve].append(fr.model.transforms.relative_imbalance(
                up.down(up(z0)), p))
    return {
        "exp": etas["exp"], "linear": etas["linear"],
        "reference_imbalance": fr.model.transforms.relative_imbalance(
            z0, p)}


# ================================================================
#  Sanity: the initial state is (numerically) purely slow
# ================================================================
def test_reference_state_is_numerically_slow(leakage_sweep):
    # z0 lives in the vortical subspace to rounding (~1e-15), so all
    # leakage seen downstream is diabatic
    assert leakage_sweep["reference_imbalance"] < 1e-10


# ================================================================
#  Gate 1: exp cycle leaks as a stretched exponential in sqrt(tau)
# ================================================================
def test_exp_leakage_is_stretched_exponential_in_sqrt_tau(
        leakage_sweep):
    # exp(-c*sqrt(tau)): log(eta) is linear in sqrt(tau) with a
    # clearly negative slope (measured ~ -2.15, R^2 ~ 0.98)
    sqrt_tau = np.sqrt(np.array(TAUS))
    log_eta = np.log(np.array(leakage_sweep["exp"]))
    basis = np.vstack([sqrt_tau, np.ones_like(sqrt_tau)]).T
    (slope, intercept), *_ = np.linalg.lstsq(basis, log_eta,
                                             rcond=None)
    predicted = basis @ np.array([slope, intercept])
    ss_res = float(np.sum((log_eta - predicted) ** 2))
    ss_tot = float(np.sum((log_eta - log_eta.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    assert slope < -1.2, (slope, leakage_sweep["exp"])
    assert r_squared > 0.95, (r_squared, leakage_sweep["exp"])


# ================================================================
#  Gate 2: exp deepens with tau; linear saturates at its floor
# ================================================================
def test_exp_deepens_while_linear_saturates(leakage_sweep):
    exp = leakage_sweep["exp"]
    lin = leakage_sweep["linear"]
    # the exp cycle drops >1000x from the shallowest to the deepest
    # ramp (measured ~1.3e-4x): the adiabatic regime is reached
    assert exp[-1] < 1e-3 * exp[0], exp
    # the linear (endpoint-nonsmooth) cycle barely moves — it floors
    # at its algebraic rate (measured deep/shallow ~0.10x)
    assert lin[-1] > 0.01 * lin[0], lin


# ================================================================
#  Gate 3: in the deep regime exp beats linear by orders of magnitude
# ================================================================
def test_exp_beats_linear_in_the_deep_regime(leakage_sweep):
    # at tau=40 the endpoint-smooth ramp leaks >100x less than the
    # linear ramp (measured ~3.4e-4x)
    exp = leakage_sweep["exp"]
    lin = leakage_sweep["linear"]
    assert exp[-1] < 1e-2 * lin[-1], (exp[-1], lin[-1])
