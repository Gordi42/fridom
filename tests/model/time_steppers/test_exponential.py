"""ETDRK4: the exponential stepper on an eigenbasis-diagonalized L.

The load-bearing claims, each with a test below:

- ``phi_functions`` are accurate on BOTH branches (the Taylor sum
  below |z| = 0.5 and the closed form above it), and hit 1, 1/2, 1/6
  at exactly z = 0 — where the vortical modes sit.
- ``exp(L dt)`` is EXACT: the linear-only answer does not depend on dt
  at all, the group law holds to machine precision, and the energy is
  exactly conserved (no damping) even at 30x the AB3 gravity CFL.
- ``time_discretization_effect`` is the identity.
- The double-counting guard fires: a model that still carries its
  linear terms raises ``LinearTermInTendencyError``.
- The nonlinear scheme is fourth order, INCLUDING through a ``Ramp``
  on ``scaling.rossby`` (which needs the per-stage clock times).
"""
import math

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model import term_predicates as terms
from fridom.model.errors import LinearTermInTendencyError
from fridom.model.time_steppers.exponential import (
    ETDRK4,
    phi_functions,
)

N = 16
COMPONENTS = ("u", "v", "p")
# the AB3 imaginary-axis limit (0.7236) over the C-grid gravity
# symbol (2*sqrt(2)*c/dx): the step size AB3 cannot exceed
AB3_DT = 0.2558 / N


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(scope="module")
def grid():
    """Return a periodic-x / walled-y channel (the engine's shape)."""
    mx = fr.spatial.meshes.IntervalMesh(
        N, (0.0, 1.0), periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(
        N, (0.0, 1.0), periodic=False, name="y")
    return fr.spatial.Grid((mx, my))


def _model(grid, stepper, rossby=0.2, *, filtered):
    """Assemble a shallow-water channel; filtered drops linear terms."""
    extra = {"term_filter": ~terms.linear} if filtered else {}
    return sw.Model(
        grid=grid, csqr=1.0, rossby_number=rossby,
        coriolis=sw.modules.FPlaneCoriolis(f0=1.0), advection=True,
        time_stepper=stepper, **extra)


@pytest.fixture(scope="module")
def basis(grid):
    """Return the UNFILTERED model's eigenbasis: the operator L."""
    return sw.eigenbasis(_model(
        grid, fr.model.time_steppers.AdamBashforth(1e-3, order=3),
        filtered=False))


@pytest.fixture(scope="module")
def state0(basis):
    """Return a balanced (vortical) initial state."""
    return sw.random_state(basis, "vortical", seed=7)


def _run(model, state, steps):
    """Reset, seed and advance; return the prognostic arrays."""
    model.reset()
    model.set_state(state)
    model.advance(steps)
    return {c: np.asarray(model.state[c].data) for c in COMPONENTS}


def _rel_error(got, want):
    """Relative L2 error over the prognostic components."""
    num = sum(np.sum(np.abs(got[c] - want[c]) ** 2) for c in COMPONENTS)
    den = sum(np.sum(np.abs(want[c]) ** 2) for c in COMPONENTS)
    return float(np.sqrt(num / den))


def _norm(arrays):
    """L2 norm over the prognostic components."""
    return float(np.sqrt(
        sum(np.sum(np.abs(arrays[c]) ** 2) for c in COMPONENTS)))


# ================================================================
#  phi-functions
# ================================================================
def test_phi_functions_at_zero_are_the_rk4_limits():
    """phi_k(0) = 1, 1/2, 1/6 -- the vortical modes sit at z = 0."""
    phi1, phi2, phi3 = phi_functions(jnp.zeros(1, dtype=complex))
    assert float(jnp.real(phi1[0])) == pytest.approx(1.0)
    assert float(jnp.real(phi2[0])) == pytest.approx(0.5)
    assert float(jnp.real(phi3[0])) == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    "magnitude",
    [pytest.param(m, id=f"|z|={m}")
     for m in (1e-8, 1e-3, 0.49, 0.51, 2.0)])
def test_phi_functions_agree_with_the_series_on_both_branches(magnitude):
    """Taylor branch (|z|<0.5) and closed form (|z|>=0.5) both hold.

    The reference is the defining series -- which is precisely what
    the closed forms lose to cancellation near the origin, so this
    pins the branch cut at 0.5 from both sides. (The series is only a
    trustworthy reference while it converges quickly, hence |z| <= 2;
    the stiff regime is covered by the identity test below.)
    """
    z = -1j * magnitude
    got = phi_functions(jnp.asarray([z]))
    want = [sum(z ** j / math.factorial(j + k) for j in range(40))
            for k in (1, 2, 3)]
    for phi, reference in zip(got, want, strict=True):
        assert complex(phi[0]) == pytest.approx(
            reference, rel=1e-10, abs=1e-13)


@pytest.mark.parametrize(
    "magnitude",
    [pytest.param(m, id=f"|z|={m}") for m in (2.0, 10.0, 50.0, 500.0)])
def test_phi_functions_satisfy_the_defining_identity_when_stiff(
        magnitude):
    """z^k phi_k == e^z - (the truncated series), in the stiff regime.

    For an OSCILLATORY z the closed forms are perfectly conditioned
    (|e^z| = 1, so the numerators never cancel) -- this is the regime
    the exponential stepper actually runs in, at omega*dt >> 1.
    """
    z = -1j * magnitude
    phi1, phi2, phi3 = (complex(p[0])
                        for p in phi_functions(jnp.asarray([z])))
    exp_z = np.exp(z)
    assert z * phi1 == pytest.approx(exp_z - 1.0, rel=1e-12)
    assert z**2 * phi2 == pytest.approx(exp_z - 1.0 - z, rel=1e-12)
    assert z**3 * phi3 == pytest.approx(
        exp_z - 1.0 - z - 0.5 * z**2, rel=1e-12)


def test_taylor_branch_reproduces_the_closed_form_at_the_cut():
    """Just below the cut, the Taylor sum matches the closed form.

    At |z| = 0.5 the closed forms still carry ~14 good digits (the
    phi3 numerator cancels only ~50x), so they are a valid reference
    exactly where the Taylor branch takes over -- which is what makes
    the branch switch safe rather than merely plausible.
    """
    z = -1j * 0.4999
    phi1, phi2, phi3 = (complex(p[0])
                        for p in phi_functions(jnp.asarray([z])))
    exp_z = np.exp(z)
    assert phi1 == pytest.approx((exp_z - 1) / z, abs=1e-13)
    assert phi2 == pytest.approx((exp_z - 1 - z) / z**2, abs=1e-13)
    assert phi3 == pytest.approx(
        (exp_z - 1 - z - 0.5 * z**2) / z**3, abs=1e-13)


# ================================================================
#  exp(L dt) is exact
# ================================================================
def test_linear_answer_is_independent_of_dt(grid, basis, state0):
    """With N = 0 the scheme IS exp(L T): dt cannot change it.

    The strongest statement of exactness available without a
    reference: one step of dt = 64x the AB3 stability limit must land
    on the same state as 64 steps of the limit itself.
    """
    results = []
    for steps in (1, 8, 64):
        dt = 64 * AB3_DT / steps
        # rossby = 0 kills the (only) nonlinear term: pure exp(L dt)
        model = _model(grid, ETDRK4(dt, basis), rossby=0.0,
                       filtered=True)
        results.append(_run(model, state0, steps))
    for other in results[1:]:
        assert _rel_error(other, results[0]) < 1e-11


def test_propagator_obeys_the_group_law(grid, basis, state0):
    """exp(L dt) applied twice == exp(L 2dt) -- to machine precision."""
    dt = 8 * AB3_DT
    coarse = _run(_model(grid, ETDRK4(2 * dt, basis), rossby=0.0,
                         filtered=True), state0, 4)
    fine = _run(_model(grid, ETDRK4(dt, basis), rossby=0.0,
                       filtered=True), state0, 8)
    assert _rel_error(fine, coarse) < 1e-11


def test_propagator_does_not_damp(grid, basis, state0):
    """|exp(-i omega dt)| = 1: the energy is EXACTLY conserved.

    This is what a semi-implicit theta-method cannot do -- it buys
    stability by damping the very waves it is asked to carry.
    """
    dt = 30 * AB3_DT
    model = _model(grid, ETDRK4(dt, basis), rossby=0.0, filtered=True)
    start = _norm(_run(model, state0, 0))
    for steps in (1, 10, 100):
        assert _norm(_run(model, state0, steps)) / start == \
            pytest.approx(1.0, abs=1e-10)


def test_time_discretization_effect_is_the_identity(basis):
    """The linear operator is exact, so omega_discrete == omega."""
    stepper = ETDRK4(0.1, basis)
    omega = np.array([0.0, 0.5, 12.0])
    got = stepper.time_discretization_effect(omega)
    assert np.allclose(got, omega.astype(complex))
    # and it does not depend on dt
    assert np.allclose(
        stepper.time_discretization_effect(omega, dt=7.5), got)


# ================================================================
#  The double-counting guard
# ================================================================
def test_unfiltered_model_raises_the_double_counting_error(
        grid, basis, state0):
    """A tendency that still holds linear terms would count them twice."""
    model = _model(grid, ETDRK4(AB3_DT, basis), filtered=False)
    model.set_state(state0)
    with pytest.raises(LinearTermInTendencyError,
                       match=r"counted twice"):
        model.advance(1)


def test_double_counting_error_names_the_offending_terms(
        grid, basis, state0):
    """The message is taught: it names the terms and the fix."""
    model = _model(grid, ETDRK4(AB3_DT, basis), filtered=False)
    model.set_state(state0)
    with pytest.raises(LinearTermInTendencyError) as excinfo:
        model.advance(1)
    assert any("gravity" in key for key in excinfo.value.terms)
    assert "term_filter" in str(excinfo.value)


# ================================================================
#  The nonlinear scheme
# ================================================================
def test_beats_the_ab3_gravity_cfl(grid, basis, state0):
    """ETDRK4 runs stably far above the step AB3 cannot cross."""
    dt = 20 * AB3_DT
    model = _model(grid, ETDRK4(dt, basis), filtered=True)
    out = _run(model, state0, 60)
    assert np.isfinite(_norm(out))
    assert _norm(out) < 10 * _norm(_run(model, state0, 0))
    # ... while AB3 at the same dt blows up (it is gravity-limited)
    ab3 = _model(grid, fr.model.time_steppers.AdamBashforth(
        dt, order=3), filtered=False)
    with pytest.raises(fr.model.results.PanicError):
        _run(ab3, state0, 60)


def test_fourth_order_through_a_ramped_parameter(grid, basis, state0):
    """A Ramp on scaling.rossby lives in N; the order must survive it.

    This is the per-stage eval_params rule: evaluating all four RK
    stages at t_n instead of at ``clock.shifted(c_i * dt)`` would
    silently drop the scheme to FIRST order, which is exactly what
    this asserts against. Self-convergence against a fine ETDRK4 run
    (an AB3 reference saturates at its own error floor long before
    ETDRK4 does).
    """
    total = 1.0
    ramp = fr.model.Ramp(0.0, 0.2, period=total, curve="cosine")

    def integrate(count):
        return _run(_model(grid, ETDRK4(total / count, basis), ramp,
                           filtered=True), state0, count)

    reference = integrate(64)
    errors = [_rel_error(integrate(count), reference)
              for count in (4, 8)]
    # halving dt must cut the error by ~2^4
    order = np.log2(errors[0] / errors[1])
    assert order > 3.0
