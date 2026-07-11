"""
Toy-ODE harness (T1/R2) for the private slaving-expansion core.

Toy A: a rotating triad with detuning (3 complex modes, energy-
conserving quadratic coupling; delta = 0 gives a stationary slow
space). Toy B: the Lorenz-86 / Lorenz-Krishnamurthy five-component
model (slow (u, v, w) triad + fast (x, z) pair).

Residual conventions (documented choice):

- ``oscillation_residual`` (spec test 2): integrate ~2 fast periods
  from the balanced state with RK4 and take the max distance of the
  fast component from the slaving manifold, ``max_t |W z(t) -
  W balance(V z(t))|``. The manifold is re-evaluated along the
  trajectory because a fixed-reference deviation is contaminated by
  the slow drift of the slaved fast component (O(eps^2)) from order
  2 on.
- ``differential_residual`` (design note S1.6; used for the order
  1..4 slope tables): ``|W F(z_b) - ds[vdot]|`` with the slaved
  prediction by central finite difference. Chosen for the tables
  because an integration-based residual at order 4 (signal eps^5)
  would need integrator error far below float64 RK4 at sane cost.
"""
import math
from dataclasses import dataclass

import numpy as np
import pytest

from fridom.framework2.transforms._slaving import (
    SlavingOps,
    balance_expansion,
)

# ================================================================
#  Generic helpers
# ================================================================


def l2(x):
    return float(np.linalg.norm(x))


def fitted_slope(eps, res):
    """Fitted log-log slope of res(eps)."""
    return float(np.polyfit(np.log(np.asarray(eps)),
                            np.log(np.asarray(res)), 1)[0])


def rk4_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_bilinear(nonlin):
    """Symmetric bilinear form of a quadratic map, by polarization."""
    def bilinear(z1, z2):
        return 0.5 * (nonlin(z1 + z2) - nonlin(z1) - nonlin(z2))
    return bilinear


def _zero_lin(z):
    return 0.0 * z


# ================================================================
#  Toy A: rotating triad with detuning
# ================================================================
@dataclass(frozen=True)
class Triad:

    """dz/dt = i diag(delta, wf, -wf) z + energy-conserving coupling."""

    coupling: float
    delta: float = 0.0
    omega_f: float = 1.0
    # unequal pair strengths: equal ones produce an accidental
    # cancellation that makes the order-3 series term vanish at
    # leading order (slopes 2, 4, 4, ... instead of the generic case)
    ca: float = 1.0
    cb: float = 0.7
    ce: float = 1.3

    def lin(self, z):
        lam = np.array([self.delta, self.omega_f, -self.omega_f])
        return 1j * lam * z

    def nonlin(self, z):
        # each of the three coupling pairs cancels in Re<z, N(z)>,
        # so sum_j |z_j|^2 is conserved exactly
        c = self.coupling
        a, b, e = c * self.ca, c * self.cb, c * self.ce
        z0, z1, z2 = z
        return np.array([
            a * z1 * z2 - b * np.conj(z0) * z1 - e * z0 * z2,
            b * z0 ** 2 - a * z0 * np.conj(z2),
            e * z0 * np.conj(z0),
        ])

    def tendency(self, z):
        return self.lin(z) + self.nonlin(z)

    def ops(self, zero_lin=False):
        wf = self.omega_f
        mask = np.array([1.0, 0.0, 0.0])
        # L_w = i diag(., wf, -wf) -> inverse is -i/wf, +i/wf
        inv_diag = np.array([0.0, 1.0 / (1j * wf), -1.0 / (1j * wf)])
        return SlavingOps(
            slow=lambda z: mask * z,
            fast=lambda z: (1.0 - mask) * z,
            inv_lw=lambda z: inv_diag * z,
            bilinear=make_bilinear(self.nonlin),
            lin=_zero_lin if zero_lin else self.lin,
        )


Z_TRIAD = np.array([0.8 + 0.3j, 0.05 - 0.02j, -0.03 + 0.04j])


# ================================================================
#  Toy B: Lorenz-86 / Lorenz-Krishnamurthy (conservative core)
# ================================================================
@dataclass(frozen=True)
class LorenzK:

    """u, v, w slow triad + x, z fast pair; global coupling scale."""

    coupling: float
    b: float = 0.5
    omega_f: float = 1.0

    def lin(self, y):
        wf = self.omega_f
        return np.array([0.0, 0.0, 0.0, -wf * y[4], wf * y[3]])

    def nonlin(self, y):
        u, v, w, _x, z = y  # x unused: no fast-fast coupling
        b = self.b
        return self.coupling * np.array([
            -v * w + b * v * z,
            u * w - b * u * z,
            -u * v,
            0.0,
            b * u * v,
        ])

    def tendency(self, y):
        return self.lin(y) + self.nonlin(y)

    def ops(self, zero_lin=False):
        wf = self.omega_f
        mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

        def inv_lw(y):
            # L_w acts on the pair as (x, z) -> (-wf z, wf x), so
            # the inverse of a pair (p, q) is the pair (q, -p) / wf
            return np.array([0.0, 0.0, 0.0, y[4] / wf, -y[3] / wf])

        return SlavingOps(
            slow=lambda y: mask * y,
            fast=lambda y: (1.0 - mask) * y,
            inv_lw=inv_lw,
            bilinear=make_bilinear(self.nonlin),
            lin=_zero_lin if zero_lin else self.lin,
        )


Y_LK = np.array([0.6, -0.4, 0.9, 0.1, -0.2])

TOYS = [
    pytest.param(lambda c: (Triad(coupling=c), Z_TRIAD), id="triad"),
    pytest.param(lambda c: (LorenzK(coupling=c), Y_LK), id="lorenz-k"),
]

EPS_RK4 = np.logspace(-2.5, -1.4, 6)      # test 2 (integration)
EPS_DIFF = np.logspace(-2.0, -1.0, 6)     # tests 3/4 (differential)
EPS_DETUNE = np.logspace(-3.0, -2.0, 5)   # test 5


# ================================================================
#  Residual measures
# ================================================================
def oscillation_residual(sys, z, order, *, nsteps=512, sample_every=4):
    """Max distance to the slaving manifold over ~2 fast periods."""
    ops = sys.ops()

    def bal(state):
        return balance_expansion(state, order=order, ops=ops).state

    y = bal(z)
    dt = 2.0 * (2.0 * np.pi / sys.omega_f) / nsteps
    dev = 0.0
    for step in range(1, nsteps + 1):
        y = rk4_step(sys.tendency, y, dt)
        if step % sample_every == 0:
            dev = max(dev, l2(ops.fast(y) - ops.fast(bal(y))))
    return dev


def differential_residual(sys, z, order, scheme="direct",
                          closure="leading", zero_lin=False):
    """|W F(z_b) - ds[vdot]| with a central-FD slaved prediction."""
    ops = sys.ops(zero_lin=zero_lin)

    def bal(state):
        return balance_expansion(state, order=order, ops=ops,
                                 scheme=scheme, closure=closure).state

    zb = bal(z)
    vdot = ops.slow(sys.tendency(zb))
    scale = l2(vdot)
    direction = vdot / scale
    v = ops.slow(z)
    h = np.finfo(float).eps ** (1.0 / 3.0) * max(l2(v), 1.0)
    ds = scale * (ops.fast(bal(v + h * direction))
                  - ops.fast(bal(v - h * direction))) / (2.0 * h)
    return l2(ops.fast(sys.tendency(zb)) - ds)


# ================================================================
#  Toy sanity
# ================================================================
def test_triad_coupling_conserves_energy():
    sys = Triad(coupling=0.7, delta=0.3)
    z = np.array([0.5 - 0.8j, -0.3 + 0.6j, 0.9 + 0.1j])
    assert abs(np.real(np.vdot(z, sys.tendency(z)))) < 1e-14


def test_lorenz_k_linear_part_is_antisymmetric():
    sys = LorenzK(coupling=0.3)
    y = np.array([0.5, -0.8, 0.3, 0.6, -0.9])
    assert abs(float(y @ sys.lin(y))) < 1e-14


# ================================================================
#  Input validation and basic structure
# ================================================================
def test_validation_errors():
    ops = Triad(coupling=0.01).ops()
    with pytest.raises(ValueError, match="order"):
        balance_expansion(Z_TRIAD, order=-1, ops=ops)
    with pytest.raises(ValueError, match="order"):
        balance_expansion(Z_TRIAD, order=True, ops=ops)
    with pytest.raises(ValueError, match="scheme"):
        balance_expansion(Z_TRIAD, order=1, ops=ops, scheme="magic")
    with pytest.raises(ValueError, match="closure"):
        balance_expansion(Z_TRIAD, order=1, ops=ops, closure="magic")


@pytest.mark.parametrize("scheme", ["direct", "telescoping"])
def test_order_zero_is_slow_projection(scheme):
    sys = Triad(coupling=0.02)
    res = balance_expansion(Z_TRIAD, order=0, ops=sys.ops(),
                            scheme=scheme)
    np.testing.assert_allclose(
        res.state, np.array([Z_TRIAD[0], 0.0, 0.0]))
    assert len(res.terms) == 1


# ================================================================
#  Test 1: collapse / regression (Toy A, delta = 0)
# ================================================================
@pytest.mark.parametrize("scheme", ["direct", "telescoping"])
def test_machenhauer_closed_form_triad(scheme):
    c, wf = 0.02, 1.0
    sys = Triad(coupling=c, omega_f=wf)
    res = balance_expansion(Z_TRIAD, order=1, ops=sys.ops(),
                            scheme=scheme)
    v0 = Z_TRIAD[0]
    # phi_1 = -L_w^-1 W B(v, v) with B(v, v) = (0, b v0^2, e |v0|^2)
    b, e = c * sys.cb, c * sys.ce
    expected = np.array([
        v0,
        1j * b * v0 ** 2 / wf,
        -1j * e * v0 * np.conj(v0) / wf,
    ])
    np.testing.assert_allclose(res.state, expected, rtol=0,
                               atol=1e-15)
    assert len(res.terms) == 2


def _v1_oracle_triad(z, order, sys):
    """Hand-rolled per-mode i/lambda recursion (the v1 oracle)."""
    # L = i diag(0, wf, -wf): 1/(i lam_j) acts as -1j * sign / wf on
    # the fast modes (delta = 0, leading closure, v1 bookkeeping)
    wf = sys.omega_f
    inv_diag = np.array([0.0, -1j / wf, 1j / wf])
    slow_mask = np.array([1.0, 0.0, 0.0])
    bilinear = make_bilinear(sys.nonlin)
    base = slow_mask * z
    slow_cache = {0: base}
    fast_cache = {}

    def slow_d(k):
        if k not in slow_cache:
            acc = np.zeros(3, dtype=complex)
            for m in range(k):
                acc = acc + math.comb(k - 1, m) * bilinear(
                    slow_d(m), slow_d(k - 1 - m))
            slow_cache[k] = slow_mask * acc
        return slow_cache[k]

    def fast_d(n, k):
        if (n, k) not in fast_cache:
            upper = slow_d(k + 1) if n == 1 else fast_d(n - 1, k + 1)
            acc = np.zeros(3, dtype=complex)
            for m in range(k + 1):
                coef = math.comb(k, m)
                for j in range(n):
                    lhs = slow_d(m) if j == 0 else fast_d(j, m)
                    rhs = (slow_d(k - m) if n - 1 - j == 0
                           else fast_d(n - 1 - j, k - m))
                    acc = acc + coef * bilinear(lhs, rhs)
            fast_cache[(n, k)] = inv_diag * (upper - acc)
        return fast_cache[(n, k)]

    out = base.copy()
    for n in range(1, order + 1):
        out = out + fast_d(n, 0)
    return out


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_direct_matches_v1_eigen_oracle(order):
    sys = Triad(coupling=0.05)
    got = balance_expansion(Z_TRIAD, order=order, ops=sys.ops()).state
    want = _v1_oracle_triad(Z_TRIAD, order, sys)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


# ================================================================
#  Test 2: eps-slope (T2), integration-based residual, N = 1, 2
# ================================================================
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("make_toy", TOYS)
def test_epsilon_slope_oscillation(make_toy, order):
    residuals = [oscillation_residual(*make_toy(c), order)
                 for c in EPS_RK4]
    slope = fitted_slope(EPS_RK4, residuals)
    assert slope >= order + 0.7


# ================================================================
#  Tests 3 + 4: slope tables (closure gate, scheme comparison)
# ================================================================
TOY_FACTORIES = {
    "triad": lambda c: (Triad(coupling=c), Z_TRIAD),
    "lorenz-k": lambda c: (LorenzK(coupling=c), Y_LK),
}
SCHEMES = ("direct", "telescoping")
CLOSURES = ("leading", "consistent")
ORDERS = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def residual_grid():
    """Differential residuals over the eps sweep for all variants."""
    grid = {}
    for toy, make in TOY_FACTORIES.items():
        for scheme in SCHEMES:
            for closure in CLOSURES:
                for order in ORDERS:
                    grid[(toy, scheme, closure, order)] = np.array([
                        differential_residual(*make(c), order,
                                              scheme=scheme,
                                              closure=closure)
                        for c in EPS_DIFF])
    return grid


def test_closure_slope_table(residual_grid):
    # the S1.4 empirical gate: report the full slope table; assert
    # only that consistent-closure slopes increase with order
    lines = ["", "closure/scheme slope table (orders 1..4, "
             f"eps in [{EPS_DIFF[0]:.3g}, {EPS_DIFF[-1]:.3g}]):"]
    slopes = {}
    for toy in TOY_FACTORIES:
        for scheme in SCHEMES:
            for closure in CLOSURES:
                s = [fitted_slope(
                    EPS_DIFF, residual_grid[(toy, scheme, closure, n)])
                    for n in ORDERS]
                slopes[(toy, scheme, closure)] = s
                lines.append(
                    f"  {toy:9s} {scheme:12s} {closure:11s}: "
                    + "  ".join(f"{x:5.2f}" for x in s))
    print("\n".join(lines))  # noqa: T201  (owner-gated decision input)
    for toy in TOY_FACTORIES:
        for scheme in SCHEMES:
            s = slopes[(toy, scheme, "consistent")]
            assert all(s[i + 1] > s[i] for i in range(len(s) - 1)), (
                f"consistent closure not monotone for {toy}/{scheme}: "
                f"{s}")


@pytest.mark.parametrize("make_toy", TOYS)
def test_all_variants_agree_at_order_two(make_toy):
    # closure corrections first enter at order 3, so all four
    # (scheme, closure) variants coincide at order 2
    sys, z = make_toy(0.05)
    ops = sys.ops()
    states = [
        balance_expansion(z, order=2, ops=ops, scheme=scheme,
                          closure=closure).state
        for scheme in SCHEMES for closure in CLOSURES]
    for other in states[1:]:
        np.testing.assert_allclose(states[0], other, rtol=0,
                                   atol=1e-14)


@pytest.mark.parametrize("closure", CLOSURES)
@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("make_toy", TOYS)
def test_schemes_coincide_at_every_order(make_toy, order, closure):
    # M2/R2 settled, and much stronger than the O(eps^(N+1))
    # agreement bound: with the corrected (V/W-split) derivative
    # line, the telescoping series is an exact term-by-term
    # resummation of the direct recursion, so the truncations
    # coincide to floating-point dust at every order and closure.
    # The historical "telescoping worse at order >= 3" cannot be a
    # property of consistent implementations of either scheme.
    for c in (EPS_DIFF[0], EPS_DIFF[-1]):
        sys, z = make_toy(c)
        ops = sys.ops()
        d = balance_expansion(z, order=order, ops=ops,
                              scheme="direct", closure=closure).state
        t = balance_expansion(z, order=order, ops=ops,
                              scheme="telescoping",
                              closure=closure).state
        assert l2(d - t) <= 1e-14 * l2(d)


@pytest.mark.parametrize("closure", CLOSURES)
@pytest.mark.parametrize("order", [3, 4])
def test_schemes_coincide_with_detuning(order, closure):
    # the identity also holds on a non-stationary slow space
    for c in (EPS_DIFF[0], EPS_DIFF[-1]):
        sys = Triad(coupling=c, delta=0.1 * c)
        ops = sys.ops()
        d = balance_expansion(Z_TRIAD, order=order, ops=ops,
                              scheme="direct", closure=closure).state
        t = balance_expansion(Z_TRIAD, order=order, ops=ops,
                              scheme="telescoping",
                              closure=closure).state
        assert l2(d - t) <= 1e-14 * l2(d)


def test_scheme_residual_constants(residual_grid):
    # M2/R2 report: which scheme has the smaller residual constant.
    # Since the truncations coincide identically, the ratio is 1 --
    # neither scheme wins anywhere (report, plus the trivial check).
    lines = ["", "telescoping/direct residual ratio "
             "(geom. mean over eps sweep, leading closure):"]
    for toy in TOY_FACTORIES:
        ratios = []
        for order in ORDERS:
            tele = residual_grid[(toy, "telescoping", "leading",
                                  order)]
            dire = residual_grid[(toy, "direct", "leading", order)]
            ratio = float(np.exp(np.mean(np.log(tele / dire))))
            ratios.append(ratio)
            assert ratio == pytest.approx(1.0, rel=1e-6)
        lines.append(
            f"  {toy:9s}: "
            + "  ".join(f"N={n}: {r:8.6f}"
                        for n, r in zip(ORDERS, ratios,
                                        strict=True)))
    print("\n".join(lines))  # noqa: T201  (owner-gated decision input)


# ================================================================
#  Test 5: generalized (non-stationary) slow space
# ================================================================
@pytest.mark.parametrize("closure", CLOSURES)
@pytest.mark.parametrize("order", [2, 3])
def test_generalized_slow_space_detuning(order, closure):
    # delta = 0.1 * omega_f * eps: keeping the L phi_0 slow-rotation
    # term must strictly beat the same recursion with it zeroed out
    with_term = []
    for c in EPS_DETUNE:
        sys = Triad(coupling=c, delta=0.1 * 1.0 * c)
        rw = differential_residual(sys, Z_TRIAD, order,
                                   closure=closure)
        rz = differential_residual(sys, Z_TRIAD, order,
                                   closure=closure, zero_lin=True)
        with_term.append(rw)
        assert rw < rz
    # the leading closure saturates the triad residual at slope 3
    # from order 3 on (see the closure table), so the order-slope
    # guarantee is asserted where the closure supports the order
    if closure == "consistent" or order <= 2:
        assert fitted_slope(EPS_DETUNE, with_term) >= order + 0.7
