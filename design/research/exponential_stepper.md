---
status: frozen
date: 2026-07-14
---

# Exponential time stepping for the shallow-water gravity-wave stiffness

Research report (see [`README.md`](README.md) for status). Motivating
question (Silvano, 2026-07-14): the shallow-water model needs a very
small `dt` under Adams-Bashforth, apparently because of the high
eigenvalues of the linearized system at high wavenumber. Can we do
better — an IMEX stepper? And would an IMEX stepper break the discrete
eigenanalysis?

**Answers, in order.** The diagnosis is right and the stiffness is
exactly the gravity CFL. IMEX is the intuitive fix and it is a **trap**
here. The eigenanalysis is *not* what an implicit treatment endangers.
The method that actually works is an **exponential integrator**, which
ships as `fr.model.time_steppers.ETDRK4`
([`../../src/fridom/model/time_steppers/exponential.py`](../../src/fridom/model/time_steppers/exponential.py)).

## 1. The stiffness is the gravity CFL, and explicit RK cannot fix it

AB3's imaginary-axis stability limit is `|omega*dt| <= 0.7236`. The
C-grid discrete gravity symbol maxes at `|omega| = 2*sqrt(2)*c/dx`
(2-D, `dx = dy`). Hence

    dt <= 0.7236 / (2*sqrt(2) * c/dx) = 0.2558 * dx/c

**independent of the flow speed** — a pure gravity-wave CFL. Measured
on a 32x32 channel (`c = 1`, `dx = 1/32`): the analytic limit is
`0.00799`, and AB3 blows up between 0.9x and 1.1x it, at `0.0082-0.0083`
for every Rossby number tested. Confirmed.

The eigenbasis engine agrees: `max|omega| = 90.46` against the analytic
`2*sqrt(2)*c/dx = 90.51`.

Explicit RK is not a way out. Imaginary-axis limits: AB3 0.7236, RK3
1.7321, RK4 2.8284. RK3 buys 2.4x the step for 3x the cost, RK4 3.9x
for 4x — a wash. (AB2 with the `eps = 0.01` damper is only 0.195, and
textbook AB2 is 0.0045: AB3 is already the right explicit choice.)

## 2. The IMEX trap: Crank-Nicolson is neutral, so it has no margin

The IMEX family already in the repo (`CNAB2`, `SBDF2`) is **worse than
AB3** on this problem — `dt_max ~ 0.005-0.03`, i.e. 25-50x *worse*.

The reason is structural, not a bug. Classical IMEX theory
(Ascher-Ruuth-Wetton) assumes the implicit part is **dissipative**
(diffusion, real negative eigenvalues). Here the implicit part is a
**wave** (purely imaginary eigenvalues), and Crank-Nicolson on an
imaginary eigenvalue is *exactly neutral*: `|amp| = 1.000000000000`,
verified out to `omega*dt = 1e8`. Zero damping margin — so any
oscillatory explicit part (the advection) pushes the root outside the
unit circle. Measured joint stability, explicit limit as a function of
the implicit phase:

| `|omega_lin*dt|` | 0 | 0.5 | 1 | 2 | 10 | 100 |
|---|---|---|---|---|---|---|
| CNAB2 explicit limit | 0.0045 | 0 | 0 | 0 | 0 | 0 |
| CNAB3 (CN + AB3) | 0.7236 | 0.5236 | 0.3236 | 0 | 0 | 0 |

The only non-damping A-stable theta is `1/2`, and it is precisely the
one that fails. **A stable IMEX for this system must damp the waves.**
An off-centred theta-method (`theta > 1/2`) with AB3 explicit does work
— `dt_max` jumps to the advective CFL — but the speed-up is *bought
with* the damping:

| theta | speed-up (Fr = 0.1) | amplitude lost per wave period, 16dx wave |
|---|---|---|
| 0.50 | unstable | — |
| 0.52 | 3.1x | 3.8% |
| 0.60 | 5.3x | 27% |
| 0.70 | 14.0x | 69% |

The geostrophic mode (`omega = 0`) takes no implicit damping, so this
is a balanced-model-with-filtered-waves. Legitimate for vortical
experiments; wrong whenever the inertia-gravity waves are the physics.

## 3. IMEX does **not** break the eigenanalysis (the actual answer)

Two separate things get conflated:

- **The spatial eigenanalysis** (`sw.Eigenmodes`: `omega(s)`, `q(s)`,
  `projector(s)`) is built purely from the grid's operator symbols
  (`GridSymbols`). It never mentions the stepper. No implicit treatment
  changes a spatial operator, so mode projections, optimal balance,
  geostrophic ICs and the energy decomposition are bit-for-bit
  untouched.
- **`time_discretization_effect(omega)`** is the only stepper-dependent
  piece. For a **fully-linear-implicit** split it stays a closed form:
  the eigenanalysis linearizes about rest, where advection contributes
  nothing, so the explicit eigenvalue is exactly zero and the
  amplification collapses to `x = (1 + (1-theta) z)/(1 - theta z)`,
  `z = -i omega dt`.

The deeper point: the discrete propagator of a linear-implicit scheme is
a **rational function of A**, and a rational function of A has *exactly
the same eigenvectors as A*. The mode decomposition therefore stays an
exactly invariant subspace of the time-stepped linear dynamics — just as
it is under explicit AB, whose propagator is a *polynomial* in A.

**The rule that follows: never split the linear operator.** If only part
of L is implicit (gravity implicit, Coriolis explicit), the two pieces
do not commute, the propagator is no longer a function of A alone, its
eigenvectors drift from A's, and a scalar `omega` is genuinely
insufficient information for `time_discretization_effect`. Putting *all*
of L implicit is both the stiffness-optimal and the analysis-preserving
choice; the two requirements coincide.

So the price of IMEX is **not** the eigenanalysis. It is wave damping.

## 4. The exponential integrator: exact linear part, no damping

`Eigenmodes.function(f, sel)` / `ChannelEigenmodesBase.function(f, sel)`
already applies *any* scalar function of the linear operator
(`sum_s P^s f(omega^s)`). The columns satisfy `L q = -i omega q`, so the
**exact** linear propagator is one line: `f = exp(-i omega dt)`.

Splitting `dX/dt = L X + N(X)` with L every `linear=True` term (gravity,
Coriolis, and `sadourny`'s `background_advection`) and integrating the
variation-of-constants formula exactly per mode gives:

- **no gravity CFL** — `dt` is set by the nonlinear scale;
- **no damping** — `|exp(-i omega dt)| = 1` exactly;
- **`time_discretization_effect` is the identity** — the eigenanalysis
  needs no correction at all.

Verified: the linear-only answer is *independent of dt* (identical at
1x, 5x, 25x and **125x** the AB3 limit — one step of `dt = 125x` the
stability limit lands on the right state); the group law
`exp(L dt)^2 = exp(L 2dt)` holds to `7.4e-15`; energy is conserved to
`1.000000000000` over 400 steps at 31x the AB3 limit.

Crucially the engine is the **channel** eigenbasis, so this is not
restricted to periodic/constant-coefficient runs: one bounded axis with
`f(y)` (beta plane) and `c^2(y)` (variable depth) is diagonalized
densely and works unchanged.

### 4.1 Multistep exponential (ETD-AB) is the second trap

The intuitive choice — exponential *Adams-Bashforth* — only reaches
2-7x. Same root cause as CN: the exact propagator of an oscillatory L is
**neutral**, so there is no damping margin, and the AB *extrapolation
across a fast rotating phase* resonates. The measured explicit stability
limit of ETD-AB3 collapses to zero near `omega*dt ~ 2-3`, and the
measured `dt_max` lands exactly at `omega_max*dt ~ 1.6-4.5`. The cap is
the multistep wrapper, not the advective CFL.

(A plain integrating factor, IF-AB, is worse still: it extrapolates
`e^{-Lt} N`, whose error constant grows like `(omega h)^3`. The ETD
coefficients are the phi-functions, which stay bounded as `|z| -> inf`.)

### 4.2 ETDRK4 is the answer

A Runge-Kutta stage structure has no cross-step phase to extrapolate.
Cox-Matthews ETDRK4, with the phi-function quadrature weights, measured
against AB3 on the same 32x32 channel (`|u|max = 1`, so `Fr = Ro`):

| Ro | AB3 | ETD-AB3 | **ETDRK4** |
|---|---|---|---|
| 0.40 | 0.00826 | 0.01837 (2.2x) | 0.11416 (**13.8x**) |
| 0.20 | 0.00836 | 0.02103 (2.5x) | 0.46670 (**55.8x**) |
| 0.05 | 0.00842 | 0.06073 (7.2x) | 2.14431 (**254.5x**) |

Cost is ~5x per step (four nonlinear evaluations plus nine spectral
contractions), and it still wins on wall-clock *and* accuracy: at
`Ro = 0.05`, `T = 1`, AB3 needs 139 steps for `9.0e-5` error while
ETDRK4 at 31x needs 4 steps for `5.0e-5` in a quarter of the time. **The
step size is set by accuracy, not stability** — the point of a stiff
solver.

The coefficients tend to `h/6` as `z -> 0`, so ETDRK4 degenerates to
classical RK4 exactly when the linear operator is absent; ETD-AB
likewise degenerates to the textbook AB rows. Both are strict
generalizations, which is a useful correctness check.

## 5. Contracts and limits

- **Double counting.** The stepper supplies L itself, so the model must
  *not* also carry the linear terms in its tendency. Assemble with
  `term_filter=~terms.linear` and build the eigenbasis from the
  UNFILTERED model. This is silently-wrong physics if forgotten, so
  `ScheduleEntry.linear` was plumbed through the composer and the
  stepper raises `LinearTermInTendencyError` at trace time.
- **Time-dependent parameters.** A parameter that lives in **N** (a
  `Ramp` on `scaling.rossby` — the optimal-balance ramp) is fully
  correct, and the RK stages are evaluated at their own clock times
  (`clock.shifted(c_i * dt)`) so fourth order survives it. Evaluating
  all four stages at `t_n` silently drops the scheme to first order —
  this was a real bug in the first prototype and is now a test.
- **A time-dependent L is NOT supported.** `L(t1)` and `L(t2)` do not
  commute (`[L_gravity, L_coriolis] != 0`), so `exp(L dt)` stops being
  the propagator. Today the question is moot — `f` and `csqr` are
  materialized as AUXILIARY *fields*, so `FPlaneCoriolis(f0=Ramp)`
  raises — but see [`../roadmap/open.md`](../roadmap/open.md) for the
  time-dependent-field thread. The recipe that *does* work, and is
  measured: keep the stiff, time-INDEPENDENT part (gravity) in the
  eigenbasis and leave the time-dependent part (rotation) in the
  tendency. `dt` is then capped by the inertial limit (`f*dt ~ 0.42`
  measured) rather than the gravity CFL — still **52.7x** AB3, because
  `f << c*k_max`. The exactly-exact special case is
  `L(t) = g(t) * L0` (a single scalar modulation), where L commutes with
  itself at different times and the phase is `integral(omega dt)`; `f(t)`
  alone does not have that structure.
- **Memory.** The channel engine stores `q` as `(n_kx, D, D)` with
  `D = 3*N_y`, i.e. O(N^3): fine at `N = 128` (~150 MB), heavy at 256
  (~1.2 GB), impractical at 512. This is a method for the idealized,
  analysable configurations — which is exactly where it is wanted. The
  fully periodic constant-coefficient case stays cheap (3x3 per mode)
  and is the natural next engine to wire.

## 6. What shipped

`fr.model.time_steppers.ETDRK4` and its `phi_functions` kernel;
`ScheduleEntry.linear`; `LinearTermInTendencyError`;
`fridom.model._eigenbasis.fourier_ops` (de-privatized so the stepper can
reuse the transform stack). ETD-AB was **not** shipped — section 4.1 is
its epitaph.
