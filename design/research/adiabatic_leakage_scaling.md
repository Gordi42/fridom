---
status: frozen
date: 2026-07-17
---

# Adiabatic leakage scaling in the ramp period — investigation

Owner question (2026-07-17): the shipped ramping gates showed diabatic
leakage decreasing only *algebraically* with ramp period `tau`
(oracle error 6.5e-2 / 4.0e-2 / 2.6e-2 at `tau` = 0.5/1/2), but the
adiabatic theorem for the Gevrey-2 "exp" ramp predicts
near-exponential decay, and the paper measured it. Defect or
measurement artifact? Probed on `dev` at R5 (single-leg experiments:
f-plane ramp `f0: 0 -> 1`, linearized sw2 channel, per the paper's
linear-leakage protocol).

## Verdict

**The machinery is correct.** The discrete AB3 propagator tracks the
continuous stretched-exponential adiabatic leakage to machine
precision; the shipped gates probed the pre-asymptotic small-`tau`
corner, and the projection-cycle oracle additionally floors at a
separate two-leg artifact.

## Evidence (32x32 channel, csqr=1, dt=0.15/32, seed 1, curve="exp")

Single up-leg, `eta = relative_imbalance(up(z0), P_slow_target)`,
`z0` random in the reference (f0=0) slow subspace:

| tau | eta exp | eta linear | ratio lin/exp |
|---|---|---|---|
| 2.5 | 6.20e-3 | 3.14e-3 | 0.51 |
| 5 | 1.50e-3 | 2.04e-3 | 1.4 |
| 10 | 4.95e-5 | 2.81e-4 | 5.7 |
| 20 | 2.94e-6 | 2.57e-4 | 87 |
| 40 | 2.61e-8 | 2.20e-4 | 8.4e3 |
| 80 | 5.66e-11 | 1.15e-4 | 2.0e6 |
| 160 | 4.50e-15 | 5.22e-5 | 1.2e10 |

- Fit (exp, pre-floor): `log eta = -2.52 sqrt(tau) - 1.32`,
  R^2 = 0.9968 (the linear-in-`tau` fit is systematically curved,
  R^2 = 0.943) — the `exp(-c sqrt(tau))` signature.
- Linear-curve control: algebraic (~1/tau) — the two ramps separate
  by 10 orders of magnitude at `tau` = 160. **The exp ramp only
  overtakes the linear ramp at `tau ≳ 5`**; below that they are
  comparable (at `tau` = 2.5 linear is slightly better). This is why
  `tau <= 3` gates cannot distinguish the curves.
- Floor diagnosis: at `tau` = 80, dt-halving changes eta by 1.0000
  (not stepper-limited — the measured leakage is the *continuous*
  adiabatic leakage); the `tau >= 160` floor (~2e-15) is float64
  roundoff. The AB cold-start bootstrap and `steps = round(tau/dt)`
  quantization inject **no** floor above 1e-15.
- Direction-symmetric (paper's up-then-down double-ramp diagnostic:
  slope -2.48, R^2 = 0.976); seed moves the prefactor ~10x, not the
  law; grid-converged by n ≈ 24 (slope stable from n = 16).

## The projection-cycle floor (separate artifact, correctly attributed)

Reproducing the shipped `AdiabaticProjection` oracle config (8x8,
beta=2, dt=1e-2) and extending `tau`: the oracle error collapses
root-exponentially, then floors at ~3.5e-7 — **bit-identical to the
cycle's idempotency residual** at every large `tau` (same on 16x16 at
9.4e-5). The floor is the two-leg cycle's reversibility residual
(O(dt) + multistep warm-up asymmetry), not ramp leakage: the pure
single-leg leakage on the same grid reaches 8.3e-11. Deep-leakage
studies must therefore use the **single-leg** diagnostic; the
projection cycle is bounded below by its own leg-dependent residual
(the plan's "leg-dependent tolerance" made quantitative).

## Consequences (applied)

- Regression shard
  `tests/model/transforms/test_adiabatic_ramping_exponential.py`
  (validated recipe: 16x16, dt=0.15/16, `tau` in {5,10,20,40}, both
  curves, ~3 s wall): asserts the sqrt-tau fit (slope < -1.5,
  R^2 > 0.98), the deep-vs-shallow collapse, and the exp/linear
  separation — a broken endpoint-flatness now fails loudly.
- The docs example's scaling figure switches to the single-leg
  diagnostic over `tau` in {5,10,20,40}, plotted against
  `sqrt(tau)`, with the linear-ramp overlay; prose states the
  `tau ≳ 5` crossover.
- Not tested here: the `gap^{3/2}` prefactor scaling (the f0-ramp's
  gap closes at the reference end; a clean gap sweep needs varying
  the target f or c). The verdict does not depend on it.
