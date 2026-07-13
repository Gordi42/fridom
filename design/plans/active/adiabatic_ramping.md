---
status: idea
date: 2026-07-13
---

# Generalized adiabatic ramping (OptimalBalance as a subclass)

Owner request, 2026-07-12. **Not scheduled** — recorded so the shape is
not lost. ROADMAP 3.8.

## Where reality stands (2026-07-13)

The dependency is discharged: ROADMAP 2.8 (state transforms) **ships**.
`fridom.model.transforms` carries `StateTransform` and its algebra,
`FixedPoint`, `Shift`, `Identity`, `Propagator`, `TimeAverage`, the
projections, and a working `OptimalBalance`
(`src/fridom/model/transforms/optimal_balance.py`, covered by
`tests/model/transforms/test_optimal_balance.py`). Nothing named
`AdiabaticRamping` exists.

Two details of the shipped code correct the earlier sketch:

- `OptimalBalance` ramps `scaling.rossby` from `0` to the **model's own
  nominal Rossby value** (so `rossby_number=0.1` ramps `0 -> 0.1`), not
  `0 -> 1`.
- The ramp vocabulary is `Ramp(v0, v1, period=..., t0=..., curve=...)`
  with `curve` in `{"linear", "cosine", "exp"}` or a callable with
  `shape(0)=0, shape(1)=1` (`src/fridom/model/time_dependent.py`) — not
  the old `ramp_type` names `exp`/`pow`/`cos`/`lin`. Evaluation is
  continuous stage-time, branch-free, and already zero-recompile under
  endpoint sweeps.

## The idea

`OptimalBalance` is today a bespoke Tier-2 transform: two owned
`Propagator` legs with a `Ramp`-valued `scaling.rossby` (up on the
forward leg, `Ramp.reversed()` down on the backward one, `TIME_STEP`
sign-flipped), wrapped in a `FixedPoint` around
`Shift(z_base) @ (Identity - P) @ ramp_cycle`.

Everything there except *the choice of what is ramped* is generic. So
invert the hierarchy:

- **`AdiabaticRamping`** — the base transform: take a model, ramp one or
  more **declared parameters** from a start to an end value over a ramp
  period, integrating the model as the parameter moves, ramp shape a
  parameter of the transform.
- **`OptimalBalance(AdiabaticRamping)`** — the special case: the ramped
  parameter is the Rossby number, the cycle is forward-then-backward
  with the base-point exchange in between, and the whole thing sits in a
  `FixedPoint`. It contributes only the balancing policy (the base
  point, the `(Identity - P)` leg, the backward-leg term filter), not the
  ramping machinery.

## What it actually buys (and what it does not)

The machinery is largely already there, which cuts both ways.
`Propagator(model, steps=..., backward=..., updates={param: Ramp(...)},
term_filter=...)` **already** drives any declared parameter with a ramp
over an internal model run, and `resolve_at` makes every scalar
parameter slot Ramp-able with no consumer changes. So a user can do
adiabatic parameter continuation today by composing a `Propagator` by
hand.

`AdiabaticRamping` is therefore an **ergonomics and naming layer plus a
re-homing of OB's leg construction**, not new capability. Its value:

- a named, documented surface for **adiabatic spin-up / parameter
  continuation** — ramp a forcing amplitude, a stratification, or a
  Coriolis parameter from a regime where the balanced state is known
  into the target regime;
- **slow-manifold initialization** variants other than optimal balance;
- one place where "how many steps does a ramp period snap to", ramp
  reversal, and multi-parameter ramps are settled, instead of once
  inside `OptimalBalance` and once in every user script.

That is a real but modest payoff, and it is why the item stays
unscheduled: build it when a second consumer (a continuation study, a
spin-up example) actually appears.

## Open questions (for when this is picked up)

- Does the base ramp **parameters** only, or any traced quantity? The
  C4 answer is now in: `MovingGeometry`
  (`src/fridom/model/modules/moving_geometry.py`) shipped with
  user-supplied schedule callables over AUXILIARY parameter fields and
  does **not** go through `Ramp`. So the geometry morph is *not* a
  consumer of this surface; the clean seam is parameters only, and the
  two merely share the "time-dependent scalar" idea.
- Is the fixed-point iteration part of the base (some continuation
  methods want it) or purely optimal-balance policy? Current reading:
  **policy** — keep `FixedPoint` in the subclass.
- Rewriting `OptimalBalance` as a subclass must not perturb its shipped
  behaviour (base-point exchange, cost accounting, divergence policy);
  the existing tests are the gate.
