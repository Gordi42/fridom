---
status: idea
date: 2026-07-12
---

# Generalized adiabatic ramping (OptimalBalance as a subclass)

Owner request, 2026-07-12. **Not scheduled** — recorded so the shape is
not lost. Depends on ROADMAP 2.8 (state transforms) being in place.

## The idea

Today `OptimalBalance` is designed as a bespoke Tier-2 transform
([`specs/model/08_state_transforms.md`](../../specs/model/08_state_transforms.md)):
two owned model variants with a `Ramp`-valued `scaling.rossby` (up on
the forward leg, down on the backward one), a sign-flipped `TIME_STEP`
on the backward leg, and a `FixedPoint` around
`Shift(z_base) @ (Identity - P) @ ramp_cycle`.

Every one of those pieces except the *choice of what is ramped* is
generic. So invert the hierarchy:

- **`AdiabaticRamping`** — the base transform. It owns the general
  procedure: take a model, adiabatically ramp one or more **declared
  parameters** from a start value to an end value over a ramp period,
  integrating the model as the parameter moves, with the ramp shape a
  parameter of the transform (`"exp"`, `"pow"`, `"cos"`, `"lin"`, or a
  user callable — the old `ramp_type` vocabulary, now continuous
  stage-time `Ramp` evaluation rather than piecewise-constant
  `theta = n/N`).
- **`OptimalBalance(AdiabaticRamping)`** — the special case: the ramped
  parameter is the **Rossby number** (0 -> 1), the cycle is
  forward-then-backward with the base-point exchange in between, and
  the whole thing is wrapped in a `FixedPoint`. It contributes the
  balancing-specific policy (the base point, the `(Identity - P)`
  projection leg, the backward-leg term filter), not the ramping
  machinery.

## Why it generalizes cleanly

The parameter-in-modules design (model D2) already makes any parameter
a first-class, traceable thing that a transform can drive; the `Ramp`
scaling machinery already exists for exactly this; and
`model.variant(term_filter=...)` already builds the owned variants.
So the base class is mostly a re-homing of code that has to exist for
optimal balance anyway.

Other members that fall out of the same base, and are the reason to
build it:

- **adiabatic spin-up / parameter continuation** — ramp a forcing
  amplitude, a stratification, or a topography parameter from a
  regime where the balanced state is known into the target regime;
- **slow-manifold initialization** other than optimal balance (e.g.
  ramping the Coriolis parameter, or a nonlinearity switch);
- **the sloped-to-flat geometry morph** of the C4 moving-geometry work
  is itself an adiabatic ramp of a mapping parameter — worth checking
  whether it should be expressed through this surface rather than its
  own schedule module (`MovingGeometry`), or whether the two should
  merely share the `Ramp` vocabulary.

## Open questions (for when this is picked up)

- Does the base class ramp **parameters** only, or any traced quantity
  (a field, a geometry schedule)? The C4 morph suggests the latter is
  wanted; the model-D2 parameter surface suggests the former is the
  clean seam. Decide before writing the base.
- What is the honest interface for "integrate while the parameter
  moves"? Optimal balance needs the *ramped* legs to be `Propagator`s
  over a stage-time-dependent model; that is exactly what the
  `Ramp`-valued scaling gives, so the base should expose it directly.
- Is the fixed-point iteration part of the base (some continuation
  methods want it) or purely optimal-balance policy? Current reading:
  **policy** — keep `FixedPoint` in the subclass.
