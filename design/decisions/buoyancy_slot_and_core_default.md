---
status: decided; implemented on feat/buoyancy-slot
date: 2026-07-23
---

# The buoyancy slot (`buoyancy=`) and the default core

**Status: owner-ratified, 2026-07-23** (Silvano, in chat). Two preset
surface changes, informed by a survey of Oceananigans' buoyancy
configuration (`tracers=` x `buoyancy=` formulation objects with
`required_tracers` validation).

## 1. `stratification=` is renamed to `buoyancy=`

The slot never held "the stratification" alone: the module it takes
declares the buoyancy variable `b`, couples it into the momentum
equation, *and* (optionally) carries the background-stratification
restoring. The name only described the third job, and it stops making
sense the moment the slot holds a formulation without a background
N^2 (a bare buoyancy tracer) or, later, temperature/salinity with an
equation of state. `buoyancy=` names what the slot decides — the
buoyancy formulation — and is the spelling Oceananigans users already
know. The old kwarg raises a taught TypeError on both presets
(`nh.Model`, `hy.Model`).

Deliberately **kept**: the class names `ConstantStratification` /
`MeridionalStratification` (they honestly describe formulations
carrying a background stratification) and the framework parameter
names `stratification.n2` / `stratification.froude` (N^2 *is* the
stratification; the params thread through diagnostics, eigenmodes and
energy of both packages). Only the preset slot is renamed.

Unlike Oceananigans, FRIDOM needs no `tracers=`/`required_tracers`
validation handshake: a formulation module *declares* its own tracers
(`FieldDeclaration.tracer`), so a formulation/tracer mismatch cannot
be spelled. Future formulations (`SeawaterBuoyancy` with T/S + an
equation of state, the Oceananigans `constant_temperature=` /
`constant_salinity=` reductions) drop into the same slot with no
framework work.

## 2. `nh.BuoyancyTracer` — a bare buoyancy tracer

`nh.BuoyancyTracer()` registers `b` and contributes the buoyancy
force alone. It is the `N^2 = 0` physics of
`ConstantStratification(n2=0)` with the restoring term absent from
the assembly instead of multiplying by zero (one fewer step-path
term). Scaling-neutral; provides no `stratification.*` scalar, so
N^2-consumers (the `1/N^2` energy metric, internal-wave eigenmodes)
refuse such a model through the missing provide, as they do a
buoyancy-less one.

## 3. `core=` defaults to `nh.Core()` on the nonhydrostatic preset

`core=None` (now the default) assembles a plain `nh.Core()` — the
common case carries no physics choice (aspect ratio 1, default solver
knobs), so requiring the argument taught nothing. **Not** extended to
`hy.Model`: `hy.Core()` with no kwargs is the *nondimensional*
variant (`gravity=None`), so a default core would refuse every
dimensional assembly — the B-3 "no default physics" ruling
(`nondimensionalization_plan.md`) stands unamended for the
hydrostatic preset's `core=`, `buoyancy=` and `free_surface=`; only
the slot name changed.
