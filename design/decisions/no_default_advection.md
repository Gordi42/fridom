---
status: decided; implemented on feat/no-default-advection
date: 2026-08-22
---

# No advection by default in the presets

**Status: owner-ratified, 2026-08-22** (Silvano, in chat, during the
coastal-upwelling example review). The three preset factories
`hy.Model`, `nh.Model` and `sw.Model` install **no advection scheme
unless asked**: the default of `advection=` is `False`.

## Why

The previous default (`advection=True`, installing `CenteredAdvection()`
on the two 3-D presets and the Sadourny module on the shallow-water one)
put a numerical scheme into a model that never named it. It surfaced in
review when a stretched-mesh example, which could not use the WENO
scheme it had named, silently ran second-order centered advection once
the keyword was dropped: nothing on the page said which scheme was
moving the tracer. The presets already refuse surprising physics in
every other slot (`core=` and `free_surface=` are required on the
hydrostatic preset, `buoyancy=` and `coriolis=` are opt-in on all of
them); the advection scheme is a choice of the same weight, and a
linear model is the honest baseline.

## What changed

- `hy.Model(advection=False)`, `nh.Model(advection=False)` and
  `sw.Model(advection=False)` are the defaults. The keyword keeps its
  meanings: a module instance is installed as given, `True` is the
  shorthand for the package's default-constructed scheme
  (`CenteredAdvection()`; Sadourny for shallow water, whose keyword is
  a plain bool).
- `hy.Model(surface_advective_flux=...)` shapes the `advection=True`
  module only, as before it shaped the default-constructed one.
- Every caller that meant the nonlinear model now says so: the gallery
  (`shallowwater/barotropic_instability.py`, the coastal upwelling page),
  the step benchmarks (`benchmarks/model/bench_step.py`, so the guard
  baselines keep measuring the nonlinear step), the docstring examples
  of `Tracer` and the spherical `sw.Model`, and the tests that relied on
  the default.

## What stays

The explicit-assembly path (`fr.model.Model(modules=...)`) is untouched,
the advection modules themselves are untouched, and the old stack is
not concerned. The tutorials under `docs/source/tutorials/` still show
the old-stack `sw.Model(mset)` and are unaffected until their own port.

## Follow-ups

The pending gallery branches authored before this ruling
(`docs/optimal-balance`, `docs/spherical-jet`) construct their models
without `advection=` and meant the nonlinear model; each gets the
explicit keyword when it is included.
