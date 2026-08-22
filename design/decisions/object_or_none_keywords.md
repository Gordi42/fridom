---
status: decided; implemented on feat/object-or-none-keywords
date: 2026-08-22
---

# Object-or-None keywords: `advection=` and `progress=` take no boolean

**Status: owner-ratified, 2026-08-22** (Silvano, in chat), superseding
the bool shorthand left in place by
[`no_default_advection.md`](no_default_advection.md) the same day.

## The rule

A keyword that can take an object takes the object or `None`. A
boolean is for a flag with no object behind it (`jit=`, `debug_nan=`,
`raise_on_nan=` on `run`). `None` means "nothing installed"; the
object names what is installed. The rule is recorded in
`design/specs/model/01_concepts.md` next to the preset factory rule.

## Why

`advection=True` installed a scheme the call site never named, which
is the defect that started the no-default ruling; keeping `True` as a
shorthand kept that defect one keyword away. The presets already spell
every other physics slot as `Module | None` (`coriolis=`, `buoyancy=`;
`core=`/`free_surface=` required), so `advection=` was the odd one out,
and its bool union forced `is True` / `is not False` branching in every
preset. `progress=True` was worse: it was not the progress bar but a
logging placeholder, which is why a hundred test sites passed
`progress=False` to silence something nobody asked for.

## What changed

- `hy.Model`, `nh.Model`, `sw.Model`: `advection: Module | None = None`.
  A boolean raises a taught `TypeError` naming `None` and the module
  spelling (`fr.model.modules.CenteredAdvection()`,
  `nh.CenteredAdvection()`, `sw.SadournyAdvection()`).
- `hy.Model(surface_advective_flux=...)` is retired (taught error): it
  only configured the module the preset used to install unasked; the
  closure is `CenteredAdvection(surface_flux=...)` on the module the
  caller passes. `hy.comparison_model` keeps its own keyword, since it
  builds that module itself.
- `sw.SadournyAdvection` is exported at the package root, and its
  `coords=` defaults to `None`: the (zonal, meridional) names are
  adopted from the grid at bind, in factor order, from the chart on a
  chart grid and from the mesh names on a flat one. A prescribed
  `background=` still needs explicit `coords=`, because its fields are
  declared before the grid is known.
- `model.run(progress=None)` and `fr.ops.Session(progress=None)`
  report nothing; a reporter object (`fr.ops.ProgressBar()`) is used
  as-is; a boolean raises a taught `TypeError`. The logging placeholder
  `_LoggingProgress` is dropped.
- The repository was swept: `advection=False` became `advection=None`,
  `advection=True` the package's module, `progress=False` was deleted
  (it is the default). Test helpers that forwarded a bool build a fresh
  module per call, since a module instance binds to one model only.

## Follow-ups

The pending gallery branches `docs/optimal-balance` and
`docs/spherical-jet` construct their models without `advection=` and
meant the nonlinear model; they get the module at inclusion.
