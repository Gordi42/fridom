---
status: frozen
date: 2026-07-17
---

# ALE on FV — does the mesh-velocity flux telescope on cell averages?

Research report (see [`README.md`](README.md) for status). Question,
from the roadmap's FV section: `MeshVelocityCorrection` is nodal-only,
so moving-geometry models are carved out of the FV default. What
exactly is the gap, does the mesh-velocity flux telescope on cell
averages, and what are the options for closing it? Method: code sweep
(operator registry, FV rows, retag seam), design-history sweep
(C4/CS-D4, FV scoping rulings, validation gates), literature survey
(GCL/DGCL, ALE-FV, ocean-model practice), plus the Reynolds-transport
derivation checked against both. AI-assisted; every load-bearing
file:line and ruling was spot-checked on `dev` at `7948bff2`.

## Verdict

**Yes — the mesh-velocity flux telescopes on cell averages.** The FV
counterpart of the nodal advective correction
:math:`\dot m\,\partial f/\partial m` is the Reynolds-transport flux
form over the moving control volume,

```
d f̄/dt |_cell = ⟨∂f/∂t|_x⟩  +  (1/V) [ (f w)_top − (f w)_bot  −  f̄ (w_top − w_bot) ]
```

whose pure-flux part telescopes exactly (interior face terms cancel
pairwise) and whose divergence part accounts exactly for the cell's
volume change. Properties, all fully discrete in space:

- **Constancy is exact** under any time integrator: reconstruction
  reproduces constants, so the bracket is identically zero for
  uniform `f` — the free-stream/GCL condition holds by construction,
  per stage. (The nodal advective form also has this; nothing lost.)
- **Tracer conservation becomes exact semi-discretely**: summed over
  cells, `Σ V·correction = boundary terms − Σ f̄ V̇`, so
  `d/dt Σ V f̄ = Σ V·physics + boundary` with zero spatial residual.
  Composed with F5's telescoping J-weighted FV advection, the whole
  spatial tracer budget closes; the C4 morph-test drift (6.9e-3,
  "no exact discrete invariant exists") collapses to a pure
  time-integration residual, resolution-independent. This is the new
  invariant closing the gap buys — the exact analogue of what F5's
  conservative advection bought on terrain (`∫(J·τ) = 0.0` exact vs
  O(1) nodal drift).
- **Exact *fully-discrete* conservation is not attainable in the
  intensive formulation**: advancing intensive `f̄` while `V(t)`
  changes leaves an O(Δt^p) residual in `Σ V f̄` because the discrete
  product rule fails for multi-stage/multi-step integrators. Exact
  budget closure needs extensive (volume-weighted) prognostics or a
  MITgcm-style step-boundary thickness rescale — option D below,
  recorded as a horizon, not proposed.

Recommended: **option C** — flux-form FV ALE as module-level
family-aware routing, the F5 pattern, no spatial-layer changes
expected. Sketch and test plan in §6; two owner calls in §7.

## 1. The gap, precisely (code level)

What ships today (`src/fridom/model/modules/moving_geometry.py`):
`MeshVelocityCorrection._correct` computes, per configured field,
`(mdot * dfdm).retag(f)` with `dfdm` from `_column_derivative`:
`physical_diff` along the mapped column, then an unconditional
`resolve("interpolate", ...)` hop back, then the pointwise product.

The break is narrower than "the column derivative is nodal":

- **`physical_diff` is already family-aware.** On a `CellAvg` column
  factor it resolves `("diff", CellAvg)` → the `FVDerivative` chain
  (`FluxDifference @ Dispatched("reconstruct")`,
  `spatial/operators/flux_diff.py:667`), i.e. `CellAvg → CellAvg`,
  metric-scaled by `MappedDerivative.expand`
  (`spatial/operators/mapped.py:398-458`).
- **The interpolate hop is wrong on FV, not missing.** After a
  nodal diff (`Center → Right`) the hop `("interpolate", Right)`
  correctly returns to `Center`. After `FVDerivative`
  (`CellAvg → CellAvg`) the same code resolves
  `("interpolate", CellAvg)` = `LinearReconstruction`
  (`CellAvg → Right|Inner`, the G4 seeding, `spatial/grid.py:2223`)
  — it moves the result *onto the faces* instead of leaving it on
  the cell. The chain was routed for the nodal row only.
- **`retag` can never cross the family.** `_bc_siblings`
  (`spatial/fields/scalar_field.py:1256-1261`) requires
  `type(src) is type(dst)`; `Center` is a `NodalSpace`, `CellAvg` an
  `AverageSpace`. Retag is a BC-tag swap by contract — the F4 walled
  solve depends on that (`div.retag(neumann_cellavg_sibling)`), so
  widening it is not on the table.
- **The staggering splits the work.** FV prognostics are average
  only transversely: `b` (and `u`, `v`) carry `CellAvg` on the
  column axis → the flux form applies; `w` is a point value
  (`Right`) on its own column axis → its correction stays advective
  at the face node, resolved through the average-kind hops — exactly
  how F5 advection treats the mixed staggering
  (`model/modules/advection.py:1649-1667`, `2335-2370`).

The carve-out around the gap (mapped-FV-default record, scoping §13
addendum): `dynamic_geometry` is a *model* property computed by the
`nh.Model` factory from `MovingGeometry` presence
(`nonhydro2/model.py:161-165`) and threaded to
`_fv_capable`/`resolve_model_family` (`nonhydro2/modules/core.py:67`,
`:289`); auto-defaults stay nodal for moving geometry, and explicit
`family="fv"` + `MeshVelocityCorrection` is a taught
`NotImplementedError` at `bind` (`moving_geometry.py:538-553`,
covered by `tests/nonhydro2/test_fv_default.py`
`test_dynamic_mapping_auto_stays_nodal_and_ale_fv_is_a_taught_gap`).

## 2. The physics: Reynolds transport on the moving control volume

For a column cell with faces at `z_bot(t), z_top(t)`, face velocities
`w_bot, w_top`, volume `V = z_top − z_bot`:

```
d/dt ∫_V f dV = ∫_V ∂f/∂t|_x dV + (f w)_top − (f w)_bot        (Reynolds transport)
dV/dt        = w_top − w_bot                                    (space conservation law)
```

Product rule on `V f̄` gives the boxed identity in the verdict. The
split matches fridom's architecture exactly: the FV physics modules
already supply `⟨∂f/∂t|_x⟩` (flux differences over the *current*
cell, metrics through the `params=` seam), so the ALE module adds
only the bracket — the same contract as the nodal module, promoted
to cell averages.

Key structural facts:

- **The spatial GCL is exact by construction.** Face positions are
  `M(ξ_face, params(t))` and the mesh velocity is
  `Σ_p (∂M/∂p) ṗ` with `ṗ` from `jax.jvp` of the schedule — the
  face velocity *is* the exact time derivative of the face position,
  so `dV/dt = w_top − w_bot` holds identically. No discrete-GCL
  construction is needed at the semi-discrete level.
- **Consistency**: the bracket = `w ∂f/∂m + O(Δm²)` — 2nd-order
  consistent with the nodal advective form, matching the family's
  order everywhere else (the midpoint identification the
  profiles-follow-the-family ruling already blesses).
- **The constancy/conservation duality** (the honest boundary):
  intensive prognostics get exact constancy for free and exact
  conservation only semi-discretely; extensive prognostics
  (`V f̄`) get exact conservation for free and need the geometry
  stepped with the *same* integrator weights to recover constancy
  (the DGCL condition — precisely the small RK3 tracer drift
  Oceananigans' ZStar hit and fixed by stepping `σ` with the tracer
  stepper). One property is free per formulation; both at once is a
  formulation-plus-integrator design, not a module patch.

## 3. Literature anchors

- GCL/DGCL: Thomas & Lombard 1979 (AIAA J. 17); Farhat, Geuzaine &
  Grandmont 2001 (JCP 174) — DGCL is necessary *and* sufficient for
  inheriting fixed-grid nonlinear stability; Guillard & Farhat 2000
  (CMAME 190) — sufficient for 1st-order time accuracy; Étienne,
  Garon & Pelletier 2009 (JCP 228) — beyond 1st order neither
  necessary nor sufficient; Mavriplis & Yang 2006 (JCP 213) — exact
  analytic face velocities do *not* by themselves satisfy the DGCL
  under BDF/IRK; the temporal residual is a quadrature-consistency
  mismatch, not a velocity error. (In the intensive flux form this
  surfaces as the conservation residual, never as a constancy error
  — the bracket vanishes on constants regardless of integrator.)
- ALE-FV form: Hirt, Amsden & Cook 1974 (JCP 14); Donea, Huerta,
  Ponthot & Rodríguez-Ferran 2004 (Encycl. Comp. Mech. ch. 14) —
  the flux-minus-divergence correction is the standard ALE integral
  form. Yamazaki, Weller, Cotter & Browne 2022 (JCP 461) is the
  closest analogue: smoothly moving (non-remap) FV mesh over
  orography targeting exact local conservation + uniform-field
  maintenance.
- Ocean practice: **MITgcm z\*** (Adcroft & Campin 2004; Campin et
  al. 2004) is the direct precedent for the intensive route — it
  achieves exact conservation with AB via a step-boundary
  thickness-ratio rescale `θⁿ⁺¹ = (hⁿ/hⁿ⁺¹)[θⁿ + Δt G]`, i.e. by
  making the step effectively extensive. **MOM6** (Griffies,
  Adcroft & Hallberg 2020), **NEMO z̃** (Leclair & Madec 2011),
  MPAS-O, and **Oceananigans ZStar** (PR #3956; constancy fix PR
  #4546) all prognose extensive `h·θ`. The compatibility theorem
  (Gross, Bonaventura & Rosatti 2002, IJNMF 38): discrete
  consistency between continuity and tracer transport is necessary
  for constancy — automatic in fridom's intensive form because the
  divergence term is built from the same face velocities as the
  flux term.

## 4. Constraints from standing rulings

- **The nodal-sibling shortcut is the rejected pattern** (F4 ruling,
  scoping `:471-482`): "bitwise-equivalent at 2nd order but bypasses
  the type discipline exactly where the codebase invests in it."
  Any closure must produce the correction *on the average family*,
  not retag a nodal result.
- **Profiles follow the family** (owner ruling 2026-07-17,
  `done.md:294`): on an FV model the `<p>_dot` `Profile` fields land
  on `CellAvg` — midpoint-initialized, products co-located. No pins;
  the ALE carve-out is the deliberate *exception*, keyed on
  `MovingGeometry` presence, not a coefficient rule.
- **The carve-out semantics survive the closure**: `dynamic_geometry`
  stays a model property; attaching a correctness module never flips
  the family. What changes on closing is `_fv_capable(dynamic)`
  itself (§7, owner call).
- **F5 precedent for shape and scope**: "F5 needed zero
  spatial-layer changes — the work was family-aware routing in the
  consuming modules plus the capability gate" (scoping `:712-715`).
  The FV ALE is the same kind of change, applied to
  `MeshVelocityCorrection`, plus the flux form itself.
- **Differentiability policy**: new step-path code ships an autodiff
  regression test; the `1/width` divisions ride `flux_diff`'s
  existing codomain-measure machinery (widths strictly positive on
  valid cells; `Inner` zero-pads boundary fluxes exactly).

## 5. Options

**A — status quo, ratified.** Moving geometry stays nodal-only;
the taught error and the auto carve-out already make it safe and
honest. Cost 0. Leaves the roadmap entry open forever or moves it to
a scoped-out decision (the cut-cells precedent). Defensible only if
moving-geometry-on-FV has no expected users.

**B — advective form on averages.** Route the column derivative
family-aware (`FVDerivative` already lands `CellAvg → CellAvg`; drop
the wrong interpolate hop on FV), sample `ṁ` at cell midpoints,
multiply co-located. Types check with zero new operators; ~a day.
But it buys *no* invariant — tracer drift keeps its O(Δx²) spatial
part — and it enshrines midpoint identification in a *numerics*
module, against the spirit of the F4 ruling (the letter of which
banned exactly this move's retag variant). Not recommended alone;
it is however the natural sub-case for `w`'s point-valued column
factor inside option C.

**C — flux-form FV ALE (recommended).** Per field, per column
factor:

- `CellAvg` column factor (`b`, `u`, `v`): reconstruct `f` onto the
  control-volume faces (`("reconstruct"/"interpolate", CellAvg)` →
  `Right|Inner`), sample the mesh velocity there
  (`grid.metric(face_space, "d<mapped>_d<p>", params=...)` — metric
  evaluation at average/face factors is supported,
  `grid.py:1176-1220`), form `f·w`, take the exact
  `("flux_diff", face)` back onto `CellAvg`, and subtract
  `f̄ · flux_diff(w)`. J/physical-width weighting module-side with
  params-threaded metrics, mirroring `_mapped_fv_divergence`
  (`advection.py:2408+`) — `FluxDifference` itself has no params
  seam, by design.
- `Right` column factor (`w`): advective form at the face node via
  the average-kind hops (the F5 corner-seam pattern) — the field is
  a point value there; no flux form exists or is needed.

New invariant (semi-discrete tracer budget closes), F5-shaped,
module-level only. Estimated a few days of well-specified Opus work
once ratified (§6).

**D — extensive prognostics / step-boundary rescale (horizon).**
Prognose `V f̄` (MOM6/ZStar pattern) or rescale by the analytic
volume ratio at step boundaries (MITgcm z\*). Buys exact
fully-discrete conservation; costs a structural change to the
prognostic set or a new stepper hook, touching every module and the
differentiability story. Out of proportion for the idealized-process
scope; record only, revisit iff budget closure to machine precision
ever becomes a requirement.

## 6. Implementation sketch + test plan (option C)

Module changes (`moving_geometry.py` only, expected):

1. `bind`: replace the blanket average-family `NotImplementedError`
   with per-field routing resolution (column factor `CellAvg` → flux
   form; nodal/point → advective form). Keep the taught error only
   for genuinely unsupported layouts, if any survive.
2. `_correct`: branch per resolved route. Flux route:
   `corr = D(w·f_face)/measure − f̄·D(w)/measure` with `D` the exact
   face→cell `flux_diff` and both terms sharing the same face `w`
   (the constancy condition); metrics via `with_params`-style
   evaluation each call, nothing cached (grid rules 2.3/3.8).
3. `extra_halo`: reconstruct + flux_diff is depth 2 — the declared
   substitute already says 2 per coordinate; re-verify, don't grow.
4. Flip `_fv_capable` / factory plumbing per the §7 owner call, and
   update the taught-error test into a capability test.

Gates (FV counterparts of the C4/nodal gates, same files):

- **Frozen-motion bitwise**: dynamic pipeline + FV ALE on a frozen
  schedule reproduces the static FV run bitwise (the §13 gate,
  extended to ALE-attached).
- **Constancy, machine zero**: uniform `b` under motion has exactly
  zero ALE tendency (single tendency evaluation, `== 0.0`).
- **Telescoping, machine zero**: `Σ V·corr + Σ f̄ V̇ − boundary = 0`
  at one evaluation (semi-discrete identity, not a run).
- **Manufactured `H(t)`**: FV counterpart of the nodal sign test
  (`tests/validation/test_moving_geometry.py:163`) — 2nd-order
  convergence against `b(z,T) = f(z·H(T))`.
- **Morph test, FV**: the sloped-to-flat morph with the tracer-drift
  bound tightened to a time-only residual (verify it shrinks with
  `Δt` at the stepper's order and is resolution-independent — the
  measurable form of the new invariant).
- **Autodiff regression** per the differentiability policy
  (`jax.grad` via `_chunk_body`, ≤8³, ≤10 steps, FD to rtol 1e-4).
- **Staggered component**: `w`'s advective route resolves the tagged
  average-family rows (counterpart of
  `test_ale_corrects_the_wall_staggered_component_too`).

## 7. Owner calls needed before implementing

1. **Does the auto default flip once capable?** Closing the gap makes
   `_fv_capable(dynamic_geometry=True)` true, and the standing rule
   "auto = FV wherever capable" would then flip existing
   moving-geometry configs from nodal to FV — a truncation-level
   numbers change for anyone relying on the auto default. Consistent
   with the twice-affirmed philosophy, but it is exactly the kind of
   silent default change the carve-out was built to avoid; needs an
   explicit ruling (flip, or keep the carve-out one release and
   announce).
2. **Momentum on the flux route**: `u`/`v` carry `CellAvg` column
   factors and naturally take the flux form (momentum content then
   telescopes too). Default yes; flagged because F5 advection
   deliberately kept velocities on the *consistent* form for its
   transverse axes — the asymmetry should be chosen, not inherited.

Option D is recorded, not proposed; no call needed unless exact
budget closure becomes a requirement.
