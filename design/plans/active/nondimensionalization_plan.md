---
status: draft
date: 2026-07-21
---

# Nondimensionalization plan — dimensional / nondimensional model variants

Owner-directed refactor (designed 2026-07-20/21, three verification
rounds against the code). **Implementation is gated**: the owner wants
the two companion designs (§B other models, §C ramping redesign)
settled first, then a joint go. Nothing here has been applied.

> **STATUS 2026-07-21 — reference-time generalization under
> discussion.** The owner ruled that the *choice of reference time
> scale* (advective / rotational / wave) must itself be a user choice,
> not hardcoded (§A silently assumes the gravity-wave clock, §B the
> rotational one). The parameter surfaces and preset APIs in §A/§B
> will be revised once that discussion settles; the mechanism-level
> designs (thickness DIAGNOSE field, live-ratio rotation mixin,
> subclass advection scaling, §C envelope) are expected to survive —
> the coefficients are live ratios (ε/Ro, (ε/Fr)²), and the clock
> choice only changes which module provides ε. Do not implement §A/§B
> parameter surfaces until this note is removed.

## Goal

Every model package offers two assembly variants sharing all tendency
modules:

- **Dimensional** — the standard formulation with physical parameters
  (shallow water: gravity `g`, depth `D(y[,t])`); **no scaling factors
  anywhere** in the assembled modules (not even `×1.0`).
- **Nondimensional** — the scaled formulation with nondimensional
  parameters. For shallow water this is the Adiabatic-paper scaling
  (gravity-wave time scale `T_gw = L/c`, height scale `H = D·Fr`):

  ∂t u + Fr (u·∇)u + f u⊥ + ∇h = 0,
  ∂t h + ∇·u + Fr ∇·(hu) = 0,
  with f = f0 + βy, **f0 = Fr/Ro = 1/√Bu**, **β = (L/R)·f0**.

Load-bearing mapping (verified): the shared sw2 terms read only the
`csqr` field, the nonlinearity scaling (reference default 1.0), and the
`f_coriolis` field. Dimensional = {p = g·h, csqr = g·D, scaling 1,
dimensional f}; nondimensional = {csqr = 1 or D̃(y) = D(y)/D0, scaling
Fr, f0 = Fr/Ro}. No term math changes; the refactor is the parameter
surface and module structure. Prognostic stays p = g·h (owner ruling —
no h-in-meters core).

A universal identity makes the rotation treatment scaling-agnostic:
with ε = T_ref/T_adv (the nonlinearity scaling) and Ro = T_rot/T_adv,
the scaled Coriolis is f0_eff = T_ref/T_rot = **ε/Ro** for *any*
reference-time choice (gravity-wave scaling: ε = Fr → f0_eff = Fr/Ro;
rotational scaling: ε = Ro → f0_eff = 1). This is what lets one
rotation mixin serve every package (§B).

## Owner rulings (2026-07-20/21)

1. Prognostic p = g·h; no generalized (γ, D) core.
2. `sw.Model(grid=, core=, time_stepper=, ...)` — all three
   **required**, no surprising defaults; `coriolis=None` optional.
3. **Thickness is a derived field** computed by the core each substage;
   Sadourny and the conserving Coriolis read `state["thickness"]` and
   carry zero scaling references.
4. No baked-in scaling in advection; the nondimensional assembly
   applies it (subclass override, see below).
5. **Ro never appears on the preset** — only on nondimensional Coriolis
   modules (`NondimFPlaneCoriolis(rossby_number=...)`, ...), tiny
   subclasses over a shared mixin.
6. The nonlinearity scaling parameter is renamed **generically**:
   `scaling.rossby` → **`scaling.nonlinearity`** (not
   rossby/froude/epsilon), across the new stack now.
7. **All parameters live/time-dependent** (g, D, Fr, Ro rampable at
   stage time; nothing frozen at build).
8. OB/AdiabaticRamping keep deforming the renamed key for now, behind a
   new guard; the term-level ramping redesign is §C.
9. Fr source for the nondim rotation: **live
   `ctx.params[scaling.nonlinearity]`** + the AR guard (single source
   of truth).
10. Nondim rotation provides `coriolis.f0` = shape (1.0) **plus**
    `coriolis.rossby`; `sw.eigenbasis` computes f0_eff = ε·f0/Ro from
    the primitives.
11. Route-B conserving Nondim classes (flat families) are **in scope**;
    chart Nondim rotation classes deferred; ≤1-ulp chart-path parity
    accepted; `DynamicalCore` removed in the same branch (golden-file
    parity, no in-tree co-existence).

## A. Shallow water (fully designed, verified)

### A1. Thickness derived field — existing machinery

`StageKind.DIAGNOSE` (S1') runs per substage after SELF_UPDATE, before
tendency terms, replace-applied into carried state
(`src/fridom/model/stages.py:43-71`,
`src/fridom/model/schedule.py:62-72,693-721,871-882`); precedent:
`HydrostaticCore` recomputes `w`/`p_hyd` this way
(`src/fridom/hydrostatic/modules/core.py:410-430`). Thickness is a
`Lifecycle.DIAGNOSTIC` field (NOT AUXILIARY — DIAGNOSE's write gate) on
p's centre space plus one DIAGNOSE stage on the core: dimensional
`csqr.to(p) + p`; nondim `csqr.to(p) + ctx.params[NONLINEARITY]·p`
(stage-time `eval_params` → Ramp-Fr live). Ordering vs the csqr
SELF_UPDATE (S1 < S1') is automatic. Halo: pointwise compute on the
full local block, ghost validity inherited from p — no mandatory
exchange (≤1 extra scalar exchange/substage multi-device); halo-trace
preserved (`_tracer_params` demotes params to floats,
`assembly.py:2071-2107`); flat-path parity guard intact. All steppers
call `stages.prepare` before every tendency evaluation (RK per Butcher
stage; AB history stores tendencies — no staleness). Cleanup: drop the
`thickness` *function* (`sw2/diagnostics.py:301-309`, name collision);
re-point `ekin_full`/`etot_full`/`pot_vort` at `state["thickness"]`;
DIAGNOSTICS dict becomes per-core (pot_vort: `(f+ζ)/h` vs
`(f_eff+ε·ζ)/h`).

### A2. Cores (`sw2/modules/core.py` split)

Shared `_ShallowWaterCore`: u/v/p/csqr/thickness declarations, gravity
term (flat/chart/immersed, unchanged math), `_derive_extra_halo`, bind
checks, thickness stage hook.

- **`DimensionalCore(gravity, depth, *, coords, meridional)`**: live
  leaves `gravity: float|Ramp`, `depth:
  float|Ramp|callable(y)|ProfileFunction`. Provides
  `shallowwater.gravity` always, `shallowwater.depth` when scalar
  (Ramp provides legal — FPlane f0 precedent). csqr AUX field
  materializes g(0)·D(y,0) via the existing default builders (now ×g);
  any time-dependent g/D → the existing SELF_UPDATE generalized to
  `ctx.params[GRAVITY]·D(y,t)`; the scalar-Ramp-csqr TypeError
  (`core.py:206-219`) becomes supported for g/D. Zero scaling
  references in the assembled dimensional module set.
- **`NondimensionalCore(froude_number, depth=1.0, *, coords,
  meridional)`**: provides `scaling.nonlinearity` from the
  `froude_number` leaf; `depth` = D̃ (float/callable/ProfileFunction,
  exactly today's csqr slot; float ≠ 1 legal → today-parity mapping);
  provides `shallowwater.csqr` when constant.

### A3. Nondimensional rotation — mixin + tiny subclasses

New `CORIOLIS_ROSSBY = ParamName("coriolis.rossby")`.
`_NondimensionalRotation` mixin in `src/fridom/model/modules/coriolis.py`
(framework-level; §B reuses it): dynamic leaf `rossby_number` (provided
as `coriolis.rossby` — introspection + live sweeps), **required**
`Param(SCALING_NONLINEARITY)` (no default → nondim rotation on a
dimensional assembly is a taught MissingParameterError at assembly; no
preset logic), and a same-name `@term` override (the in-tree
`_ConservingRotation` precedent, `sw2/modules/coriolis.py:604-722`; MRO
override-in-place `module.py:258-272`; jaxify merges dynamic attrs
across the MRO) computing `s = ε/Ro` from `ctx.params` and returning
`{k: s·v}` over the parent family's term body. State-independent s(t)
preserves `linear=True` (term stays in L); exact for all f paths
(metric_weight flux form, FieldBlend, `_stage_scalar_f`) since rotation
is linear-homogeneous in f. `linear_params` = {CORIOLIS_F0,
CORIOLIS_BETA, SCALING_NONLINEARITY, CORIOLIS_ROSSBY}.

Concrete classes (≤10 lines each): `NondimFPlaneCoriolis(f0=1.0,
rossby_number=, metric_weight=None)`;
`NondimBetaPlaneCoriolis(rossby_number=, metric_ratio=)` → base
`(f0=1, beta=metric_ratio)`. Route B (in scope, sw2-owned):
`NondimNonlinearFPlaneCoriolis`, `NondimNonlinearBetaPlaneCoriolis` —
same outer-s over the conserving term (`linear=False`;
`linear_operator_gap` carries over). Chart classes deferred (roadmap).

Route A (`CoriolisEnergyCorrection`): no subclass — drops its scaling
reference, reads `state["thickness"]`, captures a static bind-time bool
`isinstance(linear, _NondimensionalRotation)` and multiplies
(full − linear) by s with default-1.0 refs (dimensional: s = 1).
`conserving_rotation` loses its rossby argument. `metric_weight` stays
`"csqr"` for the linear rotation (skewness under the *linearized*
metric — the documented energy argument holds); thickness-weighting is
the conserving modules' job.

### A4. Advection

Base `SadournyAdvection` loses every scaling line (`sadourny.py:489,
833, 840, 853, 869-872, 949, 976-977, 1028, 1084-1086`) and reads
`state["thickness"]` (new FieldReference; immersed
`_wet_corner_thickness` and chart paths consume it as the inline
p_full today). Background term stays unscaled in both variants (the
documented Wave-A2 signed delta).
`NondimSadournyAdvection(SadournyAdvection)`: required
`Param(SCALING_NONLINEARITY)` + `advect` override multiplying the base
output by ε. A generic term-scaling wrapper was evaluated and
**rejected** (delegation surface disproportionate for four call sites;
subclass override is the in-tree pattern).

### A5. Analytic consumers read primitives

- `sw2/eigenmodes.py:912-932`: csqr = `CSQR` provide, else
  `resolve_at(GRAVITY)·resolve_at(DEPTH)`, else the existing taught
  error; f0_eff = ε·f0_shape/Ro when `CORIOLIS_ROSSBY` is provided,
  else f0 as today (snapshot-at-`at_time` is existing behavior).
- `model/energy.py:460-481`: add a GRAVITY&DEPTH branch beside `_CSQR`;
  the field-weight fallback serves both cores' variable depth;
  dsqr/hydro branches untouched. `require_constant_coriolis` checks
  presence only — unaffected.
- `ChannelEigenmodes`: no edits (probes the assembled operator
  numerically).

### A6. Rename `scaling.rossby` → `scaling.nonlinearity`

`SCALING_NONLINEARITY` replaces `SCALING_ROSSBY`
(`model/params.py:128-133`). Code sites (full inventory verified):
`model/modules/advection.py:2340,3206,3506`;
`model/transforms/optimal_balance.py:109,115,125` (semantics
unchanged); `nonhydro2/params.py:31`, `hydrostatic/params.py:39`
(alias re-exports — their cores' `rossby_number=` kwargs keep working);
`nonhydro2/diagnostics.py:119`; sw2
params/core/sadourny/coriolis/diagnostics; ~14 test files rename-only.
Optional deprecated alias `SCALING_ROSSBY = SCALING_NONLINEARITY`.

### A7. Framework guards (generic, §B-reusable)

- **Frozen-L cross-module honesty**: extend
  `_check_frozen_linear_operator` (`assembly.py:113-145`) to resolve
  each module's `linear_operator_parameters()` through the built
  `ParameterBindingTable` and refuse TimeDependent provider leaves
  (catches ramped Fr feeding rotation-in-L under ETDRK4; ~15 lines).
- **OB interim guard** (§C errata 2026-07-21: scoped to OB, NOT a
  blanket AR refusal — an AR-wide refusal of
  `⋃ linear_operator_parameters()` keys would break the *sanctioned*
  f0-ramp (`FPlaneCoriolis.linear_params` includes `CORIOLIS_F0`;
  `tests/model/transforms/test_adiabatic_ramping_exponential.py` ramps
  f0 under AB3 — deforming L is what AR is *for*; frozen-L steppers
  are already covered by the assembly check above)): guard **OB's
  automatic ε-ramp construction only** (equivalently: the
  `SCALING_NONLINEARITY` key) with a taught error naming the §C
  redesign — prevents the ε-ramp from dragging the nondim rotation
  (f0_eff = ε/Ro, live read) down with it. OB on today-shaped models
  unchanged; between §A and §C, OB on the new nondim variant raises
  the taught error — §C is what re-enables it. §C's first commit
  deletes this guard.

### A8. Preset v2 (`sw2/model.py`)

`sw.Model(*, grid, core, time_stepper, coriolis=None, advection=True,
modules_extra=(), name=None, **kwargs)` — grid/core/time_stepper
keyword-required, no default stepper; `csqr`/`rossby_number`/`coords`
kwargs removed (hard switch; taught TypeErrors naming the migration).
Preset wires: `SadournyAdvection(coords=core.coords)` vs
`NondimSadournyAdvection` (dispatch on the core), the variable-depth
`metric_weight="csqr"` check via a `core.variable_depth` property,
immersed `MaskState`, `check_rotation_modules` (unchanged — nondim
classes are FPlane/BetaPlane-family instances). No coriolis scaling
logic in the preset. `DynamicalCore` deleted in the same branch.

### A9. Commit order (one branch, `refactor/sw-nondimensionalization`)

1. Rename + `CORIOLIS_ROSSBY` (+ alias), src+tests green.
2. Framework guards (A7).
3. sw2 cores: `_ShallowWaterCore` split, the two cores, thickness
   DIAGNOSE, g/D SELF_UPDATE.
4. Consumers: sadourny de-scaling + thickness reads, conserving
   coriolis thickness + bind-flag, `NondimSadournyAdvection`.
5. Rotation mixin + Nondim classes (framework) + route-B pair (sw2).
6. Preset v2; eigenmodes/energy primitive reads; diagnostics split.
7. Test migration + parity/autodiff/multi-device gates.

Docs/examples follow on a separate owner-reviewed `docs/` branch
(AGENTS.md private-review workflow; 3 examples + ~6 docs pages use the
old `csqr=` API).

### A10. Verification

- **Golden-file parity** (captured on the pre-refactor dev commit):
  (a) nondim-equivalent assemblies (flat periodic/walled/immersed/
  chart, ± correction) — bitwise expected except chart ≤1 ulp;
  (b) dimensional vs today's `rossby_number=1.0` — bitwise all paths;
  (c) `NondimFPlaneCoriolis` vs hand-computed `FPlaneCoriolis(f0=Fr/Ro)`
  — bitwise unweighted, ≤1 ulp metric-weighted; same for the route-B
  pair.
- Behavior: thickness refreshed per substage (RK/AB agreement after
  restart); Ramp-g/D runs vs re-assembled-constant snapshots; required
  Param taught errors; frozen-L refusal of ramped Fr/Ro under the
  exponential stepper; AR guard; eigen f0_eff and g·D branches; energy
  g·D branch; existing invariant gates (mass/energy machine zero,
  etot_full, split-advance).
- Autodiff (differentiability policy): propagator-based grad
  regressions through the thickness surface wrt initial p in the
  sadourny/immersed/spherical autodiff shards; assert existing
  materialized-owner refusals.
- Multi-device: one 2-rank nonlinear step-parity test exercising the
  thickness exchange.

## B. Other models (nonhydro2, hydrostatic) — designed (2026-07-21)

### B0. Current equations and parameter flow (verified)

**nonhydro2** (state `(u, v, w, b)`, `p` diagnostic): Ro enters
**only** through the shared `_FluxFormAdvection`
(`model/modules/advection.py:2340` defaulted reference, `:3206`
`_advect`, `:3506` `_advect_perturbation`, where ε sits *inside* the
advecting velocity `U + Ro·u'`) and `linear_pot_vort`
(`nonhydro2/diagnostics.py:119`); the core declares it
(`core.py:618`) but no core term consumes it. `dsqr` = (H/L)² enters
the four `_project*` routes (`core.py:739,800,862,919`) and the
elliptic weight `Diag(1,1,1/δ²)`; solvers accept floats. The current
system is the **rotational scaling**: T_ref = 1/f, ε = Ro,
f0_eff = ε/Ro = 1 (users pass `FPlaneCoriolis(f0=1)`), and the `n2`
slot holds Ñ² = (N·T_ref)²·δ² = **Bu** = (NH/(fL))². With dsqr = 1,
ε = 1 the same term set IS the dimensional Boussinesq system — the sw
load-bearing mapping repeats exactly.

**hydrostatic** (state `(u, v, b, ps)`; `w`, `p_hyd` DIAGNOSE):
`csqr` (m²/s²) is read only by the free-surface family
(`free_surface.py:610,937` + barotropic solver, as c² or derived
g = c²·`_inv_depth`); `rossby` is declared by the core (`core.py:405`)
but read nowhere in the package — its sole consumer is the shared
advection. Today's model is already the dimensional formulation with
one scaling knob on advection; the same numbers reread
nondimensionally (c̃² = (Ld/L)², Ñ² = Bu). **No new derived fields**:
neither package has a thickness analog; w/p_hyd DIAGNOSE already
exist; the free surface is linear (advection never transports `ps`).

### B1. Advection surgery (shared, serves both packages)

`_FluxFormAdvection` (base of Centered/Upwind/WENOAdvection) loses
every scaling line: drop the `SCALING_NONLINEARITY` reference
(`advection.py:2340`); `_advect` (`:3206`) returns the unscaled
transport; `_surface_correction`/`_apply_correction`/
`_immersed_scale_2d` drop the `ro` parameter; `_advect_perturbation`
(`:3506`) uses the full velocity `U + u'` with a `scale` hook argument
on `_full_transport` (`None` = unscaled).

New framework-level **`_NondimensionalAdvection` mixin** (same file,
the `_NondimensionalRotation` twin): **required**
`Param(SCALING_NONLINEARITY)`; `_advect` override returning ε · the
parent output; `_advect_perturbation` override calling
`_full_transport(state, q, scale=ε)` (ε inside the advecting velocity
— a pure output multiply is wrong for the background split; today's
`S(U + ε·u') − S_lin(U)` semantics preserved; background term
unscaled in both variants, Wave-A2). Concrete classes (≤5 lines):
`NondimCenteredAdvection`, `NondimUpwindAdvection`,
`NondimWENOAdvection`. sw2's `NondimSadournyAdvection` (§A4) follows
the same mixin pattern.

Parity note: plain path bitwise; the H7 **surface-closure** path
(hydrostatic diagnosed-`w` grids) redistributes ε over an addition —
≤1-ulp boundary rows at ε ≠ 1 (documented concession, mirrors §A's
chart concession); exact at ε = 1.

### B2. Rotation — verbatim reuse

The §A mixin + Nondim classes are package-neutral; nonhydro2/
hydrostatic re-export them, zero new code. Under the traditional
rotational scaling s = ε/Ro = 1.0 exactly (IEEE x/x), so
`NondimFPlaneCoriolis(rossby_number=Ro)` is bitwise
`FPlaneCoriolis(f0=1)` — both spellings stay legal. Until §C lands,
the OB-compatible canonical spelling for nh/hy remains
`FPlaneCoriolis(f0=1)` (the OB interim guard refuses the Nondim
classes); the Nondim classes are the self-documenting alternative and
the enabler of non-rotational reference times.

### B3. nonhydro2 cores (`nonhydro2/modules/core.py` split)

Shared `_NonhydroCore`: declarations, family machinery (+ a public
`core.family` property for the preset), bind/extra_halo, all four
`_project*` stages — every dsqr use stays a `ctx.params[DSQR]` read.

- **`DimensionalCore(*, vertical="z", coords=..., family=None,
  solver kwargs...)`** — no physics kwargs (dimensional Boussinesq has
  no free coefficients; f and N² live on their modules).
  `parameter_declarations = ()`; a `ParameterReference(DSQR,
  default=1.0)` instead — the identity default binds static, so
  `1/1.0` folds through the projection and solvers with no structural
  change. Drops the `time_dependent_linear_parameters` dsqr hook. Zero
  traced scaling scalars.
- **`NondimensionalCore(dsqr=1.0, *, rossby_number=1.0, ...)`** —
  today's `DynamicalCore` surface: declares `nonhydro.dsqr` and
  `scaling.nonlinearity` (ε keeps the package's physical name
  `rossby_number` — the rotational-scaling analog of sw's
  `froude_number`); keeps the frozen-L dsqr hook; both leaves
  live/Ramp-able.

Stratification modules shared verbatim: `ConstantStratification`/
`MeridionalStratification` change only their DSQR reference to
`default=1.0` (`stratification.py:120,244`) so `b.to(w)/dsqr` folds
dimensionally; same for `PolarizedWaveMaker`
(`polarized_wave_maker.py:218`).

### B4. Stratification (nondim N²) — no new module (recommended)

N enters once, quadratically, in the restoring `−N²w`. General
identity: **Ñ² = (N·T_ref)²·δ² = Bu·(ε/Ro)²**; under the traditional
rotational scaling Ñ² = Bu exactly. The `n2` slot is already a bare
O(1) input the user states directly
(`ConstantStratification(n2=Bu)`); docstrings carry the general
formula. A live `NondimStratification(burger_number=...)` analog is
**deferred**: it does real work only for non-traditional reference
times and would add a second guard-blocked linear surface with no §B
use case. Same reasoning for hydrostatic. (Owner decision B-1.)

### B5. hydrostatic cores (`hydrostatic/modules/core.py` split)

Shared `_HydrostaticCore` keeps declarations, both DIAGNOSE stages,
`pressure_gradient`, and all terrain/immersed/partial-bottom
machinery unchanged (the core reads neither csqr nor rossby in any
term or stage — verified):

- **`DimensionalCore(csqr, *, vertical, horizontal)`** — declares
  `hydrostatic.csqr` only (physical m²/s²; c² = g·H stays the
  canonical single barotropic parameter — H is mesh geometry and the
  free-surface family already derives g = c²·`_inv_depth`; owner
  decision B-2). No scaling declaration.
- **`NondimensionalCore(csqr=1.0, *, rossby_number=1.0, ...)`** —
  declares `hydrostatic.csqr` (nondim reading c̃² = (Ld/L)²) and
  `scaling.nonlinearity`.

Free-surface variants, barotropic solver, terrain, `ThermalWindBackground`
(linear background couplings — unscaled in both variants, Wave-A2),
`MaskState`: untouched.

### B6. Presets v2 + diagnostics split

- `nh.Model(*, grid, core, time_stepper, coriolis=None,
  stratification=None, advection=True, modules_extra=(), name=None,
  **kwargs)` — dsqr/rossby/family/solver kwargs move onto the core
  (hard switch, taught TypeErrors); family resolution via
  `core.family`; `advection=True` dispatches on the core
  (`CenteredAdvection` vs `NondimCenteredAdvection`); passed instances
  checked for Nondim-ness mismatch (taught error).
  `stratification=None` = no stratification module (explicit opt-in;
  owner decision B-3).
- `hy.Model(*, grid, core, time_stepper, stratification,
  free_surface=None→ExplicitFreeSurface(), coriolis=None,
  advection=True, surface_advective_flux=None, ...)` —
  `stratification` required; same advection dispatch.
- `comparison_model` (HY-D6) re-targets
  `NondimensionalCore + NondimCenteredAdvection` (protocol pin updated
  once).
- Diagnostics per-core (§A1 pattern): nh nondim dict = today's;
  dimensional dict drops Ro and δ² (`ekin` = ½(u²+v²+w²),
  `linear_pot_vort` = f0/N²·∂z b + ζ). hy `DIAGNOSTICS` reads no
  scaling — shared.
- `DynamicalCore` and `HydrostaticCore` deleted in the same branch
  (golden-file parity, no co-existence).

### B7. Analytic consumers (mirror A5)

- `nh.eigenbasis` (`nonhydro2/eigenmodes.py:1307`): `dsqr` = provide
  if present else 1.0; f0_eff = ε·f0/Ro when `CORIOLIS_ROSSBY`
  provided, else f0 as today.
- `EnergyMetric.from_model` (`model/energy.py:442`): the `_DSQR`
  branch gains a dimensional-nonhydro arm (structural detection:
  prognostic `w` with Velocity role + `p` field; hydrostatic
  disambiguated by DIAGNOSTIC `w` + `_HYDRO_CSQR`) using dsqr = 1.0 —
  exact helper spelling to be re-verified at implementation time.
- `hy.eigenbasis`/`HydrostaticEigenmodes`, `nh.ChannelEigenmodes`:
  numeric operator probes — no edits.

### B8. Parity mappings and tests

- nh: `DynamicalCore(dsqr=d, rossby_number=r)` + `CenteredAdvection` ≡
  `NondimensionalCore(dsqr=d, rossby_number=r)` +
  `NondimCenteredAdvection` — bitwise;
  `DynamicalCore(dsqr=1, rossby_number=1)` ≡ `DimensionalCore()` +
  `CenteredAdvection` — bitwise (static-1.0 folds are value-exact).
- hy: analogous; bitwise except surface-closure boundary rows at
  ε ≠ 1 (≤1 ulp, B1).
- Gates: golden files pre-refactor across grid families/terrain/
  immersed × free-surface variants; background-split parity in both
  variants; taught errors; eigen fallback + f0_eff; energy-metric
  dimensional arm; existing dispersion/adjustment/energy/continuity
  gates on both cores; comparison-protocol pin; Ramp csqr/dsqr/n2 vs
  re-assembled constants; frozen-L refusals; OB regression (nondim nh
  with `FPlaneCoriolis(f0=1)`) + guard firing with
  `NondimFPlaneCoriolis`; one 2-rank step-parity test per package;
  autodiff shards re-run (no new derived fields).

### B9. Commit order (branch `refactor/nh-hy-nondimensionalization`, after §A)

1. Shared advection surgery + mixin + 3 Nondim classes, and in the
   same commit both presets flip `advection=True` to the Nondim
   classes (old cores still provide ε → physics unchanged; the Nondim
   classes cannot land before the base de-scaling — double-scaling).
2. nonhydro2: core split, preset v2, diagnostics split,
   DSQR-defaulted references, eigen/energy edits.
3. hydrostatic: core split, preset v2, comparison re-point.
4. Old cores deleted; test migration + gates. Docs on the
   owner-reviewed docs branch.

**Constraints on §A (so §B needs no rework):** keep the
`ROSSBY = SCALING_NONLINEARITY` aliases in nonhydro2/hydrostatic
params.py; touch `model/modules/advection.py` rename-only (B1 owns
the restructuring diff); keep the rotation mixin free of sw-specific
reads; keep the guards keyed on `linear_operator_parameters()`
generically; keep `model/energy.py`'s branch chain provide-keyed.

### §B owner decisions

- **B-1 Nondim stratification**: accept no-new-module (`n2` = Ñ²
  directly; Bu under the traditional scaling) — or commission
  `NondimStratification(burger_number=...)` with a live (ε/Ro)²
  factor (deferred-recommended).
- **B-2 Hydrostatic primitive**: keep `csqr = g·H` as the single
  barotropic parameter (recommended) — or an sw-style `gravity=`
  split with derived csqr.
- **B-3 Preset physics defaults**: ratify `hy.Model(stratification=)`
  required and `nh.Model` stratification default *absent* (today both
  default `ConstantStratification(n2=1.0)`) — the "no surprising
  defaults" reading — or keep today's defaults.

## C. Ramping / optimal-balance redesign — designed (2026-07-21)

Owner ruling: stop deforming the scaling parameter; ramp the
**nonlinear terms as a whole** — a time-dependent envelope
ρ(t) ∈ [0,1] on term *outputs*, independent of what scales them.
Works identically on dimensional models (no scaling parameter bound)
and nondimensional ones (envelope multiplies on top of the live ε);
the nondim rotation (f0_eff = ε/Ro, live ε) is untouched by
construction. Replaces the §A OB interim guard and OB's
stored-nominal logic (`optimal_balance.py:109-125` — OB floats the
model's ε as a ramp target and *refuses* Ramp-valued ε today; on a
model without the key it silently runs a no-ramp cycle).

### C1. Mechanism — composer-level term envelope, ρ as a bound parameter

- New `params.RAMPING_ENVELOPE = ParamName("ramping.envelope")`.
- New framework module `TendencyEnvelope(terms=<TermPredicate>,
  envelope=<float|TimeDependent>)`
  (`src/fridom/model/modules/ramping.py`): no fields/terms/stages;
  one dynamic leaf `envelope` (the `FPlaneCoriolis.f0` pattern —
  Ramp-valued leaf legal, rides the carry, stage-time resolved,
  zero-recompile sweeps); a static `terms` predicate
  (`fingerprint_token` exists on every TermPredicate); provides
  `ramping.envelope`.
- `TendencyComposer` detects at most one envelope module (two →
  taught error). After `_filter_terms`, every surviving matching
  entry's unbound fn is wrapped:
  `{k: ctx.params[RAMPING_ENVELOPE] * v}`. Dry-run, attribution, and
  the halo-trace walk all go through `entry.fn` — ρ is float-demoted
  in the tracer exactly like today's ramped ε; ρ(0)=0 preserves ghost
  demand (the existing OB-variant ⊆-verify precedent); a scalar
  multiply is halo-width-neutral.
- `ScheduleEntry.enveloped: bool` included in `static_token()` →
  enveloped and plain models never share a memoized step body or jit
  entry; existing (non-enveloped) restart fingerprints byte-identical.
- Taught refusals at assembly: predicate matches an IMPLICIT term
  (an enveloped implicit solve is unsound); matches a `linear=True`
  term (ramping L is the parameter-ramp path's job); matches nothing —
  error without a `term_filter`, warning under one (load-bearing: OB's
  `backward_filter=fr.terms.linear` leg has an inert envelope and must
  still build).
- `Model.variant(...)` gains `extra_modules=()` (appended clones;
  refused if any declares fields — the shared-FieldTable/state-treedef
  contract stays intact); `Propagator` passes it through.
- Rejected: post-hoc envelope module or stepper hook (per-term
  structure does not survive `BoundSchedule.tendency`'s single
  `sums.add`); closure-captured ρ (bakes endpoints as jit constants,
  violates the D2 no-host-capture rule, loses zero-recompile sweeps).
  The prior §A rejection of a `_collect_terms` wrap seam was for
  module-owned *physics* scaling; the envelope is transform-owned,
  cross-package, and must enter schedule identity — the seam is right
  here.

### C2. Semantics — the two homotopies, honestly

Continuum ∂t z = Lz + εN(z). Old: ε ↦ λ(t)·ε everywhere, including
the discrete PV-form thickness and conserving-rotation denominators.
New: ∂t z = Lz + ρ(t)·[terms at fixed inner ε]. Both are smooth
homotopies between the **same endpoints**: ρ=0 ≡ `linearize(model)`
by definition; ρ=1 ≡ the nominal model bitwise. The ε-in-thickness is
a PV-form discretization artifact (continuum q·(hu) = (f+εζ)u);
mid-ramp the paths differ only at that level. Route A: the correction
ramps with the nonlinearity, the pure-f rotation stays whole in L —
the physically defensible split. Route B at ρ=0 removes rotation
entirely — consistent (that L genuinely has no rotation,
`linear_operator_gap`); rotation-preserving ramping uses route A.
The nondim rotation reads live ε, which is never ramped → untouched.

### C3. API

`AdiabaticRamping(model, *, ramp_period, curve="exp", steps=None,
envelope=False, ramps=None, term_filter=None, updates=None,
name=None)`:

- `envelope=True` ≡ predicate `~fr.terms.linear & fr.terms.explicit`
  ("the nonlinearity as a whole"); a TermPredicate narrows/widens.
  Internally ρ = `Ramp(0.0, 1.0, period, curve)` stored under
  `params.RAMPING_ENVELOPE` in `self._resolved`, so `.down`,
  `.backward` (`Ramp.reversed()`, flipped TIME_STEP) and `.replace`
  reuse the existing leg machinery verbatim. First leg appends
  `TendencyEnvelope` via `variant(extra_modules=...)`; derived legs
  update the bound leaf via `updates=`.
- `ramps={key: (v0, v1)}` **stays** as the first-class parameter-
  deformation path (f0 spin-up — the load-bearing
  `test_adiabatic_ramping_exponential` recipe — depth, dsqr, g): the
  two paths answer different questions (deform L vs ramp N).

`OptimalBalance`: public signature unchanged; internally
`forward_leg = AdiabaticRamping(model, envelope=True, ...)`. Deleted:
`has_rossby`/nominal/target logic and the Ramp-ε TypeError — OB now
works on Ramp-ε models and does real balancing on dimensional models.
Effective nonlinearity on a nondim model: ρ(t)·Fr — the identical
outer multiplier to today's `Ramp(0, Fr)`; only the ε-inside-thickness
detail differs (C2). Frozen-L steppers stay valid (enveloped terms are
`linear=False` by construction).

### C4. Compatibility / rollout

§A ships the frozen-L cross-check (permanent) and the OB-scoped
interim guard (see §A errata); §C's first commit deletes the guard.
Between §A and §C, OB on the new nondim sw variant raises the taught
error — §C re-enables it. No §A rework: §C touches only framework
(composer/schedule/assembly/model/params + one new module) and the
two transforms; no sw2 module changes. §B is independent (the
envelope is term-tag-generic). Test migration:
`test_optimal_balance.py` — ramp-probe rewritten against
`ramping.envelope`, Ramp-ε-rejection test replaced by acceptance,
convergence/structure gates kept; `test_adiabatic_ramping.py` —
parameter-ramp shards unchanged, new envelope shards.

### C5. Parity / expectations

Endpoints bitwise (ρ=1 nominal; ρ=0 ≡ linearize). Mid-ramp differs
from the old ε-ramp at the discrete inner-thickness level
(O(ε·p/c²) relative, PV weights only). OB's fixed point: the same
balanced manifold, not the same bits — agreement up to the diabatic
leakage scale (the stretched-exponential O(e^{-c√τ}) the in-tree
exponential shard pins) plus fixed-point tolerance. Gates are
physics-tolerance, never bitwise-vs-old.

### C6. Files

`model/params.py` (RAMPING_ENVELOPE); `model/modules/ramping.py`
(new); `model/composer.py` (detect + wrap + refusals);
`model/schedule.py` (`enveloped` + static_token); `model/assembly.py`
(fingerprint row); `model/model.py` (`variant(extra_modules=)`);
`model/transforms/propagator.py` (passthrough);
`adiabatic_ramping.py`; `optimal_balance.py`; tests.

### C7. Tests

Endpoint exactness (leg-start ≡ linear variant, leg-end ≡ nominal,
bitwise); four-leg matrix + involutions for the envelope Ramp; taught
errors (linear match, IMPLICIT match, empty match ± filter, two
envelope modules, field-declaring extra module); memo/identity
(distinct AssemblyRecords, plain fingerprints unchanged); halo
⊆-verify on an enveloped immersed/chart variant; ETDRK4+envelope
smoke; OB on a dimensional model balances; nondim legs keep ε and
f0_eff constant through the ramp; Ramp-ε accepted; one
stretched-exponential leakage shard for the new path.

### §C owner decisions

- **C-1 Default envelope term set**: recommended
  `~fr.terms.linear & fr.terms.explicit` (everything not in L; makes
  ρ=0 exactly `linearize(model)`, but also ramps forcing and closure
  terms, which the old ε-ramp never touched). Alternative: exclude
  closures/forcing by default.
- **C-2 §A guard scoping errata**: accepted (folded into §A above) —
  guard OB's automatic ε-ramp only, preserving the sanctioned f0-ramp.
- **C-3 Parameter-ramp path**: confirm `ramps=` stays first-class
  (recommended: yes).

### C9. Ordering

§C lands after §A on its own branch with zero §A rework; §B is
independent of §C; no §B decision blocks §C.
