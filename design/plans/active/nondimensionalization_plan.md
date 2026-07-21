---
status: active
date: 2026-07-21
---

# Nondimensionalization plan — the scaling-object architecture

Owner-directed refactor (designed 2026-07-20/21; §A/§B revised to the
ratified scaling-object architecture 2026-07-21). Implementation
state: **§A (framework + shallow water) is implemented** on branch
`refactor/scaling-architecture` (Branch 1; golden-file parity against
pre-refactor dev `a1b668be`: all 16 sw2 configurations bitwise).
**§B (nonhydro2/hydrostatic, Branch 2)** and **§C (ramping envelope,
Branch 3)** are open follow-ups on this architecture.

## Goal

Every model package offers two assembly variants sharing all tendency
modules:

- **Dimensional** — physical parameters (shallow water: gravity `g`,
  depth `D(y[,t])`); **zero scaling operations** in the traced step
  (not even `×1.0`).
- **Nondimensional** — the scaled formulation whose parameters are
  the physical regime numbers (Ro, Fr, δ, metric ratio), with the
  **reference time scale itself a user choice** (`fr.scaling`).

The unifying frame (verified): with ε = T_ref/T_adv every coefficient
is a live ratio — nonlinear terms ×ε, rotation ×(ε/Ro), each wave
mechanism ×(ε/Fr_mech)² — and the scaling choice only picks **which
mechanism's number becomes ε**. Amplitude normalizations follow the
simplest-linear-operator rule per scaling (keeps analytic eigenbases
valid). Prognostic stays p = g·h (sw). All parameters live/Ramp-able;
structural refusals only.

## Owner rulings (2026-07-20/21, as ratified)

1. Prognostic p = g·h; no generalized (γ, D) core.
2. **Scaling objects** (`fr.scaling`): frozen host dataclasses
   `Dimensional() / Advective() / Rotational() / GravityWave() /
   InternalWave() / ExternalWave()`; traits `nondimensional`,
   `mechanism`; stored-only reference scales `L=, U=, g=`.
   `Model(scaling=)` is a real `fr.model.Model` kwarg carried through
   `variant()`; on the **presets** it is *optional* with
   `Dimensional()` as the default (owner spec change 2026-07-21) —
   a dimensional assembly needs no scaling argument at all.
3. **Single module families** (S3): no `Dimensional*`/`Nondim*`
   classes — mutually-exclusive constructor kwarg sets fix the
   variant (taught TypeErrors), unused leaves `None`.
4. **Thickness is a DIAGNOSE-stage field** computed by the core each
   substage; Sadourny and the conserving Coriolis read
   `state["thickness"]` and carry zero scaling references of their
   own.
5. Scaling-neutral modules (the advection families) adopt the
   variant **at bind** (`_BindTable.scaling`); no per-variant
   advection dispatch on the presets.
6. `scaling.rossby` → **`scaling.nonlinearity`** (generic name),
   renamed across the new stack.
7. **All parameters live/time-dependent** (g, D, Fr, Ro rampable at
   stage time; nothing frozen at build) — the sw core's csqr
   SELF_UPDATE generalizes to Ramp g/D and the D(y,t) law.
8. OB/AdiabaticRamping: interim guard (§C is the redesign) — a
   provider-backed ε **alias row** refuses the OB ramp with a taught
   error naming §C; absent/constant rows keep today's behavior.
9. Route-B conserving nondim variants (flat families) in scope; chart
   nondim rotation deferred (`RotationCoriolis` stays
   dimensional-only and co-assembles under any policy); ≤1-ulp
   chart-path parity accepted (measured: bitwise); `DynamicalCore`
   deleted in-branch (golden-file parity, no co-existence).

## A. Framework + shallow water — IMPLEMENTED (Branch 1)

### A1. The ε alias row (assembly)

`ParameterBindingTable.build(..., extra_rows=)` injects the canonical
`scaling.nonlinearity` row AFTER provider collection (a module
provide of the name collides, named) and BEFORE reference resolution
(defaulted references resolve against it instead of binding identity
constants). Per policy: `Advective` → constant-1.0 row (slot None);
a mechanism scaling → an **alias row** `(designated_slot,
nonlinearity_attr)` binding the canonical name onto the mechanism
module's own leaf — two names, one leaf (collision checks are
per-name); `update_parameters` through either name hits the same
leaf (double-write last-wins, documented); `Dimensional`/None → no
row (**row presence ⟺ nondimensional**; consumers dropped their
default-1.0 references). Alias names are recorded on the table
(`alias_names`, fingerprint-visible) for the OB guard.

### A2. Validation at assembly

Class traits `scaling_mechanism` / `nonlinearity_attr` + instance
`scaling_variant` on the scaled families; checks in `_scaling_rows`:
mixed variants (taught error listing offenders), nondim modules
without a nondim policy (and the converse), mechanism named but no
nondim owner (structural refusal), two owners (collision).

### A3. `sw.Core` (replaces `DynamicalCore`, deleted)

Kwarg sets: dimensional `gravity=` + `depth=` (csqr field = g·D) XOR
nondim `froude_number=` (+ optional `depth=` D̃ profile, default 1.0;
csqr field = D̃). Provides `shallowwater.gravity` / `.depth` /
`.froude` per variant (provides-implies-constancy; `shallowwater.csqr`
retired with taught errors at the old readers). Wave term: dim flux
form verbatim; nondim ×(ε/Fr)² OUTSIDE the flux on all three paths
(flat/immersed/chart), stage-time `ctx.params` reads. Thickness
DIAGNOSE (DIAGNOSTIC lifecycle, p's centre space, hydrostatic
w/p_hyd precedent): dim `csqr + p`; nondim `csqr + ε·(Fr/ε)²·p` —
the x/x spelling VERBATIM (under the matching scaling the aliased
leaves make the ratio an exact 1.0 → bitwise self-normalization).
Zero-parameter refusals (g/D/Fr = 0) at construction.

### A4. Term surgeries

Sadourny: no scaling references; reads `state["thickness"]`
everywhere `p_full` was (incl. the immersed wet-corner path);
background term unscaled both variants; bind-adopted nondim branch =
ONE outer ε per output (du, dv, dp) at the old factor positions
(bitwise placement; chart ε inside the exit rescale). Conserving
rotation reads `state["thickness"]` (rossby argument dropped);
route-B nondim = s·conserving body, s = ε/Ro; the correction scales
the (full − linear) difference by the same s, its variant adopted
from the PAIRED linear module at bind.

### A5. Shared Coriolis family

`FPlaneCoriolis(f0=)` XOR `(rossby_number=)`;
`BetaPlaneCoriolis(f0=, beta=)` XOR `(rossby_number=,
metric_ratio=)` — taught TypeErrors, Ro=0 refused. Nondim provides
`coriolis.rossby` (+ `coriolis.metric_ratio`); the carried
`f_coriolis` is the f-SHAPE (1, or 1 + metric_ratio·y); the shared
linear term scales the verbatim dimensional body by the live ε/Ro
(dimensional traces carry zero scaling ops); `linear=True`
preserved; `linear_params` unions F0/BETA/ROSSBY/METRIC_RATIO/
SCALING_NONLINEARITY (the frozen-``L`` check resolves them through
the binding table, skipping unbound names). A ramped `metric_ratio`
is a taught refusal (no shape blend yet); ramp `rossby_number`
instead. Conserving subclasses forward the dual kwargs and provide
`coriolis.rossby` in route-B nondim.

### A6. Preset, consumers, guards

`sw.Model(*, grid, core, time_stepper, scaling=None, coriolis=None,
advection=True, modules_extra=(), name=None, **kwargs)` — scaling
defaults to `Dimensional()`; taught TypeErrors for the retired
`csqr=`/`rossby_number=`/`coords=`; the advection adopts coords from
the core and the variant at bind; variable-depth
`metric_weight='csqr'` check via `core.variable_depth`. Eigen/energy
re-key on primitives: `csqr_eff = g·D` or `(ε/Fr)²·D̃`; `f0_eff = f0`
or `ε/Ro` (constant rotation ⟺ f0 provide, or `coriolis.rossby`
without `coriolis.metric_ratio`); `EnergyMetric.from_model` sw branch
re-keyed; channel/hy engines untouched (numeric operator probes).
Diagnostics: `epot`/`pot_vort`/`ekin_full` variant-aware from the
primitives; the thickness FUNCTION dropped from the DIAGNOSTICS
mapping (the state field is the user surface) — the conservation
functionals recompute h from the current state (a perturbed-state
functional must not read the one-substage-stale carry field).
Frozen-L: the assembly check additionally resolves each module's
`linear_operator_parameters()` through the binding table (cross-
module ε leaves). OB interim guard per ruling 8.

### A7. Verification (measured)

Golden files captured on pre-refactor dev `a1b668be` (4 geometries ×
{csqr=1.0, Ro=0.25, f0=0.5} dyadic and {csqr=0.5, Ro=1.0, f0=0.5}
× ±CoriolisEnergyCorrection, 10 AB3 steps): the today-parity mapping
`(csqr=X, rossby_number=r, f0=φ)` ≡ `GravityWave()` +
`sw.Core(froude_number=r, depth=X)` +
`FPlaneCoriolis(rossby_number=r/φ)` reproduces **all 16
configurations bitwise (0 ulp)** — flat, walled, immersed AND chart.
Dimensional ≡ today's `rossby_number=1.0` bitwise (gated in
`test_core_scaling`). Behavior gates: thickness restart bitwise,
Ramp g/D vs re-assembled constants, eigen/energy effective-number
branches, taught errors, propagator-based autodiff through the
thickness surface and the nondim-ratio path (FD-matched), one
forced-4 multi-device thickness step-parity shard.

## B. Other models (nonhydro2, hydrostatic) — Branch 2 (open)

On the §A architecture (`refactor/nh-hy-scaling`, after Branch 1):

- **B1 `_FluxFormAdvection` (nh/hy)**: de-scaled base + bind-adopted
  nondim branch (×ε outer; `scale=ε` inside the background-split
  full transport); H7 surface-closure boundary rows ≤1 ulp at ε≠1
  (accepted). Presets in the same commit.
- **B2 nh.Core**: **scaling-neutral** — `aspect_ratio=` (δ, live;
  squared at use sites) in every assembly; DSQR retired (cannot
  alias δ²); consumer re-points (stratification, wave maker, eigen,
  energy: `dsqr = δ²`, `n2_eff = (ε/Fr)²`).
- **B3 Stratification (nh/hy)**: `n2=` [1/s²] XOR `froude_number=`
  (internal Fr = U/(NH)); restoring −n2·w xor −(ε/Fr)²·w; owns the
  `internal_wave` mechanism.
- **B4 hydrostatic**: `hy.Core(gravity=)` provides
  `hydrostatic.gravity` (the physical constant centralizes on the
  core; nh.Core gains `gravity=` only when a T/S+EOS consumer
  lands); free surface: **no dimensional kwarg** — one `_csqr(ctx)`
  helper replaces the `ctx.params[CSQR]` reads (all three variants),
  dimensional `gravity/self._inv_depth` (H_ref inherits the roadmap
  §1 mesh-extent convention), XOR `froude_number=` (external Fr,
  the `external_wave` mechanism); barotropic term ×(ε/Fr_ext)².
- **B5**: old cores deleted; test migration; golden parity against
  the Branch-1-captured nh/hy baselines (flat dyadic configs,
  `dsqr=0.25, n2=4.0, Ro=0.25, f0=0.5`): today ≡ `Rotational()` +
  `aspect_ratio=δ` (δ²==dsqr exact) + `rossby_number=r` +
  stratification `froude_number=r/√B`; hy analogous.

### §B owner decisions (carried)

- **B-2 Hydrostatic primitive**: gravity on the core, free-surface
  c² derived per read (ratified above).
- **B-3 Preset physics defaults**: `hy.Model(stratification=,
  free_surface=)` required; `nh.Model` stratification default
  absent — the "no surprising defaults" reading.

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

## D. Unit factors and scales report — shipped (2026-07-21, framework + sw; nh/hy tables owed to Branch 2)

**Implemented** on `feat/unit-factors`: `fridom/model/units.py`
(`UnitFactor`/`FactorEntry`/`UnitsView`, the framework `t`/`T_ref`
rows, the centralized variant switch) behind `model.units`; the
shallow-water amplitude table `fridom/shallowwater2/units.py` on
`sw.Core.unit_factors` (instance property — `coords=` renames the
coordinate keys); the Coriolis family's `f_dim` row; and the writer
stamp (`Writer(units_metadata=True)` default: `fridom_scaling*`
global attrs + per-variable/coordinate/time `dimensional_factor`
attrs, the CF option-(b) time rewrite below). Rows are collected
duck-typed off `module.unit_factors` mappings, so the nh/hy tables
are a **Branch 2 follow-up with zero machinery edits**. Design
record (the shipped semantics):
**no state conversion API and no coexistence of dimensional and
nondimensional states** in the same function spaces (rejected
2026-07-21 — making that consistent would require distinct function
spaces per unit system, not worth it). Instead: the model exposes the
conversion *factors*; users apply them themselves.

- **Reference scales**: the scaling objects' reserved fields (`L=`,
  `U=`, sw/hy `g=`) — stored-only, optional.
- **The one rule**: T_ref = ε·L/U for every scaling (ε is bound:
  Fr/Ro/1 per clock choice), so every factor is derivable from (L, U
  [, g]) plus bound parameters — no per-scaling case analysis, no
  copies that can go stale under sweeps/ramps:
  x,y: L | z: δ·L | t: ε·L/U | u,v: U | w: δ·U |
  p, ps: U²/ε | sw h [m]: U²/(ε·g) | b: U²/(ε·δ·L);
  derived constants f_dim = U/(Ro·L), N_dim = U/(Fr_int·δL),
  c_dim = U/Fr. (p/b rows from the simplest-linear-operator
  normalization; the h row reproduces the paper's H = D·Fr.)
- **Surface**: `model.units` — `.factors` (component/coordinate/time
  → factor, with unit strings), `.factor(name)`, `.report()` (scales
  + derived dimensional constants + per-component factors; prints
  what the stored scales allow and marks the rest "needs U=", etc.).
  On a `Dimensional()` model all factors are 1.0 with the physical
  units — scripts stay polymorphic.
- **Writer**: NetCDF output stays in model units; the writer stamps
  metadata **by default** (owner ruling 2026-07-21: it changes no
  data, so it is free) — global attrs (scaling class, L, U, g, T_ref,
  ε, the bound numbers) and per-variable/coordinate/time
  `dimensional_factor` (+ target unit string) attributes. Users
  multiply themselves; postprocessing tools can automate it from the
  attrs.
- **Preset amendment (owner ruling 2026-07-21, supersedes the earlier
  "scaling= required")**: `scaling=` on the Model presets is
  **optional with `fr.scaling.Dimensional()` as the default** — a
  dimensional assembly needs no scaling spelling; a nondim-variant
  module under the (default) Dimensional scaling still hits the
  mixed-variant taught error, which names the fix (pass scaling=).
- Factor vocabulary: both the raw per-component factors (`p`) and the
  curated physical ones (`h`, folding the 1/g) are exposed and listed
  in the report.
- **Dimensional-model semantics (owner ruling 2026-07-21)**: raw
  component/coordinate/time factors are identity (1.0, physical unit
  strings); **curated factors keep their meaning** — `factor("h")` =
  1/g_bound on a dimensional model, so `factor("h")·p` yields meters
  in both variants (the polymorphism goal).
- **Time-axis CF metadata (owner ruling 2026-07-21, option b)**: on a
  nondimensional model the writer sets the time coordinate's
  `units = "1"` and drops the calendar anchor (model time is in units
  of T_ref; the old unconditional "seconds since <date>" claim was
  dimensionally false and auto-decoded as calendar time).
  `dimensional_factor = T_ref` is stamped alongside. Dimensional
  models keep today's CF time metadata.

## F. Docs strategy (owner rulings 2026-07-21)

For the docs rebuild's plan refresh (`docs_examples_plan.md` absorbs
this; recorded here to avoid touching that in-review file):

- **The beginner path is dimensional-only**: getting-started and the
  first model tutorials use physical parameters and never mention
  scaling (`Dimensional()` is the preset default, so no `scaling=`
  appears at all).
- **No central nondimensionalization chapter**: the nondimensional
  variants are documented **inside each model's own documentation**
  (sw docs carry the gravity-wave/paper scaling section incl. the §E
  chart-convention formulas and the `RotationCoriolis` Ω-shape
  spelling; nh/hy carry theirs), together with `model.units` /
  `units.report()` usage and the writer-metadata note.
- Examples pick their natural variant (paper-adjacent → GravityWave;
  classical setups may stay dimensional), each with a one-line
  comment naming the choice; plus a short "migrating from
  `csqr=`/`rossby_number=`" note (public repo).

## E. Chart/spherical nondimensionalization — documented convention (owner ruling 2026-07-21)

Nondimensional assemblies on chart grids are supported by
**convention, not machinery** (option 1): the user builds the sphere
grid with the nondimensional radius R̃ = R/L and the rotation with the
scaled rate Ω̃ = Ω·T_ref; the docs give the two formulas and one
worked spherical example. The docs round must also fix the
`RotationCoriolis` nondimensional kwarg spelling (the Ω-shape
normalization pairing with `rossby_number` — i.e. how Ro is defined
against 2Ω on the sphere) — defined there with a verification pass,
not improvised here. Deeper support (the grid builder consuming the
scaling object to auto-scale the radius; a scaling-aware
`RotationCoriolis`) is an **agenda item for the spherical-models plan
owner review** (`spherical_models_plan.md`), not part of this
refactor. Until the docs land, the scaling validation keeps a taught
pointer for nondim chart assemblies.
- **Explicitly out of scope**: `dimensional_state`/
  `nondimensional_state` helpers, unit tags on `State`,
  dimensional-coordinate arrays, dataset auto-conversion. Users who
  want to (non)dimensionalize fields do it with the factors.
- **Tests**: factor-table pins (paper-normalization h factor
  U²/(ε·g) = D·Fr under GravityWave; Ñ² consistency under
  Rotational), Dimensional-model identity factors, report smoke +
  missing-scale marks, writer attribute round-trip.
