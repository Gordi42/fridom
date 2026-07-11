---
status: normative
date: 2026-07-07
---

# Model layer redesign — modules, model, time stepping

Status: **draft** (design phase, no implementation yet)
Author: Silvano Rosenau (with AI-assisted brainstorming)
Date: 2026-07-06

This note set designs the model layer of `fridom.framework2` — the
`Module` system, the `Model` composition root, the state-vector
assembly, and the time steppers. It is the design reference for
ROADMAP Phase 2 (tasks 2.1–2.6); the grid layer it builds on is
designed in the sibling note set [`../grid/`](../grid/00_overview.md) and is
being implemented (ROADMAP Phase 1) in parallel.

All code snippets are **illustrative, not normative**: names and exact
signatures are expected to change during implementation.

---

## Document map

| File | Contents |
|------|----------|
| [`00_overview.md`](00_overview.md) | Motivation (section 1), inherited grid constraints (section 2), migration, precedents — this file. |
| [`01_concepts.md`](01_concepts.md) | Core concepts (section 3) and the four load-bearing design decisions (D1–D4), framed as alternatives with trade-offs. |
| [`02_rules.md`](02_rules.md) | Rules (section 4): decided rules accumulated from D1–D5 sign-offs; remaining scope listed as planned. |
| [`03_time_stepping.md`](03_time_stepping.md) | **Full D3 design** (section 5): the term surface, the stage schedule, the stepper core, IMEX/splitting, StepContext, project-the-state. |
| [`04_run_loop_io.md`](04_run_loop_io.md) | **Full D4 design** (section 6): the Model object, the assembly pipeline, the run loop, the IO seams, the post-assembly lifecycle, coupling/multi-device proofing. |
| [`05_api_sketches.md`](05_api_sketches.md) | Non-normative API sketches (section 7), drafted from the resolved decisions — the acceptance surface for class design. |
| [`06_validation.md`](06_validation.md) | Paper validation (section 8): four adversarial walks complete — findings triaged into sign-off decisions (§8.6), accepted amendments (§8.7), and the consolidated cutover-parity list (§8.8). |
| [`07_open_threads.md`](07_open_threads.md) | Open threads (section 9). |
| [`08_state_transforms.md`](08_state_transforms.md) | The state-transform algebra (section 10): composable `State -> State` transforms, model variants, the ported projection family (decision D5). |
| [`09_coupling_designfor.md`](09_coupling_designfor.md) | Coupled models, design-for constraints (section 11): precedent survey + adversarial A–O walk + architecture pre-design → the class-spec constraint list CS-1..18. Not the 3.2 design. |
| [`classes/`](classes/README.md) | The class-specification phase: six cluster files (declarations, module, model, time_steppers, transforms, io_ops) turning D1–D5 + CS-1..18 into concrete class surfaces — the bridge to ROADMAP 2.2–2.8. |
| [`../../plans/done/phase2_implementation_plan.md`](../../plans/done/phase2_implementation_plan.md) | How 2.2–2.8 is executed with parallel subagents: waves, exclusive file ownership, gates — the Phase-2 analogue of [`../../plans/done/phase1_implementation_plan.md`](../../plans/done/phase1_implementation_plan.md). |
| [`research/`](../../research/README.md) | Per-decision research reports (option analyses, precedent research) — inputs to the decisions, not normative text. |

Section numbers are stable identifiers across the files, continuing
the convention of the grid notes.

---

## 1. Motivation: pain points of the current model layer

The current model layer (`framework/model_settings_base.py`,
`framework/model.py`, `framework/modules/`,
`framework/time_steppers/`) works, but its structure blocks the
roadmap targets (no ModelSettings, one pytree, single-jit runs,
module-registered fields):

1. **`ModelSettings` is a god-object.** `ModelSettingsBase` owns the
   grid, all module containers (`tendencies`,
   `pre_step_diagnostics`, `diagnostics`), the time stepper, the
   timer, the custom-field lists, the halo, *and* (in the model
   subclasses) every physical parameter (`f0`, `beta`, `N2`, `dsqr`,
   `csqr`, `rossby_number`). Every module receives `mset` at setup
   and copies what it needs into its own attributes
   (`LinearTendency._on_setup`). Assembly is implicit in the order
   of `mset` construction, attribute mutation, and `mset.setup()`.
2. **Parameters are tangled across owners.** The `rossby_number`
   setter reaches into `tendencies.advection.scaling`; the
   `f0`/`beta` setters rebuild the `f_coriolis` field; parameters
   are consumed far from their owner (`State.pot_vort` reads `f0`
   and `N2`, the cartesian grid's eigenvectors read model physics —
   the grid notes already evict the latter).
3. **The variable set is frozen per model.** `nonhydro.State`
   hardcodes `u, v, w, b`: buoyancy exists even in unstratified
   runs, and its coupling terms sit inside `LinearTendency`
   alongside Coriolis. The target is the inverse: the nonhydrostatic
   core has no `b`; a `ConstantStratification` module *registers*
   `b` and contributes both coupling terms (`+b` in the `w`
   tendency, `-N2 w` in the `b` tendency).
4. **Field registration exists but is bolted on.**
   `mset.custom_state_fields` is a plain mutable list of
   `FieldMetadata` consumed by `State._create_default_fields`; no
   registration API, no ordering control, incomplete in
   shallowwater (`TODO` in `shallowwater/state.py`). It proves the
   seam works; the redesign makes it the *primary* mechanism.
5. **The run loop is a Python loop.** Jit lives inside module and
   time-stepper `update` methods; every step round-trips through
   the host. The roadmap target is one `jax.jit` over the whole run
   (`lax.scan` / `while_loop` + `io_callback`).
6. **The time stepper is a `Module` that drives the physics.**
   `AdamBashforth.update` calls `mset.tendencies.update` internally
   and keeps warm-up state (`it_count`, coefficient ramping) as
   Python-side logic. Staged/split stepping — IMEX partitions,
   implicit vertical mixing, pressure projection — does not fit
   this shape; the projection is currently three modules hardcoded
   at the tail of `MainTendency`, with `add_module` inserting user
   modules *before* the trio by convention.
7. **Physics and host-side infrastructure share one chain.**
   Progress bar, NaN checker, restart, and writers are modules in
   the same `update` pipeline as tendencies — incompatible with a
   fully traced run, where host effects must go through explicit
   callbacks.
8. **Setup order is fragile.** Halo is a module-owned integer
   (`required_halo`) maxed over containers; changing modules can
   re-trigger `grid.setup`; idempotence is managed with
   `setup_mode="forced"` flags. The grid redesign replaces halo
   accounting wholesale (trace-based, section 2 below); the model
   redesign must supply the orderly assembly sequence that drives
   it.

## 2. Inherited constraints from the grid redesign

These are **fixed** by the grid notes and the class designs; the
model design consumes them, it does not reopen them:

- **`State` is a `VectorField`** — named `ScalarField` components on
  per-variable `TensorProductSpace`s; the component set and names
  are stable over a run (pytree treedef); updates are functional
  (`replace`, `map`, componentwise arithmetic), no mutation
  ([`../grid/classes/fields.md`](../grid/classes/fields.md), State contract).
- **The assembly handshake is normative**
  ([`../grid/classes/grid.md`](../grid/classes/grid.md), grid lifecycle):
  Phase-2 assembly runs `grid.merge_overrides(...)` →
  `grid.negotiate(state_spaces=..., tendency=...)` →
  `grid.freeze()`; `negotiate` returns a `ReshardingReport` and the
  model re-`device_put`s its live fields once.
- **Halo is traced, not declared**: `trace_halo(tendency,
  state_spaces, registry)` dry-runs the tendency on `HaloTracer`s;
  the tendency must be traceable un-jitted, plain Python over
  fields; a module that drops to raw `.data` must declare
  `Module.extra_halo: HaloSpec`
  ([`../grid/classes/decomposition.md`](../grid/classes/decomposition.md)).
- **Modules carry dispatch-override dicts**
  (`self.dispatch["reconstruct"] = weno`) merged into the grid
  registry at assembly; the *call site* of the merge was left open
  by the grid notes ([`../grid/02_rules.md`](../grid/02_rules.md) section
  3.4) and is owed by this design (D4).
- **The grid is fully static** (no dynamic pytree leaves);
  time-dependent geometry and boundary data are *module-owned
  state*; physical parameters never live on the grid
  ([`../grid/01_concepts.md`](../grid/01_concepts.md) sections 2.6, 2.7).
- Model-side eigenmode objects (`omega`/`vec_q`/`vec_p` successors)
  consume `(grid, parameters)` and reuse `State`; they are part of
  this design's scope, not the grid's.

**Amendment owed to the grid notes**: the `State` extension contract
in [`../grid/classes/fields.md`](../grid/classes/fields.md) says physics
parameters come from "model settings/parameter objects" — once
D2 (parameter ownership) is decided, that line is updated to the
module-owned mechanism.

## 3. Target picture (from the ROADMAP)

- **No `ModelSettings`**: `Model(grid=..., tendencies=...,
  diagnostics=..., time_stepper=...)` direct assembly; every
  physical parameter lives in a module.
- **Modules register state fields**: the nonhydrostatic core
  declares `u, v, w`; `ConstantStratification` adds `b`; tracers
  and auxiliary fields are the same mechanism.
- **Everything is one pytree**: state, module state, stepper state,
  and clock form a single traced carry; the whole run is one
  `jax.jit` call.
- **Staged / split time stepping** is designed in from the start:
  tendency terms declare their integration treatment; IMEX and
  by-variable splitting compose; pressure projection is a stage.

## 4. Migration strategy

- Design on paper now (this note set), in parallel with the Phase-1
  grid implementation; the design consumes only seams the grid
  notes already fix (section 2), so it does not block on code.
- Anything discovered here that changes a grid seam is fed back as
  a normative-note amendment, the same mechanism the class docs
  used.
- Implementation follows the roadmap staging (2.1–2.6), nonhydro
  ported first (2.7); the old `framework` stays untouched until
  cutover.

## 5. Precedents

- **Oceananigans.jl** — the closest target for assembly:
  `NonhydrostaticModel(grid=..., coriolis=FPlane(f=...),
  buoyancy=BuoyancyTracer(), tracers=(:b,), closure=...,
  timestepper=:RungeKutta3)`. Pluggable buoyancy/Coriolis
  formulations as values, tracers as declared names, no
  settings object. Also: its pressure solver is a solver object
  owned by the model, not a tendency module.
- **Dedalus v3** — problem/solver split: equations are *added* to a
  problem, the solver owns an IMEX-RK/multistep timestepper that
  consumes declared linear-implicit and explicit parts — precedent
  for terms declaring their integration treatment.
- **diffrax / Equinox** — steppers as pure functions over explicit
  solver-state pytrees, scan-based loops, everything-is-a-pytree
  model composition; precedent for the single-jit run and for
  stepper warm-up state as traced state.
- **SciML (OrdinaryDiffEq.jl)** — `SplitODEProblem` /
  `SemilinearODEProblem` interfaces: the split function *is* the
  problem type, steppers dispatch on it; precedent for making the
  explicit/implicit partition part of the problem declaration
  rather than the stepper.
- **Veros / pyOM** — jax-based ocean model with a settings object:
  a demonstration of the ceiling this redesign avoids.
