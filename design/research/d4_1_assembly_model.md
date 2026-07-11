---
status: frozen
date: 2026-07-07
---

# D4.1 — Model composition and assembly pipeline

Research report (see [`README.md`](README.md) for status).

> **Reconciliation note**: this report ruled "one Model per Grid
> instance, hard rule"; the resolved design instead adopts d4_4's
> frozen-grid **verify path** (one-grid-many-models is required for
> sweep jit-cache sharing, since fields carry the grid as
> identity-hashed static aux). The amendment text in §3 is adapted
> accordingly in the consolidated design.

## 1. The constructor surface

```python
model = fr.Model(
    grid=grid,                     # assembled fr.Grid
    modules=(core, coriolis, stratification, advection, closure),
    time_stepper=fr.time_steppers.LowStorageRK3(dt=60.0),   # REQUIRED, no default
    io=(),                         # standing IO config (successor of diagnostics=)
    state_type=None,               # fallback/override for the core-supplied State class
    name=None,                     # report/log attribution (two models, one process)
)
```

- `time_stepper` required with **no default** — there is no
  physics-free default dt; presets supply the package default.
- `io=` confirmed as the `diagnostics=` successor; constructor-passed
  writers name **unbound** diagnostic primitives
  (`nh.diagnostics.pot_vort`), bound at assembly (the model doesn't
  exist yet at constructor time — resolves the D2.3 sketch's
  chicken-and-egg).
- Rejected kwargs, each replaced: `halo=` (traced), `progress_bar=`/
  `nan_checker=`/`restart_module=` (run-time policy), verbosity
  (process-global log level + the report object).
- **Preset rules sharpened**: a preset may build the module tuple,
  forward kwargs, choose default stepper/io, set name; it may not
  subclass Model, hold parameters/fields, register grid-level
  resolvers, or mutate post-construction. **Normative test**:
  preset and explicit assembly produce identical carry treedefs.

## 2. The normative assembly sequence

`fr.Model.__init__` **is** assembly — no separate `setup()`; a pure,
deterministic function of (grid, modules, stepper, io): same inputs
→ identical treedef (what makes shared-jit-cache sweeps true).

1. **Collect field declarations/references**; resolve SpacePatterns
   via grid-level `("declared_space", mesh)` resolvers → interned
   `state_spaces`; build the FieldTable; record the
   name→owner→pattern→space table. Checks: FieldCollision,
   MissingField (hints), lifecycle/role sanity, dot-free names.
   The optional `require=("x",)` pattern kwarg is **adopted** (the
   typo'd-coordinate residual: table logging + require).
2. **Collect parameter declarations/references**; build the binding
   table. Checks: one provider per dotted name, no-default names,
   provided-must-be-dynamic, the duplicate-module aliasing lint,
   explicit-wins/`USE_PROVIDED`.
3. **Collect and merge dispatch overrides** — the owed grid call
   site (§3). Resolve `(kind, SpacePattern)` keys through the step-1
   resolvers; reject `"declared_space"` kinds; same resolved key
   from two modules = `DispatchCollisionError` (module order never
   silently selects an operator). `grid.merge_overrides` exactly
   once.
4. **`bind(table)` hooks** (module order): role selections frozen to
   static tuples; grid-factor precomputes; the merged registry is
   visible (bind-time resolves see final operators); time-dependent
   reads raise unless `at_time(0.0)`.
5. **Collect terms + stages** (post-bind); build the kind-ordered
   schedule; static checks (treatments vs stepper, implicit
   collisions/merges, coverage lint with stage advances-claims,
   same-kind overlap lint, IMEX-RK × split-explicit error); compose
   the step body; collect `extra_halo`.
6. **Composer dry run on halo tracers** — per term/stage, attributed:
   write gates, contribution keys, advances cross-check, result
   spaces, read/write ordering sanity.
7. **`grid.negotiate(state_spaces, tendency=composed_step,
   halo=extra_halo)`** → `ReshardingReport`; then **`grid.freeze()`**.
8. **Allocate the carry** `(state, modules, stepper_state, clock,
   panicked)`: fields born in the negotiated layout (`default`
   evaluated with assembly-time parameters); `stepper.init`;
   float64 clock; panicked=False. `device_put` per report applies
   only to *adopted pre-built* leaves (see verification iii).
9. **Emit the assembly report** (§6).

**Ordering verification against the grid lifecycle** (fixes the
draft's bug):

- (i) **The dry run must follow `merge_overrides`**: the
  decomposition notes require halo aggregation over the registry "as
  merged" (a WENO override's wider stencil must be what the tracer
  intercepts), and dry-run space validation must see the operator
  that will actually run. Merge = step 3, dry run = step 6.
- (ii) **Merge before `bind`**: bind-time precomputes/resolves must
  see final operator identities; overrides needing negotiated
  context are **lazy-factory rows** (the transform-seeding
  mechanism), exposing `OperatorRequirements` unbound.
- (iii) The lifecycle's "walk the state and device_put" clause is
  refined: at first assembly the state doesn't exist pre-negotiate,
  so allocation is post-freeze and fields are born placed; the walk
  applies to leaves predating negotiation (module-built fields,
  `set_fields` inputs — re-homed at set time).
- (iv) Dry run before negotiate is sound: HaloTracer carries bare
  spaces; dispatch keys are layout-free — and attributed physics
  errors precede layout commitment.

ICs stay the post-assembly step (`set_fields`/`set_state`), plus the
optional one-shot constraint application (fed-forward 4f).

## 3. The dispatch-merge call site (the owed grid amendment)

**Decision**: `Module.dispatch` is a constructor-set, frozen mapping
(no `Module.setup(...)` exists — the §3.4 open question closes on
the "another assembly step" side), keyed by `kind` or
`(kind, SpacePattern)` (modules have no concrete spaces
pre-assembly); the model resolves pattern keys and calls
`grid.merge_overrides` **exactly once per assembly, at step 3** —
after field collection, before bind/dry-run/negotiate. Values may be
operator instances or lazy factories. Two modules on one resolved
key = assembly error naming both. `("declared_space", ...)` never
module-mergeable. Precedence unchanged: `(kind, space)` > kind-only
> grid default. A grid used without a model keeps its provisional
registry. *(Amendment text adapted for the frozen-grid verify path
in the consolidated design.)*

## 4. Model pytree status and set_fields spelling

**Model is a host-side driver, not a pytree**: nothing traces
through it; a Model-as-pytree would create a second flatten path to
the carry's leaves — the D2 aliasing bug by construction. It owns
the assembly artifacts (FieldTable, binding table, schedule,
composed step, report, io bindings) as host attributes plus the one
mutable slot `_carry`. **`set_fields`/`set_state` are mutating
methods, functional underneath** (new carry built, reference
swapped); `model.state` is a read-only property (the old `.z` setter
dies). `update_parameters` follows the same pattern (re-materialize
→ swap; same treedef → no recompile; logs the stale-buffer caveat
with the `rewarm` option).

## 5. Enable/disable: dropped

A traced flag can't shrink negotiated halos/layouts and costs a
per-step cond forever; a static flag recompiles anyway. Re-assembly
is the mechanism for structural changes (cheap: shared jit cache);
`fr.Ramp`-to-zero is the correct traced switch-on for continuous
ones. No `enabled` attribute, no `model.disable()`.

## 6. The assembly report

`model.report` (`AssemblyReport`), logged at INFO, printable without
device sync; `model.__repr__` is its header. Sections: header
(grid/stepper/modules), the field resolution table
(name→owner→pattern→space→lifecycle→roles→default, incl. matched
tags and `require=` outcomes), the parameter binding table (incl.
identity-defaults-in-effect), dispatch overrides, the kind-ordered
schedule (with *why* self-updates are scheduled and which implicit
merges happened), halo/layout results (+ which chain set the max),
lint warnings, and a **run-start addendum** (first `run()`): the
defaults-vs-user-initialized provenance table (discharges D1.1's
logging promise).

## 7. Multi-model audit

Process-global things checked: operator registry (grid-private ✓),
space interning (per-mesh ✓), `fr.params`/Role registries (immutable
metadata ✓), jit cache (keyed by statics ✓ — sharing is a feature),
`fr.log` (the `name=` kwarg's job), devices (`Grid(device_ids=)` ✓),
IO (per-model instances). Nothing global introduced; the Phase-3
Coupler sees a homogeneous tuple of independent drivers.

## 8. Risks / open questions

1. **jit-cache sharing is an implementation obligation**: the jitted
   chunk entry point must be framework-level with the schedule as
   hashable static data — a per-assembly closure recompiles every
   sweep member despite identical treedefs. 2.4 rule.
2. `state_type`-under-jaxify mechanics stay open; the kwarg
   unblocks.
3. Two models on one grid — superseded by d4_4's verify path.
4. Writer `derived=` binding: constructor-passed = unbound
   primitives; post-assembly attach = bound spellings (2.6 API).
5. `apply_constraints()` (initial projection convenience) —
   sketch-level, optional by D3 self-correction.
6. Lazy-factory override rows must expose requirements unbound —
   verify at class-design time.

## 9. Sketch

(See the consolidated design and the d4 sketches; the report's mock
assembly-report output is the reference for the report format:
header / fields / parameters / dispatch / schedule / halo-layout /
lint / run-start addendum.)
