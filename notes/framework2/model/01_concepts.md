# Model layer redesign — Core concepts and load-bearing decisions

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map, motivation,
and the inherited grid constraints (section 2 there).

Status: **skeleton for discussion.** Section 3 fixes the concept
vocabulary; sections D1–D4 state the four load-bearing decisions as
alternatives with trade-offs. Decisions are resolved in order
(D1 → D2 → D3 → D4): each constrains the next. Once a decision is
made, its section is rewritten as normative text and the alternatives
move to a "rejected" note, following the grid-notes convention.

---

## 3. Core concepts

The model layer decomposes into six concepts, layered on top of the
grid stack (Mesh / FunctionSpace / Field / Operator / Grid):

```
FieldDeclaration  what a module contributes to the state vector
Module            unit of physics/numerics: fields + parameters +
                  tendency + dispatch overrides + module state
State             the assembled VectorField of prognostic components
TendencyTerm      a module's tendency contribution, tagged with its
                  integration treatment (explicit/implicit/stage)
TimeStepper       pure scan-body: advances (z, stepper state) by dt
Model             composition root: grid + modules + stepper;
                  owns assembly, negotiation, and the run loop
```

plus the full dynamic carry (the successor of `ModelState`): the
pytree `(state, module states, stepper state, clock)` that flows
through the scanned step. Working name: keep **`ModelState`**, with
the variable spelled `model_state`.

**Naming rule (decided):** the state vector is always named
**`state`** in signatures, examples, and docs — never `z`, which
collides with the vertical coordinate (`lambda x, y, z: ...` in the
same file). The old `mz` / `mz.z` / `dz` spellings are retired with
it (`dz` has no successor anyway — tendency terms return
contribution dicts, D1.5). If a shorter local name is ever needed,
`sv` (state vector) is acceptable; `z` is not.

Fixed up front (not decision points):

- `State` subclasses `VectorField` (grid contract); components are
  addressed by name; treedef is stable over a run.
- Modules are `jaxify`-registered pytrees: their *structure* is
  static, their numeric attributes (parameters, owned fields) are
  dynamic leaves — so "modules can modify anything" (ROADMAP 2.3)
  falls out of the carry containing module state.
- Host-side infrastructure (progress, writers, NaN handling,
  restart) is **not** a tendency/diagnostic module in the traced
  chain; it lives at the run-loop boundary
  ([`04_run_loop_io.md`](04_run_loop_io.md)).
- **A module contributes any subset of a capability menu** — field
  declarations and references (D1), dispatch overrides, provided
  parameters (D2), tendency terms and stages (D3), and a per-step
  **self-update** of its own dynamic state (parameter leaves, owned
  AUXILIARY fields). "Computes a tendency" is one capability, not
  the definition of a module. Per-step writes are confined to the
  module's **own** state: the old Ramper-style callback that
  mutated *other* modules' parameters is replaced by (i) provided
  values that are **pure functions of the traced clock**
  (`fr.Ramp(...)`-style time-dependent parameters, evaluated by
  the owning consumer inside the trace) and (ii) owner-updated
  auxiliary fields that consumers read from the state vector.

---

## D1. Field registration and State composition

**Status: resolved (signed off 2026-07-07),** including the two
post-research refinements (the tendency read/write rule and the
`state` naming rule) and the module capability menu (section 3).
Each sub-decision was researched in depth against the concrete
consumers
(nonhydro projection, `ConstantStratification`, tracers, Sadourny,
hydrostatic, coupled models) and external precedent (Oceananigans,
Dedalus, Firedrake/Gusto, MOM6/MITgcm, SpeedyWeather, ICON-ART);
the full reports live in [`research/`](research/README.md). This
section states the consolidated proposal; the reports carry the
option analyses and rejected alternatives.

Driving example: the nonhydrostatic core declares `u, v, w`;
`ConstantStratification` registers `b` plus both coupling terms; a
user adds a passive tracer.

### D1.1 The declaration object

([report](research/d1_1_declaration_and_ics.md)) A `FieldDeclaration`
is a **transient assembly input** — plain frozen data, not a pytree,
consumed at assembly step 1 and discarded — *with one D4 amendment*:
the `default=` closures of AUXILIARY declarations are **retained in
the static assembly record** (the re-materialization table), because
`update_parameters` re-runs them; `default=` may also be an
**unbound method of the owning module** (called with the live
module; may read only the owner's own leaves). The grid-layer
`FieldMetadata` (name/units/nc-attrs) is what survives on fields.

```python
FieldDeclaration(
    name="b",                             # component key; collision unit
    space=fr.Collocated(bc=...),          # a SpacePattern (D1.2)
    lifecycle=Lifecycle.PROGNOSTIC,       # PROGNOSTIC | AUXILIARY | DIAGNOSTIC (D1.4)
    roles=frozenset({TRACER, ADVECTED}),  # opt-in consumer tags (D1.4)
    default=None,                         # background init: None->zeros | float | f(coords)
    long_name="buoyancy", units="m/s²", nc_attrs={...},
)
```

Killed relative to old `FieldMetadata`: `position`/`bc_types`/
`topo`/`is_spectral` (subsumed by the space), dtype overrides
(space `scalars`), per-field IO switches (writers default to all
PROGNOSTIC + explicit lists), the `_flags` dict (→ roles),
owner backrefs (assembly bookkeeping).

**Initial conditions are layered, not declared:** `default` is the
module-owned *background* (evaluated once at allocation, with
assembly-time parameters); user ICs are a **separate post-assembly
step** — `model.set_fields(u=callable|array|field, ...)` /
`model.set_state(state0)` (the Oceananigans `set!` pattern) — because
real ICs (jet, single wave) are State-valued and cross-component
(eigenmode + projection) and need D2-resolved parameters. IC recipes
become plain functions `(model, **params) -> State`. IC *modules*
(execute-at-start) are rejected: untraceable code posing as
pipeline, and the worst restart story. Uninitialized fields are
**zeros, not NaN-poison** (module-registered fields make poison
hostile; the run start logs which fields run at defaults).
**Restart = "skip the IC step"**: assemble normally, check a treedef
fingerprint, overwrite all carry leaves from the snapshot; the
chunked scan is the checkpoint boundary, and stepper warm-up state
restores bitwise-exactly because it is traced carry.

### D1.2 How declarations bind spaces: `SpacePattern`, not a grid hook

([report](research/d1_2_space_binding.md)) Declarations are **pure
data**; `Module.field_declarations` takes no grid. The `space` slot
holds a `SpacePattern` — name-keyed semantic tags with a default and
per-coordinate BC structure:

```python
fr.Collocated(bc={"z": BC.DIRICHLET})   # default everywhere
fr.Staggered("x")                       # staggered along x, collocated elsewhere
fr.Profile("z")                         # ConstantSpace on all axes except z (old topo)
fr.SpaceRule(lambda grid: ...)          # escape hatch: full mesh-factory power
```

The model resolves patterns at assembly step 1 through per-mesh
dispatch entries (`("declared_space", mesh)`), yielding the bare
interned `state_spaces` for `grid.negotiate`. Payoffs: the same
module works unchanged on 2D/3D grids (`Staggered("y")` on an (x,z)
grid resolves to collocated — exactly right for `v` in a 2D
rotating slice) and on mixed grids (the grid builder overrides the
z-resolver to Galerkin **once**; no module ever branches on mesh
types); declarations are printable/hashable/testable without a
grid. Resolver entries are **grid-level only** — never mergeable
from module dispatch overrides (cross-module action at a distance).
Precedent: Firedrake's mesh-free `FiniteElement` descriptors bound
by `FunctionSpace(mesh, element)`; Oceananigans' positional
`Field{Face,Center,Center}` tuples are the rejected ceiling. The
original "hook receives the grid" leaning is **rejected**:
machinery (collision checks, negotiate, treedef, debug reports)
wants data, not code, and the hook relocates the hard case
(Galerkin verticals) into every physics module as `isinstance`
ladders.

### D1.3 Everything is module-declared; `fr.Model` is generic

([report](research/d1_3_core_ownership.md)) **Option B adopted**,
with five commitments:

1. each model package ships a **dynamical-core module**
   (`nh.DynamicalCore`): declares `u, v, w` (roles + patterns),
   registers its constraint-owned field (`p` — DIAGNOSTIC
   lifecycle; the old `DiagnosticState` dissolved — `div` is a
   stage-local variable, not declared; amended with D3), contributes the
   core-coupled terms and/or the projection **stage**, owns the core
   parameters (`dsqr`; sw: `csqr`) and the `pressure_solver=` choice;
2. **`fr.Model` stays fully generic** — one class for all packages
   and Phase-3 coupling; `nonhydro.Model` is a thin factory
   returning a plain `fr.Model` (rule: presets build a module tuple
   and delegate — anything more is a design defect; test: preset
   and explicit assembly produce identical treedefs);
3. the **error contract** earns back option A's ergonomics:
   requirement failures are attributed to the requiring module with
   hint text ("velocities are declared by a dynamical-core module,
   e.g. ...");
4. the core module supplies the **State vocabulary class**
   (`state_type = nh.State`; >1 provider is an assembly error;
   fallback spelling: a `Model(state_type=...)` kwarg);
5. **owed to D3**: modules can own *stages*, ordered by declared
   kind (constraint stages after tendency accumulation), never by
   module list position — D1.3-B is contingent on this; fallback if
   D3 fails: a `core=` slot (recorded, not chosen).

Decisive evidence: the minimal nonhydro core contributes *no
tendency terms* — only declarations + a constraint stage — so an
A-style Model subclass would exist to hold three declarations and a
stage slot; and shallowwater shows "core" ≠ "pressure owner"
(`sw.DynamicalCore` declares `u, v, p` and contributes the
gravity-wave pair as ordinary terms, no stage). Coupled runs get a
homogeneous `tuple[fr.Model, ...]`. Module-only models (tracer
advection-diffusion tests) are first-class.

### D1.4 Two axes: `lifecycle` + opt-in roles

([report](research/d1_4_roles.md)) The old `_flags` conflated two
orthogonal questions; they split:

- **`lifecycle`** (mandatory enum, closed): `PROGNOSTIC` (advanced
  by the stepper, in restart) | `AUXILIARY` (module-owned carry
  data: `f_coriolis(y)`, `N²(z)`, time-dependent geometry) |
  `DIAGNOSTIC` (written during the step, read by IO/later stages:
  `p`; amended with D3 — `div` is a stage local, not declared).
  This is what the stepper, restart, and IO branch on;
  checkable (a PROGNOSTIC field nothing advances is an assembly
  error). Derived quantities (`ekin`, `pot_vort`) are *not
  declared at all* — they are functions.
- **`roles`** (frozenset of typed marker objects, open,
  PROGNOSTIC-only): `Velocity(component)`, `ADVECTED`, `TRACER`.
  **Opt-in** — nothing is advected/mixed/projected unless declared;
  templates carry the ergonomics (`FieldDeclaration.tracer("c")` ⇒
  `{TRACER, ADVECTED}`), and opt-in enables the assembly lint that
  opt-out cannot give (in-tree evidence: `NO_ADV` is never set;
  three closures follow three flag conventions today).
  `Velocity` carries an **explicit component label** — never
  derived from the space (B-grid `u`,`v` share one interned space;
  A-grid has no signal; unstructured "edge-normal" isn't per-axis);
  labels are opaque keys, the math always goes through
  space-dispatched operators. `ENABLE_FRICTION`/`ENABLE_MIXING`
  die: closures default targets by role (friction → `Velocity`
  family, mixing → `TRACER`) with name-keyed constructor overrides.
  Roles are namespaced marker objects (`Role("mybgc.nutrient")`) so
  user packages extend the vocabulary with zero framework ceremony.

**Query surface**: assembly collects declarations into a Model-owned
**`FieldTable`** (declaration order = component order); modules
resolve selections once in an assembly-time `bind(table)` hook into
static name tuples (`table.select(ADVECTED)`,
`table.velocity() -> VelocitySelector` replacing the old positional
`self[:3]` slice). No role logic inside the traced step; invisible
to `trace_halo`. **The Sadourny rule**: roles select open sets,
names couple closed sets; a scheme may use both, and every name
coupling is a declared, assembly-checked dependency. A
**coverage lint** errors on double transport (two schemes advecting
`u`) and warns on `ADVECTED` fields no scheme transports.

**Validation amendments (signed off 2026-07-08):**

- **`Velocity` on DIAGNOSTIC fields (V-H2)**: `Velocity` — alone
  among roles — may be declared on DIAGNOSTIC fields (hydrostatic
  diagnosed `w`). Role-driven **reads** (`table.velocity()` as the
  advecting flow, CFL/energy diagnostics) span both lifecycles;
  role-driven **write-targeting** (friction → the Velocity family)
  intersects PROGNOSTIC automatically — physically correct
  (diagnosed `w` has no momentum equation), and listed in
  `model.report`. `ADVECTED`/`TRACER` stay strictly PROGNOSTIC;
  `U, V` stay role-free.
- **The transverse-component rule (V-N1)**: on tensor-product
  grids, `Velocity` labels are coordinate names; a label absent
  from `grid.names` marks a **transverse (slaved) component** —
  excluded from divergence/gradient/advective-flux *directions*
  (its ∂ ≡ 0 by construction) while remaining a full member of the
  Velocity family for friction, CFL, energy, and as an advected
  quantity. This is what makes the 2D (x,z) slice's collocated `v`
  correct in every directional consumer (the space signal alone
  cannot decide it: an A-grid collocated `u` must enter `div`, the
  slice's collocated `v` must not). Assembly validation warns —
  never errors — on label-vs-staggering contradictions.
- **Coverage-lint precision (V-S3)**: the lint counts transports
  **per field across all terms** (including two terms of one
  module); a scheme's role-selected transport set **excludes its
  name-coupled components** (Sadourny transports `u, v, p` by name
  and role-selects only tracers). Untransported-`ADVECTED`
  warnings are emitted as **one aggregated line** (a linear run
  without an advection module warns once, not per field).

### D1.5 Access, collisions, and the functional-update idiom

([report](research/d1_5_access_and_collisions.md))

- **Access**: `state["b"]` is primary and the *only* form module
  code uses (modules stay generic over State classes; the halo trace
  stays State-class-independent). Per-package **vocabulary State
  subclasses** add hand-written convenience properties (`state.b`)
  that raise `MissingComponentError` with a curated hint ("add a
  stratification module, e.g. ...") and the present-component list.
  `__getattr__` fallback is rejected (it masks `AttributeError`s
  raised inside real properties like `ekin`, plus typing opacity);
  generated accessors are rejected (per-instance class synthesis
  breaks pytree registration and is invisible to type checkers).
- **Collisions**: exactly **one owner per name** — duplicate
  declaration is a `FieldCollisionError` naming both modules; no
  silent merge (it would have to reconcile spaces, roles, and
  tendency ownership). Consumers needing fields they don't own
  declare **`FieldReference(name, hint)`** entries, checked at
  assembly (`MissingFieldError` with the hint). No auto-creation —
  a reference carrying a space is a declaration in disguise. Plain
  user tracers enter as one-liner declaring modules
  (`Tracer("dye")`), replacing `mset.custom_state_fields`.
  References are the field-level face of D2's requires mechanism
  (one idiom for module authors). Registration freezes before halo
  tracing/negotiation (MOM6-style lock; treedef stability).
- **Namespace**: flat, prefix-by-convention (all surveyed systems
  are flat; structural namespacing breaks references, IO naming,
  and the vocabulary properties).
- **Tendency inputs — the read/write rule**: `tendency(self, state)`
  receives the **full assembled state vector** — one `VectorField`
  containing *all* declared components regardless of lifecycle:
  PROGNOSTIC fields, AUXILIARY parameter/geometry fields
  (`state["f_coriolis"]`, `state["n2"]`), and DIAGNOSTIC working
  fields (`state["p"]`). **Read access is uniform; write access is
  lifecycle-gated**: contribution dicts may only key PROGNOSTIC
  components (assembly-validated, like all contribution keys).
  Reading a DIAGNOSTIC component yields its value as of the **most
  recent write in stage order** — typically the previous step's
  stage write (warm-start semantics); a consumer that needs a
  diagnostic *after* a specific stage within the step is itself a
  stage (D3 owns the schedule). Note the nonhydro pressure-gradient
  correction is therefore *not* a tendency term at all: it lives
  inside the projection stage, which receives the provisional state
  and overwrites the velocities (and writes `p`) via
  `replace`. AUXILIARY components are written only by their owning
  module's own update (D2-C), never through contributions.
  *(Signature note, added with D2: the traced entry points also
  receive the per-stage resolved-parameters mapping and the clock —
  exact packaging owed to D3, see D2 reconciliation 5.)*
- **Tendency idiom**: modules **return contribution dicts**
  (`{"w": +b_term, "b": -N²w_term}`) instead of threading a shared
  `dz` accumulator — required by D3 (IMEX/staged stepping consumes
  the per-term partition, which a threaded accumulator erases), and
  better for testing and halo tracing. The composer accumulates via
  a new **`VectorField.add(**contributions)`** primitive
  (amendment owed to `classes/fields.md`); `replace` remains the
  overwrite idiom (projection stage, ICs). Contribution keys are
  validated in the assembly-time dry run, so typos fail at assembly.
- **Migration**: raising teaching-shims only (`ImmutableStateError`
  from property setters and `__setitem__`, a raising `.data` setter
  with guidance); a "temporarily working" mutation shim is rejected
  — it would be silently wrong under jit and cannot catch the
  dominant old pattern (`z.b.arr += ...`) anyway. This closes
  `classes/fields.md` open question 3: ports go fully functional
  immediately.

### D1 reconciliations (where the reports disagreed)

1. **Pattern vs grid hook** (the D1.1/D1.3/D1.4 reports assumed the
   hook; the D1.2 report argued patterns): patterns adopted;
   everything else in the other reports ports cleanly — their
   `space=` slots become patterns.
2. **`kind` vs `lifecycle`** (D1.1 proposed a two-valued kind, D1.4
   a three-valued lifecycle): lifecycle adopted; `DIAGNOSTIC` is
   real (the projection's `p`/`div` — carry-resident for warm
   starts, not required valid at step start).
3. **Who supplies the State class** (D1.5: `state_cls=` from the
   preset; D1.3: the core module): the core module, so explicit
   assembly gets the sugar too; the Model kwarg is the fallback.
4. **Tendency signature** (the D1.4 sketch threaded `dz`): the
   D1.5 contribution-dict idiom supersedes it; D1.4's `transports`
   coverage-lint property is unaffected.

### D1 residual open points

Carried in [`07_open_threads.md`](07_open_threads.md):
unstructured-factor tag vocabulary (`EDGE_NORMAL`); typo'd
coordinate names in patterns (assembly resolution-table logging /
optional `require=`); `state_type`-under-jaxify mechanics;
`DIAGNOSTIC` vs module-private storage for `div`; multi-velocity
futures (`table.velocity()` ambiguity under split-explicit /
coupling); `set_fields` mutating-vs-functional spelling (D4);
the `default`-staleness rule for 02_rules.

---

## D2. Parameter ownership and cross-module references

**Status: resolved (signed off 2026-07-07).** Researched as
four sub-questions against the full old-code consumer inventory and
external precedent (Equinox/Flax pytree-aliasing lore, Oceananigans,
Dedalus, Gusto, ClimaParams, Oceanostics); full reports in
[`research/`](research/README.md). The scaffold's leaning ("B + C
complementary, A as escape hatch") is confirmed and concretized.

### D2.1 The resolution mechanism: accessors at assembly, values per stage

([report](research/d2_1_resolution_mechanism.md)) Modules publish
scalars with `ParameterDeclaration(name, attr=...)` — the
declaration names **where the value lives** (a dynamic-leaf
attribute of the owner), never a frozen copy — and consume with
`ParameterReference(name, hint, default=REQUIRED)`, the exact twin
of `FieldReference` (one idiom for module authors; same
collision/hinted-error contract; checked and frozen at assembly).
The one divergence from fields: a scalar reference may carry a
**physically-identity default** (`scaling.rossby` → 1.0, forcing
amplitudes → 0), so dimensional models need no dummy providers;
names like `stratification.n2` are registry-marked no-default.

**Delivery semantics — the jax-aliasing answer.** Pytrees are
trees, not DAGs: a consumer storing a provider module would become
*two silently-diverging copies* on unflatten (the documented
`eqx.nn.Shared` problem). So consumers never hold provider objects
or assembly-frozen values. Resolution builds a static **binding
table** `{name: (module_slot, attr)}`; inside the trace,
`eval_params(modules, t)` reads the live leaves fresh **at each
stage time** and delivers a `params` mapping to every traced entry
point. This gets, for free: between-run leaf updates without
retrace, `fr.Ramp` values evaluated at RK sub-stage times, and one
source of truth. An assembly lint errors on any jaxified module
appearing twice in the carry. Assembly-time reads of
time-dependent values raise unless spelled `at_time(0.0)` — with
the rule "grid factor at bind, parameter factor in-step" (kills the
old `BiharmonicClosure` stale-coefficient bug class).

**Namespace**: parameter names are dotted by physics concept
(`coriolis.f0`, `stratification.n2`, `nonhydro.dsqr`,
`scaling.rossby`) — never by module class (provider
substitutability); field names stay dot-free, so the two tables are
visually disjoint (lint-enforced). A canonical-name registry
(`fr.params.CORIOLIS_F0` — name + units + default hint) mirrors
D1.4's Role markers for typo-proof cross-package interop.

### D2.2 Representation: the placement rule, and `fr.Ramp`

([report](research/d2_2_representation.md)) The old dual
`f0`-scalar + `f_coriolis`-field with syncing setters dissolves
into four rules:

- **R1 — field iff consumed pointwise** in the field algebra;
  a parameter consumed only as an algebraic coefficient (`dsqr`,
  `rossby_number`, diffusivities) is a plain dynamic leaf, never a
  field.
- **R2 — always-field within a family**: every coriolis /
  stratification / sw-`csqr` module declares the AUXILIARY field,
  the constant member on `fr.Profile()` (all-ConstantSpace, **one
  DOF** — the grid notes' broadcast makes the identical consumer
  line handle f-plane and beta plane; no `isinstance` ladders).
  The deeper rationale (sign-off note): these quantities are
  *intrinsically spatial functions* — `f(y)` on a beta plane, even
  `f(x, y)` under curvilinear coordinates — so the field is the
  honest representation and the constant is merely its special
  case; the one-DOF ConstantSpace encoding makes the honest
  representation also the free one. Corollary: declared spaces are
  static, so f-plane vs beta-plane are two module types (a factory
  can paper over ergonomics).
- **R3 — inputs are the source of truth; the field is
  owner-derived**: materialized at allocation, re-materialized by
  the functional `model.update_parameters(...)` API at run
  boundaries (same treedef → no recompile), recomputed in
  `self_update` only when time-dependent. No setters, ever.
- **R4 — scalar provides are a second read surface** for host-side
  analytic consumers, not a second source of truth. Rule: inside
  the trace the field is authoritative; provided scalars are
  assembly/host-side.

**Static/dynamic discipline** (hardened `jaxify`): dynamic leaves
are coerced through `jnp.asarray` (structural values fail loud at
construction); static attributes must be hashable non-arrays;
**provided parameters must be dynamic** (closes the
silent-recompile-per-sweep hole for exactly the parameters that get
swept).

**`fr.Ramp`** is a frozen callable pytree *value*, not a module:
shape/curve **static** (changing the curve is different math — one
recompile, correct), endpoints/timing **dynamic leaves** (sweeping
ramp targets never recompiles; the diffrax split). Branch-free
(`jnp.clip` + shape), so it is valid at every scan step. Universal
idiom: traced code reads scalar slots through
`fr.resolve_at(value, t)` — identity on plain scalars, so every
scalar parameter is Ramp-able with zero consumer changes. Boundary
with `self_update`: *a Ramp describes a curve and never writes;
scalars get time dependence via `resolve_at` at the point of use;
field-consumed parameters get it via the owner's `self_update`
rewriting the AUXILIARY field* — and assembly schedules a module's
self-update **only if** one of its inputs is time-dependent, so
static models pay nothing.

**The `rossby_number` successor**: owned and provided by the
**dynamical-core module** (Ro is a property of the
nondimensionalized equation set — the docstrings write
`∂t u + Ro u·∇u`; it pairs with `dsqr`/`csqr`, which D1.3 already
put there; the grep-verified consumer list spans advection,
closures, diagnostics, eigen/IC recipes, OptimalBalance). Generic
consumers stay Ro-ignorant via reference-valued constructor slots
(`scaling=fr.Param("scaling.rossby", default=1.0)`); the old
Ramper-plus-setter spin-up of the nonlinear term becomes
`DynamicalCore(rossby_number=fr.Ramp(...))`. A dedicated
`NondimensionalScaling` module is recorded as rejected
(a fieldless one-number module per package; the core slot already
exists).

### D2.3 Diagnostics: functions are the primitive

([report](research/d2_3_diagnostics.md)) Verified against the old
property bodies, **the parameter-free diagnostic class is nearly
empty** — even `ekin` carries `dsqr` (nh) / `Ro²·csqr` (sw); only
`rel_vort_z` (nh) and `rel_vort`/`spectral_ekin` (sw) qualify. So:

- **Primitive**: pure package-level functions
  `nh.diagnostics.pot_vort(state, *, f0, n2, rossby_number, dsqr)`
  — explicit params (scalar or field), no model back-reference,
  work on any State, jit-traceable; each carries `ParameterReference`
  annotations consumed only by the binding layer.
- **Bound form**: `model.diagnostics.pot_vort()` — assembly wraps
  the functions with parameters resolved through `model.parameters`,
  lazily (hinted `MissingParameterError` only when an
  absent-provider diagnostic is actually called). Notebook
  workflow: `model.diagnostics.etot().xr.plot()`.
- **Writer seam**: writers take named expressions
  (`derived={"pv": model.diagnostics.pot_vort}`) evaluated only at
  output cadence — every output is a pure
  `(model_state) -> Field | scalar`; where it evaluates (in-trace
  `lax.cond` + `io_callback` vs chunk boundary) is 2.6's decision.
- **Rejected as default**: diagnostic-provider modules (always-on
  step cost, carry growth, staleness confusion — MITgcm's registry
  is the anti-pattern); retained as the escape hatch for genuine
  step-frequency accumulation (time means, budgets) — *amended at
  the coupling sign-off (2026-07-08)*: the accumulation home is an
  **S6 DIAGNOSTIC-kind stage**, not `self_update` (which runs per
  substage and multi-counts under RK; `cadence=STEP` reserved) —
  see 02_rules "The S6 accumulation idiom".
- The old `epot` silent `N²=0` formula switch dies (hinted error;
  the unstratified form is user algebra); integrated scalars
  (`total_energy`) are the same primitive at rank 0 feeding a
  TimeSeries sink (2.6); the D4 constructor's `diagnostics=` kwarg
  is repurposed as IO/writer config.

### D2.4 The host-side surface and the eigenmode seam

([report](research/d2_4_host_consumers.md))

- **Three access tiers**: `model.parameters` (read-only mapping,
  values read live from the carry, hinted errors, no attribute
  sugar; **Ramp-valued slots return the Ramp object** —
  `at_time(t)` evaluates explicitly) → `model.module(Type,
  name=...)` typed lookup for unpublished knobs → direct
  constructor references, valid only up to assembly ("read
  post-assembly values through the model").
- **Mutation rules**: pre-assembly free; assembly snapshots derived
  data; **post-assembly attribute mutation raises**
  (`ImmutableParameterError` teaching-shim — it would silently miss
  baked aux fields and solver precomputes); sanctioned in-run
  dynamics are traced only (Ramp, self-update); sanctioned
  between-run changes go through `model.update_parameters(...)`
  (re-materializes, same treedef); sweeps re-assemble (identical
  treedef → shared jit cache).
- **Eigenmode seam**: both constructors —
  `nh.eigenmodes.Eigenmodes(grid, f0=..., n2=..., dsqr=...)` for
  standalone linear theory (verified: nh eigenmodes need exactly
  `f0, n2, dsqr`, *not* `Ro`), and `from_model(model)` extracting
  via `model.parameters` with structural validation (constancy via
  provides — a profile-stratification module simply doesn't provide
  the scalar; Fourier-diagonalizability; `f0 == n2 == 0` error).
  **Ramp-valued parameters error in `from_model`** unless
  `at_time=` is passed (an eigenmode set is a fixed-time snapshot;
  silent `t=0` would report wrong wave periods). Stepper dispersion
  (`time_discretization_effect`) stays a stepper concern applied by
  recipes.
- **Projections**: the old `Projection` base (mset-consuming) is
  deleted; geostrophic/wave/divergence become `em.projector(s)`
  callables; `GeostrophicTimeAverage` and `OptimalBalance` become
  host-side drivers consuming assembled `fr.Model`s (OptimalBalance's
  `update_parameters` mutation → a `fr.Ramp`-valued scaling,
  verbatim); the pressure projection is the traced in-step stage,
  on the other side of the trace boundary.
- **Pressure solvers verified: zero D2 machinery** — both old
  solvers read only `dsqr`, whose new owner also owns the solver
  slot and stage; the feed is owner-internal. *(Wording amended at
  validation, V-N: the solver binds the eigenvalue-**Symbol
  structure** at assembly and reads the live `dsqr` **leaf
  in-step** — `1/(k_h² + k_z²/dsqr)` is not factorable into
  grid × parameter, and baking the value would go stale under a
  dsqr sweep; XLA hoists the loop-invariant inverse.)*

### D2 reconciliations (where the reports disagreed)

1. **Namespace spelling** (D2.2/3/4 used flat keys, D2.1 dotted):
   dotted-by-concept adopted (disjoint from field names,
   registry-backed); the other reports' `"f0"`/`"n2"` keys read as
   `fr.params.CORIOLIS_F0`/`.STRATIFICATION_N2`.
2. **Who owns `rossby_number`** (D2.1: a NondimensionalScaling
   module; D2.2: the dynamical core, with the full consumer
   inventory): the core, per D2.2; D2.1's *mechanism*
   (default-1.0 references) is unchanged.
3. **`model.parameters` × Ramp** (D2.1: evaluate at current clock;
   D2.4: return the Ramp, explicit `at_time=`): D2.4 adopted — it
   is the safer contract and matches D2.1's own assembly-read rule.
4. **Consumer spelling** (D2.1: read `params[...]`; D2.2:
   `fr.Param(...)` constructor slots + `resolve_at(self.x, t)`):
   both, layered — `fr.Param(name, default=...)` is the
   *declaration spelling* of a defaulted `ParameterReference` in a
   constructor slot; delivery is always D2.1's binding-table/
   per-stage evaluation; owners may read their **own** leaves
   directly through `resolve_at`. Cross-module reads never touch
   another module object.
5. **Traced signatures**: D2 needs `params` (and the clock for
   `resolve_at`/forcing) in the traced entry points. Exact
   packaging — `tendency(self, state, params, clock)` vs a small
   context object — is **owed to D3** with the rest of the
   signature set; D1.5's `tendency(self, state)` is amended to
   "state plus the D3-packaged context".

### D2 residual open points

Carried in [`07_open_threads.md`](07_open_threads.md): the
explicit-wins mechanism (constructor-set value suppressing a
declared reference — `USE_PROVIDED` sentinel proposal); the
`model.update_parameters` / re-materialization lifecycle hook (D4);
restart-fingerprint treatment of parameter specs (Ramp shape =
structure, endpoints = leaves); `cfl`'s access to the stepper `dt`
(D3); `em.omega_at(k, s)` scalar accessor and non-Fourier eigenmode
families (Phase 2.7); diagnostics-result metadata rule and the
dotted-name lint (02_rules).

---

## D3. The step abstraction (staged / split stepping designed first)

**Status: resolved (signed off 2026-07-07).** Researched as
four sub-questions (term surface, stepper core, stage schedule,
IMEX/splitting) with all seven D1/D2 fed-forward requirements
discharged; the **full design lives in
[`03_time_stepping.md`](03_time_stepping.md)** (sections 5.1–5.9);
research reports in [`research/`](research/README.md) (d3_1–d3_4).
The summary:

- **Terms** are declared frozen objects
  (`TendencyTerm(name, fn, treatment, advances, transports,
  implicit, linear)`; `@fr.term` sugar), referencing **unbound**
  methods paired with module slots at compose time (the D2
  aliasing rule applied to behavior). Treatment is author-declared
  with the user override on the module constructor; implicit terms
  carry a two-capability `ImplicitOperator` (`apply` = forward
  `L·X`, `solve` = γ-agnostic `(1−dt·γ·L)⁻¹`), with mergeable
  framework families (κ-summing `VerticalDiffusion`) and at most
  one non-mergeable custom per field. Terms only add; overwrites
  are stages.
- **The step schedule** is kind-ordered, never list-positioned
  (discharging D1.3 commitment 5): per substage
  `SELF_UPDATE → DIAGNOSE → terms → ADVANCE(s) → CONSTRAINT`, plus
  a per-step epilogue (NaN seam → `DIAGNOSTIC` stages). A stage
  body is an arbitrary pure function writing its declared subset;
  reads see the nearest preceding write in schedule order
  (Gauss-Seidel falls out). `p` is DIAGNOSTIC; `div` is a stage
  local, not declared.
- **The stepper** is a pure scan body: a pytree whose only dynamic
  leaf is `dt` (sweeps and sign-flipped backward runs never
  recompile; `run_backward` dies as a method), with multistep
  history and a saturating warm-up counter as traced
  `StepperState` carry (bitwise mid-warm-up restart; the first
  chunk is the same trace as every other) and RK stage values as
  locals. The Clock is traced float64 `start/elapsed/it`; calendar
  stays host-side.
- **Splits compose**: IMEX by treatment (CNAB2/SBDF2 ship in 2.5
  against a reference vertical-diffusion consumer; IMEX-RK is a
  design-frozen slot), by-variable via module-owned ADVANCE stages
  (the split-explicit free surface is ordinary PROGNOSTIC fields
  `eta, U, V` — no Velocity role — advanced by a module-owned
  subcycle stage; nested mini-models rejected). The projection is
  **project-the-state** after every advance — exactly equivalent to
  the old project-the-tendency for explicit schemes (cutover-safe),
  self-correcting, and the only coherent IMEX arrangement
  (Oceananigans-verified, including the `p = φ/stage_dt`
  physical-pressure normalization).
- **Signatures**: every in-trace hook is
  `(self, state, ctx) -> dict` with a frozen `StepContext`
  (`params`, `clock`, `dt`, `stage_dt`, per-treatment tendency sums
  for post-tendency stages) — kwargs at the notebook boundary, ctx
  inside the trace. This amends D1.5's `tendency(self, state)` as
  D2 reconciliation 5 anticipated.

Reconciliations and residual open points: `03_time_stepping.md`
§5.8–5.9. Amendments owed on sign-off: the D1.3/D1.5 "(p, div)"
wording drops `div`; the consolidated restart-fingerprint rule
(per-term treatments + stepper statics + Ramp specs).

---

## D4. Model composition and lifecycle

**Status: resolved (signed off 2026-07-08;** NaN-check cadence is
default-per-step *pending benchmark* — a cadence knob is sanctioned
if profiling demands, see `04_run_loop_io.md`). Researched as
four sub-questions (assembly/Model object, run loop, IO seams,
lifecycle/coupling); the **full design lives in
[`04_run_loop_io.md`](04_run_loop_io.md)** (sections 6.1–6.9);
research reports in [`research/`](research/README.md) (d4_1–d4_4).
All six D2/D3 fed-forward requirements are discharged, and the last
grid-note debt (the dispatch-merge call site) is written out as an
amendment. The summary:

- **Constructor**: `fr.Model(grid, modules, time_stepper (required,
  no default), io=(), state_type=None, name=None)`; presets are
  thin factories tested by treedef identity. **`__init__` is
  assembly** — a nine-step, pure, deterministic pipeline (fields →
  parameters → **dispatch merge** → bind → terms/schedule → dry run
  → negotiate/freeze → allocate-born-sharded → report), with the
  ordering audit finding: the dry run and halo trace must see the
  registry *as merged*, so the merge is step 3.
- **The merge call site (grid amendment)**: `Module.dispatch` is a
  constructor-frozen mapping keyed by `kind` or
  `(kind, SpacePattern)`, model-resolved and merged exactly once
  (first model on the grid); same-key from two modules errors;
  `("declared_space", ...)` never module-mergeable; no
  `Module.setup()` exists.
- **Model is a host-side driver, not a pytree** (a Model-as-pytree
  would double-flatten the carry — the D2 aliasing bug); lifecycle
  methods are mutating spellings over pure carry transformers;
  `model.state` is read-only. Enable/disable flags are dropped
  (re-assembly for structure, `fr.Ramp`-to-zero for continuous
  switch-on).
- **The run loop**: `advance(steps)` is the IO-free primitive
  (coupling interleaves it), `run()` the sugar; one donated
  framework-level `step_chunk(assembly_record, carry, n)` (shared
  jit cache across identical re-assemblies is an implementation
  obligation); chunk boundaries derived from trigger unions;
  per-step `isfinite` panic flag with chunk-boundary abort (no
  `lax.cond` wrapper — per-step GPU sync tax rejected); Ctrl-C
  completes the chunk, zero steps lost; predictive walltime checks
  with `fr.io.resubmit()`.
- **IO seams**: outputs evaluate host-side at chunk boundaries —
  the compiled step is IO-free (adding a diagnostic to a writer
  never recompiles physics); declarative triggers lowered to step
  sets; the Writer's zarr is xarray/xgcm-openable; snapshots are
  true-shape leaf blobs with the fingerprint **and its source
  record** (mismatch errors diff, never silently reuse);
  `Model.restore` classmethod rejected — **persistence = script
  re-assembly + leaf snapshots** (dill dies); the `OutputStream`
  protocol is the one interface 2.6 builds behind.
- **Lifecycle**: `update_parameters(updates, rewarm=True)` resolves
  through the binding table (which now includes the stepper's
  `fr.params.TIME_STEP` — backward runs and dt sweeps use the same
  hook), re-materializes owner-derived AUX fields by re-running the
  owners' declaration defaults (allocation and update share one
  code path — correct by construction), and re-ramps the multistep
  warm-up by default (old-physics buffers); `reset()` re-warms the
  stepper and **resets the clock (= restarts Ramp legs)** without
  touching PROGNOSTIC/AUXILIARY state. **Sweeps**: the sanctioned
  idiom is one-grid-many-models via a frozen-grid **verify path**
  (a fresh grid per sweep point would recompile — fields carry the
  grid as an identity-hashed static); the cheaper leaf-only sweep
  needs no re-assembly at all.
- **Proofing**: multi-device needs nothing per-module (fields born
  sharded; entry points re-home; gather at the IO boundary);
  coupling stays a Phase-3 pure addition (`advance`, AUX-mediated
  exchange, per-model clocks with exact step-count windows, no
  process-global mutable state — audited); the OptimalBalance
  workflow composes end-to-end with no remaining gaps.

Reconciliations and residual open points:
[`04_run_loop_io.md`](04_run_loop_io.md) §6.8–6.9. Amendments owed
on sign-off: the grid-notes merge-call-site + frozen-grid verify
path; D1.1's "declarations discarded" softened (AUX default
closures retained; `default=` accepts unbound owner methods); three
02_rules entries.

---

## D5. The state-transform algebra (added 2026-07-08)

**Status: resolved (signed off 2026-07-08; NNMD descoped — its
future rewrite is a separate design exercise and will not contain a
model propagator).** The **full design lives in
[`08_state_transforms.md`](08_state_transforms.md)**
(sections 10.1–10.8); research reports d5_1–d5_3 in
[`research/`](research/README.md). The seed bullets below are kept
for the decision record; where they differ from the full design
(signature `rest` policy, `call_with_info`, the predicate set), the
full design is authoritative. The requirement: first-class,
composable `State -> State` objects — a *state algebra* mirroring
the operator algebra — e.g.

```python
vortical = nh.transforms.VorticalProjection(model)
average  = nh.transforms.TimeAverage(model)      # runs a LINEARIZED twin internally
residual = state_ini - vortical(state_ini)       # State arithmetic (already resolved)
ob_iter  = forward @ vortical @ backward         # transform composition (new)
```

The old `fr.projection` namespace (a misleading name — TimeAverage
and the ramped integrations are not projections) dissolves into
this. Seed design:

- **Two tiers, one abstraction.** Tier 1: *closed-form* transforms —
  pure traced field algebra (eigenmode projectors, filters);
  jit-able, cheap, potentially vmap-able (IC ensembles). Tier 2:
  *dynamical* transforms — run a model internally
  (`Propagator(model, steps=N)`, TimeAverage, OptimalBalance);
  host-level drivers over `reset → set_state → advance`, expensive,
  not traceable. The algebra is tier-agnostic; mixed composition is
  host-level (`traceable` flag ANDs).
- **The algebra**: `@` (right-to-left composition, matching the
  operator algebra), `+`/`-` and scalar `*` (pointwise on outputs —
  well-defined by State's vector-space structure; gives the
  complement `fr.Identity() - P`), `** n` (fixed iteration), and
  host combinators (`FixedPoint(T, tol, max_it)`, `Shift(state0)`
  for the affine pieces). No adjoints/inverses (backward is only a
  physical, not algebraic, inverse of forward); no automatic
  idempotent simplification (silent rewriting is a footgun).
- **Laws (normative once resolved)**: (1) *determinism* — `T(state)`
  is a pure function of the input given frozen config; Tier-2 calls
  are `reset(); set_state; advance; read`, and `reset()`'s
  clock-reset-restarts-Ramps semantics is what guarantees it;
  (2) *signature checking at compose time* — `A @ B` validates
  treedef/FieldTable/grid compatibility (the state-level mirror of
  operator space signatures); (3) *isolation* — a Tier-2 transform
  **owns its internal model exclusively** (never the user's; two
  transforms never share one model object — interleaved resets);
  (4) transforms map the PROGNOSTIC subset; outputs are
  `set_state`-compatible.
- **Model variants** (generalized at sign-off of the seed: the
  twin mechanism must cover more than linearity — TimeAverage may
  also want mixing/friction disabled during averaging, and OB's
  ramping propagators likewise). **`model.variant(term_filter=...,
  updates=...)`** builds a derived model: same grid (verify path),
  same module tuple, same declarations — **terms filtered, treedef
  identical by construction** (filters act on terms, never on
  modules or declarations; stages survive — the linear nonhydro
  variant still projects). `term_filter` is a composable
  **term predicate** over the collected `TendencyTerm`s (which
  carry `linear`, treatment, owner, name, advances/transports):
  `fr.terms.linear`, `fr.terms.explicit/implicit`,
  `fr.terms.owned_by(ModuleType_or_base)` (isinstance-based — so
  `~fr.terms.owned_by(fr.closures.ClosureBase)` drops all closures
  without new vocabulary), `fr.terms.named("Module/term")`, combined
  with `& | ~`. `fr.linearize(model)` ≡
  `model.variant(term_filter=fr.terms.linear)`;
  inviscid-linear ≡ `fr.terms.linear &
  ~fr.terms.owned_by(fr.closures.ClosureBase)`. The coverage lint
  downgrades to info under a filter; the filter enters the assembly
  fingerprint. Designed-for, not now: stage filtering (if a variant
  ever needs a clamp disabled), and linearization *about a state*
  (`jax.jvp` of the composed tendency — mechanically free in jax; a
  future `TangentPropagator(model, about=...)`).
- **The renamed family**: `VorticalProjection` / `WaveProjection` /
  `DivergenceProjection` (Tier 1, wrapping `em.projector`; dual
  constructors from-model / explicit-params per D2.4);
  `fr.transforms.Propagator(model, steps|runlen)` (Tier 2, the
  forward/backward building block); `TimeAverage(model, period)`
  (owns a Propagator over the linearized twin + trajectory
  averaging); `OptimalBalance` and `NNMD` as thin presets *written
  in* the algebra. Honesty note: OB does not collapse to one
  expression — the base-point exchange is affine
  (`Shift(z_base) @ (I − P)`) and the update_base_point variant
  changes the iterated map per iteration, which is `FixedPoint`
  policy, not composition; the algebra's value is reusable,
  recomposable, testable pieces.
- **Known new problems** (to harden in the research round): cost
  opacity of composed calls (a `repr` composition tree + cost
  estimates + progress hooks); transform-owned model memory
  (several carries per composition); cross-model signature equality
  rules (term_filter preserves treedefs, module swaps don't);
  vmap/ensemble semantics per tier; time-parametrized transform
  families (out of scope: transforms are autonomous maps).
  **Naming: decided (sign-off 2026-07-08)** — the namespace is
  **`fr.transforms` / `nh.transforms`**, base class
  `fr.StateTransform` (spectral transforms live in `fr.operators`,
  so the namespace is free; the terminological adjacency is
  accepted and noted in docs).

## Resolution order

1. **D1** — field registration (most concrete; the stratification
   example exercises it end to end);
2. **D2** — parameter ownership (D1's declaration/role machinery
   feeds it);
3. **D3** — step abstraction (needs D1's roles and D2's term
   ownership to know what a stage may reference);
4. **D4** — composition/lifecycle (mechanically assembles the
   other three onto the grid lifecycle).

Each resolved decision also gets its API sketch in
[`05_api_sketches.md`](05_api_sketches.md) and a validation walk in
[`06_validation.md`](06_validation.md).
