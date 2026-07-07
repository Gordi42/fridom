# D1.3 — Who declares the core prognostic variables?

Research report (see [`README.md`](README.md) for status).

## 1. Options

- **A — Model package's Model class declares the core.**
  `nonhydro.Model` (a subclass of `fr.Model`) declares `u, v, w` and
  owns the projection; modules only add extras (`b`, tracers). The
  Oceananigans/SpeedyWeather/MITgcm shape: core fixed per model
  *type*, physics and tracers pluggable around it.
- **B — Everything is module-declared, including the core.** The
  nonhydro dynamical core (velocity declarations + pressure
  projection) is itself a module; `fr.Model` is fully generic;
  `nonhydro.Model` is a thin preset/factory returning a plain
  `fr.Model`. Some module is mandatory in practice.
- **C — Required `core=` constructor slot.** `fr.Model(grid=...,
  core=..., modules=..., time_stepper=...)`: the core is an ordinary
  `Module` (same mechanism as B) but occupies a dedicated required
  argument, making "you forgot the core" a missing-argument error.
- **D — Equation-set objects (Gusto/Dedalus style).** One object
  declares *all* prognostics and all terms
  (`CompressibleEulerEquations(domain, params, active_tracers=[...])`),
  handed whole to a generic stepper.
- **E — Fixed core + tracer registry (refinement of A).** Core
  variables hardcoded, extension restricted to name-declared tracers;
  dynamical fields cannot be added at all. Out immediately — it
  cannot express the driving requirement (`b` dynamically coupled,
  present only with a stratification module) — but its
  error-ergonomics trick is worth stealing.

## 2. Precedents (condensed)

- **Oceananigans.jl**: `NonhydrostaticModel` and
  `HydrostaticFreeSurfaceModel` are *separate structs*, not presets.
  `velocities`, `pressures`, `pressure_solver` are struct fields
  (momentum structurally fixed); `buoyancy=`, `coriolis=`,
  `closure=`, `tracers=(:b,)` pluggable values. The elliptic
  constraint's *algorithm* is pluggable, its *existence* fixed by the
  model type. Cross-validation at construction: `BuoyancyTracer()`
  declares `required_tracers = (:b,)` and the constructor errors —
  the authoritative-variable-set payoff.
- **Dedalus v3**: no core at all; the user declares every field and
  every equation; pressure is just another equation. Zero physical
  validation — mistakes surface as solver failure.
- **SpeedyWeather.jl**: prognostic set fixed *per model type*;
  components pluggable by subtyping; tracers addable with
  `add!(model, Tracer(:co2))` but only **before** `initialize!`
  (allocation boundary).
- **Firedrake/Gusto**: the inversion — no Model class; a
  `PrognosticEquationSet` owns `field_names` and residuals; preset
  subclasses (`ShallowWaterEquations`: `u, D`) are thin classes over
  the generic set; a generic `Timestepper(equation, scheme, io)`
  consumes it. Duplicate names raise at construction.
- **MITgcm ptracers**: fixed Fortran core, compile-time tracer count.
- **jax-cfd**: no model object; the step is a composed closure with
  projection inside — Dedalus-grade freedom, Dedalus-grade absence of
  validation.

Synthesis: *no surveyed framework makes momentum a peer of tracers
inside one Model class* — they either fix the core in the model type
(A/E) or move the whole equation set out as one object (D). Two
transferable lessons: (i) whatever knows the required variable set at
construction time buys the good error messages; (ii) the elliptic
constraint always travels with whatever owns the core variables.
FRIDOM's Option B sits between Gusto and Dedalus: equation content
composed at module granularity with assembly-time validation Dedalus
lacks. Beyond precedent — a real (bounded) risk, mitigated by the
fact that FRIDOM's module system already *is* term-granular
composition today (only the field declarations were frozen).

## 3. Trade-off analysis against the consumers

**Nonhydro pressure projection.** Today three modules
(`TendencyDivergence` → `RFFTPressureSolver` →
`PressureGradientTendency`) hardcoded at the tail of `MainTendency`
by list position. Two structural facts decide D1.3 here:

- The minimal nonhydro core contributes **no tendency terms at
  all**: strip Coriolis, stratification, advection, closures (all
  modules already) and what remains is *declarations* (`u, v, w`)
  plus a *constraint stage*. Under A, the Model subclass would exist
  essentially to hold three field declarations and one stage — and
  would still need D3's stage mechanism and a `pressure_solver` slot
  (reintroducing the `MainTendency` tail problem as a class
  attribute). Under B, one core module co-locates the declarations
  with the constraint that gives them meaning, and takes the solver
  as a constructor argument. A user who has the velocities *cannot*
  forget the projection.
- The old `DiagnosticState` (`p`, `div`) is decisive supporting
  evidence: `p` and `div` exist *only because the projection exists*
  — structurally identical to `b` existing only because
  stratification does. Under B the core module registers them
  exactly as `ConstantStratification` registers `b`; under A they
  need yet another special home on the Model subclass. The "fields
  hide in the model class" smell reproduces itself under A
  immediately.

**Shallowwater.** `p` is core (prognostic, `∂t p = −c²∇·u`), not a
tracer — Option E dies here. Under B: `sw.DynamicalCore` declares
`u, v, p`, owns `csqr`, and contributes the coupled pair (`−∇p` in
momentum, `−c²∇·u` in `p`) as ordinary tendency terms — no
projection stage at all, demonstrating that "core" is *not* secretly
"the thing that owns the pressure solver". Also fixes an old smell:
`sw.LinearTendency` currently mixes Coriolis with the gravity-wave
coupling; under B Coriolis is a separate module and the core owns
exactly the terms coupled to its own declarations. Under A, `csqr`
must live on `sw.Model` (parameter drift back toward ModelSettings)
or in a gravity-wave module that doesn't declare the `p` it
integrates (split ownership).

**Hydrostatic (Phase 3.1).** Under B: a third core module (`u, v`
prognostic, `w` diagnostic, hydrostatic pressure stage) inside the
*same* `fr.Model`. Implicit vertical mixing is an ordinary closure
module with an `IMPLICIT` term (D3); the split-explicit free surface
is a stage — in-core or sibling module is a D3/Phase-3 question
(Oceananigans' `free_surface=` slot suggests
`hydro.DynamicalCore(free_surface=SplitExplicit(...))`).

**Passive tracers.** Identical module mechanism under all options.
The difference: under A there are permanently *two* field mechanisms
(model-declared and module-declared) with two code paths through
assembly, halo tracing, and treedef construction; under B there is
one.

**Coupled multi-model runs (Phase 3).** The strongest structural
argument for B: both coupled models are instances of the *same*
generic `fr.Model` with different module lists — one assembly
pipeline, one carry-shape convention, and the `Coupler` handles a
homogeneous `tuple[fr.Model, ...]`. Under A the coupler must be
generic over heterogeneous Model subclasses — exactly the
N-subclasses-of-`ModelSettingsBase` problem being retired. Secondary
win: module-only models are first-class — a tracer
advection-diffusion test model (ROADMAP 1.7 style) is just
`fr.Model(grid, modules=(TracerDiffusion("c"),), ...)` with no fake
core. Option C forecloses this (the slot demands a core or a `None`
escape hatch that erodes its point).

**Error ergonomics.** A/C win by construction (missing core =
missing argument). B must earn it, and the precedent shows how:
Oceananigans' `required_tracers` is a module-side
requires-declaration checked at assembly — FRIDOM already plans this
seam (D1.5 references, D2 requires). The contract that makes B's
errors as good as A's: every requirement failure is attributed to
the *requiring module* and names the missing components, and modules
may carry a hint string. Residual gap: an empty model gets a
lint-level "state vector is empty" rather than a domain-specific
message — acceptable; the preset path never hits it.

**Discoverability.** Old entry point: `nh.ModelSettings(grid=grid,
f0=f0, ...)` + `nh.Model(mset)`. B's answer is the preset:
`nonhydro.Model(grid=..., coriolis=..., stratification=...,
advection=...)` as a *factory function* returning a plain `fr.Model`
(the Gusto pattern; Oceananigans' constructor shape minus the
special struct). Arguably *better* than the old API: the kwargs are
the physics modules themselves rather than scalar parameters whose
owner is invisible.

**Option C assessed.** Buys only the missing-argument error, at the
cost of a forced taxonomy: the slot is semantically empty (field
declaration and stage contribution are uniform module capabilities
under B), tracer-only models need `core=None`, two-core compositions
are blocked, and "is a stratification module allowed in `core=`?"
has no principled answer. The preset already delivers C's ergonomics
without the slot. Reject; record as the fallback if D3 cannot
deliver module-owned stages.

**Option D assessed.** Gusto's monolithic equation set handles
optional buoyancy the way `active_tracers=` handles moisture —
flags/lists on one constructor — precisely the shape FRIDOM is
escaping (the stratification example would be a core-constructor
flag again, and its coupling terms would live inside the set, not
with the `N²` owner). In FRIDOM a module already *is* a mini
equation-set contribution; grouping is available for free as a
tuple-returning preset ("bundle"). Reject as a separate concept;
adopt the bundle idiom inside B.

## 4. Recommendation

**Option B, with five concrete commitments:**

1. **Each model package ships a dynamical-core module** (working
   name `DynamicalCore`). It declares the core prognostics
   (`u, v, w` / `u, v, p`) with their staggered spaces and roles,
   registers its constraint-owned fields (`p`, `div` for nonhydro),
   contributes the core-coupled tendency terms (SW gravity-wave
   pair) and/or constraint stages (nonhydro projection), and owns
   the core parameters (`dsqr`, `csqr`) and the solver choice
   (`pressure_solver=` constructor argument, defaulted from the
   grid, materialized at assembly).
2. **`fr.Model` stays fully generic** — one Model class for all
   packages and for Phase-3 coupling. `nonhydro.Model` /
   `shallowwater.Model` are thin factories returning a plain
   `fr.Model`; they only build the module tuple and delegate.
3. **Error contract as a normative rule**: modules declare required
   components/roles; assembly validates and attributes failures to
   the requiring module, with optional per-module hint text;
   zero-declaration assembly errors distinctly; `State`
   name-properties raise with guidance when absent (D1.5).
4. **The core module supplies the `State` subclass**
   (`state_type = nh.State`, default `fr.State`; more than one
   provider across the module list is an assembly error). Rationale:
   the core declares the components the subclass's properties
   expose, and explicit-assembly users (not only preset users) then
   get `z.u`/`z.b` with the D1.5 error behavior. (Alternative — a
   `state_type=` kwarg on `fr.Model` supplied by the preset —
   recorded as fallback if a class reference on `Module` proves
   awkward under jaxify.)
5. **Amendment owed to D3**: modules can own *stages* (not just
   tendency terms), and stage ordering is by declared *kind*
   (constraint/projection stages run after tendency accumulation
   within a step), never by module list position. D1.3-B is
   contingent on D3 honoring this; it replaces `MainTendency`'s
   list-position convention wholesale.

Why B over A, compressed: the driving example (`b` only with
stratification) already forces the module-declared mechanism to
exist and work end-to-end through assembly, halo tracing, and
treedef construction; A then keeps a *second*, privileged mechanism
whose only occupants would be three velocity declarations and a
stage slot — and the old code shows where that leads
(`DiagnosticState`, `csqr` on mset, the `MainTendency` tail).
Against the precedent counter-signal ("nobody does this"), FRIDOM is
differently positioned: its module system is already term-granular,
its Model must be uniform for Phase-3 coupling, and jax's treedef
discipline supplies the allocation boundary (assembly freeze) that
every surveyed framework independently converged on.

## 5. Risks and open questions

1. **D3 delivery risk (the load-bearing dependency).** The
   projection's privileged position must be expressible as a
   module-owned, kind-ordered stage. If D3 ends up needing a
   model-level schedule anyway, the fallback is Option C (a `core=`
   slot whose stages anchor the schedule) — record this so a D3
   failure doesn't silently reintroduce list-position conventions.
2. **Preset drift.** If `nonhydro.Model` accumulates conditional
   wiring, parameter plumbing, or state, it *is* `ModelSettings`
   again. Rule: presets build a module tuple and call `fr.Model`;
   anything else is a design defect. Test: explicit assembly and
   preset assembly must produce identical models (same treedef, same
   module list).
3. **Requires/hints quality is a D2 coupling.** Validate with the
   concrete failure `fr.Model(grid, modules=(FPlaneCoriolis(...),))`
   in `06_validation.md`.
4. **Beyond precedent.** Composing the momentum equations from peer
   modules inside one model has no direct precedent. Bounded because
   the core *module* is internally monolithic — composition happens
   between the core and its couplers, not inside the momentum
   equations.
5. **Core granularity.** One module or a bundle? Recommend a single
   module with `pressure_solver=` (possibly `None` for
   constraint-free tests); splitting can come later without API
   breakage since bundles are just tuples.
6. **Cross-package module sharing.** Is `FPlaneCoriolis` one
   framework-level module (requiring `u, v` by role) or per-package?
   Depends on D1.4's role expressiveness; either outcome compatible
   with B.
7. **Hydrostatic free surface**: in-core pluggable value vs sibling
   module — defer to D3/Phase 3; both fit B.
8. **Naming**: `DynamicalCore` vs `MomentumEquations` vs
   `NonhydrostaticEquations`. Cosmetic; decide at sketch time.
9. **`state_type` mechanics** under jaxify (class reference as
   static attribute; collision rule) — small, decide at class-design
   time.

## 6. API sketch (illustrative, not normative)

```python
grid = fr.grid.cartesian.Grid(shape=(512, 512, 32), extent=...)

# --- explicit assembly: rotating, UNSTRATIFIED nonhydro (no `b` anywhere) ---
model = fr.Model(
    grid=grid,
    modules=(
        nh.DynamicalCore(              # declares u, v, w; registers p, div;
            dsqr=1.0,                  #   owns the projection STAGE
            pressure_solver=None,      #   None -> grid-appropriate default
        ),
        fr.modules.FPlaneCoriolis(f0=1e-4),   # requires u, v; declares nothing
        nh.CenteredAdvection(),               # advects every ADVECTED component
    ),
    time_stepper=fr.time_steppers.AdamBashforth(order=3),
)
# state components: ("u", "v", "w"); z.b raises with guidance.

# --- same model WITH stratification and a passive tracer ---
model = fr.Model(
    grid=grid,
    modules=(
        nh.DynamicalCore(dsqr=1.0),
        fr.modules.FPlaneCoriolis(f0=1e-4),
        nh.ConstantStratification(n2=1e-5),   # declares b; contributes +b to
                                              #   the w-term and -N²·w to the b-term
        nh.CenteredAdvection(),
        fr.modules.PassiveTracer("dye"),      # declares dye (ADVECTED)
    ),
    time_stepper=fr.time_steppers.AdamBashforth(order=3),
)
# components: ("u", "v", "w", "b", "dye") — b enters exactly like dye does.

# --- preset (discoverability path; returns the SAME plain fr.Model) ---
model = nh.Model(                      # thin factory, not a subclass
    grid=grid,
    coriolis=fr.modules.FPlaneCoriolis(f0=1e-4),
    stratification=nh.ConstantStratification(n2=1e-5),
    advection=nh.CenteredAdvection(),
    modules=(fr.modules.PassiveTracer("dye"),),   # extras appended
    time_stepper=fr.time_steppers.AdamBashforth(order=3),
)

# --- shallowwater: p is core, no projection stage exists ---
model = fr.Model(
    grid=grid2d,
    modules=(
        sw.DynamicalCore(csqr=1.0),    # declares u, v, p; owns csqr; contributes
                                       #   -grad(p) and -c² div(u) tendency terms
        fr.modules.FPlaneCoriolis(f0=1e-4),
        sw.SadournyAdvection(),        # requires u, v, p
    ),
    time_stepper=...,
)

# --- the mandatory-module failure mode ---
fr.Model(grid=grid, modules=(fr.modules.FPlaneCoriolis(f0=1e-4),), time_stepper=...)
# AssemblyError: module 'FPlaneCoriolis' requires state components
# ('u', 'v') but no module in the list declares them.
# Hint (FPlaneCoriolis): velocities are declared by a dynamical-core
# module, e.g. fridom.nonhydro.DynamicalCore.
```
