# D1.1 — FieldDeclaration surface and the home of initial conditions

Research report (see [`README.md`](README.md) for status).

> **Reconciliation note**: this report assumed the declaration
> carries a *concrete* space (the pre-research D1.2-A leaning). The
> resolved design uses `SpacePattern`s
> ([`d1_2_space_binding.md`](d1_2_space_binding.md)); the `space`
> member below reads accordingly. It also proposed a two-valued
> `kind`; the resolved design adopts the three-valued `lifecycle`
> axis from [`d1_4_roles.md`](d1_4_roles.md), which subsumes it.

## 1. Proposed `FieldDeclaration` member list

The declaration is a **transient assembly input**: modules produce
it, the model consumes it in assembly step 1 (D4), and it is
discarded afterwards. It never lives in the carry, is not a pytree,
and never reaches jit. This is a structural break from the old
`FieldMetadata`, which rode on every `ScalarField` forever; the
grid-layer `FieldMetadata` (name/units/nc-attrs) is what survives on
fields.

```python
class FieldDeclaration:                       # plain frozen object, not a pytree
    name: str                                 # required
    space: SpacePattern                       # required (see reconciliation note)
    lifecycle: Lifecycle = PROGNOSTIC         # PROGNOSTIC | AUXILIARY | DIAGNOSTIC
    roles: frozenset[Role] = frozenset()      # D1.4 vocabulary
    default: float | Callable | None = None   # None -> zeros; mirrors create_field init=
    long_name: str = ...                      # \
    units: str = "n/a"                        #  } assembled into grid-layer FieldMetadata
    nc_attrs: Mapping[str, str] = {}          # /
```

Per-member justification:

- **`name`** — the component key in `State`; the collision unit. The
  assembly collector records *which module produced which
  declaration* as side bookkeeping for error messages — no `owner`
  member needed.
- **`space`** — subsumes the old `position`, `bc_types`, `topo`
  (via `ConstantSpace`/`Profile`), and `is_spectral` wholesale, and
  — via the resolved space's `scalars` — the dtype.
- **`lifecycle`** — deliberately **not a role**. Roles are
  open-ended, opt-in consumer tags: *who acts on this field*.
  Lifecycle is closed, structural, and load-bearing for the stepper:
  PROGNOSTIC fields receive tendencies and are advanced; AUXILIARY
  fields sit in the carry but are updated only by their owning
  module (spatially varying parameters `f(y)`, `N²(z)` profiles,
  time-dependent boundary/geometry data — D2 option C lands here).
  Mixing "does the stepper advance this" into the role set would
  make the one tag every stepper must interpret just another entry
  in an extensible vocabulary. Purely derived quantities (`ekin`,
  `pot_vort`) are **not declared at all** — they are functions
  evaluated at IO/diagnostic time; a diagnostic that needs carried
  storage (a time-mean accumulator, the projection's `p`/`div`) is a
  DIAGNOSTIC/AUXILIARY declaration by the owning module.
  *(Feedback to D2: its option C says "AUXILIARY role" — should read
  "AUXILIARY lifecycle".)*
- **`roles`** — per D1.4, referenced not re-decided. Survival path
  for the old `_flags`: `NO_ADV`/`ENABLE_MIXING`/`ENABLE_FRICTION`
  become the D1.4 role machinery, as `classes/fields.md` promises.
- **`default`** — the *default/background* initializer; the one
  genuinely new member (§2). `None` → zeros; float → constant fill;
  callable → routed through `grid.create_field(space, init=...)`
  (function of physical coordinates matched by name). The module
  constructs the declaration, so the callable naturally closes over
  module parameters — `BetaPlaneCoriolis` declares auxiliary `f`
  with `default=lambda y: f0 + beta * y`, which is what makes D2-C
  parameter fields work without a post-allocation "fill my fields"
  hook. Named `default`, not `init`, to signal it is the overridable
  base, not the IC mechanism.
- **`long_name`/`units`/`nc_attrs`** — pass-through annotation;
  assembly folds them (plus `name`) into the grid-layer
  `FieldMetadata` attached at allocation.

**Explicit kills:**

| Candidate | Verdict | Why |
|---|---|---|
| `position`, `bc_types`, `topo`, `is_spectral` | dead | subsumed by `space` |
| `dtype` / scalars override | dead | derived from the space; a complex field declares a complex-scalars space. Eigenmode States on coefficient spaces are not model-state declarations at all — constructed by the eigenmode object (sketch 4.9). |
| per-field IO on/off | dead | IO selection is run configuration, not field identity — exactly how `_flags` metastasized last time. Rule: **writers default to all PROGNOSTIC fields** and take explicit include/exclude lists; AUXILIARY is opt-in. |
| `_flags` dict | dead | replaced by `roles` |
| serialization helpers | dead here | the declaration is transient; restart persists the carry + a treedef fingerprint (§4) |
| owner/module backref | dead | assembly-side bookkeeping, not a member |

## 2. Where initial conditions live: layer (a) under (b), kill (c)

**Recommendation: defaults in the declaration (a, demoted to
"background"), user ICs as a separate step on the assembled model
(b, the user-facing mechanism), no IC modules (c). Restart replaces
exactly step (b).**

**(a) alone fails** as the user-facing home for a decisive reason:
real ICs are **State-valued and cross-component**. `SingleWave`
builds `(u,v,w,b)` jointly from one eigenvector and normalizes
across components; `Jet` superposes a coordinate-function jet with a
normalized eigenmode perturbation and applies a geostrophic
projection to the whole state (`nonhydro/initial_conditions/`). A
per-field `init=` slot cannot express any of that. Worse,
declarations are written by *module authors*, so user ICs would have
to be threaded through module constructors
(`DynamicalCore(u_init=...)`) — the god-object pattern reborn. But
(a) is exactly right for *defaults*: every module-registered field
gets a well-defined base state chosen by the module that understands
it (`ConstantStratification`: `b=0` means unperturbed
stratification), and auxiliary parameter fields self-initialize.

**(c) is rejected.** Under the single-jit run, module `update` lives
inside the traced scan; an execute-at-start phase would be a second
lifecycle stage bolted onto the module contract for something that
is naturally eager host code. Today's ICs are also structurally
untraceable (`SingleWave` raises on an unresolvable wavenumber after
a concrete check; `Jet` normalizes by a runtime `max()`) — which is
*fine*, ICs run once, eagerly, before the scan — but only if they
are not modules pretending to be pipeline. And (c) has the worst
restart story: reload must *suppress* the IC modules, which is
framework logic that (b) gets for free.

**(b) is the home.** Concretely:

- Assembly ends with the carry fully allocated from declarations:
  `grid.create_field(space, init=decl.default | data=zeros,
  metadata=...)` per field. The model is *runnable* the moment it is
  assembled — a zero/background state is a valid run.
- The user-facing surface is a `set!`-style named-field setter
  (Oceananigans precedent: build the model, then
  `set!(model; u=..., b=...)`; Dedalus v3 likewise):

  ```python
  model.set_fields(u=lambda x, y, z: ..., b=some_array)   # names checked against declarations
  model.set_state(z0)                                     # whole-State replacement, treedef-checked
  ```

  `set_fields` accepts per-name **callables** (routed through
  `grid.create_field(declared_space, init=...)`), **raw arrays**
  (`data=`), or **`ScalarField`s** (space-checked,
  `SpaceMismatchError` on mismatch — no silent interpolation).
  Crucially it re-attaches the *declared* metadata, so users never
  hand-build `FieldMetadata` or fish spaces out of the model.
  Whether the spelling is mutating (`model.set_fields(...)`) or
  functional (`model = model.with_fields(...)`) is owned by D4
  (Model pytree status); the seam — *a named-field setter between
  assembly and run* — is decided here.
- IC **recipes** (`jet`, `single_wave`, random spectra) become plain
  functions `nh.initial_conditions.jet(model, **params) -> State`
  starting from `model.z` (the defaults) and returning via
  `replace`/`map`. They consume the assembled model because it hands
  them everything in one object: the frozen grid (transforms,
  `grid.random`, correct sharding), the declared per-component
  spaces, and **resolved parameters** — `single_wave` needs `f0` and
  `N²` for the eigenmode, arriving through D2's resolution (or
  explicit module objects the user holds). The eigenmode object is
  `nh.eigenmodes.geostrophic(grid, params)` per the grid notes; the
  IC recipe is its natural consumer.
- Mental-model continuity: today
  `model.z = nh.initial_conditions.Jet(mset, ...)`; tomorrow
  `model.set_state(nh.initial_conditions.jet(model, ...))`.

## 3. Defaults and validation

**Uninitialized declared field → zeros (or the declared `default`).
Not NaN-poison.** Argued:

- Zero is the physically meaningful rest/background state for
  essentially every prognostic variable here, and *module-registered*
  fields make NaN-poison actively hostile: adding
  `ConstantStratification` or a closure that declares a work field
  would break every existing script until the user initializes a
  field they didn't knowingly create. Defaults-are-safe is the whole
  point of module-owned declarations.
- Perturbation-only ICs (initialize `u`, leave the rest) are the
  dominant use pattern (every IC in `nonhydro/initial_conditions/`
  touches a subset).
- NaN-poison interacts badly with 2.4's NaN-check/early-exit under
  scan: a forgotten IC would surface as a generic step-1 NaN abort —
  strictly worse diagnostics.
- The forgotten-IC risk is handled host-side for free: at run start
  the model logs which fields were user-initialized and which run at
  their declared defaults ("`b`: default (zeros,
  ConstantStratification)").

**Initializing a nonexistent field → immediate, eager error.**
`set_fields(psi=...)` raises at call time (host code, pre-jit)
listing declared names with provenance and a near-match suggestion:
`Unknown field 'psi'. Declared fields: u, v, w (DynamicalCore), b
(ConstantStratification). Did you mean 'b'?`. Companion checks: a
`ScalarField` on the wrong space → `SpaceMismatchError`; wrong-shape
array → caught by `create_field(data=)`; `set_state` with a
differing component set → treedef error naming the added/missing
components.

## 4. Restart interaction

Restart is clean precisely because ICs live *outside* assembly: **a
restart is "skip the IC step", nothing else.**

- The full carry `mz = (z, module states, stepper state, clock)` is
  one pytree with a stable treedef. Snapshot = serialize its leaves
  (equinox `tree_serialise_leaves` idiom / orbax / the 2.6
  TensorStore writer) plus a **treedef fingerprint** (hash of
  declaration names/spaces/lifecycles and module structure) in the
  snapshot metadata.
- Reload path: run normal assembly (defaults allocated, harmlessly),
  check fingerprint, overwrite all leaves from the snapshot, enter
  the scan. Fingerprint mismatch → clear error naming what changed
  ("snapshot has no field 'c'; declared by PassiveTracer — was this
  module added since the snapshot?"). Defaults being mere *values*
  is what makes the layering harmless: loaded leaves simply
  overwrite them. Homes (a)-alone and (c) would both need
  suppression logic.
- Under the single jit: the **chunked scan is the checkpoint
  boundary** (D4 open point, confirmed here). Snapshots happen
  between chunks at the host boundary; the restart trigger
  (wall-clock interval, SLURM resubmit — today's `restart_module.py`
  mechanics) stays host-side infrastructure. `model.run()` keeps
  today's `should_reload()` gesture.
- Because stepper warm-up state (AB ring buffers, `it_count`) is
  *traced carry*, it is captured in the snapshot and restart is
  bitwise-exact — the same reason MITgcm pickups store the AB
  tendency history. Oceananigans mirrors the layering: fields zero
  on construction, `set!` optional, checkpointer restores the full
  model state over it.

## 5. Risks and open questions

1. **`set_fields` spelling (mutating vs functional)** — blocked on
   D4's Model-pytree decision; only the seam's position
   (post-assembly, pre-run) is fixed here.
2. **Default staleness**: a `default` callable closes over module
   parameters at declaration time and is evaluated once at
   allocation. If D2 makes parameters dynamic leaves changeable
   post-assembly, defaults won't refresh. Rule for 02_rules:
   *defaults are evaluated at allocation with assembly-time
   parameters*.
3. **D2 wording**: "AUXILIARY role" → "AUXILIARY lifecycle"; confirm
   AUXILIARY fields are excluded from tendency allocation but
   included in halo negotiation (their consumers stencil them —
   `f(y)` in Coriolis does not, but `N²(z)` under vertical advection
   might).
4. **`default` expressiveness**: no `init_coeff`/`data` variants for
   now; if needed later, a tiny wrapper (`default=fr.InitCoeff(fn)`)
   extends it without new members.
5. **Lifecycle boundary cases**: a "not advanced but
   tendency-visible" category (diagnosed `w` in a hydrostatic model)
   is a *lifecycle* question, not a role — flag for the hydrostatic
   validation walk (06).
6. **Snapshot format vs 2.6**: shared with the TensorStore writer or
   a dumb leaf blob — a 2.6 decision; only the fingerprint
   requirement is set here.

## 6. Sketch

```python
# --- module side ------------------------------------------------------
class ConstantStratification(fr.Module):
    def __init__(self, n2: float = 1.0):
        self.n2 = n2                                  # D2: module-owned parameter

    field_declarations = (fr.FieldDeclaration(
        name="b", space=fr.Collocated(bc={"z": fr.BC.DIRICHLET}),
        lifecycle=fr.Lifecycle.PROGNOSTIC,
        roles=frozenset({fr.roles.TRACER, fr.roles.ADVECTED}),
        default=None,                                 # zeros: unperturbed stratification
        long_name="Buoyancy", units="m/s²"),)
    # ... TendencyTerms: +b in w, -n2*w in b (D3)

# --- user side --------------------------------------------------------
model = fr.Model(grid=grid,
                 modules=(nh.DynamicalCore(), fr.modules.FPlaneCoriolis(f0=1.0),
                          strat := ConstantStratification(n2=1.0), nh.CenteredAdvection()),
                 time_stepper=...)
# model.z now exists: u=v=w=b=0 — runnable as-is

model.set_fields(u=lambda x, y, z: ...)          # single-field IC
model.set_state(nh.initial_conditions.jet(model, strength=2, pert=0.1))

# random-spectra IC (sketch 4.9): coefficient-space construction
q = nh.eigenmodes.geostrophic(grid, coriolis=cor, stratification=strat, s=0)
z_hat = q.map(scale)                              # scale: amplitude × random phase
model.set_state(fr.operators.Fourier(grid, axes=("x", "y", "z")).backward(z_hat))

model.run(runlen=50.0)    # snapshot present? -> leaves overwrite everything above
```
