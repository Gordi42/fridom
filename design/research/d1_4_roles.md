---
status: frozen
date: 2026-07-07
---

# D1.4 — The role system

Research report (see [`README.md`](README.md) for status).

**Evidence from the old code that shaped the recommendation** (worth
keeping as motivation):

- `NO_ADV` is **never set to `True` anywhere in-tree** — the opt-out
  default exists only for hypothetical user fields; the flag has no
  exercised in-tree consumer-producer pair.
- Flag discipline already failed once:
  `nonhydro/modules/closures/biharmonic_closure.py` loops over
  **all** of `mz.z` with no flag check (it "friction-diffuses"
  buoyancy and any user tracer), `smagorinsky_lilly.py` checks
  `ENABLE_MIXING` for tracers but handles velocities by hardcoded
  name, and `diffusion.py` checks flags properly. Three closures,
  three conventions.
- The projection trio never uses flags at all: it uses
  `mz.dz.velocity`, the **positional slice** `self[:3]`
  (`nonhydro/state.py:134`) — it breaks the moment field order
  changes, which is exactly what module-registered fields will do.
- `sadourny_advection.py` skips `u, v, p` by **hardcoded name list**
  inside a role-style loop — the coexistence problem in miniature.
- Precedent survey: no surveyed framework uses free-form role tags
  as the primary mechanism. Oceananigans uses structural named
  groups (`velocities`/`tracers`/`auxiliary_fields`) plus name-keyed
  NamedTuples for per-field schemes; MOM6/MITgcm/ICON-ART use
  registration-time metadata records with per-field flags; Gusto
  expresses "is transported" as membership in a per-field scheme
  list. Recurring pain points: name-order coupling and reserved-name
  collisions (Oceananigans issue #1015); per-field flag boilerplate
  at scale (solved by ICON-ART's transport *templates*).

## 1. Proposed vocabulary: two axes, not one flag bag

The single most important structural decision: the old `_flags` dict
conflated two orthogonal questions. Split them.

**Axis 1 — `lifecycle` (mandatory, mutually exclusive enum on every
`FieldDeclaration`): who advances this field.**

| Lifecycle | Meaning | Examples |
|---|---|---|
| `PROGNOSTIC` | advanced by the time stepper from tendency terms; must be valid at step start; part of restart | `u, v, w`, `b`, tracers, shallowwater `p` |
| `AUXILIARY` | owned and updated by its declaring module; in the carry; never receives a tendency, never advected/mixed regardless of roles | spatially varying `f_coriolis(y)`, `N2(z)` profile, ramped `N2(t)`, time-dependent metric data (D2 option C) |
| `DIAGNOSTIC` | written during the step by a stage/module, read by IO or later stages; not required valid at step start; carry-resident for warm starts | nonhydro `p` and `div` (today's `DiagnosticState`) |

Justification: this is what the stepper, restart, and IO actually
branch on, and it is checkable (a `PROGNOSTIC` field no term
advances is an assembly error; a role on an `AUXILIARY` field is an
assembly error). Making these peers of `VELOCITY` in one flat set
would invite nonsense combinations (`AUXILIARY + ADVECTED`).
Computed-on-demand diagnostics (`ekin`, `pot_vort`) are *not*
declarations at all — they stay model-level functions; `DIAGNOSTIC`
declarations are only for fields that must persist in the carry
(solver warm starts, stage-to-stage handoff like `div`, cheap IO).

**Axis 2 — `roles` (frozenset of semantic tags, only meaningful on
`PROGNOSTIC` fields): what the field *is*, physically.**

| Role | Structure | Consumed by | Justification |
|---|---|---|---|
| `Velocity(component=...)` | parameterized: a component label, normally a coordinate name | projection (divergence + gradient correction), advection (as the advecting flow), friction closures, CFL/energy diagnostics, vector-aware IO | the one role that must carry structure — see §3 |
| `ADVECTED` | flat | generic advection schemes (the set of transport targets) | successor of `¬NO_ADV`; its own tag because momentum is advected too and shallowwater `p` is advected but is neither a velocity nor a tracer |
| `TRACER` | flat | mixing closures (default target set), per-tracer advection-scheme maps, BGC-style consumers | distinguishes "scalar tracer quantity" from other advected fields: sw `p` must not be mixed (spurious free-surface damping), and velocities are handled by friction, not mixing |

Typical stacks: nonhydro `u` = `{Velocity("x"), ADVECTED}`; `b` =
`{TRACER, ADVECTED}`; sw `p` = `{ADVECTED}` only; a user tracer `c` =
`{TRACER, ADVECTED}`.

**Dropped, with reasons:**

- **`ENABLE_FRICTION` / `ENABLE_MIXING` as declaration-side flags.**
  They are consumer wiring, not field identity: a field would
  declare `ENABLE_MIXING` even in a run with no mixing module, and
  the declaring module must anticipate every closure kind. Instead,
  closures default their target set *by role* (friction → `Velocity`
  family, mixing → `TRACER`) and take name-keyed overrides in their
  constructor (`include=`/`exclude=`/per-field coefficients). This
  is Oceananigans' model and it fixes the in-tree inconsistency.
- **Staggering/position/axis anything** — fully encoded by the
  `TensorProductSpace`; the role system never re-states it. But the
  space is used for *validation*, not *inference* (§3).
- **A `PRESSURE`/`CONSTRAINT` role.** The nonhydro pressure is not
  selected by generic consumers; only the projection (its declaring
  module) touches it, by name. A role with exactly one consumer that
  is also the declarer is a name in disguise — declare it
  `DIAGNOSTIC`, no roles.

## 2. Opt-in, with template constructors carrying the ergonomics

**Recommendation: roles are positively declared (opt-in); nothing is
advected, mixed, or projected unless its declaration says so.** The
ergonomic cost is paid by *declaration templates*, not an opt-out
default:

```python
FieldDeclaration.tracer("c", units="mol/m³")   # => PROGNOSTIC, {TRACER, ADVECTED}
FieldDeclaration.velocity("u", "x", space=...) # => PROGNOSTIC, {Velocity("x"), ADVECTED}
```

Rationale:

- **The architecture already made registration explicit.** Under D1
  every field enters via a module's declaration — there is no
  anonymous custom-field list for which a blanket default must guess
  intent. MOM6's "advected unless flagged off" looks opt-out but is
  really opt-in: calling `register_tracer` *is* the opt-in act;
  Oceananigans' `tracers=(:c,)` likewise. The template constructor
  is FRIDOM's equivalent of that registration act (ICON-ART's
  transport templates are the direct precedent for
  defaults-via-template rather than defaults-via-flag).
- **Opt-in is statically checkable; opt-out is not.** With opt-in,
  assembly can lint: *a `PROGNOSTIC` field that no tendency term
  advances and no role covers* → error/warning ("field 'c' is
  prognostic but nothing advances it — did you mean
  `FieldDeclaration.tracer`?"). The failure mode of a forgotten role
  is a loud, deterministic assembly diagnostic. The failure mode of
  opt-out is silent wrong physics — and the in-tree record (`NO_ADV`
  never once set) shows negation flags don't get exercised.
- **Core fields don't need protection either way**: nonhydro `p` is
  `DIAGNOSTIC` (lifecycle already excludes it); sw `p` genuinely is
  advected.

## 3. Flat vs structured: only `Velocity` is parameterized — explicitly, not space-derived

**Can the space answer "which coordinate is this component normal
to"?** On a C-grid, mostly yes. But it fails as a general mechanism:

- **A-grid**: all velocity components on `Center⊗Center⊗Center` — no
  signal at all.
- **B-grid** (kills space-derivation even on staggered tensor
  grids): `u` and `v` both live on `Right(x)⊗Right(y)` — identical,
  interned, *the same object*. The space cannot distinguish them
  even in principle.
- **Ambiguity**: a flux `u·c` or any field a module happens to place
  on `Right(x)⊗Center⊗Center` would be mis-detected as a velocity;
  space membership is not a semantic claim.
- **Sphere/unstructured** (grid notes §6.3/6.4): a triangular C-grid
  has *one* edge-normal velocity field; "normal to coordinate k" is
  not a well-posed question — the association is per-edge, encoded
  in the mesh, consumed by `div: edge-normal → cell` as an operator
  signature.

**Design:** `Velocity(component: str)` carries an explicit component
label. On tensor grids the label is a coordinate name; assembly
*validates* labels (distinct across velocity declarations; each a
grid coordinate name when the grid is a tensor product) and — where
the space is staggered — cross-checks consistency (a `Velocity("x")`
whose only `Right` factor is `y` is almost certainly a bug: warn).
On a single 2D mesh (sphere) labels are the mesh's coordinate names;
on an unstructured mesh the one edge-normal field uses a
mesh-defined label (e.g. `"normal"`), and the label is *opaque to
the framework* — consumers use it only for keying and stable
ordering, never for math. The math (divergence, gradient, Coriolis
metric terms) always goes through dispatched operators keyed on the
component *spaces*, so projection code written role-generically
survives the A-grid, the B-grid, and the unstructured mesh.

What the role system guarantees, exactly: *the set of fields
constituting the model's velocity, with stable labels and a stable,
declared order* — nothing more. Everything else was already somebody
else's job (spaces, operators). This finally replaces the `self[:3]`
positional slice.

`ADVECTED` and `TRACER` stay flat: no consumer needs per-target
structure from the declaration; per-target *treatment* (WENO for
`c1`, centered for `c2`; per-tracer κ) is consumer-side
configuration keyed by name (§5).

## 4. Extensibility and representation: typed marker objects, open by construction

**Set-of-enums** is closed — a BGC package cannot add `NUTRIENT`
without editing the framework (the MOM6 pain: new per-tracer
behavior = new registry field in framework code). **Open strings**
are typo-prone in the worst way: `select("NUTIRENT")` returns an
empty tuple, which for an opt-in system is silently-inert physics,
and two packages can collide on `"AGE"`. **Recommendation: typed
marker objects** — frozen, hashable, namespaced-key-bearing
instances, consistent with the design's "static, hashable, interned
descriptors" philosophy:

```python
# framework
ADVECTED = Role("fridom.advected")
TRACER   = Role("fridom.tracer")
class Velocity(Role):          # parameterized role family
    component: str             # key = ("fridom.velocity", component)

# user/BGC package — no framework involvement
NUTRIENT = Role("mybgc.nutrient")
```

Identity/equality/hash by key; the class-vs-instance distinction
gives family matching for free: `table.select(Velocity)` matches any
component, `table.select(Velocity("x"))` exactly one. Unknown roles
are inert by design — the model core never enumerates the
vocabulary, it only answers queries. Misspelled-role protection
comes from imports, not the framework: you can't typo an object
reference. Roles being importable objects also gives them docstrings.

## 5. Query surface: the Model answers, at assembly time; State stays thin

Roles were deliberately evicted from `FieldMetadata`
(classes/fields.md), and `State` is contractually a thin
`VectorField` — so **the role table is Model-owned assembly data,
not State data**. Assembly step 1 collects declarations into an
immutable **`FieldTable`** (declaration order = state component
order); modules receive it in their assembly-time bind hook and
resolve their selections **once**, into static tuples of names
closed over by the traced tendency. No role logic ever runs inside
the step — role queries return names (static treedef data), and
traced code just indexes `z[name]`. This keeps jit clean and makes
the role system invisible to `trace_halo`.

```python
class FieldTable:
    def select(self, role, *, exclude=()) -> tuple[str, ...]     # names, declaration order
    def velocity(self) -> VelocitySelector                        # ordered by grid coordinate order;
                                                                  # raises AssemblyError("no velocity fields:
                                                                  #   add a dynamical-core module") if empty
    def declaration(self, name) -> FieldDeclaration
    def require(self, *names) -> None                             # declared dependency, checked here

class VelocitySelector:   # static: (labels, names); callable on the carry
    def __call__(self, z: State) -> VectorField                   # components keyed by label, stable order
```

Runtime "give me all ADVECTED components" is spelled: bind-time
`names = table.select(ADVECTED)`, step-time iterate `names`. Set
operations on selections are plain tuple/set Python at bind time —
no query-language surface needed.

## 6. The Sadourny problem: roles for open sets, names for closed sets

The rule: **a role selects an open set (any number of user fields);
a name couples to a closed set (the specific fields a scheme's
stencil is written against). A scheme may — and Sadourny must — use
both, and every name it couples to is a declared dependency
(`table.require`) checked at assembly.** Sadourny under the new
system:

- bind: `table.require("u", "v", "p")`;
  `self._tracers = table.select(TRACER)` (note: `TRACER`, not
  `ADVECTED` — `u`,`v`,`p` carry `ADVECTED` but Sadourny handles
  them by name; the old `if name in ["u","v","p"]: continue`
  disappears because the query is precise).
- The energy/enstrophy-conserving vector-invariant momentum +
  flux-form `p` update use `z["u"]`, `z["v"]`, `z["p"]` directly —
  field-specific code inside a module is not a failure of the role
  system; it is the intended escape hatch, and *better* than today
  because the coupling is declared and checked instead of implicit.

**Double/missing coverage lint** (recommended, cheap,
assembly-time): every transport-providing module reports which
fields it transports (`CenteredAdvection` → its `ADVECTED`
selection; `SadournyAdvection` → `("u","v","p") + tracers`).
Assembly errors on overlap (two schemes advecting `u` ⇒ double
tendency) and warns on `ADVECTED` fields no scheme transports (the
opt-in safety net of §2). Dovetails with D3's "does a term advance
named fields" — the same declaration serves both.

## 7. Sketch

```python
from fridom.framework2.model import FieldDeclaration as FD, Lifecycle as LC
from fridom.framework2 import roles as R

class NonhydroCore(fr.Module):
    field_declarations = (
        FD("u", space=..., lifecycle=LC.PROGNOSTIC,
           roles=frozenset({R.Velocity("x"), R.ADVECTED}), units="m/s"),
        FD("v", space=..., lifecycle=LC.PROGNOSTIC,
           roles=frozenset({R.Velocity("y"), R.ADVECTED}), units="m/s"),
        FD("w", space=..., lifecycle=LC.PROGNOSTIC,
           roles=frozenset({R.Velocity("z"), R.ADVECTED}), units="m/s"),
        FD("p", space=..., lifecycle=LC.DIAGNOSTIC, units="m²/s"),
    )                                    # p: no roles — only its owner touches it

class ConstantStratification(fr.Module):
    field_declarations = (FD.tracer("b", space=..., units="m/s²"),)

user_tracer = FD.tracer("c", units="mol/m³")     # space defaults to all-collocated

class CenteredAdvection(fr.Module):
    def bind(self, table):                       # assembly-time, once
        self._vel = table.velocity()             # VelocitySelector — static
        self._targets = table.select(R.ADVECTED)
    @property
    def transports(self): return self._targets   # coverage-lint input
    def tendency(self, z):
        vel = self._vel(z)
        return {n: self.scaling * self.advect(vel, z[n]) for n in self._targets}

class PressureProjection(fr.Stage):              # owned by the core (D3)
    def bind(self, table):
        self._vel = table.velocity()             # 1-component on unstructured; 3 here
    def apply(self, z_star, carry):
        vel = self._vel(z_star)
        d = fr.operators.div(vel)                # dispatched per space
        p = self.solver(d)                       # Symbol solve, warm start from carry
        gp = fr.operators.grad(p)
        return z_star.replace(**{name: z_star[name] - self.dt * gp[label]
                                 for label, name in self._vel.items()})

class HarmonicMixing(fr.Module):
    def __init__(self, kh, kv, fields=R.TRACER, exclude=()): ...
    def bind(self, table):
        self._targets = tuple(n for n in table.select(self._fields)
                              if n not in self._exclude)

class HarmonicFriction(HarmonicMixing):
    def __init__(self, ah, av, fields=R.Velocity, exclude=()): ...  # family match

# f_coriolis(y) as module-owned auxiliary state (D2 option C):
FD("f_coriolis", space=fr.Profile("y"), lifecycle=LC.AUXILIARY, units="1/s")
```

## 8. Risks and open questions

1. **`ADVECTED` semantics must be pinned as "target of
   advective-form transport (−v·∇q) by a generic scheme."**
   Conservation/flux-form requirements (sw `p`; FV mass variables)
   are scheme-specific, expressed by field-specific schemes, not a
   role.
2. **Coverage lint depends on D3.** "Which fields does this term
   advance" must be part of the `TendencyTerm` surface for the
   double/missing-advection check to be uniform — feed forward.
3. **`TRACER` without `ADVECTED`** is representable (helpers always
   set both). Recommend: warn, suppressible (a mixed-only field is
   conceivable).
4. **Per-target consumer configuration** (per-tracer scheme, κ) is
   name-keyed constructor mappings validated at bind (`unknown
   tracer 'x' in kappa=` → error).
5. **Background/advecting flows outside the state** (today's
   `advection.background`) are module-owned `VectorField`s, outside
   the role system; document that `Velocity` means *prognostic*
   velocity.
6. **DIAGNOSTIC vs module-private state**: rule "declare it if
   another consumer (IO, later stage) reads it; keep it
   module-private otherwise" — needs a decision for `div`. Low
   stakes; affects IO defaults.
7. **IO defaults by lifecycle** (prognostic+diagnostic written,
   auxiliary opt-in) — defer to task 2.6; supported without
   additions.
8. **Multi-velocity futures** (split-explicit barotropic mode with
   its own depth-mean pair, coupled models) would make
   `table.velocity()` ambiguous. Escape hatch: `Velocity` labels
   plus a possible `group=` qualifier later — designed-for, don't
   build now.
