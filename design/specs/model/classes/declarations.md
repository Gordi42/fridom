---
status: normative
date: 2026-07-13
---

# Model layer redesign — Class designs: declarations

Part of the model-layer class designs; see [`README.md`](README.md) for the document map, the shared template, and the cross-cluster seam anchors. This cluster owns the **declarative surface modules speak**: field and parameter declarations/references, the space-pattern family, lifecycle and roles, time-dependent values, tendency terms with their implicit operators, and the term predicates. The `Module` base that carries these declarations is [`module.md`](module.md)'s; the `FieldTable`/`VelocitySelector` query surface, the assembly pipeline that consumes everything here, and the error-type registry are [`model.md`](model.md)'s. Normative sources: `../01_concepts.md` (D1 full, D2), `../02_rules.md`, `../03_time_stepping.md` §5.1/§5.5, `../08_state_transforms.md` §10.4, and the CS-1..18 constraint list (`../09_coupling_designfor.md` §11.3). Rejected alternatives are **pointed at**, never re-argued (research reports d1_1–d1_5, d2_1–d2_4, d3_1, d5_2).

---

## Module layout

The code lives in `fridom.framework2.model` (part of the parallel `fridom.framework2` package, renamed to `fridom.framework.model` at cutover). Proposed placement for this cluster; [`model.md`](model.md) holds the canonical whole-subpackage tree (the grid notes' doc-04 convention):

```
src/fridom/framework2/model/
    __init__.py            # lazypimp re-exports (below)
    declarations.py        # THIS DOC: FieldDeclaration, Lifecycle, FieldReference
    space_patterns.py      # THIS DOC: Dof, SpacePattern, Collocated, Staggered,
                           #   Profile, SpaceRule
    roles.py               # THIS DOC: Role, Velocity, ADVECTED, TRACER  (fr.roles)
    parameters.py          # THIS DOC: ParameterDeclaration, ParameterReference,
                           #   REQUIRED, Param, USE_PROVIDED
    params.py              # THIS DOC: ParamName + the canonical registry (fr.params)
    time_dependent.py      # THIS DOC: TimeDependent, Ramp, resolve_at
    terms.py               # THIS DOC: Treatment, TendencyTerm, term (@fr.term)
    implicit.py            # THIS DOC: ImplicitOperator, VerticalDiffusion,
                           #   SpectralDiagonal                        (fr.implicit)
    term_predicates.py     # THIS DOC: TermPredicate + the five leaves (fr.terms)
    module.py              # cluster 02 (module.md): Module, Stage, StepContext
    closures.py            # cluster 02: ClosureBase                  (fr.closures)
    field_table.py         # cluster 03 (model.md): FieldTable, VelocitySelector
    model.py               # cluster 03: fr.Model, TendencyComposer, ...
    time_steppers/         # cluster 04 (fr.time_steppers)
    transforms/            # cluster 05 (fr.transforms)
    io/  ops/              # cluster 06 (fr.io, fr.ops)
```

Top-level re-exports contributed by this cluster (lazypimp `__init__.py`, per repo convention — the pattern is noted, not spelled out):

| Top-level (`fr.*`) | Subpackage path | Object |
|---|---|---|
| `fr.FieldDeclaration`, `fr.Lifecycle`, `fr.FieldReference` | `fr.model.declarations` | field declaration surface |
| `fr.SpacePattern`, `fr.Dof`, `fr.Collocated`, `fr.Staggered`, `fr.Profile`, `fr.SpaceRule` | `fr.model.space_patterns` | space patterns |
| `fr.roles` | `fr.model.roles` | role namespace (`Role`, `Velocity`, `ADVECTED`, `TRACER`) |
| `fr.ParameterDeclaration`, `fr.ParameterReference`, `fr.Param`, `fr.USE_PROVIDED` | `fr.model.parameters` | parameter surface |
| `fr.params` | `fr.model.params` | canonical-name registry |
| `fr.TimeDependent`, `fr.Ramp`, `fr.resolve_at` | `fr.model.time_dependent` | time-dependent values |
| `fr.TendencyTerm`, `fr.term`, `fr.EXPLICIT`, `fr.IMPLICIT` | `fr.model.terms` | term surface |
| `fr.implicit` | `fr.model.implicit` | implicit-operator families |
| `fr.terms` | `fr.model.term_predicates` | term predicates (`linear`, `owned_by`, ...) |

`fr.terms.implicit` (the *predicate*) and `fr.implicit` (the operator *namespace*) are distinct names in distinct namespaces — every sketch spelling (7.2–7.10, `../05_api_sketches.md`) resolves unambiguously. Error types raised by this cluster's machinery (`FieldCollisionError`, `MissingFieldError`, `ParameterCollisionError`, `MissingParameterError`, `ImplicitCollisionError`, ...) live in model.md's error-type registry and are only *named* here.

## Cluster-wide rules

These apply to every class below and are not repeated per class:

- **Transient assembly data.** Declarations, references, patterns, and terms are plain frozen host objects — **never pytrees, never in the carry, never reaching jit**. They are consumed by the assembly pipeline (model.md, steps 1–6) and discarded, with exactly one signed exception: **AUXILIARY declarations with callable `default=` are retained in the static assembly record** (the re-materialization table), because `update_parameters` re-runs them (D1.1 as amended by D4; `host_writable` entries are carried in the table but flagged *exempt*, CS-2).
- **The unbound-behavior rule (jax aliasing).** Every callable slot in this cluster that references module behavior is stored **unbound** and paired with a module *slot* at compose/allocation time — `TendencyTerm.fn`, `VerticalDiffusion.kappa`, and owner-method `default=` closures. A bound method would capture the assembly-time module instance while live parameters ride the carry (the D2 aliasing trap). Assembly lints reject bound methods (`__self__` present).
- **Value semantics, hashable.** Patterns, roles, param names, and predicates are frozen, value-equal, value-hashable descriptors — they serve as dispatch-merge keys (`(kind, SpacePattern)`), selection keys, and fingerprint inputs. Unlike grid meshes/spaces they are *not* identity-hashed: two `fr.Collocated()` calls compare equal by design.
- **Namespace discipline** (D2.1, lint-enforced at assembly): field names are **dot-free**; parameter names are **dotted by physics concept**, never by module class. The state vector is always `state`, never `z`.
- **Validation homes.** This cluster defines data and its validity *rules*; the checks execute inside assembly (model.md) and are summarized per class below — this doc does not own check machinery.
- **Naming.** Classes `PascalCase`, members `snake_case`, per `AGENTS.md`. `Collocated`/`Staggered`/`Profile` are PascalCase *factory functions* returning `SpacePattern` (constructor-like sugar; indistinguishable from classes at call sites).

Task tags: `# 2.2` field/parameter registration, `# 2.5` staged stepping (terms/implicit), `# 2.8` state transforms (predicates), plus `designed-for` per the shared template.

---

### Lifecycle

The closed, mandatory lifecycle axis of every `FieldDeclaration` — *who advances this field* (D1.4).

| Aspect | Value |
|--------|-------|
| Kind | closed enum (three members, final) |
| Pytree | static marker (host-side assembly data) |
| Task | 2.2 |
| Design refs | 01 D1.4, D1.5 (read/write gating); 02 (gating, `set_aux` matrix); 03 §5.2 (`div` ruling) |

```python
"""Lifecycle: the closed structural axis of a field declaration."""
from __future__ import annotations

from enum import Enum, auto


class Lifecycle(Enum):
    """Who advances this field; what stepper/restart/IO branch on."""

    PROGNOSTIC = auto()                                        # 2.2
    AUXILIARY = auto()                                         # 2.2
    DIAGNOSTIC = auto()                                        # 2.2
```

Notes:

- **PROGNOSTIC**: advanced by the stepper from tendency contributions; must be valid at step start; in restart. A PROGNOSTIC field that nothing advances (term `advances` ∪ stage advances-claims) is an assembly error.
- **AUXILIARY**: module-owned carry data (`f_coriolis(y)`, `n2(z)`, time-dependent geometry); never receives a tendency, never advected/mixed regardless of roles; written only by its owning module (`self_update` / declaration default re-materialization). Excluded from tendency allocation but **included in halo negotiation** (consumers stencil them — D1 residual rule, stated normatively here).
- **DIAGNOSTIC**: written during the step by a stage, read by IO or later stages (`p`); not required valid at step start; carry-resident for warm starts. Reads see the **nearest preceding write in schedule order**, crossing substage and step boundaries (03 §5.2). Declare DIAGNOSTIC **iff read outside the producing stage** — `div` is a stage local, *not* declared (the D3 ruling); derived quantities (`ekin`, `pot_vort`) are never declared at all — they are functions (D2.3).
- **The three-operation matrix** (02_rules, coupling sign-off): `set_aux` writes *consented* (`host_writable`) AUXILIARY ∪ DIAGNOSTIC components; `update_parameters` re-materializes owner-derived AUXILIARY fields but **skips host-writable ones**; `reset()` zeroes DIAGNOSTIC (including consented ones) to declared defaults but never touches AUXILIARY.
- The lifecycle is deliberately **not a role**: it is the one tag every stepper must interpret; roles are open consumer vocabulary. Mixing them would invite nonsense (`AUXILIARY + ADVECTED`) — see [d1_4](../../../research/d1_4_roles.md) §1.

---

### Role, `Velocity`, and the `fr.roles` namespace

Opt-in, typed, namespaced marker objects tagging what a PROGNOSTIC field *is*, physically (D1.4).

| Aspect | Value |
|--------|-------|
| Kind | `Role` concrete (open by construction); `Velocity` final parameterized subclass; `ADVECTED`/`TRACER` module-level instances |
| Pytree | static markers (frozen, value-hashable by key) |
| Task | 2.2 |
| Design refs | 01 D1.4 incl. both validation amendments (V-H2, V-N1) and the lint precision (V-S3); 03 §5.4 (role-free `U, V`) |

```python
"""Roles: opt-in consumer tags on field declarations (fr.roles)."""
from __future__ import annotations

from typing import Final


class Role:
    """Frozen, namespaced marker; identity/equality/hash by key."""

    def __init__(self, key: str) -> None:                      # 2.2
        """A flat role with a namespaced key, e.g. Role("mybgc.nutrient")."""
        ...

    @property
    def key(self) -> str | tuple[str, ...]:                    # 2.2
        """The namespaced key ("fridom.advected"); Velocity: a pair."""
        ...

    def __eq__(self, other: object) -> bool:                   # 2.2
        """Equality by key (value semantics)."""
        ...

    def __hash__(self) -> int: ...                             # 2.2

    def __repr__(self) -> str:                                 # 2.2
        """Round-tripping repr; roles are importable, documented objects."""
        ...


class Velocity(Role):
    """The one parameterized role family: a velocity component label."""

    def __init__(self, component: str) -> None:                # 2.2
        """key = ("fridom.velocity", component); the label is opaque
        to the framework — keying and stable order only, never math."""
        ...

    @property
    def component(self) -> str:                                # 2.2
        """Explicit component label (normally a coordinate name)."""
        ...


ADVECTED: Final[Role] = ...   # Role("fridom.advected")        # 2.2
TRACER: Final[Role] = ...     # Role("fridom.tracer")          # 2.2
```

Notes:

- **Opt-in, PROGNOSTIC-only — with one signed exception**: nothing is advected/mixed/projected unless declared; templates carry the ergonomics (`FieldDeclaration.tracer`). **`Velocity` — alone among roles — may be declared on DIAGNOSTIC fields** (V-H2: the hydrostatic diagnosed `w`): role-driven *reads* (`table.velocity()` as advecting flow, CFL/energy) span both lifecycles; role-driven *write-targeting* (friction → the Velocity family) intersects PROGNOSTIC automatically — physically correct, listed in `model.report`. `ADVECTED`/`TRACER` stay strictly PROGNOSTIC (assembly error otherwise; `TRACER` without `ADVECTED` warns, suppressible).
- **The component label is explicit, never space-derived** (B-grid `u`,`v` share one interned space; A-grid has no signal; unstructured "edge-normal" is not per-axis — [d1_4](../../../research/d1_4_roles.md) §3, the rejected space-derivation analysis). Assembly *validates*: labels distinct across velocity declarations; label-vs-staggering contradictions **warn, never error** (V-N1).
- **The transverse-component rule (V-N1)**: on tensor-product grids a `Velocity` label absent from `grid.names` marks a **transverse (slaved) component** — excluded from divergence/gradient/advective-flux *directions* (its ∂ ≡ 0 by construction) while remaining a full Velocity-family member for friction, CFL, energy, and as an advected quantity. This is what makes the 2D (x,z) slice's collocated `v` correct in every directional consumer.
- **Family matching is class-vs-instance**: `table.select(Velocity)` (the class) matches any component; `table.select(Velocity("x"))` exactly one. The query surface — `FieldTable.select`, `velocity() -> VelocitySelector`, the coverage lint with its V-S3 precision (per-field counting across all terms; role-selected sets exclude name-coupled components; one aggregated untransported-`ADVECTED` warning) — is **model.md's**; no role logic runs inside the traced step (bind-time name tuples only; invisible to `trace_halo`).
- **Role-free by design**: the split-explicit `eta, U, V` carry no `Velocity` role (diagnostically-slaved transports; `table.velocity()` keeps returning the baroclinic trio — 03 §5.4, resolving the D1 residual without a `group=` qualifier); nonhydro `p` carries no roles (a role whose only consumer is its declarer is a name in disguise). The Sadourny rule: roles select open sets, names couple closed sets; every name coupling is a declared, assembly-checked dependency.
- Rejected (pointers): opt-out `NO_ADV`-style flags (never exercised in-tree), `ENABLE_FRICTION`/`ENABLE_MIXING` declaration flags (consumer wiring, not field identity — closures default targets by role with name-keyed constructor overrides), closed enums and open strings ([d1_4](../../../research/d1_4_roles.md) §4).

---

### Dof, SpacePattern, and the sugar constructors

Name-keyed semantic space tags with a default — how grid-free declarations bind function spaces (D1.2).

| Aspect | Value |
|--------|-------|
| Kind | `Dof` closed enum; `SpacePattern` concrete frozen dataclass; `Collocated`/`Staggered`/`Profile` factory functions |
| Pytree | static host data (value-hashable; also a dispatch-merge key component) |
| Task | 2.2 |
| Design refs | 01 D1.2; 04 §6.2 step 1 (`require=` adopted) and step 3 (pattern-keyed dispatch); 09 CS-16/CS-17 |

```python
"""Space patterns: semantic, grid-free space descriptors (D1.2)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import fridom.framework2 as fr


class Dof(Enum):
    """Per-coordinate semantic tag, resolved per mesh at assembly."""

    COLLOCATED = auto()   # the mesh's default "cell" representation   # 2.2
    STAGGERED = auto()    # the mesh's default dual/face representation# 2.2
    CONSTANT = auto()     # ConstantSpace factor (old topo=False)      # 2.2


@dataclass(frozen=True)
class SpacePattern:
    """Name-keyed semantic tags + default + per-coordinate BCs."""

    default: Dof = Dof.COLLOCATED                              # 2.2
    tags: tuple[tuple[str, Dof], ...] = ()                     # 2.2
    bc: tuple[tuple[str, fr.BC], ...] = ()                     # 2.2
    require: tuple[str, ...] = ()                              # 2.2
    scalars: fr.Scalars | None = None                          # 2.2 (CS-17)

    @classmethod
    def create(                                                # 2.2
        cls,
        default: Dof = Dof.COLLOCATED,
        tags: Mapping[str, Dof] | None = None,
        bc: Mapping[str, fr.BC] | None = None,
        require: Iterable[str] = (),
        scalars: fr.Scalars | None = None,
    ) -> SpacePattern:
        """Convenience constructor accepting mappings; normalizes to
        the hashable tuple-of-pairs canonical form."""
        ...

    def resolve(self, grid: Grid) -> TensorProductSpace:       # 2.2
        """Per mesh factor: pick the tag (default if no name matches)
        and BC, call grid.dispatch[("declared_space", mesh)](tag, bc);
        return the flat interned product (bare, pre-layout)."""
        ...

    def __repr__(self) -> str:                                 # 2.2
        """Round-tripping repr (assembly logs name→pattern→space)."""
        ...


def Collocated(                                                # 2.2
    *, bc: Mapping[str, fr.BC] | None = None,
    require: Iterable[str] = (),
    scalars: fr.Scalars | None = None,
) -> SpacePattern:
    """Collocated everywhere: SpacePattern()."""
    ...


def Staggered(                                                 # 2.2
    *names: str,
    bc: Mapping[str, fr.BC] | None = None,
    require: Iterable[str] = (),
    scalars: fr.Scalars | None = None,
) -> SpacePattern:
    """Staggered along the named coordinates (B-grid: two names),
    collocated elsewhere; at least one name required."""
    ...


def Profile(                                                   # 2.2
    *names: str,
    bc: Mapping[str, fr.BC] | None = None,
    require: Iterable[str] = (),
    scalars: fr.Scalars | None = None,
) -> SpacePattern:
    """ConstantSpace on all axes except the named ones (old topo);
    Profile() is all-constant — the one-DOF R2 parameter field."""
    ...
```

Notes:

- **Declarations are pure data; the hook is grid-free** — `Module.field_declarations` takes no grid; the *model* resolves at assembly step 1, yielding the bare interned `state_spaces` for `grid.negotiate`. The grid-hook alternative is **rejected** (machinery wants data, not code; it relocates the Galerkin-vertical hard case into every module as `isinstance` ladders); positional concrete tuples (the Oceananigans ceiling) are rejected with it — [d1_2](../../../research/d1_2_space_binding.md) §2, Firedrake's mesh-free `FiniteElement` as the adopted precedent.
- **Dimension generality is the feature**: names absent from the grid are simply unmatched — `Staggered("y")` on an (x,z) grid resolves to collocated (exactly right for `v` in a 2D rotating slice, paired with the V-N1 transverse rule above); `Profile("y")` degrades to all-constant. The typo cost is mitigated by (i) the assembly-logged `name → pattern → resolved space` table and (ii) **`require=`** — adopted at D4 sign-off (04 §6.2 step 1): resolution errors if a required name matched no mesh factor.
- **Resolver entries are grid-level only** (`("declared_space", mesh)` — seeded per mesh type, overridable in the grid's defaults *once* by the grid builder): **never mergeable from `Module.dispatch`** (cross-module action at a distance; 04 §6.2 step 3 enforces it at the merge call site). BC structure rides in the declaration because BCs enter the space's interning key and the declaring module owns its field's BC structure.
- **`scalars=`** requests the resolved space's scalars (a complex field declares a complex-scalars pattern — the D1.1 dtype-override kill). It is **Körper-only** (`REAL`/`COMPLEX`, no width) — the CS-17 precision axis is RESOLVED global-precision-only (2026-07-08): accumulator precision is the S6 chunk-cadence idiom's job (host-side float64 accumulation at chunk boundaries), not a per-space width — see Open questions item 3.
- **Pattern-keyed dispatch**: `Module.dispatch` may key overrides by `(kind, SpacePattern)`; the model resolves pattern keys through the same step-1 resolvers before the single merge (04 §6.2). Value equality/hashability of `SpacePattern` is load-bearing there.
- The coupler-AUX space policy (CS-16) is expressible with this vocabulary as signed: Profile-broadcast exchange fields plus an owner-declared indicator AUX for iteration 1; §3.6 trace spaces designed-for (via `SpaceRule` until then).

---

### SpaceRule

The escape hatch: a callable-wrapping pattern with the full mesh-factory power of the rejected grid hook, per field (D1.2).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final |
| Pytree | static host data (wraps a callable; not value-hashable — identity semantics, never a dispatch key) |
| Task | 2.2 |
| Design refs | 01 D1.2; [d1_2](../../../research/d1_2_space_binding.md) §3 (contract), risk 5 |

```python
"""SpaceRule: the full-power per-field space escape hatch."""
from __future__ import annotations


class SpaceRule:
    """Wraps fn(grid) -> TensorProductSpace; same resolve protocol."""

    def __init__(                                              # 2.2
        self,
        fn: Callable[[Grid], TensorProductSpace],
    ) -> None:
        """fn must be pure, static-only, and return a bare
        (pre-layout) space built from grid.factors' factories."""
        ...

    def resolve(self, grid: Grid) -> TensorProductSpace:       # 2.2
        """Call fn(grid); debug mode double-resolves and checks
        identity (cheap — spaces are interned)."""
        ...

    def __repr__(self) -> str: ...                             # 2.2
```

Notes:

- **Contract (normative, from D1.2)**: the callable is pure; runs at assembly step 1, *before* `negotiate`; may not consult the decomposition; returns bare spaces from the grid's own mesh factories. It recovers 100% of the rejected Option A's expressiveness for the ~5% of cases the semantic tags do not cover (coefficient-space AUX fields, unstructured factors until the tag vocabulary grows — Open questions).
- `SpacePattern.resolve` and `SpaceRule.resolve` satisfy one structural protocol; the `FieldDeclaration.space` slot accepts either. Restart-fingerprint note: the fingerprint hashes the *resolved bare spaces* (02_rules, refined scope), so a `SpaceRule` needs no token of its own.

---

### FieldDeclaration

What a module contributes to the state vector — plain frozen data, the collision unit, the metadata source (D1.1).

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen; two template classmethods |
| Pytree | **host object, not a pytree** — transient assembly input; AUX callable defaults retained in the static assembly record |
| Task | 2.2 |
| Design refs | 01 D1.1 (incl. the D4 amendment), D1.4, D1.5; 02 (defaults-one-path, `set_aux`, re-materialization exemption); 09 CS-1/CS-2/CS-17 |

```python
"""FieldDeclaration: a module's claim on one state component."""
from __future__ import annotations

import fridom.framework2 as fr


class FieldDeclaration:
    """Frozen declaration of one field: name, pattern, lifecycle,
    roles, background default, consent flags, annotation."""

    def __init__(                                              # 2.2
        self,
        name: str,
        *,
        space: SpacePattern | SpaceRule,
        lifecycle: Lifecycle = Lifecycle.PROGNOSTIC,
        roles: Iterable[Role] = (),
        default: float | Callable | None = None,
        host_writable: bool = False,
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
    ) -> None:
        """Normalizes roles to a frozenset and nc_attrs to tuple
        pairs; performs local validity checks (dot-free name,
        role/lifecycle compatibility, host_writable gating)."""
        ...

    # read-only attributes: name, space, lifecycle, roles, default,
    # host_writable, long_name, units, nc_attrs

    @classmethod
    def tracer(                                                # 2.2
        cls,
        name: str,
        *,
        space: SpacePattern | SpaceRule | None = None,
        default: float | Callable | None = None,
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
    ) -> FieldDeclaration:
        """Template: PROGNOSTIC + {TRACER, ADVECTED}; space defaults
        to Collocated() — the registration act of D1.4 §2."""
        ...

    @classmethod
    def velocity(                                              # 2.2
        cls,
        name: str,
        component: str,
        *,
        space: SpacePattern | SpaceRule,
        default: float | Callable | None = None,
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
    ) -> FieldDeclaration:
        """Template: PROGNOSTIC + {Velocity(component), ADVECTED}."""
        ...

    def replace(self, **changes: object) -> FieldDeclaration:  # 2.2
        """Functional update (module factories, preset tweaks)."""
        ...

    def field_metadata(self) -> fr.FieldMetadata:              # 2.2
        """Fold name/long_name/units/nc_attrs into the grid-layer
        FieldMetadata attached at allocation (what survives)."""
        ...

    def __repr__(self) -> str: ...                             # 2.2
```

Notes:

- **Transient — with the one amendment**: consumed at assembly step 1 and discarded; but the `default=` closures of AUXILIARY declarations are **retained in the static assembly record** (the re-materialization table), because `update_parameters` re-runs them (D4 amendment to D1.1). Declarations are not pytrees and never reach jit.
- **`default=` forms** (the background initializer, *not* the IC mechanism — ICs are the layered post-assembly `set_fields`/`set_state` step, model.md; [d1_1](../../../research/d1_1_declaration_and_ics.md) §2 records the rejected IC-modules and per-field-`init=` homes):
  - `None` → zeros (**not NaN-poison** — rejected, [d1_1](../../../research/d1_1_declaration_and_ics.md) §3; the run start logs which fields run at defaults);
  - `float` → constant fill;
  - a **coordinate callable** `f(x, y, z, ...)` (parameters matched by coordinate name) → routed through `grid.create_field(space, init=...)`;
  - an **unbound owner method** `(self, grid, space) -> ScalarField` (the D4 amendment; sketch 7.2) — called with the *live* module; may read **only the owner's own leaves** (cross-module-derived AUX values are `self_update` territory or a re-assembly — 02_rules).
  - *Spec concretization (called out)*: the two callable forms are disambiguated by inspection — a first positional parameter named `self` selects the owner-method form; anything else is matched against coordinate names. Confirm at 2.2 (see Open questions).
- **Defaults, one path** (02_rules, normative): the default is evaluated with the owner's *current* leaves through the **same code path** at allocation (assembly step 8) and at `update_parameters` — that identity is what makes re-materialization correct by construction.
- **`host_writable` is lifecycle-polymorphic (AUX ∪ DIAGNOSTIC)** — CS-1: it is the owner's declared consent for host-side, chunk-boundary `set_aux` writes (exchange fields, accumulator resets, assimilation increments — **externally sourced data**, never "user-tunable parameter field"; user profiles go through constructor values feeding the default). Declaring it on a PROGNOSTIC field is a validity error. Host-writable components are **exempt from re-materialization** (CS-2: the host write is their source of truth; the default is initialization-only) and are listed in `model.report`.
- **Collisions and references**: exactly one owner per name — a duplicate is a `FieldCollisionError` naming both modules; no silent merge. Registration freezes before halo tracing/negotiation (treedef stability). The namespace is flat, prefix-by-convention (D1.5).
- **Killed relative to old `FieldMetadata`** (D1.1): `position`/`bc_types`/`topo`/`is_spectral` (subsumed by `space`), dtype overrides (pattern `scalars`), per-field IO switches (writers default to PROGNOSTIC + DIAGNOSTIC with explicit lists), the `_flags` dict (→ roles), owner backrefs (assembly bookkeeping), serialization helpers (restart persists the carry + fingerprint).
- **Fingerprint**: declaration names, resolved **bare** spaces, and lifecycles enter the restart fingerprint; `Layout`/negotiation/device topology never do (02_rules, refined scope). IC leaves are deliberately invisible.
- The plain user tracer enters as the one-liner declaring module `fr.modules.Tracer("dye", units="1")` (module.md), which wraps `FieldDeclaration.tracer`.

---

### FieldReference

A consumer's checked claim on a field it does not own (D1.5).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final (NamedTuple) |
| Pytree | host object, not a pytree (transient assembly input) |
| Task | 2.2 |
| Design refs | 01 D1.5; 02 (read/write gating) |

```python
"""FieldReference: declared cross-module field dependency."""
from __future__ import annotations

from typing import NamedTuple


class FieldReference(NamedTuple):
    """Checked at assembly; MissingFieldError carries the hint."""

    name: str                                                  # 2.2
    hint: str = ""                                             # 2.2
```

Notes:

- Declared in `Module.field_references`; checked at assembly step 1; failure is a `MissingFieldError` **attributed to the requiring module** with the hint text (the D1.3 error contract: "velocities are declared by a dynamical-core module, e.g. ...").
- **No auto-creation, no space slot** — a reference carrying a space is a declaration in disguise (D1.5). A module that requires a specific BC/space may check the resolved space at `bind(table)`.
- References are the field-level face of D2's requires mechanism — one idiom for module authors; `ParameterReference` is the exact twin (with the one divergence noted there).

---

### ParameterDeclaration

A module publishes a scalar: the declaration names *where the value lives*, never a frozen copy (D2.1).

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen |
| Pytree | host object, not a pytree (transient; the binding table it feeds is static assembly data) |
| Task | 2.2 |
| Design refs | 01 D2.1, D2.2 (R4, static/dynamic discipline); 02 (provides-implies-constancy); 04 §6.2 step 2 |

```python
"""ParameterDeclaration: publish a module scalar by name + attr."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterDeclaration:
    """name -> (owner slot, attr): a live-leaf accessor, not a value."""

    name: str                    # canonical dotted name (ParamName ok) # 2.2
    attr: str = ""               # the owner attribute holding the live
                                 # value (a dynamic leaf); keyword in use # 2.2
    units: str = "n/a"                                         # 2.2
    doc: str = ""                                              # 2.2
```

Notes:

- **`attr`, not `value`** — the value is read from the live carry at use time via the assembly-built binding table `{name: (module_slot, attr)}` and delivered fresh per stage through `eval_params`/`ctx.params` (D2.1's chosen delivery semantics; the aliasing table of rejected alternatives — stored provider objects, frozen values, mirror namespaces — is [d2_1](../../../research/d2_1_resolution_mechanism.md) §2).
- **Scalars (and small static-shape arrays) only** — the boundary is "does it live on the mesh?", not size. Spatially varying parameters are AUXILIARY fields (D2.1 §1.3): *varies in space → field; scalar (possibly time-dependent) → parameter*.
- **Checks at assembly step 2** (model.md executes): one provider per name (`ParameterCollisionError` naming both modules); **provided parameters must be dynamic leaves** (closes the silent-recompile-per-sweep hole); the duplicate-module aliasing lint; the dotted-name lint.
- **Provides implies constancy** (02_rules): a module provides a scalar only when that scalar is the whole truth — `BetaPlaneCoriolis` holds an `f0` leaf but must **not** provide `coriolis.f0` (its Coriolis parameter is the `f(y)` aux field); analytic consumers (`from_model` eigenmodes) rely on the provide's existence as a constancy check. Inside the trace the field is authoritative; provided scalars are the assembly/host-side read surface (R4).
- **The stepper joins the table** as provider of `fr.params.TIME_STEP` (its `dt` leaf) at assembly step 2 — backward runs and dt sweeps use the same `update_parameters` hook.

---

### ParameterReference (and `REQUIRED`)

The exact twin of `FieldReference`, with the one divergence: a physically-identity default (D2.1).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final (NamedTuple); `REQUIRED` module-level sentinel |
| Pytree | host object, not a pytree |
| Task | 2.2 |
| Design refs | 01 D2.1; [d2_1](../../../research/d2_1_resolution_mechanism.md) risk 2 (defaults policy) |

```python
"""ParameterReference: declared cross-module scalar dependency."""
from __future__ import annotations

from typing import Any, Final, NamedTuple

REQUIRED: Final = ...   # sentinel: no default -> MissingParameterError


class ParameterReference(NamedTuple):
    """Checked and frozen at assembly; hint falls back to the
    canonical-name registry's hint."""

    name: str                                                  # 2.2
    hint: str = ""                                             # 2.2
    default: Any = REQUIRED                                    # 2.2
```

Notes:

- Declared in `Module.parameter_references`; checked at assembly with field references; an unsatisfied `REQUIRED` reference is a `MissingParameterError` attributed to the requiring module, with the hint and the list of provided parameters.
- **Defaults policy (normative)**: defaults only for **physically-identity values** — `scaling.rossby` → 1.0, forcing amplitudes → 0 — so dimensional models need no dummy providers. A default on a registry name marked `no_default` (e.g. `stratification.n2`, which would silently un-stratify a run) is an assembly error.
- Consumption is always through `ctx.params[name]` inside the trace (per-stage evaluation, Ramp-correct at RK stage times) or `model.parameters[name]` host-side; consumers never hold provider objects (the D2 aliasing rule). Assembly-time reads at `bind()` raise on time-dependent values unless spelled `at_time(0.0)` — "grid factor at bind, parameter factor in-step".

---

### ParamName and the `fr.params` registry

Typo-proof canonical parameter names — D1.4's Role markers, mirrored for parameters (D2.1 §3).

| Aspect | Value |
|--------|-------|
| Kind | `ParamName` concrete `str` subclass; registry = module-level constants |
| Pytree | static host data (it *is* a str — hashes/compares as its dotted name) |
| Task | 2.2 |
| Design refs | 01 D2.1; 04 §6.2 step 2 (`TIME_STEP`); 08 §10.7 (`SCALING_ROSSBY`) |

```python
"""fr.params: ParamName + the canonical-name registry."""
from __future__ import annotations

from typing import Final


class ParamName(str):
    """A dotted parameter name carrying units, a provider hint, and
    the no_default mark; interchangeable with its plain string."""

    def __new__(                                               # 2.2
        cls,
        name: str,
        *,
        units: str = "n/a",
        hint: str = "",
        no_default: bool = False,
    ) -> ParamName:
        """The dotted-name lint applies (must contain a dot)."""
        ...

    # read-only attributes: units, hint, no_default


TIME_STEP: Final[ParamName] = ...          # "stepper.dt", no_default   # 2.2
CORIOLIS_F0: Final[ParamName] = ...        # "coriolis.f0"              # 2.2
CORIOLIS_BETA: Final[ParamName] = ...      # "coriolis.beta"            # 2.2
STRATIFICATION_N2: Final[ParamName] = ...  # "stratification.n2",
                                           #   no_default               # 2.2
SCALING_ROSSBY: Final[ParamName] = ...     # "scaling.rossby",
                                           #   identity default 1.0     # 2.2
```

Notes:

- **`ParamName` subclasses `str`** (*spec concretization, called out*): both spellings the design uses — `params["coriolis.f0"]` and `update_parameters({fr.params.CORIOLIS_F0: f0})` — must hit the same mapping key, so the constant hashes and compares as its dotted string. Registry hints back host-side `MissingParameterError`s.
- **`TIME_STEP`** is the binding-table entry the stepper provides (its dt leaf); the snapshot manifest records its value, and `load_snapshot` errors on a dt **sign** mismatch and warns on magnitude (the successor of `run_backward`'s sign re-forcing; 02_rules). *The canonical string `"stepper.dt"` is proposed here* — see Open questions.
- Package-specific names (`"nonhydro.dsqr"` owned by `nh.DynamicalCore`, `"shallowwater.csqr"`) live in the package's own registry module (`nh.params.DSQR`, ...) using this same class; user packages prefix by convention. `fr.params` and the Role registry are **immutable module-level data** — no process-global mutable state (the coupling audit).
- Units are documentation, never computed with (D2.2 §5; full unit algebra rejected).

---

### `fr.Param` and `USE_PROVIDED`

Reference-valued constructor slots: how generic modules stay ignorant of parameters they merely scale by (D2 reconciliation 4; D2.1 risk 3).

| Aspect | Value |
|--------|-------|
| Kind | `Param` concrete frozen dataclass; `USE_PROVIDED` module-level singleton sentinel |
| Pytree | host objects, not pytrees (consumed by the binding layer at assembly) |
| Task | 2.2 |
| Design refs | 01 D2.2 §4, D2 reconciliation 4, D2 residuals; 07 §9.2 |

```python
"""fr.Param / USE_PROVIDED: reference-valued constructor slots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

USE_PROVIDED: Final = ...   # sentinel: "resolve through the table"


@dataclass(frozen=True)
class Param:
    """The declaration spelling of a defaulted ParameterReference
    sitting in a constructor slot."""

    name: str                                                  # 2.2
    default: Any = REQUIRED                                    # 2.2
```

Notes:

- **Semantics (explicit-wins)**: a constructor slot whose default is `fr.Param("scaling.rossby", default=1.0)` declares a `ParameterReference(name, default=default)` **only when the caller leaves the slot untouched**; an explicit number or `fr.Ramp` is an owned value — no reference declared, no linkage claimed (Option-A coexistence, bounded). `USE_PROVIDED` is the converse sentinel: passed explicitly to a slot that normally owns its value, it forces resolution through the binding table. The binding layer (model.md) performs the conversion; module code reads the slot uniformly through `resolve_at`/`ctx.params`.
- The flagship consumer: generic advection stays Ro-ignorant via `scaling=fr.Param("scaling.rossby", default=1.0)`; the ramped nonlinear spin-up is `DynamicalCore(rossby_number=fr.Ramp(...))` with zero consumer changes (D2.2 §4; the dedicated `NondimensionalScaling` module is recorded as rejected).
- Ownership rule of thumb (D2.1 §4): *constructor-arg what you own; provide what others consume; require what you don't own; field-reference what varies in space.*

---

### TimeDependent, `fr.Ramp`, and `resolve_at`

The time-dependent scalar value: a frozen callable pytree, never a module (D2.2 §3).

| Aspect | Value |
|--------|-------|
| Kind | `TimeDependent` ABC; `Ramp` concrete, final; `resolve_at` free function |
| Pytree | `jaxify`: curve/shape **static**, `v0`/`v1`/`t0`/`period` **dynamic leaves** (the diffrax split) |
| Task | 2.2 (`reversed()` consumed at 2.7/2.8 — the OB backward leg) |
| Design refs | 01 D2.2 §3, D2.4 (`at_time`); 02 (signed endpoints, `reversed()`, fingerprint: shape = structure, endpoints = leaves); 08 §10.5 (OB Ramps) |

```python
"""Time-dependent scalar values: TimeDependent, Ramp, resolve_at."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


class TimeDependent:
    """A pure, branch-free time curve; describes, never writes."""

    def __call__(self, t: Array) -> Array:                     # 2.2
        """Traced evaluation at clock time t; valid for all t."""
        ...

    def at_time(self, t: float) -> Array:                      # 2.2
        """Explicit host/assembly-side evaluation — the sanctioned
        spelling for assembly reads (bare reads there raise)."""
        ...

    # scalar composition (needed by OptimalBalance); results are
    # derived TimeDependent nodes. Deliberately NO field/array
    # arithmetic: a consumer forgetting resolve_at fails loudly in
    # the assembly dry run.
    def __mul__(self, other: complex) -> TimeDependent: ...    # 2.2
    def __rmul__(self, other: complex) -> TimeDependent: ...   # 2.2
    def __add__(self, other: complex) -> TimeDependent: ...    # 2.2
    def __radd__(self, other: complex) -> TimeDependent: ...   # 2.2
    def __sub__(self, other: complex) -> TimeDependent: ...    # 2.2
    def __rsub__(self, other: complex) -> TimeDependent: ...   # 2.2
    def __neg__(self) -> TimeDependent: ...                    # 2.2


@partial(fr.utils.jaxify, dynamic=("v0", "v1", "t0", "period"))
class Ramp(TimeDependent):
    """v0 + (v1 - v0) * shape(clip((t - t0)/period, 0, 1))."""

    def __init__(                                              # 2.2
        self,
        v0: float,
        v1: float,
        *,
        period: float,
        t0: float = 0.0,
        curve: str | Callable[[Array], Array] = "linear",
    ) -> None:
        """curve is STATIC: a named curve ("linear" | "cosine" |
        "exp") or a callable with shape(0)=0, shape(1)=1; endpoints
        and timing are dynamic leaves. t0/period are SIGNED times."""
        ...

    def __call__(self, t: Array) -> Array:                     # 2.2
        """Branch-free (jnp.clip + shape); valid at every scan step."""
        ...

    def reversed(self) -> Ramp:                                # 2.8
        """Reflect the active window across t = 0 (02_rules V-S2):
        same values/curve, t0 -> -(t0 + period) — a backward leg
        (dt < 0 from clock 0) retraces this ramp's values exactly;
        the naive endpoint swap over [0, T] clips to a constant."""
        ...

    def __repr__(self) -> str: ...                             # 2.2


def resolve_at(value: object, t: Array) -> Array:              # 2.2
    """value(t) if isinstance(value, TimeDependent) else value —
    identity on plain scalars, so every scalar slot is Ramp-able
    with zero consumer changes (the universal idiom)."""
    ...
```

Notes:

- **The static/dynamic split is the point**: changing the curve is different math — one recompile, correct; sweeping ramp endpoints/timing never recompiles. A scalar↔Ramp swap in a slot changes the treedef → one recompile plus a restart-fingerprint change (documented; the fingerprint treats the Ramp *spec/shape* as structure, endpoints as leaves — 02_rules).
- **Signed times (V-S2)**: backward legs run over negative clock times, so a backward Ramp spans `[−T, 0]` (`t0 = −T`). `reversed()` is the blessed endpoint-reversal spelling; *the concrete construction here* — a pure window reflection `t0 → −(t0 + period)` with values and curve unchanged — *is a spec concretization of the 02_rules sentence*; it satisfies the retrace law `r.reversed()(−s) == r(t0 + period − s + t0)` (backward progress s sees the forward value at remaining time) for **every** curve, symmetric or not. Confirm against the ported OB legs (Open questions).
- **The boundary with `self_update`** (D2.2, normative): *a Ramp describes a curve and never writes.* Scalars get time dependence via `resolve_at` at the point of use; field-consumed parameters get it via the owner's `self_update` rewriting the AUXILIARY field. Assembly schedules a module's self-update **only if** an input is `TimeDependent` (or its declaration names state `reads=` — V-H5, module.md) — static models pay nothing.
- **Read rules**: inside the trace, evaluation happens at **stage time** (`eval_params` per substage; a ramped scalar and a ramped N² profile see the same time). Host-side, `model.parameters` returns the **Ramp object** — `at_time(t)` evaluates explicitly; eigenmode `from_model` errors on Ramp-valued parameters unless `at_time=` is passed (D2.4). Assembly-time (`bind`) reads of a `TimeDependent` raise unless spelled `at_time(0.0)` — this kills the `BiharmonicClosure` stale-coefficient bug class.
- The old `Ramper` module (per-step mutation of *other* modules' parameters) is dissolved by this class plus provided-value evaluation; its untraceable `if time < start` is subsumed by `clip` + the shape contract. Rejected alternatives (value-based static filtering, mutating time-dependence): [d2_2](../../../research/d2_2_representation.md) §2–3.
- `reset()` resets the clock, **which is what restarts Ramp legs** (D4); `variant(updates=)` may change value specs (scalar → Ramp, `TIME_STEP` sign) because it is an assembly (08 §10.4).

---

### Treatment, TendencyTerm, and the `@fr.term` decorator

A module's tendency contribution as declared frozen data — the FieldDeclaration pattern applied to behavior (D3, 03 §5.1).

| Aspect | Value |
|--------|-------|
| Kind | `Treatment` closed enum; `TendencyTerm` concrete frozen dataclass; `term` decorator function |
| Pytree | host objects, not pytrees — transient assembly inputs (the composed step closes over slot indices + unbound fns) |
| Task | 2.5 |
| Design refs | 03 §5.1, §5.5 (hook signature), §5.4 (buffer partition); 01 D3; 02 (fingerprint incl. per-term treatments); 08 §10.4 (predicate inputs) |

```python
"""Tendency terms: Treatment, TendencyTerm, the @fr.term decorator."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final


class Treatment(Enum):
    """Integration treatment of a term (closed, it-1)."""

    EXPLICIT = auto()                                          # 2.5
    IMPLICIT = auto()                                          # 2.5


EXPLICIT: Final[Treatment] = Treatment.EXPLICIT   # fr.EXPLICIT # 2.5
IMPLICIT: Final[Treatment] = Treatment.IMPLICIT   # fr.IMPLICIT # 2.5


@dataclass(frozen=True)
class TendencyTerm:
    """One declared tendency contribution; transient assembly input."""

    name: str                                                  # 2.5
    fn: Callable | None = None      # UNBOUND (module, state, ctx)
                                    # -> {prognostic: increment};
                                    # None iff implicit set        # 2.5
    treatment: Treatment = Treatment.EXPLICIT                  # 2.5
    advances: tuple[str, ...] | None = None                    # 2.5
    transports: tuple[str, ...] = ()                           # 2.5
    implicit: ImplicitOperator | None = None                   # 2.5
    linear: bool = False                                       # 2.5

    def __repr__(self) -> str: ...                             # 2.5


def term(                                                      # 2.5
    fn: Callable | None = None,
    *,
    name: str | None = None,
    treatment: Treatment = Treatment.EXPLICIT,
    advances: tuple[str, ...] | None = None,
    transports: tuple[str, ...] = (),
    implicit: ImplicitOperator | None = None,
    linear: bool = False,
) -> Callable:
    """@fr.term — stamp a module method as a TendencyTerm (bare and
    parenthesized forms); name defaults to the method name; the
    method stays plainly callable (halo trace runs it un-jitted)."""
    ...
```

Notes:

- **`fn` is stored unbound** and paired with a module *slot* at compose time; the composer calls `term.fn(carry.modules[slot], state, ctx)` inside the trace (the D2 aliasing rule applied to behavior; bound-method capture is the rejected alternative — [d3_1](../../../research/d3_1_term_surface.md) §1). The attribution key is `"Module/term"`, composed by the collector from the owning module's name and `term.name` — the key `fr.terms.named(...)` and `TermEvaluationError` use.
- **Collection**: `Module.tendency_terms()` (module.md) defaults to scanning `@fr.term`-stamped methods in definition order; modules with constructed terms override it. Collection runs **after `bind(table)`**, so `advances` may come from role selections.
- **Hook signature** (seam anchor): `(self, state, ctx) -> dict` — contribution dicts key **PROGNOSTIC components only** (write gate), receive the full state vector (all lifecycles readable), and are applied via `VectorField.add` in deterministic order (module order, declaration order — float summation is not associativity-stable). **Terms only ever add**; anything that overwrites is a stage (module.md).
- **Treatment is author-declared; the user override lives on the module constructor** (`VerticalMixing(kv=..., treatment=fr.IMPLICIT)` — the Oceananigans `time_discretization=` precedent; Model-level override dicts rejected). `IMPLICIT` without `implicit=` is an assembly error; an implicit term under a purely explicit stepper is an **assembly error, never silent demotion**.
- **Write-once**: a term with `implicit=op` may omit `fn` — the explicit path is derived from `op.apply`, so flipping treatment cannot desynchronize the two (Dedalus's declared linear part, mechanically).
- **`advances`** is optional-declared, always dry-run-verified (contribution keys are static under jit, so derivation is fully reliable; a mismatch is an assembly error catching wrong-but-valid-component bugs). **`transports`** is declared-only intent feeding the D1.4 coverage lint (not derivable: diffusion and advection both write `b`; only one transports it).
- **`linear` is strict**: linear in the state at fixed params/aux; state-independent forcing is *not* linear. Consumers: `fr.terms.linear`, `fr.linearize`, the IMEX partition sanity. The JVP debug lint (evaluate at s and 2s) is prioritized in 2.5 now that `linearize` consumes the tag (08 §10.8).
- All attribution, validation, and accumulation-order machinery lives in the **TendencyComposer** (model.md); per-term treatments enter the restart fingerprint (02_rules).

---

### ImplicitOperator (protocol)

Two capabilities, nothing more — the minimal surface every IMEX family needs (03 §5.1).

| Aspect | Value |
|--------|-------|
| Kind | structural `Protocol` (runtime-checkable) |
| Pytree | implementations are static assembly data riding on terms; live coefficients are read through unbound callables at trace time |
| Task | 2.5 |
| Design refs | 03 §5.1 (surface, merge rules), §5.4 (family table, atomic blocks); 02 (lagged coefficients) |

```python
"""ImplicitOperator: the two-capability implicit-term surface."""
from __future__ import annotations

from typing import Hashable, Protocol, runtime_checkable

import fridom.framework2 as fr


@runtime_checkable
class ImplicitOperator(Protocol):
    """apply = forward L·state; solve = (1 - dt_gamma·L)^{-1}."""

    fields: tuple[str, ...]   # advanced PROGNOSTIC subset, MANDATORY  # 2.5

    def apply(                                                 # 2.5
        self, module, state, ctx,
    ) -> dict[str, fr.ScalarField]:
        """L·state — the forward evaluation (CNAB rhs; the derived
        explicit path when the term omits fn)."""
        ...

    def solve(                                                 # 2.5
        self, module, rhs: dict[str, fr.ScalarField],
        dt_gamma, ctx,
    ) -> dict[str, fr.ScalarField]:
        """(1 - dt_gamma·L)^{-1} rhs; keys exactly `fields`.
        γ-agnostic: the scheme owns γ; dt_gamma is a TRACED scalar
        positional (warm-up γ switching / adaptive dt never retrace)."""
        ...

    # --- merge mechanics (spec-proposed; implements the signed
    # --- family-merge rules, see the deviation note below) --------
    def merge_key(self) -> Hashable | None:                    # 2.5
        """Grouping key for exact merging; None = non-mergeable."""
        ...

    def merged_with(self, other: ImplicitOperator) -> ImplicitOperator:
        """Exact combination within one merge_key group."""
        ...                                                    # 2.5
```

Notes:

- **The forward apply is mandatory** — the CN solve-only trick is unsound here (recovering `L·Xⁿ` from the previous solve reads the pre-projection state: an O(dt) error every step; 03 §5.1).
- **Coupled blocks are atomic**: `fields=("u", "v")` (semi-implicit Coriolis) is one operator, indivisible under by-variable splitting (03 §5.4).
- **Lagged coefficients** (02_rules): state-dependent implicit coefficients (CATKE-like) are evaluated on the state passed into the stage (predictor/lagged values); the solve itself stays linear.
- **Merge rules (signed, 03 §5.1)**: mergeable framework families combine *exactly* (κ-summing is `(1 − dtγ(L₁+L₂))` — the combined operator; sequential opaque solves would be Lie splitting inside an IMEX stage, order-degrading); **at most one non-mergeable custom implicit operator per field** (`ImplicitCollisionError` naming both terms). *Deviation note*: the `merge_key()`/`merged_with()` protocol hooks are **spec-proposed mechanics** — the design fixes the merge *rules* but no discovery mechanism; the alternative (composer special-casing framework types) hard-codes the family list. Recorded in `../07_open_threads.md` if 2.5 implementation prefers otherwise.
- **Halo/shard story**: a family's solve is a grid-bound registry **Operator** declaring `layout local along the solve axis` — negotiated like transforms, intercepted generically by `HaloTracer`, no `.data` bypasses (retires the `RFFTPressureSolver` hand-sharding pattern).
- **Empty implicit partition is graceful** (08 §10.4): a variant filtering all implicit terms leaves an empty operator set; the solve loop is not emitted at trace time and CNAB2 degenerates to textbook AB2 (same order, not bitwise — documented, no eps injection).

---

### `fr.implicit.VerticalDiffusion`

The framework-owned mergeable tridiagonal family (03 §5.1).

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen; implements `ImplicitOperator` |
| Pytree | static assembly data; κ read live through the unbound `kappa` at trace time |
| Task | 2.5 (ships with the reference vertical-diffusion consumer, exact 1D decay + stiff-κ column tests) |
| Design refs | 03 §5.1, §5.7; 01 D3 |

```python
"""fr.implicit.VerticalDiffusion: the mergeable tridiagonal family."""
from __future__ import annotations

from dataclasses import dataclass

import fridom.framework2 as fr


@dataclass(frozen=True)
class VerticalDiffusion:
    """Diffusion along one axis, solved as ONE tridiagonal per field."""

    axis: str                                                  # 2.5
    fields: tuple[str, ...]                                    # 2.5
    kappa: Callable   # UNBOUND (module, state, ctx, field_name)
                      # -> scalar | ScalarField (coefficients,
                      # not a solver)                          # 2.5

    def apply(self, module, state, ctx) -> dict[str, fr.ScalarField]:
        """L·state via the dispatched second-derivative operator."""
        ...                                                    # 2.5

    def solve(                                                 # 2.5
        self, module, rhs: dict[str, fr.ScalarField],
        dt_gamma, ctx,
    ) -> dict[str, fr.ScalarField]:
        """One Thomas solve per field; boundary rows from the
        field's declared space BCs; flux BCs are explicit forcing."""
        ...

    def merge_key(self) -> Hashable:                           # 2.5
        """Same axis + same family => mergeable (per shared field)."""
        ...

    def merged_with(self, other: VerticalDiffusion) -> VerticalDiffusion:
        """Sum κ contributions into one solve (exact)."""
        ...                                                    # 2.5
```

Notes:

- **κ-summing is the composition answer** (Oceananigans' coefficient-merging precedent): two closures contributing implicit vertical diffusion on one field become **one** tridiagonal solve per field — the composer groups per field across same-axis instances. `kappa` follows the unbound-behavior rule; coefficients read `ctx.params`/owner leaves at stage time (Ramp-correct).
- No assembly-time factorization caching in it-1 (`dt_gamma` and κ are traced); noted as a constant-coefficient optimization.

---

### `fr.implicit.SpectralDiagonal`

The coefficient-space diagonal family — merges by summing eigenvalues (03 §5.1).

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen; implements `ImplicitOperator` |
| Pytree | static assembly data; Symbol *structure* bound at assembly, live parameter factors read in-step (the D2.4 pressure-solver pattern) |
| Task | designed-for (protocol slot frozen at 2.5; build deferred — earliest consumer the sw semi-implicit gravity-wave pair, 2.7) |
| Design refs | 03 §5.1; 01 D2.4 (bind-structure/read-leaf rule) |

```python
"""fr.implicit.SpectralDiagonal: diagonal solves in coefficient space."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralDiagonal:
    """(1 - dt_gamma·λ)^{-1} as a per-mode division."""

    fields: tuple[str, ...]                                    # later
    symbol: Symbol | Callable   # a Symbol, or an unbound factory
                                # (module, grid, space) -> Symbol,
                                # bound at assembly             # later

    def apply(self, module, state, ctx) -> dict: ...           # later

    def solve(self, module, rhs, dt_gamma, ctx) -> dict: ...   # later

    def merge_key(self) -> Hashable:                           # later
        """Same coefficient space => mergeable."""
        ...

    def merged_with(self, other: SpectralDiagonal) -> SpectralDiagonal:
        """Sum eigenvalues (exact)."""
        ...                                                    # later
```

Notes:

- The eigenvalue structure binds at assembly; parameter factors (a κ, a `dsqr`) multiply in-step from live leaves — baking values would go stale under sweeps (the V-N amendment to D2.4, stated for the pressure solver, applies verbatim). Kept deliberately brief; nothing in iteration 1 depends on it.

---

### `fr.terms` — the term-predicate family

Composable keep-predicates over collected `TendencyTerm`s — the `model.variant` filter vocabulary (D5, 08 §10.4).

| Aspect | Value |
|--------|-------|
| Kind | `TermPredicate` abstract base; five leaves (three module-level instances, three factories); internal node classes private |
| Pytree | host objects, not pytrees — frozen, hashable, reprable expression trees; **never traced** |
| Task | 2.8 |
| Design refs | 08 §10.4 (vocabulary, fingerprint tokens, rejections), §10.8; 01 D5 |

```python
"""fr.terms: the term-predicate vocabulary (linear, owned_by, ...)."""
from __future__ import annotations

from typing import Final


class TermPredicate:
    """Frozen keep-predicate expression tree; & | ~ combinable."""

    def matches(self, term: TendencyTerm, owner: Module) -> bool:
        """Host-side match against a collected term + its owner."""
        ...                                                    # 2.8

    def fingerprint_token(self) -> str:                        # 2.8
        """Stable canonical token (enters the assembly fingerprint;
        owned_by uses qualified class names)."""
        ...

    def __and__(self, other: TermPredicate) -> TermPredicate: ...  # 2.8
    def __or__(self, other: TermPredicate) -> TermPredicate: ...   # 2.8
    def __invert__(self) -> TermPredicate: ...                     # 2.8

    def __repr__(self) -> str:                                 # 2.8
        """Canonical composition-tree repr; no simplification."""
        ...


linear: Final[TermPredicate] = ...      # the declared linear tag  # 2.8
explicit: Final[TermPredicate] = ...    # treatment == EXPLICIT    # 2.8
implicit: Final[TermPredicate] = ...    # treatment == IMPLICIT    # 2.8


def owned_by(module_type: type) -> TermPredicate:              # 2.8
    """isinstance(owner, module_type) — the family axis, e.g.
    ~fr.terms.owned_by(fr.closures.ClosureBase)."""
    ...


def named(*keys: str) -> TermPredicate:                        # 2.8
    """Exact "Module/term" attribution keys; unknown keys ERROR at
    the consuming build (kills the silent-typo footgun)."""
    ...


def advancing(*fields: str) -> TermPredicate:                  # 2.8
    """Nonempty intersection with the term's advances — splits one
    module's terms (Smagorinsky stress vs κ mixing)."""
    ...
```

Notes:

- **Five leaves, three combinators, nothing else** — the identity filter is `term_filter=None`. Rejected ([d5_2](../../../research/d5_2_variants.md) §1): `transporting(...)` (suggests per-field masking that term-granular filtering cannot deliver — "freeze one tracer" is a term rewrite, documented); bare lambdas (unfingerprintable — a tokened `where(fn, token=)` escape is designed-for); a parallel `Module.category` tag axis. `fr.closures.ClosureBase` — the isinstance anchor — is introduced in module.md, earning its keep by hosting the role-target boilerplate.
- **Validation is at the consuming build**, not construction: `model.variant` errors on unknown `named` keys and on an empty filter result; the coverage lint downgrades error → info under any filter; the filter's canonical token + updates specs enter the assembly fingerprint (all model.md's execution; 08 §10.4). Qualname tokens under interactive redefinition are an accepted limitation (02_rules note owed per d5_2).
- Canonical spellings the class design must keep expressible: `fr.linearize(model)` ≡ `model.variant(term_filter=fr.terms.linear)`; inviscid-linear ≡ `fr.terms.linear & ~fr.terms.owned_by(fr.closures.ClosureBase)`; the OB backward filter `~fr.terms.owned_by(fr.closures.ClosureBase) & ~fr.terms.implicit`; per-term budgets via `model.tendency(state, filter=fr.terms.named("CenteredAdvection/momentum"))`.
- Predicates never run in the traced step — `matches` executes during (re-)assembly only.

---

## Open questions

Genuine residuals only (from `../07_open_threads.md` and `../08_state_transforms.md` §10.8), plus the spec concretizations made above that need a confirming consumer:

1. **Unstructured-factor `SpacePattern` tags** (`EDGE_NORMAL`; a multi-name mesh factor breaks the name-keyed premise): designed-for; `SpaceRule` covers it meanwhile (07 §9.1).
2. **`require=` fine semantics**: the kwarg itself is adopted (04 §6.2 step 1); remaining: does it assert *name matched a factor* only, or *tag actually consumed*; the error type and message format — settle with the 2.2 error-reporting pass, together with the resolution-table logging.
3. **CS-17 precision plumbing** — RESOLVED (2026-07-08, Silvano): **global precision only**. `SpacePattern.scalars` stays Körper-only (`REAL`/`COMPLEX`, no width); a run is uniformly float64 (default) or float32 (x64 disabled); there is no per-space width axis. Rationale: the landed grid derives storage dtype from the global `jax_enable_x64` flag (its `Scalars` docstring explicitly disclaims storage semantics), and JAX forbids float64 arrays under x64-off — so "float64 sums under float32 fields" would force x64 ON with a default width of 32, inverting fridom's documented precision idiom ("float32 run" spelled as x64-enabled): a real grid-contract change bought for one use case. The CS-17 concern itself (float32 accumulator roundoff over long windows) is covered at the model layer with no grid change: the S6 accumulation idiom is chunk-cadence — in-trace sums span at most one chunk (~256 steps), and the chunk-boundary host read accumulates in float64 on the host (no `fr.modules.WindowAccumulator` preset is promised — 07_open_threads §9.1). A width axis remains a possible future grid extension (the space-interning `_variant` machinery would take an extra key slot gracefully) but is NOT designed for.
4. **`default=` callable disambiguation**: the first-parameter-named-`self` convention proposed here is a spec concretization — confirm at 2.2 (alternative: a tiny explicit wrapper, per d1_1 risk 4's `fr.InitCoeff` pattern).
5. ~~**`TIME_STEP` canonical string**~~ — **closed**: `fr.params.TIME_STEP` is `"stepper.dt"`, provided by the stepper's dt leaf.
6. **`USE_PROVIDED` corner**: the sentinel shipped; the corner remains — `update_parameters` targeting a name whose reference was suppressed by an explicit constructor value: error or no-op?
7. **`Ramp.reversed()` construction**: the mirrored-window translation given here shipped (`Ramp.reversed()`); confirming it against the ported OB legs in the tolerance-based cutover tests is still owed.
8. **`ImplicitOperator` merge hooks**: `merge_key()`/`merged_with()` are spec-proposed mechanics for the signed merge rules — revisit at 2.5 if the composer prefers family-internal grouping.
9. **Predicate designed-fors** (parked, 08 §10.8): `fr.terms.where(fn, token=)`, glob support in `named`, stage filtering.
10. **Multi-velocity `group=` qualifier** (07 §9.1): `table.velocity()` ambiguity under future multi-velocity configurations — designed-for, don't build (the split-explicit case is resolved role-free, 03 §5.4).
