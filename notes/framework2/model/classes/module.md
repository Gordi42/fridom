# Model layer redesign — Class designs: module

Part of the model-layer class designs; see [`README.md`](README.md)
for the document map, the shared template, and the seam anchors.
This document owns the **Module cluster**: the `Module` base class —
the D1–D5 capability menu turned into a class surface —,
`fr.closures.ClosureBase`, the `Stage`/`StageKind` schedule
vocabulary, and the `StepContext` object every in-trace hook
receives. Sibling docs own the seams this cluster touches:

| Doc | Owns |
|-----|------|
| [`declarations.md`](declarations.md) | `FieldDeclaration`/`FieldReference`, `SpacePattern`, `Lifecycle`, `Role`/`fr.roles`, `ParameterDeclaration`/`ParameterReference`/`fr.Param`, `fr.Ramp`/`resolve_at`, `TendencyTerm`/`@fr.term`, `ImplicitOperator`/`fr.implicit`, `fr.terms` predicates |
| [`model.md`](model.md) | `fr.Model` (assembly consumes everything declared here), `FieldTable`/`VelocitySelector`, `TendencyComposer` (all error attribution), the binding/re-materialization tables, `State` vocabulary classes |
| [`time_steppers.md`](time_steppers.md) | `TimeStepper`, `StepperState`, `Clock` (referenced by `StepContext`) |
| [`transforms.md`](transforms.md) | `model.variant` mechanics and the term-predicate consumers of `ClosureBase` |

Where this cluster touches those seams it uses only the fixed
anchors: every in-trace hook is `(self, state, ctx) -> dict` applied
per the kind's write gate; `fn`/`default=` references to module
behavior are **unbound**; the state vector is `state`, never `z`;
no coupling/run-loop state lives outside the carry.

---

## Module layout

The code lives in `fridom.framework2.model` (the model layer of the
parallel `fridom.framework2` package). Proposed internal layout for
this cluster:

```
src/fridom/framework2/model/
    module.py         # Module (base)
    stages.py         # Stage, StageKind, fr.self_update
    context.py        # StepContext
    closures/
        base.py       # ClosureBase (re-exported as fr.closures.ClosureBase)
```

`declarations.py`, `field_table.py`, `model.py`, `composer.py`,
`time_steppers/`, `transforms/`, and `io/` belong to the sibling
docs; model.md holds the canonical whole-subpackage tree. Import
direction: `module.py` imports the declaration types *from*
`framework2.model.declarations`; nothing here imports from
`model.py` (assembly consumes modules, never the reverse).

Top-level re-exports (lazypimp, per repo convention): `fr.Module`,
`fr.Stage`, `fr.StageKind`, `fr.StepContext`, `fr.self_update`;
`fr.closures` = `fridom.framework2.model.closures` (the namespace
that will also host the ported concrete closures at 2.7 —
`HarmonicMixing`, `VerticalMixing`, ... — which are out of this
document's scope; only `ClosureBase` is specified here).

---

## Cluster-wide rules

These apply to every class below and are not repeated per class:

- **The uniform hook signature.** Every in-trace hook — term `fn`,
  `self_update`, every stage body — is
  `(self, state, ctx) -> dict` with `state` the **full assembled
  state vector** (all lifecycles readable) and `ctx` a
  `StepContext`. The returned dict is applied per the owning kind's
  write gate (terms: `add`; everything else: `replace`). One
  validation path, one halo-trace path; **no hook ever mutates or
  returns `self`** — step-evolving module data that isn't a
  parameter leaf is a declared AUXILIARY/DIAGNOSTIC field
  (03 §5.5).
- **Read access is uniform; write access is lifecycle-gated**
  (D1.5, 02_rules): contribution dicts key PROGNOSTIC components
  only; AUXILIARY components are written only by their owning
  module; a read of any component sees the **nearest preceding
  write in schedule order, crossing substage and step boundaries**
  (03 §5.2 — Gauss-Seidel and DIAGNOSTIC warm-start semantics both
  fall out of this one rule).
- **The unbound-reference rule (jax aliasing).** Behavior
  references stored in declarations — `TendencyTerm.fn`,
  `Stage.fn`, `FieldDeclaration.default` — are **unbound** methods
  paired with a module *slot* at compose time; the composer calls
  `fn(carry.modules[slot], state, ctx)`. A bound method would
  capture the assembly-time instance while live parameters ride the
  carry (the D2 trap; 03 §5.1).
- **Modules ride the carry.** Subclasses are jaxify-registered
  pytrees (`@partial(fr.utils.jaxify, dynamic=(...))`): structure
  static, numeric attributes dynamic leaves. The hardened-jaxify
  discipline (D2.2) is normative: dynamic leaves are coerced
  through `jnp.asarray` (structural values fail loud at
  construction); static attributes must be hashable non-arrays;
  **provided parameters must name dynamic leaves**
  (assembly-checked — closes the silent-recompile hole for exactly
  the parameters that get swept). An assembly lint errors on any
  jaxified module instance appearing twice in the carry (pytrees
  are trees, not DAGs).
- **Amended (Phase-2 reconciliation, 2026-07-08) — landed-jaxify
  mechanics, normative for module authors.** In the landed jaxify
  the statics are the module's whole `__dict__` minus the declared
  dynamic leaves, captured in an aux whose *equality* is structural
  (`_values_equal`, deep comparison) but whose *hash* is the
  attribute **name set only** — so the real static discipline is
  *cheap structural comparability*; "hashable non-array" is enforced
  style, and the failure mode of a heavy or mutable static is
  **silent recompiles** (dispatch-time deep comparison), not an
  error. Consequences: host-side observers on a module (counters,
  writer handles) must be listed in `_eq_ignored_attrs` or they
  break structural equality and recompile every chunk;
  identity-hashed statics (grid, spaces) must be *interned* objects
  — jit-cache sharing across re-assemblies depends on `is`;
  `tree_unflatten` bypasses `__init__` (`object.__new__`), so no
  constructor-established invariant may be assumed in-trace; flatten
  order is the declaration-ordered `dynamic_jax_attrs` tuple (wave-0
  fix) — **never reorder declared leaves in a released module**: it
  changes the treedef and breaks snapshot compatibility. Named
  `ScalarField`-valued dynamic leaves are **forbidden** until the
  annotation-exempt metadata amendment
  ([`../../classes/fields.md`](../../classes/fields.md), 2026-07-08)
  is implemented — raw arrays via `jnp.asarray` remain the rule.
- **Purity.** Traced hooks are pure field algebra — no Python-state
  mutation, no branching on traced values, no raw `.data` escapes
  (the sanctioned escape for clamps/branches is a custom pointwise
  Operator, halo 0, registered like any stencil kernel; 02_rules
  `extra_halo` entry). `bind(table)` is the **one sanctioned
  mutation site**, host-side, once, before the registration freeze.
- **Attribution keys** are `"Module/term"` / `"Module/stage"` with
  the module part the owning class name (d3_1, d5_2:
  `fr.terms.named("CenteredAdvection/momentum")`); all attribution
  machinery — dry-run validation, `TermEvaluationError` wrapping,
  deterministic accumulation order — lives in model.md's
  `TendencyComposer`, never on `Module`.

---

### Module

Unit of physics/numerics: **a module contributes any subset of a
capability menu** — field declarations and references (D1),
provided/consumed parameters (D2), dispatch overrides (D4), tendency
terms and stages (D3), and a per-substage self-update of its own
dynamic state. "Computes a tendency" is one capability, not the
definition of a module (01 §3).

| Aspect | Value |
|--------|-------|
| Kind | concrete base, subclassable; no abstract methods (every capability optional) |
| Pytree | jaxified **per subclass** (`@partial(fr.utils.jaxify, dynamic=(...))`); rides the carry as `carry.modules`; the base class declares no leaves |
| Task | 2.2 (declarations/references/bind), 2.3 (dispatch, `extra_halo`, `state_type`, carry membership), 2.5 (terms, stages, `self_update`) |
| Design refs | 01 §3, D1.1–D1.5, D2.1–D2.2; 02_rules (gating, defaults, `extra_halo`, S6 idiom, SELF_UPDATE-first); 03 §5.1–5.2, §5.5; 04 §6.1–6.2; 08 §10.4; CS-1, CS-13 |

```python
"""The module base: fields + parameters + dispatch + terms + stages
+ self-update — any subset of the capability menu."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


class Module:
    """A unit of physics/numerics contributing capabilities to an
    assembled fr.Model."""

    # The base defines no constructor and no dynamic leaves;
    # subclasses own their __init__ and register their own pytree
    # structure:  @partial(fr.utils.jaxify, dynamic=("kh", "kv")).
    # Convention: user-facing treatment overrides live on subclass
    # constructors (VerticalMixing(kv=..., treatment=fr.IMPLICIT)),
    # never on the Model.

    # ================================================================
    #  Fields (D1) — consumed at assembly steps 1 and 4
    # ================================================================

    field_declarations: tuple[FieldDeclaration, ...] = ()      # 2.2
    """What the module contributes to the state vector (D1.1).
    Transient assembly data — except AUXILIARY ``default=``
    closures, retained in the re-materialization table. May be an
    instance property built from constructor arguments
    (sketch 7.2)."""

    field_references: tuple[FieldReference, ...] = ()          # 2.2
    """Components consumed but not owned: ``FieldReference(name,
    hint)``, checked at assembly (MissingFieldError with the hint).
    No auto-creation — a reference carrying a space is a
    declaration in disguise (D1.5)."""

    # ================================================================
    #  Parameters (D2) — consumed at assembly step 2
    # ================================================================

    parameter_declarations: tuple[ParameterDeclaration, ...] = ()  # 2.2
    """Published scalars: ``ParameterDeclaration(name, attr=...)``
    names where the value lives (a dynamic leaf of this module),
    never a frozen copy. Provides implies constancy (02_rules)."""

    parameter_references: tuple[ParameterReference, ...] = ()  # 2.2
    """Consumed scalars — the exact twin of field_references.
    ``fr.Param(name, default=...)``-valued constructor slots are
    the defaulted spelling and are collected into this set by
    assembly (D2 reconciliation 4)."""

    # ================================================================
    #  Dispatch and negotiation inputs (D4) — steps 3 and 5
    # ================================================================

    dispatch: Mapping[str | tuple[str, SpacePattern],
                      Operator | Callable[..., Operator]] = {}  # 2.3
    """Constructor-frozen registry overrides, keyed ``kind`` or
    ``(kind, SpacePattern)``; model-resolved and merged into the
    grid registry exactly once, at assembly step 3. Values may be
    lazy factories (the transform-row mechanism)."""

    extra_halo: HaloSpec | None = None                         # 2.3
    """Declared halo substitute for this module's terms/stages that
    the halo trace cannot follow (V-N2 mechanics below)."""

    state_type: type[State] | None = None                      # 2.3
    """The State vocabulary class supplied by the dynamical-core
    module (D1.3 commitment 4); >1 provider across the module list
    is an assembly error; ``fr.Model(state_type=...)`` is the
    fallback/override."""

    # ================================================================
    #  Assembly hook (host-side, runs once — step 4)
    # ================================================================

    def bind(self, table: FieldTable) -> None:                 # 2.2
        """Freeze role selections to static name tuples
        (``table.select(ADVECTED)``, ``table.velocity()``), declare
        name couplings (``table.require("u", "v", "p")``), and do
        grid-factor precomputes. Default: no-op."""
        ...

    # ================================================================
    #  Terms and stages (D3) — collected post-bind, step 5
    # ================================================================

    def tendency_terms(self) -> tuple[TendencyTerm, ...]:      # 2.5
        """Terms contributed by this module. Default implementation:
        scan for ``@fr.term``-stamped methods in definition order
        (trivial modules: zero ceremony); override for constructed
        cases. Runs after ``bind(table)``, so ``advances`` may come
        from role selections."""
        ...

    stages: tuple[Stage, ...] = ()                             # 2.5
    """Owned stages. Schedule position is a pure function of each
    stage's declared kind — never of module list position (D1.3
    commitment 5). May be an instance property."""

    # ================================================================
    #  The self-update hook (traced; S1 of every substage)
    # ================================================================

    # -- not defined on the base: defining it opts the module in --
    def self_update(                                           # 2.5
        self, state: State, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """Recompute this module's own AUXILIARY fields at substage
        time; applied via ``replace``. Scheduled iff one of the
        module's inputs is time-dependent or its declaration names
        state inputs (``reads=``). Runs per SUBSTAGE — never use it
        for step-frequency accumulation (hazard note below);
        ``cadence=`` is reserved, not built (CS-1)."""
        ...
```

Semantics, invariants, error behavior:

- **The capability menu, as assembly consumes it** (04 §6.2 — the
  full inventory of what `fr.Model.__init__` reads off a module):

  | Member | Assembly step | Consumed for |
  |--------|--------------|--------------|
  | `field_declarations`, `field_references` | 1 | FieldTable, collision/reference checks, `state_spaces` for negotiate |
  | `parameter_declarations`, `parameter_references` | 2 | the binding table `{name: (slot, attr)}`; one-provider/no-default/dynamic checks |
  | `dispatch` | 3 | `grid.merge_overrides`, exactly once |
  | `bind(table)` | 4 | role selections frozen, precomputes |
  | `tendency_terms()`, `stages`, `self_update`, `extra_halo` | 5 | the kind-ordered schedule, the composed step, negotiation halo |
  | `state_type` | — | the State vocabulary class (6.1) |

  Declarations are transient assembly data (not pytrees, discarded)
  — with the one D4 amendment: AUXILIARY `default=` closures are
  retained in the static re-materialization table, because
  `update_parameters` re-runs them.
- **Defaults: one path, owner leaves only** (02_rules).
  `FieldDeclaration.default` may be an **unbound method of the
  owning module**, called with the live module; it is evaluated
  with the owner's *current* leaves through the same code path at
  allocation and at `update_parameters` (that identity is what
  makes re-materialization correct by construction). A default
  closure may read **only the owner's own leaves** —
  cross-module-derived AUXILIARY values are `self_update` territory
  or a re-assembly. Host-writable components are exempt from
  re-materialization (CS-2; the flag lives on the declaration,
  declarations.md).
- **Per-step writes are confined to the module's own state**
  (01 §3): the old Ramper-style callback that mutated *other*
  modules' parameters is replaced by (i) provided values that are
  pure functions of the traced clock (`fr.Ramp`, evaluated by the
  owning consumer inside the trace via `resolve_at`) and (ii)
  owner-updated AUXILIARY fields that consumers read from the state
  vector. Cross-module reads never touch another module object —
  they go through `ctx.params` (the binding table) or `state[...]`.
- **`bind` semantics** (assembly step 4, module order): the merged
  dispatch registry is visible (bind runs after step 3, so
  precomputed operators are the ones that will actually run);
  role selections resolve **once** into static name tuples closed
  over by the traced hooks (no role logic inside the step —
  invisible to `trace_halo`); frozen selections are stored as
  static attributes and must obey the jaxify hashability
  discipline. **Amended (2026-07-08): `bind` precomputes operators,
  spaces, and name tuples ONLY — it must not materialize real
  fields.** Fields exist only from assembly step 8, after the final
  negotiation; a field created at bind would be stranded by the
  step-7 renegotiation, whose storage shapes depend on the
  negotiated halo. The bind/in-step split rule (D2.1): **grid factor
  at bind, parameter factor in-step** — assembly-time reads of
  time-dependent parameter values raise unless spelled
  `at_time(0.0)` (kills the `BiharmonicClosure` stale-coefficient
  bug class). The `FieldTable` surface (`select`, `velocity`,
  `declaration`, `require`) is model.md's.
- **`tendency_terms` and variants** (08 §10.4): `model.variant`
  re-runs assembly with the same module tuple and filters the
  *collected* terms at step 5 — so `tendency_terms()` must be a
  deterministic pure function of the bound module (same module,
  same tuple back). Terms only ever **add**; anything that
  overwrites is a stage. Treatment is author-declared with the user
  override on the module constructor; an implicit term under a
  purely explicit stepper is an assembly error, never silent
  demotion (03 §5.1).
- **`dispatch` rules** (04 §6.2 step 3): merged exactly once (first
  model on the grid; later assemblies verify); the same resolved
  key from two modules is an error naming both — module order never
  silently selects an operator; `("declared_space", ...)` resolver
  entries are **never module-mergeable** (grid-level only, D1.2);
  the merge precedes bind, the dry run, and negotiate, so the halo
  trace sees an override's wider stencil (e.g. WENO, sketch 7.4).
  **No `Module.setup()` exists.**
- **`extra_halo` mechanics** (V-N2, 02_rules): a module declaring
  `extra_halo` has its terms **exempted from the halo trace** (the
  declared spec substitutes, entering
  `grid.negotiate(..., halo=extra_halo)` at step 7);
  contribution-key and write-gate validation for those terms runs
  in a second dry-run mode over real zero-valued fields. The
  standard implicit families need no `extra_halo` (their solves are
  registry Operators the tracer intercepts generically, 03 §5.1).
  `extra_halo` stays per-module; a per-term refinement is recorded
  as a trivial extension if ever needed (d3_1). Combined-negotiate
  semantics (2026-07-08 amendment, model.md step 7): the negotiated
  spec is trace ∨ extra_halo via `merge_max`, never either-or — the
  landed grid's exclusive-override behavior is a pending grid work
  item ([`../../phase2_grid_followups.md`](../../phase2_grid_followups.md)).
- **`self_update` semantics** (03 §5.2, 02_rules, CS-1):
  - runs **per substage at substage time** (S1), stage-consistent
    with `eval_params` — a ramped scalar and a ramped N² profile
    see the same time; once-per-step would inject an
    O(dt²·dN²/dt) error;
  - **scheduling trigger** (V-H5): scheduled iff one of the
    module's inputs is time-dependent (the Ramp-shaped rule —
    static models pay nothing) **or** its declaration names state
    inputs (`reads=("eta",)`, assembly-checked like a
    `FieldReference`) — covering state-derived AUX such as
    z*-geometry following `eta`;
  - write gate: **own AUXILIARY only**, applied via `replace`;
  - **SELF_UPDATE-first is load-bearing** (02_rules, D5): `reset()`
    leaves AUXILIARY fields untouched, so Tier-2 transform
    determinism relies on the owner recomputing from the reset
    clock *before any consumer reads* — S1 placement, first in
    every substage. Regression test: two consecutive transform
    calls on one input, bitwise-equal outputs, with a Ramp-valued
    AUX field in the twin;
  - **the accumulation hazard (docstring-documented, normative)**:
    `self_update` multi-counts under multi-stage steppers (RK3:
    three unweighted stage-time samples per step — correct under
    AB3 only by accident). Step-frequency accumulation (time
    means, window-mean fluxes, budgets) belongs in an **S6
    DIAGNOSTIC-kind stage** — the accumulation idiom below;
    `cadence=STEP` is **reserved on the declaration, not built**;
  - boundary with `fr.Ramp`: a Ramp describes a curve and never
    writes; scalars get time dependence via `resolve_at` at the
    point of use; field-consumed parameters get it via the owner's
    `self_update` rewriting the AUXILIARY field (D2.2).
- **Declaration spelling for `self_update`** (spec-level choice —
  the notes fix the `reads=` semantics, not the spelling): the bare
  method (sketch 7.2) is equivalent to `reads=()`; naming state
  inputs uses the decorator mirroring `@fr.term`:

  ```python
  @fr.self_update(reads=("eta",))          # cadence= reserved
  def self_update(self, state, ctx):
      return {"zstar_metric": ...}
  ```

  Assembly wraps either spelling into a SELF_UPDATE-kind `Stage`
  declaration (stages.py owns the decorator). See Open question 2.
- **jaxify discipline, restated for module authors**: dynamic =
  everything numeric that may be swept or ramped (parameter leaves,
  owned precomputed arrays); static = structure (target name
  tuples, solver choices, substep counts, curve shapes — changing
  them is different math: one recompile, correct). ADVANCE-stage
  integrator *statics* (substep count, filter spec) join the
  restart fingerprint alongside stepper statics (02_rules, V-H3);
  mechanically they are the owning module's jaxify-static
  attributes, hashable by discipline.
- **Removed relative to today's `Module`** (each replaced):
  `setup(mset)` (assembly is Model-owned; the dispatch merge is
  step 3), `update(mz)` in-place mutation (the uniform functional
  hooks), the `mset` back-reference (parameters are module-owned
  leaves; cross-module reads go through the binding table),
  `enable()/disable()` flags (re-assembly for structure,
  `fr.Ramp`-to-zero for continuous switch-on; D4), Ramper-style
  callbacks (`fr.Ramp` + `self_update`), per-module timers (the
  D4 §6.7 profiling tiers; `jax.named_scope("module/term")` is
  stamped by the composer).
- **Module-only models are first-class** (D1.3): a tracer
  advection-diffusion test model is
  `fr.Model(grid, modules=(TracerDiffusion("c"),), ...)` with no
  fake core; empty-PROGNOSTIC, stage-only schedules are legal
  (CS-13 — the mediator-as-model lint guard).
- **Rejected alternatives** (pointers, never re-argued): Model
  subclasses declaring the core (d1_3 option A), the `core=` slot
  (option C — recorded as the fallback if D3 had failed to deliver
  module-owned stages; it did not), equation-set objects (option
  D), grid-receiving declaration hooks (d1_2), Model-level
  treatment override dicts (d3_1), IC modules (d1_1).

---

### fr.closures.ClosureBase

The framework closure base: the `fr.terms.owned_by` predicate
target that earns its keep by hosting D1.4's role-target resolution
boilerplate (d5_2 ruling — the old code has no closure base;
`SmagorinskyLilly` and `BiharmonicClosure` subclass `Module`
directly).

| Aspect | Value |
|--------|-------|
| Kind | abstract Module subclass (marker + helper; no hooks of its own) |
| Pytree | inherited: jaxified per concrete subclass |
| Task | 2.8 (with the term predicates); consumed by the 2.7 closure ports |
| Design refs | 08 §10.4, d5_2 §1; D1.4 (role targeting, V-H2); d1_4 §8.4 |

```python
"""The closure base: marker for owned_by + role-target resolution."""
from __future__ import annotations

import fridom.framework2 as fr


class ClosureBase(fr.Module):
    """Base for dissipative closures; hosts role-target defaults."""

    default_targets: ClassVar[Role | type[Role]]               # 2.8
    """Subclass-declared default target selection: mixing closures
    set ``fr.roles.TRACER``, friction closures the ``Velocity``
    family (class = family match, D1.4)."""

    def __init__(                                              # 2.8
        self,
        *,
        fields: Role | type[Role] | Iterable[str] | None = None,
        exclude: Iterable[str] = (),
    ) -> None:
        """Store the target override (a role, a role family, or
        explicit names) and exclusions; resolution happens at bind."""
        ...

    def bind(self, table: FieldTable) -> None:                 # 2.8
        """Resolve ``fields or default_targets`` minus ``exclude``
        into the static ``targets`` tuple; validate name-keyed
        per-field options (unknown name -> assembly error)."""
        ...

    @property
    def targets(self) -> tuple[str, ...]:                      # 2.8
        """Frozen post-bind target names, declaration order."""
        ...
```

Notes:

- **What it replaces**: the `ENABLE_FRICTION`/`ENABLE_MIXING`
  declaration-side flags are dead (D1.4) — closures default their
  target set *by role* (friction → the `Velocity` family, mixing →
  `TRACER`) and take name-keyed constructor overrides
  (`fields=`/`exclude=`/per-field coefficient mappings, validated
  at bind: `unknown tracer 'x' in kappa=` → error). This fixes the
  in-tree three-closures-three-conventions inconsistency.
- **Role-driven write-targeting intersects PROGNOSTIC
  automatically** (V-H2): `Velocity` may sit on DIAGNOSTIC fields
  (hydrostatic diagnosed `w`), and role-driven *reads* span both
  lifecycles, but a closure's write targets are PROGNOSTIC by
  construction — physically correct (diagnosed `w` has no momentum
  equation) and listed in `model.report`.
- **The predicate follows free**: closures subclass `ClosureBase`
  for the ergonomics, and `~fr.terms.owned_by(fr.closures.
  ClosureBase)` drops all closures with no new vocabulary
  (inviscid-linear variants, OB's forward/backward filters).
  Foreign closures that don't subclass stay reachable via
  `owned_by(TheirClass)` / `named(...)`.
- **Rejected** (d5_2, pointer): a parallel `Module.category` tag
  axis — a second classification vocabulary with one consumer, as
  forgettable as the base class.
- Concrete closures follow the Module conventions unchanged: the
  treatment override on their constructors
  (`VerticalMixing(kv=..., treatment=fr.IMPLICIT)`), mergeable
  `fr.implicit` families for the implicit paths, and — for a
  closure exposing a diagnosed κ to IO — a declared AUXILIARY field
  updated in `self_update` (verify ergonomics at the first real
  port; Open question 4).

---

### Stage and StageKind

The schedule vocabulary: a stage is a declared, kind-ordered,
arbitrary pure function writing its declared subset — the
generalization the projection, the split-explicit subcycle, and a
Phase-3 Coupler all need (03 §5.2).

| Aspect | Value |
|--------|-------|
| Kind | `Stage`: concrete frozen dataclass, transient assembly data (mirrors `TendencyTerm`); `StageKind`: closed enum |
| Pytree | host objects, not pytrees — consumed at assembly, discarded; the composed step closes over slot indices and unbound functions |
| Task | 2.5 |
| Design refs | 03 §5.2, §5.4, §5.5; 02_rules (S6 idiom); D1.3 commitment 5; CS-1, CS-13, CS-17 |

```python
"""Stage declarations and the closed kind vocabulary."""
from __future__ import annotations

import enum
from dataclasses import dataclass


class StageKind(enum.Enum):
    """Closed it-1 vocabulary; schedule slot is a function of kind."""

    SELF_UPDATE = enum.auto()   # S1,  per substage           # 2.5
    DIAGNOSE = enum.auto()      # S1', per substage           # 2.5
    ADVANCE = enum.auto()       # S3', per substage           # 2.5
    CONSTRAINT = enum.auto()    # S4,  per substage           # 2.5
    DIAGNOSTIC = enum.auto()    # S6,  per step (post-NaN)    # 2.5


@dataclass(frozen=True)
class Stage:
    """A module-owned stage declaration (transient assembly data)."""

    kind: StageKind                                            # 2.5
    fn: Callable | str                                         # 2.5
    """UNBOUND ``(module, state, ctx) -> dict``; a string is the
    method name, resolved to the unbound method at collection."""
    name: str | None = None                                    # 2.5
    """Attribution key part; defaults to the fn name
    ("Module/stage")."""
    order: int = 0                                             # 2.5
    """Explicit intra-kind order; ties broken by (module tuple
    index, declaration index)."""
    advances: tuple[str, ...] = ()                             # 2.5
    """ADVANCE only: the named PROGNOSTIC subset this stage
    advances; counts as "advanced" in the coverage lint."""
    reads: tuple[str, ...] = ()                                # 2.5
    """SELF_UPDATE only: state inputs (the V-H5 scheduling
    trigger); assembly-checked like a FieldReference."""
    # cadence: SELF_UPDATE only — RESERVED (CS-1), not built.


def self_update(                                               # 2.5
    *, reads: tuple[str, ...] = (),
) -> Callable:
    """Decorator wrapping a module's ``self_update`` method into a
    SELF_UPDATE Stage declaration; the bare method (no decorator)
    is equivalent to ``reads=()``. ``cadence=`` reserved."""
    ...
```

Semantics, invariants, error behavior:

- **The canonical step** the kinds slot into (03 §5.2):

  ```
  per substage:  P0 ctx → S1 SELF_UPDATE → S1' DIAGNOSE → S2 terms
                 → S3 primary ADVANCE (the stepper) → S3' ADVANCE
                 stages → S4 CONSTRAINT
  per step:      S5 NaN seam → S6 DIAGNOSTIC → clock tick → host
  ```

  Tendency contributions are **terms, not stages**; the primary
  advance (S3) is the stepper's, not a module stage.
- **Write gates per kind** (03 §5.5, the uniform hook table —
  normative):

  | kind | cadence | write gate (applied via `replace`) |
  |---|---|---|
  | `SELF_UPDATE` | per substage, iff scheduled | own AUXILIARY |
  | `DIAGNOSE` | per substage | own DIAGNOSTIC (pre-tendency writes: hydrostatic `p_hyd = ∫b dz`, diagnosed `w` — S1' placement is load-bearing: the first substage after `set_state`/restart recomputes them before any term reads) |
  | `ADVANCE` | per substage | declared `advances` PROGNOSTIC subset **∪ own AUXILIARY** (V-H3: own-AUX is the declared home of stage-owned cross-step integrator state, e.g. the barotropic increment-form forcing buffer; the stage's integrator statics join the restart fingerprint) |
  | `CONSTRAINT` | per substage | PROGNOSTIC (role-selected, e.g. the velocity trio) + own DIAGNOSTIC (`p`) |
  | `DIAGNOSTIC` | per step, post-NaN-seam | own DIAGNOSTIC; **may read its own component's previous value — the accumulation idiom** (02_rules) |

  Write sets are validated in the assembly dry run (contribution
  keys are static under jit); ADVANCE `advances` claims are
  declared and count as "advanced" in the D1.4 coverage lint (else
  `eta, U, V` fail assembly — 03 §5.4).
- **Ordering discharges D1.3 commitment 5**: schedule position is a
  pure function of declared *kind*; within a kind the ordering
  tuple is `(order, module tuple index, declaration index)`.
  **Correctness never depends on list position** — an assembly lint
  errors on same-kind stages with overlapping write (or write-read)
  sets and equal `order=`, demanding an explicit order. Bitwise
  determinism *may* tie-break by module order (permuting the tuple
  already changes the treedef). Lint mechanics live in model.md's
  assembly.
- **User extension is kind declaration**: positivity clamp →
  CONSTRAINT; step-cadence accumulator → DIAGNOSTIC; time-dependent
  geometry → SELF_UPDATE. The old `add_module`-before-the-trio hack
  becomes a theorem (terms precede constraints by kind). **No
  "insert before X" API; no open kind set** in it-1.
- **The S6 accumulation idiom** (02_rules, CS-1 — normative): a
  DIAGNOSTIC stage may read its own component's previous value and
  `replace` with the updated sum — step-cadence, post-NaN-seam
  (garbage never enters accumulators), carry-resident
  (restart-exact), host-read at chunk boundaries, host-reset via
  the extended `set_aux`. **This — not `self_update` — is the
  sanctioned home for step-frequency accumulation.** Precision is
  global (Silvano's ruling, 2026-07-08 — no per-space width axis;
  the landed `Scalars` is Körper-only and storage dtype derives from
  the global x64 flag): in reduced-precision runs the accumulator
  precision story is the S6 chunk cadence itself — in-trace sums
  span at most one chunk (~256 steps), and the chunk-boundary host
  read accumulates in float64 on the host. See the CS-17 resolution
  in [`declarations.md`](declarations.md)'s Open questions.
  `fr.modules.WindowAccumulator` ships this as the preset (module
  library, out of this doc's scope).
- **ADVANCE stages** are the by-variable split (03 §5.4):
  module-owned, advancing named PROGNOSTIC subsets reading the
  latest state (Gauss-Seidel via the read rule); buffers are owned
  per ADVANCE stage; coupled implicit blocks are atomic under the
  split. The IMEX-RK × split-explicit combination is an assembly
  error (multistep outer drivers only) — checked at step 5, not
  here.
- **`div` is not declared** (03 §5.2 ruling): declare DIAGNOSTIC
  iff read outside the producing stage; `div` is a local variable
  of the projection stage. Stages compose locally instead of
  communicating through the carry as the old trio did.
- **Rejected** (pointers): nested mini-models for the barotropic
  subcycle (03 §5.4 — hidden η, nested restart, broken treedef
  discipline); an open kind set and insert-before APIs (d3_3);
  per-substage NaN checks (S5 is once per step).

---

### StepContext

The frozen context bundle every in-trace hook receives — kwargs at
the notebook boundary, ctx inside the trace (03 §5.5).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final, frozen |
| Pytree | `fr.utils.jaxify`; all-scalar leaves in the base form; the post-TENDENCY extension additionally carries the State-valued per-treatment sums |
| Task | 2.5 |
| Design refs | 03 §5.5, §5.4 (per-treatment sums), §5.2 (P0); D2 reconciliation 5; 02_rules (float64 clock) |

```python
"""The per-substage context bundle for in-trace hooks."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@fr.utils.jaxify                    # frozen; all-scalar leaves (base)
class StepContext:
    """Frozen per-substage context; built at P0 by the composed
    step, never by user code."""

    params: Mapping[str, jax.Array]                            # 2.5
    """eval_params(modules, stage_time) — live leaves read fresh at
    each stage time through the D2 binding table; Ramp values
    evaluated at RK sub-stage times."""

    clock: Clock                                               # 2.5
    """Traced clock; ``clock.time`` == the stage time, at global
    width (02_rules, Clock precision — float64 under the default
    x64-on run)."""

    dt: jax.Array                                              # 2.5
    """Full step size — cfl-type consumers' read surface."""

    stage_dt: jax.Array                                        # 2.5
    """Increment of the current advance (the ``p = φ/stage_dt``
    pressure normalization; backward runs thread sign conventions
    through it)."""

    tendency_sums: Mapping[Treatment, State] | None = None     # 2.5
    """Per-treatment tendency sums — populated for post-TENDENCY
    hooks only (ADVANCE/CONSTRAINT/DIAGNOSTIC stages); ``None``
    for terms, SELF_UPDATE, and DIAGNOSE."""
```

Semantics, invariants:

- **Construction is composer-internal** (P0 of every substage,
  03 §5.2): `ctx_i = StepContext(eval_params(modules, t_i),
  clock@t_i, dt, stage_dt_i)`; post-TENDENCY stages receive the
  extended form. User code never builds one; pure diagnostics keep
  explicit kwargs and the binding layer converts ("kwargs at the
  notebook boundary, ctx inside the trace" — D2.3 reconciliation).
- **Why ctx beats positional growth** (d3_3): the list is four and
  growing; ergonomics stay flat (Coriolis ignores ctx; `dsqr`
  consumers read `ctx.params["nonhydro.dsqr"]`; owners resolve
  their own Ramps via `resolve_at(self.x, ctx.clock.time)`).
- **Halo-tracer indifference** (d3_3, load-bearing for the one-path
  validation claim): the tracer wraps *state*; the base ctx is
  scalars — zero mimicry machinery. The per-treatment sums are the
  one State-shaped exception, present only for post-TENDENCY
  stages; being state-shaped, they are wrapped by the same tracer
  machinery when a stage is dry-run.
- **`stage_dt` is not `dt_gamma`** (03 §5.8 reconciliation 3): the
  implicit `solve` receives γΔt as a separate positional supplied
  by the stepper (γ-specific: CN dt/2, SBDF2 2dt/3) — never read
  from ctx; `ctx.stage_dt` exists for the pressure normalization.
- **The sums' consumers** (03 §5.4): the barotropic slow forcing —
  whose *default* is the increment form computed from the
  substage-start state via the ADVANCE own-AUX gate; the raw
  per-treatment-sums variant is the module constructor knob
  (`forcing="tendency_sums"`). The sums store the **summed**
  explicit contribution per treatment (per-term history is never
  needed; the partition is consumed at accumulation time).
- Frozen means frozen: no methods, no mutation, no back-references
  to Model or modules (no coupling/run-loop state outside the
  carry).

---

## Open questions

Closed by prior sign-offs and not reopened here: the self_update
cadence question (per substage — 03 §5.8), the accumulation home
(S6 DIAGNOSTIC, `cadence=STEP` reserved — CS-1), the `div`
declaration ruling (stage local), the `eps` scope (order-2-only,
time_steppers.md), the ClosureBase-vs-category ruling (d5_2), and
the write-gate table itself (03 §5.5).

1. **`state_type`-under-jaxify mechanics** (07_open_threads, D1/D4
   residual — the cluster's one live design point): a class
   reference as a jaxify-*static* attribute is hashable and should
   be mechanically fine, but the interaction with jaxify's
   static-attribute registration (does the class reference enter
   the treedef and thus the jit cache key? does subclass
   inheritance of the attribute confuse the >1-provider check?)
   needs a decision at implementation time. The
   `fr.Model(state_type=...)` kwarg fallback stands regardless
   (D1.3 commitment 4), so nothing downstream blocks on this.
2. **The `self_update` declaration spelling**: this spec proposes
   `@fr.self_update(reads=..., cadence=RESERVED)` mirroring
   `@fr.term`, with the bare method equivalent to `reads=()`
   (sketch 7.2 stays expressible). The notes fix the `reads=`
   semantics and the reservation of `cadence=`, not the spelling —
   confirm against declarations.md's decorator family before
   implementation.
3. **DIAGNOSTIC-chain ordering** (03 §5.9 residual): explicit
   `order=` now; topological sort by declared reads/writes is the
   recorded upgrade — revisit only if real DIAGNOSTIC dependency
   chains appear in the 2.7 ports.
4. **Closure AUXILIARY-κ ergonomics** (d3_1 residual): a closure
   exposing a diagnosed κ for IO owns it as a declared AUXILIARY
   field updated in `self_update` (terms cannot write DIAGNOSTIC
   intermediates — write gate). Verify the ergonomics when porting
   the first real closure (2.7); if κ is wanted at step cadence for
   accumulation, the S6 idiom applies instead.
