---
status: normative
date: 2026-07-13
---

# Model layer redesign — Class designs: transforms

Part of the model-layer class designs; see [`README.md`](README.md)
for the document map and the shared template. Normative design:
[`../08_state_transforms.md`](../08_state_transforms.md) (§10.1–10.8,
signed off 2026-07-08; **NNMD descoped**), with the D5 summary in
[`../01_concepts.md`](../01_concepts.md), the lifecycle semantics the
laws cite in [`../04_run_loop_io.md`](../04_run_loop_io.md) §6.3/§6.5,
the SELF_UPDATE-first and Ramp-endpoint rules in
[`../02_rules.md`](../02_rules.md), and coupling constraints
**CS-12/CS-14** ([`../09_coupling_designfor.md`](../09_coupling_designfor.md)
§11.3) binding. Research: d5_1–d5_3.

Ownership boundaries: **`model.variant(...)` mechanics —
constructor, assembly integration, snapshot semantics, fingerprint —
are specced in [`model.md`](model.md)**; this file specs only their
transform-side *consumption*. The `fr.terms` predicate vocabulary and
`fr.closures.ClosureBase` are owned by
[`declarations.md`](declarations.md) /
[`module.md`](module.md). The eigenmode surface (`em.q`/`em.p`/
`em.projector`, `from_model`) is ROADMAP-2.7 territory; the
projections here consume it.

---

## Module placement and homing (amended V-S4)

- Package `fridom/framework2/transforms/` = **`fr.transforms`**: the
  base class (re-exported top-level as **`fr.StateTransform`**), the
  signature/info/cost/progress vocabulary, the error types, the
  algebra nodes, `Identity`, `Shift`, `FixedPoint`,
  `relative_l2`, `assert_idempotent`, and the **Tier-2 presets
  `Propagator`, `TimeAverage`, `OptimalBalance`** — homed
  framework-level because shallowwater is a first-class OB consumer
  (V-S4 supersedes the earlier `nh.transforms`-only homing).
- **`base=` default resolution** goes through the
  **core-module-supplied default-projector hook** — the D1.3
  commitment-4 channel (the same core-module slot family as
  `state_type`): a model package's dynamical-core module may supply
  a factory `default_projector(model) -> StateTransform`;
  `OptimalBalance(base=None)` resolves through it; if the core
  supplies none, a hinted error demands an explicit `base=`. The
  hook *slot* (exact spelling, provider-uniqueness validation) lands
  on the Module capability menu ([`module.md`](module.md)); this
  file specs the consumption.
- Model packages ship **thin alias modules** (`nh.transforms`,
  `sw.transforms`) re-exporting the fr presets plus their
  **package-specific Tier-1 projections**
  (`VorticalProjection`/`WaveProjection`/`DivergenceProjection`).
  The old `fr.projection` namespace **dissolves**; `nnmd.py` is not
  ported (descoped, see the NNMD entry).
- `fr.linearize(model)` is a top-level function (sugar over
  `model.variant`).
- Naming adjacency, accepted and documented at sign-off: spectral
  *operator* transforms live in `fr.operators` (grid layer);
  `fr.transforms` is the state-transform namespace.

**Task tags**: the whole cluster is **ROADMAP 2.8**; the projections
additionally require the 2.7 eigenmode surface, and the 2.8 → 2.7
dependency (`2.4 + 2.5 (+ eigenmodes from 2.7) ► 2.8`) is the
ROADMAP's. Members are tagged `# 2.8` or `# designed-for`.

## Cluster-wide rules (the tiers and the laws, consumed)

Transcribed from §10.1–10.3; restated here once so per-class notes
stay lean.

- **Two tiers, one abstraction.** Tier 1 (closed-form — eigenmode
  projections, filters, `Shift`, `Identity`) are **jaxify-registered
  frozen pytrees** (structure static, numeric captures dynamic
  leaves — the `fr.Ramp` pattern): jit- and vmap-able
  (`jax.jit(jax.vmap(P))(batch)` for IC ensembles; sharding
  composes). Tier 2 (dynamical — run a model internally) are **host
  objects, never pytrees**, with a **trace guard**: `__call__`
  raises a taught `TraceError` on tracer-valued inputs. `traceable`
  ANDs under every combinator; ensembles through Tier 2 are a
  Python loop (batched internal models `designed-for`).
- **Law 1 — determinism.** `T(state)` is a pure function of its
  input given frozen config. Tier-2 call semantics, normative:
  `m.reset(); m.set_state(state); m.advance(N); return
  m.state.prognostic` — PROGNOSTIC-only read-back; the final clock
  goes to *info*, never into the returned State; config = parameter
  leaves snapshotted at construction (later parent updates don't
  propagate — documented; `T.with_parameters(...)` designed-for).
  Load-bearing underneath (02_rules invariant + regression test):
  `reset()` leaves AUX untouched, so Tier-2 determinism relies on
  **SELF_UPDATE running first in every substage**. `advance` is
  §6.3's IO-free primitive; `reset`/`set_state` semantics are §6.5's.
- **Law 2 — signatures.** Compared: grid **object identity** + the
  mapped `(name, space)` subset (names by equality, spaces by
  identity, order included). Checked eagerly at compose time AND
  rechecked at call time (trace time under jit — zero steady cost).
  **Amended (reconciliation §10.7.2)**: the call-time check requires
  the input to *contain* the mapped components; extra PROGNOSTIC
  components follow the transform's declared **`rest` policy** —
  default `rest="zero"` (unmapped components map to zero in the
  output, so completeness holds restricted to the family and
  `state − P(state)` carries the full tracer); `rest="pass"` is the
  explicit opt-out. **Signature ≠ treedef** (S6):
  accumulator-augmented twins and explicit-params transforms
  interoperate with model-built ones.
- **Law 3 — isolation by construction.** A Tier-2 transform **never
  stores the model you pass it** — the constructor treats it as an
  assembly spec and builds its own internal model (via
  `model.variant(...)`, or — amended V-C6 — a **fresh assembly on
  the parent's frozen grid** when the twin adds declarations, e.g.
  the accumulator upgrade; added AUX declarations must reuse spaces
  already in the parent's state-space set, or the
  satisfiability-relaxed verify covers them). No injection path
  exists, so two transforms never share a model. **Cross-transform
  staleness (amended V-C8)**: transforms constructed on either side
  of an `update_parameters` embed *different* frozen configs — the
  staleness rule is per-transform; cross-transform consistency is
  the caller's (build both, then update neither; or `resync()` when
  it lands). Preset refinement: OB's exposed leg propagators share
  the *preset's own* two models — sequential, reset-prefixed use is
  the documented exception within one owner. Passing a pre-built
  model (e.g. a variant you assembled yourself) is **ownership
  transfer**. Internal models run `io=()`, named
  `"{transform}/internal"` for log attribution, and keep no run
  state outside their carries (the CS-11/CS-12 discipline; the
  public carry read is the autopsy path).
- **Law 4.** Transforms map the PROGNOSTIC subset; outputs are
  `set_state`-compatible (§6.5 V-C12: `set_state` leaves missing
  components untouched, so trimmed/`rest`-handled outputs feed
  larger models cleanly).
- **CS-14 — product-ready base**: signatures are opaque compared
  values; **no `isinstance(state, State)` in the base** (component
  access is duck-typed through the container surface); endo-ness is
  asserted only where required (`A ** n` with n ≥ 2, `FixedPoint`).
- **Stages vs Tier-1 transforms** (ruled: a use relation, not
  competition): the pressure projection **stays a stage** (not
  autonomous; consumes `stage_dt`), but any stage body may call a
  Tier-1 transform held as bind-time static structure (same model,
  `traceable=True`) — e.g. a relaxation CONSTRAINT stage nudging
  only the balanced part.
- **Idempotency**: a declared flag, never detected; exactly three
  consumers — the `complement` sugar, the `P @ P` info-level lint
  hint, and `assert_idempotent`.

### Spec-level completions (called out, not re-decisions)

Points where these skeletons pin something the normative notes left
unspelled; each is a completion consistent with the signed design,
none changes a decision:

1. **`TransformInfo` attribute sugar**: §10.1 pins the field list
   (iterations, errors, model_steps, elapsed_model_time, extra);
   sketch 7.9 spells `info.stopped_by`. Reconciled here: unknown
   attribute reads resolve into `extra` (read-only), so
   preset-specific detail keeps the sketch spelling without widening
   the base field list.
2. **`TransformProgress` payload**: the design names the host-side
   hooks (`on_progress`, path-prefixed through composites;
   `on_iteration`) but not the payload; the field list is pinned
   here.
3. **`Propagator` gets `filter=`/`updates=` kwargs** threaded to its
   internal `model.variant(...)` call — d5_2 §6's general rule
   ("transforms take the user's model as a spec and build their
   variant internally, `filter=` kwarg") applied to the building
   block; backward propagation is spelled
   `updates={fr.params.TIME_STEP: -dt}`.
4. **`StateSignature.rest` is excluded from signature comparison**:
   `rest` is call-time behavior of the owning transform, not type
   identity — including it would break `Identity` polymorphism and
   the mapped-subset interop that law 2 exists to provide.
5. **`FixedPoint` factory-form signature**: with a factory argument
   the wrapped transform does not exist at construction, so the
   combinator is signature-polymorphic until first composed/called
   (like `Identity`); every factory-produced transform is
   endo-checked per iteration.
6. **The TimeAverage accumulator-upgrade write mechanism**: §10.5's
   parenthetical says `self_update`; the later-signed coupling
   bundle (02_rules, "the S6 accumulation idiom") rules
   step-frequency accumulation into a DIAGNOSTIC-kind stage
   (`self_update` multi-counts under multi-stage steppers). The
   upgrade note below follows 02_rules; the design intent
   (carry-resident running sum; signature ≠ treedef load-bearing)
   is unchanged.

---

### fr.StateTransform

First-class composable `State -> State` map — the D5 base; the
algebra mirrors the operator algebra (§10.1).

| Aspect | Value |
|--------|-------|
| Kind | ABC (abstract base; Tier-1 and Tier-2 families subclass) |
| Pytree | tier-dependent: Tier-1 subclasses are jaxify-registered frozen pytrees; Tier-2 subclasses are host objects, never pytrees |
| Task | 2.8 |
| Design refs | §10.1, §10.2, §10.3 (laws), CS-14; d5_1 §1/§3/§4 |

```python
"""The state-transform base: composable State -> State maps."""
from __future__ import annotations

import fridom.framework2 as fr


class StateTransform:
    """Abstract composable State -> State map (decision D5)."""

    # ================================================================
    #  Declared structure
    # ================================================================

    @property
    def domain(self) -> StateSignature:                        # 2.8
        """Input signature: grid identity + mapped (name, space)
        subset + rest policy."""
        ...

    @property
    def codomain(self) -> StateSignature:                      # 2.8
        """Output signature (same vocabulary)."""
        ...

    @property
    def traceable(self) -> bool:                               # 2.8
        """Tier 1 iff True; ANDs under every combinator."""
        ...

    @property
    def idempotent(self) -> bool:                              # 2.8
        """Declared, never detected; default False."""
        ...

    # ================================================================
    #  Application
    # ================================================================

    def __call__(self, state) -> State:                        # 2.8
        """Apply the transform. Bitwise equal to
        ``call_with_info(state)[0]`` (law); rechecks the domain
        signature (trace-time under jit); Tier-2 implementations
        trace-guard first (TraceError on tracer-valued input)."""
        ...

    def call_with_info(                                        # 2.8
        self, state,
    ) -> tuple[State, TransformInfo]:
        """Apply and return structural info (a tree mirroring the
        composition tree). Default info is ``TransformInfo.EMPTY``.
        The one info spelling: no mutating ``last_info`` attribute,
        no ``return_details=True``."""
        ...

    def cost(self) -> TransformCost:                           # 2.8
        """Internal model steps (Tier 1: zero); sums structurally
        under composition; FixedPoint reports a flagged bound."""
        ...

    def __repr__(self) -> str:                                 # 2.8
        """The annotated composition tree (tier + step counts) —
        the cost-opacity answer."""
        ...

    # ================================================================
    #  Sugar
    # ================================================================

    @property
    def complement(self) -> StateTransform:                    # 2.8
        """``Identity() - self``; idempotent-gated — raises a
        taught TypeError on non-idempotent transforms."""
        ...

    # ================================================================
    #  The algebra (dunders build the node classes below)
    # ================================================================

    def __matmul__(self, other: StateTransform) -> StateTransform:
        """Composition, right-to-left: ``(A @ B)(s) = A(B(s))``;
        requires ``B.codomain == A.domain`` (eager check)."""
        ...                                                    # 2.8

    def __add__(self, other: StateTransform) -> StateTransform:
        """Pointwise sum on outputs; domains AND codomains equal."""
        ...                                                    # 2.8

    def __sub__(self, other: StateTransform) -> StateTransform:
        """``self + (-1) * other`` (structural)."""
        ...                                                    # 2.8

    def __mul__(self, scalar: complex) -> StateTransform:      # 2.8
        """Scalar scaling of the output; plain scalars only — a
        Ramp coefficient raises TypeError ('evaluate the ramp
        explicitly'); field-valued coefficients designed-for."""
        ...

    def __rmul__(self, scalar: complex) -> StateTransform: ... # 2.8

    def __neg__(self) -> StateTransform:                       # 2.8
        """``(-1) * self``."""
        ...

    def __pow__(self, n: int) -> StateTransform:               # 2.8
        """Fixed iteration: ``n == 0`` -> Identity(self.domain);
        ``n == 1`` -> self; ``n >= 2`` requires endo (domain ==
        codomain); ``n < 0`` raises (no inverses)."""
        ...
```

Semantics, invariants, error behavior:

- **All signatures are concrete at construction** → every algebra
  check is **eager at compose time**, plus the call-time recheck.
  `SignatureMismatchError` on any failure (see the error entry).
- **The trace guard** is the base-class front door for
  `traceable=False` transforms: tracer-valued leaves in the input
  (i.e. the transform appears under `jit`/`vmap`/`grad`) raise
  `TraceError` with taught guidance (Tier-2 transforms run models —
  hoist the call out of the traced region). Tier-1 transforms pass
  tracers through by construction.
- **Progress hooks are host-side observers**: `on_progress`
  (Tier-2 constructors) and `on_iteration` (FixedPoint) receive
  `TransformProgress` payloads, path-prefixed through composites;
  they may observe, never influence results (normative).
- **No adjoints, no inverses, no `Zero` transform** (no consumer;
  backward is a physical, not algebraic, inverse of forward).
- **`complement`** is one of idempotency's three consumers; the
  other two are the `P @ P` info-level lint hint (emitted when an
  idempotent transform is composed with itself — never rewritten)
  and `assert_idempotent`.
- **vmap/ensembles**: Tier-1 transforms vmap over a leading batch
  axis on State leaves (frozen pytrees + identity-hashed statics);
  Tier-2 ensembles are a Python loop (batched internal models
  `designed-for`, §10.8).
- **Product-readiness (CS-14)**: the base never touches concrete
  `State` types — signatures are opaque compared values; endo
  assertions only at `** n` (n ≥ 2) and `FixedPoint`.
- Rejected alternatives (research archive, never re-argued):
  `last_info` mutating attribute (d5_1 — silently wrong under jit,
  order-dependent); `return_details=True` (dies with the port);
  Ramp-valued coefficients (transforms are autonomous maps — no
  clock in the surface); automatic idempotent folding / rewriting
  (what you wrote is what runs); a `linear` flag (no consumer).

---

### StateSignature

The compared value behind transform domains/codomains: grid identity
plus the mapped `(name, space)` subset, plus the `rest` policy for
extra input components (law 2, §10.7.2).

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen dataclass, final |
| Pytree | not a pytree; lives in Tier-1 statics (identity-hashed grid/spaces inside) and on host objects |
| Task | 2.8 |
| Design refs | §10.3 law 2, §10.7.2, S6 (signature ≠ treedef), CS-14 |

```python
"""Transform signatures: what compose/call checks compare."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateSignature:
    """Grid identity + ordered mapped (name, space) subset + rest
    policy."""

    grid: Grid                                                 # 2.8
    components: tuple[tuple[str, SpaceLike], ...]              # 2.8
    rest: Literal["zero", "pass"] = "zero"                     # 2.8

    @classmethod
    def of_prognostic(cls, source) -> StateSignature:          # 2.8
        """The full ordered PROGNOSTIC (name, space) table of a
        model or state — the Tier-2 preset signature."""
        ...

    def __eq__(self, other: object) -> bool:                   # 2.8
        """Grid by object identity; names by equality; spaces by
        identity; order included. ``rest`` excluded (call-time
        policy, not type identity — spec completion 4)."""
        ...

    def __hash__(self) -> int: ...                             # 2.8

    def validate_input(self, state, *, path: str = "") -> None:
        """Call-time check: the input must *contain* the mapped
        components (name + space + order); extra PROGNOSTIC
        components are legal and handled per ``rest``. Raises
        SignatureMismatchError with the composition-tree path."""
        ...                                                    # 2.8

    def __repr__(self) -> str:                                 # 2.8
        """Grid name/id + the component table + rest policy."""
        ...
```

Notes:

- **Signature ≠ treedef** (S6) is the reason this class exists as
  its own vocabulary: an accumulator-augmented twin (extra DIAG/AUX
  declarations, different carry treedef) and an explicit-params
  Tier-1 transform both interoperate with model-built transforms
  because comparison sees only the mapped PROGNOSTIC subset on the
  same grid object.
- **The mapped subset**: Tier-2 presets map the internal twin's full
  PROGNOSTIC table (`of_prognostic`); family-built projections
  declare their family subset (nonhydro: `u, v, w, b`) explicitly.
- **`rest` policy** (reconciled, signed): default `"zero"` — output
  components outside the mapped subset are zeroed (completeness
  restricted to the family; `state − P(state)` carries the full
  tracer); `"pass"` copies them through unchanged ("the geostrophic
  state keeps its dye"). Old code never faced the choice
  (fixed-composition State).
- **CS-14**: comparison is opaque — consumers compare
  `StateSignature` values, never introspect `State` classes; no
  `isinstance(state, State)` anywhere in the check path.
- The componentwise diff table and hinted grid verdicts produced on
  failure are `SignatureMismatchError`'s payload (below);
  `validate_input` only assembles them.

---

### TransformInfo, TransformCost, TransformProgress

The info/cost/progress vocabulary of §10.1 — everything is
*returned* or *observed*, nothing mutates the transform.

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen dataclasses, final |
| Pytree | host values (never traced; produced at the host level) |
| Task | 2.8 |
| Design refs | §10.1; d5_1 §1; §10.7.1 (info spelling reconciliation) |

```python
"""Info, cost, and progress records for state transforms."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransformInfo:
    """Structural call record; a tree mirroring the composition
    tree (OB's convergence record addressable by path)."""

    iterations: int | None = None                              # 2.8
    errors: tuple[float, ...] = ()                             # 2.8
    model_steps: int = 0                                       # 2.8
    elapsed_model_time: float = 0.0                            # 2.8
    extra: Mapping[str, object] = field(default_factory=dict)  # 2.8
    children: tuple[tuple[str, TransformInfo], ...] = ()       # 2.8

    EMPTY: ClassVar[TransformInfo]                             # 2.8

    def __getitem__(self, path: str | int) -> TransformInfo:   # 2.8
        """Child info by composition-tree label or index."""
        ...

    def __getattr__(self, name: str) -> object:                # 2.8
        """Read-only fallback into ``extra`` (spec completion 1:
        keeps sketch 7.9's ``info.stopped_by`` spelling without
        widening the base field list)."""
        ...


@dataclass(frozen=True)
class TransformCost:
    """Static cost estimate: internal model steps."""

    model_steps: int = 0                                       # 2.8
    upper_bound: bool = False                                  # 2.8
    # FixedPoint reports max_it * per-iteration steps, flagged.

    def __add__(self, other: TransformCost) -> TransformCost:  # 2.8
        """Steps sum; the upper-bound flag ORs."""
        ...


@dataclass(frozen=True)
class TransformProgress:
    """Host-side observer payload for on_progress/on_iteration
    hooks (spec completion 2). Observers may never influence
    results (normative)."""

    path: str                                                  # 2.8
    steps_done: int = 0                                        # 2.8
    steps_total: int | None = None                             # 2.8
    elapsed_seconds: float = 0.0                               # 2.8
```

Notes:

- **The one law**: `T(s) == call_with_info(s)[0]` bitwise; the info
  path adds observation, never a second code path. *Clarified
  (2026-07-08)*: the law requires both spellings to execute the
  **same compiled computation** — info is host-side observation
  (Tier 2) or extracted without adding outputs to a jitted program;
  where info values are themselves traced (`FixedPoint` iteration
  counts), the law is tested per-transform or downgraded to an
  identically-compiled comparison (phase-1 finding 1).
- Composite nodes label their children for path addressing (the
  labels are the repr's node names); `FixedPoint` records one child
  per iteration plus `stopped_by`/`returned_iteration` in `extra`.
- The final internal-model clock of a Tier-2 call lands in
  `elapsed_model_time` — *never* in the returned State (law 1).
- `cost()` answers "what will this cost *before* calling"; `info`
  answers "what did it actually do". Both are the cost-opacity
  mitigation for composed Tier-2 memory/step surprises.

---

### SignatureMismatchError, TraceError

The cluster's two error entries; both register in
[`model.md`](model.md)'s error-type registry.

| Aspect | Value |
|--------|-------|
| Kind | concrete exception types, final |
| Pytree | n/a |
| Task | 2.8 |
| Design refs | §10.3 law 2/law 3; d5_1 §3 |

```python
"""Transform error types."""
from __future__ import annotations


class SignatureMismatchError(TypeError):
    """Compose- or call-time signature mismatch. Carries the
    composition-tree path, a componentwise diff table (missing /
    extra / space-mismatched, in order), and the grid-identity
    verdict with hints (same-shaped different grid object -> 'one
    grid, many models: build both on one grid')."""
    ...                                                        # 2.8


class TraceError(TypeError):
    """A Tier-2 (traceable=False) transform received tracer-valued
    input — it runs a model internally and cannot appear under
    jit/vmap/grad. Taught guidance: hoist the call to the host
    level, or use a Tier-1 transform inside traces."""
    ...                                                        # 2.8
```

---

### The algebra nodes: Compose, Sum, Scaled, Power

Internal node classes built by the dunders — never user-constructed;
named so reprs and info paths stay addressable (§10.2).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final, internal (constructed by dunders only) |
| Pytree | jaxify-registered, children dynamic subtrees — meaningful for all-Tier-1 trees; a tree with a Tier-2 child is a host object in practice (its trace guard fires under jit) |
| Task | 2.8 |
| Design refs | §10.2; d5_1 §2 |

```python
"""Algebra nodes: structural composition, sum, scaling, power."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@partial(fr.utils.jaxify, dynamic=("_parts",))
class Compose(fr.StateTransform):
    """Right-to-left composition chain: parts[0] applied last."""

    @property
    def parts(self) -> tuple[fr.StateTransform, ...]:          # 2.8
        """Flattened chain (nested Compose flattened, Identity
        elided) — what you wrote is what runs."""
        ...


@partial(fr.utils.jaxify, dynamic=("_parts",))
class Sum(fr.StateTransform):
    """Pointwise sum on outputs; equal domains AND codomains."""

    @property
    def parts(self) -> tuple[fr.StateTransform, ...]:          # 2.8
        """Flattened summands (nested Sum flattened)."""
        ...


@partial(fr.utils.jaxify, dynamic=("_inner", "_coefficient"))
class Scaled(fr.StateTransform):
    """Plain-scalar scaling of a transform's output."""

    @property
    def coefficient(self) -> complex: ...                      # 2.8

    @property
    def inner(self) -> fr.StateTransform: ...                  # 2.8


@partial(fr.utils.jaxify, dynamic=("_inner",))
class Power(fr.StateTransform):
    """Fixed n-fold iteration of an endo transform (n >= 2)."""

    @property
    def n(self) -> int: ...                                    # 2.8

    @property
    def inner(self) -> fr.StateTransform: ...                  # 2.8
```

Semantics, invariants:

- **Normalization is structural only**: flatten Compose-of-Compose
  and Sum-of-Sum, elide Identity — **no rewriting, no idempotent
  folding** (silent rewriting is a footgun; `P @ P` on an idempotent
  transform emits the info-level lint hint instead).
- Derived structure: `traceable` = AND of children; `cost()` = sum
  of children (`Power`: `n ×` inner); signatures derived at node
  construction with the eager checks of the algebra table
  (`Compose`: pairwise `B.codomain == A.domain`; `Sum`: all domains
  and all codomains equal; `Power`: endo).
- `call_with_info` assembles the mirrored info tree (children
  labeled by position and class name); `Sum` evaluates each summand
  on the *input* and sums outputs (pointwise on outputs, by State's
  vector-space structure).
- `idempotent` on nodes is `False` (never inferred; a user may not
  declare it on a node — nodes are not user-constructed).
- The `n == 0` and `n == 1` cases of `**` never construct a `Power`
  (Identity / self); `n < 0` raises at the dunder.

---

### Identity

The signature-polymorphic unit of the algebra (§10.2).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final; Tier 1 |
| Pytree | jaxify-registered, no dynamic leaves |
| Task | 2.8 |
| Design refs | §10.2; d5_1 §2 |

```python
@fr.utils.jaxify
class Identity(fr.StateTransform):
    """s -> s; signature-polymorphic until composed or called."""

    def __init__(                                              # 2.8
        self,
        domain: StateSignature | None = None,
    ) -> None:
        """Optionally pinned to a signature (``A ** 0`` pins to
        A.domain); default polymorphic."""
        ...
```

Notes: `traceable=True`, `idempotent=True`; **elided in chains**
during structural normalization (so `Identity() - P` is a real `Sum`
but `A @ Identity() @ B` is `A @ B`). While polymorphic, it matches
any signature at compose time and adopts the partner's; a pinned
Identity checks like any transform.

---

### Shift

The affine piece: `s ↦ s + state0` (§10.2) — OB's base-point
exchange building block.

| Aspect | Value |
|--------|-------|
| Kind | concrete, final; Tier 1 |
| Pytree | `jaxify, dynamic=("_state0",)` — the captured state is the dynamic leaf (the `fr.Ramp` pattern) |
| Task | 2.8 |
| Design refs | §10.2; d5_1 §2; d5_3 §3 (OB exchange) |

```python
@partial(fr.utils.jaxify, dynamic=("_state0",))
class Shift(fr.StateTransform):
    """s -> s + state0 (State arithmetic; componentwise)."""

    def __init__(self, state0) -> None:                        # 2.8
        """Capture the shift state; the endo signature derives
        from state0's components (grid + mapped subset)."""
        ...

    @property
    def state0(self) -> State: ...                             # 2.8
```

Notes: `traceable=True`, `idempotent=False`; cost zero. The sum runs
through State's componentwise arithmetic (grid identity + space join
rules of the field layer apply and produce their own errors on
mismatch — the signature check catches structural mismatch first).

---

### FixedPoint

Iteration combinator with divergence policy; absorbs OB's
`update_base_point` via the factory form (§10.2).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final |
| Pytree | host object, not a pytree (`traceable=False` in it-1; a traced `lax.while_loop` variant is designed-for, no consumer) |
| Task | 2.8 |
| Design refs | §10.2 (pinned signature), §10.7.3; d5_1 §2 |

```python
"""Fixed-point iteration over a transform (or transform factory)."""
from __future__ import annotations

import fridom.framework2 as fr


class FixedPoint(fr.StateTransform):
    """Iterate T until the norm of the update converges."""

    def __init__(                                              # 2.8
        self,
        transform: (fr.StateTransform
                    | Callable[[State], fr.StateTransform]),
        *,
        tol: float = 1e-9,
        max_it: int = 3,
        norm: Callable[[State, State], float] = fr.transforms.relative_l2,
        on_divergence: Literal["stop_best", "raise", "ignore"] = "stop_best",
        on_iteration: Callable[[TransformProgress], None] | None = None,
    ) -> None:
        """transform: an endo StateTransform, or a factory
        ``State -> StateTransform`` evaluated on the current
        iterate per iteration (absorbs OB's update_base_point)."""
        ...
```

Semantics, invariants, error behavior:

- **The pinned kwargs are the whole surface** — `tol`, `max_it`,
  `norm`, `on_divergence`, `on_iteration`; nothing model-flavored
  (a model-supplied default norm is rejected: invisible criterion,
  re-coupling).
- **Edges**: `max_it == 0` → the input is returned unchanged;
  `tol == 0` → runs exactly `max_it` iterations (no early stop).
- **Divergence** (`err > prev`): keep the old stop rule, **fix the
  old bug** — old OB stopped *after* assigning the worse iterate
  and returned it; `"stop_best"` returns the argmin-error iterate;
  `"raise"` aborts with a taught error carrying the error series;
  `"ignore"` iterates on to `max_it`. (`"stop_best"` ≡ d5_3's
  `"rollback"`, reconciliation §10.7.3; the old divergence test's
  expectation changes — pinned for the 2.7/2.8 cutover tests.)
- **The factory form**: evaluated on the current iterate each
  iteration; pure given frozen captures (law 1 holds because each
  produced transform is deterministic). Signature: transform form
  takes the wrapped endo signature (endo required at construction);
  factory form is signature-polymorphic until first use (spec
  completion 5), with per-iteration endo checks on each produced
  transform.
- **Info**: `errors` (the full series), `iterations`,
  `stopped_by` (`"tol" | "max_it" | "divergence"`) and
  `returned_iteration` in `extra`, one labeled child per iteration.
  `cost()` = `max_it ×` per-iteration cost, `upper_bound=True`.
- `on_iteration` is a host observer (`TransformProgress` payload);
  may never influence results.

---

### fr.transforms.relative_l2, fr.transforms.assert_idempotent

Module-level functions: the default stopping norm (relocated from
the old field surface, respecting D1's eviction of norms from
fields), and idempotency's third consumer (§10.2, §10.3).

| Aspect | Value |
|--------|-------|
| Kind | plain functions |
| Pytree | n/a (host-level; jit-compatible internals) |
| Task | 2.8 |
| Design refs | §10.2 (the norm); §10.3 (idempotency consumers); d5_1 §2 |

```python
def relative_l2(a: State, b: State) -> float:                  # 2.8
    """The old norm_of_diff formula, relocated:
    2 * ||a - b||_2 / (||a||_2 + ||b||_2), volume-weighted over
    the PROGNOSTIC components. Relative, dimensionless,
    parameter-free — which is exactly why it can be the
    explicit-kwarg default and the energy norm cannot."""
    ...


def assert_idempotent(                                         # 2.8
    transform: fr.StateTransform,
    state,
    *,
    norm: Callable[[State, State], float] = relative_l2,
    tol: float = 1e-9,
) -> None:
    """Validation check: norm(T(T(s)), T(s)) <= tol, else a taught
    AssertionError. Generic — one of the three declared-idempotency
    consumers (validates the declaration on a probe state)."""
    ...
```

Notes: `norm=nh.diagnostics.energy_norm(model)` is one kwarg away
for dimensional stratified runs; a model-supplied *default* norm is
rejected (recorded). `relative_l2` returns a host float (FixedPoint
is host-level in it-1; the traced-variant's 0-d-array form is the
designed-for's concern).

---

### Propagator

Run-as-transform: the forward/backward Tier-2 building block
(§10.5).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final; Tier 2 |
| Pytree | host object, not a pytree (owns a Model) |
| Task | 2.8 |
| Design refs | §10.5, §10.3 laws 1/3; d5_3 §3; §6.3 (advance), §6.5 (reset/set_state), V-S1 (sign-agnostic runlen) |

```python
"""Model propagation as a StateTransform."""
from __future__ import annotations

import fridom.framework2 as fr


class Propagator(fr.StateTransform):
    """Advance an internally-owned model by a fixed extent."""

    def __init__(                                              # 2.8
        self,
        model,
        *,
        steps: int | None = None,
        runlen: float | None = None,       # exactly one of the two
        filter=None,                       # fr.terms predicate | None
        updates: Mapping | None = None,    # assembly-time variant updates
        name: str | None = None,
        on_progress: Callable[[TransformProgress], None] | None = None,
    ) -> None:
        """model is an assembly spec (law 3): the constructor
        builds its own internal variant; passing a pre-built
        variant is ownership transfer."""
        ...

    @property
    def steps(self) -> int:                                    # 2.8
        """The resolved step count."""
        ...
```

Semantics, invariants, error behavior:

- **Call semantics are law 1's, verbatim**: `reset(); set_state;
  advance(steps); read PROGNOSTIC` — built on §6.3's IO-free
  `advance` primitive; the final clock goes to info
  (`elapsed_model_time`), never into the State.
- **Exactly one of `steps=`/`runlen=`**; `runlen` is an unsigned
  duration reduced sign-agnostically per V-S1 (direction from the
  internal model's dt sign). **Backward propagation is a
  parameter, not a mode**: build the internal variant with
  `updates={fr.params.TIME_STEP: -dt}` (and reversed Ramps where
  applicable — OB does exactly this); there is no `backward=` flag.
- `filter=`/`updates=` thread to the internal `model.variant(...)`
  call (spec completion 3; default filter is the identity,
  `term_filter=None`). Variant *mechanics* — snapshot semantics,
  the ⊆ verify lemma (variants never `GridFrozenError`), fingerprint
  — are [`model.md`](model.md)'s; what this class relies on is the
  lemma plus the copy-free parent↔variant state exchange (identical
  State treedef).
- Internal model: `io=()`, named `"{name or 'Propagator'}/internal"`;
  frozen config snapshotted at construction (V-C8 staleness note
  applies). Signature: endo on the twin's full PROGNOSTIC table
  (`StateSignature.of_prognostic`), default `rest="zero"`.
- `cost()` = `steps`; info: `model_steps`, `elapsed_model_time`.

---

### TimeAverage

The GeostrophicTimeAverage successor: nested flat time-means over a
filtered twin (§10.5) — homed `fr.transforms`, aliased
`nh.transforms.TimeAverage`.

| Aspect | Value |
|--------|-------|
| Kind | concrete, final; Tier 2 |
| Pytree | host object, not a pytree |
| Task | 2.8 |
| Design refs | §10.5; d5_3 §2; §10.3 law 3 (+ V-C6 amendment); 02_rules (S6 accumulation idiom) |

```python
"""Nested time-averaging as a StateTransform."""
from __future__ import annotations

import fridom.framework2 as fr


class TimeAverage(fr.StateTransform):
    """Nested flat time-means over an internally-owned filtered
    twin (the balanced-state estimator)."""

    def __init__(                                              # 2.8
        self,
        model,
        *,
        period: float | None = None,   # None -> inertial, 2*pi/f0
        n_ave: int = 2,                # nested passes
        equidistant: bool = True,      # descending staggered periods
        backward_forward: bool = False,
        filter=fr.terms.linear,        # the twin's term filter
        on_progress: Callable[[TransformProgress], None] | None = None,
    ) -> None:
        """model is an assembly spec: owns
        model.variant(term_filter=filter) (law 3)."""
        ...
```

Semantics, invariants, error behavior (the old algorithm, ported
faithfully — d5_3 archaeology):

- **One pass** of period `T_i`: `n = ceil(T_i / dt)` steps; the
  **flat, endpoint-inclusive mean** `(Σ_{k=0..n} z_k)/(n+1)`,
  sampled every step; the pass output seeds the next pass.
- **`n_ave` nested passes**; `equidistant=True` gives **descending
  periods** `linspace(T/2, T, n_ave+1)[1:][::-1]` — staggering the
  sinc-filter zeros widens the stopband; `equidistant=False` uses
  the full period every pass.
- **`backward_forward=True`** runs a mirrored `dt < 0` pass after
  each forward pass (symmetrizes the filter). Pass mechanics:
  `update_parameters({fr.params.TIME_STEP: ±dt})` (auto-rewarm
  covers the sign flip) → `reset()` → `set_state` → accumulate →
  normalize.
- **`period=None` reads `coriolis.f0`** through `model.parameters`
  at construction and uses the inertial period `2π/f0` — hinted
  error if the composition provides no `coriolis.f0` (implements
  what the old docstring only promised; the old
  `max_period=None` crash dissolves). The provides-implies-constancy
  rule (02_rules) is what makes the scalar read legitimate.
- **`filter=fr.terms.linear` default**; the user's inviscid
  averaging is `filter=fr.terms.linear &
  ~fr.terms.owned_by(fr.closures.ClosureBase)`. **Parity delta
  pinned for cutover (signed)**: the old "linear" twin kept
  Smagorinsky — nonlinear! — running; `fr.terms.linear` correctly
  drops it (2.7 cutover tests pin the delta).
- **Accumulation, it-1: the host `advance(1)` loop** — identical to
  the old per-step cost, zero machinery (`chunk(1)` is lazily
  compiled anyway). **Upgrade path, constructor-agnostic**:
  (i) a **carry-resident accumulator module** added to the twin —
  a *declaration* change, hence per amended law 3 (V-C6) a **fresh
  assembly on the parent's frozen grid, not a variant**; the
  accumulator's AUX/DIAG declarations must **reuse spaces already
  in the parent's state-space set** (or the satisfiability-relaxed
  verify covers them); the running sum is written per the 02_rules
  **S6 accumulation idiom** (DIAGNOSTIC-kind stage — spec
  completion 6 records that this supersedes §10.5's `self_update`
  parenthetical), one `advance(n)` per pass, fully fused — **this
  upgrade is why signature ≠ treedef matters** (the augmented twin
  still signature-matches); (ii) the 2.6 capture stream.
- Signature: endo on the twin's PROGNOSTIC table; info: per-pass
  children (`model_steps`, `elapsed_model_time`); `cost()` sums the
  pass step counts.

---

### OptimalBalance

The OB preset: two ramped variants + projection exchange, iterated
by a FixedPoint factory (§10.5) — homed `fr.transforms`
(shallowwater is a first-class consumer), aliased
`nh.transforms.OptimalBalance`.

| Aspect | Value |
|--------|-------|
| Kind | concrete, final; Tier 2 |
| Pytree | host object, not a pytree (owns two internal models) |
| Task | 2.8 |
| Design refs | §10.5, §10.4 (variant updates=), §10.7.1/.3; d5_3 §3; 02_rules (Ramp signed endpoints, V-S2); the default-projector hook (V-S4) |

```python
"""Optimal balance as a StateTransform preset."""
from __future__ import annotations

import fridom.framework2 as fr


class OptimalBalance(fr.StateTransform):
    """Ramped balance iteration: forward @ base @ backward, fixed-
    point iterated with per-iteration base-point exchange."""

    def __init__(                                              # 2.8
        self,
        model,
        *,
        ramp_period: float,
        base: fr.StateTransform | None = None,
        ramp: Literal["exp", "pow", "cos", "lin"] = "exp",
        max_it: int = 3,
        tol: float = 1e-9,
        update_base_point: bool = True,
        filter=None,                    # threads to BOTH variants
        backward_filter=None,           # the mset_backwards successor
        on_progress: Callable[[TransformProgress], None] | None = None,
        on_iteration: Callable[[TransformProgress], None] | None = None,
    ) -> None:
        """model is an assembly spec; base=None resolves through
        the core-module default-projector hook (V-S4)."""
        ...

    # ================================================================
    #  Public pieces (§10.5: reusable, recomposable, testable)
    # ================================================================

    @property
    def base(self) -> fr.StateTransform:                       # 2.8
        """The base-point projector (Tier 1; resolved or given)."""
        ...

    @property
    def backward_to_linear(self) -> Propagator:                # 2.8
        """The backward ramp leg (owns the bwd internal model)."""
        ...

    @property
    def forward_to_nonlinear(self) -> Propagator:              # 2.8
        """The forward ramp leg (owns the fwd internal model)."""
        ...

    @property
    def ramp_cycle(self) -> fr.StateTransform:                 # 2.8
        """forward_to_nonlinear @ base @ backward_to_linear —
        the reusable composed cycle."""
        ...
```

Semantics, invariants, error behavior:

- **Two owned variants** with Ramp-valued
  `fr.params.SCALING_ROSSBY` (`"scaling.rossby"`, §10.7.4):
  forward `fr.Ramp(0, Ro, ...)` over `t ∈ (0, T)` (up), backward
  `fr.Ramp(Ro, 0, ...)` over `t ∈ (−T, 0)` (down — **Ramp
  endpoints are signed times**, V-S2: a backward leg runs over
  negative clock times; a naive `[0, T]` endpoint swap clips to a
  constant) and sign-flipped `fr.params.TIME_STEP` on the backward
  one — both are `variant(updates=)`'s assembly-time value-spec
  changes (S4; mechanics in [`model.md`](model.md)). The `ramp=`
  name selects the old shape family (exp/pow/cos/lin) as the Ramp
  curve.
- **Behavior delta (signed, pinned for cutover)**: old
  piecewise-constant θ = n/N ramping becomes continuous stage-time
  Ramp evaluation — tolerance-based cutover tests, expected slight
  improvement.
- **`filter=` threads to both variants** (the user's requirement);
  **`backward_filter=` is the `mset_backwards` successor** —
  recommended content `~fr.terms.owned_by(fr.closures.ClosureBase)
  & ~fr.terms.implicit` (drop dissipation rather than sign-flip it;
  backward diffusion is ill-posed — the old "negative viscosity"
  escape was, grep-verified, never used anywhere).
- **The FixedPoint-factory exchange**: the iterated map is built
  per iteration from the current iterate `s` —
  `Shift(z_base) @ (Identity() - base) @ ramp_cycle` with
  `z_base = base(s)` refreshed each iteration when
  `update_base_point=True` (the factory form is exactly what
  absorbs OBTA); `update_base_point=False` pins `z_base` from the
  initial input. `max_it`/`tol`/`on_iteration` and the divergence
  policy pass to the internal `FixedPoint`
  (`on_divergence="stop_best"` — the signed behavior delta: old OB
  returned the *worse* iterate on divergence).
- **`base=None`** resolves through the core-module-supplied
  default-projector hook (nonhydro: `VorticalProjection(model)`);
  no hook → hinted error requiring explicit `base=`.
- **Sequential-use rule (law-3 refinement, documented)**: the
  exposed leg Propagators and `ramp_cycle` share the *preset's own*
  two internal models — exclusive ownership is per *preset*;
  sequential, reset-prefixed use is the sanctioned exception within
  one owner; they must never be driven concurrently. The secondary
  maps (to-linear/to-nonlinear with reversed ramps) **retarget the
  same two models' Ramp endpoint leaves** via `update_parameters`
  (endpoints are leaves — no recompile), same sequential rule.
- **Details via `call_with_info`** (§10.7.1 — d5_3's `ob.trace`
  attribute is superseded; it was exactly the rejected `last_info`):
  `errors`, `iterations`, `stopped_by`, `returned_iteration`,
  per-iteration children containing per-leg `model_steps`.
- Memory: two carries (plus the user's) — the known Tier-2 cost;
  the shared-internal-model alternative conflicts with independent
  leg Propagators (residual, see Open questions). `cost()` =
  `max_it × (2 × ramp steps + base)`, `upper_bound=True`.

---

### VorticalProjection, WaveProjection, DivergenceProjection

Package-specific Tier-1 projections: thin wrappers over the 2.7
`em.projector` machinery with dual source constructors (§10.5).
Specced once, `nh` spelling; `sw` mirrors.

| Aspect | Value |
|--------|-------|
| Kind | concrete, final; Tier 1 |
| Pytree | `jaxify`, eigenvector captures (`_q`, `_p` data) dynamic; grid/spaces/signature static (identity-hashed) |
| Task | 2.8 — **requires the 2.7 eigenmode surface** (`em.q`/`em.p`/`em.projector`, `from_model`) |
| Design refs | §10.5, §10.3 law 2 (rest policy); d5_3 §1; D2.4 (from_model validation, Ramp `at_time=` rule) |

```python
"""Eigenmode-family projections (model package, e.g. nh.transforms)."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@partial(fr.utils.jaxify, dynamic=("_q", "_p"))
class VorticalProjection(fr.StateTransform):
    """Tier-1 projection onto the vortical eigenmode family."""

    def __init__(                                              # 2.8
        self,
        source,                        # model | em (dual constructors)
        *,
        discrete: bool = True,
        at_time: float | None = None,
        rest: Literal["zero", "pass"] = "zero",
    ) -> None:
        """source: an assembled model (the from_model path,
        inheriting all D2.4 validation incl. the Ramp at_time=
        rule) or an Eigenmodes object (the explicit-params form;
        discrete/at_time are then illegal — the em fixed them)."""
        ...


class WaveProjection(fr.StateTransform):
    """Tier-1 projection onto the wave families: P(+1) + P(-1),
    literally — constructed as the algebra Sum of the two
    eigenmode projectors (same dual constructor surface)."""
    ...                                                        # 2.8


class DivergenceProjection(fr.StateTransform):
    """Tier-1 projection onto the divergence family (same dual
    constructor surface)."""
    ...                                                        # 2.8
```

Semantics, invariants, error behavior:

- **Dual source constructors**: `(model, *, discrete=True,
  at_time=None)` goes through the eigenmode `from_model` path —
  inheriting all D2.4 validation, including the Ramp `at_time=`
  rule (a Ramp-valued `n2`/`f0` without `at_time=` is a
  `TimeDependentParameterError`: an eigenmode set is a fixed-time
  snapshot) and the provides-implies-constancy check; `(em)` is the
  explicit-params form — the true dependency is
  `(grid, f0, n2, dsqr, discrete)` and the model form only adds
  consistency + FieldTable binding. The two forms interoperate in
  the algebra because **signature ≠ treedef** (S6).
- **Mapped subset + rest**: the projection declares its family
  components (nonhydro: `u, v, w, b`) as the signature's mapped
  subset; extra PROGNOSTIC components (a `dye` tracer) follow
  `rest` — default `"zero"` (the eigenspace has no tracer
  direction; `state − P(state)` carries the full tracer),
  `"pass"` as the explicit opt-out.
- **`idempotent=True`, declared** — enabling `P.complement`
  (`Identity() - P`, the wave-plus-divergence residual) and the
  `P @ P` lint hint; `assert_idempotent` validates the declaration.
- **Nyquist lives in `em.q`/`em.p` materialization** (grid-level
  spectral mask — 2.7's obligation, pointer only): never re-zeroed
  in transforms or user code; documented caveat: the completeness
  identity `P_vort + P_wave + P_div = I` holds on the
  Nyquist-free subspace.
- Tier-1 mechanics: `traceable=True`, jit/vmap-able
  (`jax.jit(jax.vmap(P))(batch)` — the IC-ensemble path), zero
  `cost()`, `TransformInfo.EMPTY` info.
- The old `GeostrophicSpectral`/`WaveSpectral` classes and their
  mset reads dissolve into this surface (d5_3 archaeology);
  `WaveSpectral ≡ P(+1) + P(−1)` is preserved as the literal
  implementation — the algebra at work.

---

### fr.linearize

Predicate sugar, top-level (§10.4).

| Aspect | Value |
|--------|-------|
| Kind | plain function |
| Pytree | n/a |
| Task | 2.8 |
| Design refs | §10.4; d5_2 §6 |

```python
def linearize(model) -> fr.Model:                              # 2.8
    """model.variant(term_filter=fr.terms.linear), named
    "{parent}/linear". Returns a plain fr.Model (a full lifecycle
    citizen), not a transform."""
    ...
```

Notes: consumes the declared `linear` tag — which is why the JVP
linear-tag debug lint is prioritized at 2.5 (residual pointer);
V-S3's rule (a scheme's linear background-advection piece must be a
separate `linear=True` term or `linearize` drops it) is
declarations.md's to enforce. Variant mechanics (snapshot, ⊆ verify
lemma, fingerprint, coverage-lint downgrade) live in
[`model.md`](model.md).

---

### NNMD — descoped (no class)

**NNMD is descoped at sign-off**: the old `nnmd.py` is not ported at
cutover (2.7), no `NNMD` class is specced, and the future rewrite —
its own design exercise — will *not* contain a model propagator.
The d5_3 §4 archaeology (the eigenpair table, the
`N(z)`-via-variant-tendency mechanism, the quadraticity caveat and
its possible lint, the dropped never-implemented `enable_dealiasing`
flag) is retained in the research report for that rewrite's benefit;
`em.omega_field` is deferred with it (S3). The
`model.variant(term_filter=...).tendency(z)` spelling of a
restricted tendency remains generally available regardless
([`model.md`](model.md) owns `model.tendency`).

---

## Open questions

Genuinely unresolved residuals only (§10.8, 07_open_threads item 8);
the signed behavior deltas (`rest="zero"`,
`on_divergence="stop_best"`, the TimeAverage Smagorinsky parity
delta) are **not** open — they stand, pinned for the 2.7 cutover
tests.

1. **Backward-dissipation warning** (decide at the OB port,
   2.7/2.8): the old code kept closures active on backward legs;
   should construction warn when closures survive onto a
   sign-flipped (`TIME_STEP < 0`) variant — i.e. when
   `backward_filter` (or `filter`) leaves `ClosureBase` terms in a
   backward twin?
2. **The blessed Ramp endpoint-reversal spelling**: *resolved and
   shipped* — `Ramp.reversed()` (pure window reflection,
   `t0 -> -(t0 + period)`, values and curve unchanged), satisfying
   the V-S2 retrace law for every curve. Residual: confirm the
   construction against the ported OB legs in the tolerance-based
   cutover tests.
3. **OB memory** (two carries + the user's): the
   shared-internal-model alternative halves it but conflicts with
   exposing independent leg Propagators — revisit if it bites.
4. **JVP linear-tag lint priority** (2.5): `fr.linearize` now
   consumes the `linear` tag; the lint that keeps the tag honest is
   prioritized there (owned by the 2.5 row; pointer here because
   this cluster is the consumer).
5. **The `"raise"` divergence error type**: name and registry entry
   for the taught error FixedPoint's `on_divergence="raise"` throws
   (registers in model.md's error-type registry alongside the two
   entries above).
6. **Designed-for / deferred** (parked, no consumer — listed so the
   surface doesn't preclude them): `OnComponents` adapter (subsumed
   for the common case by the mapped-subset + `rest` ruling),
   `Transform.resync()` (the frozen-config refresh),
   `T.with_parameters(...)`, `fr.terms.where(fn, token=)`, stage
   filtering in variants, batched Tier-2 ensembles, the traced
   `lax.while_loop` FixedPoint, field-valued algebra coefficients,
   `em.omega_field` (deferred with the NNMD rewrite).
