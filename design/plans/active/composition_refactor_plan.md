---
status: active
date: 2026-07-09
---

# Composition refactor — the realized-map category & one composition core (design + staged plan)

**Status: design + staged plan, 2026-07-08 (awaiting owner sign-off).**
Extends [`../../decisions/symbol_stack_design.md`](../../decisions/symbol_stack_design.md) (the realized-
map algebra) with the *implementation* decision: build the realized-map
category **together with** a consolidation of the composition machinery
that is currently duplicated across the operator, symbol, block-symbol,
and state-transform algebras. Backed by a three-way deep read of
`grid/operators/base.py`, `symbol.py`/`block_symbol.py`/`banded.py`, and
`transforms/`, plus the design-note archaeology.

## 1. The motivation — measured duplication

- **Normalization skeleton** (flatten nested chains/sums, elide the
  neutral, drop/absorb `Zero`, collapse singletons, scale/negate, power)
  is implemented **4×**: operator `_compose`/`_sum` (`base.py:1669`,
  `:1697`), `Symbol` dunders (`symbol.py:220-322`),
  `BlockMatrix.__matmul__` (`composed.py:295`), `StateTransform`
  `make_compose`/`make_sum` (`transforms/algebra.py:330`,`:355`). Near
  line-for-line, zero code reuse.
- **Tag-typing** ("validate composability on matching domain/codomain
  tags") is implemented **3×**: `Symbol._compose_spaces`/`_union_space`
  (`symbol.py:487-557`), `BlockSymbol._spaces_match`
  (`block_symbol.py:289`), operator `resolve_codomain` /
  `OperatorSum.codomain` (`base.py:1303`,`:1014`).
- The **`eigenvalues` bridge** (`base.py:1610-1620`, `:1043-1049`,
  `:1104-1110`) already drives the symbol algebra by replaying `@`/`+`
  one level down onto data — living proof the pattern nests.

## 2. Archaeology verdict (why this is new, defensible work)

No note ever proposed or rejected a *shared composition core*. "Mirror
the operator algebra" is a stated **goal** (`08_state_transforms.md:14`,
`01_concepts.md:772`) but was always built as a **separate**
implementation. The one recorded rejection is **`Symbol`-as-`Operator`**
(`operators_base.md:891-895`), on the **static-structure-vs-dynamic-data**
axis — the same axis any unification must respect. And `d5_1` leaned on
"a small pinned surface as the over-engineering guard." So: green light
in spirit, with two rails — **don't over-abstract**, and **respect
static-vs-dynamic**.

## 3. The design — shared leaf utilities + three parallel algebras

**Reuse lives at the leaf (pure functions), not in a shared base class.**

### Layer 0 — shared pure utilities
- **`compose_spaces(codomain_inner, domain_outer) -> (domain, codomain)`
  and `union_spaces(a, b)`** — the one factor-wise tag validator
  (`Constant`/`Identity` passthrough, identity on the shared index).
  A *resolver* signature (operator/transform `codomain(domain)`)
  evaluates to a concrete space first, then calls the same validator.
  Replaces `_compose_spaces`+`_union_space`+`_spaces_match`+the equality
  checks in `OperatorSum.codomain`/`resolve_codomain`.
- **`normalize_chain` / `normalize_sum`** — flatten/elide-neutral/
  collapse-singleton, parameterized by node-constructor + neutral
  predicate + optional interner. Consumed by every algebra's `@`/`+`.

### Layer 1 — three parallel algebras, each calling Layer 0
- **Operators** — static, interned, grid-free; resolver signatures; axis
  binding, `requirements`, the `eigenvalues` bridge, the field
  application/sync path. *Unchanged in kind.*
- **RealizedMaps (the new consolidation)** — the fixed-tag, data-carrying
  `ScalarField -> ScalarField` maps: `Symbol`, `BlockSymbol`, `Banded`,
  a bound transform, and `SpectralSolve`. See §4.
- **StateTransforms** — info/cost/tier/model-binding, `State`
  signatures. Keeps its nodes; *may* adopt Layer 0 (§5, low priority).

**What we do NOT do:** force the three into one class hierarchy with a
static/dynamic flag. Operators' three static invariants — interning,
zero-leaf `jaxify`, identity `__eq__`/`__hash__` — are rejected by the
data-carrying layers (already selectively broken by `ScaledOperator`'s
field coeff and by `Symbol`). Merging would trade clean specializations
for a leaky per-node-flag abstraction — the over-abstraction the notes
warn against. Reuse is Layer-0 functions, not inheritance.

## 4. The `RealizedMap` category

A protocol/mixin (a `@final` type like `Symbol` may still *inherit* a
mixin — `@final` only forbids subclassing it) with:
`domain`/`codomain` (fixed, already-resolved tags) · `__call__(field)` ·
`@` `+` `*` `inverse` `conj` · `jaxify(dynamic=<data>)` (static tags aux,
dynamic data leaves).

- **`Symbol`** — the complete reference today (has all of it).
- **`BlockSymbol`** — has the matrix algebra; needs `__call__`/`inverse`.
- **`Banded`** — promote `banded.py`'s free functions to a type: fixed
  tags on a non-transformed axis, band storage,
  `__call__`=matvec, `inverse`=Thomas solve, per-mode batching. A
  diagonal `Symbol` is the bandwidth-0 case; a `Banded` on `z` composes
  with a horizontal `Symbol` on `(x,y)` by disjoint-axis tensor product;
  nests under `BlockSymbol`.
- **bound transform / `SpectralSolve`** — a **lazy `RealizedComposite`**
  applies right-to-left via `__call__` (the existing `Composite._apply`,
  `base.py:901`, is the template), with an eager `Symbol@Symbol`
  Hadamard fast path. `SpectralSolve` becomes `backward @
  symbol.inverse() @ forward` — a composed value, retired as a class.
- **Explicit materialization** (decision A): a bare `symbol @ recipe`
  raises, pointing at `recipe.eigenvalues(grid, space)`.

## 5. Staged plan

Each stage is behavior-neutral where marked and gated by the **full**
`grid/operators` + `model` + `nonhydro2`/`shallowwater2` + `transforms`
+ `validation` suite (the D′ discipline: run it before committing).

- **S0a — `compose_spaces`/`union_spaces` validator.** Extract; route
  `Symbol`/`BlockSymbol`/operator tag checks through it. Behavior-neutral;
  prerequisite for the realized-map `@` typing. *(low risk)*
- **S0b — `normalize_chain`/`normalize_sum` helper. DROPPED (2026-07-09).**
  On inspection the three flatten implementations share only "flatten
  nested same-kind + collapse singleton"; the variations dominate
  (operator: `Zero` + interning + `Identity`; `RealizedComposite`:
  symbol-fusion + adjacency-typecheck, no `Zero`; `StateTransform`:
  `Identity`, no `Zero`). A shared helper needs ~5 hooks over ~5 lines
  each — the over-abstraction the "small pinned surface" guard rejects.
  The load-bearing shared piece was the tag validator (S0a). Not worth it.
- **S1 — `RealizedMap` protocol + `Symbol`/`BlockSymbol` adopt it.**
  Add the missing `BlockSymbol.__call__`/`inverse`; both use S0a for
  typing. Behavior-neutral for existing use; adds the shared interface.
  Depends S0a. *(medium)*
- **S2 — `RealizedComposite` + `SpectralSolve`-as-composition +
  explicit-materialization guard.** Cross-type `@` (transform @ symbol)
  via the protocol; rewire the D′ `SpectralPressureSolver` to `backward
  @ symbol.inverse() @ forward`. Depends S1. Guard: pressure/poisson
  equivalence bitwise/tolerance. *(medium — touches the pressure path)*
- **S3 — `Banded` as a `RealizedMap`.** Promote `banded.py`; band
  storage + matvec/solve + batching; the mixed Fourier×(FD/Cheb)
  substrate. Depends S1–S2. *(medium)*
- **S4 — Phase I proper.** `BlockSymbol`-of-`Banded` nesting + mixed-
  representation variable-coefficient / wall-bounded eigenmodes, on the
  consolidated realized-map layer. Depends S3. *(large)*
- **S5 (optional) — StateTransform adopts Layer 0.** `make_compose`/
  `make_sum` call `normalize_*`; its signature check a State-level
  `compose_spaces`. Behavior-neutral; low priority.

**This subsumes** the roadmap's standalone "reframe `SpectralSolve`"
follow-up (→ S2) and **Phase I** (→ S3–S4): doing them on the
consolidated realized-map layer is strictly less work than bolting
`Banded` onto today's ad-hoc `Symbol`/`BlockSymbol`/`banded.py`.

## 6. Risks & guardrails

- **Blast radius:** S0–S2 touch the `@final` `Symbol` + `base.py` that
  the whole eigenmode/pressure path rides. Same full-regression guard as
  D′; stop-and-report if a stage can't stay green.
- **Keep operators static/interned** — Layer 0 must not impose dynamic
  leaves or value-equality on the operator nodes; it's pure functions the
  operator layer *calls*, not a base it *inherits*.
- **`@final` preserved** — `RealizedMap` is a protocol/mixin, so `Symbol`
  stays `@final`.
- **Behavior-neutral stages first** (S0, and the adopt-only half of S1)
  bank the de-duplication before any new capability, so a regression is
  attributable.
