---
status: active
date: 2026-07-08
---

# Phase-2 reconciliation — grid-layer follow-up work items

Grid-layer follow-ups discovered while reconciling the Phase-2 model
design ([`model/`](../../specs/model/00_overview.md), decisions D1–D5) against the
merged Phase-1 grid implementation (2026-07-08). This is the analogue,
in the opposite direction, of
[`../done/phase1_findings.md`](../done/phase1_findings.md): where that file fed
implementation findings back into the design, this file feeds design
demands back to the grid implementers.

Severity legend: items marked **2.2-blocking** gate the model-layer
implementation start (ROADMAP 2.2); **correctness** items produce
wrong results or wrong sizing if hit; **perf** and **ergonomics**
items do not gate 2.2.

Status (2026-07-08): **items 1–8 landed** — items 1–7 in Phase-2
wave 1 ([`../done/phase2_implementation_plan.md`](../done/phase2_implementation_plan.md);
dev `040f5ad`), item 8 (ROADMAP task 1.8, sync-strategy redo) merged
into dev 2026-07-07; nothing gates 2.2. Item 8's merge also resolves
10a (the `store = pad + sync` re-trace: `store` is now pad-only).
Items 9–10 remain open (ergonomics); item 11 (BC-free boundaries)
is signed and implemented on a separate branch, not yet in dev.

## Work items

1. **`negotiate` combined-halo semantics** — **DONE (2026-07-07,
   with task 1.8)**. `tendency=` and `halo=` combine as merge_max in
   `_negotiated_halo`; the `Grid.negotiate` docstring states the
   same rule; regression test in
   `tests/framework2/grid/decomposition/test_negotiate.py`
   (`test_tendency_and_halo_combine_as_merge_max`).

2. **`merge_overrides` facade** — 2.2-blocking, small. The facade is
   a `NotImplementedError` stub, but `OperatorRegistry.merge`
   (`operators/registry.py:289-324`) already implements the full
   merge contract — only the swap is missing: call
   `self._dispatch.merge(overrides)` and swap the held instance, with
   a pre-freeze guard and a duplicate-resolved-key error naming both
   modules. Design ref:
   [`../../specs/grid/classes/grid.md`](../../specs/grid/classes/grid.md) registry placement /
   "Merge call site" resolution.

3. **`("declared_space", mesh)` resolver rows** — 2.2-blocking. The
   registry's `_normalize_key` structurally rejects mesh-keyed
   entries — it requires a `FunctionSpace`/`TensorProductSpace`
   second element (`operators/registry.py:410-420`). Either widen
   `DispatchKey` to admit mesh-keyed rows or add a dedicated resolver
   table; enforce "never module-mergeable" there. Design ref: model
   D1.2.

4. **`trace_halo` name-keyed states** — 2.2-blocking. `trace_halo`
   builds anonymous positional components `c0, c1, ...`
   (`decomposition/halo.py:647-688`) although `VectorTracer` already
   accepts a `Mapping` (`halo.py:530-644`) — the entry point just
   never passes names. Accept `Mapping[str, SpaceLike]`; positional
   naming blocks name-addressed model tendencies.

5. **`freeze()` fingerprint + verify path + `GridFrozenError`** —
   needed by 2.3. `freeze()` is a bare flag (`grid.py:295-297`) and a
   frozen `negotiate` raises bare `RuntimeError` (`grid.py:283-286`)
   — expected, since the fingerprint/verify amendment postdates the
   implementation start. Record the negotiation fingerprint
   (state-space set, merged override keys, `HaloSpec`, layout
   vocabulary) at `freeze()`; implement the demand-satisfaction
   verify (subset / less-or-equal, not equality) with
   ConstantSpace-broadcast adoption; raise `GridFrozenError` from
   both the verify failure and the post-freeze mutators. Design ref:
   [`../../specs/grid/classes/grid.md`](../../specs/grid/classes/grid.md) "Merge call site" resolution
   (model D4/D5).

6. **`VectorField.add` + metadata re-attachment in
   `replace`/`map`/`add`** — 2.2-blocking. `add()` is unimplemented;
   `replace`/`map` insert fresh components without re-attaching
   metadata. *(The perf flag this item carried is discharged by task
   1.8: `store` no longer syncs, so tendency accumulation is
   exchange-free with no storage-frame special-casing.)* Design ref:
   [`../../specs/grid/classes/fields.md`](../../specs/grid/classes/fields.md) amendment (2026-07-08).

7. **Annotation-exempt metadata equality in jaxify** — 2.2-blocking,
   correctness. Exclude the metadata annotation category from aux
   equality so treedefs are metadata-insensitive: fixes scan-carry
   breaks on metadata changes and the keys-only-hash silent
   recompiles. Design ref: [`../../specs/grid/classes/fields.md`](../../specs/grid/classes/fields.md)
   amendment (2026-07-08);
   [`../done/phase1_findings.md`](../done/phase1_findings.md) finding 2 (widened).

8. **Sync-strategy redo** — **DONE (2026-07-07; ROADMAP task 1.8,
   branch `framework2-sync-redo`)**. Implemented as decided, with
   three implementation findings recorded in the decision record
   (memoized consumption syncs, periodicity-gated kernel claims,
   width-independence + shardability cap); stage log in
   [`../done/sync_redo_plan.md`](../done/sync_redo_plan.md). Originally:
   Consumption-side sync with trace-time halo-validity depth: fields
   carry a valid-halo-depth as a static trace-time attribute;
   operator application syncs iff input depth < requirement; `store`
   stops syncing. Full mechanism recorded in the
   [`../../specs/grid/classes/decomposition.md`](../../specs/grid/classes/decomposition.md#open-questions)
   entry (now the decision record). Not blocking 2.2–2.3
   (results-neutral swap); should land before performance-sensitive
   multi-device work (2.7 benchmarks, 3.3). Subsumes item 6's
   storage-frame-`add` flag: once `store` stops syncing, tendency
   accumulation is sync-free with no special-casing. Before 1.8, every operator application synced via
   `_finalize` (`operators/base.py:1317-1365`), every field `+`/`-`
   pays `store = pad + sync` (`fields/storage.py:201-213`,
   `scalar_field.py:711-736`), pointwise products on sharded nodal
   spaces exchange too, and `OperatorSum._apply` syncs per term plus
   per pairwise addition (`base.py:982-988`) — only
   `SeparableComposite` elides (`base.py:889-910`). The mechanism
   cuts this to roughly one exchange per state component per step
   and is results-neutral by construction (syncs only rewrite ghost
   cells, never true-shape data). The `HaloTracer` accounting
   (`halo.py:340-373`) and `HaloSpec` grow/merge_max arithmetic it
   consumes are already built and tested.

9. **`fr.grid.cartesian.Grid` constructor** — ergonomics. Still an
   empty stub; known from
   [`../done/phase1_findings.md`](../done/phase1_findings.md).

10a. **Eager field creation re-traces per call under multi-device**
    — perf, surfaced by wave 4.2 (2026-07-08). Each
    `grid.create_field` (the `store = pad + sync` path) traces two
    small programs anew on every call when more than one device is
    present — the sync closure is apparently not shared across
    calls. The model layer works around it by caching its constant
    tendency template; a grid-side fix belongs with the 1.8 sync
    rework (which replaces this code path anyway). **RESOLVED** by
    the task-1.8 merge (2026-07-07): `store` is pad-only, so
    `create_field` no longer traces a sync closure per call; the
    model's template cache is now a redundant (harmless) belt.

10. **Teaching shims + small gaps** — ergonomics.
    `ImmutableStateError` on `.data` assignment
    ([`../../specs/grid/classes/fields.md`](../../specs/grid/classes/fields.md) D1.5),
    `MissingComponentError` / `_component` hint path, and
    `ScalarField.to` metadata preservation on the converting path.
    The [`../done/phase1_findings.md`](../done/phase1_findings.md) API-gap backlog
    (re-exports, BC-nodal operator rows, coefficient-space products,
    `.item()`, Chebyshev geometry accessors) remains open and is not
    repeated here.

12. **Carry-resident AUXILIARY fields break jitted `scan` treedef
    stability** — **correctness, 2.2-blocking for real models**;
    surfaced INDEPENDENTLY by both wave-6 model ports (nonhydro +
    shallowwater, 2026-07-08). **RESOLVED (2026-07-08)** by the
    grid-layer root-cause fix: the ghost-cache seam now records the
    memoized exchange in an external identity-keyed
    `WeakKeyDictionary` (`operators/base.py` `_SYNC_CACHE`) instead
    of mutating `f._data`/`f._halo_valid` in place, so a
    carry-resident field's treedef is never mutated. One exchange
    per component per step is preserved (exchange-count gate
    unchanged); validity stays treedef-participating (the unsafe
    exemption was rejected — it keys the sync-placement cache).
    Research: three paths explored (treedef-exemption proven unsafe
    by counterexample; model-layer flooring viable but containment;
    grid-layer cache chosen as root-cause). Details below. `ScalarField._halo_valid` is a plain
    **static** attribute (jaxify `dynamic=("_data",)`,
    `annotation=("_metadata",)` — `_halo_valid` is neither), so it
    participates in treedef equality. The consumption-side sync's
    ghost-cache seam mutates it **in place**
    (`operators/base.py` `_memoize_sync`: `f._halo_valid =
    synced.halo_valid`). A PROGNOSTIC field is rebuilt to zero-halo
    each step so its treedef is stable; a **carry-resident
    AUXILIARY field read by a stencil** (e.g. a beta-plane
    `f(y)`, a stratification `N²(z)`, bathymetry) has its
    `_halo_valid` mutated but is never rebuilt → the `lax.scan`
    output carry's treedef differs from the input → "carry
    input/output pytree structure differ". Existing tests never
    caught it (`TracerDiffusion` has no stenciled AUX field).
    Consequence: **`BetaPlaneCoriolis` assembles but cannot
    `advance`** on nonhydro; both ports had to route static
    parameters through the scalar+`extra_halo`+`.data` path or
    pre-sync AUX fields to full halo. **Fix is NOT the naive
    treedef-exemption** (adding `_halo_valid` to the `annotation=`
    category like `_metadata`): unlike metadata, `halo_valid` drives
    sync PLACEMENT, so excluding it from the jit-cache key could
    reuse a body compiled for a different halo state. Two safe
    directions, owner's call: (a) model-layer — `step_chunk`
    canonicalizes AUXILIARY carry fields' `_halo_valid` to their
    input state at the end of each step; (b) grid-layer — the
    ghost-cache seam must not let an in-place mutation escape into
    the returned field a caller keeps. **Belongs to the task-1.8
    sync owner** (its mechanism) coordinated with the model layer.

13. **`ConstantSpace`/`Profile()` cannot broadcast in a tendency
    term** — correctness/ergonomics; both wave-6 ports (2026-07-08).
    **RESOLVED (2026-07-08)**: the `HaloTracer` product path now
    lifts a `ConstantSpace`/`Profile()` operand onto the nodal join
    the same way the eager `ScalarField` product does (shared
    `_lift_field`), and a `ConstantSpace→nodal` `.to` broadcast
    branch was added to both eager and traced paths. Halo-0,
    results-neutral; a `Profile("y")` Coriolis `f(y)` now broadcasts
    natively in a term (assembles + advances). Details below.
    In the halo trace, `constant_field * nodal_field` raises
    `SpaceMismatchError`: the eager `ScalarField` path lifts a
    `ConstantSpace` operand via `join`, but `HaloTracer`'s product
    dispatch applies the operator to un-lifted operands, and `.to`
    from `ConstantSpace`→nodal is undefined. This blocks the notes'
    **R2 "1-DOF `fr.Profile()` field broadcast in the consumer
    line"** (D1.5), so a beta-plane `f(y)` or a `Profile("z")`
    stratification must be declared on a full `Collocated()` space
    (wasteful, and it interacts with item 12). Fix: lift
    `ConstantSpace` operands in the `HaloTracer`/coefficient-space
    product path the same way the eager path does; add the
    `ConstantSpace`→nodal `.to` broadcast row.

14. **Field arithmetic rejects a traced scalar operand** —
    ergonomics/robustness; surfaced by the wave-6 cleanup
    (2026-07-08). The `ScalarField` arithmetic dunders accept only
    Python scalars, not a traced `ctx.params[...]` value, so any
    term multiplying a field by a traced parameter scalar (Rossby
    scaling `Ro·adv`, `b/dsqr`, `-N²·w`) must drop to
    `field.with_data(scalar * field.data)` — a raw-`.data` escape
    that then forces an `extra_halo` declaration. This is why the
    wave-6 Coriolis unification routes `f` through a *field*
    (`f_coriolis`) rather than a scalar param, and why the
    nonhydro `ConstantStratification`/`CenteredAdvection` and the
    shallowwater `SadournyAdvection` still carry `extra_halo`
    bypasses. Fix: let the field arithmetic dunders (and the
    `HaloTracer` twins) accept a 0-d `jax.Array`/traced scalar
    operand (broadcast, halo-0) — the scalar analogue of the GAP-13
    `ConstantSpace` broadcast. Not blocking (the bypasses work);
    removes the last routine `extra_halo` raw-`.data` escapes from
    model terms.

11. **BC-free bounded spaces: exterior values untouchable** —
    owner-flagged design question (2026-07-07), full note in
    [`bc_free_boundaries.md`](bc_free_boundaries.md). Replace the
    BC-free extrapolation ghost fill with a legality rule: operator
    rows needing exterior values exist only on BC-structured
    spaces; explicit one-sided stencil rows are the opt-in
    replacement. Motivated by the fill's inconsistency under
    composition (the task-1.8 `d²` counterexample); unlocks bounded
    chain elision (mirror fills commute). Not blocking; decide
    before 2.2+ registers boundary-aware modules on bounded meshes.
    Subsumes the "BC-nodal operator rows" API-gap entry above.
    **Resolution proposed** in
    [`boundary_plan.md`](boundary_plan.md) (R1-R4, staged plan;
    awaiting owner sign-off).
