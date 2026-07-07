# Phase-2 reconciliation — grid-layer follow-up work items

Grid-layer follow-ups discovered while reconciling the Phase-2 model
design ([`model/`](model/00_overview.md), decisions D1–D5) against the
merged Phase-1 grid implementation (2026-07-08). This is the analogue,
in the opposite direction, of
[`phase1_findings.md`](phase1_findings.md): where that file fed
implementation findings back into the design, this file feeds design
demands back to the grid implementers.

Severity legend: items marked **2.2-blocking** gate the model-layer
implementation start (ROADMAP 2.2); **correctness** items produce
wrong results or wrong sizing if hit; **perf** and **ergonomics**
items do not gate 2.2.

## Work items

1. **`negotiate` combined-halo semantics** — 2.2-blocking,
   correctness. `tendency=` and `halo=` must combine as
   trace(tendency) `merge_max` extra_halo, not shadow: today
   `_negotiated_halo` treats `halo=` as an exclusive override and
   silently discards the trace
   (`decomposition/decomposition.py:542-560`), so a step-7 call with
   any `extra_halo` would under-provision every traced module. The
   `Grid.negotiate` docstring states a third, different precedence
   (`grid.py:262-267`) — fix it to the merge_max rule too. Design
   ref: model assembly step 7
   ([`model/classes/model.md`](model/classes/model.md)).

2. **`merge_overrides` facade** — 2.2-blocking, small. The facade is
   a `NotImplementedError` stub, but `OperatorRegistry.merge`
   (`operators/registry.py:289-324`) already implements the full
   merge contract — only the swap is missing: call
   `self._dispatch.merge(overrides)` and swap the held instance, with
   a pre-freeze guard and a duplicate-resolved-key error naming both
   modules. Design ref:
   [`classes/grid.md`](classes/grid.md) registry placement /
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
   [`classes/grid.md`](classes/grid.md) "Merge call site" resolution
   (model D4/D5).

6. **`VectorField.add` + metadata re-attachment in
   `replace`/`map`/`add`** — 2.2-blocking. `add()` is unimplemented;
   `replace`/`map` insert fresh components without re-attaching
   metadata. Also flag for the perf pass: `add` is the hottest sync
   site of the composed step (one exchange per term per component
   under the current store-syncs contract) — it deserves a
   storage-frame path when designed. Design ref:
   [`classes/fields.md`](classes/fields.md) amendment (2026-07-08).

7. **Annotation-exempt metadata equality in jaxify** — 2.2-blocking,
   correctness. Exclude the metadata annotation category from aux
   equality so treedefs are metadata-insensitive: fixes scan-carry
   breaks on metadata changes and the keys-only-hash silent
   recompiles. Design ref: [`classes/fields.md`](classes/fields.md)
   amendment (2026-07-08);
   [`phase1_findings.md`](phase1_findings.md) finding 2 (widened).

8. **Sync-strategy redo** — perf, high value; **DECIDED (owner
   sign-off, 2026-07-08; ROADMAP task 1.8)**, no longer a candidate.
   Consumption-side sync with trace-time halo-validity depth: fields
   carry a valid-halo-depth as a static trace-time attribute;
   operator application syncs iff input depth < requirement; `store`
   stops syncing. Full mechanism recorded in the
   [`classes/decomposition.md`](classes/decomposition.md#open-questions)
   entry (now the decision record). Not blocking 2.2–2.3
   (results-neutral swap); should land before performance-sensitive
   multi-device work (2.7 benchmarks, 3.3). Subsumes item 6's
   storage-frame-`add` flag: once `store` stops syncing, tendency
   accumulation is sync-free with no special-casing. Today every operator application syncs via
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
   [`phase1_findings.md`](phase1_findings.md).

10. **Teaching shims + small gaps** — ergonomics.
    `ImmutableStateError` on `.data` assignment
    ([`classes/fields.md`](classes/fields.md) D1.5),
    `MissingComponentError` / `_component` hint path, and
    `ScalarField.to` metadata preservation on the converting path.
    The [`phase1_findings.md`](phase1_findings.md) API-gap backlog
    (re-exports, BC-nodal operator rows, coefficient-space products,
    `.item()`, Chebyshev geometry accessors) remains open and is not
    repeated here.
