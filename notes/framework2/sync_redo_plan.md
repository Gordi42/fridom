# Task 1.8 implementation plan — consumption-side sync

Implements the signed decision (owner sign-off 2026-07-08): replace
the iteration-1 sync-after-every-operator placement with
**consumption-side sync with trace-time halo-validity tracking**.
Decision record: [`classes/decomposition.md`](classes/decomposition.md#open-questions)
(the per-step sync-amplification entry); work item 8 in
[`phase2_grid_followups.md`](phase2_grid_followups.md); ROADMAP task
1.8. Branch: `framework2-sync-redo` (off `dev`); the swap is
results-neutral, so the concurrently running model-layer
implementation needs no edits and no coordination beyond merge order
(§7).

## 1. What is being replaced

Today every exchange is supply-side: `_finalize` appends `Sync` after
every operator application (`operators/base.py:1317-1365`), `store =
pad + sync` runs on every true-shape field construction
(`fields/storage.py:201-230`) — so every `+`/`-`/product pays an
exchange — and `OperatorSum._apply` pays per term plus per pairwise
addition (`base.py:982-988`). Target: **roughly one exchange per
state component per step**, mechanically guaranteed to never read a
stale ghost.

## 2. Mechanism (normative)

**Validity attribute.** Each `ScalarField` carries
`_halo_valid: HaloSpec` — per-name count of *currently valid* ghost
layers (depth-remaining; the `HaloTracer`'s existing depth is the
depth-consumed dual). `VectorField` delegates per component. Zero
everywhere is the default.

**Construction.** `store` drops the sync: `pad` only, validity zero.
Ghost slots may hold pad garbage; nothing reads them un-synced (the
consumption check below is the mechanical guarantee). `pad` must
zero-fill, never leave memory uninitialized (WENO's data-dependent
weights would otherwise see NaN/inf in slots we later mark invalid —
zeros are harmless there). All `store` call sites (`_wrap`,
`create_field`, `integrate`, `spectral`, `random_fields`,
`immersed_domain`) inherit validity-zero construction unchanged.

**Consumption rule.** In the shared application templates
(`UnaryOperator.__call__` base.py:407, `BinaryOperator.__call__`
base.py:498), before `_apply`: resolve this application's per-axis
halo requirement (separable: `requirements(factor).halo` on the
resolved axis; whole-space: every non-constant name — same
resolution `HaloTracer._grown` already implements, halo.py:313-338).
If any required axis has `valid < required`, sync the operand
(`grid.sync` fills **all** axes to the negotiated width; validity :=
the negotiated `HaloSpec`). `Sync` stays internal-only; it moves
from post-kernel to pre-kernel. Requirement-0 applications
(pointwise products, `Where`, transforms on halo-0 coefficient
spaces) never sync — this alone removes the sharded-nodal-product
exchanges.

**Result validity.** `_finalize` stops syncing and instead stamps the
result: `valid_out[axis] = valid_in[axis] − r` on consumed axes,
carry-over elsewhere; binary ops take the elementwise min of operand
validities first. This is sound with **no kernel changes**:
`apply_staggered` already computes every output ghost slot whose
stencil window lies in storage and zero-fills only the unreachable
ones (`staggering.py:148-150`) — the bookkeeping just marks which
computed layers came from valid inputs. Stage B audits each kernel
family and any family that can't certify this claims **zero** output
validity (conservative = extra exchange, never wrongness).
Store-constructed results (arithmetic via `_wrap`, transforms via
`store`) are validity-zero by construction; the true-shape combine
of `_linear_combine` (the wave-4B choice) is kept — it now costs pad
only.

**Composites and sums.** `SeparableComposite`: the pre-apply check
uses the chain-sum requirement (already `_chain_requirements`), so
one sync at chain entry covers the whole chain; `_apply_factor`
stays a raw kernel seam. `OperatorSum._apply`: hoist — ensure
validity for the per-axis max over terms once, then apply the terms
to that same field; term applications then see sufficient validity
and place no syncs. Tendency accumulation (`+` on results) is
store-constructed and sync-free — this subsumes follow-ups item 6's
storage-frame-`add` flag.

**Memoization (the n-consumers case).** A consumption-triggered sync
writes the synced storage and new validity **back onto the operand
field object** (in-place ghost rewrite; semantically invisible —
ghosts only, never true-shape data). n modules consuming the same
state component then pay one exchange total: the first consumer
syncs, the rest see validity W. Trace-safety guard: skip the
write-back when the operand's `_data` is a concrete array but the
synced result is a tracer (a closure-captured field consumed inside
someone else's trace — rare; those consumers just re-sync). Fields
passed through jit/scan boundaries are re-materialized per trace by
unflatten, so per-trace objects are private and the write-back is
safe there.

**Reshard.** `redistribute` re-blocks storage; validity := 0 on the
moved axes (`_trace_reset_names` is the existing trace rule),
carry-over elsewhere.

**Pytree rule.** `_halo_valid` goes into the jaxify aux data and
**participates in treedef equality**: a cached jit trace embeds sync
placement decisions, so a cache hit with a different validity would
silently skip a needed sync — validity must key the cache.
Consequences: the validity vocabulary is small (a few `HaloSpec`
values per space), so recompile churn is bounded; `scan` demands a
validity fixed point on the carry, which the model step naturally
has (state updates are store-constructed → validity zero at every
persistent seam). If a mismatch ever surfaces, the sound fix is
flooring the carry validity down to the entry validity (claiming
less than you have is always safe) — provide an internal
`_declare_valid` for that; do not exempt validity from equality.

**Width-independence (the central invariant).** Correctness is
independent of the negotiated width W as long as W ≥ every single
application's requirement: a chain that exhausts validity mid-way
simply syncs again. W tunes the exchange count only. This is what
makes the swap results-neutral and lets negotiation cap W freely
(§3).

## 3. Negotiation and tracer changes

- `HaloTracer._trace_apply` (halo.py:340-373) stops resetting depth
  to zero after every application; it accumulates (chains sum,
  parallel branches max — the arithmetic already there) and the
  recorder observes the running maximum. The tracer thereby
  simulates the new runtime rule exactly: one arithmetic, two
  consumers.
- Negotiated width = `max(traced accumulation, per-op registry
  floor)`, optionally **capped** for shardability
  (`cells/shard ≥ width + 1` bit us in Phase 1 on small meshes):
  widths may *grow* under accumulation (a `diff∘diff` step traces 2
  where iteration 1 negotiated 1), trading halo memory for exchange
  count. Cap → runtime places a mid-chain sync automatically (§2
  width-independence). Floor = the max single-application
  requirement from the registry (`_registry_halo`).
- Fix follow-ups **item 1** in passing (same function): `tendency=`
  and `halo=` combine as `merge_max`, not shadow
  (`decomposition/decomposition.py:542-560`), and align the
  `Grid.negotiate` docstring (`grid.py:262-267`).

## 4. Work breakdown (stages = commits, each green)

**A. Validity plumbing, behavior-neutral.** `_halo_valid` on
`ScalarField`/`VectorField` (aux, treedef-participating; plumbing
constructor threading); `store` and `_finalize` keep syncing but
stamp validity = negotiated widths; debug assertions that the
bookkeeping matches the always-synced reality. Full suite untouched
— this lands the treedef implications (jit/scan) before any
behavior change. Files: `fields/scalar_field.py`,
`fields/vector_field.py`, `fields/storage.py`,
`decomposition/halo.py` (a `merge_min` helper), `operators/base.py`.

**B. Kernel ghost audit.** Certify the `valid_in − r` output claim
per family: staggering-based (FD, interp, flux_diff — via
`apply_staggered`), WENO/reconstruct, select, transforms/spectral
(halo-0), integrate. Deliverable: a claim table in the code (per-op
hook or default) with conservative zero where uncertifiable, and the
`pad` zero-fill check. Files: `operators/staggering.py` +
per-family modules; no behavior change yet.

**C. Placement flip (the one behavioral commit).** Consumption check
in both `__call__` templates + composite chain-sum + `OperatorSum`
hoist; `_finalize` and `store` stop syncing; memoization write-back
with tracer guard; `Reshard` validity reset. Update the contract
tests (`test_movement.py`'s "base appends sync", wave-1/3
integration halo assertions) and add the **exchange-count gate**
(§5). Files: `operators/base.py`, `operators/movement.py`,
`fields/storage.py`, tests.

**D. Negotiation/tracer + docs.** §3 in full; re-validate
multi-device widths (small-mesh shardability, cap logic);
benchmarks; docs flip: `classes/decomposition.md` (contract section
+ decision record → implemented), `classes/operators_base.md`
application path, `04_decomposition.md`, ROADMAP 1.8 → done,
`phase2_grid_followups.md` items 1, 6 (flag), 8 → done.

Single agent (or in-session), sequential — this touches the hot core
files; parallel waves would only add merge risk against the running
model agent.

## 5. Gates

- Full `tests/framework2` plain **and** forced-4
  (`XLA_FLAGS=--xla_force_host_platform_device_count=4
  FRIDOM_TEST_FORCED_DEVICES=4`); full repo `-n 8 --dist loadfile`;
  ruff clean; ≥95% branch coverage on touched modules.
- **Exchange-count tests** (counter-instrumented
  `Decomposition.sync`): (a) `f.diff("x").diff("x")` under W=2 → 1
  exchange; (b) an n-term tendency sum consuming one field → 1; (c)
  pointwise product on a sharded nodal space → 0; (d) field
  arithmetic → 0; (e) n modules consuming the same component → 1
  (memoization).
- Bitwise 1-vs-4 device invariance (`test_multi_device.py`)
  unchanged; `tests/framework2/validation` all green (results
  neutrality).
- Benchmarks (exclusive, per the compute budget in
  [`implementation_plan.md`](implementation_plan.md)): re-run
  `bench_operators` composed-tendency cases; expect the wave-4B
  re-fusion pathology class (+40–65% on composed chains) to shrink
  and multi-device exchange counts to drop to ~1/component/step.
  Record in `benchmarks/framework2/RESULTS.md`.

## 6. Risks

- **Stale-ghost wrongness** is the failure class the old contract
  couldn't have; defense in depth: the mechanical consumption check,
  conservative stage-B claims, the bitwise multi-device gate, and
  the validation suite.
- **Corner ghosts**: multi-axis consumption relies on syncs filling
  corners ("local fills run first", `tensor.py:552-553`) and on the
  per-axis joint validity arithmetic — same model `HaloTracer`
  already uses; covered by the 2D validation cases.
- **Treedef churn**: memoization mutates validity on eager-held
  fields → one extra compile per call site, then stable. Accept;
  revisit only if it shows up in benchmarks.
- **Wider negotiated halos** can break small-mesh sharding — the cap
  (§3) is the mitigation and is exercised by a dedicated test.
- **Tracer leak through memoization** — guarded (§2); the guard gets
  a regression test (closure-captured field inside jit).

## 7. Merge coordination

The model-layer agent works on its own branch against the current
(iteration-1) sync behavior; since the swap is results-neutral and
the field surface is unchanged, merge order is free. Preferred:
land `framework2-sync-redo` → `dev` after gates, model branch
rebases and inherits the speedup. Only shared-file risk: ROADMAP
row 1.8 and `phase2_grid_followups.md` status flips (line-disjoint
from model edits; trivial conflicts at worst).
