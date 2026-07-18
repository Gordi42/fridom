---
status: frozen
date: 2026-07-18
---

# Storage-halo width — the biased-buffer +1 layer, probed and resolved

Charge (roadmap, from
[`upwind5_revisit.md`](upwind5_revisit.md) §6): biased order-5 pads
storage to `n+8` per axis where the nominal stencil reach needs `n+6`
(centered: `n+4` vs `n+2`) — ~6% inflation on every upwind5 buffer at
96³ (−3.0% bytes at 192³), est. 2–3 ms/step @192³ on RTX-3060-class
hardware. Flagged "a core staggering-policy question, parity-sensitive,
unprobed."

Answer up front: **probed, mechanism identified, and the width is
recoverable.** The extra layer is not staggering and not even-rounding —
it is an interval-arithmetic over-approximation in the halo trace: the
sync-free chain rule **sums each operator's symmetrized scalar reach**,
losing the biased window's asymmetry. The true composed footprint of
`flux_diff ∘ reconstruct` is 3 cells/side for order 5 (1/side for
centered) — width 3/1 storage is sufficient for **bit-identical**
physics (measured). Two-sided (interval) halo accounting recovers it
with no extra syncs; implemented and shipped (§3): upwind5/weno5 now
negotiate width 3 (`n+6`), sync-count-invariant, bitwise-parity
verified, full new-stack suite green. One qualifier: at the *model*
level the centered family stays at width 2 (`n+4`) because
`DynamicalCore.extra_halo = 2` floors the negotiation — the centered
half of the roadmap estimate needs that floor revisited (left open).

## 1. Mechanism — where `n+6`-worth of reach becomes `n+8` of storage

Storage per axis is `n + 2*width`, symmetric, no rounding
(`spatial/decomposition/tensor.py:431`; multi-device block
`cells + 1 + 2*width` at `:435` — the `+1` is the stagger reserve for
n-vs-n+1 staggered pairs and is orthogonal to this item). `width` is
the negotiated `HaloSpec`: the **max sync-free accumulated depth over
the traced tendency chain**, where sequential un-synced composition
**sums** per-operator scalar reaches (`halo.py:120–145` `grow`,
`base.py:1706–1721` `_chain_requirements`) and parallel terms merge by
max.

The biased order-5 reconstruction declares `halo = order//2 + 1 = 3`
(`weno.py:823`, `model/modules/advection.py:1151`); the trailing flux
difference adds 1 (`flux_diff.py:316`): width 4 → `n+8`. Centered:
1 + 1 → width 2 → `n+4`. Upwind3 negotiates 3 (odd) — the
"rounded to even" hypothesis is refuted directly.

The sum is over-tight by exactly 1/side because the scalar accounting
symmetrizes an asymmetric window before summing. In cell indices (face
`j` between cells `j` and `j+1`): the biased-pair union window at face
`j` is cells `j−2..j+3` — offsets `[−2,+3]`, symmetrized to scalar 3.
The flux difference at cell `i` reads faces `i−1, i` — offsets
`[−1,0]`. The composed footprint is the Minkowski sum
`[−1,0] ⊕ [−2,+3] = [−3,+3]`: **3/side**, not `1+3=4`. Centered:
`[−1,0] ⊕ [0,+1] = [−1,+1]`: 1/side, not 2. The same tightening falls
out for every direction-alternating chain (a `diff∘diff` Laplacian is
`[−1,+1]`, width 1, not 2; upwind3 `[−2,+2]`, width 2, not 3).

Note the width applies to the **step-transient sealed/flux/
reconstruction buffers**, not the persistent state (state is stored at
true shape and sealed to full width once per step,
[`stencil_lowering.md`](stencil_lowering.md) §"seal"): the savings land
precisely on the buffers the byte attribution identified as upwind5's
advection-phase peak (9 flux products `[n+8]³` + 6 face
reconstructions, [`upwind5_revisit.md`](upwind5_revisit.md) §3).

## 2. Empirical probe — forcing the width (scripts: [`storage_halo_width/probe/`](storage_halo_width/probe/))

Single device, CPU, triperiodic `nh.Model` `family="nodal"` (biased FV
assembly is still blocked on dev, `upwind5_revisit.md` §7.1). Width
forced below traced demand by clamping the two negotiation seams
(`_negotiated_halo` + `Grid._demanded_halo`; an explicit `halo=` can
only widen via `merge_max`).

- **E1 parity (32³, 10 steps):** upwind5 at width 3 is **exactly
  bit-identical** to width 4 on u,v,w,b. Centered at width 1 agrees to
  ~1.3e-15 abs (~4e-16 rel; fp reassociation from the shape change).
  Width 3/1 is physically sufficient — the composite-window arithmetic
  above, confirmed end to end.
- **E4 floor:** width 2 fails loudly at the assembly dry-run
  (`reconstruct.py:281`, "negotiated halo width 2 … too small for the
  5-point stencil"). The consumption-side guards make
  under-provisioning a taught error, never garbage.
- **E2 the catch:** with today's scalar accounting, the narrow store
  triggers `_ensure_valid` mid-chain re-syncs on the reconstructed
  intermediates (upwind5: 7 → 16 syncs per advection evaluation,
  14 → 26 per step; centered w1 the same) — the runtime still demands
  the summed depth 4/2 and refills. Physics unchanged (E1), but
  storage bytes are traded for exchanges: single-device this is cheap
  wrap copies (E5: still ~13% faster at 64³ CPU, indicative), on
  multi-device they become real ppermute exchanges and would likely
  cancel or reverse the win. **Forcing alone is not shippable; the
  chain requirement itself must carry the tight composite.**
- **E3 bytes:** compiled-step `memory_analysis` (arg/temp/output) all
  track `(n+6)³/(n+8)³` to 4 s.f.: −8.1% @64³, −5.7% @96³ (→ −3.0%
  @192³). The saving is the ghost-shell volume, exactly as estimated.

## 3. The fix — two-sided (interval) halo accounting (shipped)

Design: `HaloSpec` stores a per-name two-sided reach `(below, above)`
internally and presents the per-side **maximum** as the symmetric
storage width (`__getitem__`, `.widths`), so every storage consumer
(`tensor.py`, stencil bounds checks, exchange) is untouched.
`OperatorRequirements` gains `reach: tuple[int, int] | None` (the
scalar `halo` remains as the symmetric collapse). The trace
accumulates the two-sided window (Minkowski sum along un-synced
chains, per-side max across parallel terms), and the runtime
`halo_valid` carries the interval so `_ensure_valid` no longer
re-syncs a side that has spare validity. Per-op reaches are derived
from the **implemented** stencil geometry
(`staggering.window_reach` / `exterior_reach` at each kernel's own
alignment), not hardcoded: e.g. upwind5's left bias measures
`Center→Right = [−2,+2]`, the right bias `[−1,+3]`, the select
max-merges to `[−3,+3]`. 13 source files, +648/−214. One-sided
closure variants deliberately stay symmetric (their edge patches use
wider one-sided true-DOF stencils the interior reach doesn't
describe); `extra_halo` module demands stay scalar/symmetric.

Gates (all green):

- **Widths:** upwind5 → 3, weno5 → 3 on the assembled model (dev: 4);
  plain-grid/registry centered chains → 1 (dev: 2); `diff∘diff`
  Laplacian → 1; bounded shrinking stencils (`Center→Inner`) → 0.
  Asserted by new tests in `test_negotiate.py` / `test_fv_default.py`.
- **Sync-count invariance:** upwind5/weno5/centered per-step
  `grid.sync` count identical to dev (control: narrow storage with
  *scalar* validity re-syncs 28 vs 16 — two-sided validity is
  load-bearing, cf. §2 E2).
- **Parity:** narrow vs forced-wide same-code runs bitwise identical
  (maxdiff exactly 0.0, upwind5 and centered, 32³×10 steps).
- **Suites:** full new-stack sweep green (5401 passed at
  implementation time; 6329 passed / 0 failed re-run after merging
  the same-day dev movement — FV fusion-guard ratchet and windowed
  walled arms included), forced-4
  multi-device decomposition suite 334 passed, all three autodiff
  regression files pass, ruff clean. The reblock HLO golden was
  regenerated (pure width-2→1 shape shift; lowering structure
  unchanged).

Honest deltas beyond the accounting itself:

1. **Latent cache bug fixed:** `ImmersedDomain._cache` and
   `Grid._measures` cached padded concrete arrays keyed only on
   `(space, kind)`; a model re-negotiating *wider* (e.g.
   `HydrostaticCore.extra_halo = 2`) got stale narrow arrays. Keys now
   include the negotiated halo. The old base-2 provisional masked
   this.
2. **One FV assertion honestly relaxed:** the FV mapped-pressure
   cross-family check was bitwise on dev only because the FV
   measure-weighted inner product's pairwise-reduction tree happened
   to coincide at width 2; it is not width-invariant (~2 ulp). Relaxed
   to `atol=1e-14` with an explanatory comment; the residual gate is
   unchanged.
3. **Width-pin updates:** ~20 test files pinned the old scalar widths
   (registry provisional 2, bounded 1, triple-diff 3, ...); every new
   pinned value was re-derived from the kernel geometry, not adjusted
   to pass.

## 3.1 What the model level actually gets

The assembled nonhydro model's negotiated width is
`max(traced chain, module extra_halo)`. upwind5/weno5 chains dominate
their floors → **width 3, storage `n+6`, the full §2 byte saving**.
The centered chain tightens to 1 but `DynamicalCore.extra_halo = 2`
floors the model at width 2 → the centered `n+4 → n+2` half of the
roadmap estimate is **not yet realized**; it now needs the
`extra_halo = 2` declaration revisited (same shape as the
shallow-water `extra_halo` item previously reverted as "not
mechanical", §6.2). Left on the roadmap.

## 4. Parity sensitivity, resolved

"Parity-sensitive" in the roadmap = result-parity (the corpus uses
"parity" exclusively for bitwise/machine-precision reproducibility;
even/odd is refuted by upwind3's width 3). Measured: upwind5's
narrowing is bit-transparent on the tendency path; centered moves by
~1 ulp (reassociation under changed buffer shapes). The multi-device
HLO goldens pin storage shapes and need a regen; the sharded≡serial
invariants are width-independent by construction. Prototype evidence:
narrow-vs-wide same-code parity is bitwise (0.0) on both schemes, the
forced-4 multi-device suite passes wholesale, and the only genuine
parity casualty found is the FV mapped-pressure pairwise-reduction
coincidence (§3 item 2) — ~2 ulp, documented at the assertion.

## 5. Perf expectation and the manual GPU protocol

Structural savings: every step-transient buffer shrinks by the
ghost-shell ratio (−3.0% bytes at 192³, −5.7% at 96³, more at smaller
n); upwind5's §3 attribution puts the wider-halo share of the
+82%-over-centered delta at ~25%/13% (128/192³), so the realizable
wall-clock win is the roadmap's 2–3 ms/step @192³ on RTX-3060-class
hardware for the biased/WENO family. (The centered-path `n+4 → n+2`
win waits on the `extra_halo` floor, §3.1.) GPU wall-clock validation
is deliberately **not** run here
(owner ruling: no automated cluster submissions,
[`../plans/active/perf_guard_plan.md`](../plans/active/perf_guard_plan.md)
§5.1); protocol when wanted: the step suite A/B at 128³/192³ upwind5 +
centered, single A100, fresh process per variant, plus the
`benchmarks/comparison` weno5/centered rows.

## 6. Side findings

1. [`upwind5_revisit.md`](upwind5_revisit.md) §3's "`n+8` vs centered's
   `n+4` — 2x the nominal reach on both" overstates upwind5: the
   accurate framing is +1 layer/side (`n+8` vs `n+6`), as §6 and the
   roadmap have it. (Record is frozen; noted here rather than edited.)
2. Adjacent but distinct: `sw.DynamicalCore.extra_halo` is an
   unconditional 2/coordinate (linear shallow-water pays 2× halo
   bytes); previously attempted and reverted as "not mechanical"
   ([`../plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md)
   §1.3). Interval accounting does not touch it (it merges as a
   symmetric user demand).
