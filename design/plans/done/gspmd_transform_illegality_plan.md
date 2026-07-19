---
status: done
date: 2026-07-18
---

# GSPMD transform illegality — remaining phases

Owner-approved campaign (2026-07-18): the naive GSPMD transform path
becomes illegal, phased. Phases 0–1 shipped the same day (study and
implementation record:
[`../../research/gspmd_naive_transform_illegality.md`](../../research/gspmd_naive_transform_illegality.md)).
**Complete 2026-07-19**: every phase and follow-up item shipped (the
final four — Tier-2, Wave B, no-gather synthesis, trig/mixed — as
merges `dec698e2`, `4367d2f9`, `8462be11`, `48cfc25c`; outcome and
deviations in the "Shipped" entries below and the
[roadmap done entry](../../roadmap/done.md)). The only remaining
related work is the GPU-scoped validation checkpoint, owner-batched,
tracked in [`../../roadmap/open.md`](../../roadmap/open.md).

## Shipped

- **Phase 0 — Tier-1 guard** (`83fbc56c`): sharded-transform-axis use
  of `Transform.forward/backward` is a taught error; probe gather in
  `numeric_eigenpairs`; suite pinned. Fallout waves (fixture pins
  `a6bc4b39`, interval-accounting fixes — see
  [`../../research/halo_sharding_invariants.md`](../../research/halo_sharding_invariants.md))
  are closed; forced-4 on `tests/spatial|model|nonhydro2|shallowwater2`
  is clean of unmarked guard hits.
- **Phase 1 — 3-D channel synthesis** (`e316987d`):
  `ContractPlan.synthesize`; `mode()`/`channel_random_state` run on
  sharded 3-D channels, bit-identical parity, finite VJP.
- **Phase 2 — 2-D channel: transpose pipeline supersedes the gather**
  (`8752170a`). The owner's transpose directive (2026-07-19) rejected
  the gather path: the 2-D channel is served **exactly and without any
  gather** by `Channel2DPlan` — the fused transpose contraction (park
  the shardedness on the bounded axis, run the local `rfft`, per-`kx`
  dense `Q diag(w) Qᴴ M`). `_eigenbasis._contract_planes` /
  `_backward_synthesis` route projection, `f(L)` and synthesis through
  it; the taught error narrows to the non-1-D-mesh remainder. So the
  gather item is retired, not implemented.
- **Phase 3 core — the general distributed transform apply**
  (`8752170a`). `DistributedTransform` (in
  `operators/distributed_transform.py`) consumes the transform's
  `distributed_forward_plan`/`distributed_backward_plan` in a
  layout-preserving fused `backward(middle(forward(.)))` `shard_map`
  region (the `SlabPlan` shape generalized beyond the solve, plus the
  shared `TransposeGeometry`/`transpose_forward`/`transpose_backward`
  primitive `Channel2DPlan` is built on). `resolve_distributed_transform`
  serves a plain Fourier transform on a 1-D device mesh, declines
  (None) elsewhere. Parity 0.0–1.8e-15, no all-gather HLO-asserted.
- **Phase 3 consumer wave — the per-mode diagonal fused route**
  (`feat/distributed-transform-consumers`). `DistributedTransform.apply_diagonal`
  threads a per-mode diagonal through the region's `in_specs` (sharded
  on the sharded axis `a`), so a symbol that varies along the sharded
  axis runs per shard — the capability the shard-agnostic closure
  `middle` cannot carry. `Transform.apply_diagonal(f, symbol_factory)`
  is the consumer surface: it routes `backward(symbol(forward))` through
  the fused apply when the space is servable and the operand shards a
  stage axis, else the plain sandwich (single-device / replicated,
  bit-for-bit unchanged); the symbol is built per frame (the internal
  distributed frame differs from the single-device codomain) and must be
  endo. **Consumer served:** the Krylov CG `SpectralDerivative` apply
  (`test_krylov::test_solution_is_device_count_invariant` converted from
  a taught-error skip to a real device-count-invariance gate).

- **Analytic all-periodic route — Wave A** (`f8358720`, 2026-07-19).
  `DistributedTransform.apply_matrix`/`project`/`synthesize` (fused
  multi-component region, per-mode D×D matrix threaded sharded);
  `GridSymbols(coeff_spaces=...)` frame hook + `Eigenmodes._reframe`
  + `operator_matrix` (host `np.any` Nyquist gates → jnp masks, sw2
  DC patch folded as a k==0 mask); router `model/analytic_distributed.py`.
  Serves nh2/sw2 projections (parity ≤1.9e-15), balance/NNMD
  (≤7.7e-16), random-state/`mode()` (0.0 or ≤1.3e-15); single-device
  bit-identical; grad FD-matched; HLO all-to-all only.
- **ETDRK4 — Wave C** (`bce54cff`, 2026-07-19). `ContractPlan`/
  `Channel2DPlan` grew fused `project`/`synthesize_amplitudes` halves
  (documented amplitude sharding contract; padded lanes provably stay
  zero through the RK arithmetic); `ETDRK4._forward`/`_backward`
  route through them on sharded periodic axes (5 projects + 4
  synthesizes per step, shard-local phi/exp arithmetic). Invariance
  <1e-10, round-trip 1e-11, grad FD-matched, single-device
  bit-identical.

- **Tier-2 illegality** (merge `dec698e2`, 2026-07-19; owner-decided
  the same day). `Transform._reject_replicating_transform`: a taught
  error when a multi-device operand shards a *non-stage* axis while
  every stage axis is local (the forward's `CoefficientSpace`
  codomain is replicated by the storage contract, so GSPMD silently
  all-gathers — verified in HLO: the full array is materialized per
  device before the FFT). Replicated operands and collapsed
  decompositions stay exempt, so explicit gathers remain legal. The
  escape is `SpectralSolve(..., allow_replicated=True)`: an explicit
  replicate-then-compute (`reshard(Layout({}))` → composite →
  reshard back) that satisfies both guards via the replicated-operand
  exemption, echoing the decomposition's `allow_replicated`
  vocabulary. **Two deviations from the decision record, owner
  attention flagged:** (1) the study's "zero Tier-2-only test
  breakage" was wrong by one test —
  `test_reshard_transform_backward_round_trip` was a *deliberately
  designed* legal-Tier-2 pin, converted to the taught-error +
  explicit-`Layout({})`-escape pattern (a real silent→explicit
  capability change, not a no-op); (2) the recorded escape sites
  (Chebyshev-vertical, mismatched-layout composite) empirically trip
  *Tier-1* on tested grids (the composite's Fourier part sits on the
  sharded periodic axis), so `allow_replicated` is wired as a
  general replicate-then-compute rescue serving them regardless of
  tier, and the "Wave-B tier" escape was never needed in code
  (walled-vertical analytic consumers are Tier-1 territory; Wave B
  landed the same day). Guard-active `tests/spatial` forced-4: 3388
  passed, 0 failed.
- **Wave B — walled-vertical analytic tier** (merge `4367d2f9`,
  2026-07-19). `WalledVerticalTransform` in
  `spatial/operators/distributed_transform.py`: one fused
  `shard_map` region runs the two periodic axes through the shared
  transpose pipeline while the bounded trig axis rides device-local
  (per-shard DST/DCT), `ModeChart.embed` lifts each component onto
  the shared union lattice, and the per-mode D×D matrix
  (`assemble_walled_operator_matrix`, built frame-locally via the
  Wave-A hook) applies as one einsum. Serves nh2 walled-vertical
  projections (≤1.6e-15), balance/NNMD orders (≤2.8e-16),
  `mode()`/random-state (0.0, via the documented replicated
  fallback); HLO all-to-all only; grad FD-matched 2.2e-12. Two
  recorded deviations from design §4: the Fourier pair runs
  fully-complex (no third local axis to carry the rfft half), and
  the fused synthesize is *architecturally impossible* for walled
  frames (the internal frame never coincides with the single-device
  random-phase frame) — `AnalyticDistributedRoute.can_synthesize`
  (False for walled routes) gates `synthesize_columns` back to the
  replicated backward, the §4 fallback. sw2 walled is out of scope
  (a horizontal channel, served by the numeric `ChannelEigenmodes`).
- **No-gather random synthesis** (merge `8462be11`, 2026-07-19).
  `hermitian_reframe` (`model/analytic_distributed.py`) completes
  the device-independent gain columns onto the transpose engine's
  re-designated internal frame: stored half kept, missing half =
  conjugated multi-axis reflection, and on the self-conjugate planes
  (DC, Nyquist at even extents) the `(stored + reflected-conj)/2`
  average — the load-bearing subtlety: the analytic gains are not
  Hermitian there for `u`/`b`, and the average reproduces exactly
  the Hermitian projection `irfft` was silently applying. The shared
  `synthesize_columns` tail routes random-state and `mode()` through
  the fused backward (no all-gather, HLO-asserted); the replicated
  backward remains only where genuinely unserved (single device,
  walled routes via `can_synthesize=False`, non-1-D layouts).
  Invariance nh2 1.33e-15 / sw2 4.44e-16; the O(field) replicated
  gain *columns* keep the pre-existing coefficient-replication
  contract (the numeric channel `mode()` precedent) — the eliminated
  gather is the backward-transform collective on the IC path.
- **Trig/mixed transform families** (merge `48cfc25c`, 2026-07-19).
  `ComposedTransform.apply_diagonal`: a sharded operand routes
  through the walled solve's `SlabPlan` (`apply_plan_diagonal`,
  shared with `SlabSolve.__call__`) with the symbol built on the
  plan's internal coefficient frame — serving the walled-channel
  spectral apply (shard a periodic axis) at 1.2e-14 invariance,
  2 all-to-alls, grad FD-matched 1.3e-13; the Tier-1 route hint now
  points mixed frames at `resolve_transform(...).apply_diagonal`.
  **Deviation from the §4 steer (deliberate):** reused the tested
  `SlabPlan` geometry (bounded axis as transpose partner, rfft half
  on the local Fourier axis, 2 all-to-alls) instead of a
  `WalledVerticalTransform.apply_diagonal` (whose z-local
  union-lattice shape exists for the multi-component contraction and
  costs 4 all-to-alls) — less code, strictly faster.
  **Pure-trig homogeneous shapes stay declined, deliberately:** no
  in-repo consumer reaches them; the machinery was verified ready
  (1.3e-14 probe) should one appear. One constraint surfaced: the
  operator's `eigenvalues` must be an endomorphism on the full mixed
  frame — a purely-horizontal operator fails both the fused route
  and the single-device sandwich consistently (taught error).

## Related owner items (from the same campaign)

- GPU validation of the campaign's multi-device paths at the next
  owner-batched checkpoint (all GPU-scoped: the fused synthesis
  parity tests, the 3-D `ContractPlan` ETDRK4 end-to-end run and its
  distributed grad — real `eigh` bases are CPU-unsafe, jax#39292).
  Agents do not submit GPU jobs. Tracked in
  [`../../roadmap/open.md`](../../roadmap/open.md).
- (The halo ratifications formerly listed here were both resolved by
  owner rulings 2026-07-19, shipped in `fc2a3b66` — entries in
  `roadmap/done.md`.)
