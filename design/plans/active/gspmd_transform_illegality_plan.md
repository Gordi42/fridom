---
status: active
date: 2026-07-18
---

# GSPMD transform illegality — remaining phases

Owner-approved campaign (2026-07-18): the naive GSPMD transform path
becomes illegal, phased. Phases 0–1 shipped the same day (study and
implementation record:
[`../../research/gspmd_naive_transform_illegality.md`](../../research/gspmd_naive_transform_illegality.md)).
This plan tracks the remainder. Each phase is independently mergeable.

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

## Open

- **Tier-2 — DECIDED (owner, 2026-07-19): illegal, as
  recommended.** All-local-axes naive transforms on a multi-device
  mesh (silent all-gather; numerically correct but unscalable)
  become illegal, with the explicit allow-replicated escape for the
  irreducible cases: the Chebyshev-vertical solve (block-diagonal,
  cannot go slab), the mismatched-layout composite solve, plus the
  Wave-B tier below (until Wave B lands). Empirically zero
  Tier-2-only test breakage. Implementation open.
- **Wave B — walled-vertical analytic tier.** The analytic route
  (`f8358720`) serves plain-Fourier all-periodic frames;
  walled-vertical analytic grids (`ComposedTransform` trig z stage,
  `ModeChart` embed/restrict) keep the taught error. Needs a
  `ContractPlan`-shaped region absorbing the bounded axis into the
  stacked column
  ([`../../research/analytic_eigenmode_distributed_route.md`](../../research/analytic_eigenmode_distributed_route.md)
  §4).
- **No-gather random synthesis (frame-mismatch cases).** Random-state
  gains/phases are built on the device-independent single-device
  frame (`grid.random` keys on the global storage index); when the
  sharded axis is that frame's half axis (nh2 x-sharded, sw2 2-D),
  synthesis falls back to a replicated backward — device-invariant
  and correct, but a gather on the IC path. A pure fused route needs
  a Hermitian half-axis re-expression of the gains.
- **Trig/mixed transform families.** `resolve_distributed_transform`
  serves plain Fourier only; `ComposedTransform` declines to the
  taught error.

## Related open owner items (from the same campaign, tracked here)

- GPU validation of the campaign's multi-device paths at the next
  owner-batched checkpoint (all GPU-scoped: the fused synthesis
  parity tests, the 3-D `ContractPlan` ETDRK4 end-to-end run and its
  distributed grad — real `eigh` bases are CPU-unsafe, jax#39292).
  Agents do not submit GPU jobs.
- (The halo ratifications formerly listed here were both resolved by
  owner rulings 2026-07-19, shipped in `fc2a3b66` — entries in
  `roadmap/done.md`.)
