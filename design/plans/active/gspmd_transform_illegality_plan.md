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

## Open

- **Phase 3 — remaining consumers (deferred debt).** These hit the
  Tier-1 taught error on sharded grids and are **not** served by the
  single-field diagonal route: each needs a multi-component per-mode
  **matrix** contraction (the all-periodic analog of `ContractPlan` —
  stacked component fields, the per-mode eigenvector matrix `Q Qᴴ M`
  in the internal frame, a multi-field fused region) or intermediate
  materialized amplitude state, deliberately not built here (do not
  force a re-gathering path):
  - the **exponential time stepper** (`ETDRK4._forward`/`_backward`,
    `model/time_steppers/exponential.py`): `fourier_ops` per component
    plus the eigenbasis column contraction, with amplitudes flowing
    across per-stage physical tendency evaluations. No forced-4 test
    exists today; the seam is the channel-contraction shape but split
    into a `project`-to-amplitudes and a `synthesize`-from-amplitudes
    half (raw sharded amplitude arrays between them). Servable by
    giving `ContractPlan`/`Channel2DPlan` those two halves; deferred.
  - the **analytic (all-periodic) eigenmode projections** (nh/sw
    `eigenmodes.py` `GridEigenmodes.projector`/`function`, `kit.forward`
    + per-mode eigenvector matrix + `kit.backward`, via `BoundTransform`
    on `Transform.forward`/`backward`).
  - the **balance / NNMD state transforms** (`transforms/balance_expansion.py`)
    that ride the same analytic `kit`. (`optimal_balance` is a step-path
    `Propagator` — orthogonal to the transform seam, out of scope.)
  The **numeric channel** eigenmode projections / `f(L)` / synthesis
  are already served (Phases 1–2, `ContractPlan`/`Channel2DPlan`).
- **Tier-2 decision (OWNER).** Whether all-local-axes naive
  transforms on a multi-device mesh (silent all-gather; numerically
  correct but unscalable) also become illegal. Empirically zero
  Tier-2-only test breakage today, but the guard would need an
  explicit allow-replicated escape for the irreducible cases
  (Chebyshev-vertical solve — block-diagonal, cannot go slab;
  mismatched-layout composite solve). Recommended only after phase 3,
  so the escape list is minimal.

## Related open owner items (from the same campaign, tracked here)

- Ratify the verify-side treatment of explicit `halo=` (capped like
  negotiate — the shipped behavior) and decide the `_cap_for_sharding`
  over-reach question (it caps non-sharded ghost axes too; fixing it
  changes negotiated widths, i.e. storage/perf).
- GPU validation of the campaign's multi-device paths (the fused
  synthesis parity tests are GPU-scoped) at the next owner-batched
  guard/validation checkpoint; agents do not submit GPU jobs.
