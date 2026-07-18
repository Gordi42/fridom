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

## Open

- **Phase 2 — 2-D channel gather path (M; OWNER RATIFICATION
  PENDING).** Same ratification item as the eigen-remainder entry in
  `roadmap/open.md`: a gather path scoped to 2-D channels (exact,
  negligible cost at dense-`eigh`-buildable sizes, `em.q` already
  replicated), validated in
  [`../../research/eigen_remainder_investigation.md`](../../research/eigen_remainder_investigation.md)
  §3.2. Implementation after the go-ahead: projection + synthesis both
  route through the gather; the taught error narrows to non-1-D
  meshes only.
- **Phase 3 — standalone distributed Transform apply (L).** Consume
  the so-far-unconsumed `distributed_forward_plan`/
  `distributed_backward_plan` in a general
  forward → pointwise → backward distributed pipeline (the `SlabPlan`
  shape, generalized beyond the solve). Re-legalizes on sharded
  grids: analytic/numeric eigenmode transforms (nh + sw), the state
  transforms (balance expansion, optimal balance, NNMD), the
  exponential time stepper, and the Krylov CG's `SpectralDerivative`
  apply. Also converts the residual marked-test debt (listed in the
  research record) back to real device-count-invariance tests. Wide
  blast radius: `operators/transform.py` or a new module,
  `symbols.py`/`realized.py`, nh/sw transform modules.
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
