---
status: active
date: 2026-07-13
---

# Phase-2 reconciliation — grid-layer follow-up work items

> **Sized 2026-07-13: SMALL (~2-3 days, ~300-450 LOC).** Roadmap: a
> **next step** — the `cartesian.Grid` convenience constructor should land
> *before* the docs rebuild writes Getting Started (the existing docs pages
> use the old stack's equivalent). The coefficient-space product/power rows
> are **deferred**: not a missing row but a semantics decision (elementwise
> multiplication of Fourier coefficients is a convolution, not a product),
> and they block nothing.

Grid-layer follow-ups discovered while reconciling the Phase-2 model
design ([`model/`](../../specs/model/00_overview.md), decisions D1–D5)
against the merged Phase-1 grid implementation (2026-07-08). This is
the analogue, in the opposite direction, of
[`../done/phase1_findings.md`](../../research/phase1_findings.md): where that
file fed implementation findings back into the design, this file fed
design demands back to the grid implementers.

Status (2026-07-13): **everything that gated the model layer has
landed**. What is left is one ergonomics item, kept below. The file
stays active only as the home of that one; it is also cited as the
record of the `_halo_valid`/treedef failure mode (item R11), e.g. by
[`krylov_scan_plan.md`](../done/krylov_scan_plan.md).

## Resolved (original numbering, one line each)

- **R1 `negotiate` combined-halo semantics** — `tendency=`/`halo=`
  combine as merge_max (`e506a91d`); regression test
  `tests/spatial/decomposition/test_negotiate.py`.
- **R2 `merge_overrides` facade** — implemented on `Grid`
  (`8ad33bf9`; `spatial/grid.py:318`), used by the model assembly and
  both models.
- **R3 `("declared_space", mesh)` resolver rows** — dedicated resolver
  table in the registry, never module-mergeable (`8ad33bf9`;
  `spatial/operators/registry.py`, `_DECLARED_SPACE`).
- **R4 `trace_halo` name-keyed states** — accepts
  `Mapping[str, SpaceLike]` (`e506a91d`;
  `spatial/decomposition/halo.py:967`).
- **R5 `freeze()` fingerprint + verify path + `GridFrozenError`** —
  `NegotiationFingerprint`, demand-satisfaction verify, and
  `GridFrozenError` (`8ad33bf9`; `spatial/grid.py:439`,
  `spatial/errors.py:133`; `tests/spatial/test_freeze_fingerprint.py`).
- **R6 `VectorField.add` + metadata re-attachment** — `add`/`replace`/
  `map` all route through `_keep_metadata` (`040f5ad5`;
  `spatial/fields/vector_field.py`).
- **R7 annotation-exempt metadata equality in jaxify** — the
  `annotation=` aux category is excluded from treedef equality
  (`040f5ad5`; `framework/utils/jax_utils.py`).
- **R8 sync-strategy redo** — consumption-side sync with trace-time
  halo-validity depth; `store` is pad-only (ROADMAP 1.8, merge
  `e7d0bb37`; stage log [`../done/sync_redo_plan.md`](../done/sync_redo_plan.md)).
  Also resolves the "eager field creation re-traces per call under
  multi-device" perf item: `create_field` no longer builds a sync
  closure.
- **R11 carry-resident AUXILIARY fields break jitted `scan` treedef
  stability** — root-caused in the grid layer: the ghost-cache seam
  records the memoized exchange in an external identity-keyed
  `_SYNC_CACHE` instead of mutating `f._halo_valid` in place
  (`f79f9fbf`; `spatial/operators/base.py`). The treedef-exemption
  route stays **rejected** — `halo_valid` drives sync *placement*, so
  excluding it from the jit-cache key could reuse a body compiled for
  a different halo state. Same failure class as the Krylov carry
  (see [`krylov_scan_plan.md`](../done/krylov_scan_plan.md)).
- **R12 `ConstantSpace`/`Profile()` broadcast in a tendency term** —
  the `HaloTracer` product path lifts a constant operand onto the
  nodal join like the eager path (shared `_lift_field`), plus a
  `ConstantSpace→nodal` `.to` row (`f79f9fbf`).
- **R13 field arithmetic rejects a traced scalar operand** — the
  arithmetic dunders take a 0-d `jax.Array` (`cff59a93`,
  `309bbdd6`; `_is_0d_array` in `spatial/fields/scalar_field.py`).
  The `extra_halo` declarations still present in model modules are
  V-N2 trace exemptions, not raw-`.data` escapes.
- **R14 BC-free bounded spaces** — superseded: the R1 flip landed
  (merge `9a95202a`), the extrapolation fill is gone, one-sided rows
  are the opt-in closure. Note moved to
  [`../done/bc_free_boundaries.md`](../done/bc_free_boundaries.md);
  the remaining boundary work (Robin dynamic data) lives in
  [`boundary_plan.md`](boundary_plan.md), which also owns the
  BC-nodal operator-row gap.
- **R15 `fr.spatial.cartesian.Grid` constructor** — built 2026-07-15
  (the decision was keep, not withdraw): the `shape=`/`extent=`/
  `periodic=`/`names=` convenience subclass, filled from
  [`../../specs/grid/classes/grid.md`](../../specs/grid/classes/grid.md),
  alongside the new `fr.spatial.spherical.Grid` fast assemble and the
  `fr.spatial.charts.lonlat_sphere` chart primitive. Record:
  [`../done/grid_ergonomics_plan.md`](../done/grid_ergonomics_plan.md).

## Open work items

1. **Teaching shims + remaining API gaps** — ergonomics, not
   blocking. What is left of the
   [`../done/phase1_findings.md`](../../research/phase1_findings.md) backlog:
   - `ImmutableStateError` on `.data` assignment
     ([`../../specs/grid/classes/fields.md`](../../specs/grid/classes/fields.md)
     D1.5): `.data` is a bare read-only property today, so an
     assignment raises an unhelpful `AttributeError`.
   - Transform classes (`Fourier`, `Sine`, `Cosine`, `Chebyshev`,
     `SpectralDerivative`, `PhaseShift`) and `NodeSet` are not
     re-exported at the `fr.spatial.*` level; `grid.dispatch` is
     typed `object`.
   - Coefficient-space fields have no product/power rows, and
     constant→coefficient broadcast is blocked, so spectral operator
     coefficients still drop to `.data`.

   Landed from that backlog and no longer tracked here:
   `MissingComponentError` and the `_component` hint path,
   `ScalarField.to` metadata preservation (operator application
   carries `result.metadata`), `.item()`, biased nodal stencils
   (upwind/WENO), and WENO's halo (modules declare no `extra_halo`;
   negotiation picks the width). Clenshaw–Curtis measures on
   `ChebyshevMesh` are tracked in
   [`high_order_mapped_plan.md`](high_order_mapped_plan.md).
