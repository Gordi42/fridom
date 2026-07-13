---
status: done
date: 2026-07-13
---

# Graded-order boundary fallback operator — implementation plan

**Shipped.** The graded near-wall closure — a wide high-order kernel in
the interior, progressively narrower *interior-only* stencils on the `K`
faces adjacent to each physical wall — is in the tree, on both the
finite-volume (WENO) and the nodal (C-grid advection) route, and works on
a sharded bounded axis. It reads no exterior value and needs no BC, so a
wide bounded reconstruction is R1-legal by construction; WENO's
periodic-only restriction is retired.

## Goal (as delivered)

`WenoReconstruction(order, boundary="graded")` mints a `Fallback`
operator that accepts a bounded factor (`CellAvg -> Inner`) and grades
the order down (p → p−2 → … → 1) over the wall rows using interior DOFs
only. Plain `WenoReconstruction(order)` still raises on bounded, so the
default operator table stays closed under R1; graded is opt-in, a
module-override row.

## Landed

- **F1 — `Fallback` core.** `src/fridom/spatial/operators/fallback.py`:
  the `Fallback` separable operator, the `graded_ladder` /
  `graded_reconstruction` builders, `UpwindOne` (innermost rung), the
  per-rung WENO kernels. Interned on its static structure; no edits to
  `reconstruct.py` / `base.py`. (`f0937e56`)
- **F2 — the `boundary="graded"` knob.** `weno.py` `_BOUNDARY_MODES =
  ("none", "graded")`; the graded spelling returns a `Fallback` whose
  `codomain` accepts bounded. (`f0937e56`)
- **F3 — validation suite.** `tests/spatial/operators/test_fallback.py`,
  `test_fallback_validation.py`: NaN-poison exterior-halo gate
  (`test_weno3_nan_poison_gate_proves_interior_only`), interior
  convergence, graceful wall-row degradation, jit single-trace across a
  value sweep. (`f0937e56`)
- **F6 — distributed bounded axis.** `layout="local"` is gone;
  `Fallback.requirements` returns `layout="any"` everywhere. The wall
  patch sits behind the decomposition-layer seam `patch_physical_ends`
  (`decomposition/decomposition.py`, implemented in
  `decomposition/tensor.py`), so the operator stays shard-blind.
  Gated by `tests/spatial/decomposition/test_fallback_multi_device.py`
  (forced-4 device-count invariance + the interior-only gate under
  sharding). (`f0937e56`)
- **Shared ladder + nodal route.** The ladder arithmetic, wall-window
  builder and multi-device patch were factored into
  `spatial/operators/graded.py`, and the nonhydro2 upwind/WENO advection
  grew its own `boundary="graded"` nodal closure on top of it —
  the practical consumer of this work.
  (`4eac7ccf`, `5005f985`; merge `719ff4cd`)
- **Configurable bottom rung.** The ladder's wall-adjacent rung is a
  knob (`biased_specs` / `centered_*` in `graded.py`), and the biased
  advection can request a centered near-wall rung instead of the
  1st-order upwind cell.
  (`773af55b`, `0526d06b`; merge `1b5eeb9b`)

## Constraint added after the plan was written

The biased reconstructions refuse stretched (mapped) meshes
(`2884efcb`, `63bb6dd2`; merge `fe24b9f7`), so the graded closure is
uniform-mesh-only today. Lifting that is a separate work item, not part
of this plan.

## Not done (deferred, optional)

- **F4 — linear graded.** No `boundary="graded"` knob on
  `finite_difference.py` / `interp.py`; the `Fallback` machinery would
  carry it (the ladder is order/kind-agnostic), but nothing needs it
  yet. With it would come the `f = y(1−y)` / `d² = −2` legality case
  (`bc_free_boundaries.md:22-35`): a bounded second derivative computed
  by a graded linear stencil with no extrapolation fill. Pick this up
  only when a consumer asks for it.

## Still designed-for (unchanged non-goals)

- Immersed-boundary graded reconstruction (the reduced-order zone is a
  geometry mask, so it needs a `jnp.where`/gather path, not the static
  slice partition used here).
- Order-*preserving* one-sided WENO (one-sided smoothness indicators).
- The R1 flip / width-aware legality guard on the *default* rows.
