---
status: open
date: 2026-07-07
---

# BC-free bounded spaces: exterior values are untouchable

**Owner-flagged open question (2026-07-07, raised during task 1.8).
Not scheduled; needs an owner decision before the model layer
registers BC-dependent modules on bounded meshes.** Cross-refs:
work item 11 in [`phase2_grid_followups.md`](phase2_grid_followups.md);
the BC-row API gap in [`../done/phase1_findings.md`](../done/phase1_findings.md);
the task-1.8 periodicity-gated validity claims in the
[decomposition contract](../../specs/grid/classes/decomposition.md) (whose recorded
mirror-commutation refinement this question subsumes).

## Proposed rule

On a bounded axis, a **BC-free** space makes values beyond the
boundary *undefined* — no operation may read them, and the storage
layer must not invent them. An operator signature is legal on a
BC-free bounded operand **iff its true-shape output needs no
exterior values**; signatures that do need them exist only for
**BC-structured** operand spaces, where the boundary closure is
declared physics, not a guess.

This replaces the iteration-1 BC-free closure (one-sided linear
extrapolation ghost fill, `tensor.py`), which is a *hidden
numerical scheme*: consumed once it is a defensible second-order
one-sided stencil, but composed it is inconsistent — the
extrapolation kills curvature, so `f.diff("y").diff("y")` on
`f = y(1−y)` returns **0 instead of −2** at the boundary cells
(n = 4 centers on [0, 1]: the extrapolated ghost gives the
boundary-face derivative as 0.5 where the true value is 1.0; the
second difference then cancels exactly). An implicit closure that
is fine once and garbage twice, applied silently by the storage
layer, is the "silent wrongness" class the framework exists to
forbid. Task 1.8 *contained* it (validity claims reset on bounded
axes, so it can no longer leak through chains) — this rule would
*remove* it.

## The legality split is already type-expressible

The face-set structure of the nodal family encodes exactly which
operations are total without boundary knowledge:

| signature | exterior values needed? | BC-free legality |
|---|---|---|
| `Center -> Inner` (diff, interp) | no — every inner face lies between two centers | legal |
| `Outer -> Center` | no — the boundary faces are true DOFs | legal |
| `Center -> Outer` | the two boundary faces | **BC-structured only** (Dirichlet: the face value *is* the BC datum; nothing exterior is read) |
| `Inner -> Center` | the two boundary faces (ghost slots) | **BC-structured only** — this is where the `d²` counterexample leaked |

So "legal" is per registry row, not a new concept: dispatch is
keyed on `(kind, space)`, BC-structured spaces are distinct interned
spaces, and a missing row is already a clean `DispatchError`. The
rule is enforced by *which rows exist*, plus deleting the BC-free
bounded fill (interior shard edges still exchange — those ghosts
are wrap-like copies and unproblematic; only the physical-boundary
slots become unreadable).

## What already exists

- **The pattern, in miniature**: `FluxDifference` on the Inner
  domain refuses the extrapolation explicitly ("the BC-free ghost
  extrapolation must never leak in here") and closes the boundary
  with the exact zero fluxes of the no-normal-flow contract —
  boundary values from declared physics, never a storage-layer
  guess.
- **BC-structured spaces and fills**: `mesh.nodal(..., bc=...)`
  variants are interned, shape-correct (only Dirichlet reduces),
  and their mirror fills (odd / even-about-member) are implemented
  and tested.
- **Consumption-side sync (task 1.8)** removed the structural need
  for total fills: nothing blanket-syncs results anymore, so a
  space may legitimately have *no* boundary fill.

## What it unlocks

- Outlawing the one non-commuting fill leaves only mirror fills on
  bounded axes — which commute with the stencil families — so the
  task-1.8 validity claims can extend to bounded axes and bounded
  chains elide exchanges like periodic ones (the refinement
  recorded in the decomposition decision record).
- Boundary handling becomes auditable: every boundary closure is
  either a declared BC structure or an explicitly registered
  one-sided stencil row (see below), never implicit.

## Costs / migration

1. **BC-structured operator rows** across the stencil families —
   the already-filed API gap; today BC-free is the only fully
   usable bounded path, which is *why* the extrapolation is
   load-bearing. This inverts it: BC-structured becomes the
   supported path.
2. **Explicit one-sided boundary stencils** as the opt-in
   replacement where no BC is physically available (e.g. open
   boundaries, diagnostics): registered rows a user chooses, with
   visible order/stencil — the honest form of what the
   extrapolation fill did implicitly. Scope per family (FD, interp,
   reconstruct/WENO) to be decided.
3. **Migration** of everything leaning on the fill: the bounded
   validation cases and the FV/WENO boundary story.
4. **Model-layer tie-in**: inhomogeneous BCs
   (`grid.sync(boundary_data=...)`, designed-for) fold in naturally
   — the BC-structured space says *how* the boundary closes, the
   module supplies *what*. Decide together with the Robin/mixed-BC
   open question in [`../../specs/grid/classes/spaces.md`](../../specs/grid/classes/spaces.md)
   (static structure vs dynamic BC data).

## Rejected alternatives

- **Shrinking codomains** (`Inner -> "interior centers"`): a zoo of
  offset spaces, misaligned arithmetic, and state spaces that
  shrink per step — breaks fixed-point time stepping.
- **Per-field defined-region masks**: duplicates the
  immersed-domain layer for a two-slot problem.

## Status

Non-blocking: task 1.8's claim reset confines the extrapolation to
single-consumption uses, where it is second-order correct. The
decision should land before ROADMAP 2.2+ registers boundary-aware
modules on bounded meshes. **Resolution proposed** (with the
Robin/mixed question, one principle for both) in
[`boundary_plan.md`](boundary_plan.md) — decisions R1/R2 there.
