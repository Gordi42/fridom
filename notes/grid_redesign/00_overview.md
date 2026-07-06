# Grid abstraction redesign

Status: **draft** (design phase, no implementation yet)
Author: Silvano Rosenau (with AI-assisted brainstorming)
Date: 2026-07-05

This document records the design of the new grid abstraction for FRIDOM.
It captures the decisions made so far, the rationale behind them, a
non-normative sketch of the intended API, and paper-validations of the
design against grid types we do not implement yet but must not preclude.

All code snippets are **illustrative, not normative**: names and exact
signatures are expected to change during implementation.

The notes are split across several files; this overview holds the
motivation, migration strategy, and precedents. See the document map
below for the rest. This design is the reference for ROADMAP Phase 4
(the function-space grid rewrite); the mapping from files to roadmap
tasks is in that document.

---

## Document map

The redesign notes are organized as follows (read roughly in order):

| File | Contents |
|------|----------|
| [`00_overview.md`](00_overview.md) | Motivation (section 1), migration strategy (section 8), precedents (section 9) — this file. |
| [`01_concepts.md`](01_concepts.md) | Core concepts (section 2): Mesh, FunctionSpace, TensorProductSpace, Field, Operator, Grid, coordinate data. |
| [`02_rules.md`](02_rules.md) | Rules (section 3): strict algebra, coefficient spaces, dispatch, shapes, the three boundary kinds, FV semantics, discretization. |
| [`03_api_sketches.md`](03_api_sketches.md) | Non-normative API sketches (section 4). |
| [`04_decomposition.md`](04_decomposition.md) | Domain decomposition (section 5). |
| [`05_validation.md`](05_validation.md) | Paper validation against five future grid types (section 6). |
| [`06_open_threads.md`](06_open_threads.md) | Open threads (section 7) — all resolved; a stub mapping each former thread to the section that now carries the decision. |
| [`07_iteration1_api.md`](07_iteration1_api.md) | Iteration-1 public API cheat sheet (section 10): the small surface a day-one user actually types. |

Section and sketch numbers are stable identifiers across the files;
cross-references between files are linked, references within a file are
left as plain "section X.Y".

---

## 1. Motivation: pain points of the current abstraction

The current `fr.grid.GridBase` and its cartesian implementation work well
for the uniform C-grid case but have structural limitations:

1. **Fat, partially enforced interface.** `GridBase` mixes geometry,
   Fourier transforms, domain-decomposition passthroughs, reductions,
   and *model physics* (`omega`, `vec_q`, `vec_p`). Every model package
   must subclass every grid just to attach its eigenmodes, and the
   hydrostatic model already deviates from the signature.
2. **Minimal staggering model.** Staggering is a per-axis
   `CENTER`/`FACE` enum stored in *field metadata*. The FFT (DST-I vs
   DST-II selection), water mask, mesh offsets, and cumulative
   integrals each re-implement position handling independently.
3. **Boundary conditions live in field metadata** (`bc_types`), and
   `SpectralDiff` mutates them (DIRICHLET <-> NEUMANN flip) as a side
   effect — a symptom that BCs really belong to the basis.
4. **`topo` (reduced dimensions) is bolted on**: many operations carry
   `TODO: partial topo not supported` (fft, sync, diff, interpolate).
5. **The transform API does not fit real solvers.**
   `RFFTPressureSolver` bypasses `grid.fft` entirely and talks to the
   decomposition directly (rfft, axis subsets, custom dct).
6. **Spectral-consistency machinery is homeless.**
   `discrete_spectral_operators` (`k_hat`, `one_hat`, ...) lives in
   `grid.cartesian` but is used framework-wide (projections,
   eigenvectors, pressure solvers); it is part of the grid contract
   without being in the interface.
7. **One global halo integer** for all axes; the grid rebuilds the
   decomposition on halo mismatch.

The redesign targets, additionally, capabilities the current design
cannot express at all:

- **Mixed discretizations per axis**, e.g. uniform finite volume in the
  horizontal x Chebyshev-Galerkin in the vertical.
- **Non-tensor factors**, e.g. unstructured horizontal meshes combined
  with a structured vertical (ICON/FESOM-style prisms), or spherical
  geometry with metric terms.
- **General staggering** beyond per-axis CENTER/FACE (A/B/C grids,
  edge-normal velocities on triangular C-grids).

---

## 8. Migration strategy

- Build the new abstraction as a **parallel subpackage** with its own
  mirrored tests (95% branch coverage applies from the start). It lives
  under the transitional name `framework.grid2` while the old grid
  still works, and is **renamed to the canonical `framework.grid` once
  the old grid is deleted**. The assembly root is the model-agnostic
  `fr.Grid` (`meshes=`/`names=`), with `fr.grid.cartesian.Grid`
  (`shape=`/`extent=`/`periodic=`) a convenience subclass — the split is
  fixed in [section 2.6](01_concepts.md#26-grid--the-assembly-object).
  The mesh factors live
  in `fr.meshes` and the free-standing operators in `fr.operators`
  (plural collection namespaces, matching `fr.modules`/`fr.time_steppers`);
  function spaces need no top-level namespace — they are produced by
  mesh factories (`mx.center`, `mz.galerkin(...)`).
- Port model packages one by one (nonhydro first, as the most complete
  consumer), keeping the old `framework.grid` working throughout.
- Delete the old grid last, together with `Position`, `bc_types`,
  `topo`, `is_spectral`, and `FFTPadding` call sites.

## 9. Precedents

- **shenfun** — tensor products of 1D bases; closest to the
  mesh-factor idea; source for Shen bases with built-in BCs.
- **Dedalus v3** — coordinates / distributor / bases separation;
  fields carry their bases; transform scheduling.
- **Firedrake + Gusto** — FEEC view of staggering (compatible FE);
  naming compatibility target, machinery intentionally not adopted.
- **Oceananigans.jl** — `Field{Face, Center, Center}` per-axis
  staggering; essentially the current FRIDOM design and a demonstration
  of its ceiling; also: immersed boundaries.
