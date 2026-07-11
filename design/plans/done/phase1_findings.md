---
status: done
date: 2026-07-07
---

# Phase-1 validation findings (input to Phase 2)

From the ROADMAP 1.7 validation gate (2026-07-07): findings from
driving the grid layer through hand-rolled PDEs the way a user would.
Numerics passed at or beyond spec everywhere (FD order 1.99, FV
conservation 4.4e-16, WENO ENO-bounded, diffusion/Poisson ~1e-15,
99.1 % branch coverage); everything below is about contracts and
ergonomics, to be consumed by the Phase-2 design (ROADMAP 2.1/2.2).

## Contract findings

1. **Whole-step `jax.jit` is device-count invariant only to ~2 ulp,
   not bitwise** — XLA fuses the sharded and unsharded programs
   differently; eager op-by-op execution is exactly bitwise (that is
   the operator layer's tested contract). Likewise eager-vs-jit
   differs by ~1 ulp on one device. The model layer must not promise
   bitwise 1-vs-N equality for jitted runs; invariance tests must
   compare identically-compiled paths.
2. **`ScalarField` arithmetic resets metadata, and metadata is
   static treedef aux**, so `u + dt*du` on a *named* field changes
   the treedef and `lax.scan` rejects named-field carries
   (`VectorField` preserves component metadata and is fine).
   Candidate fixes for Phase 2: preserve metadata like VectorField
   does, or move metadata out of the treedef.
3. **Sync amplification** is tracked separately
   ([decomposition open questions](../../specs/grid/classes/decomposition.md#open-questions));
   the 1.7 runs used eager stepping and per-operator sync throughout.

## API-gap backlog (ergonomics; report-only in 1.7)

- `fr.grid.cartesian.Grid` (the 07_iteration1_api day-one
  `shape=/extent=` constructor) is still an empty stub — slipped
  through the wave assignments; users hand-build `IntervalMesh`
  factors.
- Transform classes (`Fourier`, `Sine`, `Cosine`, `Chebyshev`,
  `SpectralDerivative`, `PhaseShift`) and `NodeSet` are not
  re-exported at the `fr.grid.*` level; `grid.dispatch` is typed
  `object`.
- BC-structured nodal spaces have almost no operator rows (no
  diff/interpolate/integrate/multiply): bounded-BC diffusion is only
  expressible via the trig-transform Laplacian; `f.mean()` on a
  Neumann field raises.
- No first-class biased nodal stencil pair (upwinding needs the
  artificial-diffusion identity or the FV route).
- Coefficient-space fields have no product/power rows; spectral
  operator coefficients (−1/|k|²) force a drop to `.data`, and
  constant→coefficient broadcast is blocked (`kx² + ky²` is not
  expressible as field algebra).
- All-constant reductions return shape-(1,…,1) fields; an
  `.item()`-style accessor is missing.
- WENO5 needs a manual `grid.negotiate(halo=HaloSpec({...: 3}))`
  before field creation, discovered only via the halo ValueError
  (deep import of `HaloSpec` required).
- `ChebyshevMesh` geometry accessors (`evaluation_nodes`, `measure`,
  `init=`) are unimplemented — mixed-grid ICs pass hand-computed
  `data=`.
