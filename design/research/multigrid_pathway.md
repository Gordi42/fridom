---
status: frozen
date: 2026-07-17
---

# Multigrid pathway — grid-to-grid transfer on the new stack

Research report (see [`README.md`](README.md) for status); the input
behind [`../plans/active/multigrid_pathway_plan.md`](../plans/active/multigrid_pathway_plan.md).
Run 2026-07-17 as five parallel passes — three code maps, one
empirical sharding probe (forced 4 host cpu devices), one external
prior-art survey — against a detached worktree pinned at `731089fc`.
The probe scripts were throwaway (scratchpad, per the research-
worktree convention); every load-bearing observation is inlined here.
File:line references are as of the pinned commit.

## The question

Can multigrid solvers be built by reusing the existing grid /
function-space structures — construct coarser sibling `Grid`s and add
transfer methods between them — and what would that take? Sub-
questions: what breaks, how the multi-device (GSPMD) layer behaves
under grid transfers, and whether the approach is sound multigrid
practice at all.

## F1 — The framework substrate is favorable

- **Two grids at different resolutions coexist cleanly** in one
  process and one jit trace. `Grid` is fully static, identity-hashed,
  holds no arrays (nodes/measures/metrics/fractions recompute from
  callables), and has no global state keyed by grid; all caches are
  instance-level (`grid.py:308`, `mesh._refined_cache`,
  `immersed_domain.py:165`). Verified live: two grids under forced 4
  devices share the same interned mesh objects and device mesh, and
  fine-grid operators keep working after the coarse grid is built.
- **One exact resolution-change mechanism already ships**: the padded
  spectral transforms land fields on `mesh.refined(pad.factor)` — a
  different-sized mesh coexisting on the same decomposition
  (`operators/transform.py:391`, `dealias.py`). `refined()` accepts
  coarsening factors (`factor < 1`, `structured_1d.py:245`).
- **The transfer cannot be `.to()`**: field arithmetic and the whole
  `.to` conversion family are hard-walled to one grid
  (`GridMismatchError`, `scalar_field.py:935-943`; every conversion
  kind asserts `src.mesh is dst.mesh`). A grid-pair operator is the
  only shape that fits — and the coupling pre-design already names it:
  CS-15 `Regrid` (grid-pair-bound, built after both freezes), with
  §11.1 equating the coupling restriction to "the grid layer's
  cell-average restriction"
  ([`../specs/model/09_coupling_designfor.md`](../specs/model/09_coupling_designfor.md)).

## F2 — The solver seam is already open

- `ConjugateGradient` (`spatial/operators/krylov.py:184`) consumes
  `preconditioner=` as an opaque field→field callable; a V-cycle slots
  in with zero CG-core changes. Requirements it inherits: SPD in the
  measure-weighted L² product, pure closure inside `lax.scan`, static
  iteration count (CS-D2); the peeled first iteration
  (`krylov.py:74-81`) absorbs trace-time setup. The nullspace
  projection is pluggable per solve (`projection=`, IP-D6 wet-mean).
- **Coarse operators come by re-discretization, not Galerkin
  products**: the mapped and immersed Laplacians are matrix-free
  compositions of registry `diff`/`interpolate` rows plus coefficient
  fields derived from the grid (metrics by autodiff of the chart,
  fractions by re-quadrature of the analytic wet-volume callable) —
  `mapped_pressure.py:602-657`, `immersed_pressure.py:246-320`. A
  coarse `Grid` re-derives all of it.
- **Reusable primitives**: the tridiagonal-per-column banded kernel
  (`operators/banded.py`, used by the IMEX `VerticalDiffusion`) is the
  vertical line smoother; the same-mesh `Restriction`/`interpolate`
  staggering machinery is the structural template for transfer
  kernels.
- **The measured case** (from
  [`../plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md)
  §4b and [`../plans/done/immersed_partial_cells_plan.md`](../plans/done/immersed_partial_cells_plan.md)):
  steep mapped terrain (4.5× depth ratio) takes ~44-45 PCG iterations
  to 1e-10, resolution-independent, vs the 30-iteration default
  budget; genuine immersed partial cells take ~60+; the spectral
  preconditioner is 58% of a ~6.15 ms/iteration at 256³ on one A100.
  Line-Jacobi as the *sole* preconditioner was measured 4-20× worse
  than spectral — that measurement does not condemn it as an MG
  *smoother*, where the coarse-grid correction supplies the global
  horizontal coupling it lacks standalone.
- A second consumer exists: the hydrostatic implicit free surface's
  designed-for variable-`csqr` route is the same CG architecture in
  2D (`free_surface.py:528`, `hydrostatic_model_plan.md` §7).

## F3 — Decomposition layer: precedent, constraints, one hazard

- Iteration 1 shards exactly one coordinate axis over a 1-D device
  mesh (`tensor.py:179-218`); refined-mesh fields already reshard per
  their own `ceil(n/P)` blocking on the same mesh (`tensor.py:306-359`)
  — the hierarchy precedent.
- **Minimum coarse size per axis on P devices**: representability
  requires the last shard non-empty (`tensor.py:352-358`; on P=4,
  `n ∈ {5,6}` are already unrepresentable), and halo correctness
  requires last-shard cells ≥ halo+1 — but that second gate runs
  **only at fine-grid negotiation** (`decomposition.py:974`). A coarse
  mesh reusing a fine decomposition can pass storage construction and
  then **silently corrupt** in halo exchange (`tensor.py:1873` slices
  past true cells; no runtime guard on the sync path). Per-level
  grids negotiate fresh and avoid this; a validation gate makes it
  structural.
- When nothing is shardable, auto-selected devices fall back to one
  device; **explicitly requested devices raise**
  (`decomposition.py:700-707`). A hierarchy needs all levels on one
  device set, so negotiation needs a replicated fallback mode (the
  replicated `Layout({})` already exists).
- Shape-changing transfer is genuinely new machinery: `redistribute`
  only transposes same-shape layouts; the padded-even reblock frames
  (`even_shape`/`pad_even`, `tensor.py:811-938`) are the right
  substrate for the non-nesting fallback path.

## F4 — Empirical probe (forced 4 host cpu devices, GSPMD)

Restriction/prolongation/V-cycle on arrays sharded exactly as fridom
shards (verified against a real grid field):

1. Factor-2 restriction (strided or block-mean): output stays
   sharded, **zero collectives**, while per-shard extents divide by 2.
2. Prolongation: nearest = zero collectives; linear along the sharded
   axis = 3 `collective-permute` (halo-class, not gathers).
3. A 3-level V-cycle fuses into **one jit**, runs multi-device,
   **compute-bound**: 6 permutes per stencil sweep, 0 per restriction,
   3 per linear prolongation, exactly (90 total, matching the op
   count); communication volume is one ghost plane per exchange.
4. Below `size < devices` inside jit, GSPMD **silently
   auto-replicates** (redundant compute + one gather-class collective)
   — graceful degradation. Eager `device_put` at indivisible sizes
   raises `IndivisibleError`; a forced sharding constraint there is
   silently dropped to replicated.
5. **No in-jit agglomeration onto a device subset** — a sharding
   constraint naming fewer devices than the inputs is a hard error
   ("incompatible devices"); in-jit the only option is
   replicate-onto-all (`all-gather`). Eager cross-mesh `device_put`
   at a jit boundary works and is cheap at coarse sizes.
6. Field storage frame ≠ logical frame: logical 32³ lives in a
   (52, 36, 20) storage array (halos + ceil-block padding). Transfers
   must operate on logical `.data` and re-inject through grid/halo
   machinery, never write raw `_data`.

Corollary: keep levels sharded while the sharded axis divides
`P · 2^(sharded levels)`; replicate below a static threshold. The 32³
+5.5% unamortized-collective tail already measured for the
distributed transforms says coarse levels should not run distributed
spectral machinery anyway.

## F5 — External prior art (survey)

- **No mature JAX-native multigrid exists** to adopt: jax-cfd never
  built theirs (unmaintained; FFT/CG only); JAX-AMG (arXiv:2606.09001)
  is an AmgX/CUDA wrapper distributed by MPI rank; Veros uses
  BiCGSTAB+Jacobi. The one JAX-native geometric-MG precedent —
  Fast(er)PM (arXiv:2607.10983, no public repo) — validates the exact
  architecture: fixed-count V-cycles, whole cycle in one compiled
  program, sharded fine levels with explicit halo exchange, and
  **gather-to-replicated redundant solves on the deepest levels**
  (the PETSc `redundant`/`telescope` pattern under GSPMD).
- Oceananigans — the closest analogue — evaluated AMG (CPU-only PCG
  preconditioner, PR #2654) and runs FFT-preconditioned CG on GPU for
  immersed boundaries; MITgcm uses PCG everywhere with a **vertical
  tridiagonal line preconditioner** for cg3d. AMReX/HPGMG are the
  block-structured GMG references (GSRB/Chebyshev smoothers,
  piecewise-constant restriction, linear+ prolongation, rediscretized
  coarse operators, AMG escape hatch on cut cells).
- **Transfer-order rule** (Trottenberg §2.7-2.8; Hemker 1990): for a
  second-order operator, transfer orders must satisfy mP + mR > 2.
  Constant prolongation + cell-average restriction (the naive MAC
  pairing) sums to exactly 2 — marginal for V-cycles (documented
  W-cycle-only convergence). Trilinear cell-centered prolongation
  (9/16-3/16-1/16 weights in 2D) with its adjoint restriction passes,
  and P = Rᵀ in the weighted product transmits SPD to every level.
- **Anisotropy**: point smoothing + full coarsening provably degrades
  at ocean aspect ratios (ρ→1). The two endorsed cures are vertical
  line relaxation (+ standard coarsening) and vertical line relaxation
  + **horizontal semicoarsening** (Müller & Scheichl, operational NWP
  MG). Sigma-coordinate cross-derivative terms are handled by the same
  vertical line smoother (Adams & Smolarkiewicz 2001).
- **Cut cells**: tiny partial cells spread operator eigenvalues over
  ~15 orders of magnitude and break pointwise smoothers; mitigations
  are an `hFacMin`-style fraction floor (fridom has one), re-deriving
  geometry per level (fridom's analytic re-quadrature does this
  natively), Galerkin coarsening of the masked part as the fallback,
  and boundary-aware extra smoothing. Nullspace handling per level
  must stay the wet-volume-weighted mean, never point pinning.
- **Fixed cycle count as preconditioner**: one V(1,1) per CG
  iteration is the standard (hypre default). A fixed-count cycle with
  stationary symmetric smoothers is a *fixed linear operator* —
  plain PCG stays valid, no FGMRES/flexible-CG needed. Symmetry (not
  stationarity) is the constraint to engineer: symmetric smoothers
  (damped Jacobi / line-Jacobi / Chebyshev) or transposed pre/post
  ordering, plus adjoint transfer pairs.

## Implications

Carried into the plan
([`../plans/active/multigrid_pathway_plan.md`](../plans/active/multigrid_pathway_plan.md)):
build the grid-pair transfer layer first (dual-use with coupling
CS-15/§11.1); hierarchy = independent per-level `Grid`s (fresh
negotiation, no `refined_from` adoption leakage — the adoption chain
would let the fine grid silently accept coarse spaces); horizontal
semicoarsening + vertical line smoothing; order-2 adjoint transfer
pair; replicated coarse levels below a static threshold; V-cycle as a
fixed-count symmetric PCG preconditioner behind the existing
`preconditioner=` seam; the V-cycle build stays gated on the
steep-bathymetry / partial-cells workload trigger, de-risked by a
two-level spike.
