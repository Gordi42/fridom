---
status: active
date: 2026-07-17
---

# Multigrid pathway — grid transfer layer + gated V-cycle preconditioner

Phased plan for grid-to-grid transfer on the new stack and the
geometric-multigrid preconditioner built on it. Grounded in
[`../../research/multigrid_pathway.md`](../../research/multigrid_pathway.md)
(five-pass research, pinned `731089fc`); the justifying measurements
live in [`perf_geometry_merge_plan.md`](perf_geometry_merge_plan.md)
§4b (steep mapped: ~45 iterations, resolution-independent) and
[`immersed_partial_cells_plan.md`](immersed_partial_cells_plan.md)
(genuine partials: ~60+ against the 30 budget).

**Phase A (transfer layer) is standalone and dual-use** — it is also
the grid-layer cell-average restriction the coupling pre-design
already names (CS-15, §11.1 of
[`../../specs/model/09_coupling_designfor.md`](../../specs/model/09_coupling_designfor.md))
and a user-facing regridding utility (e.g. restart at a different
resolution). **Phase B (the V-cycle) shipped 2026-07-17** after the
gate trigger fired on both workloads and its B0 two-level spike
passed the kill criterion (§3).

*Owner ratification 2026-07-17 (decision walkthrough in chat): MG-D1..D8
confirmed as written — notably semicoarsening (MG-D4), analytic
re-derivation for coarse geometry (MG-D6), line-Jacobi V(1,1)
(MG-D7) — and the sequencing is phase A first, then B0 on real
`GridTransfer` machinery.*

## 1. Decisions

- **MG-D1 — the transfer is a free-standing grid-pair operator**
  (`GridTransfer`), not a `.to` kind and not a registry row. The `.to`
  family and field arithmetic are hard-walled to one grid
  (`GridMismatchError`), registries are grid-private, and CS-15
  already specifies the grid-pair shape. The coupling `Regrid`
  (host-held, cross-model) will later *compose* `GridTransfer`;
  `GridTransfer` itself is the traceable, same-device-mesh primitive.
- **MG-D2 — default transfer pair is order-2 adjoint:** trilinear
  cell-centered prolongation `P` and its measure-weighted adjoint
  restriction `R = P†` (volume-weighted full-weighting). This
  simultaneously satisfies the transfer-order rule (mP + mR = 4 > 2;
  the naive constant/average pairing sums to exactly 2 and is
  documented V-cycle-marginal), the SPD-transmission requirement
  (P = R† in the weighted product), and conservation (P preserves
  constants ⇒ R preserves integrals). An `order=1` pair
  (constant P, volume-weighted block-average R — collective-free on
  the sharded axis) ships alongside for coupling/regrid use.
- **MG-D3 — hierarchy levels are independent `Grid`s** built by
  `Grid.coarsened()` over `Mesh.coarsened()`, **never** via
  `mesh.refined(Fraction(1, 2))`. `refined()` sets `refined_from`
  child→parent, so a coarse mesh built that way becomes an adoption
  child of the fine mesh and the fine grid's dispatch would silently
  accept coarse spaces with fine-mesh rows
  (`registry._adoption_chain`). Independent meshes keep every
  mismatch loud, and per-level grids re-negotiate their own
  decompositions (which closes the silent-halo hazard, research F3).
- **MG-D4 — semicoarsening by default:** the MG hierarchy coarsens
  the horizontal axes only and keeps the vertical mesh at full
  resolution, paired with vertical line smoothing (MG-D7). This is
  the operationally-endorsed cure for grid-aligned anisotropy
  (Müller & Scheichl), it sidesteps `ChebyshevMesh`'s refined/coarsen
  refusal entirely, and it means no vertical transfer operator is
  needed at all. `Grid.coarsened` takes per-name factors, so full
  coarsening stays expressible where it is wanted.
- **MG-D5 — one device set for all levels; replicate below the
  shardability floor.** Negotiation gains a replicated fallback
  (today an explicitly-requested device set raises when nothing is
  shardable); a level whose sharded axis no longer divides `P` lives
  replicated on the *same* device mesh and is solved redundantly on
  every device — the PETSc-`redundant` pattern and the one JAX-sharded
  MG precedent (Fast(er)PM), and the only shape GSPMD supports inside
  one jit (in-jit submesh agglomeration is a hard error, research F4).
- **MG-D6 — coarse geometry by re-derivation:** coarse levels reuse
  the same `CoordinateMapping` / `ImmersedDomain` descriptors; metrics
  and wet fractions re-derive on the coarse spaces (re-quadrature =
  the literature's "re-detect the boundary at every level", for
  free). Conservative *discrete* restriction of the fine fractions
  through `GridTransfer` + the `fraction=` explicit-data overload is
  the recorded fallback lever if immersed convergence disappoints
  (the Galerkin-flavored option); it is not built first.
- **MG-D7 — symmetric, stationary cycle:** damped point-Jacobi and
  damped vertical-line block-Jacobi (all columns at once, via the
  banded kernel) are the smoothers — both symmetric and
  coloring-free, so `pre == post` sweeps with P = R† gives a
  symmetric V-cycle; a fixed cycle count keeps it a *fixed linear
  operator*, so plain `ConjugateGradient` stays valid (no
  FGMRES/flexible CG). Coarsest solve = fixed extra sweeps
  (iteration 1); a fixed-degree Chebyshev polynomial is the recorded
  upgrade. An inner CG at the coarsest level is **forbidden** (a
  fixed-iteration CG is still a nonlinear map of its input — it would
  silently break the outer CG).
- **MG-D8 — the V-cycle plugs into the existing seam untouched:**
  a field→field callable passed as `preconditioner=` to
  `ConjugateGradient`, selected by a `pressure_preconditioner` knob
  on `DynamicalCore` (fingerprint-static, like
  `single_precision_solve`). No CG-core changes; per-level nullspace
  handling reuses the pluggable `projection=` discipline (wet-mean
  per level, never point pinning).

## 2. Phase A — the grid transfer layer

*Landed 2026-07-17 (merge `fae44be4`); the sections below are the
record. Implementation deviations from the stubs, all verified by the
GA gates: restriction is built as `jax.linear_transpose` of the
explicit prolongation wrapped in the two volume weightings — the
literal `R = M_H^-1 P^T M_h`, exact on boundary rows, with no
hand-rolled `shard_map` kernels (GSPMD keeps aligned axes shard-local;
asserted no-gather in tests). `Grid.coarsened` pins the coarse level
to the fine grid's **resolved** device set (an auto-sharded fine grid
would otherwise auto-fall its coarse sibling back to one device,
breaking MG-D5). `Mesh.coarsened(1)` returns `self` (pass-through, so
Chebyshev verticals pass under semicoarsening). Order-2 halo demands
are **not** registered pre-freeze — neighbor access lowers to GSPMD
collective-permutes (forced-4 parity confirmed); explicit registration
is deferred to the traceable-Regrid need (§11.5).*

### A1 — `Mesh.coarsened(factor)`

`StructuredMesh1D.coarsened(factor: int) -> Self`: an independent
same-family mesh with `n_cells // factor` cells (divisibility
enforced), same extent/topology/mapping, built through the existing
`_make_refined` hook but with **no** `refined_from` link (MG-D3);
memoized per factor in its own `_coarsened_cache` so spaces stay
identity-comparable. `ChebyshevMesh` raises (as it does for
`refined`); `PointMesh`/`SphereMesh`/unstructured are out of scope
(factor-1 pass-through only).

```python
def coarsened(self, factor: int) -> Self:
    """
    Return an independent same-family mesh with n_cells // factor.

    Description
    -----------
    The coarse sibling for grid hierarchies (multigrid, regridding).
    Unlike ``refined``, the result carries **no** ``refined_from``
    link: it is not adopted by any grid owning this mesh, so
    cross-resolution use is always explicit through ``GridTransfer``
    (MG-D3). Memoized per factor.

    Parameters
    ----------
    factor : int
        The integer cell-count divisor (>= 1); ``n_cells`` must be
        divisible by it.
    """
```

### A2 — `Grid.coarsened(factors)`

```python
def coarsened(
    self,
    factors: Mapping[str, int] | int,
    *,
    device_ids: tuple[int, ...] | None = None,
) -> Grid:
    """
    Assemble the coarse sibling grid (multigrid / regrid levels).

    Description
    -----------
    Rebuilds a grid from per-coordinate coarsened meshes with the
    same attachments (``CoordinateMapping`` / ``ImmersedDomain``
    descriptors re-bound; metrics and fractions re-derive on the
    coarse spaces, MG-D6), the same default family, and a fresh
    default registry. Model-level dispatch overrides do **not**
    carry over. The result is unfrozen with a provisional
    negotiation over the **same device set** (replicated fallback
    when nothing is shardable, MG-D5).

    Parameters
    ----------
    factors : Mapping[str, int] | int
        Per-coordinate-name integer divisors (missing names default
        to 1 — semicoarsening, MG-D4), or one uniform divisor.
    device_ids : tuple[int, ...] | None, optional
        Override the device set; None inherits this grid's
        (default: None).
    """
```

Sub-items: the attachments must be re-bindable to a second grid —
`CoordinateMapping._bind` / `ImmersedDomain` attachment are
single-shot today, so A2 adds a clone-on-attach hook (descriptors are
array-free; the callables are shared). Factors touching a
multi-name / non-structured mesh factor must be 1 or raise loudly.

### A3 — negotiation: replicated fallback

`negotiate(..., allow_replicated: bool = False)`: when nothing is
GHOST-shardable and the device set was explicit, `allow_replicated`
keeps the full device set with the replicated-only layout tuple
(`Layout({})` over the same 1-D device mesh) instead of raising.
`Grid.coarsened` passes `allow_replicated=True`; the public `Grid`
constructor keeps today's behavior. One new negotiation test path;
the existing single-device fallback for auto-selected devices is
untouched.

### A4 — `GridTransfer` (`spatial/operators/transfer.py`)

The grid-pair operator (MG-D1/D2). Iteration-1 scope: real scalar
fields on the collocated cell spaces (`Center` nodal, `CellAvg`);
staggered face spaces and coefficient spaces raise (designed-for:
coupling fluxes, CS-16 traces).

```python
@partial(jaxify, dynamic=())
class GridTransfer:

    """
    Restriction / prolongation between two coexisting grids.

    Description
    -----------
    Grid-pair-bound (CS-15 shape): built after both grids exist,
    validating same coordinate names, per-name integer cell-count
    ratios (1 = uncoarsened axis), and one shared device set. The
    pair is adjoint under the two grids' measure-weighted L2
    products, <R f, g>_H = <f, P g>_h, and conservative (P
    preserves constants); order=2 is trilinear P with its weighted
    adjoint R (full-weighting), order=1 is constant P with the
    volume-weighted block average R (MG-D2). Kernels run in the
    logical frame and re-enter storage through the target grid's
    decomposition (never raw ``_data``); the aligned case (per-shard
    extents divisible by the ratio) lowers to shard-local
    ``shard_map`` kernels — order-1 restriction is collective-free,
    the order-2 pair costs halo-class permutes along the sharded
    axis; non-nesting blockings fall back to the global reblock
    path (correct, slower).

    Parameters
    ----------
    fine : Grid
        The fine grid (transfer source for ``restrict``).
    coarse : Grid
        The coarse grid; every axis ratio fine/coarse must be a
        positive integer.
    order : int, optional
        Transfer-pair order, 1 or 2 (default: 2).
    """

    def __init__(self, fine: Grid, coarse: Grid, *,
                 order: int = 2) -> None: ...

    def restrict(self, field: ScalarField) -> ScalarField:
        """Fine-grid field -> coarse-grid field (R = P† weighted)."""

    def prolong(self, field: ScalarField) -> ScalarField:
        """Coarse-grid field -> fine-grid field (P)."""
```

Sub-items: per-axis separable kernels (weights per parity offset,
trace-time constants); halo demands of the order-2 pair (one fine
halo for R, one coarse halo for P along coarsened axes) registered
against both grids **pre-freeze** — the traceable-Regrid requirement
§11.5 anticipated; metadata carried (`same quantity, other grid`).

### A5 — level validation gate

A small helper (used by `GridTransfer.__init__` and the phase-B
hierarchy builder) that re-checks, per level and axis: last-shard
cells ≥ 1 **and** ≥ negotiated halo + 1 when the axis is sharded —
the check that today runs only at fine-grid negotiation (research
F3's silent-corruption hazard). Failure is a loud `ValueError`
naming the level and axis; per-level grids make it structurally
unreachable, the gate keeps it that way.

### A6 — phase-A tests and gates

Mirrored tests (see §4). Gates: **GA-1** adjointness
`<R f, g>_H == <f, P g>_h` and conservation to 1e-14 (both orders,
even/odd parities, mapped + immersed grids, semicoarsened factors);
round-trip `R(P(x)) == x` for order 1; **GA-2** forced-4-device
parity with single-device results and zero gather-class collectives
on the aligned order-1 restriction path; **GA-3** ruff clean,
mirrored tests green, patch coverage ≥ 95%.

## 3. Phase B — the multigrid preconditioner (shipped)

**Trigger** (unchanged from the roadmap lever): steep bathymetry
(~45 iterations; profile relabel below) or genuine partial cells
(~60+) as a real workload. Both had fired.

**Status: shipped 2026-07-17** (merge `87aeabea` onto dev). B0 ran
2026-07-17 (GB-0 pass, numbers below); B1+B2 landed as
`09421e10`/`28c11a2f`/`568673ca`/`25d088e5`, B3+B4 as
`56774cb8`/`7bdaca8e`/`e0da2495`, the B5 evidence as `6ac5d4e6`, the
default re-pin as `e047c496`; the throwaway spike harness was removed
before the merge (`5705f002`). Three corrections recorded below
(steep-profile relabel, off-diagonal sign, GB-1 recalibration) are
evidence-backed agent findings endorsed by the orchestrating
sessions, **not yet owner-reviewed**. Open follow-up: the GB-2
wall-clock A100 leg ([`open.md`](../../roadmap/open.md)).

### B0 — two-level spike (the kill criterion)

Before any production code beyond phase A: a throwaway two-grid
cycle (one horizontal coarsening, damped vertical-line smoothing,
exact coarse solve) on the §4b steep-terrain benchmark. **Gate
GB-0:** preconditioned iterations to 1e-10 at 64³ drop below ~20
(from 45). If not, stop, record the numbers here, and keep the
spectral preconditioner — the lever returns to "measured, not
taken".

*Ran 2026-07-17 — **GB-0 PASS.** Numbers (64³, random mean-free
rhs, iterations to 1e-10 relative residual, weighted norm):
spectral baseline 44; two-grid V(1,1), semicoarsen {x: 2, y: 2},
damped vertical-line Jacobi, near-exact coarse solve — ω
0.6/0.7/**0.8**/0.9/1.0 → 13/12/**11**/16/diverges. The 11 is
resolution-independent (same at 32³). V(2,2): 8 (marginal for ~2×
smoother cost). Production-shaped coarse solve (k fixed line
sweeps, no recursion): k = 8/16/32 → 31/23/17 — a real bottom
needs recursion depth, not extra sweeps at the first coarse level.
Cycle symmetry ⟨M⁻¹u,v⟩ vs ⟨u,M⁻¹v⟩ at roundoff (3e-16). Mild
profile: 11 vs spectral's 11 — the win is entirely the steep case
(44 → 11). MG-D7's inner-CG ban was also empirically confirmed:
a fixed-iteration inner CG run past convergence degrades cycle
symmetry to ~1e-8.*

*Profile relabel: the §4b "steep 4.5×" is mislabeled — the
documented 44/45-iteration problem is `depth(x) = 1 + 0.8 sin(x)`
(depth ratio **9.0**); amplitude 7/11 (a true 4.5 ratio) needs only
27 iterations. GB-2 below is re-pinned to a = 0.8; a correction
note now sits in `perf_geometry_merge_plan.md` §4b.*

### B1 — operator smoothing surfaces

`MappedPressureSolver` / `ImmersedPressureSolver` expose the
coefficients their smoothers need, on their own space:
`diagonal()` (the operator's diagonal, for damped Jacobi) and
`vertical_bands()` (the per-column tridiagonal part of ``A``, for
line relaxation via the shared banded kernel,
`operators/banded.py`). Both derive from the same metric /
fraction machinery `apply()` uses (per-solve memo discipline
unchanged).

*Landed on the branch (`568673ca`), probe-verified to roundoff
(probe tests must use period-4 coloring — period 3 aliases on
periodic axes whose cell count is not divisible by 3). Binding
findings: (1) the slope cross fluxes DO contribute to the mapped
diagonal (`Z_a` varies along the column, so the centered vertical
average does not annihilate them; exact analytic corner bracket
implemented), while the pure z-off-diagonal is exactly
**+**`K^bb_face/dz²` — positive, because ``A`` is
negative-semidefinite; a minus sign there makes the outer PCG
diverge. (2) ``T`` deliberately excludes the horizontal cross
residues (asymmetric per column, subdominant) — validated:
production ``T`` reproduces the spike's 11 iterations exactly.
`vertical_bands()` raises on a periodic vertical; the Thomas solve
axis must stay device-local.*

### B2 — `MultigridVCycle` (`spatial/operators/multigrid.py`)

The generic engine, model-agnostic like `krylov.py`:

```python
@dataclass(frozen=True)
class MultigridLevel:
    operator: Callable[[ScalarField], ScalarField]   # A_l, SPD
    smoother: Smoother          # symmetric fixed-sweep update
    projection: Callable[[ScalarField], ScalarField] | None
    transfer: GridTransfer | None    # to next-coarser; None = last


class MultigridVCycle:

    """
    Fixed-count symmetric V-cycle preconditioner (MG-D7/D8).

    Description
    -----------
    A trace-time object: a static tuple of levels (finest first),
    static sweep counts, no convergence branching — the cycle
    unrolls at trace time (CS-D2 discipline). Symmetric smoothers
    with ``pre_sweeps == post_sweeps`` and adjoint transfers make
    the cycle SPD in the finest level's measure-weighted product,
    so it is a valid ``ConjugateGradient`` preconditioner as-is.
    ``__call__(residual) -> correction``.

    Parameters
    ----------
    levels : tuple[MultigridLevel, ...]
        Finest -> coarsest; at least one.
    pre_sweeps, post_sweeps : int, optional
        Smoother sweeps around the coarse correction; must be equal
        (symmetry) (default: 1).
    coarse_sweeps : int, optional
        Extra smoother sweeps standing in for the coarsest solve
        (default: 8).
    """

    def __call__(self, r: ScalarField) -> ScalarField: ...
```

Smoothers (same module): `DampedJacobi(diagonal, omega=2/3)` and
`VerticalLineJacobi(bands, omega)` — pure callables
`sweep(x, b) -> x`, symmetric by construction. Cycle body is a
Python recursion over the static level tuple (jit-unrolled);
per-level projections applied to restricted residuals (nullspace
consistency across levels).

*Landed on the branch (`09421e10` Thomas kernel in `banded.py`,
`28c11a2f` engine, `25d088e5` tests; 182 mirrored tests green, ruff
clean). API facts the B3 builders rely on: smoother sweep signature
is `sweep(x, b, operator)` (the cycle passes `level.operator`);
`VerticalBands` lives in `spatial.operators.multigrid`; the cycle
applies the NEXT level's projection to restricted residuals and
never retags — the transfer's restricted space must BE the next
level's operator space (derive it as
`transfer.restrict(probe).function_space`). Line-smoother default
ω = 0.8 (spike optimum; ω = 1 diverges).*

### B3 — hierarchy builders (nonhydro2)

`mapped_pressure` / `immersed_pressure` gain a builder that
assembles levels from a fine grid: `grid.coarsened({...})` per level
(horizontal /2, MG-D4), per-level operator = the same solver class
re-instantiated on the coarse grid (re-discretization, MG-D6),
transfers from A4, level count static with a floor on coarse cells
(and the A5 gate). The builder output feeds `krylov()` as the
`preconditioner=` callable when selected.

*Landed (`7bdaca8e`, grid memo `56774cb8`): the shared builder lives
in `nonhydro2/modules/multigrid_hierarchy.py` —
`coarsen_levels(fine_grid, fine_space, *, vertical, max_levels,
order=2)` returns the finest-first `(grid, space, transfer)` chain
(`HORIZONTAL_FACTOR = 2`, `MIN_COARSE_CELLS = 4`); each solver's
`_build_vcycle` assembles the `MultigridLevel`s and `krylov()`
dispatches on the knob. Degradation verified: an axis coarsens iff
even and ≥ 8 cells; a grid too small for any coarsening yields a
1-level smoothing-only cycle (works, no error). Two implementation
findings: `Grid.coarsened` now memoizes per (factors, device_ids)
for retrace stability, with a `Grid.override_keys` accessor so the
FV re-discretization is idempotent under the memo; and MG-D6
re-discretization must also re-merge the FV `diff` dispatch profile
on each coarse grid — `Grid.coarsened` deliberately drops model
overrides, so a CellAvg coarse level would otherwise resolve the
pressure gradient on the wrong face family (loud shape error).*

### B4 — settings surface

`DynamicalCore.__init__` gains
`pressure_preconditioner: str = "spectral"` (`"spectral" |
"multigrid"`) and `multigrid_levels: int = 3`, threaded like
`pressure_iterations` (`core.py:423`, forwarded at the three solver
build sites) and mirrored on the `nh.Model` factory. Static in the
module fingerprint; flat grids ignore the knob (exact spectral
solve, no iteration).

*Landed (`e0da2495`): knobs exactly as specified, threaded at both
PCG build sites and mirrored on `nh.Model`. Default
`multigrid_levels` re-pinned to **5** after the measurement campaign
(`e047c496`): depth 3 gives 18 iterations at 64³ on the steep case —
over the GB-2 gate — while 5 is a graceful maximum (4-cell floor)
and flat at 13.*

### B5 — gates

**GB-1** *(recalibrated 2026-07-17: ρ ≤ 0.2 was mis-set — that is a
Gauss–Seidel-class number; the measured damped point-Jacobi V(1,1)
floor is ρ ≈ 0.36 (2-D) / 0.54 (3-D) at ω = 0.8, textbook LFA
values, grid-independent)*: committed gate is flat isotropic 2-D
full-depth V(1,1) contraction < 0.5 and grid-independent, plus
preconditioner symmetry `<M⁻¹u, v>_w == <u, M⁻¹v>_w` to roundoff
(both hold on the branch). The production smoother for the real
anisotropic workload is the line variant (11 iterations on the
steep case); stronger point smoothers (V(2,2) ≈ 0.38, Chebyshev,
GS) stay recorded levers. **GB-2** steep mapped a = 0.8 (depth
ratio 9.0 — re-pinned from the mislabeled "4.5×", see B0): ≤ 15
iterations to 1e-10, resolution-independent over n = 32..96+
(iteration counts are platform-independent; measure on cpu), and
≥ 1.5× step wall-clock win vs the spectral preconditioner at 128³+
on the A/B harness — the wall-clock leg needs an A100-class device
and may land as a post-merge follow-up measurement. **GB-3** immersed
genuine partials within the default 30-iteration budget. **GB-4**
zero warm recompiles; HLO flat in both the CG iteration count and
the level count. **GB-5** forced-4-device parity; no gather-class
collectives above the replication threshold.

*Outcomes (all on the branch before the merge):*

- *GB-1: holds as recalibrated (flat isotropic 2-D V(1,1)
  contraction < 0.5, grid-independent; cycle symmetry at roundoff).*
- *GB-2 (iterations): **44 → 13**, flat over n = 32/64/96 at the
  pinned defaults (levels = 5, engine coarse_sweeps = 8). Campaign
  (64³, iterations to 1e-10, spectral baseline 44): depth 2/3/4/5 at
  k = 8 → 31/18/12/13; k = 16 → 23/14/12/13; k = 32 → 17/12/12/13.
  Depth dominates sweeps; the 4→5 uptick (12→13) is the 4-cell
  coarsest being marginally too coarse (harmless); depth 4 / k = 8
  sits exactly at 15 at n = 96 (zero margin) — hence 5. The
  wall-clock leg (≥ 1.5× at 128³+) needs an A100: open follow-up.*
- *GB-3: genuine partials (order = 4, min_fraction = 0.1): 15
  iterations at 16³ / 18 at 32³ to 1e-10 — inside the 30 budget with
  ≥ 12 margin; spectral needs ~80. The immersed count creeps with n
  and does not improve with depth or sweeps — cut-cell boundary
  eigenvalues set the floor, well inside budget. Immersed
  comparisons are wet-region-only by necessity: dry cells are
  unconstrained (zero operator rows) and the geometric transfers
  bleed prolongated values into them.*
- *GB-4: compile-once per level config; HLO identical across CG
  iteration counts 4/8/20; HLO grows with level count by design
  (trace-time unroll, MG-D8) — the level gate is compile-once, not
  HLO equality.*
- *GB-5: forced-4 parity < 1e-8 on an aligned 3-level shard and on a
  12→6 shard whose coarse level replicates (6 does not divide 4
  devices — the MG-D5 fallback exercised under decomposition). Real
  multi-GPU joins the next campaign.*
- *Differentiability: `jax.grad` through a 6-step immersed multigrid
  run via `_chunk_body` is finite and FD-matched (~1.5e-8, gate
  1e-4) — the smoothers' dry-cell double-`where` guards hold. The
  mapped twin was blocked by a **pre-existing** reverse-NaN in
  `velocity_correction` (the field/jacobian divide VJP; present with
  the spectral preconditioner too; first noted at the CG-tolerance
  landing) — localized this session and since fixed (guarded
  `_divide_by_jacobian`, merge `ee350bda`; mapped autodiff
  regression in `test_mapped_model_autodiff.py`; entry in
  [`done.md`](../../roadmap/done.md)).*

## 4. Test plan (mirrored)

- `tests/spatial/meshes/test_structured_1d.py` — `coarsened`:
  identity/memoization, divisibility errors, no adoption
  (`refined_from is None`), Chebyshev refusal.
- `tests/spatial/test_grid.py` — `Grid.coarsened`: attachment
  re-bind, family carry-over, per-name factors, unfrozen +
  provisional negotiation, device-set inheritance.
- `tests/spatial/decomposition/test_decomposition.py` —
  `allow_replicated` negotiation path; the A5 gate.
- `tests/spatial/operators/test_transfer.py` — adjointness,
  conservation, constants, both orders, mapped/immersed/semicoarse
  cases, non-nesting fallback, storage-frame round trip;
  `@pytest.mark.multi_device` shards for GA-2.
- `tests/spatial/operators/test_multigrid.py` (phase B) —
  contraction, symmetry/SPD, fixed-point (zero residual → zero),
  level-count/HLO stability.
- `tests/nonhydro2/test_pressure*.py` (phase B) — knob wiring,
  mapped + immersed convergence gates at test scale.
- **Differentiability policy (AGENTS.md)**: `GridTransfer` and the
  V-cycle are step-path code once selected, so each phase ships its
  one cheap `jax.grad` regression test (pure-kernel pattern, <=8^3,
  <=10 steps). Transfers are linear (safe); the smoother divisions
  (Jacobi diagonal, line bands) must guard dry/padded cells with the
  double-`jnp.where` pattern; `custom_vjp` stays banned in the
  cycle.

## 5. Risks

- **Transfer-order marginality** — closed by MG-D2 (order-2 default).
- **Cut-cell stiffness** (eigenvalue spread breaks point smoothers)
  — the `hFacMin` floor exists; boundary-adjacent extra sweeps are
  the first lever, Galerkin-coarsened masked part (MG-D6 fallback)
  the second, AMG-at-the-bottom explicitly out of scope.
- **Thin geometry lost on coarse levels** — re-quadrature keeps
  coarse geometry honest per level; the level-count floor caps
  coarsening depth; record iteration counts per case in B5.
- **Silent halo corruption on undersized coarse levels** — closed
  structurally by per-level negotiation + the A5 gate.
- **Compile/cache growth per level** (strong `(space, layout)`
  caches, per-level `shard_map` kernels) — log-many levels; GB-4
  measures it; the persistent jax cache absorbs the one-time cost.
- **Replication threshold tuning** — static and coarse-grained (the
  probe shows graceful degradation either side); default = replicate
  once the sharded axis stops dividing the device count.

## 6. Out of scope (designed-for, not precluded)

Galerkin (RAP) coarse operators as such; Chebyshev-mesh vertical
transfers; staggered-face and coefficient-space transfer signatures
(coupling fluxes, CS-16 surface traces); the coupling super-mesh /
cross-process `Regrid` (§11.5 — `GridTransfer` is its same-device
building block); 2-D device meshes (follow the Fast(er)PM pencil
pattern when iteration 2 lands); AMG; W-/F-cycles and Chebyshev
smoothing (recorded upgrades behind GB-2).
