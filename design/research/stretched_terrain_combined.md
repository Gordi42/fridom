---
status: frozen
date: 2026-07-17
---

# Stretched + terrain-following combined — does the J-weighting compose with the physical-width division?

Research report (see [`README.md`](README.md) for status). Question,
from the roadmap's FV section: a grid that is both
`MappedIntervalMesh`-stretched and terrain-following "would J-weight
the conservative form on top of `flux_diff`'s physical-width
division; correctness there is unverified" (scoping §13 follow-up).
What actually happens on such a grid, what must change to support it,
and is there one unified method that serves the hydrostatic and
nonhydrostatic models simultaneously? Method: design-history sweep
(F0–F6 scoping, mapped-Jacobian spike, high-order mapped plan, ALE
record, hydrostatic plan), code sweep (measure/metric machinery, FV
rows, mapped advection and pressure, hydrostatic core), and a
numerical probe battery on live combined grids (constructibility,
factor trace, conservation, constancy, operator symmetry, hydrostatic
seams). AI-assisted; every load-bearing file:line and number was
produced on `dev` at `b77f8582`; the pressure-preconditioner claims
were re-verified after merging the multigrid phase-B dev head
(`87aeabea`) into this branch (§3).

## Verdict

**The feared double-count does not exist.** On a combined grid the
two geometries factor exactly and complementarily: the stretching
lives entirely in the mesh's physical widths (`grid.measure`, which
`flux_diff`/`integrate`/`CumulativeIntegral` divide/weight by), and
the chart Jacobian `J = d<mapped>_d<base>` from `grid.metric` carries
*only* the terrain factor — probed: `J == H(x)` to `0.0`, constant
along the stretched column. F5's conservative J-weighted FV tracer
advection is therefore **already correct and exactly conservative on
the combined grid** (probed budget residual `-8.9e-16`), with no code
change.
The composition was designed for: `MappedIntervalMesh` explicitly
rules "cross-factor mappings (terrain-following) are grid-level
metric data, not mesh structure" (`mapped_interval.py:44-48`), and
the chart's base coordinate is the mesh's own (stretched) coordinate,
so the total vertical metric `dz_p/ds` never needs to exist as one
object.

What is actually broken is downstream, and different per model:

- **nonhydro2 (both families): the mapped pressure solve dies on a
  stretched column.** The spectral preconditioner cannot build
  (`SpectralSolve` needs a `transform`; `MappedIntervalMesh` has
  none → cryptic `DispatchError`), and the operator's corner cross
  hops lose their exact transpose pairing under the non-uniform
  measure (probed rel asymmetry `2.6e-5` vs ≤1e-12 on a uniform
  column) — the SPD/CG license the F5 proof carried "under the
  uniform computational measure" does not extend. Both are fixable:
  the measure-adjoint down-hop restores exact symmetry (probed to
  `3.7e-16`, recipe in §3), and the preconditioner needs a
  measure-aware column route (the multigrid V-cycle is the candidate
  once its column machinery reads measure widths — today it raises
  on a stretched factor too, §3). No capability gate catches the
  combination: the grid silently auto-resolves `family="fv"` and
  fails at the first projection.
- **hydrostatic: no terrain support at all** (stretched-only is
  already exact through the measures) — and it fails *silently*: on
  a terrain grid `hy.Model` builds and runs with no complaint while
  the diagnosed `p_hyd` omits the column Jacobian entirely (probed:
  27% relative error at `H ∈ [0.7, 1.3]`). The chart metric is
  missing at four scattered, currently metric-free sites: the
  `p_hyd` column integral, the `w` diagnosis (vertical increment
  *and* the slope term of its horizontal divergence), the explicit
  baroclinic pressure gradient (the classic sigma-coordinate
  pressure-gradient-error site), and the free-surface depth mean /
  2D barotropic solve (physical depth `H(x,y)` ≠ computational
  extent). The `CumulativeIntegral(jacobian=...)` seam the
  hydrostatic plan names is designed but **not wired** for `maps=`
  terrain grids (§4).

**Unified method: yes — one convention, per-consumer application.**
The convention that already governs the working parts and extends to
every broken one: *stretching is read from `grid.measure` (never
re-derived), terrain from `grid.metric` rows (`J`, `1/J`, slopes
`Z_i`, `sqrt_g`), J-weighting is applied module-side (operators keep
no params seam), and every weight/divisor pair is matched so
telescoping survives*. The shared advection family
(`fr.model.modules.advection`) already implements it and serves both
models. What remains is per-consumer: the nonhydro pressure operator
needs the measure-adjoint corner hop plus a non-spectral column
preconditioner; the hydrostatic sites need module-side J-weighting
(the designed `jacobian=` seam is unwired — §4), slope-corrected
horizontal derivatives, and a variable-depth free surface.
Sections 6–7: options and owner calls.

## 1. Why a double-count was feared

Scoping §13 correction 1 established two *different* readings of the
FV DOF, one per geometry:

- **Pure terrain** (uniform meshes + `CoordinateMapping`):
  `flux_diff` divides by the *computational* cell width (a constant),
  so the DOF is the **chart-cell average**; physical conservation
  `∫q dV = ∫Jq dξ` is delivered by the explicit J-weighted advection
  form, not by DOF typing.
- **Pure stretch** (`MappedIntervalMesh`): `flux_diff` divides by the
  *physical* per-cell width field, so the DOF is the
  **physical-volume average**; no J-weighting exists or is needed.

On a combined grid both mechanisms are active at once — the measure
is a genuine per-cell field *and* the module J-weights on top — and
no record reconciled the two readings. If `J` had absorbed any part
of the stretching (or the measure any part of the terrain), the
vertical metric would be counted twice.

## 2. Why it does not double-count (code + probe)

The layering is exact, not approximate:

1. **Widths.** `grid.measure` materializes per-factor 1D widths from
   the *mesh's own* `coordinate_map` only (`grid.py:1988-2039`,
   `_mapped_measure_vector`); the chart never enters the measure. On
   the combined grid the vertical measure is the stretched
   base-coordinate width `w_i` (staggered differences of the mapped
   node positions).
2. **Chart metrics.** `grid.metric` derives `J = d<mapped>_d<base>`
   by `jax.jvp` of the map callable *at the mesh's evaluation nodes*
   (`coordinate_mapping.py:455-484`). Because the map's base
   coordinate is the mesh's (stretched) coordinate, `J` is the pure
   terrain factor. Probed on `maps={"zp": lambda sigma, H: ...}` over
   a tanh-stretched σ column: `max|J − H| = 0.0`, and `J` has zero
   variation along σ (`ptp = 0.0`).
3. **The vertical flux term, factor by factor**
   (`advection.py:2408-2491`): the code computes
   `(1/J_i) · ((J F)_{i+1/2} − (J F)_{i−1/2}) / w_i` — flux
   J-weighted at faces before `flux_diff`
   (`advection.py:2481-2485`), division by the physical width `w_i`
   inside `flux_diff` (`flux_diff.py:391-395` via
   `divide_by_codomain_measure`, `staggering.py:539-584`), outer
   `1/J` after (`advection.py:2489-2491`). The physical `z_p`-width
   of cell *i* is `J_i · w_i`, so the product is exactly the physical
   flux divergence: the two divisors are complementary factors of one
   Jacobian, not two Jacobians.
4. **Conservation telescopes for the same reason:** `integrate`
   weights by the same `w_i` that `flux_diff` divides by, and the
   J-weighted budget multiplies by the same `J_i` that the tendency
   divides by, so the cell sum collapses to boundary fluxes:
   `integrate(J · tendency) = -8.9e-16` on the combined grid
   (probe, random fluxes) — the F5 invariant `∫(J·τ) = 0.0` extends
   unchanged.

Constructibility: the combined grid builds without complaint,
`_fv_capable = True`, and `resolve_model_family` auto-selects
`"fv"` (`nonhydro2/modules/core.py:67-134, 289-362`) — no gate
inspects whether the mapped column's base mesh is stretched.

Constancy/free-stream: for `q ≡ 1` the flux-form tendency reduces to
the module's own discrete mapped divergence of the velocity, so
constancy holds exactly iff the projection enforces the same
discrete divergence. Probed: for a random staggered velocity pair,
`J ·` (advection tendency of `q ≡ 1`) equals the divergence leg that
`MappedPressureSolver.divergence` builds to rel `2.2e-16` — same
`flux_diff`, same cross hops, same J placement, differing only by
the outer `1/J` convention, which is finite-nonzero and cancels in
the free-stream condition. So a projection-accepted velocity
transports constants exactly on the combined grid.

## 3. What breaks in nonhydro2: the pressure solve on a stretched column

`MappedPressureSolver` (`nonhydro2/modules/mapped_pressure.py`)
assumes the mapped column rides a *uniform* computational mesh — in
its own words: "No measure weighting enters the cross hops: the
mapped column rides uniform computational meshes and all geometry
lives in the `K` coefficients" (`mapped_pressure.py:73-75`, echoed
`:416-418`, `core.py:307-308`). A stretched base column violates this
twice:

1. **Hard failure first: no preconditioner can run.** The default
   spectral preconditioner composes `SpectralSolve(Div @ Diag @
   Grad)`; `SpectralSolve` resolves a `transform` per axis, and
   `MappedIntervalMesh` deliberately has no spectral basis
   (`mapped_interval.py:109-127`). Result: `DispatchError: no
   operator registered for kind 'transform' on CellAvg(sigma, ...)`
   — cryptic, and far from the grid-construction site that created
   the mismatch (at solver build on the probe base `b77f8582`; at
   the first `solve` on the post-multigrid head). Re-probed on the
   merged `87aeabea` head: the new `preconditioner="multigrid"`
   option *builds* but its V-cycle also dies on the stretched
   column — the column machinery reaches `uniform_spacing`
   (`staggering.py:439-443`) and raises `NotImplementedError:
   ... has no uniform cell width; nonuniform spacing enters through
   the grid.measure fields`. Neither route works; the constructor's
   capability checks validate the coupled axes but never the base
   column's uniformity.
2. **The SPD license is lost.** The corner cross hops
   (`_cross_to_face`/`_cross_to_column`, `:548-610`) are exact
   transposes of each other only under the uniform computational
   measure — precisely the F5 correction-2 ruling ("chart-uniform,
   NOT measure-weighted"), whose recorded caveat was exactly this
   case. On the stretched column the probe measures
   `⟨Ap,q⟩ − ⟨p,Aq⟩` at rel `2.6e-5` (matrix asymmetry `8.6e-5`;
   uniform-column reference ≤ 1e-12); `A·1` stays machine zero
   (compatibility survives), and the diagonal
   `flux_diff`/`face_diff` legs keep their summation-by-parts
   pairing under any measure — the asymmetry is *entirely* the
   stretched base-axis down-hop, whose `0.5/0.5` stencil is uniform
   even on the stretched mesh.

Why only now: CG's inner product is the **physical measure-weighted
L²** (`ConjugateGradient._dot`, `krylov.py:412-435` —
`Σ (a·b).integrate()`), so "SPD" means self-adjoint under the
physical measure. On a uniform column physical = computational and
the plain hops are already adjoint; a stretched base column is
exactly where the two measures diverge.

**Fix, de-risked by probe.** Replace the base-axis corner down-hop
(`_to_cell`'s unweighted `"average"`, `mapped_pressure.py:394-435`)
with the measure-weighted adjoint of the up-hop:

```
down_b = diag(1/m_cell) · up_bᵀ · diag(m_inner)
```

with `m_cell` the physical stretched cell widths and `m_inner` the
physical center-to-center face measures (both from `grid.measure`).
Probed on the combined grid: rel asymmetry `2.6e-5 → 3.7e-16`
(matrix `8.6e-5 → 1.1e-16`), `A·1` machine zero, diagonal legs
byte-identical (the down-hop is used only in `_cross_to_face`). The
coupled-axis (x) hops are uniform-periodic and need no change. This
answers the F5 correction-2 caveat rather than reversing the ruling:
uniform columns keep the chart-uniform hops (physical = uniform
there), stretched columns weight them. The preconditioner needs a
separate, non-spectral column route — candidates in §6.

Both families are affected (the solver is family-switched at the
corner-hop kind only), so this is not an FV regression: the *nodal*
stretched+terrain model is equally unbuildable today.

## 4. What breaks in hydrostatic: terrain at all

The hydrostatic model (`src/fridom/hydrostatic`) is nodal-only (no
family machinery) and today has **zero chart support**; only
stretched-z works, automatically, because every vertical increment
rides `grid.measure` (`CumulativeIntegral._accumulate`,
`cumulative.py:342-386`: increment = `f · measure`, `sqrt_g` only
under `jacobian=`, which the core never passes). The chart metric is
missing at four sites, none with a nonhydro2 counterpart (nonhydro2
folds all of them into the one CG operator):

1. **`p_hyd` column integral** (`hydrostatic/modules/core.py:365-389`)
   — `-∫ b dz_p` needs the column Jacobian on the increment. The
   designed seam (`CumulativeIntegral(jacobian=...)`, hydrostatic
   plan §3) turns out to be **not wired for `maps=` terrain grids**,
   with a trap on each side (probed on the combined column):
   `jacobian=("zp",)` — the chart-coordinate name the convention
   suggests — is a **silent no-op** (the gate at
   `cumulative.py:373-376` keys on the *axis* name, and `zp` is not
   a factor: output bitwise equal to no-jacobian, error O(1),
   non-convergent), while `jacobian=("sigma",)` fires the gate and
   then **raises** (`unknown metric 'sqrt_g'` — an analytic `maps=`
   mapping derives `d<p>_d<q>` rows but no `sqrt_g`; only the
   embedding `chart=` form supplies one). The integral machinery
   itself is sound: hand-feeding the terrain-weighted increment
   (`cumint(J·b)`, J from `grid.metric`) converges at order `1.98`
   on the combined stretched column.
2. **`w` diagnosis** (`core.py:283-323`) — two gaps: the vertical
   increment misses `J` (same `jacobian=` omission), and the
   horizontal divergence `u.diff(x) + v.diff(y)` is taken at constant
   *computational* z, missing the slope term `-(Z_i/J)∂_b` — the
   divergence itself is not the physical one on a chart.
3. **Baroclinic pressure gradient** (`core.py:394-413`) —
   `p_hyd.diff(x)` at constant computational z is the classic
   sigma-coordinate pressure-gradient error; needs the same slope
   correction from the same `grid.metric` rows the mapped advection
   already uses (`column_corrections`).
4. **Free surface** (`free_surface.py`) — `_inv_depth` is frozen from
   the *mesh extent* and `Integral()[z]` carries no `sqrt_g`; on a
   chart the physical depth `H(x,y)` is variable and ≠ the
   computational extent, so both the depth-mean divisor and the
   integral weight are wrong, and the 2D barotropic Helmholtz solve
   becomes variable-coefficient (`c² = gH(x,y)`) — the hydrostatic
   analogue of `mapped_pressure.py`, which does not exist (the
   "variable-csqr solve route — specified, not built").

Today's failure mode on a chart grid, probed: `hy.Model` **builds
and runs with no raise** (with and without advection; steps stay
finite) while `p_hyd` omits the terrain Jacobian — `max` relative
error 27% at a mild `H ∈ [0.7, 1.3]`. Silent wrong physics, the
failure class the taught-error convention exists to prevent. One
invariant to guard in any fix: the exact surface-energy cancellation
(`w(0)` ↔ half-cell `p_hyd`, `core.py:24-32`) is proven under the
plain measure and must be re-derived under the J-weighted vertical.

Note the asymmetry with nonhydro2: for hydrostatic, "stretched +
terrain combined" is not a special case on top of working terrain —
terrain support itself is the missing layer, and building it
J-correct from the start makes the combined grid come out for free
(the stretched widths are already right).

## 5. Constraints from standing rulings

- **J-weighting stays module-side** — `FluxDifference` has no params
  seam by design; F5 and the ALE record (option C) both weight in the
  consuming module with params-threaded metrics. A combined-grid form
  is module/model work, not a spatial-operator change.
- **Same-row discrete Jacobian** (mapped-Jacobian spike): divisors of
  wide rows must be the same-row static Jacobian; *order 2 is exactly
  the boundary where "divide by the two-point measure" is a valid
  same-row Jacobian* — which is why the current 2nd-order combined
  advection is exact. Any order > 2 lift on a stretched column
  inherits the spike's route (high-order mapped plan §3), unchanged
  by this record.
- **FV-D4**: no nodal-retag shortcuts — a combined-grid closure must
  produce its result on the average family.
- **FV-D2**: momentum is not FV-conserved; velocities keep the
  consistent mapped divergence. The combined grid inherits this
  asymmetry — only the tracer budget telescopes.
- **F5 correction 2** ruled the corner hops chart-uniform *for
  uniform computational columns*, with the stretched column recorded
  as the open caveat — §3's measure-adjoint hop is the answer to that
  caveat, not a reversal of the ruling.
- **ALE (option C, when it lands)** is the same flux-form pattern;
  its mesh-velocity flux would J-weight and divide identically, so
  the convention below covers moving terrain over a stretched column
  with no extra mechanism.
- **Biased/WENO advection self-rejects stretched axes** (the
  `mapped_factor` refusal); on the combined grid only the 2nd-order
  centered family runs — unchanged by this record, lifted only by the
  high-order mapped plan.

## 6. The unified method

One convention, already live in the working parts:

> **Stretching is read from `grid.measure`; terrain from
> `grid.metric`; J-weighting is applied module-side; every
> weight/divisor is a matched pair.** Concretely: (i) any vertical
> increment/divisor is the codomain's measure field — never a scalar
> `dx`, never re-derived; (ii) any terrain factor is a
> `grid.metric` row (`J`, `1/J`, `Z_i`, `sqrt_g`) evaluated on the
> requesting staggered space — never folded into a mesh or a
> measure; (iii) a module that multiplies `J` onto a flux divides
> the matching `J` out of the tendency (telescoping pairs); (iv) a
> transpose-sensitive operator (CG/SPD) pairs its hops as adjoints
> under the measure-weighted inner product.

Per-consumer work items this implies:

**nonhydro2** (makes the already-correct advection reachable):
- N1 (immediate, cheap): taught error — `MappedPressureSolver`
  rejects a stretched base column by name at construction, pointing
  here; kills the cryptic `DispatchError`. Also correct the
  `mapped_pressure.py:73-75` docstring, which asserts the
  uniform-column assumption as if unconditional.
- N2: the measure-adjoint base-axis down-hop (§3 recipe), gated on a
  stretched base column, restoring exact SPD under the physical
  inner product; hop-local (`_resolve_corner_rows`/`_to_cell`),
  diagonal legs untouched.
- N3: a stretched-capable column preconditioner. The multigrid
  V-cycle (pathway phase B, `87aeabea`) is the natural candidate but
  is not one yet: re-probed on the merged head, its column machinery
  reads uniform spacing and raises on a stretched factor — it needs
  to consume `grid.measure` widths in the vertical-line smoother /
  coarsening instead (the same convention shift as N2). Fallbacks: a
  measure-aware vertical-line (Thomas) preconditioner alone, or
  plain-CG as a correctness-test stopgap.
- N4: extend the C3-style validation battery to a combined-grid case
  (projection idempotence, divergence-free steady state, energy).

**hydrostatic** (buys terrain and the combined grid in one move):
- H1: J-weight the two column integrals and the free-surface
  `Integral`. Two routes, since the `jacobian=` seam is unwired for
  `maps=` grids (§4): (a) **module-side** — the core multiplies the
  integrand by `J = grid.metric(..., "d<mapped>_d<base>")` before
  the plain cumint (probed: order 1.98 on the combined column; the
  F5 pattern, no spatial-layer change); (b) wire the seam — derive
  `sqrt_g` for analytic `maps=` mappings and re-key the
  `jacobian=` gate so chart-coordinate names resolve (spatial-layer
  work; also fixes the silent-no-op trap for every future consumer).
  Route (a) ships the model; (b) is the operator-hygiene follow-up —
  at minimum the silent no-op should become a taught error.
- H2: slope-corrected horizontal derivatives for the `w`-diagnosis
  divergence and the baroclinic pressure gradient — consume the same
  `column_corrections` metrics the mapped advection uses; this is
  the pressure-gradient-error site, so ship with the standard
  seamount/rest-state test (zero flow over topography at rest).
- H3: variable-depth free surface: physical `H(x,y)` =
  `Integral(jacobian=...)` of 1 over the column; the implicit variant
  needs the variable-csqr CG (specified in the hydrostatic plan §7).
- H4: re-derive the surface-energy cancellation under the J-weighted
  vertical; gate with the existing linear-energy test on a chart
  grid.
- H0 (immediate, cheap): until H1–H3 land, a taught error at model
  assembly on chart grids — today it runs silently with a 27%-wrong
  `p_hyd` (§4).

**shared**: the advection module needs nothing — it already serves
both models on the combined grid. ALE composes once its option C
lands.

Ordering note: H1–H4 are worthwhile *without* stretching (plain sigma
terrain hydrostatic); N1–N3 are worthwhile *without* terrain (the
stretched-column preconditioner also serves plain stretched grids if
their solve ever leaves the spectral path). The combined grid is the
intersection gate, not a separate mechanism — which is what makes the
method unified.

## 7. Owner calls needed before implementing

1. **Scope/priority.** Is combined stretch+terrain wanted for
   nonhydro2 now (N2–N4), or is the immediate ask only the taught
   errors (N1/H0) with the lift deferred? The advection layer is
   already correct either way; nothing regresses by gating.
2. **Hydrostatic terrain.** H1–H4 is a feature (hydrostatic
   sigma-coordinates), not a bugfix; it subsumes the combined case.
   Build now, or record as the designed route and gate (H0)?
3. **Preconditioner route** for the stretched column (N3): wait for
   the multigrid pathway (its natural consumer) vs. a dedicated
   vertical-line preconditioner now. Interacts with the multigrid
   plan's phase ordering.
4. **Gate placement.** The taught errors key on "mapped column whose
   base mesh is stretched" — confirm the predicate lives in the
   consuming solvers (F5 precedent: capability gates at the module),
   not in grid construction (the grid itself is valid and the
   advection layer uses it correctly).

### §7 addendum — rulings (owner, 2026-07-17)

All four calls ruled the day the record froze:

1. **nonhydro2 scope:** N1 + N2 now, with a plain-CG stopgap so
   correctness runs and the N4 battery are possible; the
   preconditioner lift proceeds on the ratified route (ruling 3),
   not deferred behind measurements.
2. **Hydrostatic:** gate **and** build the core — H0, H1, H2, H4,
   plus the depth-integral fix for the explicit/split-explicit free
   surfaces; the variable-csqr *implicit* free surface (H3) stays
   deferred behind a taught error.
3. **Preconditioner route:** the multigrid V-cycle learns
   `grid.measure` widths directly (column smoother + coarsening);
   no interim Thomas-only preconditioner.
4. **The `jacobian=` seam is wired properly:** `sqrt_g` derived for
   analytic `maps=` mappings, the gate re-keyed so chart-coordinate
   names resolve, and unresolvable names become a taught error.
   Consequence: H1 consumes the wired seam (§6 route b),
   **superseding** the module-side route-(a) recommendation.

Gate placement stays as recommended: capability gates live in the
consuming solvers/modules (F5 precedent), never in grid
construction.

## Appendix: probe battery

Probes live outside the tree (scratch); recipes and numbers:

- **Factor trace / no-double-count**: 2D x-periodic × stretched-σ
  (tanh clustering) with `maps={"zp": lambda sigma, H: ...}`,
  `H = H(x)` parameter field. `max|J − H| = 0.0`, `ptp_σ(J) = 0.0`;
  vertical measure = stretched widths.
- **Conservation**: random `Inner` fluxes, `integrate(J · (1/J)D_σF)
  = -8.9e-16`.
- **Constancy/divergence alignment**: random staggered velocity
  pair; `max|J·adv(q≡1) − press_div| = 7.1e-15` (rel `2.2e-16`)
  against `MappedPressureSolver.divergence` rebuilt from its own
  operator calls (the spectral preconditioner is bypassed — it does
  not build on the stretched column).
- **Symmetry**: current operator `⟨Ap,q⟩−⟨p,Aq⟩` rel `2.576e-5`
  (matrix asymmetry `8.57e-5`); measure-adjoint-down-hop variant
  `3.69e-16` (matrix `1.12e-16`); `max|A·1|` ≤ `2.3e-13` for both;
  diagonal legs byte-identical across the change. CG inner product:
  physical measure-weighted L² (`krylov.py:412-435`).
- **Hydrostatic seams**: `cumint(jacobian=("zp",))` bitwise equal to
  no-jacobian (silent no-op; error `0.169`, order `−0.05`);
  `jacobian=("sigma",)` raises `unknown metric 'sqrt_g'`;
  `cumint(J·b)` error `9.9e-4 → 2.5e-4`, order `1.98`. `hy.Model`
  on a terrain grid: builds, steps finite, no raise; `p_hyd` rel
  error 27% at `H ∈ [0.7, 1.3]`.
- **Accuracy** (manufactured fluxes vs the operator's continuum
  limit, interior cells): mild stretch (1.34:1) order `1.90`;
  strong tanh stretch (15.4:1) order `2.00`, with the
  vertical-aligned and cross-term parts at `1.99` each — **the
  cross term does not degrade** under strong stretching (the plain
  `0.5/0.5` hops sit inside a divergence over a smooth map;
  boundary cells show a wall-closure effect only, an impermeability
  artifact of manufactured fluxes, not a stretching-order effect).
