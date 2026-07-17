---
status: active
date: 2026-07-17
---

# Immersed partial cells — all dimensions, all three models

**Goal (owner, 2026-07-17):** make the immersed grid *work* in all
three models (`nonhydro2`, `shallowwater2`, `hydrostatic`), with
**partial cells in every dimension** — a sloping boundary declared in
x, `B(y, z)`, must produce genuine partial cells in the x direction,
not just partial bottom cells. As generic as possible: the weighting
machinery lives in `spatial` / `fr.model`, the model packages change
minimally.

This is fidelity-ladder point 2 of the masked-domain spec
(`design/specs/grid/02_rules.md` §3.7): cut-cell **volume fractions on
cell spaces, area fractions on face spaces, entering integrals and
flux operators as weights** — the boolean mask staying the `{0, 1}`
special case. It subsumes the immersed half of FV stage F5
(`fv_nonhydro_scoping.md` §6) and the "variable-`csqr` solve route —
specified, not built" deferral of the hydrostatic plan (§7).

## 1. Current state (recon 2026-07-17, four sweeps)

- **Grid layer: complete for booleans, scaffolded for fractions.**
  `spatial/immersed_domain.py` derives per-space masks/fractions on
  demand (slip-rule staggering, dry exterior, BC-drop parity via
  `_drop_constrained`), but `_cell_mask` is collocation-at-centers
  thresholded at 0.5, so `fraction()` returns `{0, 1}` only.
- **Model layer: nothing consumes masks.** The only model touchpoint
  is the capability gate `_require_fv_capable`
  (`nonhydro2/modules/core.py:97-138`) forcing immersed grids onto the
  nodal family — where they then run **unmasked**. shallowwater2 and
  hydrostatic have zero immersed references. Two hand-rolled
  prototypes exist in `tests/validation/`: a Brinkman penalization
  loop (`test_paper_validation.py:136-163`) and a masked flux-form
  Poisson/CG projection using fractions as face weights
  (`test_terrain_following_pressure.py:291-333`) — the latter is the
  pattern this plan promotes into the real solver.
- **Solvers.** The spectral solves (nonhydro flat path, hydrostatic
  `ImplicitFreeSurface._solve`) are separable constant-coefficient
  eigenbasis inversions — unadaptable to spatially-variable
  fractions. The `MappedPressureSolver` + `ConjugateGradient`
  (`spatial/operators/krylov.py`) pair is the architecture to reuse:
  matrix-free SPD operator under the measure-weighted product,
  spectral inverse as preconditioner, correction derived from the
  operator's own fluxes. Every solver is constructed inline in one
  CONSTRAINT stage per model, so the branch points are local.
- **Precedent (external).** MITgcm's hFac machinery is exactly this
  design, and its `hFacW`/`hFacS` *are* lateral partial cells
  (Adcroft, Hill & Marshall 1997 shaved cells; MITgcm
  cg2d/cg3d = masked PCG). Established practice adopted here:
  min-rule face fractions, hFacMin small-cell floor, div/grad
  adjointness for SPD, wet-mean nullspace projection,
  FFT-preconditioned CG that degrades gracefully to the exact
  spectral solve.

## 2. The discrete formulation

Notation, per cell `c` and face `f` (any axis): full cell volume
`V_c` (product of primal measures), full face area `A_f`
(product of transverse measures), center-to-center distance `d_f`
(dual measure), volume fraction `θ_c ∈ [0, 1]`, face-area fraction
`α_f ∈ [0, 1]`.

- **Masked divergence** (what continuity means on a masked grid):
  `div_θ(u)|_c = (1/(θ_c V_c)) Σ_f ± α_f A_f u_f`. Tendencies of
  cell quantities divide by the *wet* volume `θ_c V_c`; every flux
  is weighted by the *open* area `α_f A_f`. Flux telescoping is then
  exact over the wet region: total tracer content `Σ θ V q` is
  conserved to machine zero, and a face with `α = 0` is a free
  no-normal-flow wall.
- **Pressure gradient**: the plain two-point difference
  `G_f(p) = (p_R − p_L)/d_f`, **unweighted** — the open area enters
  only where the divergence gathers the flux back. This pairing is
  what keeps the assembled operator symmetric.
- **The cut-cell Poisson operator**, row-scaled by the *full* volume
  (not the wet volume):
  `L p|_c = (1/V_c) Σ_f ± α_f A_f G_f(p)`.
  Then `⟨q, L p⟩_V = Σ_c V_c q_c (L p)_c = − Σ_f α_f A_f d_f G_f(q) G_f(p)`
  — symmetric negative-semidefinite **in the existing
  measure-weighted product** that `ConjugateGradient` already
  implements. (The `1/(θV)`-scaled row is the same equation after
  diagonal row scaling; choosing `1/V` keeps CG's inner product
  untouched.) The RHS is the same-scaled masked divergence of the
  provisional velocity, `r|_c = (1/V_c) Σ_f ± α_f A_f u*_f`, so the
  projection removes exactly the divergence the operator measures.
- **Nullspace / dry cells.** Dry rows of `L` are identically zero and
  the RHS is wet-supported, so CG residuals stay wet-supported for
  the whole iteration; the preconditioner may paint values onto dry
  cells but they are multiplied by `α = 0` everywhere they could act
  (a face with `α > 0` has two wet neighbors under the min rule).
  The physical nullspace is the wet-region constant `e` (the boolean
  wet indicator): project its **V-orthogonal complement**,
  `p − (∫_wet V p / ∫_wet V) · e` — the orthogonal projector in the
  V-weighted product in which this row scaling of `L` is symmetric.
  (**Correction, I2**: the θ-weighted global mean this plan first
  specified is the projector for the *differently scaled* operator
  and re-introduces incompatibility on genuine partial cells —
  measured residual plateau ~4e-2. The e-form holds at machine
  zero.) Compatibility holds by telescoping: masked wall faces carry
  zero flux, so `Σ_c V_c r_c = 0` exactly.
- **Velocity correction**: `u_f ← u*_f − m_f G_f(p)` (vertical leg
  `1/dsqr`-weighted as in the flat solver), with `m_f` the boolean
  face mask `α_f > 0` — the correction never injects velocity into a
  closed face.
- **Preconditioner**: the existing flat `SpectralSolve` of the
  unmasked Laplacian (`nonhydro2/modules/pressure.py` machinery,
  Neumann/Dirichlet sibling seams on walled grids), **masked onto
  the wet cells**: `z = e ⊙ M⁻¹(r)`. (**Correction, I2**: the raw
  unmasked inverse destabilizes CG — its global `k = 0` gauge mixes
  with the wet-constant nullspace, measured residual → 1e17. With
  wet-supported residuals the masked form equals the symmetric
  `D M⁻¹ D`, SPD on the wet subspace.) When no immersed domain is
  attached the model never routes here (structurally unchanged
  spectral path); when the declared domain is all-wet the operator
  equals the preconditioner's and CG converges in ~1 iteration (the
  mapped-flat identity precedent).

## 3. Decisions

### IP-D1 — fraction semantics: quadrature opt-in, collocation default

`ImmersedDomain(init, *, slip=Slip.NO_SLIP, order=None,
min_fraction=0.1)`:

- `order=None` (default): iteration-1 semantics **bitwise** — the
  indicator is collocation-sampled at cell centers and thresholded at
  0.5, `θ ∈ {0, 1}` (staircase). Every existing test and consumer is
  untouched.
- `order=q` (the partial-cell opt-in): `θ_c` is the per-cell
  `q`-point Gauss–Legendre tensor quadrature of the declared
  indicator — the G8 `_quadrature_discretize` machinery
  (`grid.py:1226-`), which already honors `coordinate_map` stretch
  per cell. On face spaces the same machinery point-evaluates the
  nodal factor and cell-averages the transverse factors, which *is*
  the face-area fraction — but the **default face rule is the
  min-transfer** (below), and direct face quadrature is recorded as a
  designed-for refinement (it can violate `α ≤ min(θ_L, θ_R)`
  consistency on shaved geometries).
- The declared callable may return an indicator **or** a genuine
  local fraction; quadrature averages either. **Explicit `fraction=`
  data stops being thresholded** — module-owned genuine fractions
  pass through (boolean explicit data unchanged). Users with exactly
  known geometry (e.g. columnar `B(y, z)`) pass exact per-cell
  fractions this way.

### IP-D2 — staggering transfer: min-rule fractions, `θ > 0` masks

- `fraction(space)`: cell axes keep `θ`; face axes combine the two
  adjacent cells with **min** (dry exterior on bounded meshes) —
  the geometric, slip-independent transfer whose boolean restriction
  is exactly today's AND (`WaterMask` parity, MITgcm
  `hFacW = min(hFacC)` precedent). Multi-axis staggering composes
  per axis, as today.
- `mask(space, slip=...)`: staggered boolean of `θ > 0` — AND (min)
  under `NO_SLIP`, OR (max) under `FREE_SLIP`. On the `order=None`
  default this is bitwise iteration-1.
- BC-constrained DOF drops (`_drop_constrained`) unchanged.

### IP-D3 — small cells: hFacMin floor at materialization

`min_fraction` (default **0.1**, the MITgcm-typical hFacMin; `0.0`
disables): after quadrature, `θ < min_fraction/2 → 0`, else
`θ < min_fraction → min_fraction`. Applied once at cell-fraction
materialization; face fractions inherit through the min rule. Boolean
`{0, 1}` fractions are unaffected, so the default path never sees it.
Rationale: tendencies divide by `θ V`; unfloored slivers make the
explicit step unstable and the elliptic operator ill-conditioned —
the literature fixes this at geometry time, not at runtime.

### IP-D4 — weighting enters at the term level, not the registry

Probed: advection's flux-divergence legs resolve the *same*
`("diff", factor)` rows as the pressure chain
(`model/modules/advection.py:2252-2263` — `flux.diff(axis)`), and on
a C-grid one key serves two semantics (`("diff", CellAvg)` is both
the pressure gradient and the x-leg of momentum flux divergence). A
blanket immersed dispatch override therefore **cannot** express
"weight the flux divergence, not the gradient" and is rejected.

Instead, per the spec's ownership rule ("masking correctness is owned
by operators and modules"), fractions enter as **explicit field
arithmetic at each flux-form term**, concentrated in shared
machinery:

- a new `fr.model.modules.immersed` helper module owns the
  per-space fraction/mask lookups (cached grid-side like
  `grid._measures`, concrete-only — the memoization §3.7 allows) and
  the weighting idioms;
- `_FluxFormAdvection` (shared by all three models) weights every
  face flux by `fraction(flux_space)` and divides the divergence by
  `fraction(codomain)` (guarded: dry → 0) when `grid.immersed` is
  present — momentum control volumes use the staggered fractions
  `fraction(u_space)` etc., which the min-transfer already serves;
- the diffusion closures weight their fluxes identically (or gate,
  see IP-D8);
- each model core weights its own bespoke reductions (divergence,
  depth means, cumulative integrals) — thin, local edits.

Unimmersed grids take none of these branches (structural no-op —
the parity guard).

### IP-D5 — dry-DOF hygiene: one shared masking stage

`fr.model.modules.immersed.MaskState` — a CONSTRAINT-stage module,
auto-added by each model factory when `grid.immersed is not None`
(one line per factory): multiplies every prognostic field by its
boolean per-space mask (velocity-role fields under the domain's slip
rule, cell fields under the cell mask). This keeps dead DOFs dead
against the modules that legitimately do not consult masks (Coriolis,
wave makers, pressure-gradient tendencies) without touching them —
the old stack's sync-mask mechanism, done once, model-agnostically.

### IP-D6 — the immersed pressure solve (nonhydro2)

New `nonhydro2/modules/immersed_pressure.py`, the
`MappedPressureSolver` sibling (§2 formulation):

- operator/RHS/correction as in §2, built from the fraction fields
  and the existing `FaceDifference`/`FluxDifference` rows by explicit
  composition (no registry changes), `1/dsqr` on the vertical leg;
- `ConjugateGradient` with the flat spectral inverse as
  preconditioner, `iterations=pressure_iterations` (the existing
  knob, default 30), and the **wet-mean projection**: `krylov.py`
  gains an optional `projection: Callable[[FieldLike], FieldLike]`
  parameter generalizing `project_mean` (backwards-compatible;
  `project_mean=True` ≡ the global-mean projection);
- `DynamicalCore._project` branches to it when
  `grid.immersed is not None` (exactly like the mapped branch);
  walls compose through the Neumann-sibling seam the preconditioner
  already implements; **mapped + immersed stays a taught error**.

### IP-D7 — family policy: immersed model runs are FV

`_fv_capable` becomes `grid.mapping is None` (immersed grids are
FV-capable); auto-family = FV iff unmapped. An explicit
`family="nodal"` on an immersed grid becomes a **taught error**
("immersed physics is finite-volume: fractions are volume/area
weights; today's nodal path ignores the mask") — strictly better
than the current silent unmasked nodal run. Grid-layer
masks/fractions stay family-agnostic as today.

### IP-D8 — taught gates for what iteration 2 does not do

- biased/upwind/WENO advection on immersed grids: taught error
  (stencils reach across dry cells; the graded-fallback closure
  keyed on masks is designed-for). `CenteredAdvection` is the
  supported family — it never reads a dry value with nonzero weight
  (min-rule faces have two wet neighbors).
- analytic eigenmode kits / `from_model` transforms on immersed
  grids: taught error (the eigenbasis of the masked operator is not
  the tensor basis).
- `MeridionalDiffusion`/Smagorinsky-type closures: weight if the
  flux-form edit is mechanical during I2, else taught error — never
  a silent unmasked run.
- mapped + immersed: taught error (unchanged).

### IP-D9 — hydrostatic integration

- **`w` diagnosis**: bottom-up cumulative integral of the masked
  horizontal transport divergence, solved as
  `α^z_{k+1/2} w_{k+1/2} = α^z_{k-1/2} w_{k-1/2} − (1/A_c) Σ_h ± α_h A_h u_h |_k`,
  i.e. fraction-weighted increments and a guarded division by `α^z`
  (`α^z = 0 → w = 0`). Gate: masked continuity to machine zero.
- **`p_hyd`**: unweighted top-down cumsum, masked output (values
  under topography are dead). The partial-bottom-cell
  pressure-gradient refinement (Pacanowski & Gnanadesikan's concern)
  is recorded as designed-for, not built.
- **Implicit free surface**: with fractions the operator becomes
  `ε η − dt'² ∇_h·(csqr H̃ ∇_h η)` with per-column transport depths
  `H̃_u = Σ_k α^x dz / H` (and `H̃_v`) from summed face fractions —
  variable-coefficient, so the solve flips to the **CS-D2 route the
  plan already specifies**: SPD operator + `ConjugateGradient`,
  flat spectral inverse (mean depth) as preconditioner. `ε = 0`
  (rigid lid) uses the wet-column-mean projection. The
  `SpectralSolve` fast path stays for unimmersed grids.
- **Split-explicit**: barotropic fluxes gated by the summed face
  fractions, per-column wet depths in the depth-mean reductions and
  the SM2005 commit; the scalar `1/H`-from-extent bind freeze
  becomes a per-column field when immersed (guarded on land columns).
- **Explicit free surface**: same weighting, no solve.

### IP-D10 — shallowwater2 integration

2D fractions: `θ` = plan-area fraction, `α` = face-width fraction.
The linear core weights continuity `∂_t p = −(1/θ) ∇·(α csqr u)` and
masks the gravity gradient; `SadournyAdvection` runs under **boolean
masks** (masked corner vorticity and mass fluxes — its
energy/enstrophy telescoping is pinned interior-exact, boundary
conservation documented as approximate); genuine-fraction Sadourny
weighting is designed-for.

## 4. Stages and gates

| Stage | Work | Branch | Gate |
|---|---|---|---|
| **I0 — genuine fractions** (`spatial`) | IP-D1/D2/D3: `order=` quadrature, min-transfer, `θ > 0` masks, floor, explicit-fraction passthrough, grid-side caching | `feat/immersed-fractions` | `order=None` bitwise iteration-1 parity across the whole existing suite; quadrature fractions vs analytic geometries (slope, disk) converge in `q`; floor semantics pinned; forced-4 device invariance |
| **I1 — shared plumbing** (`spatial` + `model`) | IP-D5/D6a: CG `projection=` param + wet-mean helper; `fr.model.modules.immersed` (fraction helpers + `MaskState`) | same branch as I0 | existing krylov tests untouched; `project_mean` ≡ `projection=global-mean` bitwise; `MaskState` unit-tested per role/slip |
| **I2 — nonhydro2** | IP-D4/D6/D7/D8: `ImmersedPressureSolver`, `_project` branch, gate flip, advection weighting, taught gates | `feat/immersed-nonhydro` | **staircase equivalence**: face-aligned immersed box ≡ walled FV model over 12 jitted steps (≤ 1e-11); post-projection masked divergence ≈ machine zero; all-wet immersed ≡ unimmersed (tight tol); manufactured masked Poisson with sloping `B(y, z)` (true x-partials) converges at 2nd order; `θ`-weighted tracer mass conserved to machine zero; taught errors pinned; CG residual reported and sane at default iterations |
| **I3 — hydrostatic** | IP-D9 | `feat/immersed-hydrostatic` | masked continuity (w) machine zero; **column equivalence**: flat immersed bottom ≡ shallower unimmersed domain (geostrophic + Poincaré parity); implicit-FS immersed solve converges vs explicit oracle; split-explicit volume conservation machine zero on masked columns |
| **I4 — shallowwater2** | IP-D10 | `feat/immersed-shallowwater` | mass conservation machine zero; geostrophic steady state preserved away from the mask; masked Sadourny channel stable, interior conservation pinned |
| **I5 — close-out** | records, roadmap move, cleanup | direct-to-dev | zero leftover branches/worktrees; roadmap hygiene rule honored |

Sequencing: I0+I1 (one branch) → I2 → I3 ∥ I4 → I5. I2 first because
it forges the weighting and solver patterns I3/I4 copy. Every stage:
mirrored tests (95% branch coverage), ruff clean, model smoke file
where core machinery is touched (per AGENTS.md).

## 5. Risks / open items

- **Preconditioner quality on heavily-masked domains** — the unmasked
  spectral inverse degrades as the wet region shrinks; the fixed
  iteration budget (30) is the first thing to fail. Mitigation: the
  residual is reported per solve; the gate pins it; the knob is
  user-facing. Multigrid is the recorded fallback lever (roadmap
  "mapped-solve residual levers").
- **Momentum advection near partial cells** — the dual-cell fraction
  algebra is standard but fiddly; the staircase-equivalence gate
  (bit-comparable to walls) is the safety net.
- **Split-explicit consistency** — barotropic vs baroclinic
  transport-depth mismatch leaks mass at coasts; the volume
  conservation gate targets exactly this.
- **Distributed correctness** — fraction fields ride the ordinary
  `store` + `sync` path (already device-count invariant in tests);
  the forced-4 suite must cover the new solver's fast paths; real
  multi-GPU validation joins the next GPU campaign (same status as
  the FV default flip).
- **Perf** — unimmersed paths are structurally untouched (no
  override, no branch taken); the step-parity guards stand. Immersed
  runs pay one CG (≈ mapped-solve cost profile); no committed
  baselines change.

## 6. Implementation record

**I0+I1 shipped 2026-07-17** (merge `ee257bc0`; branch
`feat/immersed-fractions`). Genuine fractions
(`order=`/`min_fraction=` per IP-D1/D3, min-transfer per IP-D2,
`mask = staggered θ > 0`, explicit fractions unthresholded,
concrete-only memoization), CG `projection=` hook, `MaskState` in
`fr.model.modules.immersed` (factory wiring deferred to I2–I4).
Gates: bitwise `order=None` parity across the whole pre-existing
suite; 100% branch coverage on all three touched files; forced-4
device invariance; ruff clean. Corrections found:

1. **`order=1` aliases `None`** (the collocation staircase), matching
   `grid.py`'s midpoint shortcut — documented and pinned, not an
   error.
2. **Hard indicators quadrature-converge slowly** — Gauss–Legendre
   quadrature of a discontinuous `x < B` indicator is O(1/q)-ish and
   oscillatory (measured: cell-true 0.75 → 0.50 at q=2, 0.755 at
   q=64, 0.746 at q=128). Consequence for gates and users: accuracy
   claims (the I2 manufactured-Poisson 2nd-order gate) must declare
   geometry via **smooth/analytic local fractions or the explicit
   `fraction=` path**; users with exactly known columnar geometry
   (`B(y, z)`) should pass exact per-cell fractions explicitly.
3. Direct `FaceAvg` quadrature stays guarded in `grid.py`
   (designed-for); face-area fractions go through the min-transfer,
   as designed.

**I2 shipped 2026-07-17** (merge `b447b8e5`; branch
`feat/immersed-nonhydro`). `ImmersedPressureSolver`
(`nonhydro2/modules/immersed_pressure.py`), the `_project_immersed`
branch, the family-gate flip, IP-D4 fraction weighting in
`_FluxFormAdvection`, `MaskState` factory wiring, and the IP-D8
taught gates (biased advection, closures, eigenmodes, nodal+immersed,
mapped+immersed). Gates: operator symmetry exact (rel-diff 0.0);
post-projection masked divergence 9e-14 (CG residual 2e-16);
staircase ≡ walled FV at ~1e-16 over 12 jitted steps; all-wet ≡
unimmersed ~1e-15 with ~2 PCG iterations; manufactured masked Poisson
with genuine x/z-partials at L2 order 2.38; θ-weighted buoyancy
conservation exactly 0.0; dry-DOF hygiene exactly 0.0; forced-4
device invariance; patch coverage 95.7%. The merge onto `dev`
composed IP-D7 with the concurrent `fv-mapped-default` owner ruling
("FV wherever capable"): the merged predicate is **FV-capable iff
immersed or static-mapped-or-flat; only a dynamically driven mapping
(ALE, nodal-only) keeps the nodal auto default**, and mapped+immersed
routes to FV so the specific composition error fires. Corrections
found (both folded into §2 above): the preconditioner must be masked
(`e ⊙ M⁻¹`), and the nullspace projection is the V-orthogonal
boolean-wet mean, not the θ-weighted mean. Also recorded for I3/I4:
two scalings coexist (solver rows `1/V`, physical tendencies `1/θV`);
the velocity correction uses the boolean face mask (α·m = α exactly);
isolated `min_fraction`-floored slivers add nullspace modes — use
`min_fraction=0` (or guaranteed connectivity) for manufactured-RHS
convergence studies (the physical projection RHS is always in range);
preconditioner quality degrades on genuine partials (~60+ iterations,
plan risk 1 confirmed — multigrid stays the recorded fallback);
`MaskState` must be halo-trace exempt, sort after the projection, and
capture the immersed descriptor at bind.

**I4 shipped 2026-07-17** (merge `a5aec29d`; branch
`feat/immersed-shallowwater`). Fraction-weighted linear core
(`_gravity_immersed`: `∂_t p = −(1/θ)∇·(α c² u)`, masked pressure
gradient), masked Sadourny (`_advect_immersed`: fraction-weighted
thickness transport, boolean-masked corner PV and momentum
tendencies), `MaskState` wiring, eigenmode taught gates; all local to
`shallowwater2` (a package-local `immersed_weighting.py` mirrors the
I2 idioms; no shared-file changes). Gates: mass `Σ θ V p` 2.2e-16
over a nonlinear masked run; staircase channel vs the walled model
~2.8e-17 over 12 steps; all-wet ≡ unimmersed ~5.6e-17; genuine
lateral partials stable with mass diff 4.2e-17; dry-DOF hygiene
exactly 0; 100% coverage on changed files; forced-4 invariance.
Corrections: (1) the staircase channel reproduces the **free-slip**
walled Sadourny under the default `NO_SLIP` mask — the AND-combined
corner mask zeroes wall-corner `ζ`, exactly the walled `ζ = 0`
closure, while tangential `u` sits unmasked at cell centers; (2) the
Sadourny **thickness transport is genuinely fraction-weighted** (not
merely boolean-masked as IP-D10's wording suggested) — that is what
conserves mass to machine zero on genuine partials with advection
active; the boolean qualifier applies to the momentum machinery only;
(3) all-wet parity is near-bitwise (~6e-17), not bitwise: the
immersed path is halo-trace exempt (halo 2 vs auto-1), so XLA fuses
differently — within the plan's tolerance; (4) shallowwater2 has no
`family=` concept, so IP-D7's nodal gate has no sw2 analogue — the
grid-default nodal C-grid carries the masked paths directly.

**I3 shipped 2026-07-17** (merge `3858d977`; branch
`feat/immersed-hydrostatic`). Masked continuity `w` (fraction-weighted
transport divergence through `CumulativeIntegral`, guarded `α_z`
division), wet-column free surface (masked depth-mean, per-column
transport depths, variable-coefficient implicit barotropic PCG
`_solve_immersed` with wet-column-masked spectral preconditioner and
`pressure_iterations` knob, masked explicit/implicit corrections,
per-column split-explicit subcycle), `MaskState` wiring, eigenmode
taught gates. Gates: masked continuity ~4.4e-16 (w exactly 0 on closed
faces); column equivalence ≤ 8.9e-16 explicit / 2.0e-15 implicit;
all-wet implicit ≡ unimmersed 2.2e-16 at 1 CG iteration; rigid-lid
masked depth-mean divergence 1.95e-15; split-explicit θ-mass 1.9e-16
with all-wet byte-identical and land-column transport exactly 0;
genuine x-partials θ-mass drift ≤ 1.1e-14; 100% coverage on changed
files; forced-4 invariant. Corrections to IP-D9:

1. **The hydrostatic model is nodal-only** (no FV family machinery);
   the mask rides explicit fraction arithmetic — family-agnostic, and
   the 2nd-order stencils are the same numbers anyway. No
   nodal+immersed taught error exists there (nothing silently ignores
   the mask); IP-D7's family gate is a nonhydro2-only concept.
2. **`csqr` stays `g × mesh extent`**, the wet-column wave speed is
   recovered through the coefficient `H̃ = H_wet/H_ref` — column
   comparisons must match physical `g`, not `csqr`.
3. **`_depth_mean_div` keeps the scalar reference depth** `1/H_ref`
   (the volume-conserving transport form); per-column wet depths
   enter only the operator coefficient `csqr·H̃` and the split
   subcycle — putting them in the RHS is the mass-leak trap IP-D9
   warned about, from the inside.
4. **The Outer-face `α_z` needs a surface-face override** to the
   surface-cell fraction: the min-rule's dry exterior would zero the
   physical surface DOF `w(0)`.
5. **Genuine partial cells need `min_fraction > 0`** (default 0.1) on
   explicit paths — the `÷θ` tendency blows up otherwise; and
   quadrature `init` callables must use `jnp`, not `np` (traced).
6. Shared-file change: `MaskState` skips prognostics with a
   `ConstantSpace` factor (barotropic `ps`/`U`/`V` — wet-region
   hygiene owned by the free-surface module; `ImmersedDomain.mask`
   rejects unresolved spaces).

**I5 close-out 2026-07-17.** Autodiff compliance sweep (merge of
`test/immersed-autodiff`): every immersed guarded division audited
against the new AGENTS.md differentiability policy — all sites
already reverse-safe (double-`where`, or the safe-denominator Sadourny
PV style; note the two styles differ in forward value at guarded
cells — exact 0 vs finite-masked-downstream — relevant to anyone
tightening forward parity gates there); three autodiff regression
shards added (`test_immersed_model_autodiff.py` per model), `jax.grad`
through the masked PCG / Sadourny / wet-column paths FD-matched at
≤ 1e-8 on genuine partial cells. Roadmap entry moved to `done.md`;
residuals (heavily-masked preconditioning → multigrid pathway plan,
graded-mask biased advection, mapped+immersed, partial-bottom `p_hyd`
refinement, fraction Sadourny momentum, masked closures, 4-GPU
validation) tracked in `open.md`. Stages I0–I5: **all shipped**.

## 7. Out of scope (designed-for, not precluded)

Direct face-area quadrature (shaved-cell faces); ghost-cell
immersed-boundary fill (ladder point 3) and Brinkman penalization
module (point 4, the spectral-basis route); graded/fallback biased
advection near masks; partial-cell hydrostatic pressure-gradient
correction; fraction-weighted Sadourny; level-set representation;
mapped + immersed composition; moving immersed geometry
(`MovingGeometry` seam); old stack untouched.
