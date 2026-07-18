---
status: active
date: 2026-07-18
---

# Multigrid generalization — terrain barotropic solve + coarsening-axis freedom

Driver (owner, 2026-07-18): build the multigrid-preconditioned solve for
the hydrostatic **implicit free surface on terrain grids** (the deferred
H3 item, today a taught error), generalizing the shipped multigrid
machinery where needed — and use the same generalization to measure the
open question the shipped design left unanswered: **horizontal
semicoarsening vs full 3-D coarsening** on the nonhydrostatic mapped
solve.

Two parts, one plan: (a) the 2-D variable-coefficient barotropic
Helmholtz solve (closes H3); (b) coarsening-axis freedom in the
hierarchy builder plus the measured comparison (no default change
implied).

## 0. Context — what the research established (2026-07-18)

- The V-cycle engine is already generic: `MultigridVCycle` /
  `MultigridLevel` / the `Smoother` ABC
  ([multigrid.py](../../../src/fridom/spatial/operators/multigrid.py))
  interact purely through interfaces, and `DampedJacobi` (an isotropic
  point smoother) already exists and satisfies the contract. All
  "horizontal semicoarsening + full-vertical line smoother" hardwiring
  lives in the *solvers* (`_build_vcycle` always picks
  `VerticalLineJacobi`) and in the hierarchy builder's one-line
  `name == vertical` skip
  ([multigrid_hierarchy.py:99](../../../src/fridom/nonhydro2/modules/multigrid_hierarchy.py)).
- `GridTransfer` coarsens exactly the axes the coarse grid coarsened
  (any named axis, vertical included; `MappedIntervalMesh` accepted),
  with the measure-weighted adjoint restriction `R = M_H^-1 P^T M_h`.
  Its **only** blocker for the barotropic use is the input gate: it
  rejects fields with a `ConstantSpace` factor
  ([transfer.py:282-310](../../../src/fridom/spatial/operators/transfer.py)),
  which is precisely the z-factor of a `Profile("x","y")` field.
- `Grid.coarsened` clones-and-rebinds the chart and re-derives all
  metrics on the coarse spaces (MG-D6); it has no horizontal
  assumption. A mapped vertical coarsens; a `ChebyshevMesh` vertical
  cannot (`_make_refined` raises).
- MG-D5 replication + per-level shard floors already cover the
  multi-device story for tiny coarse levels; `ConstantSpace` factors
  are never sharded.
- The point-smoother refutation
  ([multigrid_kernel_study.md](../../research/multigrid_kernel_study.md))
  is **scoped to the 3-D semicoarsened hierarchy** (vertical error
  modes neither smoothed nor coarsened). For the isotropic 2-D case the
  records carry the positive datum: the GB-1 gate measured 2-D
  point-Jacobi V(1,1) at contraction ρ ≈ 0.36, grid-independent
  ([multigrid_pathway_plan.md](multigrid_pathway_plan.md) §B5).
- MG-D4 (semicoarsening) explicitly left full coarsening "expressible
  where it is wanted"; no record measures semicoarsen-vs-full — part
  (b) is genuinely new.
- The hydrostatic side already builds the terrain RHS and correction:
  `_depth_mean_div`'s flux-form transport divergence and the z-uniform
  `-dt' grad ps` correction
  ([free_surface.py](../../../src/fridom/hydrostatic/modules/free_surface.py)),
  and `_solve_immersed` (IP-D9) is the architectural template: per-axis
  field-arithmetic SPD operator, flat-spectral preconditioner, CG with
  `pressure_iterations`/`pressure_tolerance`. The taught error to
  remove is `ImplicitFreeSurface.bind` (free_surface.py:684-695).

## 1. Decisions (owner-ratified 2026-07-18)

*Owner ratification 2026-07-18: the plan and all recommendations are
ratified as proposed — GM-D1 resolves to option 1 (volume-exact);
GM-D9 was ruled the same day (full 3-D coarsening default). Phase E
(warm-started pressure solves) was added at the same ratification on
the owner's question.*

**GM-D1 — the terrain operator is the volume-exact ("variable-csqr")
form; the energy-form is the recorded alternative. RATIFIED: option 1
(owner, 2026-07-18).**
With `T*` the *un-normalized* post-advance transport divergence
`∫[∂x(Ju) + ∂y(Jv)] dz`, `H_a(x,y) = ∫J dz` on the a-face, `H_ref` the
constant vertical mesh extent, and `g = csqr / H_ref`:

    (ε I − dt'² g ∇h·(H_face ∇h)) ps = ε ps^n − dt' g T*        (option 1)

- **Option 1 (recommended, volume-exact):** SPD in the plain 2-D
  product for ε > 0 (PSD with constant nullspace at ε = 0);
  conserves plain `∫ps` (barotropic volume) to round-off by
  construction — the stated *purpose* of the deferred implicit solve
  ([stretched_terrain_combined.md](../../research/stretched_terrain_combined.md)
  §6 H3: "exact volume needs the variable-c²"); constant `1/g` energy
  weight; local barotropic wave speed `g·H(x,y)` — the physical one.
  No `1/H(x,y)` division anywhere in the solve path (the RHS uses raw
  `T*`, not `_depth_mean_div`'s normalized form), which also removes
  the guarded-division autodiff hazard from the implicit path.
- **Option 2 (energy-form):** mass term `ε·H(x,y)/H_ref` instead of
  `ε` — also plain-SPD; consistent with the shipped *explicit* terrain
  variant's gravity term (constant csqr over variable depth, energy
  conserved, `∫ps` drifts O(slope)).
- The two share the identical flux part, so the implementation carries
  both trivially; the ruling picks the default. Flag: option 1 makes
  the implicit variant's discrete barotropic physics differ from the
  explicit terrain variant's (physical `gH(x,y)` vs constant `csqr`
  wave speed). Whether the explicit variant should later adopt the
  volume-exact form too (it is *simpler* — no `1/H` division — but
  changes shipped physics) is out of scope here and is its own owner
  decision.

**GM-D2 — solver home and seam.** A new
`src/fridom/hydrostatic/modules/barotropic_pressure.py` houses the
terrain barotropic solver (mirroring the mapped/immersed pressure-solver
shape: operator via per-axis `registry.resolve("diff", ...)` field
arithmetic exactly as `_solve_immersed` does, `diagonal()`,
`_build_vcycle`, per-solve metric derivation — never cached across
solves). `ImplicitFreeSurface` gains `_solve_terrain` calling it; the
`bind` gate is removed. Knobs mirror the nonhydro naming:
`pressure_preconditioner ∈ {"spectral", "multigrid"}` (default
`"spectral"`), `multigrid_levels: int | None = None` (floor-limited,
the ratified default semantics), reusing the existing
`pressure_iterations` / `pressure_tolerance`. The velocity correction
keeps using the same discrete `ps.diff(a).to(vel)` gradient as the
operator's legs — the exact-cancellation identity is the gate, not an
aspiration. Flat and immersed paths stay byte-identical.

**GM-D3 — hierarchy representation: 3-D coarse grids + Profile fields;
`GridTransfer` learns `ConstantSpace` pass-through.** The barotropic
hierarchy reuses `coarsen_levels` unchanged in semantics (coarsen x,y,
keep z): coarse levels are full 3-D grids whose charts re-derive
`H_face`/`H_cell` per level from `grid.metric` — MG-D6 re-discretization
preserved, no Galerkin coefficient restriction. The one machinery
change: `GridTransfer` accepts fields with `ConstantSpace` factors
(shape-(1) axes: never transferred, skipped by the volume weighting).
Rejected alternative: genuine 2-D grids per level — transfers work
today, but a 2-D grid carries no chart, so coefficients would have to
be *restricted* from the fine level (Galerkin), breaking MG-D6 symmetry
with the rest of the multigrid stack and adding a model-side
projection seam.

**GM-D4 — smoother: damped point-Jacobi, symmetric V(1,1),
`coarse_sweeps=8`, floor depth.** Consistent with the GB-1 datum
(ρ ≈ 0.36 grid-independent, 2-D isotropic); the point-smoother
refutation binds only the 3-D semicoarsened hierarchy. Structural note
for the owner: the §7-addendum ruling 3 ("the multigrid V-cycle learns
`grid.measure` widths directly — column smoother + coarsening") was
written for the 3-D stretched-column pressure solve (N3); the
barotropic solve space has **no column at all**, so a point smoother is
not a contradiction of that ruling — N3 stays open and unclaimed.

**GM-D5 — hierarchy-builder promotion + coarsening-axis freedom.**
`coarsen_levels` moves from `fridom.nonhydro2.modules` to
`fridom.spatial` (it is grid/space-generic and now has two model
consumers); nonhydro2 imports it from there. Its signature gains
`vertical: str | None` (None = nothing excluded) and
`coarsen_vertical: bool = False`; when True the vertical coarsens under
the same `MIN_COARSE_CELLS` floor as the horizontal axes. MG-D4's
semicoarsening **default is unchanged**; full coarsening becomes
expressible (the door MG-D4 recorded). The immersed solver stays
horizontal-only (its `uniform_spacing` reads block stretched z — the N3
adjacency, untouched here). A Chebyshev vertical cannot coarsen and
keeps raising with a taught message.

**GM-D6 — walls.** The barotropic operator supports walled horizontal
axes from day one via the `mapped_pressure._resolve_flux_rows`
precedent (divergence legs keyed on Dirichlet-tagged faces = zero
barotropic transport through the wall; identity on periodic axes). The
immersed template's doubly-periodic shortcut is not inherited; a
channel test is a Phase B gate.

**GM-D7 — rigid lid (ε = 0) gauge.** Nullspace = constants; the
V-orthogonal projection under option 1 is the **plain mean** removal
(the engine's `_mean_free` pattern), applied per level and at the CG
seam; the RHS is orthogonal to constants by telescoping of `T*`. ε > 0
is non-singular: no projection anywhere.

**GM-D8 — halo.** The terrain implicit variant declares the 2-cell
metric halo per horizontal coordinate (the `core.py` terrain/immersed
precedent), replacing the flat path's 1-cell FD halo; flat grids keep
1 cell.

**GM-D9 — full 3-D coarsening becomes the mapped-solver default
(owner-ratified 2026-07-18).** Originally scoped measurement-only; the
owner ruled after spike 2 (§5: identical 10-iteration convergence at
every size, −6..−11% per CG iteration, line smoother kept). The flip
lands in Phase D behind its gates: the forced-4 / real multi-GPU leg
must come back neutral-or-better first, and the builder must degrade
gracefully where the vertical *cannot* coarsen — a Chebyshev vertical
(`_make_refined` raises), an indivisible n_z, or the immersed solver's
`uniform_spacing` limit — by keeping semicoarsening for that
configuration automatically. The auto-fallback is safe by the
prefer-explicit rule's own criterion: the choice is measured
convergence-neutral and never physics-affecting (preconditioner cost
only). Semicoarsening stays expressible via the knob.

## 2. Phases and gates

**Phase A — spatial machinery** (`spatial/operators/transfer.py`,
hierarchy-builder promotion + `coarsen_vertical`).
Gate GA-1: `GridTransfer` unit tests including a numeric Profile-field
adjointness check (`⟨R f, g⟩_H = ⟨f, P g⟩_h`, rel < 1e-12).
Gate GA-2: the shipped nonhydro2 multigrid tests pass unchanged (the
promoted builder produces the identical hierarchy; semicoarsening
default untouched).

**Phase B — terrain implicit surface, spectral-preconditioned CG**
(the functional H3 close; `barotropic_pressure.py` +
`ImplicitFreeSurface._solve_terrain`; preconditioner = flat spectral
inverse at the mean depth).
Gate GB-1: exact-cancellation — at ε = 0 the corrected depth-mean
transport divergence is ≤ 1e-13 · scale (machine precision).
Gate GB-2: flat-chart limit — the terrain path on an `a = 0` chart
matches the shipped `SpectralSolve` path to machine precision.
Gate GB-3: volume — plain `∫ps` conserved to round-off over a
multi-step run (option 1); energy-skew analog recorded for the chosen
form.
Gate GB-4: a walled-channel configuration runs and passes GB-1.
Gate GB-5: autodiff regression (house pattern: `jax.grad` through a
short run, finite and matching central FD to rtol 1e-4).
Gate GB-6: the taught error is replaced; its test flips to asserting
the solve engages.

**Phase C — 2-D multigrid preconditioner** (point-Jacobi V-cycle over
the Phase A machinery; `pressure_preconditioner="multigrid"`).
Gate GC-1: h-independent PCG iterations on steep terrain (flat across
64² → 512² at a = 0.8, matching the spike counts).
Gate GC-2: at a = 0.8 the multigrid-preconditioned count beats or
matches the flat-spectral one; SPD/symmetry checks green.
Gate GC-3: forced-4-device parity for the multigrid path (MG-D5
replication exercised on the 2-D hierarchy).

**Phase D — full 3-D coarsening as the mapped default (GM-D9,
owner-ratified).** The single-GPU measurement is done (spike 2, §5).
Remaining work: the production `coarsen_vertical` knob over the
Phase A builder (default: coarsen the vertical wherever the mesh
supports it, with the automatic semicoarsening fallback of GM-D9),
tests including the fallback configurations, and the multi-GPU leg.
Gate GD-1: forced-4 / real multi-GPU parity and cost neutral-or-better
(z is unsharded, so z-transfers are shard-local — expected neutral,
must be verified) before the default flips.
Gate GD-2: graceful-degradation tests (Chebyshev vertical, indivisible
n_z, immersed solver) keep semicoarsening without error.
Gate GD-3: a research record with the full table including the
multi-device leg; step-baseline impact is left to the owner's batched
guard checkpoints (perf-guard policy).

**Phase E — warm-started pressure solves (owner-requested
2026-07-18).** Thread the existing, currently-unused `x0` seam
(`ConjugateGradient.__call__(rhs, x0=None)`, already plumbed through
both pressure solvers' `solve`) to the three iterative call sites: the
mapped and immersed projections pass the previous stored pressure
rescaled to the stage increment (`x0 = state["p"] · ctx.stage_dt`; the
zero-initialized first step is unchanged), and the implicit free
surface passes the previous `ps`. The stopping test is RHS-relative
and the masked-scan early exit is measured real, so saved achieved
iterations are wall time (estimate: ~2–3 of 10 multigrid iterations,
~9–13 of 36 spectral-preconditioned, flow-dependent; never negative —
correctness is start-independent).
Gate GE-1: warm-started solves match the zero-start solution within
tolerance; the mean gauge stays enforced under a non-mean-free `x0`;
the autodiff regressions stay green.
Gate GE-2: a measured step-time / achieved-iteration reduction on the
GB-2 config, recorded in the implementation record.

## 3. Risks

- The `ConstantSpace` pass-through could reach deeper than the input
  gate (sibling-space construction, volume weighting). The spike
  measures exactly this; if it snowballs, the fallback is the rejected
  2-D-grid alternative of GM-D3 (workable, uglier).
- Tiny 2-D coarse levels under GSPMD: expected to ride MG-D5
  replication; if the replicated 2-D levels misbehave, cap the
  hierarchy depth on sharded grids (the `multigrid_levels` int knob
  already expresses this).
- Order-2 transfer weights are index-space linear — on a *stretched*
  coarsened vertical (Phase D) that is first-order in physical space;
  acceptable for a preconditioner, recorded as a fidelity note.
- Moving geometry: the mapped 3-D `_build_vcycle` refuses grid-bound
  parameters; terrain charts in the hydrostatic model are static, so
  the barotropic solver inherits the same refusal for safety.

## 4. Out of scope (designed-for, not precluded)

- N3 — the measure-aware stretched-column smoother for the 3-D
  nonhydro solve (§7-addendum ruling 3's consumer) and the immersed
  solver's `uniform_spacing` limitation.
- Immersed + terrain combined (the base-class mutual-exclusion gate
  stays).
- Adopting the volume-exact form in the *explicit*/split-explicit
  terrain variants (own owner decision; GM-D1 flag).
- The `EnergyMetric`/eigenmodes plain-extent `ps` weight inconsistency
  on terrain (recorded in the terrain record; untouched).
- Changing the 3-D semicoarsening default (GM-D9).

## 5. Evidence — de-risking spikes (2026-07-18)

Two spikes ran before this plan froze (throwaway worktree branches
`perf/spike-barotropic-mg`, `perf/spike-vertical-coarsening`, removed
after reporting; the source deltas they validated are exactly GM-D3's
transfer pass-through and GM-D5's builder flag):

- **Spike 1 (2-D barotropic): COMPLETE — both design questions
  answered yes.** The whole chain (hacked `GridTransfer` →
  `coarsen_levels(…, vertical="z")` to the 4×4 floor with z kept as
  ConstantSpace → per-level chart re-derivation of `H_face` →
  `DampedJacobi` V(1,1) → manual PCG mirroring `krylov._step`) runs on
  Profile fields end to end. The `ConstantSpace` pass-through is
  **shallower than feared**: two small edits (`_is_cell_factor`,
  `_sibling_factor` honoring `factor.is_constant`); `_axis_specs` /
  `_volume` / `_coarsened_names` need nothing (the constant factor
  gets ratio 1 automatically). All checks at machine precision: SPD
  symmetry ≤ 1.3e-15 (both GM-D1 options, ε ∈ {0,1}, a ∈ {0,0.4,0.8}),
  Profile transfer adjointness ≤ 1.3e-16, flat-limit equivalence to
  the spectral operator 4e-15 with 1-iteration PCG. Iteration counts
  (rel 1e-8, stiffness dt'²csqr/dx² = 2500, sizes 64²–512²):
  **multigrid is h- AND steepness-independent at 11–13 iterations in
  every cell of the table**; flat-spectral is size-flat but
  steepness-degrading (1 at a=0 → 13 at a=0.4 → 28 at a=0.8; the
  size-flatness is specific to smooth single-mode terrain). Crossover:
  spectral wins mild terrain outright, parity near a≈0.4, multigrid
  ~2× fewer iterations at a=0.8 — supporting GM-D2's
  spectral-default + multigrid-knob shape. **Option 1 vs option 2:
  counts identical to within 1 iteration everywhere** (at ε=0 the
  operators coincide), so GM-D1 is a pure physics ruling, not a solver
  trade. Unpreconditioned CG blows up ~linearly in n (554+ at 256²) —
  both preconditioners are load-bearing.
- **Spike 2 (vertical coarsening): COMPLETE — full 3-D coarsening is
  a modest, real win with the line smoother; the point smoother is
  refuted under full coarsening too.** GB-2 steep mapped protocol
  (a=0.8, FV, tol 1e-8, cuSPARSE, one clean A100). Line-smoothed full
  coarsening converges in the **identical 10 PCG iterations** at
  128³/256³/512³ (true measure-weighted residuals verified ≤ 6.5e-9);
  the measure-weighted z-transfers on the mapped column are adjoint to
  machine precision (≤ 2.5e-15) — they just work. Costs
  (semicoarsen → fullcoarsen): per V-cycle 1.99→1.97 / 14.2→12.3 /
  108.5→93.6 ms; per CG iteration 3.08→2.90 / 16.6→15.7 /
  134.2→119.1 ms; projected solve 1341.6→1190.6 ms at 512³ (**−6% at
  128³/256³, −11% at 512³** — the saving grows with n because
  semicoarsening's coarse levels keep full n_z). Realized floor depth
  identical (coarsest 4×4×4 vs 4×4×512); peak memory equal; 512³
  compile 464→396 s. `DampedJacobi` under full coarsening **hits the
  100-iteration budget at 6.5e-3 residual** at both 128³ and 256³:
  proportional coarsening preserves the vertical anisotropy (the
  `1/dsqr` weighting + mapped column), so line smoothing stays
  load-bearing and the 3-D point-smoother refutation extends to full
  coarsening. Single-GPU numbers; GSPMD untested (z is unsharded, so
  z-transfers are shard-local — expected neutral, to be verified in
  Phase D).

## 6. Roadmap tie-in

Closes the implicit half of the open.md item "Variable-depth implicit +
split-explicit free surfaces (H3)" (the split-explicit chart variant is
already built; the roadmap line bundles history). Extends the
"Multigrid preconditioner — follow-up measurements" line with the
Phase D comparison. References, not claims: IP-D9 (immersed barotropic
sibling), CS-D2 (the CG-route ancestor), N3.
