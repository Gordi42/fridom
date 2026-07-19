---
status: active
date: 2026-07-16
---

# Roadmap — open work

The single open-work tracker for FRIDOM, ordered: **next steps** first,
then **long-term goals**. Task numbers are historical and stable — other
records cite them ("ROADMAP 3.5") — so they do not run in order.

**Hygiene rule (binding): this file holds only open work.** The moment
something ships, move its record to [`done.md`](done.md) — in the same
change that reports it shipped — and trim the open entry here to what
actually remains. Status narrative ("shipped", "landed", "resolved",
"update: ... now exists") must not accumulate in this file; where
context about finished work is needed, one pointer to the `done.md`
entry is enough.

# Next steps

## Naive GSPMD transform path — phased illegality (remaining)

Owner-approved 2026-07-18; phases 0–3 and every named consumer wave
are shipped (entries in [`done.md`](done.md); latest: the analytic
all-periodic route + ETDRK4 halves, merges `f8358720`/`bce54cff`
2026-07-19, design in
[`../research/analytic_eigenmode_distributed_route.md`](../research/analytic_eigenmode_distributed_route.md)).
Still open:

- **Tier-2 illegality — owner-decided 2026-07-19, implementation
  open.** All-local naive transforms on a multi-device mesh (the
  silent all-gather) become illegal, as recommended, with an
  explicit allow-replicated escape for the irreducible cases: the
  Chebyshev-vertical solve, the mismatched-layout composite solve,
  plus the recorded Wave-B tier below (until Wave B lands).
- **Wave B — walled-vertical analytic tier.** The analytic eigenmode
  route serves plain-Fourier all-periodic frames; walled-vertical
  analytic grids (`ComposedTransform`, trig z stage + `ModeChart`
  embed/restrict) keep the taught error and need a
  `ContractPlan`-shaped region absorbing the bounded axis into the
  stacked column (design record §4).
- **No-gather random synthesis for frame-mismatch cases.** When the
  sharded axis is the single-device half axis (nh2 x-sharded, sw2
  2-D), random-state synthesis falls back to a replicated backward —
  correct and device-invariant, but a gather on the IC path; a pure
  fused route needs a Hermitian half-axis re-expression of the
  device-independent gains.
- **Trig/mixed transform families**: `resolve_distributed_transform`
  declines them (plain Fourier only).

## Channel eigenmodes on multi-device — remaining gap

Projection, synthesis, and every reachable sharded channel case are
served (fused contraction `e60259de`, half-axis re-designation
`feade7fa`, the 2-D channel via `Channel2DPlan` `8752170a`; both
surfaced eigenbasis faults fixed — entries in [`done.md`](done.md)).
One defensive decline remains:

- **Non-1-D meshes**: **unreachable today** (the decomposition
  negotiates only single-axis layouts; a hand-built 2-axis mesh dies
  at decomposition build) — keep the decline. The pencil primitive
  (per-mesh-axis `all_to_all` in one 2-D-mesh `shard_map`) is proven
  composable for the day a 2-D backend lands.

Evidence, provenance probes, and the full re-attribution history:
[`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md).

## shallowwater2 physical-components flip — campaign

Ruling (c) of
[`../decisions/physical_state_components.md`](../decisions/physical_state_components.md)
(owner-ratified 2026-07-19): move the spherical prognostics from the
chart convention (`dlon/dt`, `dphi/dt`) to physical m/s components —
the NEMO/MITgcm curvilinear standard — completing the "state
components are physical" invariant (ruling (a)) across all packages.
Conversions are pointwise diagonal metric rescales, but the flip
reverses a deliberate recorded design: it touches the chart operator
plumbing (`lower_index`/`curl`/`div` flows), the Sadourny
energy-conserving spellings, the energy correction and the eigen
machinery, and must re-prove the energy-exactness gates. Retires
`u_physical` / `v_physical` (the interim conversion points) and
brings physical IC input to the sphere. Standalone campaign — plan
before implementation.

## Finite-volume nonhydro — decisions and validation

All FV stages (F0–F6) are shipped — every non-immersed grid serves
`family="fv"`, unmapped/unimmersed grids are FV by *default*, moving
geometry now runs on FV too (ALE-on-FV closed, scoping §13 addendum 2),
and the FV nonhydro is feature-complete against nodal except cut cells
(out of scope by decision; entries in [`done.md`](done.md), records in
the scoping §10–§13). Open:

- **Stretched + terrain-following combined — residuals.** The
  correctness question and the implementation campaign shipped
  2026-07-17 (entry in [`done.md`](done.md); research + rulings in
  [`../research/stretched_terrain_combined.md`](../research/stretched_terrain_combined.md)).
  Open:
  - **Variable-depth split-explicit free surface** (H3 residual):
    still a taught error on charts. The *implicit* half shipped
    2026-07-18 (multigrid_generalization_plan phase B: the
    volume-exact variable-csqr solve, resolving the plan §8
    volume-vs-energy tension for the implicit variant); the
    subcycle's terrain transport form is the remaining half.
  - **Terrain + walled-horizontal** (the one remaining layer of the
    walled-horizontal gap; the flat/immersed gap itself is closed —
    entry in [`done.md`](done.md)): a hydrostatic model on a
    sigma-chart terrain grid with a walled *horizontal* axis still
    fails to assemble. This layer is a genuine missing conversion,
    not a tag relabel: the mapped slope gradient
    (`hydrostatic/modules/core.py` `_slope_gradient` →
    `spatial/coordinate_mapping.py` `_at_space` → `field.to`)
    needs to *interpolate* a wall-normal-face quantity along the
    walled axis, and the BC-free `('interpolate', Inner(x))` row
    (correctly) does not exist. Likely spelling: retag the
    face quantity onto its Dirichlet sibling first (the odd-parity
    claim of `_dirichlet_mid`), so the registered tagged
    interpolate row resolves — but the seam sits in the shared
    `coordinate_mapping` machinery, so the claim needs a
    per-call-site justification, not a blanket arm.

[`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md)

## Immersed partial cells — residuals

The immersed grid works in all three models with genuine partial
cells in every dimension (stages I0–I4 shipped 2026-07-17; entry in
[`done.md`](done.md), record + per-stage corrections in
[`../plans/active/immersed_partial_cells_plan.md`](../plans/active/immersed_partial_cells_plan.md)).
Open, none blocking:

- **sw2 mapped+immersed** (the last open leaf of the composition
  follow-ups — everything else closed 2026-07-19, entry in
  [`done.md`](done.md)): stays a taught error, now at BOTH bind sites
  (`SadournyAdvection` and `DynamicalCore`, so linear models are
  refused too). Scoped 2026-07-19: the explicit-tendency wiring
  (α on the sqrt_g-weighted flux, sealed /θ divergence, masked
  momentum) is mechanical, but the lift is gated on (a) a
  sqrt_g-weighted embedding-chart fraction quadrature (the MI-D1
  Jacobian path serves column corrections only, not `chart_coords` —
  the separable per-axis fallback is not the physical wet volume on a
  non-separable induced metric) and (b) re-deriving the semi-discrete
  energy-antisymmetry proof with the combined α·sqrt_g corner weight
  (the stage-B energy gate has no mapped analog). No in-repo consumer
  needs it; revisit when a concrete curvilinear-with-islands use case
  appears
  ([`../plans/active/mapped_immersed_composition_plan.md`](../plans/active/mapped_immersed_composition_plan.md)).
- **Partial-bottom `p_hyd` on terrain charts** (the flat/stretched
  correction itself — the third residual — shipped 2026-07-19; entry
  in [`done.md`](done.md)): the terrain-chart (+immersed) leg is the
  recorded PB-D3 deferral — status quo there is the uncorrected
  O(dz)-at-cuts behavior, nothing regresses
  ([`../plans/active/partial_bottom_phyd_plan.md`](../plans/active/partial_bottom_phyd_plan.md)).
- **Immersed closure deferrals** (the fourth residual itself —
  fraction Sadourny + harmonic closures — shipped 2026-07-19; entry
  in [`done.md`](done.md)): no-slip immersed side-drag, Smagorinsky
  immersed (walled Smagorinsky first), VerticalMixing immersed (the
  wet-aware variable-dz tridiagonal, its own item when picked up) —
  all taught errors with recorded designs
  ([`../plans/active/immersed_closures_sadourny_plan.md`](../plans/active/immersed_closures_sadourny_plan.md)
  §5).
- **All-wet ≤32-ulp pin — owner ratification pending** (same plan,
  gate B-G4): the immersed-Sadourny all-wet parity test pins the JIT
  momentum tendency to ≤32 ulp of field scale instead of bitwise —
  the eager path *is* exactly bitwise; the residual is XLA
  FMA-contraction on the mandated identity-at-α=1 fraction ops,
  measured worst case 1.0 ulp.

## Diffusion/friction closures at walls and on terrain — residuals

Stages 0–4 shipped 2026-07-17 (walls free/no-slip on the nodal
family, implicit no-slip rows, mapped along-σ, the `VerticalMixing`
stretched/terrain gates, the measure-divide VJP seal), and the FV
walled lift shipped 2026-07-18 (walled `CellAvg` targets take the
same flux-retag closure — both families now covered; entries in
[`done.md`](done.md), record + §9 addendum
[`../research/diffusion_walls_terrain_scoping.md`](../research/diffusion_walls_terrain_scoping.md)).
Open:

- **Measure-aware implicit column** — the `VerticalMixing`
  stretched/terrain gates stand until the banded column learns
  `grid.measure` widths + the terrain Jacobian (the multigrid V-cycle
  already consumes measure widths on stretched columns — N3, entry in
  [`done.md`](done.md) — this is its implicit-diffusion twin; pairs
  with the flagged variable-kappa follow-up, `implicit.py`).
- **Stage 5** — the geopotential-correct full-metric (then rotated)
  diffusion tensor; deferred, separate plan (record §3.6 A/C).
- **Owner ratification** — `slip="free"` default **ratified
  2026-07-19** (record §6 call 1). Still shipped-unreviewed:
  biharmonic same-treatment-both-passes (incl. no-slip), along-σ as
  the first terrain deliverable (record §6 calls 2–3).
- **`grid.measure` pre-assembly ordering (lead)** — querying a
  mapped mesh's measure before `Model` assembly freezes the
  decomposition early; the stale cached measure then
  shape-mismatches the step frame. Normal build→run ordering is
  unaffected (probe artifact; matters for diagnostics workflows).
- Partial slip (NEMO `shlat`-style) = a Robin wall flux — a
  boundary-closure 2e consumer (entry above).

## Mapped-solve residual levers — measured
*The parent line — "multi-device compile and execution cost", formerly
roadmap 3.9 — closed 2026-07-16 (entry in [`done.md`](done.md)). What
survives is a short list of measured, deliberately-not-taken levers —
promote one only when its trigger appears:*

- **Single-precision distributed solve** — `single_precision_solve` is
  a no-op on multi-device walled/mapped grids (the full-precision
  distributed solve takes precedence; documented at
  `spectral_solve.py`). Worth roughly the single-device −10% if a
  multi-device user ever asks.
- **Distributed-transform planner size floor** — small problems pay
  unamortized collective latency (32³ walled/mapped +5.5%); a size
  floor on the planner is the lever if toy-size multi-device runs ever
  matter
  ([`distributed_transform_plan.md`](../plans/active/distributed_transform_plan.md)).
- **Surplus staggered reblock leg (`n = n_cells + 1`)** — deliberately
  still on the global reblock path (needs a `P*(cells+1)` frame plus
  one realigning collective-permute; gate documented in
  `decomposition/tensor.py`). No hot-loop consumer exists — the Neumann
  pressure sibling keeps the pressure-space shape. Revisit only if a
  Neumann-outer field enters a hot loop.

## Gaps against the Oceananigans reference comparison

The 2026-07 matched-protocol comparison against Oceananigans.jl
0.105.3 (suite and full report live in the untracked
`benchmarks/comparison` — out-of-tree by design, see its README)
confirmed the new stack wins where it is pressure-solve-bound. The
three trailing areas it identified are largely closed — single-GPU
memory ceiling, time-to-first-step, WENO throughput (entries in
[`done.md`](done.md)). Still open:

- **Re-run the full comparison suite** on post-fix dev. The
  2026-07-17 single-GPU recheck already re-measured the changed rows
  (weno5 512³ 130.5 ms/step, oc edge 1.57×); what remains is the
  full-table refresh — including the multi-GPU scaling rows — blocked
  on a 4-GPU allocation. New runs report the honest `compile_s`
  metric (chunk metric fixed 2026-07-18; entry in
  [`done.md`](done.md)). Owner 2026-07-19: sequence this as the
  **last step before the docs-rebuild writing pass** — it may
  surface further regressions that docs content should not bake in.

## Docs & examples rebuild

The bulk of the rebuild: **12 example ports** (only
`shallowwater/barotropic_instability.py` is on the new stack) and the
**entire prose page tree** — `docs/source/` still holds the old-stack
pages. Also retires the pre-rendered-media machinery
(`@skip_on_doc_build`, the git-LFS videos).

Two upstream gaps it needs: an `fr.io` root alias, and `pot_vort` on
shallowwater2 (documented, absent). Reader-facing content is
owner-reviewed privately before it reaches `dev` (see AGENTS.md).
[`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md)

## Adiabatic-ramping docs — example review (deferred at landing)

ROADMAP 3.8 shipped 2026-07-17
([`done.md`](done.md) §3.8;
[`../plans/done/adiabatic_ramping.md`](../plans/done/adiabatic_ramping.md)),
with R6 (the double-ramp example + Advanced Topics docs page) merged
**on owner instruction without the private content-review pass**.
Open work: the owner review of
`examples/shallowwater/adiabatic_double_ramp.py` and
`docs/source/advanced/adiabatic_ramping.rst` (projection onto the
working tree, `REVIEW:` markers, sweep-and-apply per AGENTS.md), plus
the style-guide tensions flagged at preparation: citation
infrastructure for the unpublished JFM draft (bibtex vs the current
prose citation), whether the one-chapter Advanced Topics scaffold
stands or folds into the docs rebuild, doctest wiring for inline
snippets, and API cross-refs as literals until the new-stack API
reference lands. Owner 2026-07-19: deliberately parked — run this
review as part of the docs-rebuild review cycle, not before it.

## Cutover — retire the old stack

Gated on the docs rebuild (above) being far enough along not to break the
build, plus owner sign-off of the intentional-deltas table. **Physics
parity is closed**; what remains is mechanical. The old packages are still
on disk (136 modules, 107 test files) and still exported from
`src/fridom/__init__.py`.

- Rehome the two survivors (`framework/utils/`, `framework/logger.py`) —
  everything else depends on this decision (likely a top-level
  `fridom.utils`); then rewrite the 45 source + 16 test imports.
- Delete the old packages and their tests; rename `nonhydro2` /
  `shallowwater2`; fix the root exports, `tests/conftest.py`, the CI
  multi-device path, coverage config, benchmarks.

The old "rename `framework2` → `framework`" plan is obsolete: the split
already landed the new stack as `spatial` + `model`, so only the two model
packages still carry a `2`.
Records: [`../plans/active/cutover_parity_plan.md`](../plans/active/cutover_parity_plan.md)
(parity, sign-off) and
[`../plans/active/cutover_checklist.md`](../plans/active/cutover_checklist.md)
(the executable swap list).


---

# Sized, deferred — real work, but nothing is waiting on it

Real work, but nothing is waiting on any of it. The first two were
sized on 2026-07-13 against real diffstats of comparable landed work;
none is hard to justify *technically*, all fail the "who wants it" test
today. Promote an item the moment a consumer appears.


## Boundary closures, stage 2e — the Robin dynamic `(α, g)` path

*Medium (1-2 weeks; ~400-700 LOC source, ~300-400 test, against the
`9a95202a` analogue of 21 files / +1054).* **No consumer exists**: `BC.ROBIN`
appears only in `bc.py`, two `NotImplementedError` raises, and the tests
asserting those raises. Both models use only Dirichlet/Neumann walls.
Note (2026-07-17): flux-type boundary forcing shipped as a *tendency*
module (`BoundaryFlux`, [`done.md`](done.md)) and deliberately does
**not** consume this path (plan §BF-D1) — 2e's consumers remain
inhomogeneous boundary *values* (moving lids, prescribed wall
buoyancy) and dynamic Robin data, and none exists yet. Do it when a
model needs one — otherwise it ships untested-by-use.

Two genuine design questions, unspecified in the record, are why this is
not days: **(1)** "seeded at assembly" fights "compiles once across an α
sweep" — `Grid` is not a pytree and the registry carries no dynamic
fields, so a row seeded at assembly makes α a captured constant and a
swept α *does* retrace; α/g must arrive as traced module params read at
apply time. **(2)** Under sharding the fill runs *inside* `shard_map`, so
an inhomogeneous `g` on a boundary face is a transverse-sharded operand
needing its own `in_specs`/masking.

Stage 2f (the mirror/halo-claim widening) is small but pure perf, gated on
profiling nobody has done. Defer.
[`../plans/active/boundary_plan.md`](../plans/active/boundary_plan.md)

## Open boundaries — sponge is small, through-flow is large

*Sponge tier: SMALL (days). Genuine through-flow: LARGE (3–6 weeks,
four shippable stages; scoped 2026-07-17).* **No consumer exists** —
nothing on the roadmap needs net inflow/outflow today, and a
sponge-emulated open boundary (`Relaxation` + coordinate mask next to
a wall) is already possible with zero code. The scoping record
([`../research/open_boundaries_scoping.md`](../research/open_boundaries_scoping.md))
sizes the tiers: a packaged `SpongeLayer` module (days); prescribed
through-flow — open-side space structure (the wall face becomes a
DOF; widest blast radius), boundary data via the FV wall-flux slot or
stage 2e ghost-fill rows, an MITgcm-style volume-flux balance pass
(the Poisson solvability condition; the spectral solver itself needs
**no** surgery — predictor-only inflow, the Oceananigans route), and
Orlanski/perturbation-advection outflow as CONSTRAINT-stage
overwrites. A shallowwater2 Flather module (~1 wk) is the cheap pilot
— no elliptic solve. Five owner calls are listed in the record (§4)
before Tier 2 starts. Doing Tier 2 finally gives boundary-closure 2e
its consumer.

## Eigenmode phase I — `Banded` as a first-class realized map

*Medium (1-2 weeks; ~500-700 LOC source, 8-12 files) for the FD/nodal
vertical route — but **large** if the `Fourier ⊗ Chebyshev` headline is
taken literally.* **No consumer exists**: the walled nonhydro pressure
solve is already purely diagonal (`Fourier×Fourier×Cosine`);
variable-coefficient wall-bounded eigenmodes are served by the shipped
channel engine (`model/eigen_channel.py`); and the terrain-mapped pressure
has cross terms, so it is not block-diagonal and `Banded` would not serve
it either. The one plausible win — a stretch-only vertical map, where a
per-mode Thomas solve would replace fixed-iteration CG exactly — is
speculative and unrequested.

**Landmine:** `d/dz` in *Chebyshev coefficient* space is dense-triangular,
not banded. A real Fourier⊗Chebyshev solve needs the Shen/Galerkin BC
basis and Clenshaw–Curtis measures, neither of which is built. Only the
FD/nodal-vertical variant is actually in reach.
[`../plans/active/projection_eigenmode_roadmap.md`](../plans/active/projection_eigenmode_roadmap.md)

## High-order mapped stencils — the full lift (spike answered)

*Medium (1-2 weeks).* Held back from "next steps" on payoff, not
difficulty. The C-grid biased advection tendency is only 2nd order on
**any** mesh once the advecting velocity varies, so the mapped divisor
does **not** buy asymptotic order back for `UpwindAdvection` /
`WENOAdvection`. The real payoff is ENO/dispersion behaviour restored on
stretched meshes (the reason those schemes exist), honest design order
for standalone `WenoReconstruction` and `FiniteDifference` order > 2,
and mapped one-sided FD. That is quality on stretched meshes, not a new
capability headline — worth doing, not urgent.

The plan's blocker is answered: the 2026-07-16 Jacobian spike identified
the **same-row discrete Jacobian** as the divisor (restores design order
*and* satisfies the discrete metric identity exactly; entry in
[`done.md`](done.md)). All that remains is the lift itself, on the
recorded route
([`../plans/active/high_order_mapped_plan.md`](../plans/active/high_order_mapped_plan.md)
§3, numbers in
[`../research/mapped_jacobian_spike.md`](../research/mapped_jacobian_spike.md)).

## TangentPropagator — the D5 forward-mode surface

*Small.* `jax.jvp` of `model.tendency` (spec
[`../specs/model/04_run_loop_io.md`](../specs/model/04_run_loop_io.md));
the shared name-resolution piece shipped with `Model.propagator` (the
public reverse-mode surface — entry in [`done.md`](done.md)). **No
consumer exists** (NNMD descoped); build when one appears. Plan §5.4:
[`../plans/active/differentiability_plan.md`](../plans/active/differentiability_plan.md).

---

# Long-term goals

| #   | Task | Notes |
|-----|------|-------|
| 3.1 | **Hydrostatic model — external comparison legs** | The model itself shipped 2026-07-17 (entry in [`done.md`](done.md); record [`../plans/active/hydrostatic_model_plan.md`](../plans/active/hydrostatic_model_plan.md) §8). The **Oceananigans leg executed 2026-07-17** (out-of-tree `benchmarks/comparison` harness, single A100; machine-precision linear parity, full HY-D6 ladder — results in the bench repo's `results/HYDRO_REPORT.md`; it also surfaced the implicit+advection surface-closure instability, root-caused and fixed same day, plan §H7). Still open: the **Veros and pyOM3 legs** (**pyOM3 source access needs the owner**) — and the **owner review of `examples/hydrostatic/comparison_baseline.py`** — landed on `dev` 2026-07-17 by owner authorization *before* review (deviation from the examples-review workflow, owner instruction in chat); the review itself is still owed (owner 2026-07-19: deferred to the docs-rebuild review cycle) — sweep `REVIEW:` markers / direct edits when it happens. Designed-fors (T/S + EOS, topography / variable-`csqr` CG solve, z*/ALE, spherical) stay in plan §7. |
| 3.7 | **Spherical nonhydro** | The 3D spherical chart (`X(lon, lat, h)`, so the metric comes out diagonal and `w = dh/dt` is already physical) needs the C2 chart metrics and the C3 elliptic machinery to meet: the pressure operator becomes the Laplace–Beltrami on the chart — still SPD under the sqrt(g)-weighted product, so the PCG structure carries over, but the operator assembly must be written. Not the first 3D-spherical consumer: a hydrostatic model needs no pressure solve and is the likelier first use (3.1). |
| 3.2 | **Coupled models — design** | `jax.distributed`, field exchange between models on different meshes/devices/processes, a `Coupler` module plus regridding operators, a synchronization schedule. **Pre-designed** in [`../specs/model/09_coupling_designfor.md`](../specs/model/09_coupling_designfor.md) (precedent survey + adversarial walk + architecture; the class specs carry its CS-1..18 constraints, so 3.2 stays a pure addition). |
| 3.3 | **Coupled models — implementation** | Same-process multi-device, then multi-host. Depends on 3.2. Its old cost prerequisite (3.9/3.10 — "decomposed runs must be affordable before coupling them is credible") is met: the multi-device execution-cost line closed 2026-07-16 ([`done.md`](done.md)). |

---

## Known gaps in the shipped surface

Small, but they are promises the specs make that the code does not keep
(surfaced by the 2026-07-13 spec audit):

- `model.blank_state()` / `model.state_space(name)` — specified, never
  built. Either add the two one-liners or strike them from the spec.
- `fr.modules.WindowAccumulator` — named as shipping in four places;
  does not exist.
- `add_prognostic` — every stepper family's combine step wants it.

## Cross-cutting rules

- Mirrored tests (95% branch coverage gate), ruff-clean.
- Benchmarked with the 0.1 infrastructure (runtime, compile, memory).
- The old `framework` stays runnable until the cutover; new work does not
  go into it.
- Unstructured grids stay out of scope; the designed-for grid extensions
  must not be precluded by the iteration-1 core.
