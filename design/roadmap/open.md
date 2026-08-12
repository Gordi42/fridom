---
status: active
date: 2026-07-16
---

# Roadmap — open work

The tracker for work that is actually next. Companions:
[`done.md`](done.md) — the shipped record; [`deferred.md`](deferred.md)
— parked work, each entry with a promotion trigger;
[`declined.md`](declined.md) — decided against, with reasons (check
it before re-proposing). Historical task numbers (3.1, 3.7, ...) are
stable — other records cite them.

**Hygiene rule (binding): this file holds only open work.** When
something ships, its record moves to `done.md` — in the same change
that reports it shipped. When something is deferred, its entry moves
to `deferred.md` with a trigger; when something is decided against,
to `declined.md` with the reason. Status narrative ("shipped",
"landed", "resolved") must not accumulate here; one pointer to the
`done.md` entry is enough.

# The pipeline (owner-sequenced 2026-07-19)

Everything ahead of the docs rebuild is **finished before docs
writing starts** (owner ruling 2026-07-19). The perf-guard
checkpoint runs after all physics changes and before the
Oceananigans re-run.

## 1. Wave residuals (the 2026-07-19 eleven-stream wave landed — entry in [`done.md`](done.md))

- **Spherical plan owner review** —
  [`../plans/active/spherical_models_plan.md`](../plans/active/spherical_models_plan.md)
  (SP-D1..D9) awaits the owner before spherical implementation
  starts.
- **Owner calls filed by the wave:**
  - `H_ref` convention on mapped grids — **dissolved by the
    gravity-first hydrostatic API** (owner + Branch-2 verification
    2026-07-21; entry in [`done.md`](done.md)): with `gravity=` as
    the input, every step-path `c²·(1/H_ref)` site collapses to the
    plain g (verified site-by-site, no lone c² exists); depths enter
    only as the volume-exact local H_a and the flat-only physical
    extent in analytics/reporting.
  - **`hy.energy.hydrostatic_energy_weights` is vestigial**
    (superseded docstring, flat-only test consumers, wrong on any
    non-unit depth): delete + repoint its two tests (recommended),
    or fix the docstring. Public `hy.energy` export.
  - **`ThermalWindBackground` naming** — keep the shipped name
    (recommended) or rename to the roadmap's old candidate
    `ThermalWindShear` (public export, wide-ish rename).
  - **`Model.set_state` stale-multistep footgun — resolved** (owner
    ruling 2026-07-23: both prognostic writes re-warm
    unconditionally; entry in [`done.md`](done.md)).

## 2. Spherical 3-D models (3.7 — promoted, owner 2026-07-19)

Pre-docs feature: **hydrostatic first, nonhydro after; sw2 is the
metric reference.** Campaign; runs on the owner-approved plan
(`../plans/active/spherical_models_plan.md`, in flight above).
Scoping evidence:
[`../research/spherical_models_scoping.md`](../research/spherical_models_scoping.md).
Key deltas: metric-aware momentum advection on charts (the shared,
wide-blast-radius piece — flux-form advection is currently
metric-blind on charts, fenced by taught errors); hydrostatic chart
core + explicit free surface (a minimal spherical hydrostatic ships
without any elliptic work); barotropic Helmholtz on charts
(implicit/split-explicit); nonhydro `coords` threading + chart
Laplace–Beltrami pressure operator + metric projection. A torus
preset chart is a candidate pole-free test chart.

## 2b. Nondimensionalization — SHIPPED except docs (2026-07-21; entry in [`done.md`](done.md))

All code landed on dev: the scaling-object architecture across all
three packages, the term-envelope ramping (OB/AR redesign), and the
unit-factors/report/writer-metadata feature. **What remains is
reader-facing and belongs to the docs rebuild (item 5)**: `examples/`
and the old-stack `benchmarks/` files still spell the retired
`csqr=`/`rossby_number=`/`dsqr=` surface, plus the per-model
nondimensionalization doc sections (plan §E/§F).
Plan: [`../plans/active/nondimensionalization_plan.md`](../plans/active/nondimensionalization_plan.md).


## 2c. Units and field metadata — SHIPPED (2026-08-12)

Both owner-reported defects and all four collateral ones are fixed
on dev (rulings + outcome in §11/§12 of the record). `FieldMetadata`
stores the physical unit plus a `nondimensional` flag and derives
`units` from the pair; the assembly stamps the flag once from the
scaling, so nondimensional runs report `"1"` in the store, in `.xr`
and in memory, and the three writer seams answer from one rule.
Derived quantities declare their own annotation through
`ScalarField.new_quantity`; a per-package name-identity gate fails
when one carries a borrowed or default record.

The scaling frame also rides through the algebra:
`FieldMetadata.cleared()` drops the identity a value-computing op
invalidates and keeps `nondimensional`, which describes the value
system rather than the quantity — so no declaration site spells the
scaling, and an unregistered derived quantity can no longer ship a
false physical claim. (That pass also found `_lift_field` dropping
metadata outright on the constant-space broadcast, which is why
`f_coriolis` lost its annotation inside `sw.pot_vort`.)

**What remains** (§5.5 re-scoped 2026-08-12 — the "two copies of one
fact" framing was wrong; the measured overlap is five component rows
and the rest is complementary). The naming and drift halves are
closed: `UnitFactor.unit` is now `target_unit` (the unit of
`factor * value`, whose misreading produced the lat-lon bug) and a
per-package lint pins the component overlap.

Derived-quantity conversion shipped 2026-08-12 (§5.6): a `derived`
row kind, 14 rows across the three packages, a writer fallback from
the user-chosen output key to the canonical name, and
`Writer(unit_factors=)` + `UnitsView.resolve` for ad-hoc quantities.

**What remains**: `sw.ekin_full`, `sw.etot_full` and `sw.pot_vort`
carry no row, deliberately — their factors turn on the geopotential
thickness convention (the `thickness` row is `(U/Fr)^2` while `p` is
`U^2/eps`, which do not reconcile under a non-`GravityWave` sw
scaling). **Owner call.** A test asserts the absence so it reads as
a decision, not an oversight; an absent row leaves the quantity
honestly unconvertible where a guessed one would repeat §5.1.

Record: [`../research/units_metadata_investigation.md`](../research/units_metadata_investigation.md).

**Reader-facing consequence for the docs rebuild (item 5)**:
nondimensional coordinates now carry `units="1"` where they
previously carried no attribute, so CDFViewer renders `x [1]` on
`barotropic_jet.py` and `barotropic_instability.py`
(`Data.jl:690` brackets any non-empty unit; `"1"` is not a
registered display unit so it is never converted away). **Open —
awaiting the owner**: accept the `[1]` label, or suppress `"1"` in
the viewer's label composition (the recommendation: the store stays
CF-correct and the decision sits in the presentation layer).


## 2d. Operator-algebra gaps surfaced by the eddy inversion (2026-08-12)

Found while building the general streamfunction inversion
(`fridom.model.streamfunction`, record
[`../research/eddy_streamfunction_inversion.md`](../research/eddy_streamfunction_inversion.md)).
None blocks that work, which routes around them, but each is a
real limit the next caller will hit.

- **No mixed `Sine x Cosine` transform product.** `_seed_transform_rows`
  binds each trig family instance over *all* axes its meshes ground,
  so on a grid walled in x and z the sine instance is
  `Sine(grid, axes=("x", "z"))` and a Dirichlet-x tensor Neumann-z
  product raises `no DST signature on CellAvg(z, bc=NEUMANN)`. The
  pressure solve never hits this because it tags every axis Neumann.
  The inversion routes around it by keeping one trig family across
  bounded axes, which is sound only while the vertical is passive
  (the symbol never reads its mode index). A caller that genuinely
  needs `DST(x) x DCT(z)` needs axis-restrictable trig transforms.
- **The FV staggering family does not enforce R1.**
  `LinearInterp.codomain` calls `require_grounded_bounded_sides`;
  `LinearReconstruction._physical_codomain` does not, so three
  bounded BC-free rows stay seeded although their true-shape output
  reaches outside the true region: `Inner -> CellAvg` (exterior
  reach `(1, 1)`), `Right -> CellAvg` `(1, 0)` and
  `FaceAvg -> Center` `(1, 1)`. They read an unrepaired wall ghost
  instead of raising — the silence that let nh2's half-tagged
  `rel_vort_z` report an `O(v/dx)` wall column for as long as it did
  (fixed; [`done.md`](done.md)). The class docstring already commits
  to R1 and `_target_codomain` spells it by hand for
  `CellAvg -> Outer`, so this is a gap rather than a policy.
  Adding the guard needs a `reach=` override on
  `require_grounded_bounded_sides` plus an `fv_exterior_reach` twin
  of `staggering.exterior_reach` (`fv_node_offset` alignment), and
  un-seeds exactly those three rows with periodic rows untouched.
  **Blocked on one thing.** `_FluxFormAdvection._flux_divergence`'s
  mapped-column correction (`model/modules/advection.py`) hops a
  BC-free `Inner(z)` onto `CellAvg(z)` on every terrain-following FV
  run, so the guard breaks that assembly. Measured: the ghost it
  reads today holds an exact zero, so no number is presently wrong.
  A homogeneous Dirichlet claim there is bitwise identical on the FV
  path but **moves the nodal mapped trajectory**, because the nodal
  family already seeds `LinearInterp(boundary="one_sided")` for
  `("interpolate", Inner(z))` on a mapped grid — the R2 designed
  closure. The FV analogue is the missing piece: a
  `boundary="one_sided"` `Inner -> CellAvg` reconstruction row
  (today the opt-in grounds `CellAvg -> Outer` only), seeded the same
  way. That row lands first, then the guard, and the mapped-FV
  trajectory moves once — correctly.

- **`spectral_sibling` refuses a 3-D operand on a walled horizontal
  plus a rigid lid.** On `channel-x+lid` and `box-xy+lid` a walled
  horizontal axis commits the trig family to Dirichlet, and the
  passive `Outer(z)` face factor has no Dirichlet origin (n+1
  against n-1 DOFs), so the sibling raises. The eddy factories route
  around it by `jax.vmap`-ing the 2-D solve over the vertical
  (`_invert_horizontal`), which is bitwise identical to a per-level
  loop and cheaper than the 3-D solve anyway, so nothing is blocked.
  Verified **not** a distributed regression: the unmodified vorticity
  branch already refuses a sharded horizontal transform axis.

Also parked from the same pass: `build_flat_spectral_solve` is
reusable for non-pressure elliptic problems once `_neumann_sibling`
grows an `is_free` guard (a no-op in the pressure path, whose space
is always the BC-free cell scalar); today it raises on every walled
topology.

## 2e. Thin-axis remainder (elision + the walled fill shipped 2026-08-12; entry in [`done.md`](done.md))

- **A100 confirmation of flat-axis elision** (owner-submitted,
  confirmatory — the cpu evidence already carried the decision):
  `benchmarks/ci/` job written and dry-run verified. `S >= 0.50` at the
  largest size confirms; predicted 0.83–0.87 for WENO5. A leg-B
  arithmetic intensity above ~4.8 FLOP/byte would be the genuine
  surprise, overturning the bandwidth-bound premise.
- **Real `srun -n N` verification** (owner-submitted). Forced-4 host
  devices pass (462 tests), and a size-1 axis is never sharded at any
  device count, so the flat path only ever takes the single-shard
  branch — low risk, unverified under a true multi-host launch.
- **The merely-thin axis is still uncovered:** `n = 2` pays 3x/5x and no
  flat rule reaches it. Extending the widening to a modular tiled gather
  for `n < halo` looks worth costing given the measured size of the win.
- **`examples/nonhydro/dancing_eddies.py`** prose still says a thin
  vertical rules out the wide reconstruction stencils. Now doubly stale
  (the wrap fix lifted it; elision makes the flat axis the *cheap* one).
  Docs-review scope — owner-reviewed privately, per AGENTS.md.

## 3. Perf-guard checkpoint (owner-run)

After **all** physics changes above land, before the Oceananigans
re-run (owner 2026-07-19). Overdue by cadence: baselines last
re-recorded at `3130b1e4`; ~69 src-touching merges since, plus this
wave. `benchmarks/ci/step_guard.sbatch` — the owner submits; agents
never do.

## 4. Oceananigans comparison re-run

Full-table refresh on post-fix dev, including the multi-GPU scaling
rows (needs a 4-GPU allocation; new runs report the honest
`compile_s` metric). Sequenced as the last step before the docs
writing pass — it may surface regressions docs content must not bake
in. Suite lives out-of-tree in `benchmarks/comparison` (by design).

## 4b. Generalized source module — P4 remainder (P1–P3 shipped 2026-07-24, entry in [`done.md`](done.md))

What remains of
[`../plans/active/source_module_plan.md`](../plans/active/source_module_plan.md):
**P4, the example re-spells** — `internal_wave_maker.py` and
`multiple_wave_makers.py` still call the deleted
`nh.GaussianWaveMaker`/`nh.PolarizedWaveMaker`, and `wave_package.py`
(plus any other example using it) the renamed `gaussian_envelope` —
all ride the docs cycle in §5 under the private owner-review
workflow. Candidate follow-up parked in the plan: `Model.propagator`
blanket-refuses `wrt=` parameters of modules that own materialized
AUX fields, so `source.*` gradients need the `_chunk_body` surface.

## 5. Docs & examples rebuild

The bulk: **11+ example ports** and the **entire prose page tree**
(`docs/source/` still holds the old-stack pages); retires the
pre-rendered-media machinery.
**Plan refresh first** (owner-sequenced): the plan predates the
hydrostatic package, terrain/sigma charts, immersed cells,
differentiability (`Model.propagator`), and the spherical promotion —
re-enumerate the examples (two on-disk examples are uninventoried),
add the missing chapters (Models/Hydrostatic, spherical, terrain,
differentiability), per-chapter planning notes for Advanced.
Reviews parked into this cycle (owner 2026-07-19): the
adiabatic-ramping page review (R6 landed unreviewed;
[`../plans/done/adiabatic_ramping.md`](../plans/done/adiabatic_ramping.md);
the companion `adiabatic_double_ramp.py` example was **deleted** —
owner decision 2026-07-23 — so the review also reconciles the
chapter's remaining in-prose references to it) and the
`examples/hydrostatic/comparison_baseline.py` review — sweep
`REVIEW:` markers / direct edits when they run. Reader-facing
content is owner-reviewed privately before it reaches `dev`
(AGENTS.md).
[`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md)

**CDFViewer upstream wishlist** (owner 2026-07-23): improvements to
the `cdfviewer --record` animations that need CDFViewer.jl features —
the owner implements upstream; revisit the example scripts'
`cdfviewer` invocations when they ship. Collected so far:
model-time label displayed as the animation plays (in progress
upstream); colorbar height auto-matched to the plot height (a manual
`figsize=` tuned to the domain aspect works around it); axis labels
that carry the units read from the store metadata (manual
`xlabel=`/`ylabel=` kwargs work around it); tick labels in scaled
units (2000 km rather than 2.0x10^6). Title and label sizes turned
out to be plain kwargs (`titlesize=`, `xlabelsize=`) — applied in
the equatorial-waves example 2026-07-23, no upstream work needed.
Related and separately owned: the writer names the zarr record
dimension `iteration`, so `cdfviewer -a time` warns
("Animation dimension 'time' not found") and falls back to the
iteration axis. The animation is still correctly time-ordered, and
swapping the flag to `-a iteration` was rejected — the fix belongs on
the time-dimension side.

## 6. Hydrostatic external comparison legs (3.1 remainder)

After docs (owner 2026-07-19): the **Veros and pyOM3 legs** (pyOM3
source: `github.com/ceden/pyOM3`, verified reachable).
`ThermalWindBackground` (shipped) unblocks an Eady leg. Full narrative
of the shipped model + Oceananigans leg: [`done.md`](done.md) §3.1.
Designed-fors (T/S + EOS, z*/ALE) stay in
[`../plans/active/hydrostatic_model_plan.md`](../plans/active/hydrostatic_model_plan.md)
§7.

## 7. Cutover — retire the old stack

Gated on the docs rebuild being far enough along not to break the
build, plus owner sign-off of the intentional-deltas table.
**Physics parity is closed**; what remains is mechanical. The old
packages are still on disk (136 modules, 107 test files) and still
exported from `src/fridom/__init__.py`.

- Rehome the two survivors (`framework/utils/`, `framework/logger.py`)
  — target **ratified (owner 2026-07-19): top-level `fridom.utils`**;
  then rewrite the 45 source + 16 test imports.
- Delete the old packages and their tests; rename `nonhydro2` /
  `shallowwater2`; fix the root exports, `tests/conftest.py`, the CI
  multi-device path, coverage config, benchmarks.

Records:
[`../plans/active/cutover_parity_plan.md`](../plans/active/cutover_parity_plan.md)
(parity, sign-off) and
[`../plans/active/cutover_checklist.md`](../plans/active/cutover_checklist.md)
(the executable swap list).

---

# Long-term goals

| # | Task | Notes |
|-----|------|-------|
| 3.2/3.3 | **Coupled models** (design + implementation, merged 2026-07-19) | `jax.distributed`, field exchange between models on different meshes/devices/processes, a `Coupler` module plus regridding operators, a synchronization schedule. **Pre-designed** in [`../specs/model/09_coupling_designfor.md`](../specs/model/09_coupling_designfor.md) (CS-1..18; the class specs carry the constraints, so implementation is a pure addition). Order: same-process multi-device, then multi-host. The old cost prerequisite is met (multi-device execution-cost line closed 2026-07-16). |
| 3.11 | **Unstructured grids** | Promoted from an out-of-scope note (owner 2026-07-19). Binding constraint meanwhile: the core must not preclude the designed-for grid extensions (the designed-for sections of the grid specs are the enforcement surface). |
| — | **ETDRK4 on the analytic eigenbasis** | The exponential stepper (`model/time_steppers/exponential.py`) only consumes the dense **channel** eigenbasis (`ChannelEigenmodes`): it reads `eigenbasis.q`/`.omega`/`.metric` as arrays and needs a `bounded_axis`. On a doubly-periodic grid `sw.eigenbasis` returns the analytic fully-spectral `Eigenmodes` (there `.q` is a *method*, no `bounded_axis`), so `ETDRK4(dt, basis)` fails at construction — the barotropic-instability example (doubly periodic) cannot use it. Teach ETDRK4 to consume the analytic eigenbasis: it then applies on doubly-periodic grids **and** — better — the per-step cost drops from the dense O(N²) column contraction (measured ~90–110× an AB3 step at nx=96: a *net slowdown* even at a stable 16× dt, channel bench 2026-07-22) to an FFT-based transform, so the exact-linear big-dt advantage can actually pay off. Related open call surfaced by the same test: `linear=True` dissipative closures (`BiharmonicFriction`) are silently dropped under `term_filter=~linear` (ETDRK4 keeps only `real(omega)`) — decide how the exponential should carry a diagonal dissipative part. |

---

# Cross-cutting rules

- Mirrored tests (95% branch coverage gate), ruff-clean.
- Benchmarked with the 0.1 infrastructure (runtime, compile, memory).
- The old `framework` stays runnable until the cutover; new work does
  not go into it.
