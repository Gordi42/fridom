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

Open: **derived quantities carry no conversion row**, so a
nondimensional store has `vort` with `units="1"` and no
`dimensional_factor` while `u` carries one — honest but
unrecoverable. Investigated 2026-08-12 (§5.6): the factor **cannot**
be derived from the unit string (measured: declared/from-dimensions
ratios are `eps` for `u`/`p` and `delta` for `b` — each amplitude is
an independent normalization and `delta*L` is a second length), so
per-quantity rows are the only route; no new machinery is needed
(`module.unit_factors` + live-parameter `UnitFactor`s); the writer
must additionally fall back from the user-chosen output name to
`field.metadata.name`, which the name-identity gate already pins.
Ten of the seventeen are mechanical (`U/L` for the vorticity /
divergence family, `U^2` for the energies), `nh.linear_pot_vort` is
`U/(eps*L)` rather than `U/L` because its definition folds in `eps`,
and `sw.ekin_full` / `etot_full` / `pot_vort` need the geopotential
conventions settled before their rows can be written — an absent row
is honest, a wrong one repeats §5.1. Recommendation: package rows +
a user-declarable factor on the `derived=` channel, staged.

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
