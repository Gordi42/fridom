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

## 4b. Generalized source module (plan filed 2026-07-24, awaits owner)

[`../plans/active/source_module_plan.md`](../plans/active/source_module_plan.md)
(SRC-D1..D8): one generic `fr.model.modules.Source` (separable
`Q(x)·g(t)` term, complex-pattern quadrature, `Harmonic` law,
`gaussian_envelope` → `gaussian` in a new `shapes.py`,
`wave_package(quadrature=True)`) replaces and deletes the two
v1-ported wave makers. Sequencing is the owner's call; it gates the
`internal_wave_maker` / `multiple_wave_makers` example ports in §5,
so it wants to land before those.

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
