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

## 1. In flight — the 2026-07-19 wave

Eleven parallel streams, launched 2026-07-19 (owner rulings recorded
in the cited records):

- **Full-suite validation + push** of the tested sha.
- **Explicit terrain free surface → volume-exact** (GM-D1 option 1,
  owner-ruled — all three variants share one discrete barotropic
  physics), then **split-explicit terrain+immersed** (the `∫αJ dz`
  composition; implicit M5 quadrature as reference).
- **Hydrostatic `EnergyMetric`/`eigenmodes` terrain ps weight** —
  physically consistent energy diagnostics on charts.
- **`ThermalWindShear` module** — `−M²v` buoyancy restoring +
  `w·dU/dz` tilt; unblocks Eady setups (and a future Eady
  comparison leg).
- **Eigenvector-gauge canonicalization** in the numeric channel
  basis build (largest component real-positive; cross-build parity
  becomes assertable).
- **`VerticalMixing` immersed** — wet-aware variable-dz tridiagonal
  ([`../plans/active/immersed_closures_sadourny_plan.md`](../plans/active/immersed_closures_sadourny_plan.md)
  §5).
- **API surface**: `fr.io` root alias; `model.blank_state()` +
  `model.state_space(name)` built (specs updated; IC recipes follow);
  `add_prognostic` struck (`state.add` blessed); `WindowAccumulator`
  preset struck (idiom stays) — see [`declined.md`](declined.md).
- **`pot_vort` on shallowwater2** (metric-aware; docs upstream gap).
- **`grid.measure` pre-assembly cache fix** (recompute-on-
  renegotiation, owner-ruled; sibling-cache audit).
- **Walled Smagorinsky W1→W2→W3**
  ([`../research/smagorinsky_walls_scoping.md`](../research/smagorinsky_walls_scoping.md),
  owner-ratified: slip="free" default, no-slip via wall-row
  correction, per-cell filter width; van Driest/Scotti declined).
- **Spherical models implementation plan** (design-only; owner
  reviews the plan before implementation starts).

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

## 5. Docs & examples rebuild

The bulk: **12+ example ports** and the **entire prose page tree**
(`docs/source/` still holds the old-stack pages); retires the
pre-rendered-media machinery.
**Plan refresh first** (owner-sequenced): the plan predates the
hydrostatic package, terrain/sigma charts, immersed cells,
differentiability (`Model.propagator`), and the spherical promotion —
re-enumerate the examples (two on-disk examples are uninventoried),
add the missing chapters (Models/Hydrostatic, spherical, terrain,
differentiability), per-chapter planning notes for Advanced.
Reviews parked into this cycle (owner 2026-07-19): the
adiabatic-ramping example/page review (R6 landed unreviewed;
[`../plans/done/adiabatic_ramping.md`](../plans/done/adiabatic_ramping.md))
and the `examples/hydrostatic/comparison_baseline.py` review — sweep
`REVIEW:` markers / direct edits when they run. Reader-facing
content is owner-reviewed privately before it reaches `dev`
(AGENTS.md).
[`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md)

## 6. Hydrostatic external comparison legs (3.1 remainder)

After docs (owner 2026-07-19): the **Veros and pyOM3 legs** (pyOM3
source: `github.com/ceden/pyOM3`, verified reachable).
`ThermalWindShear` (in flight) unblocks an Eady leg. Full narrative
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

---

# Cross-cutting rules

- Mirrored tests (95% branch coverage gate), ruff-clean.
- Benchmarked with the 0.1 infrastructure (runtime, compile, memory).
- The old `framework` stays runnable until the cutover; new work does
  not go into it.
