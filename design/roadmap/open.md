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
  - **`H_ref` convention on mapped grids** — the volume-exact
    reference depth is the vertical *mesh* extent (GM-D1 literal,
    shipped), so a pure vertical re-parameterization changes the
    barotropic wave speed (a nonlinear stretch with physical depth
    1.4 over base extent 1.0 gets `g = csqr/1.0`). Alternative: the
    physical reference depth (chart-invariant). Small change if
    taken (`free_surface.py` reference depth + the metric
    one-liner).
  - **`hy.energy.hydrostatic_energy_weights` is vestigial**
    (superseded docstring, flat-only test consumers, wrong on any
    non-unit depth): delete + repoint its two tests (recommended),
    or fix the docstring. Public `hy.energy` export.
  - **`ThermalWindBackground` naming** — keep the shipped name
    (recommended) or rename to the roadmap's old candidate
    `ThermalWindShear` (public export, wide-ish rename).

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

## 2b. Nondimensionalization — Branch 3 on the scaling-object architecture (Branches 1+2 implemented 2026-07-21)

**Branch 1 (framework + shallow water)** and **Branch 2
(nonhydro2 + hydrostatic, `refactor/nh-hy-scaling`, awaiting owner
integration)** are implemented: `fr.scaling` policy objects and the
alias row, `sw.Core` / `nh.Core(aspect_ratio=)` / gravity-first
`hy.Core(gravity=)`, dual-kwarg Coriolis / stratification /
free-surface families, the de-scaled bind-adopting advection, the
re-keyed eigen/energy/diagnostics consumers, and golden-file parity
(sw 16/16 bitwise; nh dim+Rotational bitwise; hy dim/ExternalWave/
Rotational bitwise with the accepted H7 closure-row roundoff on the
closure-ON dim config). Remaining:

- **Branch 3** (`feat/ramping-envelope`, §C; independent of 2):
  `TendencyEnvelope`, composer wrap, AR `envelope=True`, OB
  rewritten (deletes the interim alias-row guard). Until it lands,
  OptimalBalance refuses mechanism-scaled models (taught error).

Must land **before the docs rebuild** — Branches 1+2 changed the
public assembly API (`csqr=`/`rossby_number=`/`coords=`/`dsqr=`/
`dt=` removed from the presets; `examples/` and `benchmarks/`'s
old-stack files still spell the old surface and are part of the
docs-rebuild pass).
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

---

# Cross-cutting rules

- Mirrored tests (95% branch coverage gate), ruff-clean.
- Benchmarked with the 0.1 infrastructure (runtime, compile, memory).
- The old `framework` stays runnable until the cutover; new work does
  not go into it.
