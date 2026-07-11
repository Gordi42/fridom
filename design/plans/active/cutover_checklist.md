---
status: active
date: 2026-07-08
---

> Salvaged 2026-07-11 from the retired sibling checkout (it was never
> committed there). Written 2026-07-08, before the wave program and the
> package split, so read with:
> [`cutover_parity_plan.md`](cutover_parity_plan.md) is the live status
> — Blockers 1 and 2 below are since CLOSED (waves A/A2/B/C: closures,
> forcings, WENO/upwind, ICs, walled pressure solve, transforms/
> projections, eigenmode tiers); only docs/examples and the swap remain.
> Naming: `framework2` became `fridom.spatial` + `fridom.model`
> (2026-07-11), so the swap step "rename framework2 -> framework" is
> obsolete; the rename now only concerns `nonhydro2`/`shallowwater2`.
> Open question the split adds: old `framework/utils` + `logger.py`
> survive the cutover — decide their new home (e.g. `fridom.utils`).
> The **mechanical swap** section is the enduring value here: the
> consumer map (conftest, CI paths, pyproject coverage, test_init,
> benchmarks, docs rst) still applies.

# Cutover checklist — retire `framework` / `nonhydro` / `shallowwater`

Goal: delete the old `src/fridom/{framework,nonhydro,shallowwater}` packages and
rename `framework2`/`nonhydro2`/`shallowwater2` to the canonical names, per the
[ROADMAP Cutover section](../../../ROADMAP.md).

Status legend: `[ ]` todo · `[~]` partial · `[x]` done.

Derived from a five-angle deep-research sweep (2026-07-08): model parity
(nonhydro, shallowwater), framework-level parity, consumer map, and cutover
mechanics.

## Key structural facts (why this is tractable)

- **Coupling is already clean.** The *only* thing `framework2`/`nonhydro2`/
  `shallowwater2` import from old `framework` is `framework.utils` (+ its
  transitive `framework/logger.py`). No new code touches old grid/model/field/
  module/projection code. ~24 import lines, listed in the research.
- **Survivors at cutover:** `framework/utils/` and `framework/logger.py` stay;
  everything else in old `framework/` is deletable once blockers below clear.
- **Hydrostatic: dropped** (decided 2026-07-08). Old `src/fridom/hydrostatic`
  + `tests/hydrostatic` removed; it was self-contained (never exported, never
  imported outside its own tree). ROADMAP 3.1 is now a pure greenfield future
  feature, **no longer a cutover gate**.

---

## Blocker 1 — Finish the model ports (ROADMAP 2.7, the primary gate)

Shared work (do once, used by both nonhydro2 and shallowwater2):

- [ ] **Initial conditions library.** No `*2` equivalent exists at all; all 13
      example scripts depend on it. Port:
  - [ ] nonhydro: `SingleWave`, `KelvinWave`, `WavePackage`, `BarotropicJet`,
        `Jet`, `CoherentEddy`, `RandomGeostrophicSpectra`,
        `geostrophic_energy_spectrum` (`src/fridom/nonhydro/initial_conditions/`)
  - [ ] shallowwater: `Jet`, `SingleWave`, `CoherentEddy`, `EquatorialWave`,
        the `geostrophic_spectra` family
        (`src/fridom/shallowwater/initial_conditions/`)
- [ ] **Diagnostics.** Restore parameterful diagnostics:
  - [ ] nonhydro2: add `epot`, `etot`, `pot_vort` (full Ertel PV), `cfl`,
        `local_rossby_number`, full `rel_vort`(x/y/vector) — currently only
        `ekin` + `linear_pot_vort`
  - [ ] shallowwater2: create `shallowwater2/diagnostics.py` and wire it into
        `DynamicalCore` (mirror `nonhydro2/modules/core.py`). Its `state.py`
        docstring already *promises* `ekin/epot/pot_vort` but none exist.
- [ ] **Closures.** Port harmonic/biharmonic Mixing & Friction (+ Smagorinsky
      for nonhydro). framework2 has only `ClosureBase` stub + IMEX
      `VerticalDiffusion`. Needed by ~5 nonhydro examples. *(Or confirm
      deliberately dropped.)*
- [ ] **Forcings / wave-makers.** `GaussianWaveMaker`, `PolarizedWaveMaker`
      (nonhydro). *(Or confirm dropped.)*

nonhydro-specific:

- [ ] **Non-periodic pressure solver.** Port `RFFTPressureSolver` (DCT, walled
      boxes, multi-GPU). nonhydro2 ported only the periodic Fourier solver;
      walled/convection configs can't run today.
- [ ] **Advection schemes.** Port `UpwindAdvection`, `WENO` (only
      `CenteredAdvection` ported).
- [ ] **Eigenmode branches.** Add the divergent branch (`s="d"`) — nonhydro2
      has `s=0,±1` only.

shallowwater-specific:

- [ ] **Discrete eigenmodes.** Implement the discrete staggered-C-grid modes
      (`use_discrete=True`, old default); shallowwater2 has continuous only.
- [ ] Minor: `State.velocity` / `State.tracers` convenience accessors.

Tests:

- [ ] Mirrored tests for every ported item above, to hold the 95% patch-coverage
      gate. New model tests live under `tests/framework2/{nonhydro2,shallowwater2}/`.

## Blocker 2 — State transforms / projections (ROADMAP 2.8, "wave 7", not started)

`src/fridom/framework2/transforms/__init__.py` is an empty stub. Depends on the
wave-6 eigenmode objects (themselves an open 2.7 item). Design is complete:
[`design/specs/model/08_state_transforms.md`](../../specs/model/08_state_transforms.md).

- [ ] `fr.StateTransform` + algebra (`@`, arithmetic, `FixedPoint`, `Shift`).
- [ ] Vortical / Wave / Divergence projections.
- [ ] `Propagator`, `TimeAverage`, `OptimalBalance`.
- [ ] `model.variant(term_filter=...)` + term predicates + `fr.closures.ClosureBase`.
- [ ] Mirrored tests.
- **Deferred, NOT a blocker:** `nnmd.py` (descoped — future rewrite, no model
      propagator).

## ~~Blocker 3 — Plotting / animation & writers~~ — RESOLVED (dropped, 2026-07-08)

Decided: **not a blocker.** No porting needed; these subsystems are dropped.

- [x] **Plotting** is handled by the xarray conversion (`f.xr` export) — no
      plotting module in framework2 by design.
- [x] **Live plotting during the run is dropped**: old
      `framework/modules/animation/` (live animation, `video_writer.py`) and
      `figure_saver.py` are not ported.
- [x] **NetCDF and (old) Zarr writers dropped** in favor of the tensorstore
      strategy: `framework/modules/{netcdf_writer,zarr_writer}.py` are not
      ported; `fr.io.Writer` (zarr-format store via tensorstore, xarray/
      xgcm-openable) is the sole output path.
- [ ] Follow-through at cutover: drop the tutorial content that teaches live
      animation / netCDF output (part of the docs sweep below).

## Cutover — mechanical swap (once Blockers 1–3 clear)

- [ ] Keep `framework/utils/` + `framework/logger.py`; delete the rest of old
      `framework/`, plus old `nonhydro/`, `shallowwater/`.
- [ ] Rename `framework2`→`framework`, `nonhydro2`→`nonhydro`,
      `shallowwater2`→`shallowwater`.
- [ ] Rewrite the ~24 `from fridom.framework.utils import ...` /
      `import fridom.framework as fr` (utils-only) lines to the co-located utils.
- [ ] `src/fridom/__init__.py`: exports currently list `framework`, `nonhydro`,
      `shallowwater` (old) and do **not** list `nonhydro2`/`shallowwater2` — fix
      both the `TYPE_CHECKING` block and `all_modules_by_origin`.
- [ ] `tests/conftest.py:18` — `import fridom.framework as fr` (`capture_logs`
      fixture uses `fr.log`); repoint to surviving logger.
- [ ] Delete old test trees `tests/{framework,nonhydro,shallowwater}/`; move the
      `tests/framework2/...` trees to `tests/framework/...` etc.
- [ ] `examples/` — 13 scripts + 3 `GALLERY_HEADER.rst` import old packages;
      rewrite to the new API (pulled into docs via sphinx-gallery,
      `docs/source/conf.py:74`).
- [ ] `benchmarks/bench_{nonhydro,shallowwater,operators,fields}.py` — import
      old packages; CI smoke-runs them (`.github/workflows/tests.yml:80`).
- [ ] `docs/` — `source/fridom_api.rst`, `source/tutorials/using_models/
      fridom_api_names.rst`, and the ~10 tutorial `.rst` code-blocks.
- [ ] CI — `.github/workflows/tests.yml:40` hard-codes the old
      `tests/framework/domain_decomposition/...` multi-device path.
- [ ] `pyproject.toml` — coverage `source_pkgs=["fridom"]` denominator shifts;
      tidy the cosmetic `zarr_writer.py` comment.
- [ ] `tests/test_init.py` iterates the top-level exports — will fail until they
      are corrected.
- [ ] Docs/meta prose: `README.md`, `AGENTS.md` (import-alias + layout sections).

## Done

- [x] **Drop hydrostatic** — removed `src/fridom/hydrostatic` + `tests/hydrostatic`
      (2026-07-08). Self-contained; no external imports; not in package exports.
