---
status: active
date: 2026-07-13
---

# Cutover-parity plan (retire framework / nonhydro / shallowwater)

The physics/parity half of the cutover is **done**: nothing in the old
packages blocks the new stack any more. What is left is the mechanical
swap (delete the old trees, rehome `framework/utils` + `logger.py`,
rename `nonhydro2`/`shallowwater2`) plus the docs/examples rebuild that
runs on its own plan.

Companion records:

- [`cutover_checklist.md`](cutover_checklist.md) — the mechanical
  consumer map (conftest, CI paths, pyproject, `test_init`, benchmarks,
  docs rst). Its blocker sections are stale; its swap section is live.
- [`docs_examples_plan.md`](docs_examples_plan.md) — owns the
  examples/docs rebuild (owner-approved 2026-07-11), including the
  private review workflow.
- [`../../research/parity_audit.md`](../../research/parity_audit.md) —
  frozen §8.8 audit (23/23 rows covered, 0 flags).

## Standing decisions

- **Hydrostatic is out of the cutover.** Decided 2026-07-08, executed
  2026-07-11 (`a78ca3d3`): `src/fridom/hydrostatic` and
  `tests/hydrostatic` are gone. It is a greenfield future feature
  (ROADMAP 3.1), not a gate. The plan's old "keep the sources parked"
  instruction is void — there are no sources.
- **`framework2` no longer exists.** It was split into `fridom.spatial`
  + `fridom.model` (`0260d779`). The swap step "rename framework2 →
  framework" (ROADMAP Cutover, still worded that way) is obsolete; the
  new core packages already carry their final names. Only
  `nonhydro2`/`shallowwater2` still need renaming.
- **Background flow is O(1), not Ro-scaled** (Wave A2 convention
  change; the old stack scaled it). Linearity is a term property
  (spec ruling V-S3): the advection module owns a separate linear
  background term, `S_lin(U, q)`; there is no separate linear module.
  `fr.linearize` replaces the old `disable_nonlinear`.
- **Deliberately not ported** (no successor planned): NetCDF and old
  zarr writers, live animation / `VideoWriter` / `figure_saver`
  (`fr.io.Writer` — tensorstore zarr — is the sole output path, and
  plotting is `f.xr`); old `nonhydro.BiharmonicClosure` (grid-scaled,
  mask-BC variant); spatially varying diffusion coefficients
  (AUXILIARY-field design point, door open); advection orders > 5;
  implicit closure treatment; mpi4py multi-host runs (`jax.distributed`
  is ROADMAP 3.2/3.3).

## Landed

- **Wave A — physics modules** (2026-07-10): closures (`ClosureBase`,
  harmonic/biharmonic diffusion + friction, nh2 `SmagorinskyLilly`,
  `14e7af5`); forcings (`Relaxation`, nh2 `GaussianWaveMaker`,
  `PolarizedWaveMaker`, `013ee2d..90d91c4`); advection (`UpwindAdvection`,
  `WENOAdvection` orders 3/5 over the grid WENO kernels, `5e5a061`;
  walled support added later in `719ff4cd`, honest-order note in
  `ee351efe`).
- **Wave A2 — background-flow advection** (2026-07-10, `8ee94f41`):
  difference-form split (owner option 3) in nh2 Centered/Upwind/WENO and
  sw2 Sadourny; old-stack tendency parity ~6e-16 at Ro=1, Doppler shift
  matches the discrete symbol (≤1e-12). Sheared `U` runs in the linear
  model; eigh-based engines refuse it via the Hermiticity guard (an
  `eig` backend is a separate future item).
- **Wave B — analytic initial conditions** (2026-07-10): sw2
  `single_wave`/`jet`/`coherent_eddy`/`equatorial_wave`; nh2
  `single_wave`/`kelvin_wave` (exact labeled channel mode, replacing the
  old approximate V=0 ansatz)/`wave_package`/`barotropic_jet`/`jet`/
  `coherent_eddy`. Wave factories return `(omega, state)` on the
  `em.mode` phase convention; `use_discrete` dropped (the eigenmodes
  *are* discrete).
- **NNMD rewrite** (2026-07-11, `859eab32`): `fr.transforms.BalanceExpansion`
  — the one descoped item, closed. The old `framework/projection/nnmd.py`
  was broken for nonhydro anyway (SW eigenvector sign convention
  hard-coded).
- **§8.8 audit closed** (2026-07-11): both owner flags wired —
  `p = φ/stage_dt` normalization (`49dfc092`) and the central `rest`
  policy in `StateTransform.call_with_info` (`b22412a4`, `48603040`).
  Audit state: 23/23 rows covered, 0 flags, 1 subtle sketch open
  (one-period dispersion, Appendix A Sketch B) and a 16-row
  intentional-deltas table still awaiting owner sign-off.
- **Pressure solves cover the old capability**: `SpectralPressureSolver`
  runs walled grids on the Neumann-tagged cosine (DCT-II) spaces, and
  `mapped_pressure.py` adds the PCG solve for mapped/terrain-following
  grids (`1d598055`). The old `RFFTPressureSolver` is therefore not a
  capability gap; only its real-FFT memory halving is unported (an
  optimization, unclaimed).

## Remaining

1. **Rehome `framework/utils` + `logger.py`.** This is the *only*
   coupling left: 41 `from fridom.framework.utils import ...` lines
   across `spatial`/`model`/`nonhydro2`/`shallowwater2`, plus four
   `import fridom.framework as fr` lines in `spatial/operators/`
   (`base`, `symbol`, `registry`, `realized` — used solely for
   `fr.utils.jaxify`), one in `benchmarking/measure.py`, and
   `tests/conftest.py:18` (the `capture_logs` fixture uses `fr.log`).
   Decide the destination (`fridom.utils` is the obvious one) before the
   deletion, since every other step depends on it.
2. **The swap.** Delete old `framework/` (85 modules), `nonhydro/` (31),
   `shallowwater/` (20) and their 107 test files
   (`tests/{framework,nonhydro,shallowwater}`); rename
   `nonhydro2`→`nonhydro`, `shallowwater2`→`shallowwater` (with their
   test trees); fix `src/fridom/__init__.py`, which today exports the
   *old* `nonhydro`/`shallowwater` and does not export the `*2`
   packages at all; repoint `.github/workflows/tests.yml` (the
   multi-device job hard-codes
   `tests/framework/domain_decomposition/...`); retire the old
   benchmarks (`bench_nonhydro`, `bench_shallowwater`, `bench_fields`,
   `bench_operators` — `benchmarks/{model,spatial}` are the successors);
   update `pyproject.toml` coverage, `README.md`, `AGENTS.md`. The
   checklist's consumer map is the working list.
3. **Docs & examples** — 12 of the 13 `examples/*.py` and all 13
   tutorial/API `.rst` files still import the old stack. Owned by
   `docs_examples_plan.md`; explicitly **not a hold on the swap** (the
   new content is written against the current `*2` names, and the rename
   is a find/replace pass over `examples/` and `docs/`).
4. **Diagnostics gap** (decide: port or declare a non-port). nh2 exposes
   `ekin`, `epot`, `linear_pot_vort`; the old stack also had full Ertel
   `pot_vort`, `cfl`, `local_rossby_number` and the `rel_vort` family.
   sw2 has `ekin`/`epot`/`thickness` plus the thickness-weighted
   `*_full` set. This is the last unresolved *content* difference; it
   blocks nothing that runs today.
5. **Residual**: `nonhydro2.diagnostics.epot` returns `inf` at N²=0 (it
   divides by `params[STRATIFICATION_N2]` unguarded — the guard lives
   only in `EnergyMetric.from_model`).
6. **Owner sign-off** on the audit's 16-row intentional-deltas table,
   and (optional, for a strong pass) the one-period-dispersion sketch.

## Not cutover blockers

Waves 10–11 of the 2.9 substrate (linear-term blocks, symbolic
`BlockSymbol`): the probe-based eigen stack covers the functionality;
the substrate refinement proceeds independently
([`linear_term_blocks_plan.md`](../../archive/linear_term_blocks_plan.md)).
