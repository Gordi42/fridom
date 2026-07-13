---
status: active
date: 2026-07-13
---

# Cutover checklist — retire `framework` / `nonhydro` / `shallowwater`

The mechanical swap: delete the old
`src/fridom/{framework,nonhydro,shallowwater}` packages, rehome the two
survivors (`framework/utils/`, `framework/logger.py`), and rename
`nonhydro2`/`shallowwater2` to the canonical names.

**Scope.** This file is the *executable* swap list — the consumer map and
the order of operations. It does **not** track parity:

- Is the new stack at parity? →
  [`cutover_parity_plan.md`](cutover_parity_plan.md) (waves A–C, the
  parity audit, the intentional-deltas table awaiting owner sign-off).
- Docs and examples? → [`docs_examples_plan.md`](docs_examples_plan.md)
  (they are rebuilt, not ported; steps 10–11 below are pointers only).

Status legend: `[ ]` todo · `[~]` partial · `[x]` done.
All counts below were re-verified against `dev` on 2026-07-13.

## Already settled (no action)

- [x] **Hydrostatic dropped** — `src/fridom/hydrostatic` + `tests/hydrostatic`
      removed (a78ca3d3, 2026-07-08). ROADMAP 3.1 is a greenfield future
      feature, not a cutover gate.
- [x] **`framework2` rename is obsolete** — the package split (2026-07-11,
      `plans/done/spatial_model_split_plan.md`) already landed the new
      framework as `fridom.spatial` + `fridom.model`. Only
      `nonhydro2`/`shallowwater2` still carry a `2`.
- [x] **Blockers 1–2 closed** (waves A/A2/B/C, see the parity plan): initial
      conditions, diagnostics, closures (`Harmonic/Biharmonic
      Diffusion|Friction`, `SmagorinskyLilly`), forcings (`Relaxation`,
      both wave makers), `Upwind`/`WENOAdvection`, walled pressure solve,
      eigenmodes, and the whole `model/transforms/` package (projections,
      `Propagator`, `TimeAverage`, `OptimalBalance`, `BalanceExpansion`).
- [x] **Plotting / animation / NetCDF + old Zarr writers dropped**, not
      ported (2026-07-08). Output goes through `fr.io.Writer` (zarr via
      tensorstore) and `f.xr`.

## Gate (both must hold before step 1)

- [ ] Parity sign-off: the intentional-deltas table in the parity audit
      accepted by the owner; full suite + 95% coverage + ruff green.
- [ ] Docs/examples rebuild far enough along that deleting the old
      packages does not break the doc build (`docs_examples_plan.md`).

## Step 1 — Rehome the survivors

`framework/utils/` (10 modules: `array_ops`, `decorators`, `dtypes`,
`filesystem`, `formatting`, `jax_utils`, `mpi`, `numpy_utils`,
`printing`, `__init__`) and `framework/logger.py` are the only old-stack
code the new stack uses. The coupling is utils-only and shallow.

- [ ] **Open decision:** new home. Proposal: a top-level `fridom.utils`
      (+ `fridom.log`); no `src/fridom/utils` exists today.
- [ ] Rewrite **45 import lines in 45 new-stack source files**
      (spatial 14, model 25, nonhydro2 5, shallowwater2 1). 41 are
      `from fridom.framework.utils import ...`; 4 are `import
      fridom.framework as fr` in `spatial/operators/{base,realized,
      registry,symbol}.py`, which touch **only** `fr.utils`.
- [ ] Rewrite the same import in **16 new-stack test files**
      (`tests/{spatial,model,nonhydro2,shallowwater2}`).
- [ ] Cosmetic leftovers: the string tuples `("fridom.framework",
      "fridom.spatial", "fridom.model")` in `model/_eigenbasis.py:1351`
      and `model/transforms/balance_expansion.py:109`, the docstring
      reference in `model/transforms/time_average.py:14`, and the
      `import fridom.framework as fr` inside the `benchmarking/measure.py`
      docstring example.

## Step 2 — Freeze or delete the old-stack cross-checks

Two new-stack tests import the old stack (`import fridom.framework as
frold`) to regress against it:

- [ ] `tests/model/transforms/test_balance_expansion.py` (old NNMD).
- [ ] `tests/nonhydro2/test_advection.py` (old tendency parity).

Either delete them at the swap or freeze their reference values into
static arrays first.

## Step 3 — Delete the old packages

- [ ] `src/fridom/framework` (85 modules), minus the survivors of step 1.
- [ ] `src/fridom/nonhydro` (31 modules), `src/fridom/shallowwater`
      (20 modules).
- [ ] `tests/framework` (62 files), `tests/nonhydro` (28),
      `tests/shallowwater` (17) — 107 test files.

## Step 4 — Rename

- [ ] `src/fridom/nonhydro2` → `nonhydro`, `shallowwater2` →
      `shallowwater`; likewise `tests/nonhydro2` → `tests/nonhydro`,
      `tests/shallowwater2` → `tests/shallowwater`.

## Step 5 — Package exports

- [ ] `src/fridom/__init__.py` lists `benchmarking, framework, model,
      nonhydro, shallowwater, spatial` in both the `TYPE_CHECKING` block
      and `all_modules_by_origin` — drop `framework`, add the new home of
      utils/logger if it is top-level.
- [ ] `tests/test_init.py` parametrizes over those exports; it fails
      until they are correct.

## Step 6 — Test harness

- [ ] `tests/conftest.py:18` — `import fridom.framework as fr`; the
      `capture_logs` fixture uses `fr.log`. Repoint to the surviving
      logger.

## Step 7 — Benchmarks

- [ ] Retire the four old-stack benchmarks
      `benchmarks/bench_{nonhydro,shallowwater,operators,fields}.py`.
      The new-stack suites already live in `benchmarks/model/` and
      `benchmarks/spatial/`. CI smoke-runs the directory
      (`python -m fridom.benchmarking run benchmarks`,
      `.github/workflows/tests.yml:101`).
- [ ] `benchmarks/nonhydro_shallowwater_new_vs_old.md` is a historical
      comparison record — keep it, but it must stop being executable
      input.

## Step 8 — CI

- [ ] `.github/workflows/tests.yml:61` — the multi-device job hard-codes
      `tests/framework/domain_decomposition/test_domain_decomposition.py`
      next to `tests/spatial/decomposition`. Drop the old path.

## Step 9 — pyproject

- [ ] `pyproject.toml:81-84` — the `filterwarnings` comment points at
      `fridom/framework/modules/zarr_writer.py`; check whether the
      zarr-consolidated-metadata ignore (and the imageio/ffmpeg fork
      ignore below it) is still needed at all once the old writers and
      the video writer are gone.
- [ ] Coverage `source_pkgs = ["fridom"]` — the denominator shifts when
      ~136 old modules disappear; re-check `fail_under = 95`.

## Step 10 — Examples (owned by `docs_examples_plan.md`)

- [ ] 12 of 13 example scripts import the old packages (all of
      `examples/nonhydro/*.py` and `examples/shallowwater/
      equatorial_waves.py`; only `shallowwater/barotropic_instability.py`
      is already on the new stack). The three `GALLERY_HEADER.rst` files
      are clean.

## Step 11 — Docs (owned by `docs_examples_plan.md`)

- [ ] 13 `.rst` files reference the old packages: `source/fridom_api.rst`,
      `source/getting_started.rst`, `source/tutorials/using_models/*`
      (8 files), `source/tutorials/more_tutorials/{backend,precision}.rst`.

## Step 12 — Prose

- [ ] `AGENTS.md` — the layout section (the `2`-suffix note), the
      import-alias section (`import fridom.nonhydro2 as nh`, "old-stack
      code keeps ..."), and the `__init__.py` example that uses
      `fridom.framework.grid.cartesian`. `README.md` has no old-stack
      references.
