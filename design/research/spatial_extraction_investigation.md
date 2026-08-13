---
status: frozen
date: 2026-08-13
---

# Extracting `fridom.spatial` into its own repository — investigation

Research report (see [`README.md`](README.md) for status). Question:
should `src/fridom/spatial` become a standalone repository and
distribution, on the model of Julia's ClimaCore? **Owner ruling
2026-08-13: no.** The decline is recorded in
[`../roadmap/declined.md`](../roadmap/declined.md); this record holds
the measurements behind it, and the three bugs the investigation
surfaced (all fixed the same day).

## Verdict

Extraction is **mechanically easy and semantically expensive**. The
dependency graph is already a DAG with spatial at the bottom; nothing
blocks a split. The cost is not untangling — it is converting a
freely-refactorable internal seam into a published API of ~130 symbols
across 40 modules, permanently, at bus factor 1, for a consumer that
does not exist.

## 1. Coupling — clean outward, wide inward

**Outward (spatial → rest):** 16 import lines, 4 symbols —
`dtype_real`, `dtype_comp`, `jaxify`, `jaxjit`, all from
`framework/utils`. Zero imports from `model/`, `nonhydro2/`,
`shallowwater2/`, `io/`, `ops/`, `benchmarking/`. No import cycles, no
import-time registration into spatial, interning fully internal.
`Grid.merge_overrides()` — the one write into spatial's dispatch table
— is a data handoff through a public method, not a hook. All remaining
`fr.io.` / `fr.operators.` / `fr.modules.` strings inside spatial are
docstrings referring to the *root* alias.

**Inward (rest → spatial):** `model/` alone imports 34 spatial
submodules and 85 symbols. Consumers bypass the lazypimp facade (~25
names) and deep-import ~130 symbols across 40 modules. `__all__`
appears in **0 of 95** spatial modules. There are 13 private-symbol
imports (WENO internals, the `_ensure_valid`/`_finalize`/
`_required_halo` operator lifecycle, Krylov reduction helpers) and
`ScalarField._data` is reached at **14 sites**.
`model/modules/advection.py` alone accounts for 7 of the 13 and 8 of
the 14.

## 2. Size and churn

| | value |
|---|---|
| `src/fridom/spatial` | 95 files, 48,424 LOC — 34% of src |
| `tests/spatial` | 123 files, 39,589 LOC, 3,652 tests — 30.5% |
| spatial blobs in history | 22.9 MB of 313 MB (7.3%) |
| package age | **created 2026-07-11** (`0260d779`) |
| commits touching spatial since 2026-02 | 213 |
| …that **also** touch model/nh2/sw2 | **64 (30%)** |

Each of those 64 becomes a two-repo, two-CI, ordered-merge transaction
under a split. The boundary is also still moving: `7a1be9f0` /
`b4e08512` (2026-08-12) pushed the model's scaling frame down into
spatial's `FieldMetadata`, and the next campaign (spherical 3-D) is
explicitly wide-blast-radius through `spatial/charts`.

## 3. The stated motivation dissolves at the cutover

Nine of twelve runtime dependencies (`scipy`, `netCDF4`, `zarr`,
`tqdm`, `dill`, `IPython`, `coloredlogs`, `regex`, `pillow`) are
**old-stack only**. After the swap, `pip install fridom` resolves to
jax + numpy + lazypimp + tensorstore, and an outsider already gets a
light install and `import fridom.spatial`. What a split would add over
that is the absence of the word "fridom" in the import path.

## 4. Prior art (verified subset)

- **Firedrake re-absorbed its splits.** [PyOP2](https://github.com/OP2/PyOP2)
  is archived — *"PyOP2 can now be found inside the Firedrake
  repository"* — as is TSFC. The most aggressively multi-repo project
  in scientific Python reversed course.
- **[jax-cfd is unmaintained](https://github.com/google/jax-cfd):**
  *"🚨 JAX-CFD is no longer maintained."* Google shipped a reusable
  structured-grid FVM layer attached to a PNAS paper; it acquired no
  external dependents.
- **PhiML** (106 stars) and **jaxsw** (18 stars) exist with
  essentially only their parent projects as users. Both analogies are
  weaker than they first appear — PhiML is a math/tensor layer, not a
  discretization layer, and jaxsw is an ocean model, not a framework.

Claims *not* independently verified and deliberately excluded: the
diffrax scope quote, survival-analysis hazard ratios, jaxdf LOC
counts, all download statistics.

## 5. Bugs surfaced (all fixed 2026-08-13)

1. **x64 was never enabled by `import fridom`.** Live bug, not just a
   post-cutover hazard: `jax.config.update("jax_enable_x64", val=True)`
   lived only in `framework/__init__.py`, and lazypimp defers
   everything, so a bare `import fridom` measured `False float32` on
   `dev` until something transitively pulled in a `framework` module.
   The cutover's deletion of `framework/` would have made it
   permanent. Moved to the root `__init__.py`; subprocess regression
   test in `tests/test_init.py`.
2. **`jaxify`'s structural-equality gate hardcoded `"fridom"`**
   (`jax_utils.py`). It decides whether two pytree treedefs compare
   structurally or by identity — i.e. whether jax reuses a jit cache
   entry — and a rename would have sent 167 classes down the identity
   path silently. Now derived as `_ROOT_PACKAGE = __name__.partition(".")[0]`.
   Note the mechanism: `jaxify` sets `cls.__eq__ = _structural_eq`, so
   jaxified classes never reach the branch; it serves non-jaxified
   objects held as static aux data (`ConjugateGradient`,
   `MultigridVCycle`, `Decomposition`, …).
3. **Inverted test import** — `tests/spatial` reached up into
   `fridom.model.errors` to assert a re-export. Moved to
   `tests/model/test_errors.py`.

## 6. Known gaps, not closed

- **Spatial's own coverage is partly borrowed.** Three of five CI
  pytest legs are `nonhydro2` multigrid files chosen specifically to
  guard `spatial/operators/transfer.py` (the SPMD miscompile in
  [`semicoarsen_multidevice_regression.md`](semicoarsen_multidevice_regression.md)).
  Unmeasured: `tests/spatial`-only branch coverage of
  `operators/{transfer,multigrid,multigrid_hierarchy,krylov}.py`. This
  is a real question independent of any split —
  `uv run pytest tests/spatial --cov=src/fridom/spatial`.
- **Second upward test import** still open:
  `tests/spatial/decomposition/test_graded_nodal_multi_device.py:27`
  imports `_BiasedFaceReconstruction` from `fridom.model`. Relocating
  it to `tests/model/` would drop it from the forced-4 CI leg, whose
  glob covers only `tests/spatial/decomposition` — the workflow glob
  must widen in the same change.
- **Two more hardcoded root-package tests** in
  `model/_eigenbasis.py:1580` and `model/transforms/balance_expansion.py:108`.
  Left alone deliberately: they are dispatch guards off the jit-cache
  path whose rename failure mode is a loud, tested `ValueError`, and
  each hardcodes three subpackage names plus error-message text.
  Deriving only the root would make them *look* rename-proof.
