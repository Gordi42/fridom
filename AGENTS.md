# AGENTS.md

Guidance for AI agents working in **FRIDOM** (Framework for Idealized Ocean
Models). Follow the conventions below; they describe the intended house style,
which the modern code (e.g. `framework/modules/module.py`,
`framework/modules/advection/advection_base.py`,
`nonhydro/modules/linear_tendency.py`) already follows.

## Project layout

- `src/fridom/` — package source (setuptools src-layout).
  - `spatial/` — spatial discretization: meshes, function spaces, fields,
    operators, domain decomposition, the `Grid` assembly root.
  - `model/` — model machinery: model core (assembly, run loop, schedule),
    tendency modules, time steppers, state transforms, io, ops.
  - `nonhydro2/`, `shallowwater2/` — concrete models built on
    `spatial` + `model` (the "2" suffix drops when the old stack is
    removed).
  - `framework/`, `nonhydro/`, `shallowwater/` — the **old stack**, kept
    only until the cutover completes
    (`design/plans/active/cutover_parity_plan.md`); do not build on it.
- `tests/` — pytest suite; mirrors the `src/fridom` tree.
- `docs/` — Sphinx docs (sphinx-book-theme + sphinx-gallery).
- `examples/` — sphinx-gallery example scripts.
- `design/` — internal design records (specs, decisions, plans, research;
  see `design/README.md`). `design/roadmap/open.md` tracks **open work
  only**: when something ships, move its record to
  `design/roadmap/done.md` in the same change and trim the open entry
  to what remains — never leave "shipped/landed/resolved" narrative in
  `open.md`.
- `assets/` — project assets (logo).

## Commands

```bash
uv sync --extra dev                        # create/refresh .venv with dev deps
uv run pytest tests/ -n 8 --dist loadfile  # run the full test suite (parallel)
uv run pytest tests/framework              # run a subset (serial)
uv run pytest tests/ -n 8 --dist loadfile --cov  # full suite with coverage

uv run ruff check src tests                # lint (must stay at zero errors)
uv run pre-commit install                  # install the ruff pre-commit hook

# multi-device suite (reruns the decomposition tests on 4 forced host devices)
XLA_FLAGS=--xla_force_host_platform_device_count=4 FRIDOM_TEST_FORCED_DEVICES=4 \
  uv run pytest tests/framework/domain_decomposition/test_domain_decomposition.py

# multi-process (multi-host) run: N OS processes, one GPU each, via SLURM.
# The script must call jax.distributed.initialize() BEFORE importing fridom.
JAX_PLATFORMS=cuda srun -n 4 --gpu-bind=none .venv/bin/python your_script.py
```

- **Forced host devices vs. real multi-process.** Two different
  parallelisms, do not conflate them. `XLA_FLAGS=--xla_force_host_platform_device_count=4`
  gives **one process** N devices (single-controller GSPMD): the whole
  global array is addressable by that one process, so host fetches
  (`np.asarray`, `.xr`) and single-writer I/O just work. This is what the
  test suite and CI use. **Real multi-process** (`srun -n N` +
  `jax.distributed.initialize()`) gives **N processes** one device each:
  every array is sharded across processes and each process addresses only
  its own shard. Code that host-fetches a *global* array
  (`np.asarray(distributed_array)`) raises `Fetching value for jax.Array
  that spans non-addressable devices` — gather it with
  `jax.experimental.multihost_utils.process_allgather(arr, tiled=True)`
  instead, and coordinate any shared-file writes across ranks
  (`process_index()` + `multihost_utils.sync_global_devices(tag)`).
  A path that passes forced-4 can still be multi-process-broken; verify
  multi-host behaviour under a real `srun -n N` launch.

- **Multi-process launch recipe (DKRZ A100 nodes).** Use
  `--gpu-bind=none`, **not** `--gpus-per-task=1`: the latter makes each
  task's cgroup expose one GPU as local index 0, while jax's SLURM
  auto-detection binds local index = `SLURM_LOCALID`, so ranks 1..N-1 get
  "no supported devices found for platform CUDA" and the coordinator
  hangs. With `--gpu-bind=none` all tasks see all GPUs and
  `jax.distributed.initialize()` (called before importing fridom, which
  touches the backend) assigns one GPU per local rank. Guard a run under
  `timeout` — a rank that dies leaves the others blocked at the
  coordination barrier until the heartbeat times out. Caveat
  (2026-07-17, jax 0.10.2): bare `initialize()` SLURM auto-detect can
  segfault binding the coordinator to `[::]` (IPv6). If it does,
  initialize explicitly — still before importing fridom:
  `coordinator_address="localhost:<free port>"`,
  `num_processes=int(os.environ["SLURM_NTASKS"])`,
  `process_id=int(os.environ["SLURM_PROCID"])`,
  `local_device_ids=[int(os.environ["SLURM_LOCALID"])]`.

- For full-suite runs use pytest-xdist with `--dist loadfile`: tests in the
  same file share jit-compilation caches, so grouping by file minimizes
  redundant compilations across workers. For single files or debugging
  (`-x`, `--pdb`), run serially.

- `-n 8` is the sweet spot even on many-core nodes. The suite is
  compilation-bound with per-file distribution, so a tail of slow tests
  plus per-worker jax-import startup set the floor; raising the worker
  count past ~16 is counterproductive (measured slower), and extra CPUs
  do not help.

- `tests/conftest.py` auto-enables a **persistent jax compilation cache**
  (`.jax_cache/`, gitignored) so the many small compilations are only
  paid once; warm runs are roughly twice as fast on both cpu and gpu.
  Override the cache dir with `FRIDOM_TEST_JAX_CACHE_DIR` (set it empty to
  disable). The cache is keyed on HLO + jaxlib version + backend, so it is
  safe across code changes.

- `tests/conftest.py` also **evicts jax's in-process compilation cache at
  every test-file boundary** (`pytest_runtest_teardown`). jax never evicts
  that cache on its own, so over a long xdist worker session it grows
  without bound and peaks near ~20 GB across four workers on the full
  suite — enough to OOM a 16 GB CI runner (the cause of the intermittent
  "lost connection" single-device CI failures). Boundary eviction caps the
  suite at ~13 GB at no wall-time cost (a new file traces fresh shapes, so
  little live reuse is lost, and the persistent on-disk cache turns any
  genuine reuse back into a cheap disk read). This is **standing test
  infrastructure** — keep it on; new tests inherit it automatically and
  need do nothing. Do not disable it (`FRIDOM_TEST_CLEAR_CACHE=0`) except
  for a deliberate memory experiment. Two caveats: the eviction fires only
  *between* files, so a single very large test file keeps its whole
  footprint on one worker (split such a file rather than clearing within
  it — clearing mid-file forces recompiles on the critical-path worker for
  little gain); and `jax.clear_caches()` returns memory well only when
  called often, which the many-file boundaries of the full suite provide.

- The suite runs on the gpu out of the box: `conftest.py` sets
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` so pytest-xdist workers share the
  single device instead of each preallocating ~75% of its memory (which
  otherwise stalls all but one worker). Force the backend with
  `JAX_PLATFORMS=cpu` / `JAX_PLATFORMS=cuda`. The micro-test suite is
  fastest on cpu (gpu kernel-launch latency dominates the tiny problems);
  the gpu is primarily for the `benchmarks/` suite.

- The repo ships a uv-managed environment (`.venv` + `uv.lock`); run everything
  through `uv run` (or activate `.venv`) so the correct interpreter and pinned
  dependencies are used. `uv sync --extra dev` provisions the dev toolchain
  (pytest, coverage, ruff, ...).
- FRIDOM is **jax-only**: jax is a core dependency and all compute arrays are
  `jax.numpy` arrays. The compute platform (cpu/gpu/tpu) is selected through
  JAX directly (e.g. the `JAX_PLATFORMS` env var); tests run on whatever
  platform jax picks.

### Testing policy

- Run only the tests for the files you changed; do **not** run the full
  test suite unless the user explicitly asks for it (CI runs the full
  suite on every push).
- Map each edited source file to its mirrored test file:
  `src/fridom/<pkg>/<path>/<mod>.py` ->
  `tests/<pkg>/<path>/test_<mod>.py`.
  - If the mirrored test file does not exist, run the nearest mirrored
    test directory instead (e.g. `src/fridom/framework/utils/*.py` ->
    `tests/framework/utils`, falling back to `tests/framework`).
  - When editing an `__init__.py`, also run the sibling `test_init.py`.
  - When editing a test file, run that test file.
  - **Oversized-module exception:** the one-test-module-per-source-module
    rule bends when a single module's test file grows so large that it
    both hurts readability and (because `--dist loadfile` cannot split one
    file) pins a whole xdist worker as the suite's wall-clock floor. Such a
    module may be covered by several **prefix-mirrored shards**
    `test_<mod>_<aspect>.py` (e.g. `test_advection.py`,
    `test_advection_walls.py`, `test_advection_background.py`,
    `test_advection_mapped.py`) that together mirror the module. Keep the
    `test_<mod>` prefix so the changed-file -> tests mapping still resolves
    by glob (`test_advection*.py`), and split along cohesive concerns.
    Prefer this to splitting a cohesive *source* module purely to speed
    the tests. Because the suite uses `--import-mode=importlib` with no
    shared-helper convention, each shard is self-contained: duplicate the
    few small shared builders rather than importing across test files.
- Changes to framework core machinery (`framework/utils/`, fields, the
  module system, `model.py`, `model_settings_base.py`, time steppers,
  grid base classes) affect all model packages. After the mirrored
  tests pass, additionally run one model smoke file:
  `uv run pytest tests/nonhydro/test_linear_model.py`.
- Full-suite runs (only when explicitly requested):
  `uv run pytest tests/ -n 8 --dist loadfile`.

### Differentiability policy (new stack)

- The new-stack step path is reverse-mode differentiable end to
  end: `jax.grad` through a model run w.r.t. any bound parameter
  (`friction.nu`, `dt`, an initial field) is exact to
  finite-difference precision, including the CG pressure solve (see
  `design/research/jax_grad_run_investigation.md` for the status,
  the kernel recipe, and the known hazards). This is a tested
  invariant, not an accident: treat a NaN gradient as a bug.
- New or changed step-path code (tendency modules, closures,
  spatial operators used in tendencies, time steppers) ships one
  small autodiff regression test in its mirrored test file:
  `jax.grad` of a quadratic loss through a short run via the pure
  kernel (`fridom.model.model._chunk_body`; pattern:
  `tests/model/test_model_autodiff.py`) is finite and matches a
  central finite difference to rtol 1e-4. Keep it cheap: <=16^2/8^3
  grids, <=10 steps, one test per feature — a few seconds of
  compile. Host-side code (io, reporting, assembly) is exempt.
- The usual poison is a masked singularity: `x / y` or `x ** 0.5`
  where sealing/clipping keeps the forward value finite but the VJP
  is singular (`y == 0` in never-valid padding, `sqrt` at a clipped
  0). Guard with the double-`jnp.where` pattern, or — when the
  guard costs step time — a `custom_jvp` whose primal is untouched.
  Never `custom_vjp` in step-path code: it forecloses forward-mode
  `jvp`.

### Coverage policy

- The project enforces **95% branch coverage** (`fail_under = 95` in
  `pyproject.toml`; Codecov gates project and patch coverage at 95%).
- New or changed code must ship with tests in the mirrored test file so
  that patch coverage stays >= 95%.
- Coverage is opt-in locally: add `--cov` to the pytest invocation
  (pytest-cov combines data across xdist workers automatically).
- Prefer testing hard-to-test code (I/O, animation, MPI) with
  monkeypatching/mocks. Use `# pragma: no cover` only for lines that are
  truly unreachable in tests; structural exclusions
  (`if TYPE_CHECKING:`, `@abstractmethod`, `raise NotImplementedError`)
  are already configured in `[tool.coverage.report]`.

## Conventions

### Imports

Order: module docstring -> `from __future__ import annotations` -> stdlib ->
third-party -> first-party `fridom`. Separate groups with a blank line.

```python
"""Base class for advection schemes."""
from __future__ import annotations

from abc import abstractmethod
from functools import partial

import jax.numpy as jnp
import numpy as np

import fridom.framework as fr
```

- `from __future__ import annotations` is **required** in every source module.
- New-stack code uses the **root alias**: `import fridom as fr`, then
  `fr.spatial.Grid`, `fr.model.Model` (lazy subpackages). Model packages
  by alias: `import fridom.nonhydro2 as nh`,
  `import fridom.shallowwater2 as sw`.
- Old-stack code (only) keeps `import fridom.framework as fr`,
  `import fridom.nonhydro as nh`, etc.
- Use **absolute** imports in source modules. Relative imports (`from .`) are
  allowed **only** inside `__init__.py` `TYPE_CHECKING` blocks.
- Guard annotation-only / circular imports with
  `if TYPE_CHECKING:  # pragma: no cover`.

### `__init__.py` (lazy loading via lazypimp)

Every `__init__.py` uses `lazypimp.setup`. Duplicate the real imports inside a
`TYPE_CHECKING` block (so IDEs/type checkers resolve names), then declare the
`all_modules_by_origin` / `all_imports_by_origin` dicts and call `setup(...)`
as the last statement.

```python
"""Base module for Cartesian grids."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import discrete_spectral_operators

    # import all classes
    from .fft import FFT
    from .grid import Grid

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework.grid.cartesian"

all_modules_by_origin = {
    base: ["discrete_spectral_operators"],
}

all_imports_by_origin = {
    f"{base}.fft": ["FFT"],
    f"{base}.grid": ["Grid"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
```

### Arrays (jax)

Compute code uses **jax** directly:

- Arrays: `import jax.numpy as jnp` (never `import numpy` for compute code;
  numpy is fine for host-side work like time handling or I/O).
- Scipy: `import jax.scipy as jsp`.
- Dtypes: `fr.utils.dtype_real()` / `fr.utils.dtype_comp()` — these follow
  the jax `jax_enable_x64` flag, which fridom enables at import (float64 by
  default).
- In-place-style updates: `fr.utils.modify_array(arr, where, value)`
  (wraps `arr.at[where].set(value)`).
- jit: decorate with `@fr.utils.jaxjit`; register classes as pytrees with
  `@fr.utils.jaxify` (subclasses of jaxified classes are auto-registered;
  apply the decorator only to mark additional `dynamic=(...)` attributes).
- Modules in jit-compiled containers (e.g. `mset.tendencies`) must be
  **pure**: no Python-side state mutation, no branching on traced values.
  Stateful modules (e.g. `Counter`) belong in `mset.diagnostics`.

### Docstrings

NumPy-style, including the project's custom `Description` section. Use `r"""`
whenever the docstring contains LaTeX. Every module needs a (usually one-line)
module docstring.

```python
def fft(self, axes: tuple[int] | None = None) -> fr.FieldBase:
    r"""
    Perform a Fast Fourier Transform (FFT) on the field.

    Description
    -----------
    Computes the FFT of the field. The axes parameter selects the
    axes to transform.

    Parameters
    ----------
    axes : tuple[int] | None, optional
        The axes to transform; None transforms all axes (default: None).

    Returns
    -------
    fr.FieldBase
        The FFT of the field.
    """
```

- Parameter entries use plain `name : type` (no backticks around name/type).
- Optional/default values: `name : type, optional`, with the default stated in
  the description text as `(default: X)`.
- reST directives are used for maths and code: `.. math::`, `:math:`,
  `.. code-block:: python`.

### Typing

- Use PEP 604 unions: `str | int | None`. Do **not** use `typing.Union`,
  `typing.Optional`, or quoted forward-reference strings (rely on
  `from __future__ import annotations`).
- Instance attributes are typed inline in `__init__`
  (e.g. `self.mset: fr.ModelSettingsBase | None = None`).

### Structure & naming

- Classes: `PascalCase`. Functions/methods/variables: `snake_case`.
  Module-level constants: `UPPER_CASE`. Private members: leading `_`.
  Modules/files: `snake_case`.
- Base/interface classes must inherit from `abc.ABC` and mark their interface
  methods with `@abstractmethod` (so abstractness is enforced).
- Model components use the `@fr.utils.jaxify` decorator (or
  `@partial(fr.utils.jaxify, dynamic=(...))`).
- Expose private `_attr` fields through `@property` in a dedicated section.
- Organize files with banner comments (64 `=` chars) and lighter `-` dividers:

  ```python
  # ================================================================
  #  Properties
  # ================================================================
  ```

- Double quotes for strings. Max line length **79/80** characters.
- No semicolon statement-joining. No trailing whitespace.
- Task markers: `# TODO(Silvano): ...`, `# FIXME(Silvano): ...`.

### Linting

- Linting is done with **ruff**, configured in `pyproject.toml`
  (`[tool.ruff.lint]` with `select = ["ALL"]` and a small set of ignores, plus
  per-file ignores for `tests/`, `__init__.py`, and `examples/`).
- Run `ruff check src tests` and keep new code clean. Use targeted inline
  `# noqa: <RULE>` suppressions only when justified.
- `pylint` config also exists in `pyproject.toml` (extension allow-list and a
  `good-names` list for common numerics variables like `ax`, `dx`, `dy`, `dz`).

### Tests

- pytest, **function-based** (`def test_*`), no `class Test...` containers.
  Files named `test_*.py`, mirroring the `src/fridom` layout.
- Each package dir has a `test_init.py` that parametrizes over the
  `__init__.py` re-exports to assert every public name imports.
- Tests requiring a specific jax device count are marked with
  `@pytest.mark.multi_device` (skipped on one device) or
  `@pytest.mark.single_device` (skipped on several devices); unmarked
  tests must pass on any device count (see the multi-device suite
  command above).
- Use `@pytest.fixture` (including `params=`/`autouse`) and
  `@pytest.mark.parametrize` with `pytest.param(..., id=...)`.
- Use plain `assert` and `pytest.raises(..., match=...)`.

## Git workflow (binding for agents)

- `dev` is the integration branch. `main` advances only by merging `dev`
  (releases); never commit to `main` directly.
- **Direct-to-dev** is allowed only for changes that touch nothing under
  `src/`, `tests/`, `benchmarks/`, or `examples/` — i.e. `design/`,
  `docs/` prose, `notes`-style records, comment/markdown fixes.
- **Everything else goes on a short-lived branch** named
  `<type>/<kebab-topic>` with
  `type` in `{feat, fix, refactor, perf, test, chore, docs}`
  (e.g. `feat/robin-boundaries`, `refactor/operators-split`). No other
  branch-name forms (no `worktree-agent-*`, no bare topic names).
- Parallel agents never share a checkout: one branch + one worktree per
  agent, and the worktree branch carries the proper `<type>/<topic>`
  name.
- **Merge gate:** the mirrored tests for every edited source file (see
  Testing policy) and `uv run ruff check src tests` must pass before
  merging.
- Land with `git merge --no-ff <branch>` onto `dev`, then **delete the
  branch and remove its worktree in the same session**. Never end a
  session with a leftover branch or worktree.
- GitHub PRs are the exception, not the rule: open one only when Silvano
  explicitly asks for a reviewable record. Push the branch, open the PR
  with `gh`, and delete the remote branch after the merge.
- **Exception: docs content is owner-reviewed privately** (standing
  request, 2026-07-11). Any change to reader-facing documentation
  content — `docs/` pages and `examples/` scripts — is reviewed by
  Silvano **before** it reaches `dev`, and the review stays off
  GitHub (the repo is public; review discussion is not). Mechanics:
  the agent works on a local `docs/<topic>` branch (**never pushed**
  until approved), builds a local preview (`make html` /
  `sphinx-autobuild`), and hands off for review by **projecting the
  branch onto the main checkout as unstaged changes**:
  `git restore --source=docs/<topic> -- docs/ examples/` with the
  checkout on `dev` (owner preference 2026-07-11: he reviews
  working-tree diffs in lazygit). Feedback arrives in that projected
  working tree: direct edits and discarded hunks (authoritative), or
  anchored markers at the exact spot: `.. REVIEW: ...` in rst,
  `# REVIEW: ...` in Python examples, `<!-- REVIEW: ... -->` in
  Markdown. Agents sweep markers (`grep -rn "REVIEW:" docs examples`),
  apply each, delete it, fold generalizable corrections into the docs
  style guide (`design/specs/docs/style_guide.md`, later
  `docs/STYLE.md`), and reconcile the reviewed working tree back onto
  the branch (his tree state wins over the branch). **Merge gate:**
  zero open markers AND Silvano's explicit approval in chat; only
  then merge onto `dev`, push, and clean the projection out of the
  working tree. Design records under `design/` and docs build
  *infrastructure* (`conf.py`, CI, templates) follow the normal
  workflow above.
- Commit messages: `<scope>: <short lowercase summary>` where scope is
  the affected package or area (`spatial: ...`, `model: ...`,
  `nonhydro2: ...`, `tests: ...`, `design: ...`), matching the existing
  history style.
