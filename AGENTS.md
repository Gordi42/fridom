# AGENTS.md

Guidance for AI agents working in **FRIDOM** (Framework for Idealized Ocean
Models). Follow the conventions below; they describe the intended house style,
which the modern code (e.g. `framework/modules/module.py`,
`framework/modules/advection/advection_base.py`,
`nonhydro/modules/linear_tendency.py`) already follows.

## Project layout

- `src/fridom/` — package source (setuptools src-layout).
  - `framework/` — core base classes and shared machinery (grids, modules,
    fields, time steppers, projections, jax utilities).
  - `nonhydro/`, `shallowwater/`, `hydrostatic/` — concrete models built on
    `framework`.
- `tests/` — pytest suite; mirrors the `src/fridom` tree.
- `docs/` — Sphinx docs (sphinx-book-theme + sphinx-gallery).
- `examples/` — sphinx-gallery example scripts.
- `notes/` — HPC/ops notes. `src.bak/` — vestigial, ignore.

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
```

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
- Changes to framework core machinery (`framework/utils/`, fields, the
  module system, `model.py`, `model_settings_base.py`, time steppers,
  grid base classes) affect all model packages. After the mirrored
  tests pass, additionally run one model smoke file:
  `uv run pytest tests/nonhydro/test_linear_model.py`.
- Full-suite runs (only when explicitly requested):
  `uv run pytest tests/ -n 8 --dist loadfile`.

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
- Import packages by alias: `import fridom.framework as fr`,
  `import fridom.nonhydro as nh`, `import fridom.shallowwater as sw`,
  `import fridom.hydrostatic as hs`.
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
