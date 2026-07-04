# AGENTS.md

Guidance for AI agents working in **FRIDOM** (Framework for Idealized Ocean
Models). Follow the conventions below; they describe the intended house style,
which the modern code (e.g. `framework/modules/module.py`,
`framework/modules/advection/advection_base.py`,
`nonhydro/modules/linear_tendency.py`) already follows.

## Project layout

- `src/fridom/` — package source (setuptools src-layout).
  - `framework/` — core base classes and shared machinery (grids, modules,
    fields, time steppers, projections, config/backend).
  - `nonhydro/`, `shallowwater/`, `hydrostatic/` — concrete models built on
    `framework`.
- `tests/` — pytest suite; mirrors the `src/fridom` tree.
- `docs/` — Sphinx docs (sphinx-book-theme + sphinx-gallery).
- `examples/` — sphinx-gallery example scripts.
- `notes/` — HPC/ops notes. `src.bak/` — vestigial, ignore.

## Commands

```bash
pip install -e '.[dev]'                 # install with dev extras
./run_tests.sh                          # run tests across all backends
FRIDOM_BACKEND=numpy pytest tests/      # run tests on a single backend
ruff check src tests                    # lint
```

- Tests select the array backend via the `FRIDOM_BACKEND` env var
  (`numpy`, `cupy`, `jax_cpu`, `jax_gpu`). `run_tests.sh` loops over all of
  them with aggregated coverage.

## Conventions

### Imports

Order: module docstring -> `from __future__ import annotations` -> stdlib ->
third-party -> first-party `fridom`. Separate groups with a blank line.

```python
"""Base class for advection schemes."""
from __future__ import annotations

from abc import abstractmethod
from functools import partial

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

### Array backend (numpy / cupy / jax)

Never `import numpy` for compute code. The active backend lives in
`fridom.framework.configuration` and is reached through `fr.config`:

- Arrays: `fr.config.ncp` (the numpy-compatible module).
- Scipy: `fr.config.scp`.
- Dtypes: `fr.config.dtype_real`, `fr.config.dtype_comp`.

Preferred idiom is a **function-local alias** in code that uses it heavily:

```python
def some_method(self) -> ndarray:
    ncp = fr.config.ncp
    return ncp.zeros_like(self.arr)
```

### Docstrings

NumPy-style, including the project's custom `Description` section. Use `r"""`
whenever the docstring contains LaTeX. Every module needs a (usually one-line)
module docstring.

```python
def fft(self, padding: FFTPadding = FFTPadding.NOPADDING) -> fr.FieldBase:
    r"""
    Perform a Fast Fourier Transform (FFT) on the field.

    Description
    -----------
    Computes the FFT of the field. The padding parameter controls the
    zero-padding strategy.

    Parameters
    ----------
    padding : fr.grid.FFTPadding, optional
        The zero-padding strategy (default: FFTPadding.NOPADDING).

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
- Use `@pytest.fixture` (including `params=`/`autouse`) and
  `@pytest.mark.parametrize` with `pytest.param(..., id=...)`.
- Use plain `assert` and `pytest.raises(..., match=...)`.
- Backend-aware tests branch on `fr.config` (e.g. `fr.config.backend_is_jax`).
