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
uv run pytest tests/                       # run the test suite
uv run pytest tests/framework              # run a subset
uv run ruff check src tests                # lint (must stay at zero errors)
uv run pre-commit install                  # install the ruff pre-commit hook
```

- The repo ships a uv-managed environment (`.venv` + `uv.lock`); run everything
  through `uv run` (or activate `.venv`) so the correct interpreter and pinned
  dependencies are used. `uv sync --extra dev` provisions the dev toolchain
  (pytest, coverage, ruff, ...).
- FRIDOM is **jax-only**: jax is a core dependency and all compute arrays are
  `jax.numpy` arrays. The compute platform (cpu/gpu/tpu) is selected through
  JAX directly (e.g. the `JAX_PLATFORMS` env var); tests run on whatever
  platform jax picks.

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
