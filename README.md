[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/fridom/badge/?version=latest)](https://fridom.readthedocs.io/en/latest/index.html)
[![codecov](https://codecov.io/github/Gordi42/fridom/graph/badge.svg?token=6LY1CFM6KU)](https://codecov.io/github/Gordi42/fridom)
[![DOI](https://zenodo.org/badge/714260615.svg)](https://doi.org/10.5281/zenodo.14536978)

[![](assets/logo/fridom-final.svg)](https://www.youtube.com/watch?v=Fotni4P2ZQs)

# Framework for Idealized Ocean Models (FRIDOM)

FRIDOM is a Python framework for building idealized ocean and geophysical fluid
models. You assemble a model from a grid and a handful of modules; the result is
a tool for research and process studies.

## Why FRIDOM

**One compiled run.** No Python time loop: FRIDOM lowers the whole integration
into a single `jax.jit` over a chunked `lax.scan`, compiles it once, and runs it
on CPU, GPU, or TPU.

**No silent errors.**

- *Operators never read past their data.* Fields track how deep their halo is
  valid and operators track how far their stencil reaches, so FRIDOM exchanges
  the halo it needs (once, shared across readers) instead of reading stale or
  out-of-bounds cells. Bypassing this by dropping to raw arrays is rejected
  unless declared.
- *Inconsistent models don't build.* Assembly validates the model first and
  refuses duplicate fields, unsatisfied references, or an implicit term under an
  explicit stepper, naming the module at fault.
- *Bad numbers stop the run.* Non-finite values halt the integration at the next
  chunk boundary and report the first bad step.

**Composable modules.** No settings object; each parameter (Coriolis frequency,
stratification, wave speed) lives in the module that uses it. Swap a module to
change the physics.

**Function-space grid.** Fields live on function spaces; operators map between
them and compose with `@`, where `laplacian` is literally `div @ grad`.
Spectral, finite-difference, and finite-volume (WENO) schemes share one
interface.

**Runs anywhere, analysis-ready output.** Multi-device decomposition uses native
JAX sharding, no MPI. Initialise fields from `f(x, y)`, export any field to
xarray with `field.xr`, and write output as a zarr store through TensorStore.

## Example

The snippet below builds a rotating shallow-water model, puts a bump on the free
surface, integrates it, and plots the relative vorticity.

```python
import numpy as np

import fridom as fr
import fridom.shallowwater2 as sw

# A doubly-periodic square grid: one mesh per axis.
mx = fr.spatial.meshes.IntervalMesh(128, (0.0, 1.0), periodic=True, name="x")
my = fr.spatial.meshes.IntervalMesh(128, (0.0, 1.0), periodic=True, name="y")
grid = fr.spatial.Grid((mx, my))

# Assemble a shallow-water model from composable modules.
model = sw.Model(
    grid=grid,
    csqr=1.0,
    rossby_number=0.2,
    coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt=2e-3, order=3),
)

# Set the initial free-surface field p(x, y) from a function of the coordinates.
def bump(x, y):
    return 0.05 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.1 ** 2))

model.set_fields(p=bump)

# Integrate. The whole loop is compiled once into a single JAX function.
model.run(steps=200)

# Read the relative vorticity as an xarray DataArray and plot it.
model.state.rel_vort.xr.plot(cmap="RdBu_r")
```

> The core packages `fridom.spatial` (spatial discretization) and
> `fridom.model` (model machinery) carry their final names. The rewritten
> models still ship as `nonhydro2` and `shallowwater2` alongside the previous
> versions; they take over the primary names once the port is finished.

## Available models

- **nonhydro**: a 3D non-hydrostatic Boussinesq model, adapted from
  [ps3d](https://github.com/ceden/ps3d).
- **shallowwater**: a 2D rotating shallow-water model.

FRIDOM is under active
development and the models are being rebuilt on the new core, so the API still
moves between versions.

## Installation

FRIDOM needs Python 3.11 or newer. Clone the repository and install it in
editable mode:

```bash
git clone https://github.com/Gordi42/FRIDOM
cd FRIDOM
pip install -e .            # CPU
pip install -e '.[cuda]'    # NVIDIA GPU (CUDA 12)
```

The project is developed with [uv](https://docs.astral.sh/uv/), so `uv sync`
(or `uv sync --extra dev`) reproduces the pinned environment. The plotting in
the example above needs `xarray` and `matplotlib`, which come with the `dev`
extra. The current code lives on the development branch, and the `fridom`
release on PyPI predates this rewrite, so install from source for the version
described here. See the
[installation guide](https://fridom.readthedocs.io/en/latest/installation.html)
for more.

## Documentation

The full documentation, tutorials, and example gallery live at
[fridom.readthedocs.io](https://fridom.readthedocs.io/en/latest/index.html).

## Related work

FRIDOM builds on ideas from several fluid- and ocean-modeling projects:

- **[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)**, a mature
  Julia finite-volume ocean model spanning idealized to near-global setups, on
  CPU and GPU.
- **[Veros](https://github.com/team-ocean/veros)**, a pyOM2-derived
  primitive-equation ocean model in Python with a JAX backend. It is FRIDOM's
  closest JAX relative, though Veros is a fixed model where FRIDOM is a
  framework for assembling them.
- **[pyOM2](https://github.com/ceden/pyOM2)** and
  **[ps3d](https://github.com/ceden/ps3d)**, the Fortran ocean and
  pseudo-spectral flow solvers that FRIDOM's idealized experiments descend from.
- **[Dedalus](https://github.com/DedalusProject/dedalus)**, a general spectral
  PDE solver where you enter equations symbolically.
- **[Shenfun](https://github.com/spectralDNS/shenfun)**, a spectral-Galerkin
  framework that shares FRIDOM's view of fields on function spaces with
  operators between them.
- **[jax-cfd](https://github.com/google/jax-cfd)**, differentiable CFD in JAX,
  the same accelerator-first substrate applied to generic flows.

## Gallery

https://github.com/Gordi42/FRIDOM/assets/118457787/66cca07d-5893-4c1b-af13-901dc78bdd6b

## How to cite

```
@software{Rosenau_fridom_2024,
          author = {Rosenau, Silvano Gordian},
          doi = {10.5281/zenodo.14536979},
          month = dec,
          title = {{Fridom: A framework for idealized ocean models.}},
          url = {https://github.com/Gordi42/fridom},
          version = {0.0.1},
          year = {2024}
}
```

## Author

Silvano Rosenau

## License

[MIT](LICENSE)
