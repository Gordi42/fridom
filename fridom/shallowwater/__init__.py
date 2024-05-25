"""
# Single Layer Shallow Water Model
A scaled rotating shallow water model. The discretization is based on a
energy conserving finite difference scheme on a staggered Arakawa C-grid.
Based on Sadourny [1975].

## Classes:
    ModelSettings     : used to construct the model grid
    Grid              : used to construct model, initial cond., etc.
    FieldVariable     : used to store scalar variables
    State             : stores u,v,h
    ModelState        : contains the model state of the shallow water model
    Model             : the model itself
    Plot              : Plotting class
    PlotContainer     : Container for plotting routines
    ModelPlotter      : For animating the model
    DiagnoseImbalance : diagnostic tool for measuring imbalances

## Modules:
    initial_conditions: contains initial conditions
    projection        : for flow decomposition
    source            : contains source terms to force the model
    eigenvectors      : eigenvectors module

## Example:
```python
import fridom.shallowwater as sw
mset = sw.ModelSettings()                  # create model settings
mset.N = [2**7, 2**7]                      # set resolution
mset.L = [4, 4]                            # set domain size
grid = sw.Grid(mset)                       # create grid
model = sw.Model(grid)                     # create model
model.z = sw.initial_conditions.Jet(grid)  # set initial conditions
model.run(runlen=2)                        # run model

# plot top view of final kinetic energy
sw.Plot(model.z.ekin())(model.z)
```
"""

# import all the classes and functions from the nonhydro module
from .model_settings import ModelSettings
from .grid import Grid
from .state import State
from .model_state import ModelState
from .model import Model
from .plot import Plot, PlotContainer
from .model_plotter import ModelPlotter
from .diagnosed_imbalance import DiagnoseImbalance

# move the field variable into the nonhydro namespace
from fridom.framework.field_variable import FieldVariable

# import modules
from . import modules
from . import initial_conditions
from . import projection
from . import source
from . import eigenvectors