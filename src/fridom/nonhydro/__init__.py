"""
# Nonhydrostatic model
A 3D non-hydrostatic Boussinesq model. Based on the ps3D model by
Prof. Carsten Eden ( https://github.com/ceden/ps3D ).

## Classes:
    ModelSettings     : used to construct the model grid
    Grid              : used to construct model, initial cond., etc.
    FieldVariable     : used to store scalar variables
    State             : stores u,v,w,b
    ModelState        : contains the model state of the nonhydrostatic model
    Model             : the model itself
    Plot              : Plotting class
    PlotContainer     : Container for plotting routines
    ModelPlotter      : For animating the model
    DiagnoseImbalance : diagnostic tool for measuring imbalances

## Modules:
    projection        : for flow decomposition
    initial_conditions: contains initial conditions
    source            : contains source terms to force the model
    eigenvectors      : eigenvectors module
    time_steppers     : time steppers

## Example:
```python
import fridom.nonhydro as nh
mset = nh.ModelSettings()                  # create model settings
mset.N = [2**7, 2**7, 2**4]                # set resolution
mset.L = [4, 4, 1]                         # set domain size 
grid = nh.Grid(mset)                       # create grid
model = nh.Model(grid)                     # create model
model.z = nh.initial_conditions.Jet(grid)  # set initial conditions
model.run(runlen=2)                        # run model

# plot top view of final kinetic energy
nh.Plot(model.z.ekin()).top(model.z)
```
"""

# import all the classes and functions from the nonhydro module
from .model_settings import ModelSettings
from .grid import Grid
from .state import State
from .plot import Plot, PlotContainer
from .model_plotter import ModelPlotter
from .diagnose_imbalance import DiagnoseImbalance

# move the classes from the framework into the nonhydro namespace
from fridom.framework.field_variable import FieldVariable
from fridom.framework.model_state import ModelState
from fridom.framework.model import Model

# importing modules
from . import modules
from . import projection
from . import initial_conditions
from . import eigenvectors
from fridom.framework import config

# move the time steppers into the nonhydro namespace
from fridom.framework import time_steppers