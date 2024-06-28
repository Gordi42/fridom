"""
Nonhydrostatic Model
===
A 3D non-hydrostatic Boussinesq model. Based on the ps3D model by
Prof. Carsten Eden ( https://github.com/ceden/ps3D ).

Description
-----------
TODO

Modules
-------
`config`
    Module that stores all configuration options for the framework.
`grid`
    Module that stores all available grids for the non-hydrostatic model.
`time_steppers`
    Time steppers for the non-hydrostatic model.

Classes
-------
`ModelSettings`
    The model settings class.
`FieldVariable`
    A field variable class.
`ModelState`
    The model state class.
`Model`
    The model class.

Examples
--------
>>> # Provide examples of how to use this module.
>>> import fridom.nonhydro as nh
>>> mset = nh.ModelSettings()                  # create model settings
>>> mset.N = [2**7, 2**7, 2**4]                # set resolution
>>> mset.L = [4, 4, 1]                         # set domain size 
>>> grid = nh.Grid(mset)                       # create grid
>>> model = nh.Model(grid)                     # create model
>>> model.z = nh.initial_conditions.Jet(grid)  # set initial conditions
>>> model.run(runlen=2)                        # run model
>>> 
>>> # plot top view of final kinetic energy
>>> nh.Plot(model.z.ekin()).top(model.z)
"""
# ----------------------------------------------------------------
#  Importing model specific classes and modules
# ----------------------------------------------------------------
# importing modules
from . import grid

# importing classes
from .model_settings import ModelSettings

# ----------------------------------------------------------------
#  Importing generic classes and modules
# ----------------------------------------------------------------
# importing modules
from fridom.framework import config
from fridom.framework import time_steppers

# importing classes
from fridom.framework.field_variable import FieldVariable
from fridom.framework.model_state import ModelState
from fridom.framework.model import Model



# # import all the classes and functions from the nonhydro module
# from .model_settings import ModelSettings
# from .grid import Grid
# from .state import State
# from .plot import Plot, PlotContainer
# from .model_plotter import ModelPlotter
# from .diagnose_imbalance import DiagnoseImbalance

# # move the classes from the framework into the nonhydro namespace
# from fridom.framework.field_variable import FieldVariable
# from fridom.framework.model_state import ModelState
# from fridom.framework.model import Model

# # importing modules
# from . import modules
# from . import projection
# from . import initial_conditions
# from . import eigenvectors
# from fridom.framework import config

# # move the time steppers into the nonhydro namespace
# from fridom.framework import time_steppers