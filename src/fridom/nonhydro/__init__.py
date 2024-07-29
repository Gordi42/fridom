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
`utils`
    Module that stores all utility functions for the framework.
`grid`
    Module that stores all available grids for the non-hydrostatic model.
`modules`
    Module that stores all available modules for the non-hydrostatic model.
`time_steppers`
    Time steppers for the non-hydrostatic model.

Classes
-------
`ModelSettings`
    The model settings class.
`FieldVariable`
    A field variable class.
`State`
    The state class.
`ModelState`
    The model state class.
`Model`
    The model class.
`Plot`
    The plot class.

Examples
--------
>>> # Provide examples of how to use this module.
>>> import fridom.nonhydro as nh
>>> mset = nh.ModelSettings()                  # create model settings
>>> mset.N = (2**7, 2**7, 2**4)                # set resolution
>>> mset.L = [4, 4, 1]                         # set domain size 
>>> grid = nh.Grid(mset)                       # create grid
>>> model = nh.Model(grid)                     # create model
>>> model.z = nh.initial_conditions.Jet(grid)  # set initial conditions
>>> model.run(runlen=2)                        # run model
>>> 
>>> # plot top view of final kinetic energy
>>> nh.Plot(model.z.ekin()).top(model.z)
"""
import sys
from types import ModuleType
import importlib
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # ----------------------------------------------------------------
    #  Importing model specific classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from . import grid
    from . import modules
    from . import projection
    from . import eigenvectors
    from . import initial_conditions

    # importing classes
    from .model_settings import ModelSettings
    from .state import State
    from .plot import Plot, PlotContainer

    # ----------------------------------------------------------------
    #  Importing generic classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from fridom.framework import config
    from fridom.framework import utils
    from fridom.framework import time_steppers

    # importing classes
    from fridom.framework.field_variable import FieldVariable
    from fridom.framework.model_state import ModelState
    from fridom.framework.model import Model
    

# ================================================================
#  Define all the possible imports
# ================================================================

# Set up dictionary that maps an import to a path
# items in the all_modules_by_origin dictionary are imported as modules
all_modules_by_origin = { 
    "fridom.nonhydro": ["grid", 
                        "modules", 
                        "projection", 
                        "eigenvectors",
                        "initial_conditions"],
    "fridom.framework": ["config", "time_steppers", "utils"],
}

# items in the all_imports_by_origin dictionary are imported as elements of a module
all_imports_by_origin = { 
    "fridom.nonhydro.model_settings": ["ModelSettings"],
    "fridom.nonhydro.state": ["State"],
    "fridom.nonhydro.plot": ["Plot", "PlotContainer"],
    "fridom.framework.field_variable": ["FieldVariable"],
    "fridom.framework.model_state": ["ModelState"],
    "fridom.framework.model": ["Model"],
}

# ================================================================
#  Set up the import system
# ================================================================

origins = {}
_all_modules = []
for origin, items in all_modules_by_origin.items():
    for item in items:
        _all_modules.append(item)
        origins[item] = origin

_all_imports = []
for origin, items in all_imports_by_origin.items():
    for item in items:
        _all_imports.append(item)
        origins[item] = origin

# load submodules on demand
class _module(ModuleType):
    def __getattr__(self, name):
        # check if the attribute is a module
        if name in _all_modules:
            res = importlib.import_module(origins[name] + "." + name)
        # check if the attribute is an import
        elif name in _all_imports:
            mod = importlib.import_module(origins[name])
            res = getattr(mod, name)
        # if the attribute is not found
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        # set the attribute in the current module such that it is not loaded again
        setattr(self, name, res)
        # return the attribute
        return res

sys.modules[__name__].__class__ = _module
__all__ = _all_modules + _all_imports



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


# ================================================================
#  TODO: Update the following imports
# ================================================================
# :: updated but not tested ::
# modules/linear_tendencies.py
# modules/pressure_gradient_tendency.py
# modules/tendency_divergence.py
# modules/pressure_solvers/spectral_pressure_solver.py

# start with:
# modules/main_tendency.py

# initial_conditions/barotropic_jet.py
# initial_conditions/jet.py
# initial_conditions/single_wave.py
# initial_conditions/vertical_mode.py
# initial_conditions/wave_package.py
# initial_conditions/__init__.py
# modules/advection/centered_advection.py
# modules/advection/second_order_advection.py
# modules/advection/__init__.py
# modules/diffusion/biharmonic_friction.py
# modules/diffusion/biharmonic_mixing.py
# modules/diffusion/harmonic_friction.py
# modules/diffusion/harmonic_mixing.py
# modules/diffusion/__init__.py
# modules/forcings/gaussian_wave_maker.py
# modules/forcings/polarized_wave_maker.py
# modules/forcings/__init__.py
# modules/interpolation/interpolation_module.py
# modules/interpolation/linear_interpolation.py
# modules/interpolation/polynomial_interpolation.py
# modules/interpolation/__init__.py
# modules/pressure_solvers/cg_pressure_solver.py
# modules/pressure_solvers/__init__.py
# modules/diagnostics.py
# modules/netcdf_writer.py
# modules/__init__.py
# projection/divergence_spectral.py
# projection/geostrophic_spectral.py
# projection/nnmd.py
# projection/wave_spectral.py
# projection/__init__.py
# diagnose_imbalance.py
# diagnostic_state.py
# eigenvectors.py
# plot.py