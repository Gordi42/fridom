"""
# Modules
Base classes for creating modules in the model

## Base Class for Modules
- `Module`:        Base class for all modules
- `update_module`: Decorator for the update method of the module
- `start_module`:  Decorator for the start method of the module
- `stop_module`:   Decorator for the stop method of the module
- `ModuleContainer`: Base class for a container of modules

## Modules
- `NetCDFWriter`:  Module for writing the model output to a NetCDF file
- `animation`:     Contains modules for creating animated output of the model
"""
# importing the base class and functions
from .module import Module, update_module, start_module, stop_module
from .module_container import ModuleContainer

# importing the Classes
from .netcdf_writer import NetCDFWriter

# importing modules
from . import animation