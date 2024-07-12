"""
Modules
===
Base classes and functions for creating modules in the model.

Modules
-------
`animation`
    Contains modules for creating animated output of the model.

Classes
-------
`Module`
    Base class for all modules.
`ModuleContainer`
    Base class for a container of modules.
`BoundaryConditions`
    Module for setting boundary conditions to the model state.
`NetCDFWriterBase`
    Base Module for writing the model output to a NetCDF file.
`NetCDFWriterSingleProc`
    Module for writing the model output to a NetCDF file on a single processor.

Functions
---------
`update_module`
    Decorator for the update method of the module.
`start_module`
    Decorator for the start method of the module.
`stop_module`
    Decorator for the stop method of the module.
"""
# importing modules
from . import animation

# importing the Classes
from .module import Module
from .module_container import ModuleContainer
from .boundary_conditions import BoundaryConditions
from .netcdf_writer_single_proc import NetCDFWriterSingleProc
from .netcdf_writer import NetCDFWriter

# importing the functions
from .module import setup_module, module_method

