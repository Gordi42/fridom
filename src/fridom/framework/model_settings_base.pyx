# cython: language_level=3

# Import external modules
import numpy as np
from fridom.framework.grid.grid_base cimport GridBase

cdef class ModelSettingsBase:
    def __init__(self, grid: GridBase | None, **kwargs) -> None:
        # grid
        self.grid = grid

        # model name
        self.model_name = "Unnamed model"

        # time parameters
        from fridom.framework.time_steppers.adam_bashforth import AdamBashforth
        self.time_stepper = AdamBashforth()

        # modules
        from fridom.framework.modules.module_container import ModuleContainer
        from fridom.framework.modules.boundary_conditions import BoundaryConditions
        # List of modules that calculate tendencies
        self.tendencies = ModuleContainer(name="All Tendency Modules")
        # List of modules that do diagnostics
        self.diagnostics = ModuleContainer(name="All Diagnostic Modules")
        # Boundary conditions  (should be set by the child class)
        self.bc = BoundaryConditions(field_names=[])

        # Output
        self.enable_verbose = False   # Enable verbose output
        
        # Starttime
        self.start_time = np.datetime64(0, 's')

        # State Constructors
        from fridom.framework.state_base import StateBase
        self.state_constructor = lambda: StateBase(self, [])
        self.diagnostic_state_constructor = lambda: StateBase(self, [])

        # Set attributes from keyword arguments
        self.set_attributes(**kwargs)

    def set_attributes(self, **kwargs):
        # Set attributes from keyword arguments
        for key, value in kwargs.items():
            # Check if attribute exists
            if not hasattr(self, key):
                raise AttributeError(
                    "ModelSettings has no attribute '{}'".format(key)
                    )
            setattr(self, key, value)
        return

    cpdef void setup(self):
        self.grid.setup(self)
        return

    def print_verbose(self, *args, **kwargs):
        if self.enable_verbose:
            print(*args, **kwargs)
        return

    def __repr__(self) -> str:
        res = "================================================\n"
        res += "  Model Settings:\n"
        res += "================================================\n"
        res += "  Model name: {}\n".format(self.model_name)
        res += f"{self.time_stepper}"
        res += f"{self.tendencies}"
        res += f"{self.diagnostics}"
        res += "------------------------------------------------\n"
        return res

    def _to_numpy(self, memo):
        from copy import deepcopy
        from fridom.framework.to_numpy import to_numpy
        mset = deepcopy(self)
        for attr in dir(mset):
            if attr.startswith("__"):
                continue
            if callable(getattr(mset, attr)):
                continue
            setattr(mset, attr, to_numpy(getattr(mset, attr), memo))
        return mset

    # ================================================================
    #  Properties
    # ================================================================

    property model_name:
        def __get__(self):
            return self.model_name
        def __set__(self, str value):
            self.model_name = value
    
    property grid:
        def __get__(self):
            return self.grid
        def __set__(self, GridBase value):
            self.grid = value
    
    property time_stepper:
        def __get__(self):
            return self.time_stepper
        def __set__(self, object value):
            self.time_stepper = value
    
    property tendencies:
        def __get__(self):
            return self.tendencies
        def __set__(self, object value):
            self.tendencies = value
    
    property diagnostics:
        def __get__(self):
            return self.diagnostics
        def __set__(self, object value):
            self.diagnostics = value
    
    property bc:
        def __get__(self):
            return self.bc
        def __set__(self, object value):
            self.bc = value
    
    property state_constructor:
        def __get__(self):
            return self.state_constructor
        def __set__(self, object value):
            self.state_constructor = value
    
    property diagnostic_state_constructor:
        def __get__(self):
            return self.diagnostic_state_constructor
        def __set__(self, object value):
            self.diagnostic_state_constructor = value
    
    property start_time:
        def __get__(self):
            return self.start_time
        def __set__(self, object value):
            self.start_time = value
    
    property enable_verbose:
        def __get__(self):
            return self.enable_verbose
        def __set__(self, bint value):
            self.enable_verbose = value
    
