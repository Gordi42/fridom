# cython: language_level=3

# Import external modules
from fridom.framework.grid.grid_base cimport GridBase

cdef class ModelSettingsBase:
    cdef str model_name
    cdef GridBase grid
    cdef object time_stepper
    cdef object tendencies
    cdef object diagnostics
    cdef object bc
    cdef object state_constructor
    cdef object diagnostic_state_constructor
    cdef object start_time
    cdef bint enable_verbose

    cpdef void setup(self)