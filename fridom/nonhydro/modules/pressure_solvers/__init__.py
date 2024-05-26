"""
# Pressure solvers
This module contains the pressure solvers for the non-hydrostatic model.

Classes:
    - SpectralPressureSolver: Spectral pressure solver.
    - CGPressureSolver: Conjugate gradient pressure solver.
"""

from .spectral_pressure_solver import SpectralPressureSolver
from .cg_pressure_solver import CGPressureSolver