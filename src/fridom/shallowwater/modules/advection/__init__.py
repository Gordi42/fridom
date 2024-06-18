"""
# Advection module
This module contains the advection modules for the shallow water model.

## Available Modules:
    - SadournyAdvection: Using finite differences. (Standard advection scheme)
    - SpectralAdvection: Using spectral methods.
"""

from .sadourny_advection import SadournyAdvection
from .spectral_advection import SpectralAdvection