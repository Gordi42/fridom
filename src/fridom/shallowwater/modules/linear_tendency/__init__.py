"""
# Linear Tendency Module

This module computes the linear tendency of the model. The linear tendency is 
the sum of the Coriolis force, the pressure gradient force, and the horizontal 
divergence.

## Modules:
- LinearTendency: Using finite differences.
- LinearTendencySpectral: Using spectral methods.
"""

from .linear_tendency import LinearTendency
from .linear_tendency_spectral import LinearTendencySpectral