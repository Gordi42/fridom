"""
# Forcings

This module contains classes that generate forcing terms for the shallow water model.

## Classes:
    - GaussianWaveMaker: forces the u-component of the velocity field.
    - PolarizedWaveMaker: A polarized wave maker that creates a wave package.
"""

from .gaussian_wave_maker import GaussianWaveMaker
from .polarized_wave_maker import PolarizedWaveMaker