"""
# Forcings

This module contains classes that generate forcing terms for the shallow water model.

## Classes:
    - GaussianWaveMaker: forces the u-component of the velocity field.
    - PolarizedWaveMaker: A polarized wave maker that creates a wave package.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gaussian_wave_maker import GaussianWaveMaker
    from .polarized_wave_maker import PolarizedWaveMaker