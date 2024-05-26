"""
# Diffusion module
This module contains the diffusion modules (friction and mixing) of the model.

## Available modules:
- HarmonicTendency: Harmonic friction tendency module.
- BiharmonicTendency: Biharmonic friction tendency module.
"""

from .harmonic_tendency import HarmonicTendency
from .biharmonic_tendency import BiharmonicTendency
