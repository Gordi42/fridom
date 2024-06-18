"""
# Diffusion module
This module contains the diffusion modules (friction and mixing) of the model.

## Available modules:
- HarmonicFriction: Harmonic friction module.
- HarmonicMixing: Harmonic mixing module.
- BiharmonicFriction: Biharmonic friction module.
- BiharmonicMixing: Biharmonic mixing module.
"""

from .harmonic_friction import HarmonicFriction
from .harmonic_mixing import HarmonicMixing
from .biharmonic_friction import BiharmonicFriction
from .biharmonic_mixing import BiharmonicMixing
