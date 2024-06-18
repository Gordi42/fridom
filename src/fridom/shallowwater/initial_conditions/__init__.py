"""
# Initial Conditions for the Nonhydrostatic Model

## Available Initial Conditions:
    - SingleWave: A single polarized sine wave in the whole domain
    - WavePackage: A polarized wave package with a Gaussian profile 
    - Random: A random field with a given energy spectrum
    - WaveSpectra: A random field with a GM-like wave spectrum
    - GeostrophicSpectra: A random field with a geostrophic spectrum
    - RandomPhase: WaveSpectra + GeostrophicSpectra
    - Jet: An unstable jet
"""
# Waves
from .single_wave import SingleWave
from .wave_package import WavePackage

# Random initial conditions
from .random import Random
from .wave_spectra import WaveSpectra
from .geostrophic_spectra import GeostrophicSpectra
from .random_phase import RandomPhase

# Jet
from .jet import Jet