"""
Initial Conditions for the Nonhydrostatic Model
===============================================

Classes
-------
`SingleWave`
    A single polarized sine wave in the whole domain
`WavePackage`
    A polarized wave package with a Gaussian profile
`VerticalMode`
    A polarized vertical mode over the whole domain
`BarotropicJet`
    A barotropic unstable jet with horizontal shear
`Jet`
    An unstable jet with horizontal and vertical shear
"""

# import classes directly to avoid having to type single_wave.SingleWave()
from .single_wave import SingleWave
from .wave_package import WavePackage
from .vertical_mode import VerticalMode
from .barotropic_jet import BarotropicJet
from .jet import Jet