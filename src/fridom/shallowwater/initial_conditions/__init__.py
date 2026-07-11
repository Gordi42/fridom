"""Initial Conditions for the shallow water model."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from .coherent_eddy import CoherentEddy
    from .equatorial_wave import EquatorialWave
    from .geostrophic_spectra import (
        PrescribedSpectraRandomPhase,
        RandomGeostrophicSpectra,
        RandomWaveSpectra,
        ShallowWaterRandom,
        geostrophic_energy_spectrum,
        gm_energy_spectrum,
    )
    from .jet import Jet
    from .single_wave import SingleWave

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.shallowwater.initial_conditions"

all_modules_by_origin = { }

all_imports_by_origin = {
    f"{base_path}.coherent_eddy": ["CoherentEddy"],
    f"{base_path}.equatorial_wave": ["EquatorialWave"],
    f"{base_path}.geostrophic_spectra": [
        "PrescribedSpectraRandomPhase",
        "RandomGeostrophicSpectra",
        "RandomWaveSpectra",
        "ShallowWaterRandom",
        "geostrophic_energy_spectrum",
        "gm_energy_spectrum",
    ],
    f"{base_path}.jet": ["Jet"],
    f"{base_path}.single_wave": ["SingleWave"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
