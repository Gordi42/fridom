r"""
A 3D hydrostatic Boussinesq model.

Description
-----------
The model is based on pyOM2 by Prof. Carsten Eden
( https://github.com/ceden/pyOM2 ).

System of Equations
-------------------
The model solves the scaled hydrostatic Boussinesq equations given by
the momentum equations:

.. math::
    \partial_t u + Ro~\, \boldsymbol{u} \cdot \nabla u =
        f v - \partial_x p + \boldsymbol{F}_u

.. math::
    \partial_t v + Ro~\, \boldsymbol{u} \cdot \nabla v =
        -f u - \partial_y p + \boldsymbol{F}_v

The buoyancy equation:

.. math::
    \partial_t b + Ro~, \boldsymbol{u} \cdot \nabla b = -w N^2 + \boldsymbol{F}_b

The vertical velocity is given by the continuity equation:

.. math::
    \partial_z w = -\left( \partial_x u + \partial_y v \right)

The pressure is given by the hydrostatic balance:

.. math::
    \partial_z p = b

with:
    - :math:`\boldsymbol{u} = (u, v, w)` the velocity vector,
    - :math:`p` the model pressure,
    - :math:`b` the model buoyancy,
    - :math:`f` the Coriolis parameter,
    - :math:`N` the buoyancy frequency,
    - :math:`\boldsymbol{F}_i` are sources and sinks,
    - :math:`Ro` the Rossby number (1 for unscaled equations),
    - :math:`\delta` the aspect ratio (1 for unscaled equations),

Derivation
----------
The model equations can be directly derived from the non-hydrostatic equations
that are detailed in :py:mod:`fridom.nonhydro`. The only difference is in the
vertical momentum equation. Which is originally given by:

.. math::
    \partial_t w + Ro~\, \boldsymbol{u} \cdot \nabla w =
        \delta^{-2} b - \delta^{-2} \partial_z p + \boldsymbol{F}_w

by assuming a small aspect ratio (e.g. :math:`\delta = H/L \ll 1`). The vertical
momentum equation is then replaced by the hydrostatic balance equation:

.. math::
    \partial_z p = b

Meaning of Pressure and Buoyancy
--------------------------------
The pressure :math:`p`, buoyancy :math:`b`, and buoyancy frequency :math:`N`
are related to the real pressure and density by the following equations:

.. math::
    p = \frac{\pi - \pi_s}{\rho_0}
    ~ , \quad
    b = -\frac{g}{\rho_0} \rho'
    ~ , \quad
    N^2 = -\frac{g}{\rho_0} \partial_z \rho_s = \partial_z b_s

with:
    - :math:`\pi` the real pressure,
    - :math:`\pi_s` the hydrostatic background pressure,
    - :math:`\rho'` the density perturbation,
    - :math:`\rho_0` the constant Boussinesq density,
    - :math:`\rho_s` the background density profile.

For more details on the derivation see :py:mod:`fridom.nonhydro`.

"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework import projection, time_steppers, utils
    from fridom.framework.clock import Clock, TimingFormat
    from fridom.framework.clock_trigger import ClockTrigger
    from fridom.framework.configuration import config
    from fridom.framework.field_metadata import FieldMetadata
    from fridom.framework.logger import log
    from fridom.framework.model import Model
    from fridom.framework.model_state import ModelState
    from fridom.framework.scalar_field import ScalarField
    from fridom.framework.tensor_field import TensorField
    from fridom.framework.vector_field import VectorField
    from fridom.hydrostatic import grid

    from .model_settings import ModelSettings
    from .state import DiagnosticState, State

# ================================================================
#  Setup lazy loading
# ================================================================
all_modules_by_origin = {
    "fridom.framework": ["time_steppers", "utils", "projection"],
    "fridom.hydrostatic": ["grid"],
}

hs_base = "fridom.hydrostatic"

all_imports_by_origin = {
    "fridom.framework.configuration": ["config"],
    "fridom.framework.logger": ["log"],
    "fridom.framework.field_metadata": ["FieldMetadata"],
    "fridom.framework.scalar_field": ["ScalarField"],
    "fridom.framework.vector_field": ["VectorField"],
    "fridom.framework.tensor_field": ["TensorField"],
    "fridom.framework.model_state": ["ModelState"],
    "fridom.framework.model": ["Model"],
    "fridom.framework.clock": ["Clock", "TimingFormat"],
    "fridom.framework.clock_trigger": ["ClockTrigger"],
    f"{hs_base}.model_settings": ["ModelSettings"],
    f"{hs_base}.state": ["State", "DiagnosticState"],

}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
