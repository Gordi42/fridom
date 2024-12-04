r"""
A 3D non-hydrostatic Boussinesq model.

Description
-----------
The model is based on the ps3D model by Prof. Carsten Eden 
( https://github.com/ceden/ps3D ).

System of Equations
-------------------
The model solves the scaled non-hydrostatic Boussinesq equations given by 
the momentum equations:

.. math::
    \partial_t u + Ro~\, \boldsymbol{u} \cdot \nabla u =
        f v - \partial_x p + \boldsymbol{F}_u

.. math::
    \partial_t v + Ro~\, \boldsymbol{u} \cdot \nabla v =
        -f u - \partial_y p + \boldsymbol{F}_v

.. math::
    \partial_t w + Ro~\, \boldsymbol{u} \cdot \nabla w =
        \delta^{-2} b - \delta^{-2} \partial_z p + \boldsymbol{F}_w

The buoyancy equation:

.. math::
    \partial_t b + Ro~, \boldsymbol{u} \cdot \nabla b = -w N^2 + \boldsymbol{F}_b

And the continuity equation:

.. math::
    \nabla \cdot \boldsymbol{u} = 0

with:
    - :math:`\boldsymbol{u} = (u, v, w)` the velocity vector,
    - :math:`p` the model pressure,
    - :math:`b` the model buoyancy,
    - :math:`f` the Coriolis parameter,
    - :math:`N` the buoyancy frequency,
    - :math:`\boldsymbol{F}_i` are sources and sinks,
    - :math:`Ro` the Rossby number (1 for unscaled equations),
    - :math:`\delta` the aspect ratio (1 for unscaled equations),

Meaning of Pressure and Buoyancy
--------------------------------
**TL;DR:**

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

Let :math:`\rho` be the real density (the one that we would measure in a
real fluid), and :math:`\pi` be the real pressure. In this section, we
link the model pressure :math:`p`, the model buoyancy :math:`b`, and the
buoyancy frequency :math:`N` to the real pressure and density. For that,
we start with the full Boussinesq equations (neglect scaling and source terms):

.. math::
    \rho_0 (D_t u - fv) = - \partial_x \pi
    ~ , \quad
    \rho_0 (D_t v + fu) = - \partial_y \pi
    ~ , \quad
    \rho_0 D_t w = - \partial_z \pi - \rho g

where :math:`D_t = \partial_t + \boldsymbol{u} \cdot \nabla` is the material
derivative, :math:`\rho_0` is the constant Boussinesq density. We further 
decompose the density into a part that only depends on the vertical coordinate
:math:`\rho_s(z)`, and a perturbation :math:`\rho'`:

.. math::
    \rho(\boldsymbol{x}, t) = \rho_0 + \rho_s(z) + \rho'(\boldsymbol{x}, t)

We now solve the vertical momentum equation for :math:`w=0`, and :math:`\rho' = 0`:

.. math::
    \partial_z \pi_s = - g \rho_0 - g \rho_s(z)

where :math:`\pi_s` is the hydrostatic background pressure. Defining the
pressure :math:`p`, and buoyancy :math:`b` as:

.. math::
    p = \frac{\pi - \pi_s}{\rho_0}
    ~ , \quad
    b = -\frac{\rho' g}{\rho_0}

the Boussinesq equations become:

.. math::
    D_t u - fv = - \partial_x p
    ~ , \quad
    D_t v + fu = - \partial_y p
    ~ , \quad
    D_t w = - \partial_z p + b
    

To obtain a tendency equation for the buoyancy, we compute its differential:

.. math::
    db = - \frac{g}{\rho_0} d\rho' = - \frac{g}{\rho_0} d\rho + \frac{g}{\rho_0} d\rho_s
       = - \frac{g}{\rho_0} d\rho - N^2 dz

where the buoyancy frequency :math:`N` is defined as:

.. math::
    N^2 = - \frac{g}{\rho_0} \partial_z \rho_s = \partial_z b_s

Divide the differential by :math:`dt` to obtain the tendency equation:

.. math::
    D_t b = - \frac{g}{\rho_0} D_t \rho - N^2 w

By further assuming that the density is conserved, we obtain the buoyancy equation:

.. math::
    D_t b = - w N^2


Scaling
-------
The system is nondimensionalized by introducing the following nondimensional
coordinates:

.. math::
    x' = \frac{x}{L}
    ~ , \quad
    y' = \frac{y}{L}
    ~ , \quad
    z' = \frac{z}{H}
    ~ , \quad
    t' = \Omega t

where :math:`L` is the horizontal length scale, :math:`H` is the vertical length
scale, and :math:`\Omega` is the rotation rate. The nondimensional variables are
alwqays denoted with the prime. The nondimensional model parameters are:

.. math::
    f' = \frac{f}{\Omega}
    ~ , \quad
    N' = \frac{N}{N^*}

where :math:`N^*` is the order of magnitude of the buoyancy frequency.
The nondimensional model variables are:

.. math::
    u' = \frac{u}{U}
    ~ , \quad
    v' = \frac{v}{U}
    ~ , \quad
    w' = \frac{w}{W}
    ~ , \quad
    p' = \frac{p}{P}
    ~ , \quad
    b' = \frac{b}{B}

where :math:`U` is the order of magnitude of the horizontal velocity, :math:`W`
is the order of magnitude of the vertical velocity, :math:`P` is the order of
magnitude of the pressure, and :math:`B` is the order of magnitude of the
buoyancy. We define the scaling parameters as follows:

.. math::
    Ro = \frac{U}{\Omega L}
    ~ , \quad
    \delta = \frac{H}{L}
    ~ , \quad
    L_r = \frac{N^* H}{\Omega}

where :math:`Ro` is the Rossby number, :math:`\delta` is the aspect ratio, and
:math:`L_r` is the Rossby deformation radius. Inserting the nondimensional
variables into the continuity equation, we obtain the following scaling relation
for the aspect ratio:

.. math::
    \delta = \frac{H}{L} = \frac{W}{U}

We assume a small Rossby number and that the momentum equations are in balance (e.g. the time derivatives 
vanish). This leads to the following scaling relation:

.. math::
    B = \Omega U = \frac{P}{L}

We further assume that the length scale is of the same order of magnitude as the
Rossby deformation radius. Or in other words, a Froude number of order one. From
this assumption, we can relate the buoyancy frequency to the rotation rate and
the aspect ratio:

.. math::
    L \approx L_r \Rightarrow N^* = \frac{\Omega}{\delta}

Inserting everything into the full Boussinesq equations, we obtain the scaled
equations that are given at the beginning.


Time Stepping
-------------
We recall that the full system of equations is given by (neglecting scaling):

.. math::
    \partial_t \boldsymbol{u} = \Delta_{\boldsymbol{u}} - \nabla p
    ~ , \quad
    \partial_t b = \Delta_b
    ~ , \quad
    \nabla \cdot \boldsymbol{u} = 0

where :math:`\Delta_{\boldsymbol{u}}` includes all tendency terms on the right
hand side of the momentum equations except the pressure gradient term. And 
the same for :math:`\Delta_b`. Taking the divergence of the momentum equations,
we obtain the Poisson equation for the pressure:

.. math::
    \nabla^2 p = \nabla \cdot \Delta_{\boldsymbol{u}}

This means that the pressure is implicitly given by the velocity and buoyancy.
Hence, the full system of equations can be written as:

.. math::
    \partial_t \boldsymbol{z} = \boldsymbol{f}(\boldsymbol{z}, t)

where :math:`\boldsymbol{z} = (\boldsymbol{u}, b)` is the state vector, and 
:math:`\boldsymbol{f}` is the right hand side of the equations. For the explicit
time stepping schemes, we must evaluate the right hand side of the equations at
a given time level. This is done by first computing the tendency terms
:math:`\Delta_{\boldsymbol{u}}` and :math:`\Delta_b`, and then solving the
Poisson equation for the pressure. And finally, we remove the pressure gradient
term from the tendency terms to obtain the right hand side of the equations.
For more details on the pressure solver, see 
:py:mod:`fridom.nonhydro.modules.pressure_solvers`.
"""
from typing import TYPE_CHECKING
from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # ----------------------------------------------------------------
    #  Importing model specific classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from . import grid
    from . import modules
    from . import initial_conditions

    # importing classes
    from .model_settings import ModelSettings
    from .state import State, DiagnosticState

    # ----------------------------------------------------------------
    #  Importing generic classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from fridom.framework import utils
    from fridom.framework import time_steppers
    from fridom.framework import projection

    # import logger and config
    from fridom.framework.configuration import config
    from fridom.framework.logger import log

    # importing classes
    from fridom.framework.field_variable import FieldVariable
    from fridom.framework.model_state import ModelState
    from fridom.framework.model import Model

# ================================================================
#  Setup lazy loading
# ================================================================
all_modules_by_origin = {
    "fridom.nonhydro": ["grid", "modules", "initial_conditions"],
    "fridom.framework": ["time_steppers", "utils", "projection"],
}

all_imports_by_origin = {
    "fridom.framework.configuration": ["config"],
    "fridom.framework.logger": ["log"],
    "fridom.nonhydro.model_settings": ["ModelSettings"],
    "fridom.nonhydro.state": ["State", "DiagnosticState"],
    "fridom.framework.field_variable": ["FieldVariable"],
    "fridom.framework.model_state": ["ModelState"],
    "fridom.framework.model": ["Model"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
