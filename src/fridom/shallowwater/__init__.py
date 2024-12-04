r"""
A 2D scaled rotating shallow water model

System of equations
-------------------

.. math::

    \partial_t \boldsymbol{u} + Ro~\, \boldsymbol{u} \cdot \nabla \boldsymbol{u}
        = - f \underset{\neg}{\boldsymbol u} 
          - \nabla p + \boldsymbol{F}_\boldsymbol{u}

.. math::

    \partial_t p + Ro~\, \nabla \cdot \left( p \boldsymbol{u} \right)
        = - c^2 \nabla \cdot \boldsymbol{u} + \boldsymbol{F}_h

with:

- :math:`\boldsymbol{u} = (u,v)` the horizontal velocity vector, the hook
  operator rotates the vector by 90 degrees: :math:`\underset{\neg}{\boldsymbol u} = (-v,u)`
- :math:`p=g \eta` is the pressure perturbation. (:math:`\eta` is the 
  free surface displacement, :math:`g` is the acceleration due to gravity)
- :math:`\boldsymbol{F}` are source and sink terms
- :math:`f` is the Coriolis parameter
- :math:`c^2` is the phase speed of the gravity waves (:math:`c^2 = g H`) in 
  the unscaled system (with :math:`H` the depth of the fluid). In the scaled
  system :math:`c^2` corresponds to the Burger number
- :math:`Ro = \frac{U}{f L}` is the Rossby number (only for the scaled system,
  in unscaled system it is :math:`Ro=1`)

Derivation
----------
Start with the hydrostatic Navier Stokes equations with constant density and 
neglect the source terms:

.. math::

    D_t \boldsymbol{u} = - f \underset{\neg}{\boldsymbol u} 
                         - \frac{1}{\rho_0} \nabla p'
    ~, \quad
    \partial_z p' = - \rho_0 g
    ~, \quad
    \nabla \cdot \boldsymbol{u} + \partial_z w = 0

where :math:`D_t = \partial_t + \boldsymbol{u} \cdot \nabla` is the material
derivative and :math:`p'` is the pressure.
Let the free surface displacement be given by :math:`\eta(x,y,t)`. By integrating
the hydrostatic equation from some depth level :math:`z` to the surface, we 
obtain the pressure:

.. math:: 

    p'(\boldsymbol{x},z) 
        = p'(\boldsymbol{x}, \eta) + \int_{\eta}^z \partial_{z'} p' dz'
        = p'(\boldsymbol{x}, \eta) + \int_z^{\eta} \rho_0 g dz' 
        = p'(\boldsymbol{x}, \eta) + \rho_0 g (\eta + z)

Define new pressure variable :math:`p = g \eta` and assume constant atmospheric
pressure, i.e. :math:`\nabla p'(\boldsymbol{x}, \eta) = 0`. Then the horizontal 
pressure gradient is:

.. math:: 

    \frac{1}{\rho_0} \nabla p' = \nabla p

Inserting the new pressure variable into the momentum equation yields the
momentum equation in the form given above (with :math:`Ro=1`). To derive an 
equation for the new pressure variable, we first need to derive an equation
for the free surface displacement. For that, we need the differential of the
free surface displacement:

.. math::

    d\eta = \nabla \eta \cdot d\boldsymbol{x} + \partial_t \eta dt

The total time derivative of the surface displacement should be equal to the
vertical velocity at the surface, while the vertical velocity at the bottom
(:math:`z=-H`) should be zero:

.. math::

    w|_{z=\eta} = \frac{d\eta}{dt} = (\nabla \eta) \cdot \boldsymbol{u} +
                                      \partial_t \eta
                = D_t \eta
    ~, \quad
    w|_{z=-H} = 0

Now we relate the vertical velocity at the top to the horziontal velocity
field. For this, we integrate the continuity
equation from the bottom to the surface and assume barotropic conditions
(i.e. :math:`\partial_z \boldsymbol{u} = 0`):

.. math::

    0 = \int_{-H}^{\eta} \nabla \cdot \boldsymbol{u} + \partial_z w dz
      = (\eta + H) \nabla \cdot \boldsymbol{u} + w|_{z=\eta} - w|_{z=-H}
    
    \Rightarrow \quad
    D_t \eta + (\eta + H) \nabla \cdot \boldsymbol{u} = 0

multiply with the gravitational acceleration :math:`g` yields

.. math::

    D_t p + (p + c^2) \nabla \cdot \boldsymbol{u} = 0
    \quad \Longleftrightarrow \quad
    \partial_t p + \nabla \cdot (p \boldsymbol{u}) = - c^2 \nabla \cdot \boldsymbol{u}

with :math:`p = g \eta` and :math:`c^2 = g H`.

Scaling
.......
We start with the unscaled shallow water equations from above:

.. math::

    \partial_t \boldsymbol{u} + \boldsymbol{u} \cdot \nabla \boldsymbol{u}
        = - f \underset{\neg}{\boldsymbol u} - \nabla p 
    ~, \quad
    \partial_t p + \nabla \cdot \left( p \boldsymbol{u} \right)
        = - c^2 \nabla \cdot \boldsymbol{u}

and introduce the following scaling:

.. math::

    t = \frac{1}{\Omega} t'
    ~, \quad
    \boldsymbol{x} = L \boldsymbol{x}'
    ~, \quad
    f = \Omega f'
    ~, \quad
    \boldsymbol{u} = U \boldsymbol{u}'
    ~, \quad
    p = P p'

where the variables with a prime are the nondimensional variables.
we further introduce the Rossby number :math:`Ro`, the Froude number :math:`Fr`,
and assume geostrophic balance on first order:

.. math::

    Ro = \frac{U}{\Omega L}
    ~, \quad
    Fr = \frac{U}{c}
    ~, \quad
    \Omega U = \frac{P}{L}

Inserting the nondimensional variables into the shallow water equations yields:

.. math::

    \partial_{t'} \boldsymbol{u}' + Ro~\, \boldsymbol{u}' \cdot \nabla' \boldsymbol{u}'
        = - f' \underset{\neg}{\boldsymbol u}' - \nabla' p'
    ~, \quad
    \partial_{t'} p' + Ro~\, \nabla' \cdot \left( p' \boldsymbol{u}' \right)
        = - \frac{{Ro}^2}{{Fr}^2} \nabla' \cdot \boldsymbol{u}'

Drop the primes, and use the Burger number :math:`Bu = \frac{{Ro}^2}{{Fr}^2}`
yields the scaled shallow water equations.

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
    from .state import State

    # ----------------------------------------------------------------
    #  Importing generic classes and modules
    # ----------------------------------------------------------------
    # importing modules
    from fridom.framework import utils
    from fridom.framework import time_steppers
    from fridom.framework import projection

    # import logger and config
    from fridom.framework.logger import log
    from fridom.framework.configuration import config

    # importing classes
    from fridom.framework.field_variable import FieldVariable
    from fridom.framework.model_state import ModelState
    from fridom.framework.model import Model

# ================================================================
#  Setup lazy loading
# ================================================================
base_fr = "fridom.framework"
base_sw = "fridom.shallowwater"
all_modules_by_origin = { 
    base_fr: ["time_steppers", "utils", "projection"],
    base_sw: ["grid", "modules", "initial_conditions"],
}

all_imports_by_origin = { 
    f"{base_fr}.configuration": ["config"],
    f"{base_fr}.logger": ["log"],
    f"{base_sw}.model_settings": ["ModelSettings"],
    f"{base_sw}.state": ["State"],
    f"{base_fr}.field_variable": ["FieldVariable"],
    f"{base_fr}.model_state": ["ModelState"],
    f"{base_fr}.model": ["Model"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
