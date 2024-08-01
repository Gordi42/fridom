"""
Time Steppers
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from . import runge_kutta

    from .time_stepper import TimeStepper
    from .adam_bashforth import AdamBashforth
    from .runge_kutta import RungeKutta, RKMethods

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.framework.time_steppers"

all_modules_by_origin = {base_path: ["runge_kutta"], }

all_imports_by_origin = { 
    f"{base_path}.time_stepper": ["TimeStepper"],
    f"{base_path}.adam_bashforth": ["AdamBashforth"],
    f"{base_path}.runge_kutta": ["RungeKutta", "RKMethods"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)