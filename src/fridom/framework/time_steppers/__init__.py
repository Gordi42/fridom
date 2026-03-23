"""Time Steppers."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .adam_bashforth import AdamBashforth, _perform_time_step
    from .runge_kutta import ButcherTableau, RKMethods, RungeKutta
    from .time_stepper import TimeStepper

# ================================================================
#  Setup lazy loading
# ================================================================
base_path = "fridom.framework.time_steppers"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{base_path}.time_stepper": ["TimeStepper"],
    f"{base_path}.adam_bashforth": ["AdamBashforth", "_perform_time_step"],
    f"{base_path}.runge_kutta": ["RungeKutta", "RKMethods", "ButcherTableau"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
