"""
The time-stepper namespace (``fr.time_steppers``).

Description
-----------
Owning class spec: ``notes/framework2/model/classes/time_steppers.md``.
Wave 4 adds ``TimeStepper``/``StepperState`` and ``AdamBashforth``;
Wave 5 adds ``ExplicitRungeKutta``, ``LowStorageRK3``, ``tableaus``,
``IMEXMultistep``, ``CNAB2``, ``SBDF2``.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from .adam_bashforth import ABState, AdamBashforth
    from .base import StepperState, TimeStepper

base = "fridom.framework2.model.time_steppers"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{base}.base": ["TimeStepper", "StepperState"],
    f"{base}.adam_bashforth": ["AdamBashforth", "ABState"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
