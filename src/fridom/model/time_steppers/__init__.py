"""
The time-stepper namespace (``fr.time_steppers``).

Description
-----------
Owning class spec: ``design/specs/model/classes/time_steppers.md``.
Wave 4 added ``TimeStepper``/``StepperState`` and ``AdamBashforth``;
Wave 5 adds the explicit RK family (``ExplicitRungeKutta``,
``LowStorageRK3``, ``ButcherTableau``, ``tableaus``) and the IMEX
multistep family (``IMEXMultistep``, ``IMEXState``, ``CNAB2``,
``SBDF2``); the exponential family adds ``ETDRK4`` (and its
``phi_functions`` kernel), which integrates the linear operator
EXACTLY through the model's eigenbasis — no gravity-wave CFL, no
wave damping, and an identity ``time_discretization_effect``.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from .adam_bashforth import ABState, AdamBashforth
    from .base import StepperState, TimeStepper
    from .exponential import ETDRK4, phi_functions
    from .imex import CNAB2, SBDF2, IMEXMultistep, IMEXState
    from .runge_kutta import (
        ButcherTableau,
        ExplicitRungeKutta,
        LowStorageRK3,
        tableaus,
    )

base = "fridom.model.time_steppers"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{base}.base": ["TimeStepper", "StepperState"],
    f"{base}.adam_bashforth": ["AdamBashforth", "ABState"],
    f"{base}.runge_kutta": [
        "ExplicitRungeKutta", "LowStorageRK3", "ButcherTableau",
        "tableaus"],
    f"{base}.imex": ["IMEXMultistep", "IMEXState", "CNAB2", "SBDF2"],
    f"{base}.exponential": ["ETDRK4", "phi_functions"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
