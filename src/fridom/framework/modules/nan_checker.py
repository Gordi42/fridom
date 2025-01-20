"""A module to check if a state vector contains NaN values."""
from __future__ import annotations

import fridom.framework as fr


class NaNChecker(fr.modules.Module):

    """
    Check if a state vector contains NaN values.

    Description
    -----------
    This module checks if the state vector contains NaN values. If it does, the
    model panic flag is set to True and the model is stopped. Since it can be
    computationally expensive to check for NaNs every iteration, the module
    has a clock trigger that specifies when to check for NaNs.

    Parameters
    ----------
    clock_trigger : fr.ClockTrigger, optional
        When to check for NaNs. If not specified, the module will check every
        100 iteration steps.

    """

    name = "NaN Checker"

    def __init__(self, clock_trigger: fr.ClockTrigger | None = None) -> None:
        super().__init__()
        self.clock_trigger = clock_trigger or fr.ClockTrigger(step_size=100)

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102

        if not self.clock_trigger.check(mz.clock):
            return mz

        if mz.z.has_nan():
            msg = "State variable contains NaNs. Stopping model."
            fr.log.critical(msg)
            mz.panicked = True

        return mz

    def _on_reset(self) -> None:
        self.clock_trigger.reset()

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def clock_trigger(self) -> fr.ClockTrigger:
        """When to check for NaNs."""
        return self._clock_trigger

    @clock_trigger.setter
    def clock_trigger(self, value: fr.ClockTrigger) -> None:
        self._clock_trigger = value
