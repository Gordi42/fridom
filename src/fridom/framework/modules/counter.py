"""A counting module for debugging purposes."""
from __future__ import annotations

import fridom.framework as fr


class Counter(fr.modules.Module):

    """
    A counting module for debugging purposes.

    Description
    -----------
    Every time the update method is called, the counter is incremented by one.
    Additionally, a clock trigger can be set to only increment the counter
    for specific clock events.

    Parameters
    ----------
    clock_trigger : fr.ClockTrigger, optional
        A clock trigger to only increment the counter for specific clock events.
        If not set, the counter is incremented every time the update method is
        called.

    """

    name = "Counter"

    def __init__(self, clock_trigger: fr.ClockTrigger | None = None) -> None:
        super().__init__()
        self.clock_trigger = clock_trigger
        self.counter: int

    def _on_setup(self) -> None:
        self.clock_trigger = self.clock_trigger or fr.ClockTrigger()
        self.counter = 0

    def _on_reset(self) -> None:
        self.clock_trigger.reset()
        self.counter = 0

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        if not self.clock_trigger.check(mz.clock):
            return mz
        self.counter += 1
        return mz
