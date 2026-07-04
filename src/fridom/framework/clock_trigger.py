"""Emits signals based on the state of a clock."""
from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

import fridom.framework as fr

if TYPE_CHECKING:
    from collections.abc import Callable


class ClockTrigger:

    """
    Emits signals based on the state of a clock.

    Description
    -----------
    Some modules should not be active all the time. For example you may want
    a module that is only active every 10 time steps. This class provides
    some basic functionality to check if a start condition is met, if a
    stop condition is met, and if the module should advance.

    Parameters
    ----------
    start_date : np.datetime64, float, optional
        The start date of the module. If provided, the module will start when
        the model time is greater or equal to this date.
    start_step : int, optional
        The start step of the module. If provided, the module will start when
        the model step is greater or equal to this step.
    stop_date : np.datetime64, float, optional
        The stop date of the module. If provided, the module will stop when
        the model time is greater or equal to this date.
    stop_step : int, optional
        The stop step of the module. If provided, the module will stop when
        the model step is greater or equal to this step.
    time_interval : np.timedelta64, float, optional
        The time interval between each advance of the module. If provided, the
        object will trigger an advance every `time_interval` seconds.
    step_number : int, optional
        The number of steps between each advance of the module. If provided,
        the object will trigger an advance every `step_number` steps.

    """

    def __init__(self,
                 start_date: np.datetime64 | float | None = None,
                 start_step: int | None = None,
                 stop_date: np.datetime64 | float | None = None,
                 stop_step: int | None = None,
                 time_interval: np.timedelta64 | float | None = None,
                 step_size: int | None = None) -> None:
        # check for too many arguments
        fr.exceptions.TooManyArgumentsError.check(
            max_args=1, time_interval=time_interval, step_number=step_size)

        self._start_date = start_date
        self._start_step = start_step
        self._stop_date = stop_date
        self._stop_step = stop_step
        self._time_interval = time_interval
        self._step_size = step_size

        self._start_callback = self._set_start_callback(start_date, start_step)
        self._stop_callback = self._set_stop_callback(stop_date, stop_step)
        self._number_of_advanced_steps = 0
        self._started = False
        self._stopped = False
        # convert the time interval to seconds
        if (time_interval is not None
                and isinstance(time_interval, np.timedelta64)):
            time_interval = fr.utils.to_seconds(time_interval)

        self._trigger_on_first_step = True
        self._time_interval = time_interval
        self._step_size = step_size
        self._start_time = None
        self._start_it = None

    def _set_start_callback(self,
                            start_date: np.datetime64 | float | None = None,
                            start_step: int | None = None,
                            ) -> Callable[[fr.Clock], bool]:
        fr.exceptions.TooManyArgumentsError.check(
            max_args=1, start_date=start_date, start_step=start_step)

        # create a trigger callback for the start condition
        if start_date is None and start_step is None:
            # No start condition
            def _start_callback(_clock: fr.Clock) -> bool:
                return True
        elif isinstance(start_date, np.datetime64):
            # convert the start date to a float in seconds
            start_time = fr.utils.to_seconds(start_date)
            def _start_callback(clock: fr.Clock) -> bool:
                return clock.time >= start_time
        elif start_date is not None:
            # use the start date as a float in seconds
            def _start_callback(clock: fr.Clock) -> bool:
                return clock.time >= start_date
        elif start_step is not None:
            # use the start step
            def _start_callback(clock: fr.Clock) -> bool:
                return clock.it >= start_step
        else:
            msg = "Invalid start condition."
            raise ValueError(msg)
        return _start_callback

    def _set_stop_callback(self,
                           stop_date: np.datetime64 | float | None = None,
                           stop_step: int | None = None,
                           ) -> Callable[[fr.Clock], bool]:
        fr.exceptions.TooManyArgumentsError.check(
            max_args=1, stop_date=stop_date, stop_step=stop_step)

        if stop_date is None and stop_step is None:
            # No stop condition
            def _stop_callback(_clock: fr.Clock) -> bool:
                return False
        elif isinstance(stop_date, np.datetime64):
            # convert the stop date to a float in seconds
            stop_time = fr.utils.to_seconds(stop_date)
            def _stop_callback(clock: fr.Clock) -> bool:
                return clock.time >= stop_time
        elif stop_date is not None:
            # use the stop date as a float in seconds
            def _stop_callback(clock: fr.Clock) -> bool:
                return clock.time >= stop_date
        elif stop_step is not None:
            # use the stop step
            def _stop_callback(clock: fr.Clock) -> bool:
                return clock.it >= stop_step
        else:
            msg = "Invalid stop condition."
            raise ValueError(msg)
        return _stop_callback

    def should_start(self, clock: fr.Clock) -> bool:
        """
        Check if the module should start.

        Description
        -----------
        This method checks if a start requirement is met. And returns
        `True` if this requirement is met and the start condition has not
        been met before. Hence, this method will only return `True` once.

        Parameters
        ----------
        clock : fr.Clock
            The clock of the model.

        Returns
        -------
        bool
            `True` if the module should start, `False` otherwise.

        """
        should_start = not self._started and self._start_callback(clock)
        if should_start:
            self._started = True
            self._start_it = copy(clock.it)
            self._start_time = copy(clock.time)
        return should_start

    def should_stop(self, clock: fr.Clock) -> bool:
        """
        Check if the module should stop.

        Description
        -----------
        This method checks if a stop requirement is met. And returns `True` if
        this requirement is met and the stop condition has not been met before.
        Hence, this method will only return `True` once.

        Parameters
        ----------
        clock : fr.Clock
            The clock of the model.

        Returns
        -------
        bool
            `True` if the module should stop, `False` otherwise.

        """
        should_stop = not self._stopped and self._stop_callback(clock)
        if should_stop:
            self._stopped = True
        return should_stop

    def should_advance(self, clock: fr.Clock) -> bool:
        """
        Check if the module should advance.

        Description
        -----------
        This method checks if the module should advance. It returns `True` if
        the module should advance and `False` otherwise.

        Parameters
        ----------
        clock : fr.Clock
            The clock of the model.

        Returns
        -------
        bool
            `True` if the module should advance, `False` otherwise.

        """
        if not self._started or self._stopped:
            return False

        n_steps = self._number_of_advanced_steps
        if self._time_interval is not None:
            target_time = self._start_time + n_steps * self._time_interval
            should_advance = clock.time >= target_time
        elif self._step_size is not None:
            target_it = self._start_it + n_steps * self._step_size
            should_advance = clock.it >= target_it
        else:
            should_advance = True

        if should_advance:
            self._number_of_advanced_steps += 1

            if (not self.trigger_on_first_step
                    and self._number_of_advanced_steps == 1):
                should_advance = False

        return should_advance

    def check(self, clock: fr.Clock) -> bool:
        """
        Check if the module should advance with automatic start and stop.

        Description
        -----------
        This method checks if the module should advance. It will automatically
        start and stop the module if the start and stop conditions are met.

        Parameters
        ----------
        clock : fr.Clock
            The clock of the model.

        Returns
        -------
        bool
            `True` if the module should advance, `False` otherwise.

        """
        self.should_start(clock)
        self.should_stop(clock)
        return self.should_advance(clock)

    def reset(self) -> None:
        """Reset the module."""
        self._number_of_advanced_steps = 0
        self._started = False
        self._stopped = False

    def _configuration(self) -> tuple:
        """Return the configuration (excluding the mutable state)."""
        return (self._start_date, self._start_step,
                self._stop_date, self._stop_step,
                self._time_interval, self._step_size,
                self._trigger_on_first_step)

    def __eq__(self, other: object) -> bool:
        """
        Compare two clock triggers by their configuration.

        Description
        -----------
        Only the configuration (start/stop conditions and intervals) is
        compared, not the mutable trigger state. Trigger decisions are
        made on the host side and can never influence jit-compiled
        computations, so equally configured triggers are considered
        equal. This keeps the jit-cache keys of objects that contain a
        clock trigger stable.
        """
        if not isinstance(other, ClockTrigger):
            return NotImplemented
        return self._configuration() == other._configuration()

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        res = "ClockTrigger("
        if self._start_date is not None:
            res += f"start_date={self._start_date}, "
        elif self._start_step is not None:
            res += f"start_step={self._start_step}, "
        else:
            res += "start_date=None, "
        if self._time_interval is not None:
            res += f"time_interval={self._time_interval}, "
        elif self._step_size is not None:
            res += f"step_size={self._step_size}, "
        else:
            res += "time_interval=None, "
        if self._stop_date is not None:
            res += f"stop_date={self._stop_date})"
        elif self._stop_step is not None:
            res += f"stop_step={self._stop_step})"
        else:
            res += "stop_date=None)"
        return res

    @property
    def trigger_on_first_step(self) -> bool:
        """Check if the module should trigger on the first step."""
        return self._trigger_on_first_step

    @trigger_on_first_step.setter
    def trigger_on_first_step(self, value: bool) -> None:
        self._trigger_on_first_step = value
