"""clock.py - Keep track of the model time."""

from __future__ import annotations

from enum import Enum, auto
from functools import partial

import numpy as np

import fridom.framework as fr


class TimingFormat(Enum):
    """The timing format for the model clock."""

    SECONDS = auto()
    DATETIME = auto()

@partial(fr.utils.jaxify, dynamic=("_start_time", "_passed_time"))
class Clock:
    """A clock to keep track of the model time.

    Parameters
    ----------
    start_date : np.datetime64, optional
        The start date of the model run.
    start_time : float, optional
        The start time of the model run in seconds.

    Raises
    ------
    ValueError
        If both `start_date` and `start_time` are provided.

    """

    def __init__(
        self,
        start_date: np.datetime64 | None = None,
        start_time: float | None = None,
    ) -> None:
        # check that only one of the two is provided
        fr.exceptions.TooManyArgumentsError.check(
            1, start_date=start_date, start_time=start_time
        )

        self._timing_format = TimingFormat.SECONDS
        self.start_time = start_time or 0
        self.start_date = start_date
        self._passed_time = 0

    def reset(self) -> None:
        """Reset the passed time to zero."""
        self._passed_time = 0

    def tick(self, time_step: float) -> None:
        """Increase the passed time by the time step.

        Parameters
        ----------
        time_step : float
            The time step in seconds.

        """
        self._passed_time += time_step

    def get_total_time(self, passed_time: float) -> np.datetime64 | float:
        """Get the total time of the model run.

        Parameters
        ----------
        passed_time : float
            The passed time in seconds.

        Returns
        -------
        `np.datetime64` or `float`
            The total time of the model run. Either a datetime object
            corresponding to the date or a float corresponding to the
            time in seconds.

        """
        match self._timing_format:
            case TimingFormat.DATETIME:
                deltatime = np.timedelta64(int(passed_time), "s")
                return self.start_date + deltatime
            case TimingFormat.SECONDS:
                return self.start_time + passed_time

    def set_start(self, time: np.datetime64 | float) -> None:
        """Set the start time of the model run.

        Parameters
        ----------
        time : `np.datetime64` or `float`
            The start time of the model run.

        """
        if isinstance(time, np.datetime64):
            self.start_date = time
        else:
            self.start_time = time

    def __repr__(self) -> str:
        res = "Clock(\n"
        if self._timing_format == TimingFormat.DATETIME:
            res += f"  start_date = {self.start_date},\n"
        else:
            res += f"  start_time = {self.start_time},\n"
        res += f"  passed_time = {self.passed_time},\n"
        res += f"  current_time = {self.get_total_time(self.passed_time)})"
        return res

    @property
    def start_time(self) -> float:
        """Get the start time in seconds."""
        return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        self._start_time = value
        self._timing_format = TimingFormat.SECONDS

    @property
    def start_date(self) -> np.datetime64 | None:
        """Get the start date."""
        return self._start_date

    @start_date.setter
    def start_date(self, value: np.datetime64 | None) -> None:
        self._start_date = value
        if value is None:
            return
        self._start_time = fr.utils.to_seconds(value)
        self._timing_format = TimingFormat.DATETIME

    @property
    def passed_time(self) -> float:
        """Get the passed time in seconds."""
        return self._passed_time

    @property
    def time(self) -> float:
        """Get the model time in seconds."""
        return self.start_time + self.passed_time

    @time.setter
    def time(self, value: float) -> None:
        self._passed_time = value - self.start_time
