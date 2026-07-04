"""A progress bar module to display the progress of the simulation."""
from __future__ import annotations

import io
import sys
import time

import numpy as np
from tqdm import tqdm

import fridom.framework as fr


class ProgressBar(fr.modules.Module):

    """
    A progress bar module to display the progress of the simulation.

    Description
    -----------
    The progress bar class is a wrapper around the tqdm progress bar. It
    has a custom format and handles the output to the stdout when the
    stdout is a file.

    Parameters
    ----------
    disable : bool
        Whether to disable the progress bar.

    """

    name = "Progress Bar"
    def __init__(self) -> None:
        super().__init__()
        self._pbar = None
        self._file_output = None
        self._output = None
        self._last_it = None
        self._last_call = None
        self._aimed_interval = None
        self._main_loop_type = None
        self._datetime_formatting = None
        self._start_value = None
        self._final_value = None
        self._interval = None

    @fr.modules.module_method
    def start(self) -> None:  # noqa: D102
        # TODO(Silvano): only the main rank should print the progress bar
        disable = False

        # ----------------------------------------------------------------
        #  Set the progress bar format
        # ----------------------------------------------------------------
        bar_format = "{percentage:3.2f}%|{bar}| "
        bar_format += "[{elapsed}<{remaining}]{postfix}"

        # ----------------------------------------------------------------
        #  Check if the stdout is a file
        # ----------------------------------------------------------------
        file_output = fr.utils.stdout_is_file()
        if file_output:
            # if the stdout is a file, tqdm would print to the stderr by default
            # we could instead print to the stdout, but this would mess up
            # the look of the progress bar due to "\r" characters
            # so we create a StringIO object to capture the output
            # and adjust the progress bar accordingly
            output = io.StringIO()
        else:
            output = sys.stdout

        # ----------------------------------------------------------------
        #  Create the progress bar
        # ----------------------------------------------------------------
        pbar = tqdm(
            total=100,
            disable=disable,
            bar_format=bar_format,
            unit="%",
            file=output)

        # ----------------------------------------------------------------
        #  Set the attributes
        # ----------------------------------------------------------------
        self._pbar = pbar
        self._file_output = file_output
        self._output = output
        self._last_it = None
        self._last_call = time.time()
        self._aimed_interval = None
        self._main_loop_type = None
        self._datetime_formatting = None
        self._start_value = None
        self._final_value = None

    @fr.modules.module_method
    def stop(self) -> None:  # noqa: D102
        if self._pbar is not None:
            self._pbar.close()
        self._pbar = None
        self._file_output = None
        self._output = None
        self._last_it = None
        self._last_call = None
        self._aimed_interval = None
        self._main_loop_type = None
        self._datetime_formatting = None
        self._start_value = None
        self._final_value = None

    def set_options(self,
                    main_loop_type: str,
                    datetime_formatting: bool,
                    start_value: float,
                    final_value: float) -> None:
        self._main_loop_type = main_loop_type
        self._datetime_formatting = datetime_formatting
        self._start_value = start_value
        self._final_value = final_value

    def print_progress_bar(self, mz: fr.ModelState) -> None:
        if not self.is_enabled():
            return None

        it = int(mz.clock.it)
        last_it = self._last_it or it - 1
        n_its = it - last_it

        # get the time between the last call (in milliseconds)
        now = time.time()
        elapsed = 1e3 * (now - self._last_call) / max(n_its, 1)
        self._last_call = now
        self._last_it = it
        self._aimed_interval = max(min(1000, 200 / max(elapsed, 1e-3)), 1)
        elapsed = f"{int(elapsed)} ms/it"


        # Get the current progress value
        match self._main_loop_type:
            case "for loop":
                value = it
            case "while loop":
                value = mz.clock.time

        # map the value to a percentage
        value = 100 * ( (value - self._start_value)
                       / (self._final_value - self._start_value) )

        # clamp the value between 0 and 100
        value = float(max(0, min(100, value)))

        # Create a postfix string for the progress bar
        if self._datetime_formatting:
            time_str = np.datetime64(int(mz.clock.time), "s")
        else:
            time_str = fr.utils.humanize_number(float(mz.clock.time), unit="seconds")

        postfix = f"It: {it} - Time: {time_str}"

        # update the progress bar
        self._pbar.n = value
        self._pbar.set_postfix_str(f"{elapsed}  at {postfix}")

        if not self._file_output:
            return mz

        # print the progress to the stdout
        fr.log.info(self._output.getvalue().split("\r")[1])

        # clear the output string
        self._output.seek(0)

        return None


    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        if self._start_value is None:
            return mz

        it = int(mz.clock.it)

        if self._last_it is None:
            self._last_it = it

        n_its = it - self._last_it


        # check if an interval was set
        if self._interval and n_its % self._interval != 0:
            return mz

        if (self._interval is None and self._aimed_interval is not None
                and n_its % int(self._aimed_interval) != 0):
            return mz

        self.print_progress_bar(mz)

        return mz

    @property
    def interval(self) -> int | None:
        """The interval at which to update the progress bar."""
        return self._interval

    @interval.setter
    def interval(self, value: int | None) -> None:
        self._interval = value
