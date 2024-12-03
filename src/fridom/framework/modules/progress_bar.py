import fridom.framework as fr
import numpy as np
import time


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
    `disable` : `bool`
        Whether to disable the progress bar.
    """
    name = "Progress Bar"
    def __init__(self) -> None:
        super().__init__()
        self._pbar = None
        self._file_output = None
        self._output = None
        self._last_call = None
        self._main_loop_type = None
        self._datetime_formatting = None
        self._start_value = None
        self._final_value = None
        return

    @fr.modules.module_method
    def start(self) -> None:
        # only the main rank should print the progress bar
        if fr.utils.I_AM_MAIN_RANK:
            disable = False
        else:
            disable = True
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
            import io
            output = io.StringIO()
        else:
            import sys
            output = sys.stdout

        # ----------------------------------------------------------------
        #  Create the progress bar
        # ----------------------------------------------------------------
        from tqdm import tqdm
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
        self._last_call = time.time()
        self._main_loop_type = None
        self._datetime_formatting = None
        self._start_value = None
        self._final_value = None
        return

    @fr.modules.module_method
    def stop(self) -> None:
        self._pbar.close()
        self._pbar = None
        self._file_output = None
        self._output = None
        self._last_call = None
        self._main_loop_type = None
        self._datetime_formatting = None
        self._start_value = None
        self._final_value = None
        return

    def set_options(self, 
                    main_loop_type: str, 
                    datetime_formatting: bool,
                    start_value: float | int,
                    final_value: float | int):
        self._main_loop_type = main_loop_type
        self._datetime_formatting = datetime_formatting
        self._start_value = start_value
        self._final_value = final_value
        return

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        if self._start_value is None:
            return

        # get the time between the last call (in milliseconds)
        now = time.time()
        elapsed = now - self._last_call
        self._last_call = now
        elapsed = f"{int(elapsed*1e3)} ms/it"

        # Get the current progress value
        match self._main_loop_type:
            case "for loop":
                value = mz.it
            case "while loop":
                value = mz.clock.time

        # map the value to a percentage
        value = 100 * ( (value - self._start_value) 
                       / (self._final_value - self._start_value) )

        # clamp the value between 0 and 100
        value = max(0, min(100, value))

        # Create a postfix string for the progress bar
        if self._datetime_formatting:
            time_str = np.datetime64(int(mz.clock.time), 's')
        else:
            time_str = fr.utils.humanize_number(mz.clock.time, unit="seconds")
        postfix = f"It: {mz.it} - Time: {time_str}"

        # update the progress bar
        self._pbar.n = value
        self._pbar.set_postfix_str(f"{elapsed}  at {postfix}")

        # I had some problem with the bar not updating, so I added the refresh
        # method below. However this takes a lot of time and it seems to be 
        # working fine without it now.
        # self._pbar.refresh()

        if not self._file_output:
            return

        # print the progress to the stdout
        fr.log.info(self._output.getvalue().split("\r")[1])

        # clear the output string
        self._output.seek(0)

        return
