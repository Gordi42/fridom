"""Model class for the fridom framework."""
from __future__ import annotations

import numpy as np

import fridom.framework as fr


class Model:

    """
    The main model class.

    Parameters
    ----------
    mset : ModelSettingsBase
        The model settings.

    """

    def __init__(self, mset: fr.ModelSettingsBase) -> None:
        self.mset = mset
        self.model_state = fr.ModelState(mset)

    def start(self) -> None:
        """Prepare the model for running."""
        # start all modules
        self.timer.total.start()
        for module in self._modules:
            module.start()
        self.model_state.panicked = False

    def stop(self) -> None:
        """Finish the model run."""
        for module in self._modules:
            module.stop()
        self.timer.total.stop()
        self.progress_bar.stop()

    def reset(self) -> None:
        """Reset the model (pointers, tendencies)."""
        for module in self._modules:
            module.reset()
        self.model_state.reset()
        self.timer.reset()

    # ============================================================
    #   RUN MODEL
    # ============================================================

    def run_backward(self, steps: int | None = None) -> None:
        """
        Run the model backward in time.

        Description
        -----------
        This method runs the model backward in time for a given number of steps.
        """
        # Prepare the model for running
        if self.restart_module.should_reload():
            self.load(self.restart_module.file)
        self.time_stepper.dt = -abs(self.time_stepper.dt)  # ensure dt is negative
        start_value = self.model_state.clock.it
        # step count always increases even when running backward
        final_value = start_value + steps

        self.start()

        # Execute the first time step
        self._execute_first_time_step()

        # Start the progress bar
        self.progress_bar.start()
        self.progress_bar.set_options(
            main_loop_type="for loop",
            datetime_formatting=False,
            start_value=start_value,
            final_value=final_value)

        # ----------------------------------------------------------------
        #  Main loop
        # ----------------------------------------------------------------
        self._main_loop_steps(start_value+1, final_value)

        # stop the model
        self._finalize_run()

    def run(self,
            steps: int | None = None,
            runlen: np.timedelta64 | float | None = None,
            start_step: int = 0,
            start_time: np.datetime64 | float = 0,
            end_time: np.datetime64 | float | None = None) -> None:
        """
        Run the model.

        Parameters
        ----------
        steps : int (default: None)
            Number of steps to run.
        runlen : np.timedelta64 | float | int (default: None)
            Length of the run.
        start_step : int (default: 0)
            Start iteration of the run.
        start_time : np.datetime64 | float | int (default: 0)
            Start time of the run.
        end_time : np.datetime64 | float | int (default: None)
            End time of the run.

        """
        #  Check input
        fr.exceptions.TooManyArgumentsError.check(
            max_args=1, steps=steps, runlen=runlen, end_time=end_time)

        #  Convert time parameters to seconds
        datetime_formatting = self._datetime_formatting(
            start_time, end_time, runlen)
        start_time, end_time, runlen = self._initialize_run(
            start_time, end_time, runlen, start_step)

        # Determine loop type and final values
        main_loop_type, start_value, final_value = self._determine_loop_type(
            steps, start_time, runlen, end_time)

        # check if the final value is reached
        if start_value >= final_value:
            return

        # Prepare the model for running
        if self.restart_module.should_reload():
            self.load(self.restart_module.file)

        # make sure the time stepper has a positive time step
        self.time_stepper.dt = abs(self.time_stepper.dt)

        self.start()

        # Execute the first time step
        self._execute_first_time_step()

        # Start the progress bar
        self.progress_bar.start()
        self.progress_bar.set_options(
            main_loop_type=main_loop_type,
            datetime_formatting=datetime_formatting,
            start_value=start_value,
            final_value=final_value)

        # ----------------------------------------------------------------
        #  Main loop
        # ----------------------------------------------------------------
        if main_loop_type == "for loop":
            self._main_loop_steps(start_value+1, final_value)
        elif main_loop_type == "while loop":
            self._main_loop_time(final_value)

        # stop the model
        self._finalize_run()

    def _datetime_formatting(self,
                             start_time: np.datetime64 | float,
                             end_time: np.datetime64 | float | None,
                             runlen: np.timedelta64 | float | None) -> bool:
        """Check if some of the input parameters are in datetime format."""
        if isinstance(start_time, np.datetime64):
            return True
        if isinstance(end_time, np.datetime64):
            return True
        return bool(isinstance(runlen, np.timedelta64))

    def _initialize_run(self,
                        start_time: np.datetime64 | float,
                        end_time: np.datetime64 | float | None,
                        runlen: np.timedelta64 | float | None,
                        start_step: int) -> tuple[float, float, float]:
        """Convert time parameters and initialize the clock."""
        self.model_state.clock.set_start(start_time)
        self.model_state.clock.time = fr.utils.to_seconds(start_time)
        self.model_state.clock.it = start_step
        return (fr.utils.to_seconds(start_time),
                fr.utils.to_seconds(end_time),
                fr.utils.to_seconds(runlen))

    def _determine_loop_type(self,
                             steps: int | None,
                             start_time: float,
                             runlen: float | None,
                             end_time: float | None) -> tuple[str, int, int]:
        """Determine the loop type and final values."""
        if runlen is not None:
            end_time = start_time + runlen

        if steps is not None:
            main_loop_type = "for loop"
            start_value = self.model_state.clock.it
            final_value = start_value + steps
        else:
            main_loop_type = "while loop"
            start_value = start_time
            final_value = end_time

        return main_loop_type, start_value, final_value

    def _execute_first_time_step(self) -> None:
        """Print the timing of the first time step."""
        # compile the modules
        from time import time
        if fr.config.backend_is_jax:
            fr.log.notice("Compiling modules at first time step")
            start_time = time()

        # Execute modules that should run at the start
        for module in self.diagnostics.module_list:
            if module.execute_at_start:
                self.model_state = module.update(self.model_state)

        # Execute the first time step
        was_enabled = self.progress_bar.is_enabled()
        self.progress_bar.disable()

        self._safe_step()

        # re-enable the progress bar if it was enabled before
        if was_enabled:
            self.progress_bar.enable()

        # Print the compilation time
        if fr.config.backend_is_jax:
            fr.log.notice(
                f"Compilation finished in {time()-start_time:.2f} seconds")

    def _safe_step(self) -> None:
        """Run a single time step and catch any exceptions."""
        if self.mset.raise_error_when_something_goes_wrong:
            self.step()
            return
        try:
            self.step()
        except Exception as e:  # noqa: BLE001
            fr.log.error("An error occurred during the model run.")
            fr.log.error(e)
            self.model_state.panicked = True

    def _main_loop_steps(self, start_value: int, final_value: int) -> None:
        start_it = self.model_state.clock.it
        fr.log.info(
            f"Running model from iteration {start_value} to {final_value}")

        # loop over the given number of steps
        for _ in range(start_it, final_value):
            self._safe_step()

            if self.model_state.panicked:
                fr.log.warning(
                    "Something went wrong. Stopping model.")
                break

    def _main_loop_time(self, end_time: float) -> None:
        fr.log.info(
            f"Running model from {self.model_state.clock.time} to {end_time}")

        # loop until the end time is reached
        while self.model_state.clock.time < end_time:
            self._safe_step()

            if self.model_state.panicked:
                fr.log.warning(
                    "Something went wrong. Stopping model.")
                break

    def _finalize_run(self) -> None:
        """Finalize the model run."""
        self.progress_bar.print_progress_bar(self.model_state)
        # finalize the model
        self.stop()

        fr.log.info(
            "Model run finished at it: %d, time: %f",
            self.model_state.clock.it,
            self.model_state.clock.time)
        fr.log.info(self.mset.timer)


    # ============================================================
    #   SINGLE TIME STEP
    # ============================================================

    def step(self) -> None:
        """Update the model state by one time step."""
        # perform the time step
        self.model_state = self.time_stepper.update(mz=self.model_state)

        # check if there are any nans in the state variable
        self.model_state = self.nan_checker.update(self.model_state)

        # make diagnostics
        self.model_state = self.diagnostics.update(mz=self.model_state)

        # Update the progress bar
        self.progress_bar.update(self.model_state)

        # check if the model should restart
        if self.restart_module.should_restart(self.model_state):
            self.restart_module.restart(self)

    # ============================================================
    #   Getters and setters
    # ============================================================

    @property
    def z(self) -> fr.VectorField:
        """Returns the current state variable."""
        return self.model_state.z

    @z.setter
    def z(self, value: fr.VectorField) -> None:
        """Set the current state variable."""
        self.model_state.z = value

    @property
    def timer(self) -> fr.Timer:
        """The timing module."""
        return self.mset.timer

    @property
    def nan_checker(self) -> fr.modules.NaNChecker:
        """The NaN checker module."""
        return self.mset.nan_checker

    @property
    def progress_bar(self) -> fr.modules.ProgressBar:
        """The progress bar module."""
        return self.mset.progress_bar

    @property
    def restart_module(self) -> fr.modules.RestartModule:
        """The restart module."""
        return self.mset.restart_module

    @property
    def time_stepper(self) -> fr.time_steppers.TimeStepper:
        """The time stepper."""
        return self.mset.time_stepper

    @property
    def tendencies(self) -> fr.modules.ModuleContainer:
        """The module container for all tendencies."""
        return self.mset.tendencies

    @property
    def diagnostics(self) -> fr.modules.ModuleContainer:
        """The module container for all diagnostics."""
        return self.mset.diagnostics

    @property
    def _modules(self) -> list[fr.modules.Module]:
        """List of all modules."""
        return [
            self.nan_checker,
            self.restart_module,
            self.time_stepper,
            self.tendencies,
            self.diagnostics,
        ]

    # ============================================================
    #   OTHER METHODS
    # ============================================================

    def load(self, file: str) -> None:
        """
        Load a model from a file.

        Parameters
        ----------
        file : str
            The filename to load the model from

        """
        # underscores are not allowed in the filename
        from pathlib import Path

        import dill
        # get a list of all files in the directory that start with the filename
        with Path(file).open("rb") as f:
            model = dill.load(f)  # noqa: S301
            model.mset.grid = self.mset.grid

        for key, attr in vars(model).items():
            setattr(self, key, attr)

    def save(self, file: str) -> None:
        """
        Save the full model to a file.

        Parameters
        ----------
        file : str
            The filename to save the model to

        """
        from pathlib import Path

        import dill
        with Path(file).open("wb") as f:
            fr.log.verbose(f"Saving model to {file}")
            grid = self.mset.grid
            # remove the grid from the model before pickling
            self.mset.grid = None
            dill.dump(self, f)
            # restore the grid
            self.mset.grid = grid
