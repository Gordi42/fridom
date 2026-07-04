"""Create a mp4 video from the model."""
from __future__ import annotations

import multiprocessing as mp
import queue
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import fridom.framework as fr


# FIXME(Silvano): The parallel option is not working with jax anymore
class VideoWriter(fr.modules.Module):

    """
    Create a mp4 video from the model.

    Description
    -----------
    To create a mp4 video from the model, one must provide a `ModelPlotter`
    that will be used to create the figure. The video writer does not support
    MPI parallelism.

    Parameters
    ----------
    model_plotter : ModelPlotter
        The model plotter that will be used to create the figure.
    write_interval : np.timedelta64 | float
        The interval at which the data should be written to the file.
    filename : str, optional (default="output.mp4")
        The filename of the video (will be stored in videos/filename).
    fps : int, optional (default=30)
        The frames per second of the video.
    parallel : bool, optional (default=True)
        If True, the video writer will use parallelism to create the video.
    max_jobs : float, optional (default=0.4)
        The maximum fraction of the available threads that will be used.

    """

    name = "Video Writer"
    def __init__(self,
                 model_plotter: fr.modules.animation.ModelPlotter,
                 model_time_per_second: np.timedelta64 | float,
                 filename: str = "output.mp4",
                 fps: int = 30,
                 parallel: bool = True,
                 max_jobs: float = 0.2,
                 ) -> None:
        super().__init__()

        # Convert the times to seconds
        if isinstance(model_time_per_second, np.timedelta64):
            model_time_per_second = fr.utils.to_seconds(
                model_time_per_second)
        # Compute the write interval
        write_interval = model_time_per_second / fps

        # add .mp4 extension if it is not there
        path = Path(filename)
        if not path.suffix:
            filename += ".mp4"

        self.model_plotter = model_plotter
        self.write_interval = write_interval
        self.filename = str(Path("videos") / filename)
        self.fps = fps
        self.max_jobs = max_jobs
        # set the flag for MPI availability
        self.mpi_available = False
        self.writer = None
        self.parallel = parallel
        self.fig = None
        self._last_write_time = None
        self._last_checkpoint_time = None

    def _on_setup(self) -> None:
        # create video folder if it does not exist
        if not Path("videos").exists():
            fr.log.info("Creating videos folder")
            Path("videos").mkdir(parents=True)

        # delete the file if it already exists
        if Path(self.filename).exists():
            fr.log.notice(f"Deleting existing video file {self.filename}")
            Path(self.filename).unlink()

        # use maximum of 40% the available threads
        if self.parallel:
            self.maximum_jobs = int(self.max_jobs*mp.cpu_count())
        self.fig = None

    @fr.modules.module_method
    def start(self) -> None:  # noqa: D102
        # list for the jobs and queues for creating the figures
        self.running_jobs = []       # Processes
        self.open_queues  = []       # Queues

        # start the writer
        if self.writer is not None and not self.writer.closed:
            fr.log.warning(
                "VideoWriter.start() called without closing the previous"
                " writer.",
                "Continue with the previous writer.")
        else:
            import imageio  # noqa: PLC0415 (deferred import of optional/heavy dependency)
            self.writer = imageio.get_writer(self.filename, fps=self.fps)

    @fr.modules.module_method
    def stop(self) -> None:  # noqa: D102
        # collect all figures
        while len(self.running_jobs) > 0:
            fr.log.info("Collecting remaining figures")
            self.collect_figures()
        if self.writer is not None:
            fr.log.debug("Closing the video writer")
            self.writer.close()
            fr.log.debug("Video writer closed")
        if self.fig is not None:
            import matplotlib.pyplot as plt  # noqa: PLC0415 (deferred import of optional/heavy dependency)
            plt.close(self.fig)
            self.fig = None
        self._last_write_time = None
        self._last_checkpoint_time = None

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        time = mz.clock.time
        # ----------------------------------------------------------------
        #  Check if it is time to write
        # ----------------------------------------------------------------
        if self._last_write_time is None or self._last_checkpoint_time is None:
            time_to_write = True
        else:
            next_write_time = self._last_write_time + self.write_interval
            if (self._last_checkpoint_time < next_write_time and
                time >= next_write_time):
                time_to_write = True
            else:
                time_to_write = False
        self._last_checkpoint_time = time

        if not time_to_write:
            return mz
        self._last_write_time = time

        if self.parallel:
            self.parallel_update(mz)
        else:
            self.single_update(mz)
        return mz

    def parallel_update(self, mz: fr.ModelState) -> None:
        """Create a new figure in a separate process and queue it."""
        # collect finished figures
        self.collect_figures()

        # wait until there is space for a new job
        while len(self.running_jobs) >= self.maximum_jobs:
            self.collect_figures()

        # create a new figure
        q = mp.Queue()
        diagnostics = self.mset.diagnostics
        self.mset.diagnostics = None

        kw = {"kwargs": self.model_plotter.prepare_arguments(mz),
              "output_queue": q,
              "model_plotter": self.model_plotter}
        job = mp.Process(target=VideoWriter.p_make_figure, kwargs=kw)
        self.mset.diagnostics = diagnostics

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            job.start()

        self.open_queues.append(q)
        self.running_jobs.append(job)

    def single_update(self, mz: fr.ModelState) -> None:
        """Create a figure and append it to the video (serial mode)."""
        if self.fig is None:
            self.fig = self.model_plotter.create_figure()
        else:
            self.fig.clear()
        kw = self.model_plotter.prepare_arguments(mz)

        self.model_plotter.update_figure(fig=self.fig, **kw)
        img = self.model_plotter.convert_to_img(self.fig)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.writer.append_data(img)

    def collect_figures(self) -> None:
        """Collect finished figures and append them to the video."""
        while len(self.running_jobs) > 0:
            try:
                img = self.open_queues[0].get(timeout=0.05)
            except queue.Empty:
                break

            # add the figure to the video
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.writer.append_data(img)

            # remove the finished job and queue
            self.running_jobs[0].join()
            self.running_jobs.pop(0)
            self.open_queues.pop(0)

    def show_video(self, width: int = 600) -> Any:
        """Display the video in a Jupyter notebook."""
        from IPython.display import (  # noqa: PLC0415 (optional dependency, only needed in notebooks)
            Video,
        )
        return Video(self.filename, width=width, embed=True)

    @property
    def info(self) -> dict:  # noqa: D102
        res = super().info
        res["filename"] = self.filename
        res["fps"] = self.fps
        res["max_jobs"] = self.max_jobs
        return res

    def __to_numpy__(self, memo: dict) -> VideoWriter:
        return self.__deepcopy__(memo)

    def __deepcopy__(self, memo: dict) -> VideoWriter:
        dont_copy = ["writer", "fig"]
        # now deepcopy the object
        new = self.__class__.__new__(self.__class__)
        for key in self.__dict__:
            if key in dont_copy:
                setattr(new, key, getattr(self, key))
            else:
                setattr(new, key, deepcopy(getattr(self, key), memo))
        return new

    # =====================================================================
    #  PARALLEL FUNCTIONS
    # =====================================================================

    def p_make_figure(**kwargs: Any) -> None:
        """
        Make the image of a ModelPlotter object (parallel function).

        Gets a ModelPlotter object, makes the image of it and puts it in
        the output queue.

        Arguments:
            modelplot (ModelPlotter): model plotter object
            output_queue (mp.Queue) : output queue
        """
        # get output queue
        output_queue = kwargs["output_queue"]
        model_plotter = kwargs["model_plotter"]
        kw = kwargs["kwargs"]
        fig = model_plotter.create_figure()
        model_plotter.update_figure(fig=fig, **kw)

        img = model_plotter.convert_to_img(fig)
        output_queue.put(img)
