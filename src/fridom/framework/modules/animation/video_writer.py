"""Create a mp4 video from the model."""
from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import fridom.framework as fr


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
    filename : str, optional
        The filename of the video (will be stored in videos/filename) (default:
        "output.mp4").
    fps : int, optional
        The frames per second of the video (default: 30).

    """

    name = "Video Writer"
    def __init__(self,
                 model_plotter: fr.modules.animation.ModelPlotter,
                 model_time_per_second: np.timedelta64 | float,
                 filename: str = "output.mp4",
                 fps: int = 30,
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
        # set the flag for MPI availability
        self.mpi_available = False
        self.writer = None
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

        self.fig = None

    @fr.modules.module_method
    def start(self) -> None:  # noqa: D102
        # start the writer
        if self.writer is not None and not self.writer.closed:
            fr.log.warning(
                "VideoWriter.start() called without closing the previous"
                " writer. Continue with the previous writer.")
        else:
            import imageio  # noqa: PLC0415 (deferred import of optional/heavy dependency)
            self.writer = imageio.get_writer(self.filename, fps=self.fps)

    @fr.modules.module_method
    def stop(self) -> None:  # noqa: D102
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

        self.single_update(mz)
        return mz

    def single_update(self, mz: fr.ModelState) -> None:
        """Create a figure and append it to the video."""
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
