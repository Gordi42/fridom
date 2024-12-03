import fridom.framework as fr
import numpy as np
from typing import Union
import warnings
import os
from pathlib import Path


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
    `model_plotter` : `ModelPlotter`
        The model plotter that will be used to create the figure.
    `write_interval` : `np.timedelta64 | float`
        The interval at which the data should be written to the file.
    `filename` : `str`, optional (default="output.mp4")
        The filename of the video (will be stored in videos/filename).
    `fps` : `int`, optional (default=30)
        The frames per second of the video.
    'parallel' : `bool`, optional (default=True)
        If True, the video writer will use parallelism to create the video.
    `max_jobs` : `float`, optional (default=0.4)
        The maximum fraction of the available threads that will be used.

    """
    name = "Video Writer"
    def __init__(self, 
                 model_plotter: 'fr.ModelPlotter', 
                 model_time_per_second: Union[np.timedelta64, float],
                 filename: str="output.mp4", 
                 fps: int=30,
                 parallel: bool=True,
                 max_jobs: float=0.2,
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
        self.filename = os.path.join("videos", filename)
        self.fps = fps
        self.max_jobs = max_jobs
        # set the flag for MPI availability
        self.mpi_available = False
        self.writer = None
        self.parallel = parallel
        self.fig = None
        self._last_write_time = None
        self._last_checkpoint_time = None
        return

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        super().setup(mset)
        # create video folder if it does not exist
        if not os.path.exists("videos"):
            fr.log.info("Creating videos folder")
            os.makedirs("videos")

        # delete the file if it already exists
        if os.path.exists(self.filename):
            fr.log.notice(f"Deleting existing video file {self.filename}")
            os.remove(self.filename)

        # use maximum of 40% the available threads
        if self.parallel:
            import multiprocessing as mp
            self.maximum_jobs = int(self.max_jobs*mp.cpu_count())
        self.fig = None
        return

    @fr.modules.module_method
    def start(self):
        """
        Method to start the writer process.
        """
        # list for the jobs and queues for creating the figures
        self.running_jobs = []       # Processes
        self.open_queues  = []       # Queues

        # start the writer
        if self.writer is not None and not self.writer.closed:
            fr.log.warning(
                "VideoWriter.start() called without closing the previous writer.",
                "Continue with the previous writer.")
        else:
            import imageio
            self.writer = imageio.get_writer(self.filename, fps=self.fps)
        return

    @fr.modules.module_method
    def stop(self):
        """
        Method to stop the writer process.
        """
        # collect all figures
        while len(self.running_jobs) > 0:
            fr.log.info("Collecting remaining figures")
            self.collect_figures()
        fr.log.debug("Closing the video writer")
        self.writer.close()
        fr.log.debug("Video writer closed")
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
        self._last_write_time = None
        self._last_checkpoint_time = None
        return

    @fr.modules.module_method
    def update(self, mz: 'fr.ModelState') -> 'fr.ModelState':
        """
        Update method of the parallel animated model.
        """
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

    def parallel_update(self, mz: 'fr.ModelState'):
        # collect finished figures
        self.collect_figures()

        # wait until there is space for a new job
        while len(self.running_jobs) >= self.maximum_jobs:
            self.collect_figures()

        # create a new figure
        import multiprocessing as mp
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
        return
    
    def single_update(self, mz: 'fr.ModelState'):
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
        return

    def collect_figures(self):
        while len(self.running_jobs) > 0:
            try :
                img = self.open_queues[0].get(timeout=0.05)
            except:
                break

            # add the figure to the video
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.writer.append_data(img)

            # remove the finished job and queue
            self.running_jobs[0].join()
            self.running_jobs.pop(0)
            self.open_queues.pop(0)
        return

    def show_video(self, width=600):
        from IPython.display import Video
        return Video(self.filename, width=width, embed=True) 

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    filename: {self.filename}\n"
        res += f"    interval: {self.interval}\n"
        res += f"    fps: {self.fps}\n"
        res += f"    max_jobs: {self.max_jobs}\n"
        return res

    def __to_numpy__(self, memo):
        return self.__deepcopy__(memo)

    def __deepcopy__(self, memo):
        dont_copy = ["writer", "fig"]
        # now deepcopy the object
        from copy import deepcopy
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

    def p_make_figure(**kwargs):
        """
        Parallel function that gets a ModelPlotter object, makes the image 
        of it and puts it in the output queue.

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
        return