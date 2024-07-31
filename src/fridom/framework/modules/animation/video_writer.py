# Import external modules
from typing import TYPE_CHECKING
import warnings
# Import internal modules
from fridom.framework import config
from fridom.framework.to_numpy import to_numpy
from fridom.framework.modules.module import Module, setup_module, module_method
# Import type information
if TYPE_CHECKING:
    from fridom.framework.modules.animation import ModelPlotterBase
    from fridom.framework.state_base import StateBase
    from fridom.framework.model_state import ModelState


class VideoWriter(Module):
    """
    Create a mp4 video from the model.
    
    Description
    -----------
    To create a mp4 video from the model, one must provide a `ModelPlotter`
    that will be used to create the figure. The video writer does not support
    MPI parallelism.
    
    Parameters
    ----------
    `model_plotter` : `ModelPlotterBase`
        The model plotter that will be used to create the figure.
    `interval` : `int`, optional (default=50)
        The interval (time steps) at which the plot will be updated.
    `filename` : `str`, optional (default="output.mp4")
        The filename of the video (will be stored in videos/filename).
    `fps` : `int`, optional (default=30)
        The frames per second of the video.
    'parallel' : `bool`, optional (default=True)
        If True, the video writer will use parallelism to create the video.
    `max_jobs` : `float`, optional (default=0.4)
        The maximum fraction of the available threads that will be used.
    
    Examples
    --------
    The following example shows how to create a video from a nonhydrostatic
    model using the SingleWave initial condition.

    .. code-block:: python

        import fridom.nonhydro as nh
        import numpy as np
        import matplotlib.pyplot as plt

        # create the video writer
        class MyPlotter(nh.modules.animation.ModelPlotterBase):
            def create_figure():
                return plt.figure(figsize=(8, 6), tight_layout=True, dpi=100)

            def update_figure(fig, mz: nh.ModelState) -> None:
                nh.Plot(mz.z.b).front(fig=fig)

        vid_writer = nh.modules.animation.VideoWriter(
            MyPlotter, interval=10, filename="single_wave.mp4", fps=30)

        # create the model
        grid = nh.grid.cartesian.Grid(
            N=[128]*3, L=[1]*3, periodic_bounds=(True, True, True))
        mset = nh.ModelSettings(grid=grid, dsqr=0.02, Ro=0.0)
        mset.time_stepper.dt = np.timedelta64(10, 'ms')
        # add the video writer
        mset.diagnostics.add_module(vid_writer)
        mset.setup()
        z = nh.initial_conditions.SingleWave(mset, kx=2, ky=0, kz=1)
        model = nh.Model(mset)
        model.z = z
        model.run(runlen=np.timedelta64(10, 's'))
        # show the video
        vid_writer.show_video()

    """
    def __init__(self, 
                 model_plotter: 'ModelPlotterBase', 
                 interval: int=50,
                 filename: str="output.mp4", 
                 fps: int=30,
                 parallel: bool=False,
                 max_jobs: float=0.4,
                 name="Video Writer") -> None:
        import os
        filename = os.path.join("videos", filename)
        super().__init__(name=name, 
                         model_plotter=model_plotter,
                         interval=interval,
                         filename=filename,
                         fps=fps,
                         max_jobs=max_jobs)
        # set the flag for MPI availability
        self.mpi_available = False
        self.writer = None
        self.parallel = parallel
        self.fig = None
        return

    @setup_module
    def setup(self):
        import os
        # create video folder if it does not exist
        if not os.path.exists("videos"):
            config.logger.info("Creating videos folder")
            os.makedirs("videos")

        # delete the file if it already exists
        if os.path.exists(self.filename):
            config.logger.notice(f"Deleting existing video file {self.filename}")
            os.remove(self.filename)

        # use maximum of 40% the available threads
        if self.parallel:
            import multiprocessing as mp
            self.maximum_jobs = int(self.max_jobs*mp.cpu_count())
        self.fig = None
        return

    @module_method
    def start(self):
        """
        Method to start the writer process.
        """
        # list for the jobs and queues for creating the figures
        self.running_jobs = []       # Processes
        self.open_queues  = []       # Queues

        # start the writer
        if self.writer is not None and not self.writer.closed:
            config.logger.warning(
                "VideoWriter.start() called without closing the previous writer.",
                "Continue with the previous writer.")
        else:
            import imageio
            self.writer = imageio.get_writer(self.filename, fps=self.fps)
        return

    @module_method
    def stop(self):
        """
        Method to stop the writer process.
        """
        # collect all figures
        while len(self.running_jobs) > 0:
            config.logger.info("Collecting remaining figures")
            self.collect_figures()
        self.writer.close()
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
        return

    @module_method
    def update(self, mz: 'ModelState') -> 'ModelState':
        """
        Update method of the parallel animated model.
        """
        # check if its time to update the plot
        if mz.it % self.interval != 0:
            return mz

        if self.parallel:
            self.parallel_update(mz)
        else:
            self.single_update(mz)
        return mz

    def parallel_update(self, mz: 'ModelState'):
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

        kw = {"mz": to_numpy(mz), "output_queue": q, "model_plotter": self.model_plotter}
        job = mp.Process(target=VideoWriter.p_make_figure, kwargs=kw)
        self.mset.diagnostics = diagnostics

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            job.start()

        self.open_queues.append(q)
        self.running_jobs.append(job)
        return
    
    def single_update(self, mz: 'ModelState'):
        if self.fig is None:
            self.fig = self.model_plotter.create_figure()
        else:
            self.fig.clear()
        self.model_plotter.update_figure(fig=self.fig, mz=mz)
        img = self.model_plotter.convert_to_img(self.fig)
        self.writer.append_data(img)
        return

    def collect_figures(self):
        while len(self.running_jobs) > 0:
            try :
                img = self.open_queues[0].get(timeout=0.05)
            except:
                break

            # add the figure to the video
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
        config.set_backend(config.Backend.NUMPY, silent=True)
        # get output queue
        output_queue = kwargs["output_queue"]
        model_plotter = kwargs["model_plotter"]
        fig = model_plotter.create_figure()
        model_plotter.update_figure(fig=fig, mz=kwargs["mz"])

        img = model_plotter.convert_to_img(fig)
        output_queue.put(img)
        return