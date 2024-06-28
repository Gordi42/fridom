# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.to_numpy import to_numpy
from fridom.framework.modules.module import \
    Module, start_module, stop_module, update_module
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
    `max_jobs` : `float`, optional (default=0.4)
        The maximum fraction of the available threads that will be used.
    
    Methods
    -------
    `start()`
        Start the video writing process.
    `stop()`
        Stop the video writing process.
    `update(mz, dz)`
        Add a new frame to the video.
    `show_video(width)`
        Show the video in the Jupyter notebook.
    
    Examples
    --------
    >>> TODO: add example from nonhydrostatic model
    """
    def __init__(self, 
                 model_plotter: 'ModelPlotterBase', 
                 interval: int=50,
                 filename: str="output.mp4", 
                 fps: int=30,
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
        return

    @start_module
    def start(self):
        """
        Method to start the writer process.
        """
        import os

        # create video folder if it does not exist
        if not os.path.exists("videos"):
            os.makedirs("videos")

        # delete the file if it already exists
        if os.path.exists(self.filename):
            os.remove(self.filename)

        # list for the jobs and queues for creating the figures
        self.running_jobs = []       # Processes
        self.open_queues  = []       # Queues

        # use maximum of 40% the available threads
        import multiprocessing as mp
        self.maximum_jobs = int(self.max_jobs*mp.cpu_count())

        # start the writer
        import imageio
        self.writer = imageio.get_writer(self.filename, fps=self.fps)
        return

    @stop_module
    def stop(self):
        """
        Method to stop the writer process.
        """
        # collect all figures
        while len(self.running_jobs) > 0:
            self.collect_figures()
        self.writer.close()
        return

    @update_module
    def update(self, mz: 'ModelState', dz: 'StateBase'):
        """
        Update method of the parallel animated model.
        """
        # check if its time to update the plot
        if mz.it % self.interval != 0:
            return

        # collect finished figures
        self.collect_figures()

        # wait until there is space for a new job
        while len(self.running_jobs) >= self.maximum_jobs:
            self.collect_figures()

        # create a new figure
        import multiprocessing as mp
        q = mp.Queue()
        kw = {"mz": to_numpy(mz), "output_queue": q, "model_plotter": self.model_plotter}
        job = mp.Process(target=VideoWriter.p_make_figure, kwargs=kw)
        job.start()

        self.open_queues.append(q)
        self.running_jobs.append(job)
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
        fig = model_plotter.create_figure()
        model_plotter.update_figure(fig=fig, mz=kwargs["mz"])

        img = model_plotter.convert_to_img(fig)
        output_queue.put(img)
        return