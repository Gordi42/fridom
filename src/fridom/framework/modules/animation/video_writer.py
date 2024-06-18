from fridom.framework.modules.module import \
    Module, start_module, stop_module, update_module
from fridom.framework.modules.animation import ModelPlotterBase
from fridom.framework.state_base import StateBase
from fridom.framework.model_state import ModelState


class VideoWriter(Module):
    """
    Module for creating a mp4 video from the model output.
    """
    def __init__(self, 
                 model_plotter:ModelPlotterBase, 
                 interval: int=50,
                 filename: str="output.mp4", 
                 fps: int=30,
                 max_jobs: float=0.4,
                 name="Video Writer") -> None:
        """
        Constructor of the parallel mp4 output writer module

        Arguments:
            model_plotter       : class with the plotting functions
            interval (int)      : interval between frames
            filename (str)      : filename of the video
            fps (int)           : frames per second of the video
            max_jobs (float)    : maximum fraction of the available threads
        """
        import os
        filename = os.path.join("videos", filename)
        super().__init__(name=name, 
                         model_plotter=model_plotter,
                         interval=interval,
                         filename=filename,
                         fps=fps,
                         max_jobs=max_jobs)
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
    def update(self, mz: ModelState, dz: StateBase):
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
        kw = {"mz": mz.cpu(), "output_queue": q, "model_plotter": self.model_plotter}
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

# remove symbols from namespace
del ModelPlotterBase, StateBase, ModelState, \
    Module, start_module, stop_module, update_module