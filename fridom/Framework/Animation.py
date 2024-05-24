class ModelPlotterBase:
    def create_figure():
        return None

    def update_figure(fig):
        return None

    def convert_to_img(fig):
        return None


class LiveAnimation:
    def __init__(self, live_plotter:ModelPlotterBase) -> None:
        self.live_plotter = live_plotter
        self.fig = self.live_plotter.create_figure()

    def update(self, **kwargs):
        # first clear the figure
        self.fig.clf()
        # update the figure
        self.live_plotter.update_figure(fig=self.fig, **kwargs)
        # display the figure
        from IPython import display
        display.display(self.fig)
        # clear the output when the next figure is ready
        display.clear_output(wait=True)


class VideoAnimation:
    """
    Class for parallel animated plotting of the model.
    """
    def __init__(self, live_plotter:ModelPlotterBase, filename:str, fps:int,
                 max_jobs=0.4) -> None:
        """
        Constructor of the parallel animated model.

        Arguments:
            live_plotter        : class with the plotting functions
            filename (str)      : filename of the video
            fps (int)           : frames per second
            max_jobs (float)    : maximum fraction of the available threads
        """
        self.live_plotter = live_plotter
        self.fps = fps

        # create video folder if it does not exist
        import os
        if not os.path.exists("videos"):
            os.makedirs("videos")
        filename = os.path.join("videos", filename)
        # delete the file if it already exists
        if os.path.exists(filename):
            os.remove(filename)
        self.filename = filename

        # list for the jobs and queues for creating the figures
        self.running_jobs = []       # Processes
        self.open_queues  = []       # Queues

        # use maximum of 40% the available threads
        import multiprocessing as mp
        self.maximum_jobs = int(max_jobs*mp.cpu_count())
        return

    def start_writer(self):
        """
        Method to start the writer process.
        """
        import imageio
        self.writer = imageio.get_writer(self.filename, fps=self.fps)
        return

    def stop_writer(self):
        """
        Method to stop the writer process.
        """
        # collect all figures
        while len(self.running_jobs) > 0:
            self.collect_figures()
        self.writer.close()
        return

    def update(self, **kwargs):
        """
        Update method of the parallel animated model.
        """
        # collect finished figures
        self.collect_figures()

        # wait until there is space for a new job
        while len(self.running_jobs) >= self.maximum_jobs:
            self.collect_figures()

        # create a new figure
        import multiprocessing as mp
        q = mp.Queue()
        kw = kwargs.copy()
        kw["output_queue"] = q
        kw["model_plotter"] = self.live_plotter
        job = mp.Process(target=VideoAnimation.p_make_figure, kwargs=kw)
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
        model_plotter.update_figure(fig=fig, **kwargs)

        img = model_plotter.convert_to_img(fig)
        output_queue.put(img)
        return