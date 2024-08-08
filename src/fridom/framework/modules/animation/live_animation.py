import fridom.framework as fr


class LiveAnimation(fr.modules.Module):
    """
    Create a live plot of the model that gets updated at regular intervals 
    during the simulation.
    
    Description
    -----------
    To create a live animation of the model, one must provide a `ModelPlotter`
    that will be used to create the figure. Note that live animations only work
    in Jupyter notebooks (no MPI support).

    .. warning::
        The live animation module clashes with the Progress bar module. Make
        sure to disable the progress bar module when using the live animation.
    
    Parameters
    ----------
    `model_plotter` : `ModelPlotterBase`
        The model plotter that will be used to create the figure.
    `interval` : `int`, optional (default=50)
        The interval (time steps) at which the plot will be updated.
    """
    name = "Live Animation"
    def __init__(self, 
                 model_plotter: fr.modules.animation.ModelPlotter,
                 interval: int = 50,
                 ) -> None:
        super().__init__()

        self.interval = interval
        self.model_plotter = model_plotter
        self.mpi_available = False
        self.fig = None
        return

    @fr.modules.module_method
    def start(self) -> None:
        """
        Initialize the figure.
        """
        self.fig = self.model_plotter.create_figure()

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        """
        Update the figure from the model state and display it.
        
        Parameters
        ----------
        `mz` : `ModelState`
            The model state to be used to update the figure.
        """
        # check if its time to update the plot
        if mz.it % self.interval != 0:
            return mz

        # first clear the figure
        self.fig.clf()
        # update the figure
        args = self.model_plotter.prepare_arguments(mz)
        self.model_plotter.update_figure(fig=self.fig, **args)
        # display the figure
        from IPython import display
        display.display(self.fig)
        # clear the output when the next figure is ready
        display.clear_output(wait=True)
        return mz

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    Interval: {self.interval}\n"
        return res