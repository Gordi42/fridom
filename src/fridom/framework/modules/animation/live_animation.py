# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.modules.module import \
    Module, start_module, stop_module, update_module
# Import type information
if TYPE_CHECKING:
    from .model_plotter import ModelPlotterBase
    from fridom.framework.state_base import StateBase
    from fridom.framework.model_state import ModelState

class LiveAnimation(Module):
    def __init__(self, 
                 model_plotter: 'ModelPlotterBase',
                 interval: int = 50,
                 ) -> None:
        super().__init__(
            name="Live Animation",
            model_plotter=model_plotter,
            interval=interval)
        self.fig = None
        return

    @start_module
    def start(self) -> None:
        self.fig = self.model_plotter.create_figure()

    @update_module
    def update(self, mz: 'ModelState', dz: 'StateBase'):
        # check if its time to update the plot
        if mz.it % self.interval != 0:
            return

        # first clear the figure
        self.fig.clf()
        # update the figure
        self.model_plotter.update_figure(fig=self.fig, mz=mz)
        # display the figure
        from IPython import display
        display.display(self.fig)
        # clear the output when the next figure is ready
        display.clear_output(wait=True)
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    Interval: {self.interval}\n"
        return res