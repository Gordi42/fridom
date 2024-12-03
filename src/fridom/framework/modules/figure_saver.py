import fridom.framework as fr
import os

class FigureSaver(fr.modules.Module):
    r"""
    Saves a figure created by a Plotter module to a file.

    Description
    -----------
    This module saves a figure created by a Plotter module to a file
    at a specified model time. The figure is saved in the 'figures'
    directory, if it does not exist, it is created.

    Parameters
    ----------
    `thumbnail_path` : `str`
        The path to the file where the figure should be saved.
    `model_time` : `float`
        The model time at which the figure should be saved.
    `plotter` : `ModelPlotter`
        The Plotter module that creates the figure.
    `dpi` : `int`
        The resolution of the figure in dots per inch.
    """
    def __init__(self, 
                 filename: str,
                 model_time: float,
                 plotter: fr.modules.animation.ModelPlotter,
                 dpi: int = 256):
        super().__init__()
        self.filename = filename
        self.model_time = model_time
        self.plotter = plotter
        self.dpi = dpi
        self._created = False
        self.mpi_available = False

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        if mz.clock.time < self.model_time:
            return mz
        if self._created:
            return mz
        os.makedirs("figures", exist_ok=True)
        self.plotter(mz).savefig(self.filename, dpi=self.dpi)
        self._created = True
        return mz