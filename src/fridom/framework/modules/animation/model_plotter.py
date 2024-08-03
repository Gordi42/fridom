# Import external modules
from typing import TYPE_CHECKING
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState

class ModelPlotter:
    """
    A model plotter contains the logic for creating and updating a figure object
    
    Description
    -----------
    The Model Plotter must be overwriten by child classes to implement the
    following methods:

    `create_figure()`: 
        create a figure object (e.g. matplotlib figure)

    `prepare_arguments(mz: ModelState) -> dict`:
        prepare the arguments for the update_figure method (e.g. extract the
        field to be plotted and convert it to numpy/xarray)
    
    `update_figure(fig, **kwargs)`:
        update the figure object with the given arguments from the
        prepare_arguments method 
    
    `convert_to_img(fig)`:
        convert the figure object to a numpy image array. If matplotlib is used,
        this method does not need to be overwritten. However, if a different
        plotting library is used, this method must be overwritten.
    """
    def __init__(self):
        return

    def create_figure():
        """
        This method should create a figure object 
        (e.g. matplotlib figure) and return it.
        """
        import matplotlib.pyplot as plt
        return plt.figure()

    def prepare_arguments(mz: 'ModelState') -> dict:
        """
        This method should prepare the arguments for the update_figure method.
        """
        raise NotImplementedError

    def update_figure(fig, **kwargs) -> None:
        """
        This method should update the figure object with the
        given model state.
        """
        raise NotImplementedError

    def convert_to_img(fig):
        """
        This method should convert the figure object to a numpy image array.
        """
        import numpy as np
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer._renderer)
        return img