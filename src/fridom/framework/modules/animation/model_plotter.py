# Import external modules
from typing import TYPE_CHECKING
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState

class ModelPlotterBase:
    """
    A model plotter contains the logic for creating and updating a figure object
    
    Description
    -----------
    The Model Plotter must be overwriten by child classes to implement the
    following methods:
    - create_figure: create a figure object (e.g. matplotlib figure)
    - update_figure: update the figure object
    By default, the model plotter base assumes that matplotlib is used for 
    plotting. However, when using a different plotting library, the user must
    overwrite the following method:
    - convert_to_img: convert the figure object to a numpy image array
    
    Methods
    -------
    `create_figure()`
        Create a figure object and return it.
    `update_figure(fig, mz)`
        Update the figure object with the given model state.
    `convert_to_img(fig)`
        Convert the figure object to a numpy image array.
    
    Examples
    --------
    >>> TODO: add example from nonhydrostatic model
    """
    def __init__(self):
        return

    def create_figure():
        """
        This method should create a figure object 
        (e.g. matplotlib figure) and return it.
        """
        raise NotImplementedError

    def update_figure(fig, mz: 'ModelState') -> None:
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