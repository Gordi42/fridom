"""model_plotter.py - A module for creating and updating a figure object"""
import numpy as np
import fridom.framework as fr

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
    def __new__(cls, mz: 'fr.ModelState'):
        fig = cls.create_figure()
        cls.update_figure(fig, **cls.prepare_arguments(mz))
        return fig

    @staticmethod
    def create_figure():
        """
        This method should create a figure object 
        (e.g. matplotlib figure) and return it.
        """
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        return plt.figure()

    @staticmethod
    def prepare_arguments(mz: 'fr.ModelState') -> dict:
        """
        This method should prepare the arguments for the update_figure method.
        """
        raise NotImplementedError

    @staticmethod
    def update_figure(fig, *args, **kwargs) -> None:
        """
        This method should update the figure object with the
        given model state.
        """
        raise NotImplementedError

    @staticmethod
    def convert_to_img(fig):
        """
        This method should convert the figure object to a numpy image array.
        """
        # first we draw the figure
        fig.canvas.draw()
        # access the renderer
        renderer = fig.canvas.get_renderer()
        # get the rgba buffer from the renderer
        rgba_buffer = renderer.buffer_rgba()
        # convert the rgba buffer to a numpy array
        img = np.array(rgba_buffer)
        # return the image
        return img
