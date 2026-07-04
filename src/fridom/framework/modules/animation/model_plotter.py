"""A module for creating and updating a figure object."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import fridom.framework as fr


class ModelPlotter:

    """
    A model plotter contains the logic to create and update a figure object.

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
        convert the figure object to a numpy image array. If matplotlib is
        used, this method does not need to be overwritten. However, if a
        different plotting library is used, this method must be overwritten.
    """

    def __new__(cls, mz: fr.ModelState) -> Any:
        """Create a figure and update it with the given model state."""
        fig = cls.create_figure()
        cls.update_figure(fig, **cls.prepare_arguments(mz))
        return fig

    @staticmethod
    def create_figure() -> Any:
        """Create a figure object (e.g. matplotlib figure) and return it."""
        import matplotlib.pyplot as plt  # noqa: PLC0415 (deferred import of optional/heavy dependency)
        return plt.figure()

    @staticmethod
    def prepare_arguments(mz: fr.ModelState) -> dict:
        """Prepare the arguments for the update_figure method."""
        raise NotImplementedError

    @staticmethod
    def update_figure(fig: Any, *args: Any, **kwargs: Any) -> None:
        """Update the figure object with the given model state."""
        raise NotImplementedError

    @staticmethod
    def convert_to_img(fig: Any) -> np.ndarray:
        """Convert the figure object to a numpy image array."""
        # first we draw the figure
        fig.canvas.draw()
        # access the renderer
        renderer = fig.canvas.get_renderer()
        # get the rgba buffer from the renderer
        rgba_buffer = renderer.buffer_rgba()
        # convert the rgba buffer to a numpy array
        return np.array(rgba_buffer)
        # return the image
