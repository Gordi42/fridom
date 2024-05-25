from fridom.framework.model_state import ModelStateBase

class ModelPlotterBase:
    """
    Base class for the model plotter.

    Child classes should implement the following methods:
    - create_figure: create a figure object (e.g. matplotlib figure)
    - update_figure: update the figure object
    - convert_to_img: convert the figure object to a numpy image array
    """
    def create_figure():
        """
        This method should create a figure object 
        (e.g. matplotlib figure) and return it.
        """
        raise NotImplementedError

    def update_figure(fig, mz: ModelStateBase) -> None:
        """
        This method should update the figure object with the
        given model state.
        """
        raise NotImplementedError


    def convert_to_img(fig):
        """
        This method should convert the figure object to a numpy image array.
        """
        raise NotImplementedError

# remove symbols from namespace
del ModelStateBase