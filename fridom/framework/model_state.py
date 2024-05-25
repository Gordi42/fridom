from fridom.framework.state_base import StateBase
from fridom.framework.grid_base import GridBase

class ModelStateBase:
    """
    The base class for model states. It contains the state vector, the time step
    and the model time. Child classes may add more attributes as for example the
    diagnostic variables needed for the model.

    All model state variables should be stored in this class.

    ## Attributes
    - z (StateBase)   : State vector
    - it (int)        : Time step
    - time (float)    : Model time
    """
    def __init__(self, grid: GridBase, is_spectral: bool) -> None:
        """
        The base constructor for the ModelStateBase class.
        """
        self.z: StateBase = None
        self.it = 0
        self.time = 0.0

# remove symbols from the namespace
del StateBase, GridBase