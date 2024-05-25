from fridom.framework.state_base import StateBase

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
    def __init__(self, z: StateBase, it: int, time: float) -> None:
        """
        The base constructor for the ModelStateBase class.
        """
        self.z = z
        self.it = it
        self.time = time

# remove symbols from the namespace
del StateBase