from fridom.framework.state_base import StateBase
from fridom.framework.grid_base import GridBase

class ModelState:
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
    def __init__(self, 
                 grid: GridBase, 
                 initialize_state = True) -> None:
        """
        The base constructor for the ModelStateBase class.
        """
        if initialize_state:
            self.z = grid.mset.state_constructor(grid)
            self.z_diag = grid.mset.diagnostic_state_constructor(grid)
        else:
            self.z: StateBase = None
            self.z_diag: StateBase = None
        self.it = 0
        self.time = 0.0

    def cpu(self) -> 'ModelState':
        """
        If the model runs on the cpu, this function returns the object itself.
        If the model runs on the gpu, this function creates a copy of the model
        state on the cpu.
        """
        if self.z.grid.mset.gpu:
            mz = ModelState(self.z.grid, False)
            mz.z = self.z.cpu()
            mz.z_diag = self.z_diag.cpu()
            mz.it = self.it
            mz.time = self.time
            return mz
        else:
            return self

# remove symbols from the namespace
del StateBase, GridBase