from fridom.framework.model_state import ModelStateBase

class ModelState(ModelStateBase):
    """
    Contains the model state of the shallow water model
    """
    def __init__(self, 
                 grid,
                 is_spectral: bool = False) -> None:
        """
        # Args:
        - grid (GridBase)       : Grid of the model.
        - is_spectral (bool)    : If the model is spectral or not.
        """
        super().__init__(grid, is_spectral)
        from fridom.shallowwater.state import State
        self.z = State(grid, is_spectral)
        return

    def cpu(self) -> 'ModelState':
        """
        If the model runs on the cpu, this function returns the object itself.
        If the model runs on the gpu, this function creates a copy of the model
        state on the cpu.
        """
        if self.z.grid.mset.gpu:
            mz = ModelState(self.z.grid, self.z.is_spectral)
            mz.z = self.z.cpu()
            mz.it = self.it
            mz.time = self.time
            return mz
        else:
            return self

# remove symbols from the namespace
del ModelStateBase