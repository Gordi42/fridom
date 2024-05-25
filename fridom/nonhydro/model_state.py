from fridom.framework.model_state import ModelStateBase

class ModelState(ModelStateBase):
    """
    Contains the model state of the nonhydrostatic model 
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

        # Add State
        from fridom.nonhydro.state import State
        self.z = State(grid, is_spectral)

        # Add pressure and divergence variables
        from fridom.framework.field_variable import FieldVariable
        from fridom.nonhydro.boundary_conditions import PBoundary
        self.p = FieldVariable(grid, 
                    name="Pressure p", bc=PBoundary(grid.mset))
        self.div = FieldVariable(grid,
                    name="Divergence", bc=PBoundary(grid.mset))
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
            mz.p = self.p.cpu()
            mz.div = self.div.cpu()
            return mz
        else:
            return self

# remove symbols from the namespace
del ModelStateBase