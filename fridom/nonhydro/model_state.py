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

# remove symbols from the namespace
del ModelStateBase