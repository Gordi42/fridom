# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.model_settings_base import ModelSettingsBase
# Import type information
if TYPE_CHECKING:
    from fridom.framework.grid.grid_base import GridBase

class ModelSettings(ModelSettingsBase):
    """
    Model settings for the 3D non-hydrostatic model.
    
    Parameters
    ----------
    `grid` : `Grid`
        The grid object.
    
    Attributes
    ----------
    `f_coriolis` : `float | np.ndarray`
        Coriolis parameter.
    `N2` : `float | np.ndarray`
        Background stratification (squara of Brunt-Vaisala frequency)
    `dsqr` : `float`
        Square of aspect ratio.
    `Ro` : `float`
        Rossby number.
    """
    _dynamic_attributes = ModelSettingsBase._dynamic_attributes + [
        "f_coriolis", "N2", "dsqr", "Ro" ]
    def __init__(self, grid: 'GridBase', **kwargs):
        super().__init__(grid)
        dtype = config.dtype_real


        # main tendency
        from fridom.nonhydro.modules.main_tendency import MainTendency
        tendencies = MainTendency()

        # boundary condition
        from fridom.nonhydro.modules.boundary_conditions import BoundaryConditions
        bc = BoundaryConditions()

        # state constructor
        from fridom.nonhydro.state import State
        state_constructor = lambda: State(self, is_spectral=False)
        from fridom.nonhydro.diagnostic_state import DiagnosticState
        diag_state_constructor = lambda: DiagnosticState(self, is_spectral=False)

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.model_name = "3D - Nonhydrostatic model"
        self.f_coriolis   = dtype(1)
        self.N2   = dtype(1)
        self.dsqr = dtype(1)
        self.Ro   = dtype(1)
        self.tendencies = tendencies
        self.bc   = bc
        self.state_constructor = state_constructor
        self.diagnostic_state_constructor = diag_state_constructor

        self.set_attributes(**kwargs)
        return

    @property
    def parameters(self) -> dict:
        res = super().parameters
        res["coriolis parameter f"] = f"{self.f_coriolis} 1/s"
        res["Stratification N²"] = f"{self.N2} 1/s^2"
        res["Aspect ratio dsqr"] = f"{self.dsqr}"
        res["Rossby number Ro"] = f"{self.Ro}"
        return res


utils.jaxify_class(ModelSettings)
