# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.modelsettings_base import ModelSettingsBase
# Import type information
if TYPE_CHECKING:
    from fridom.framework.grid.grid_base import GridBase

class ModelSettings(ModelSettingsBase):
    """
    Model settings for the 3D non-hydrostatic model.
    
    Description
    -----------
    
    
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
    def __init__(self, grid: 'GridBase', **kwargs):
        super().__init__(grid)
        dtype = config.dtype_real


        # main tendency
        from fridom.nonhydro.modules.main_tendency import MainTendency
        tendencies = MainTendency()

        # state constructor
        from fridom.nonhydro.state import State
        state_constructor = lambda: State(self, is_spectral=False)
        from fridom.nonhydro.diagnostic_state import DiagnosticState
        diag_state_constructor = lambda: DiagnosticState(self, is_spectral=False)

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.model_name = "NonHydrostatic"

        self.f_coriolis   = dtype(1)
        self.N2   = dtype(1)
        self.dsqr = dtype(1)
        self.Ro   = dtype(1)
        self.tendencies = tendencies
        self.state_constructor = state_constructor
        self.diagnostic_state_constructor = diag_state_constructor

        self.set_attributes(**kwargs)
        return


    def __str__(self) -> str:
        """
        String representation of the model settings.

        Returns:
            res (str): String representation of the model settings.
        """
        res = super().__str__()
        res += "  Physical parameters:\n"
        res += "    f    = {:.3f}\n".format(self.f_coriolis)
        res += "    N2   = {:.3f}\n".format(self.N2)
        res += "    dsqr = {:.3f}\n".format(self.dsqr)
        res += "    Ro   = {:.3f}\n".format(self.Ro)
        res += "================================================\n"
        return res

    def __repr__(self) -> str:
        """
        String representation of the model settings (for IPython).

        Returns:
            res (str): String representation of the model settings.
        """
        return self.__str__()