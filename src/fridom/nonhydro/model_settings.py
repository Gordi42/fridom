# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.field_variable import FieldVariable
from fridom.framework.model_settings_base import ModelSettingsBase
# Import type information
if TYPE_CHECKING:
    from fridom.framework.grid.grid_base import GridBase
    from numpy import ndarray

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

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.model_name = "3D - Nonhydrostatic model"
        self._f0 = dtype(1)
        self._beta = dtype(0)
        self._f_coriolis   = None
        self.N2   = dtype(1)
        self.dsqr = dtype(1)
        self.Ro   = dtype(1)
        self.tendencies = tendencies
        self.bc   = bc

        self.set_attributes(**kwargs)
        return

    def setup(self):
        super().setup()

        # Coriolis parameter
        f_coriolis = FieldVariable(
            self, 
            name="f",
            long_name="Coriolis parameter",
            units="1/s",
            position=self.grid.cell_center,
            topo=[False, True, False],
        )
        f_coriolis[:] = self._f0 + self._beta * self.grid.X[1][None,0,:,0,None]
        self._f_coriolis = f_coriolis
        return

    def state_constructor(self):
        from fridom.nonhydro.state import State
        return State(self, is_spectral=False)

    def diagnostic_state_constructor(self):
        from fridom.nonhydro.diagnostic_state import DiagnosticState
        return DiagnosticState(self, is_spectral=False)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def parameters(self) -> dict:
        res = super().parameters
        res["coriolis parameter f0"] = f"{self.f0} 1/s"
        res["beta term"] = f"{self.beta} 1/(m*s)"
        res["Stratification N²"] = f"{self.N2} 1/s^2"
        res["Aspect ratio dsqr"] = f"{self.dsqr}"
        res["Rossby number Ro"] = f"{self.Ro}"
        return res

    @property
    def f0(self) -> 'float':
        """The constant term of the  Coriolis parameter (f=f0 + beta*y)."""
        return self._f0
    
    @f0.setter
    def f0(self, value: 'float'):
        self._f0 = value
        if self._f_coriolis is not None:
            self.f_coriolis[:] = self._f0 + self._beta * self.grid.X[1][None,:,None]
        return

    @property
    def beta(self) -> 'float':
        """The beta term of the Coriolis parameter (f=f0 + beta*y)."""
        return self._beta
    
    @beta.setter
    def beta(self, value: 'float'):
        self._beta = value
        if self._f_coriolis is not None:
            self.f_coriolis[:] = self._f0 + self._beta * self.grid.X[1][None,:,None]
        return

    @property
    def f_coriolis(self) -> 'FieldVariable':
        """The Coriolis parameter (f=f0 + beta*y)."""
        return self._f_coriolis
    
    @f_coriolis.setter
    def f_coriolis(self, value: 'FieldVariable | float | ndarray'):
        if isinstance(value, FieldVariable):
            self._f_coriolis = value
        else:
            self._f_coriolis[:] = value
        return


utils.jaxify_class(ModelSettings)
