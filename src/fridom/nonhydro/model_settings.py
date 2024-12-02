from fridom.framework.grid.grid_base import GridBase
import fridom.nonhydro as nh
from functools import partial
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

@partial(utils.jaxify, dynamic=("f_coriolis", "N2", "dsqr", "Ro"))
class ModelSettings(ModelSettingsBase):
    """
    Model settings for the 3D non-hydrostatic model.
    
    Parameters
    ----------
    `grid` : `Grid`
        The grid object.
    """
    model_name = "3D - Nonhydrostatic model"

    def __init__(self, grid: GridBase, **kwargs) -> None:
        super().__init__(grid)
        # Set standard parameters
        self.tendencies = nh.modules.MainTendency()
        self._f0 = 1             # constant coriolis parameter f0
        self._beta = 0           # beta term d(f)/dy
        self._f_coriolis = None  # the coriolis parameter field
        self._N2 = 1             # stratification N²
        self._N2_field = None    # stratification N² field
        self._dsqr = 1           # aspect ratio
        self._Ro = 1             # Rossby number

        # Finally, set attributes from keyword arguments
        self.set_attributes(**kwargs)

    def setup_settings_parameters(self):
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
        # make sure that the advection term is scaled by the Rossby number
        self.tendencies.advection.scaling = self.Ro
        return

    def state_constructor(self):
        from fridom.nonhydro.state import State
        return State(self, is_spectral=self.grid.spectral_grid)

    def diagnostic_state_constructor(self):
        from fridom.nonhydro.state import DiagnosticState
        return DiagnosticState(self, is_spectral=self.grid.spectral_grid)

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

    @property
    def N2(self) -> 'float':
        """The stratification N²."""
        return self._N2
    
    @N2.setter
    def N2(self, value: 'float'):
        self._N2 = value
        return

    @property
    def Ro(self) -> 'float':
        """The Rossby number."""
        return self._Ro
    
    @Ro.setter
    def Ro(self, value: 'float'):
        self._Ro = value
        # scale the advection term
        self.tendencies.advection.scaling = value
        return

    @property
    def dsqr(self) -> 'float':
        r"""The aspect ratio. :math:`\delta^2`."""
        return self._dsqr
    
    @dsqr.setter
    def dsqr(self, value: 'float'):
        self._dsqr = value
        return
