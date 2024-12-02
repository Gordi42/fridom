import fridom.framework as fr
import fridom.shallowwater as sw
import numpy as np


class ModelSettings(fr.ModelSettingsBase):
    """
    Model settings for the 2D shallow water model.
    
    Parameters
    ----------
    `grid` : `Grid`
        The grid object.
    """
    model_name = "ShallowWater"

    def __init__(self, grid: 'fr.grid.GridBase', **kwargs) -> None:
        super().__init__(grid)

        # Set standard parameters
        self.tendencies = sw.modules.MainTendency()
        self._f0 = 1             # constant coriolis parameter f0
        self._beta = 0           # beta term d(f)/dy
        self._f_coriolis = None  # the coriolis parameter field
        self._csqr = 1           # speed of waves squared
        self._csqr_field = None  # c² field (for varying depth)
        self._Ro = 1             # Rossby number

        # Finally, set attributes from keyword arguments
        self.set_attributes(**kwargs)

    def setup_settings_parameters(self):
        # Coriolis parameter
        f_coriolis = fr.FieldVariable(
            self, 
            name="f",
            long_name="Coriolis parameter",
            units="1/s",
            position=self.grid.cell_center,
            topo=[False, True],
        )
        f_coriolis.arr = self._f0 + self._beta * self.grid.X[1][None,0,:]
        self._f_coriolis = f_coriolis

        # Speed of waves squared
        csqr_field = fr.FieldVariable(
            self,
            name="csqr",
            long_name="Speed of waves squared",
            units="m²/s²",
            position=self.grid.cell_center,
        )
        csqr_field += self._csqr
        self._csqr_field = csqr_field
    
        # make sure that the advection term is scaled by the Rossby number
        self.tendencies.advection.scaling = self.Ro
        return

    def state_constructor(self):
        return sw.State(self, is_spectral=False)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def parameters(self) -> dict:
        res = super().parameters
        res["coriolis parameter f0"] = f"{self.f0} s⁻¹"
        res["beta term"] = f"{self.beta} m⁻¹ s⁻¹)"
        res["Phase velocity c²"] = f"{self._csqr} m²s⁻²"
        res["Rossby number Ro"] = f"{self.Ro}"
        return res

    @property
    def f0(self) -> 'float':
        """The constant term f0 of the  Coriolis parameter (f=f0 + beta*y)."""
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
    def f_coriolis(self) -> 'fr.FieldVariable':
        """The variable coriolis parameter field"""
        return self._f_coriolis
    
    @f_coriolis.setter
    def f_coriolis(self, value: 'fr.FieldVariable | float | np.ndarray'):
        if isinstance(value, fr.FieldVariable):
            self._f_coriolis = value
        else:
            self._f_coriolis[:] = value
        return

    @property
    def csqr(self) -> 'float':
        """The phase speed of the gravity waves."""
        return self._csqr
    
    @csqr.setter
    def csqr(self, value: 'float'):
        self._csqr = value
        if self._csqr_field is not None:
            self.csqr_field[:] = value
        return

    @property
    def csqr_field(self) -> 'fr.FieldVariable':
        """The variable c²(x,y) field."""
        return self._csqr_field

    @csqr_field.setter
    def csqr_field(self, value: 'fr.FieldVariable | float | np.ndarray'):
        if isinstance(value, fr.FieldVariable):
            self._csqr_field = value
        else:
            self._csqr_field[:] = value
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
