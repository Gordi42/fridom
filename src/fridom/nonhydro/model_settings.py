"""Model settings for the nonhydrostatic model."""
from __future__ import annotations

from functools import partial
from typing import Literal

import fridom.framework as fr
import fridom.nonhydro as nh


@partial(fr.utils.jaxify, dynamic=("f_coriolis", "N2", "dsqr", "Ro"))
class ModelSettings(fr.ModelSettingsBase):

    """
    Model settings for the 3D non-hydrostatic model.

    Parameters
    ----------
    grid : Grid
        The grid object.

    """

    model_name = "3D - Nonhydrostatic model"

    def __init__(self, grid: fr.grid.GridBase, **kwargs: any) -> None:
        super().__init__(grid)

        # TODO(Silvano): rename variables to meaningful names

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

    def setup_settings_parameters(self) -> None:  # noqa: D102
        # Coriolis parameter
        f_coriolis = fr.ScalarField(self,
            name="f",
            long_name="Coriolis parameter",
            units="1/s",
            position=self.grid.cell_center,
            topo=(True, True, True),  # TODO(Silvano): don't need topo in x and z
        )
        self._f_coriolis = f_coriolis
        self._update_coriolis()
        # Background stratification
        stratification = fr.ScalarField(self,
            name="N²",
            long_name="Stratification",
            units="1/s^2",
            position=self.grid.cell_center,
            topo=(True, True, True),  # TODO(Silvano): don't need topo in x and y
        )
        self._N2_field = stratification
        self.N2 = self.N2
        # make sure that the advection term is scaled by the Rossby number
        self.tendencies.advection.scaling = self.Ro

    def state_constructor(self) -> nh.State:  # noqa: D102
        return nh.State(self, is_spectral=self.grid.spectral_grid)

    def diagnostic_state_constructor(self) -> nh.DiagnosticState:  # noqa: D102
        return nh.DiagnosticState(self, is_spectral=self.grid.spectral_grid)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def parameters(self) -> dict:  # noqa: D102
        res = super().parameters
        res["coriolis parameter f0"] = f"{self.f0} 1/s"
        res["beta term"] = f"{self.beta} 1/(m*s)"
        res["Stratification N²"] = f"{self.N2} 1/s^2"
        res["Aspect ratio dsqr"] = f"{self.dsqr}"
        res["Rossby number Ro"] = f"{self.Ro}"
        return res

    @property
    def f0(self) -> float:
        """The constant term of the  Coriolis parameter (f=f0 + beta*y)."""
        return self._f0

    @f0.setter
    def f0(self, value: float) -> None:
        self._f0 = value
        self._update_coriolis()

    @property
    def beta(self) -> float:
        """The beta term of the Coriolis parameter (f=f0 + beta*y)."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = value
        self._update_coriolis()

    @property
    def f_coriolis(self) -> fr.ScalarField:
        """The Coriolis parameter (f=f0 + beta*y)."""
        return self._f_coriolis

    @f_coriolis.setter
    def f_coriolis(self, value: fr.ScalarField) -> None:
        if not isinstance(value, fr.ScalarField):
            msg = "The coriolis parameter must be a ScalarField."
            raise TypeError(msg)
        self._f_coriolis = value

    def _update_coriolis(self) -> None:
        if self._f_coriolis is None:
            # nothing to be updated if the coriolis parameter is not set
            return
        _x, y, _z = self.f_coriolis.get_mesh()
        self._f_coriolis.arr = self.f0 + self.beta * y

    @property
    def N2(self) -> float:
        """The stratification N²."""
        return self._N2

    @N2.setter
    def N2(self, value: float) -> None:
        # update the stratification field
        if self._N2_field is not None:
            x, y, z = self._N2_field.get_mesh()
            self._N2_field.arr = z*0 + value
        self._N2 = value

    @property
    def N2_field(self) -> fr.ScalarField:
        """The stratification N² field."""
        return self._N2_field

    @N2_field.setter
    def N2_field(self, value: fr.ScalarField) -> None:
        if not isinstance(value, fr.ScalarField):
            msg = "The stratification field must be a ScalarField."
            raise TypeError(msg)
        self._N2_field = value

    @property
    def Ro(self) -> float:
        """The Rossby number."""
        return self._Ro

    @Ro.setter
    def Ro(self, value: float) -> None:
        self._Ro = value
        # scale the advection term
        self.tendencies.advection.scaling = value

    @property
    def dsqr(self) -> float:
        r"""The aspect ratio. :math:`\delta^2`."""
        return self._dsqr

    @dsqr.setter
    def dsqr(self, value: float) -> None:
        self._dsqr = value
