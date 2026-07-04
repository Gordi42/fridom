"""Model settings for the hydrostatic model."""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp

import fridom.framework as fr
import fridom.hydrostatic as hs


@partial(fr.utils.jaxify, dynamic=("coriolis_parameter",
                                   "background_stratification",
                                   "rossby_number"))
class ModelSettings(fr.ModelSettingsBase):

    """
    Model settings for the 3D hydrostatic model.

    Parameters
    ----------
    grid : Grid
        The grid object.
    coriolis_parameter : float | fr.ScalarField, optional
        The coriolis parameter. Can be a constant or a field (default: 0).
    background_stratification : float | fr.ScalarField, optional
        The background stratification. Also referred to as N² (default: 0).
    rossby_number : float, optional
        The Rossby number for scaling the nonlinearity (default: 1).

    """

    model_name = "3D - Hydrostatic model"

    def __init__(self, grid: fr.grid.GridBase, **kwargs: any) -> None:
        super().__init__(grid)

        # Set standard parameters
        self.tendencies = hs.modules.MainTendency()
        self.coriolis_parameter = 0
        self.background_stratification = 0
        self.rossby_number = 1

        # Finally, set attributes from keyword arguments
        self.set_attributes(**kwargs)

    def setup_settings_parameters(self) -> None:  # noqa: D102
        # This will make sure that the coriolis parameter is a scalar field
        self.coriolis_parameter = self.coriolis_parameter
        # Setup the background stratification
        self.background_stratification = self.background_stratification

    def state_constructor(self) -> hs.State:  # noqa: D102
        return hs.State(self, is_spectral=self.grid.spectral_grid)

    def diagnostic_state_constructor(self) -> None:  # noqa: D102
        return hs.DiagnosticState(self, is_spectral=self.grid.spectral_grid)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def parameters(self) -> dict:  # noqa: D102
        res = super().parameters
        res["Coriolis parameter"] = self._format_coriolis_parameter()
        res["Background stratification"] = (
            self._format_background_stratification())
        res["Rossby number"] = f"{self.rossby_number}"
        return res

    @property
    def coriolis_parameter(self) -> float | fr.ScalarField:
        r"""
        The Coriolis parameter.

        Description
        -----------
        The coriolis parameter is given by:

        .. math::
            f = 2 \Omega \sin(\phi)

        where :math:`\Omega` is the Earth's rotation rate and :math:`\phi`
        is the latitude. The Coriolis parameter is used to account for the
        Coriolis force in the momentum equations. A typical value for
        mid-latitudes is

        .. math::
            f_0 = 10^{-4} \, \text{s}^{-1}

        """
        return self._coriolis_parameter

    @coriolis_parameter.setter
    def coriolis_parameter(self, value: float | fr.ScalarField) -> None:
        # We need to make sure that the coriolis parameter is a scalar field
        if isinstance(value, fr.ScalarField):
            self._coriolis_parameter = value
            return
        # If the model is not yet setup, we cannot create a scalar field
        if not self.is_setup:
            self._coriolis_parameter = value
            return
        # If the value is a float, and we already have a scalar field,
        # we update it
        if ( hasattr(self, "_coriolis_parameter") and
             isinstance(self._coriolis_parameter, fr.ScalarField) ):
            self._coriolis_parameter.arr = jnp.full_like(
                self._coriolis_parameter.arr, value)
            return
        # Else we have to create a new scalar field
        coriolis_parameter = fr.ScalarField(self,
            name="f",
            long_name="Coriolis parameter",
            units="1/s",
            position=self.grid.cell_center,
            # TODO(Silvano): don't need topo in x and z
            topo=(True, True, True),
        )
        coriolis_parameter += value
        self._coriolis_parameter = coriolis_parameter

    def _format_coriolis_parameter(self) -> str:
        coriolis_parameter = self.coriolis_parameter
        if not isinstance(coriolis_parameter, fr.ScalarField):
            return f"{coriolis_parameter} 1/s"

        # Check if the coriolis parameter is a constant scalar field
        if not coriolis_parameter.is_constant:
            return "Variable"

        return f"{coriolis_parameter.value} 1/s"

    @property
    def background_stratification(self) -> float | fr.ScalarField:
        r"""
        The background stratification.

        Description
        -----------
        The background stratification is given by:

        .. math::
            N^2 = -\frac{g}{\rho_0} \partial_z \rho_s = \partial_z b_s

        where :math:`g` is the gravity, :math:`\rho_0` is the reference
        density, and :math:`\rho_s` is the background density. The variable
        :math:`b_s` would
        correspond to the background buoyancy. A typical value for the
        background stratification in the ocean is

        .. math::
            N^2 = 2.5 \times 10^{-5} \, \text{s}^{-2}

        """
        return self._background_stratification

    @background_stratification.setter
    def background_stratification(self,
                                  value: float | fr.ScalarField) -> None:
        # We need to make sure that the background stratification is a
        # scalar field
        if isinstance(value, fr.ScalarField):
            self._background_stratification = value
            return
        # If the model is not yet setup, we cannot create a scalar field
        if not self.is_setup:
            self._background_stratification = value
            return
        # If the value is a float, and we already have a scalar field,
        # we update it
        if ( hasattr(self, "_background_stratification") and
             isinstance(self._background_stratification, fr.ScalarField) ):
            self._background_stratification.arr = jnp.full_like(
                self._background_stratification.arr, value)
            return
        # Else we have to create a new scalar field
        background_stratification = fr.ScalarField(self,
            name="N2",
            long_name="Background stratification",
            units="1/s^2",
            position=self.grid.cell_center.shift(axis=2),
            # TODO(Silvano): don't need topo in x and z
            topo=(True, True, True),
        )
        background_stratification += value
        self._background_stratification = background_stratification

    def _format_background_stratification(self) -> str:
        background_stratification = self.background_stratification
        if not isinstance(background_stratification, fr.ScalarField):
            return f"{background_stratification} 1/s^2"

        # Check if the background stratification is a constant scalar field
        if not background_stratification.is_constant:
            return "Variable"

        return f"{background_stratification.value} 1/s^2"

    @property
    def rossby_number(self) -> float:
        r"""
        The Rossby number.

        Description
        -----------
        The Rossby number is a dimensionless number that is used to scale the
        nonlinearity in the momentum equations. It is given by:

        .. math::
            Ro = \frac{U}{f L}

        where :math:`U` is the typical velocity, :math:`f` is the Coriolis
        parameter, and :math:`L` is the typical length scale. A typical
        value for the Rossby number in the ocean is typically much smaller
        than 1.

        """
        return self._rossby_number

    @rossby_number.setter
    def rossby_number(self, value: float) -> None:
        self._rossby_number = value
        # TODO(Silvano): scale the advection term
