"""Model settings for the 2D shallow water model."""
from __future__ import annotations

import fridom.framework as fr
import fridom.shallowwater as sw


class ModelSettings(fr.ModelSettingsBase):

    """
    Model settings for the 2D shallow water model.

    Parameters
    ----------
    grid : Grid
        The grid object.

    """

    model_name = "ShallowWater"

    def __init__(self, grid: fr.grid.GridBase, **kwargs: any) -> None:
        super().__init__(grid)

        # TODO(Silvano): rename variables to meaningful names

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

    def setup_settings_parameters(self) -> None: # noqa: D102
        # Coriolis parameter
        f_coriolis = fr.ScalarField(self,
            name="f",
            long_name="Coriolis parameter",
            units="1/s",
            position=self.grid.cell_center,
            topo=(True, True),  # TODO(Silvano): don't need topo in x
        )
        self._f_coriolis = f_coriolis
        self._update_coriolis()

        # Speed of waves squared
        csqr_field = fr.ScalarField(
            self,
            name="csqr",
            long_name="Speed of waves squared",
            units="m²/s²",
            position=self.grid.cell_center,
        )
        self._csqr_field = csqr_field + self._csqr

        # make sure that the advection term is scaled by the Rossby number
        self.tendencies.advection.scaling = self.Ro

    def state_constructor(self) -> sw.State:  # noqa: D102
        return sw.State(self, is_spectral=self.grid.spectral_grid)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def parameters(self) -> dict:  # noqa: D102
        res = super().parameters
        res["coriolis parameter f0"] = f"{self.f0} s⁻¹"
        res["beta term"] = f"{self.beta} m⁻¹ s⁻¹)"
        res["Phase velocity c²"] = f"{self._csqr} m²s⁻²"
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
        r"""
        The beta term of the Coriolis parameter (f=f0 + beta*y).

        Unscaled System
        ---------------
        For the unscaled system, the beta term is given by:

        .. math::
            \beta = \frac{df}{dy} = \frac{2\Omega \cos(\phi)}{R}

        where :math:`\Omega` is the angular velocity of the Earth, :math:`\phi`
        is the latitude, and :math:`R` is the radius of the Earth.

        Scaled System
        -------------
        In the scaled system, the beta term is given by:

        .. math::
            \beta = \frac{\beta' L}{\Omega}

        where :math:`\beta'` is the unscaled beta term, :math:`L` is the
        domain extent in the y-direction, and :math:`\Omega` is the angular
        velocity of the Earth. We can rewrite the domain extent in terms of
        the latitude extent :math:`\Delta \phi`:

        .. math::
            L = R \Delta \phi

        where :math:`R` is the radius of the Earth. Thus, the beta term in the
        scaled system is given by:

        .. math::
            \beta = 2 \cos(\phi) \Delta \phi

        """
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
        _x, y = self.f_coriolis.get_mesh()
        self._f_coriolis.arr = self.f0 + self.beta * y

    @property
    def csqr(self) -> float:
        """The phase speed of the gravity waves."""
        return self._csqr

    @csqr.setter
    def csqr(self, value: float) -> None:
        self._csqr = value
        if self._csqr_field is not None:
            self.csqr_field *= 0
            self.csqr_field += value

    @property
    def csqr_field(self) -> fr.ScalarField:
        """The variable c²(x,y) field."""
        return self._csqr_field

    @csqr_field.setter
    def csqr_field(self, value: fr.ScalarField) -> None:
        if not isinstance(value, fr.ScalarField):
            msg = "Field must be a ScalarField."
            raise TypeError(msg)
        self._csqr_field = value

    @property
    def Ro(self) -> float:
        """The Rossby number."""
        return self._Ro

    @Ro.setter
    def Ro(self, value: float) -> None:
        self._Ro = value
        # scale the advection term
        self.tendencies.advection.scaling = value
