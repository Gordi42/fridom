"""Base class for advection schemes."""
from __future__ import annotations

from abc import abstractmethod
from functools import partial

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("_scaling", "_background"))
class AdvectionBase(fr.modules.Module):

    r"""
    Base class for advection schemes.

    Description
    -----------
    This class implements the base interface for 1D, 2D, and 3D advection
    schemes. For that, it is assumed that the velocity field is stored in
    the state vector as the components `u`, `v`, and `w`. Child classes
    must implement the `advection` method to calculate the advection term:

    .. math::
        \mathcal{A}(\boldsymbol{v}, q) = -\boldsymbol{v} \cdot \nabla q

    where :math:`q` is the quantity to be advected and
    :math:`\boldsymbol{v}` is the velocity field, which is the sum of the
    velocity field in the state vector and the background velocity field,
    stored in the `background` attribute.
    This update routine of this module adds the advection term multiplied
    by the nonlinear scaling factor to the tendency term of all fields
    that are not flagged with `NO_ADV`:

    .. math::
        \partial_t q \leftarrow \partial_t q
                            + \rho \mathcal{A}(\boldsymbol{v}, q)

    where :math:`\rho` is the nonlinear scaling factor.
    """

    name = "Advection Base"

    def __init__(self) -> None:
        super().__init__()
        self._scaling = 1
        self._background = None
        self._disable_nonlinear = False

    @abstractmethod
    def advection(self,
                  velocity: fr.VectorField,
                  quantity: fr.ScalarField) -> fr.ScalarField:
        """Advect a quantity using the given velocity field."""

    def advect_state(
            self, z: fr.VectorField, dz: fr.VectorField) -> fr.VectorField:
        """Advect the state vector."""
        if self.background is None and self.disable_nonlinear:
            return dz
        if self.disable_nonlinear:
            zf = self.background
        else:
            # Compute the full state vector (including the background)
            zf = z if self.background is None else z + self.background

        # calculate the advection term
        for field in z:
            if field.flags["NO_ADV"]:
                continue
            dz[field.name] += self.scaling * self.advection(zf.velocity, field)
        return dz

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = self.advect_state(mz.z, mz.dz)
        return mz

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def disable_nonlinear(self) -> bool:
        """
        Whether to disable advection by the state vector itself.

        Advection by the background state is still enabled.
        """
        return self._disable_nonlinear

    @disable_nonlinear.setter
    def disable_nonlinear(self, value: bool) -> None:
        self._disable_nonlinear = value

    @property
    def scaling(self) -> float:
        """
        A scaling factor for the nonlinear terms (default: 1.0).

        Description
        -----------
        Some modules require to scale the nonlinear terms, as for example the
        optimal balance projection
        (:py:class:`fridom.framework.projection.OptimalBalance`). This
        parameter provides an interface to set this scaling factor.
        """
        return self._scaling

    @scaling.setter
    def scaling(self, value: float) -> None:
        self._scaling = value

    @property
    def background(self) -> fr.VectorField:
        """The background state."""
        return self._background

    @background.setter
    def background(self, value: fr.VectorField) -> None:
        self._background = value
