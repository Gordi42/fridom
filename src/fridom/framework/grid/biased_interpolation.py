"""Base class for biased interpolation methods."""
from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

import fridom.framework as fr

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp


@fr.utils.jaxify
class BiasedInterpolationModule(fr.modules.Module):

    """The base class for biased interpolation methods."""

    name = "Biased Interpolation Module"

    @abstractmethod
    def _interpolate_axis(self,
                          x: jnp.ndarray,
                          bias: jnp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> jnp.ndarray:
        """Interpolate the field along a single axis."""
        raise NotImplementedError

    def interpolate(self,
                    f: fr.ScalarField,
                    bias: fr.ScalarField,
                    destination: fr.grid.Position) -> fr.ScalarField:
        """
        Interpolate the field to the destination position.

        Parameters
        ----------
        f : fr.ScalarField
            The field to interpolate.
        bias : fr.ScalarField
            A helper field that determines the bias of the interpolation.
        destination : fr.grid.Position
            The position to interpolate to.

        Returns
        -------
        fr.ScalarField
            The interpolated field.

        """
        # check that the destination is only one axis different from the
        # field position
        diff_axes = [i for i in range(f.arr.ndim)
                     if f.position[i] != destination.positions[i]]

        if len(diff_axes) != 1:
            msg = ("Upwind interpolation can only be used to interpolate "
                   "between positions that differ along exactly one axis.")
            raise ValueError(msg)

        # check that the bias field has the same position as the destination
        if isinstance(bias, fr.ScalarField) and bias.position != destination:
            msg = ("The bias field must have the same position as the "
                   "destination position.")
            raise ValueError(msg)

        # only pass the array of the bias field to the interpolation function
        if isinstance(bias, fr.ScalarField):
            bias = bias.arr

        if f.position == destination:
            return f

        if not f.topo[diff_axes[0]]:
            # no interpolation when the field has no extend along the axis
            return f

        mdata = deepcopy(f.mdata)
        mdata.position = destination
        arr = self._interpolate_axis(f.arr, bias, axis=diff_axes[0],
                                     destination=destination.positions[diff_axes[0]])
        return fr.ScalarField(mset=f.mset, mdata=mdata, arr=arr)
