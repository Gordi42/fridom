"""Base class for interpolation methods."""
from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy

import fridom.framework as fr

ncp = fr.config.ncp

@fr.utils.jaxify
class InterpolationModule(fr.modules.Module):

    """
    The base class for interpolation methods.

    Description
    -----------
    An interpolation module is a class that interpolates a field from one position
    to another. For example, from the cell face to the cell center.
    """

    name = "Interpolation Module"
    _is_mod_submodule = True

    @abstractmethod
    def _interpolate_axis(self,
                          x: ncp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> ncp.ndarray:
        """Interpolate the field along a single axis."""
        raise NotImplementedError

    def interpolate(self,
                    f: fr.ScalarField,
                    destination: fr.grid.Position) -> fr.ScalarField:
        """
        Interpolate the field to the destination position.

        Parameters
        ----------
        f : fr.ScalarField
            The field to interpolate.
        destination : fr.grid.Position
            The position to interpolate to.

        Returns
        -------
        fr.ScalarField
            The interpolated field.

        """
        arr = f.arr

        for axis in range(f.arr.ndim):
            if f.position[axis] == destination.positions[axis]:
                # no interpolation needed
                continue
            if not f.topo[axis]:
                # no interpolation when the field has no extend along the axis
                continue

            arr = self._interpolate_axis(arr, axis, destination.positions[axis])

        mdata = deepcopy(f.mdata)
        mdata.position = destination
        return fr.ScalarField(mset=f.mset, mdata=mdata, arr=arr)
