"""Finite difference differentiation module for Cartesian grids."""
from __future__ import annotations

from copy import deepcopy
from functools import partial

import fridom.framework as fr

ncp = fr.config.ncp

@partial(fr.utils.jaxify, dynamic=("_dx1", ))
class FiniteDifferences(fr.grid.DiffModule):

    """
    Finite difference differentiation for Cartesian grids.

    Description
    -----------
    If a field is defined at the cell center, the field is differentiated using
    a forward difference, and the resulting field is defined at the cell face.
    If a field is defined at the cell face, the field is differentiated using
    a backward difference, and the resulting field is defined at the cell center.

    """

    name = "Finite Differences"
    def __init__(self) -> None:
        super().__init__()
        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.required_halo = 1
        self._dx1 = None

    def _on_setup(self) -> None:
        if not isinstance(self.mset.grid, fr.grid.cartesian.Grid):
            msg = "Finite differences only work with Cartesian grids."
            raise TypeError(msg)

        self._dx1 = 1 / ncp.array(self.mset.grid.dx, dtype=fr.config.dtype_real)

    def diff(self,  # noqa: D102
             f: fr.ScalarField,
             axis: int) -> fr.ScalarField:

        destination = f.position.shift(axis)

        view = fr.grid.Stencil(
            grid=self.grid, size=2, offset=0, destination=destination[axis],
        ).view(f.arr, axis=axis)

        diff = (view[1] - view[0]) * self._dx1[axis]

        # update the metadata
        mdata = deepcopy(f.mdata)
        mdata.position = destination

        return fr.ScalarField(mset=f.mset, mdata=mdata, arr=diff)
