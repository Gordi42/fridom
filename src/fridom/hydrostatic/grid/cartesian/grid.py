"""Cartesian grid class for hydrostatic model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import fridom.framework as fr

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):

    """
    Cartesian grid with equidistant grid spacing for the hydrostatic model.

    Parameters
    ----------
    shape : tuple[int]
        The number of grid points in each direction.
    length : tuple[float]
        The length of the domain in each direction (in meters).
    periodic : tuple[bool] | None
        Whether the domain is periodic in each direction.

    """

    def __init__(self,
                 shape: tuple[int],
                 length: tuple[float],
                 periodic: tuple[bool] | None = None,
                 ) -> None:
        super().__init__(N=shape,
                         L=length,
                         periodic_bounds=periodic)

    def omega(self,
              kvec: tuple[float] | tuple[np.ndarray],
              method: Literal["discrete", "continuous"] = "discrete",
              ) -> np.ndarray:
        """
        Compute the dispersion relation of the model.

        Parameters
        ----------
        kvec : tuple[float] | tuple[np.ndarray]
            The wave vectors in the form (kx, ky, kz).
        method : Literal["discrete", "continuous"]
            Whether to include space-discretization effects in computation
            of the dispersion relation.

        Returns
        -------
        np.ndarray
            The dispersion relation omega(k).

        """
        raise NotImplementedError

    def vec_q(self,
              mode: int,
              method: Literal["discrete", "continuous"] = "discrete",
              ) -> fr.VectorField:
        """
        Compute the eigenvector of the linear operator of the given mode.

        Parameters
        ----------
        mode : int
            The mode (which eigenvalue / eigenvector to compute).
        method : Literal["discrete", "continuous"]
            Whether to include space-discretization effects in computation
            of the dispersion relation.

        Returns
        -------
        fr.VectorField
            The eigenvector of the linear operator.

        """
        raise NotImplementedError


    def vec_p(self,
              mode: int,
              method: Literal["discrete", "continuous"] = "discrete",
              ) -> fr.VectorField:
        """
        Compute the projection vector of the linear operator of the given mode.

        Parameters
        ----------
        mode : int
            The mode (which eigenvalue / eigenvector to compute).
        method : Literal["discrete", "continuous"]
            Whether to include space-discretization effects in computation
            of the dispersion relation.

        Returns
        -------
        fr.VectorField
            The eigenvector of the linear operator.

        """
        raise NotImplementedError
