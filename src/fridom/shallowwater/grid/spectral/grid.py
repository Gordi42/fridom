"""Spectral grid for the shallow water model."""

from __future__ import annotations

from numpy import ndarray

import fridom.framework as fr
import fridom.shallowwater as sw


@fr.utils.jaxify
class Grid(fr.grid.spectral.Grid):

    def __init__(self, N: list[int], L: list[int], *args: any, **kwargs: dict) -> None:

        if "periodic_bounds" in kwargs and not all(kwargs["periodic_bounds"]):
            msg = "Only periodic boundaries are supported in the spectral grid."
            raise ValueError(msg)

        super().__init__(N=N, L=L, periodic_bounds=[True, True])

    def omega(self,
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False,
              ) -> ndarray:
        # always use non-discrete eigenvalues even if use_discrete is True
        return sw.grid.cartesian.eigenvectors.omega(
            mset=self.mset, s=1, k=k, use_discrete=False)

    def vec_q(self, s: int, use_discrete=False) -> sw.State:
        # always use non-discrete eigenvectors even if use_discrete is True
        return sw.grid.cartesian.eigenvectors.vec_q(
            mset=self.mset, s=s, use_discrete=False)

    def vec_p(self, s: int, use_discrete=False) -> sw.State:
        # always use non-discrete eigenvectors even if use_discrete is True
        return sw.grid.cartesian.eigenvectors.vec_p(
            mset=self.mset, s=s, use_discrete=False)
