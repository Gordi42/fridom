"""Cartesian grid for the shallow water model."""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom.framework as fr
import fridom.shallowwater as sw

if TYPE_CHECKING:  # pragma: no cover
    from numpy import ndarray


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):

    """Cartesian grid of the shallow water model."""

    def __init__(self, shape: list[int], domain_size: list[int],
                 periodic_bounds: list[bool] | None = None,
                 domain_decomp: (
                     fr.domain_decomposition.DomainDecomposition | None
                 ) = None,
                 ) -> None:
        if periodic_bounds is None:
            periodic_bounds = [True, True]
        super().__init__(shape, domain_size, periodic_bounds,
                         domain_decomp=domain_decomp)

    def omega(self,
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False
              ) -> ndarray:
        """Compute the eigenvalues of the system matrix."""
        return sw.grid.cartesian.eigenvectors.omega(
            mset=self.mset, s=1, k=k, use_discrete=use_discrete)

    def vec_q(self, s: int, use_discrete: bool = True) -> sw.State:
        """Compute the eigenvectors of the system matrix."""
        return sw.grid.cartesian.eigenvectors.vec_q(
            mset=self.mset, s=s, use_discrete=use_discrete)

    def vec_p(self, s: int, use_discrete: bool = True) -> sw.State:
        """Compute the projection vectors of the system matrix."""
        return sw.grid.cartesian.eigenvectors.vec_p(
            mset=self.mset, s=s, use_discrete=use_discrete)
