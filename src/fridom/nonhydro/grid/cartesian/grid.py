"""Cartesian grid for the nonhydrostatic model."""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import fridom.framework as fr
import fridom.nonhydro as nh

if TYPE_CHECKING:
    from numpy import ndarray


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):

    """Cartesian grid of the nonhydrostatic model."""

    def __init__(self,
                 shape: list[int],
                 domain_size: list[float],
                 periodic_bounds: list[bool] | None = None,
                 domain_decomp: (
                     fr.domain_decomposition.DomainDecomposition | None
                 ) = None,
                 diff_mod: fr.grid.DiffModule | None = None,
                 interp_mod: fr.grid.InterpolationModule | None = None
                 ) -> None:
        super().__init__(shape=shape,
                         domain_size=domain_size,
                         periodic_bounds=periodic_bounds,
                         domain_decomp=domain_decomp,
                         diff_mod=diff_mod,
                         interp_mod=interp_mod)

    def omega(self,
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False
              ) -> ndarray:
        """Return the eigenvalues (frequencies) of the system matrix."""
        return nh.grid.cartesian.eigenvectors.omega(
            s = 1, f0=self.mset.f0,
            stratification_n2=self.mset.stratification_n2,
            dsqr=self.mset.dsqr,
            k=k, dx=self.dx, use_discrete=use_discrete)

    @cache  # noqa: B019 (cache bound to grid lifetime)
    def vec_q(self, s: int, use_discrete: bool = True) -> nh.State:
        """Return the eigenvectors of the system matrix."""
        return nh.grid.cartesian.eigenvectors.vec_q(
            mset=self.mset, s=s, use_discrete=use_discrete)

    @cache  # noqa: B019 (cache bound to grid lifetime)
    def vec_p(self, s: int, use_discrete: bool = True) -> nh.State:
        """Return the projection vectors of the system matrix."""
        return nh.grid.cartesian.eigenvectors.vec_p(
            mset=self.mset, s=s, use_discrete=use_discrete)
