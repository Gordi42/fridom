from functools import cache

from numpy import ndarray

import fridom.framework as fr
import fridom.nonhydro as nh


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):
    def __init__(self,
                 shape: list[int],
                 domain_size: list[float],
                 periodic_bounds: list[bool] | None = None,
                 domain_decomp: fr.domain_decomposition.DomainDecomposition | None = None,
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
        return nh.grid.cartesian.eigenvectors.omega(
            s = 1, f0=self.mset.f0, stratification_n2=self.mset.stratification_n2, dsqr=self.mset.dsqr,
            k=k, dx=self.dx, use_discrete=use_discrete)

    @cache
    def vec_q(self, s: int, use_discrete=True) -> nh.State:
        return nh.grid.cartesian.eigenvectors.vec_q(
            mset=self.mset, s=s, use_discrete=use_discrete)

    @cache
    def vec_p(self, s: int, use_discrete=True) -> nh.State:
        return nh.grid.cartesian.eigenvectors.vec_p(
            mset=self.mset, s=s, use_discrete=use_discrete)
