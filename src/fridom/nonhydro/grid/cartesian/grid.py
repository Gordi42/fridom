from numpy import ndarray
import fridom.nonhydro as nh
import fridom.framework as fr


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):
    def __init__(self, 
                 N: list[int],
                 L: list[float],
                 periodic_bounds: list[bool] | None = None,
                 domain_decomp: fr.domain_decomposition.DomainDecomposition | None = None,
                 diff_mod: fr.grid.DiffModule | None = None,
                 interp_mod: fr.grid.InterpolationModule | None = None
                 ) -> None:
        super().__init__(N=N,
                         L=L,
                         periodic_bounds=periodic_bounds,
                         domain_decomp=domain_decomp,
                         diff_mod=diff_mod,
                         interp_mod=interp_mod)

    def omega(self, 
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False
              ) -> ndarray:
        return nh.grid.cartesian.eigenvectors.omega(
            mset=self.mset, s=1, k=k, use_discrete=use_discrete)

    def vec_q(self, s: int, use_discrete=True) -> nh.State:
        return nh.grid.cartesian.eigenvectors.vec_q(
            mset=self.mset, s=s, use_discrete=use_discrete)

    def vec_p(self, s: int, use_discrete=True) -> nh.State:
        return nh.grid.cartesian.eigenvectors.vec_p(
            mset=self.mset, s=s, use_discrete=use_discrete)
