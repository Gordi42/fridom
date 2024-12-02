import fridom.framework as fr
import fridom.shallowwater as sw
from numpy import ndarray


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):
    def __init__(self, N: list[int], L: list[int],
                 periodic_bounds: list[bool] = [True, True]):
        super().__init__(N, L, periodic_bounds)

    def omega(self, 
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False
              ) -> ndarray:
        return sw.grid.cartesian.eigenvectors.omega(
            mset=self.mset, s=1, k=k, use_discrete=use_discrete)

    def vec_q(self, s: int, use_discrete=True) -> sw.State:
        return sw.grid.cartesian.eigenvectors.vec_q(
            mset=self.mset, s=s, use_discrete=use_discrete)

    def vec_p(self, s: int, use_discrete=True) -> sw.State:
        return sw.grid.cartesian.eigenvectors.vec_p(
            mset=self.mset, s=s, use_discrete=use_discrete)
