from numpy import ndarray
import fridom.nonhydro as nh
import fridom.framework as fr


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):
    def __init__(self, N: list[int], L: list[int],
                 periodic_bounds: list[bool] = [True, True, True],
                 decomposition: str = 'slab'):
        if decomposition == 'slab':
            shared_axes = [0, 1]
        elif decomposition == 'pencil':
            shared_axes = [0]
        else:
            raise ValueError(f"Unknown decomposition {decomposition}")
        super().__init__(N, L, periodic_bounds, shared_axes)

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
