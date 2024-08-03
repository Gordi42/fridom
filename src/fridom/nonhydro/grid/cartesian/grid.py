from numpy import ndarray
import fridom.nonhydro as nh
import fridom.framework as fr

class Grid(fr.grid.cartesian.Grid):
    # update the list of dynamic attributes
    _dynamic_attributes = fr.grid.cartesian.Grid._dynamic_attributes + [
        'k2_hat', 'k2_hat_zero']
    
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

    def setup(self, mset: nh.ModelSettings):
        super().setup(mset)

        # discretized wave number squared

        from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
        k_dis = [dso.k_hat_squared(kx, dx, use_discrete=True)
                 for (kx,dx) in zip(self.K, self.dx)]

        # scaled discretized wave number squared
        k2_hat = k_dis[0] + k_dis[1] + k_dis[2] / mset.dsqr

        k2_hat_zero = fr.config.ncp.where(k2_hat == 0)
        fr.utils.modify_array(k2_hat, k2_hat_zero, 1)

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.k2_hat = k2_hat
        self.k2_hat_zero = k2_hat_zero
        return

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

fr.utils.jaxify_class(Grid)