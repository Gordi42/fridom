from fridom.framework.projection_base import \
    GeostrophicSpectralBase, Projection, WaveSpectralBase, \
    DivergenceSpectralBase, GeostrophicTimeAverageBase
from fridom.framework.optimal_balance_base import OptimalBalanceBase
from fridom.framework.nnmd_base import NNMDBase

from fridom.shallowwater.grid import Grid


class GeostrophicSpectral(GeostrophicSpectralBase):
    def __init__(self, grid: Grid) -> None:
        from fridom.shallowwater.eigenvectors import VecP, VecQ
        super().__init__(grid, VecQ, VecP)

class WaveSpectral(WaveSpectralBase):
    def __init__(self, grid: Grid) -> None:
        from fridom.shallowwater.eigenvectors import VecP, VecQ
        super().__init__(grid, VecQ, VecP)

class DivergenceSpectral(DivergenceSpectralBase):
    def __init__(self, grid: Grid) -> None:
        from fridom.shallowwater.eigenvectors import VecP, VecQ
        super().__init__(grid, VecQ, VecP)

class GeostrophicTimeAverage(GeostrophicTimeAverageBase):
    """
    Geostrophic projection using time-averaging.
    """
    def __init__(self, grid: Grid, 
                 n_ave=4,
                 equidistant_chunks=True,
                 max_period=None,
                 backward_forward=False
                 ) -> None:
        """
        Geostrophic projection using time-averaging.

        Arguments:
            grid      (Grid)          : The grid.
            n_ave             (int)   : Number of averages to perform.
            equidistant_chunks(bool)  : Whether to split the averaging periods 
                                        into equidistant chunks.
            max_period        (float) : The maximum period of the time averages.
                                        if None, the maximum period is set to
                                        the inertial period.
            backward_forward  (bool)  : Whether to use backward-forward averaging.
        """
        from fridom.shallowwater.model import Model
        super().__init__(grid, Model, n_ave, 
                         equidistant_chunks, max_period, backward_forward)
        return

class OptimalBalance(OptimalBalanceBase):
    def __init__(self, grid: Grid, 
                 base_proj: Projection, 
                 ramp_period=1, ramp_type="exp", 
                 enable_forward_friction=False, 
                 enable_backward_friction=False, 
                 update_base_point=True,
                 max_it=3, stop_criterion=1e-9,
                 return_details=False) -> None:
        from fridom.shallowwater.model import Model
        super().__init__(grid=grid, 
                         Model=Model, 
                         base_proj=base_proj, 
                         ramp_period=ramp_period, 
                         ramp_type=ramp_type, 
                         enable_forward_friction=enable_forward_friction, 
                         enable_backward_friction=enable_backward_friction, 
                         update_base_point=update_base_point,
                         max_it=max_it, 
                         stop_criterion=stop_criterion, 
                         return_details=return_details)

class NNMD(NNMDBase):
    def __init__(self, grid: Grid, 
                 order=3, enable_dealiasing=True) -> None:
        from fridom.shallowwater.model import Model
        from fridom.shallowwater.eigenvectors import VecP, VecQ
        from fridom.shallowwater.state import State
        super().__init__(grid, Model, State, VecQ, VecP, order, enable_dealiasing)

# remove symbols from the namespace
del GeostrophicSpectralBase, Projection, WaveSpectralBase, \
    DivergenceSpectralBase, GeostrophicTimeAverageBase, OptimalBalanceBase, \
    NNMDBase, Grid