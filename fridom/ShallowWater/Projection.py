from fridom.Framework.ProjectionBase import \
    GeostrophicSpectralBase, Projection, WaveSpectralBase, \
    DivergenceSpectralBase, GeostrophicTimeAverageBase
from fridom.Framework.OptimalBalanceBase import OptimalBalanceBase
from fridom.Framework.NNMDBase import NNMDBase

from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid


class GeostrophicSpectral(GeostrophicSpectralBase):
    def __init__(self, mset: ModelSettings, grid: Grid) -> None:
        from fridom.ShallowWater.Eigenvectors import VecP, VecQ
        super().__init__(mset, grid, VecQ, VecP)

class WaveSpectral(WaveSpectralBase):
    def __init__(self, mset: ModelSettings, grid: Grid) -> None:
        from fridom.ShallowWater.Eigenvectors import VecP, VecQ
        super().__init__(mset, grid, VecQ, VecP)

class DivergenceSpectral(DivergenceSpectralBase):
    def __init__(self, mset: ModelSettings, grid: Grid) -> None:
        from fridom.ShallowWater.Eigenvectors import VecP, VecQ
        super().__init__(mset, grid, VecQ, VecP)

class GeostrophicTimeAverage(GeostrophicTimeAverageBase):
    """
    Geostrophic projection using time-averaging.
    """
    def __init__(self, mset: ModelSettings, grid: Grid, 
                 n_ave=4,
                 equidistant_chunks=True,
                 max_period=None,
                 backward_forward=False
                 ) -> None:
        """
        Geostrophic projection using time-averaging.

        Arguments:
            mset      (ModelSettings) : Model settings.
            grid      (Grid)          : The grid.
            n_ave             (int)   : Number of averages to perform.
            equidistant_chunks(bool)  : Whether to split the averaging periods 
                                        into equidistant chunks.
            max_period        (float) : The maximum period of the time averages.
                                        if None, the maximum period is set to
                                        the inertial period.
            backward_forward  (bool)  : Whether to use backward-forward averaging.
        """
        from fridom.ShallowWater.Model import Model
        super().__init__(mset, grid, Model, n_ave, 
                         equidistant_chunks, max_period, backward_forward)
        return

class OptimalBalance(OptimalBalanceBase):
    def __init__(self, mset: ModelSettings, grid: Grid, 
                 base_proj: Projection, 
                 ramp_period=1, ramp_type="exp", 
                 enable_forward_friction=False, 
                 enable_backward_friction=False, 
                 update_base_point=True,
                 max_it=3, stop_criterion=1e-9,
                 return_details=False) -> None:
        from fridom.ShallowWater.Model import Model
        super().__init__(mset=mset, 
                         grid=grid, 
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
    def __init__(self, mset: ModelSettings, grid: Grid, 
                 order=3, enable_dealiasing=True) -> None:
        from fridom.ShallowWater.Model import Model
        from fridom.ShallowWater.Eigenvectors import VecP, VecQ
        from fridom.ShallowWater.State import State
        super().__init__(mset, grid, Model, State, VecQ, VecP, order, enable_dealiasing)

# remove symbols from the namespace
del GeostrophicSpectralBase, Projection, WaveSpectralBase, \
    DivergenceSpectralBase, GeostrophicTimeAverageBase, OptimalBalanceBase, \
    NNMDBase, ModelSettings, Grid