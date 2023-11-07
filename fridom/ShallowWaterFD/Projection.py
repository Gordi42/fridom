from fridom.Framework.ProjectionBase import GeostrophicSpectralBase, Projection, WaveSpectralBase, DivergenceSpectralBase, GeostrophicTimeAverageBase
from fridom.Framework.OptimalBalanceBase import OptimalBalanceBase
from fridom.Framework.NNMDBase import NNMDBase

from fridom.ShallowWaterFD.ModelSettings import ModelSettings
from fridom.ShallowWaterFD.Grid import Grid
from fridom.ShallowWaterFD.Eigenvectors import VecP, VecQ
from fridom.ShallowWaterFD.State import State


class GeostrophicSpectral(GeostrophicSpectralBase):
    def __init__(self, mset: ModelSettings, grid: Grid) -> None:
        super().__init__(mset, grid, VecQ, VecP)

class WaveSpectral(WaveSpectralBase):
    def __init__(self, mset: ModelSettings, grid: Grid) -> None:
        super().__init__(mset, grid, VecQ, VecP)

class DivergenceSpectral(DivergenceSpectralBase):
    def __init__(self, mset: ModelSettings, grid: Grid) -> None:
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
        from fridom.ShallowWaterFD.Model import Model
        super().__init__(mset, grid, Model, n_ave, 
                         equidistant_chunks, max_period, backward_forward)
        return

class OptimalBalance(OptimalBalanceBase):
    def __init__(self, mset: ModelSettings, grid: Grid, 
                 base_proj: Projection, 
                 ramp_period=1, ramp_type="exp", 
                 enable_forward_friction=False, 
                 enable_backward_friction=False, 
                 max_it=3, stop_criterion=1e-9,
                 return_details=False) -> None:
        from fridom.ShallowWaterFD.Model import Model
        super().__init__(mset, grid, Model, base_proj, ramp_period, ramp_type, enable_forward_friction, enable_backward_friction, max_it, stop_criterion, return_details)

class NNMD(NNMDBase):
    def __init__(self, mset: ModelSettings, grid: Grid, 
                 order=3, enable_dealiasing=True) -> None:
        from fridom.ShallowWaterFD.Model import Model
        super().__init__(mset, grid, Model, State, VecQ, VecP, order, enable_dealiasing)
