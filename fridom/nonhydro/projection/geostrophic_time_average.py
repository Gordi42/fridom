from fridom.nonhydro.grid import Grid
from fridom.framework.projection \
    .geostrophic_time_average import GeostrophicTimeAverageBase


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
        from fridom.nonhydro.model import Model
        super().__init__(grid, Model, n_ave, 
                         equidistant_chunks, max_period, backward_forward)
        return

# remove symbols from namespace
del Grid, GeostrophicTimeAverageBase