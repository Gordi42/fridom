from fridom.nonhydro.grid import Grid
from fridom.framework.projection.projection import Projection
from fridom.framework.projection.optimal_balance import OptimalBalanceBase


class OptimalBalance(OptimalBalanceBase):
    def __init__(self, grid: Grid, 
                 base_proj: Projection, 
                 ramp_period=1, ramp_type="exp", 
                 enable_forward_friction=False, 
                 enable_backward_friction=False, 
                 update_base_point=True,
                 max_it=3, stop_criterion=1e-9,
                 return_details=False) -> None:
        from fridom.nonhydro.model import Model
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

# remove symbols from namespace
del Grid, Projection, OptimalBalanceBase