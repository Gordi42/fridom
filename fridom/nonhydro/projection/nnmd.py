from fridom.nonhydro.grid import Grid
from fridom.framework.projection \
    .nnmd import NNMDBase

class NNMD(NNMDBase):
    def __init__(self, grid: Grid, 
                 order=3, enable_dealiasing=True) -> None:
        from fridom.nonhydro.model import Model
        from fridom.nonhydro.eigenvectors import VecP, VecQ
        from fridom.nonhydro.state import State
        super().__init__(grid, Model, State, VecQ, VecP, order, enable_dealiasing)

# remove symbols from the namespace
del Grid, NNMDBase