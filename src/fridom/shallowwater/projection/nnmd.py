from fridom.shallowwater.grid import Grid
from fridom.framework.projection \
    .nnmd import NNMDBase

class NNMD(NNMDBase):
    def __init__(self, grid: Grid, 
                 order=3, enable_dealiasing=True) -> None:
        from fridom.shallowwater.model import Model
        from fridom.shallowwater.eigenvectors import VecP, VecQ
        from fridom.shallowwater.state import State
        super().__init__(grid, Model, State, VecQ, VecP, order, enable_dealiasing)

# remove symbols from the namespace
del Grid, NNMDBase