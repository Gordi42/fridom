from copy import deepcopy
import fridom.framework as fr
from functools import partial


@partial(fr.utils.jaxify, dynamic=('_dx1', ))
class FiniteDifferences(fr.grid.DiffModule):
    name = "Finite Differences"
    def __init__(self) -> None:
        super().__init__()
        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.required_halo = 1
        self._dx1 = None

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        super().setup(mset)
        from .grid import Grid
        if not isinstance(self.mset.grid, Grid):
            raise ValueError("Finite differences only work with Cartesian grids.")

        conf = fr.config
        self._dx1 = 1 / conf.ncp.array(self.mset.grid.dx, dtype=conf.dtype_real)

    def diff(self, 
             f: fr.ScalarField,
             axis: int) -> fr.ScalarField:
        # differentiate the field
        match f.position[axis]:
            case fr.grid.AxisPosition.CENTER:
                f = self._diff_forward(f, axis)
            case fr.grid.AxisPosition.FACE:
                f = self._diff_backward(f, axis)

        return f
        if all(self.grid.periodic_bounds):
            return f
        return self.grid.water_mask.apply_mask(f)
        return f.apply_water_mask()

    def _diff_forward(self, 
                      f: fr.ScalarField, 
                      axis: int) -> fr.ScalarField:
        # update the metadata
        mdata = deepcopy(f.mdata)
        mdata.position = f.position.shift(axis)

        @self.grid.domain_decomp.shard_map
        def _diff(arr):
            rolled = fr.config.ncp.roll(arr, shift=-1, axis=axis)
            return (rolled - arr) * self._dx1[axis]

        return fr.ScalarField(mset=f.mset, mdata=mdata, arr=_diff(f.arr))

    def _diff_backward(self,
                       f: fr.ScalarField, 
                       axis: int) -> fr.ScalarField:
        # update the metadata
        mdata = deepcopy(f.mdata)
        mdata.position = f.position.shift(axis)

        @self.grid.domain_decomp.shard_map
        def _diff(arr):
            rolled = fr.config.ncp.roll(arr, shift=1, axis=axis)
            return (arr - rolled) * self._dx1[axis]

        return fr.ScalarField(mset=f.mset, mdata=mdata, arr=_diff(f.arr))
