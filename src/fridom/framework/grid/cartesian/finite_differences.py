import fridom.framework as fr
from functools import partial


@partial(fr.utils.jaxify, dynamic=('_dx1', 'water_mask'))
class FiniteDifferences(fr.grid.DiffModule):
    name = "Finite Differences"
    def __init__(self) -> None:
        super().__init__()
        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.required_halo = 1
        self._dx1 = None
        self.water_mask = None

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        super().setup(mset)
        from .grid import Grid
        if not isinstance(self.mset.grid, Grid):
            raise ValueError("Finite differences only work with Cartesian grids.")
        
        conf = fr.config
        self._dx1 = 1 / conf.ncp.array(self.mset.grid.dx, dtype=conf.dtype_real)
        self.water_mask = self.mset.grid.water_mask
        return

    @partial(fr.utils.jaxjit, static_argnames=('axis', 'order'))
    def diff(self, 
             f: fr.FieldVariable,
             axis: int,
             order: int = 1) -> fr.FieldVariable:
        # differentiate the field
        match f.position[axis]:
            case fr.grid.AxisPosition.CENTER:
                f = self._diff_forward(f, axis)
            case fr.grid.AxisPosition.FACE:
                f = self._diff_backward(f, axis)

        # check if we need to differentiate more
        if order == 1:
            return f
        else:
            return self.diff(f, axis, order-1)

    @partial(fr.utils.jaxjit, static_argnames=('axis',))
    def _diff_forward(self, 
                      f: fr.FieldVariable, 
                      axis: int) -> fr.FieldVariable:
        res = fr.FieldVariable(**f.get_kw())
        new_pos = f.position.shift(axis)
        mask = self.water_mask.get_mask(new_pos)

        next = tuple(slice(1, None) if i == axis else slice(None) 
                     for i in range(f.arr.ndim))
        prev = tuple(slice(None, -1) if i == axis else slice(None) 
                     for i in range(f.arr.ndim))

        @self.grid.domain_decomp.shard_map
        def _diff(arr):
            diff = (arr[next] - arr[prev]) * self._dx1[axis]
            return fr.utils.modify_array(arr, prev, diff)

        res.arr = _diff(f.arr) * mask
        res.position = new_pos

        return res

    @partial(fr.utils.jaxjit, static_argnames=('axis',))
    def _diff_backward(self,
                       f: fr.FieldVariable, 
                       axis: int) -> fr.FieldVariable:
        res = fr.FieldVariable(**f.get_kw())
        new_pos = f.position.shift(axis)
        mask = self.water_mask.get_mask(new_pos)

        next = tuple(slice(1, None) if i == axis else slice(None) 
                     for i in range(f.arr.ndim))
        prev = tuple(slice(None, -1) if i == axis else slice(None) 
                     for i in range(f.arr.ndim))

        @self.grid.domain_decomp.shard_map
        def _diff(arr):
            diff = (arr[next] - arr[prev]) * self._dx1[axis]
            return fr.utils.modify_array(arr, next, diff)

        res.arr = _diff(f.arr) * mask
        res.position = new_pos

        return res
