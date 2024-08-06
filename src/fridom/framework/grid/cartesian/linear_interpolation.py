import fridom.framework as fr
from functools import partial


@partial(fr.utils.jaxify, dynamic=('water_mask',))
class LinearInterpolation(fr.grid.InterpolationModule):
    r"""
    Simple linear interpolation for cartesian grids.

    .. math::
        f(x + 0.5 \Delta x) = \frac{1}{2} (f(x) + f(x + \Delta x))
    """
    def __init__(self) -> None:
        super().__init__(name="Linear Interpolation")
        self.ndim: int = None
        self._nexts: tuple[slice] = None
        self._prevs: tuple[slice] = None
        self.water_mask = None
        return

    @fr.modules.setup_module
    def setup(self) -> None:
        self.ndim = ndim = self.mset.grid.n_dims
        self._nexts = tuple(self._get_slices(axis)[0] for axis in range(ndim))
        self._prevs = tuple(self._get_slices(axis)[1] for axis in range(ndim))
        self.water_mask = self.mset.grid.water_mask
        return

    @fr.utils.jaxjit
    def interpolate(self, 
                    f: fr.FieldVariable,
                    destination: fr.grid.Position) -> fr.FieldVariable:
        for axis in range(f.arr.ndim):
            f = self.interpolate_axis(f, axis, destination.positions[axis])
        mask = self.water_mask.get_mask(destination)
        f.arr = f.arr * mask
        return f

    @partial(fr.utils.jaxjit, static_argnames=('axis', 'destination'))
    def interpolate_axis(self, 
                         f: fr.FieldVariable,
                         axis: int,
                         destination: fr.grid.AxisPosition) -> fr.FieldVariable:
        if not f.topo[axis]:
            # no interpolation when the field has no extend along the axis
            return f

        if f.position[axis] == destination:
            # no interpolation needed
            return f

        res = fr.FieldVariable(**f.get_kw())
        next = self._nexts[axis]; prev = self._prevs[axis]
        average = 0.5 * (f.arr[next] + f.arr[prev])

        # get the destination slice
        match destination:
            case fr.grid.AxisPosition.CENTER:
                dest_slice = next
            case fr.grid.AxisPosition.FACE:
                dest_slice = prev

        res.arr = fr.utils.modify_array(res.arr, dest_slice, average)
        res.position = f.position.shift(axis)
        return res

    def _get_slices(self, axis):
        next = tuple(slice(1, None) if i == axis else slice(None) 
                     for i in range(self.ndim))
        prev = tuple(slice(None, -1) if i == axis else slice(None) 
                     for i in range(self.ndim))
        return next, prev
