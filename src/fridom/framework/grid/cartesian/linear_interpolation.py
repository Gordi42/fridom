from copy import deepcopy

import fridom.framework as fr


@fr.utils.jaxify
class LinearInterpolation(fr.grid.InterpolationModule):

    r"""
    Simple linear interpolation for cartesian grids.

    .. math::
        f(x + 0.5 \Delta x) = \frac{1}{2} (f(x) + f(x + \Delta x))
    """

    name = "Linear Interpolation"
    required_halo = 1

    def interpolate(self,
                    f: fr.ScalarField,
                    destination: fr.grid.Position) -> fr.ScalarField:
        for axis in range(f.arr.ndim):
            f = self._interpolate_axis(f, axis, destination.positions[axis])
        return f
        if all(self.grid.periodic_bounds):
            return f
        return self.grid.water_mask.apply_mask(f)

    def _create_scalar_field(self,
                             f: fr.ScalarField,
                             axis: int,
                             arr: fr.config.ncp.ndarray) -> fr.ScalarField:
        mdata = deepcopy(f.mdata)
        mdata.position = f.position.shift(axis)
        return fr.ScalarField(mset=f.mset, mdata=mdata, arr=arr)

    def _interpolate_axis(self, 
                         f: fr.ScalarField,
                         axis: int,
                         destination: fr.grid.AxisPosition) -> fr.ScalarField:

        if f.position[axis] == destination:
            # no interpolation needed
            return f
        if not f.topo[axis]:
            # no interpolation when the field has no extend along the axis
            return self._create_scalar_field(f, axis, f.arr)

        shift = -1 if destination == fr.grid.AxisPosition.FACE else 1

        @self.grid.domain_decomp.shard_map
        def _interpolate(arr):
            rolled = fr.config.ncp.roll(arr, shift=shift, axis=axis)
            return 0.5 * (arr + rolled)

        return self._create_scalar_field(f, axis, _interpolate(f.arr))
