import fridom.framework as fr
from functools import partial


@partial(fr.utils.jaxify, dynamic=('water_mask', ))
class PolynomialInterpolation(fr.grid.InterpolationModule):
    r"""
    Polynomial interpolation for cartesian grids.

    Description
    -----------
    Consider the following grid points:

    .. math::
        x_i = (i - n/2) \Delta x, \quad i = 0, 1, \ldots, n

    where :math:`n` is the (odd) order of the polynomial interpolation. For
    example for :math:`n = 3` we have the following grid points:

    ::

            We want to interpolate the field to this point (x=0)
                                    ↓
                |   x_0   |   x_1   |   x_2   |   x_3   |
        x/dx =     -3/2      -1/2       1/2       3/2

    Let :math:`f_i` be the field values at :math:`x_i`. We define the
    continuous extension of the field as:

    .. math::
        f(x) = \sum_{i=0}^{n} \left(
            \prod_{j=0, j \neq i}^{n} \left(
                \frac{x - x_j}{x_i - x_j} f_i
            \right)
        \right)

    By definition, :math:`f(x_i) = f_i` holds. Finally, to interpolate the 
    field to the point :math:`x=0`, we insert :math:`x=0` into the above
    expression. Note that the grid spacing :math:`\Delta x` cancels out.

    .. math::
        f(0) = \sum_{i=0}^{n} c_i f_i

    with the coefficients :math:`c_i` given by:
    
    .. math::
        c_i = \prod_{j=0, j \neq i}^{n} \frac{j-n/2}{j - i}
    """
    name = "Polynomial Interpolation"
    def __init__(self, order: int = 1):
        super().__init__()
        # order must be an odd number
        assert order % 2 == 1

        self.required_halo = order // 2 + 1
        self.order = order
        self._coeffs = None
        self._slices = None
        self._nexts = None
        self._prevs = None
        self.water_mask = None
        return

    @fr.modules.module_method
    def setup(self, mset: 'fr.ModelSettingsBase') -> None:
        super().setup(mset)
        self.ndim = ndim = self.mset.grid.n_dims
        # coefficients for the polynomial interpolation
        order = self.order
        coeffs = []
        for i in range(order+1):
            c = fr.config.dtype_real(1)
            for j in range(order+1):
                if j != i:
                    c *= (j - order/2) / (j - i)
            coeffs.append(c)
        self._coeffs = coeffs

        # slices to get certain parts of the array
        slices = [slice(i, -order + i) for i in range(order)]
        slices.append(slice(order, None))

        all_slices = []
        for axis in range(ndim):
            sl = []
            for sli in slices:
                s = [slice(None)] * ndim
                s[axis] = sli
                sl.append(tuple(s))
            all_slices.append(sl)
        self._slices = all_slices

        self._nexts = tuple(self._get_slices(axis)[0] for axis in range(ndim))
        self._prevs = tuple(self._get_slices(axis)[1] for axis in range(ndim))

        # water mask
        self.water_mask = self.mset.grid.water_mask
        return

    @fr.utils.jaxjit
    def interpolate(self, 
                    f: fr.FieldVariable,
                    destination: fr.grid.Position) -> fr.FieldVariable:
        for axis in range(f.arr.ndim):
            f = self.interpolate_axis(f, axis, destination.positions[axis])
        mask = self.water_mask.get_mask(destination)
        f.arr *= mask
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

        # get the destination slice
        match destination:
            case fr.grid.AxisPosition.CENTER:
                dest_slice = self._nexts[axis]
            case fr.grid.AxisPosition.FACE:
                dest_slice = self._prevs[axis]

        @self.grid.domain_decomp.shard_map
        def interpolate(arr):

            average = sum(arr[s] * self._coeffs[i] 
                        for i, s in enumerate(self._slices[axis]))
            return fr.utils.modify_array(arr, dest_slice, average)
        
        res.arr = interpolate(f.arr)
        res.position = f.position.shift(axis)
        return res

    def _get_slices(self, axis):
        n = self.order // 2
        if n == 0:
            end = None
        else:
            end = -n
        next = tuple(slice(n+1, end) if i == axis else slice(None) 
                     for i in range(self.ndim))
        prev = tuple(slice(n, -1-n) if i == axis else slice(None) 
                     for i in range(self.ndim))
        return next, prev
