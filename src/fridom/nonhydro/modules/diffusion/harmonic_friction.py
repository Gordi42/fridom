import fridom.framework as fr
import fridom.nonhydro as nh
from numpy import ndarray
from functools import partial


class HarmonicFriction(fr.modules.Module):
    r"""
    Harmonic friction module

    Description
    -----------
    The tendency of the harmonic friction module on velocity field :math:`u` 
    is given by:
    .. math::
        \Delta u = \nabla \cdot \left (\boldsymbol{A} \cdot \nabla u \right)

    with:
    .. math::
        \boldsymbol{A} = \begin{bmatrix} a_h \\ a_h \\ a_v \end{bmatrix}

    where :math:`a_h` is the horizontal harmonic friction coefficient and
    :math:`k_h` is the vertical harmonic friction coefficient. For
    :math:`v` and :math:`w` the same equation applies.

    Parameters
    ----------
    `ah` : `float`
        Horizontal harmonic friction coefficient.
    `av` : `float`
        Vertical harmonic friction coefficient.
    `diff` : `fr.grid.DiffBase | None`, (default=None)
        Differentiation module to use. If None, the differentiation module of
        the grid is used.
    """
    _dynamic_attributes = ["mset", "ah", "av", "water_mask"]
    def __init__(self, 
                 ah: float = 0, 
                 av: float = 0,
                 diff: fr.grid.DiffBase | None = None):
        super().__init__(name="Harmonic Friction")
        self.ah = fr.config.ncp.array(ah)
        self.av = fr.config.ncp.array(av)
        self._diff = diff

    @fr.modules.setup_module
    def setup(self):
        # setup the differentiation modules
        if self.diff is None:
            self.diff = self.mset.grid._diff_mod
        else:
            self.diff.setup(mset=self.mset)
        self.water_mask = self.grid._water_mask
        return

    @partial(fr.utils.jaxjit, static_argnames=("pos"))
    def harmonic(self, arr: ndarray, pos: fr.grid.Position) -> ndarray:
        r"""
        Compute the harmonic second order derivative.

        Description
        -----------
        Computes the harmonic friction term for the given field variable.

        Parameters
        ----------
        `arr` : `ndarray`
            The field variable.

        Returns
        -------
        `ndarray`
            The harmonic friction term.
        """
        # div = fr.config.ncp.zeros_like(arr)
        # a = [self.ah, self.ah, self.av]
        # for axis in range(3):
        #     # match pos[axis]:
        #     #     case fr.grid.AxisPosition.CENTER:
        #     #         first_dif = "forward"
        #     #         second_dif = "backward"
        #     #     case fr.grid.AxisPosition.RIGHT:
        #     first_dif = "backward"
        #     second_dif = "forward"
        #     #     case fr.grid.AxisPosition.LEFT:
        #     # first_dif = "forward"
        #     # second_dif = "backward"

        #     # new_pos = pos.shift(axis, first_dif)
        #     print("yo")
        #     print(new_pos)
        #     print("yo")
        #     grad = mask * self.diff.diff(arr, axis, first_dif) * a[axis]
        #     div += self.diff.diff(grad, axis, second_dif)
        # mask = self.grid.get_water_mask(pos)
        new_pos = self.grid.cell_center
        mask = self.water_mask.get_mask(new_pos)
        return mask * arr

    # @fr.utils.jaxjit
    def friction(self, z: nh.State, dz: nh.State) -> nh.State:
        """
        Compute the harmonic friction term.
        """
        dz.u.arr += self.harmonic(z.u.arr, z.u.position)
        dz.v.arr += self.harmonic(z.v.arr, z.v.position)
        # dz.w.arr += self.harmonic(z.w.arr, z.w.position)
        return dz

    @fr.modules.module_method
    def update(self, mz: nh.ModelState) -> nh.ModelState:
        mz.dz = self.friction(mz.z, mz.dz)
        return mz

    @property
    def info(self) -> dict:
        res = super().info
        res["ah"] = self.ah
        res["av"] = self.av
        res["diff"] = self.diff
        return res

    @property
    def diff(self) -> fr.grid.DiffBase:
        """The differentiation module."""
        return self._diff
    
    @diff.setter
    def diff(self, value: fr.grid.DiffBase):
        self._diff = value
        return

fr.utils.jaxify_class(HarmonicFriction)