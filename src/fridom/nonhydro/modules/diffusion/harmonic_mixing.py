import fridom.framework as fr
import fridom.nonhydro as nh
from numpy import ndarray
from functools import partial


class HarmonicMixing(fr.modules.Module):
    r"""
    Harmonic mixing module

    Description
    -----------
    The harmonic mixing module on a scalar field :math:`b` is given by:

    .. math::
        \Delta b = \nabla \cdot \left (\boldsymbol{K} \cdot \nabla b \right)

    with:
    .. math::
        \boldsymbol{K} = \begin{bmatrix} k_h \\ k_h \\ k_v \end{bmatrix}

    where :math:`k_h` is the horizontal harmonic mixing coefficient and
    :math:`k_v` is the vertical harmonic mixing coefficient. All components
    with the flag "ENABLE_MIXING" are mixed.

    Parameters
    ----------
    `kh` : `float`
        Horizontal harmonic mixing coefficient.
    `kv` : `float`
        Vertical harmonic mixing coefficient.
    `diff` : `fr.grid.DiffBase | None`, (default=None)
        Differentiation module to use. If None, the differentiation module of
        the grid is used.
    """
    _dynamic_attributes = ["mset", "kh", "kv"]
    def __init__(self, 
                 kh: float = 0, 
                 kv: float = 0,
                 diff: fr.grid.DiffBase | None = None):
        super().__init__(name="Harmonic Mixing")
        self.kh = fr.config.ncp.array(kh)
        self.kv = fr.config.ncp.array(kv)
        self._diff = diff

    @fr.modules.setup_module
    def setup(self):
        # setup the differentiation modules
        if self.diff is None:
            self.diff = self.mset.grid._diff_mod
        else:
            self.diff.setup(mset=self.mset)
        return

    # @partial(fr.utils.jaxjit, static_argnames=("pos"))
    # def harmonic(self, arr: ndarray, pos: fr.grid.Position) -> ndarray:
    #     r"""
    #     Compute the harmonic second order derivative.

    #     Description
    #     -----------
    #     Computes the harmonic friction term for the given field variable.

    #     Parameters
    #     ----------
    #     `arr` : `ndarray`
    #         The field variable.

    #     Returns
    #     -------
    #     `ndarray`
    #         The harmonic friction term.
    #     """
    #     # calculate the gradient of the field variable
    #     div = fr.config.ncp.zeros_like(arr)
    #     k = [self.kh, self.kh, self.kv]
    #     for axis in range(3):
    #         # new_pos = pos.shift(axis, "forward")
    #         # mask = self.grid.get_water_mask(new_pos)
    #         grad = self.diff.diff(arr, axis, "forward") * k[axis] #* mask
    #         div += self.diff.diff(grad, axis, "backward")
    #     # mask = self.grid.get_water_mask(pos)
    #     return div# * mask

    # @fr.utils.jaxjit
    # def mixing(self, z: nh.State, dz: nh.State) -> nh.State:
    #     r"""
    #     Compute the harmonic mixing term.
    #     """
    #     # z = self.mset.bc.apply_boundary_conditions(z)
    #     for name, field in z.fields.items():
    #         if field.flags["ENABLE_MIXING"]:
    #             dz.fields[name].arr += self.harmonic(field.arr, field.position)
    #     return dz

    @partial(fr.utils.jaxjit)
    def harmonic(self, field):
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
        # calculate the gradient of the field variable
        div = fr.config.ncp.zeros_like(field.arr)
        k = [self.kh, self.kh, self.kv]
        for axis in range(3):
            # new_pos = field.position.shift(axis, "forward")
            # mask = self.grid.get_water_mask(new_pos)
            grad = self.diff.diff(field.arr, axis, "forward") * k[axis] #* mask
            div += self.diff.diff(grad, axis, "backward")
        # mask = self.grid.get_water_mask(field.position)
        new_field = fr.FieldVariable(arr=div, **field.get_kw())
        return new_field

    @fr.utils.jaxjit
    def mixing(self, z: nh.State, dz: nh.State) -> nh.State:
        r"""
        Compute the harmonic mixing term.
        """
        # z = self.mset.bc.apply_boundary_conditions(z)
        for name, field in z.fields.items():
            if field.flags["ENABLE_MIXING"]:
                dz.fields[name].arr += self.harmonic(field).arr
        return dz

    def intermediate(self, mz: nh.ModelState) -> nh.ModelState:
        mz.dz = self.mixing(mz.z, mz.dz)
        mz.dz.b.arr.block_until_ready()
        return mz

    @fr.modules.module_method
    def update(self, mz: nh.ModelState) -> nh.ModelState:
        # wait untils jax array is read
        mz.z.b.arr.block_until_ready()
        return self.intermediate(mz)

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

fr.utils.jaxify_class(HarmonicMixing)