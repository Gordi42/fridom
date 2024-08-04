import fridom.framework as fr
import fridom.nonhydro as nh
from numpy import ndarray


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

    @fr.utils.jaxjit
    def harmonic(self, arr: ndarray) -> ndarray:
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
        grad = self.diff.grad(arr)
        # multiply the gradient with the harmonic friction coefficients
        grad = [self.kh * grad[0], self.kh * grad[1], self.kv * grad[2]]
        # return the divergence of the gradient
        return self.diff.div(grad)

    @fr.utils.jaxjit
    def mixing(self, z: nh.State, dz: nh.State) -> nh.State:
        r"""
        Compute the harmonic mixing term.
        """
        z = self.mset.bc.apply_boundary_conditions(z)
        for name, field in z.fields.items():
            if field.flags["ENABLE_MIXING"]:
                dz.fields[name].arr += self.harmonic(field.arr)
        return self.mset.bc.apply_boundary_conditions(dz)

    @fr.modules.module_method
    def update(self, mz: nh.ModelState) -> nh.ModelState:
        mz.dz = self.mixing(mz.z, mz.dz)
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

fr.utils.jaxify_class(HarmonicMixing)