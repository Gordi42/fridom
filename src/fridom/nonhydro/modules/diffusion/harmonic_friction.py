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
    _dynamic_attributes = ["mset", "ah", "av"]
    def __init__(self, 
                 ah: float = 0, 
                 av: float = 0,
                 diff: fr.grid.DiffModule | None = None):
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
        return

    @fr.utils.jaxjit
    def friction(self, z: nh.State, dz: nh.State) -> nh.State:
        """
        Compute the harmonic friction term.
        """
        for name in ["u", "v", "w"]:
            dfdx, dfdy, dfdz = self.diff.grad(z.fields[name])
            dfdx *= self.ah; dfdy *= self.ah; dfdz *= self.av
            div = self.diff.div((dfdx, dfdy, dfdz))
            dz.fields[name] += div
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
    def diff(self) -> fr.grid.DiffModule:
        """The differentiation module."""
        return self._diff
    
    @diff.setter
    def diff(self, value: fr.grid.DiffModule):
        self._diff = value
        return

fr.utils.jaxify_class(HarmonicFriction)