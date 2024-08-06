import fridom.framework as fr
import fridom.nonhydro as nh


class BiharmonicFriction(fr.modules.Module):
    r"""
    Biharmonic friction module

    Description
    -----------
    Following Griffiies et al. (2000), the biharmonic friction operator iterates 
    twice over the harmonic friction operator. For a velocity scalar field 
    :math:`u` it is given by:

    .. math::
        \Delta b = - \nabla \cdot \left (\boldsymbol{A} \cdot \nabla b 
            \nabla \cdot \left (\boldsymbol{A} \cdot \nabla b \right)
            \right)

    with:

    .. math::
        \boldsymbol{A} = \begin{pmatrix} \sqrt{a_h} \\ \sqrt{a_h} \\ \sqrt{a_v} \end{pmatrix}

    where :math:`a_h` is the horizontal biharmonic friction coefficient and
    :math:`a_v` is the vertical biharmonic friction coefficient. The biharmonic
    operator is applied to all velocity components (u, v, w).

    Parameters
    ----------
    `ah` : `float`
        Horizontal biharmonic friction coefficient.
    `av` : `float`
        Vertical biharmonic friction coefficient.
    `diff` : `fr.grid.DiffModule | None`, (default=None)
        Differentiation module to use. If None, the differentiation module of
        the grid is used.
    """
    _dynamic_attributes = ["mset", "_ah", "_av"]
    def __init__(self, 
                 ah: float = 0, 
                 av: float = 0,
                 diff: fr.grid.DiffModule | None = None):
        super().__init__(name="Harmonic Mixing")
        self.required_halo = 2
        self._ah = None
        self._av = None
        self._diff = diff
        # set the coefficients
        self.ah = ah
        self.av = av
        return

    @fr.modules.setup_module
    def setup(self):
        # setup the differentiation modules
        if self.diff is None:
            self.diff = self.mset.grid._diff_mod
        else:
            self.diff.setup(mset=self.mset)
        return

    @fr.utils.jaxjit
    def harmonic_friction(self, f: fr.FieldVariable) -> fr.FieldVariable:
        """
        Compute the harmonic friction term.
        """
        ah = self._ah; av = self._av
        dfdx, dfdy, dfdz = self.diff.grad(f)
        dfdx *= ah; dfdy *= ah; dfdz *= av
        div = self.diff.div((dfdx, dfdy, dfdz))
        return div

    @fr.utils.jaxjit
    def friction(self, z: nh.State, dz: nh.State) -> nh.State:
        r"""
        Compute the biharmonic friction term.
        """
        for name in ["u", "v", "w"]:
            div1 = self.harmonic_friction(z.fields[name])
            div2 = self.harmonic_friction(div1)
            dz.fields[name] -= div2
            dz.fields[name]
        return dz

    @fr.modules.module_method
    def update(self, mz: nh.ModelState) -> nh.ModelState:
        mz.dz = self.friction(mz.z, mz.dz)
        return mz

    @property
    def ah(self) -> float:
        """The horizontal biharmonic friction coefficient."""
        return self._ah ** 2
    
    @ah.setter
    def ah(self, value: float):
        self._ah = fr.config.ncp.sqrt(value)
        return

    @property
    def av(self) -> float:
        """The vertical biharmonic friction coefficient."""
        return self._kv ** 2
    
    @av.setter
    def av(self, value: float):
        self._av = fr.config.ncp.sqrt(value)
        return

    @property
    def info(self) -> dict:
        res = super().info
        res["diff"] = self.diff
        res["ah"] = self.ah
        res["av"] = self.av
        return res

    @property
    def diff(self) -> fr.grid.DiffModule:
        """The differentiation module."""
        return self._diff
    
    @diff.setter
    def diff(self, value: fr.grid.DiffModule):
        self._diff = value
        return

fr.utils.jaxify_class(BiharmonicFriction)