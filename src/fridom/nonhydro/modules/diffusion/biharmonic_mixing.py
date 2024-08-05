import fridom.framework as fr
import fridom.nonhydro as nh


class BiharmonicMixing(fr.modules.Module):
    r"""
    Biharmonic mixing module

    Description
    -----------
    Following Griffiies et al. (2000), the biharmonic mixing operator iterates 
    twice over the harmonic mixing operator. For a scalar field :math:`b` it is 
    given by:

    .. math::
        \Delta b = - \nabla \cdot \left (\boldsymbol{K} \cdot \nabla b 
            \nabla \cdot \left (\boldsymbol{K} \cdot \nabla b \right)
            \right)

    with:

    .. math::
        \boldsymbol{K} = \begin{pmatrix} \sqrt{k_h} \\ \sqrt{k_h} \\ \sqrt{k_v} \end{pmatrix}

    where :math:`k_h` is the horizontal biharmonic mixing coefficient and
    :math:`k_v` is the vertical biharmonic mixing coefficient. All components
    with the flag "ENABLE_MIXING" are mixed.

    Parameters
    ----------
    `kh` : `float`
        Horizontal biharmonic mixing coefficient.
    `kv` : `float`
        Vertical biharmonic mixing coefficient.
    `diff` : `fr.grid.DiffBase | None`, (default=None)
        Differentiation module to use. If None, the differentiation module of
        the grid is used.
    """
    _dynamic_attributes = ["mset", "_kh", "_kv"]
    def __init__(self, 
                 kh: float = 0, 
                 kv: float = 0,
                 diff: fr.grid.DiffModule | None = None):
        super().__init__(name="Harmonic Mixing")
        self.required_halo = 2
        self._kh = None
        self._kv = None
        self._diff = diff
        # set the coefficients
        self.kh = kh
        self.kv = kv
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
    def harmonic_mixing(self, f: fr.FieldVariable) -> fr.FieldVariable:
        """
        Compute the harmonic mixing term.
        """
        kh = self._kh; kv = self._kv
        dfdx, dfdy, dfdz = self.diff.grad(f)
        dfdx *= kh; dfdy *= kh; dfdz *= kv
        div = self.diff.div((dfdx, dfdy, dfdz))
        return div

    @fr.utils.jaxjit
    def mixing(self, z: nh.State, dz: nh.State) -> nh.State:
        r"""
        Compute the harmonic mixing term.
        """
        for name, field in z.fields.items():
            if field.flags["ENABLE_MIXING"]:
                # dz.fields[name] += self.biharmonic_mixing(field)
                div1 = self.harmonic_mixing(field)
                div2 = self.harmonic_mixing(div1)
                dz.fields[name] -= div2
        return dz

    @fr.modules.module_method
    def update(self, mz: nh.ModelState) -> nh.ModelState:
        mz.dz = self.mixing(mz.z, mz.dz)
        return mz

    @property
    def kh(self) -> float:
        """The horizontal harmonic mixing coefficient."""
        return self._kh ** 2
    
    @kh.setter
    def kh(self, value: float):
        self._kh = fr.config.ncp.sqrt(value)
        return

    @property
    def kv(self) -> float:
        """The vertical harmonic mixing coefficient."""
        return self._kv ** 2
    
    @kv.setter
    def kv(self, value: float):
        self._kv = fr.config.ncp.sqrt(value)
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

fr.utils.jaxify_class(BiharmonicMixing)