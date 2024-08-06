import fridom.framework as fr

@fr.utils.jaxjit
def _get_coefficients(diffusion_coefficients):
    ncp = fr.config.ncp
    coeffs = []
    for coeff in diffusion_coefficients:
        if isinstance(coeff, fr.FieldVariable):
            kappa = ncp.sqrt(ncp.abs(coeff.arr))
            kappa = fr.FieldVariable(arr=kappa, **coeff.get_kw())
        else:
            kappa = ncp.sqrt(ncp.abs(coeff))
        coeffs.append(kappa)
    return coeffs

class BiharmonicDiffusion(fr.modules.Module):
    r"""
    Biharmonic diffusion module

    Description
    -----------
    Following Griffiies et al. (2000), the biharmonic mixing operator 
    :math:`\mathcal{B}` iterates twice over the harmonic mixing operator
    :math:`\mathcal{H}`. For a scalar field :math:`u` it is given by:

    .. math::
        \mathcal{B}(u) = - \mathcal{H} \left( \mathcal{H}(u) \right)

    where we use the biharmonic diffusion coefficient :math:`\sqrt{|\kappa_i|}`. 
    The index :math:`i` refers to the direction of the diffusion.

    Parameters
    ----------
    `field_flags` : `list[str]`
        A list of strings that indicate which fields should be diffused.
        For example, if `field_flags=["ENABLE_MIXING"]`, all fields with the
        flag "ENABLE_MIXING" will be diffused. For more information on possible
        flags, see :py:mod:`fridom.framework.FieldVariable`.
    `diffusion_coefficients` : `tuple[float | fr.FieldVariable]`
        A tuple of diffusion coefficients. The length of the tuple must match
        the number of dimensions of the grid.
    `diff_module` : `fr.grid.DiffModule | None`, (default=None)
        Differentiation module to use. If None, the differentiation module of
        the grid is used.
    `interp_module` : `fr.grid.InterpolationModule | None`, (default=None)
        Interpolation module to interpolate the diffusion coefficients.
    `name` : `str`, (default="Harmonic Diffusion")
        Name of the module.
    """
    _dynamic_attributes = ["mset", "_diffusion_coefficients"]

    def __init__(self,
                 field_flags: list[str],
                 diffusion_coefficients: list[float | fr.FieldVariable],
                 diff_module: fr.grid.DiffModule | None = None, 
                 interp_module: fr.grid.InterpolationModule | None = None,
                 name: str = "Biharmonic Diffusion"):
        super().__init__(name)
        self.field_flags = field_flags

        self.harmonic = fr.modules.closures.HarmonicDiffusion(
            field_flags=field_flags,
            diffusion_coefficients=_get_coefficients(diffusion_coefficients),
            diff_module=diff_module,
            interp_module=interp_module)

        self.diffusion_coefficients = diffusion_coefficients
        self.diff_module = diff_module
        self.interp_module = interp_module
        return

    @fr.modules.setup_module
    def setup(self):
        # setup the differentiation modules
        if self.diff_module is None:
            self.diff_module = self.mset.grid._diff_mod
        else:
            self.diff_module.setup(mset=self.mset)
        if self.interp_module is None:
            self.interp_module = self.mset.grid._interp_mod
        else:
            self.interp_module.setup(mset=self.mset)
        self.harmonic.setup(mset=self.mset)
        return

    @fr.utils.jaxjit
    def diffusion_operator(self, u: fr.FieldVariable) -> fr.FieldVariable:
        r"""
        Applies the biharmonic diffusion operator on a scalar field :math:`u`.
        """
        # apply the first harmonic diffusion operator
        div1 = self.harmonic.diffusion_operator(u)
        # apply the second harmonic diffusion operator
        div2 = self.harmonic.diffusion_operator(div1)
        return div2
    
    @fr.utils.jaxjit
    def diffuse(self, z: fr.StateBase, dz: fr.StateBase) -> fr.StateBase:
        # loop over all fields
        for name, field in z.fields.items():
            if not any([field.flags[flag] for flag in self.field_flags]):
                # skip the field if it does not have any of the field flags
                continue

            # apply the diffusion operator
            dz.fields[name] -= self.diffusion_operator(field)
        return dz

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz = self.diffuse(mz.z, mz.dz)
        return mz


    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------

    @property
    def field_flags(self) -> list[str]:
        """A list of field flags that indicate which fields should be diffused."""
        return self._field_flags

    @field_flags.setter
    def field_flags(self, value):
        self._field_flags = value
        return

    @property
    def diff_module(self) -> fr.grid.DiffModule:
        """The differentiation module."""
        return self._diff_module
    
    @diff_module.setter
    def diff_module(self, value):
        self.harmonic.diff_module = value
        self._diff_module = value
        return

    @property
    def interp_module(self) -> fr.grid.InterpolationModule:
        """The interpolation module to interpolate the diffusion coefficients."""
        return self._interp_module

    @interp_module.setter
    def interp_module(self, value):
        self.harmonic.interp_module = value
        self._interp_module = value
        return

    @property
    def diffusion_coefficients(self) -> list[float | fr.FieldVariable]:
        """A list of diffusion coefficients."""
        return self._diffusion_coefficients
    
    @diffusion_coefficients.setter
    def diffusion_coefficients(self, value):
        self.harmonic.diffusion_coefficients = _get_coefficients(value)
        self._diffusion_coefficients = value
        return
    
fr.utils.jaxify_class(BiharmonicDiffusion)