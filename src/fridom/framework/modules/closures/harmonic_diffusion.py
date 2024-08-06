import fridom.framework as fr


class HarmonicDiffusion(fr.modules.Module):
    r"""
    Harmonic diffusion module

    Description
    -----------
    The harmonic diffusion operator :math:`\mathcal{H}` on a scalar field 
    :math:`u` is given by:

    .. math::
        \mathcal{H}(u) = \nabla \cdot \left (\mathbf{A} \cdot \nabla u \right)

    with the diagonal diffusion tensor :math:`\mathbf{A}` given by:
    
    .. math::
        \mathbf{A} = \begin{pmatrix} \kappa_1 & \dots  & 0 \\ 
                                     \vdots   & \ddots & \vdots \\ 
                                     0        & \dots  & \kappa_n \end{pmatrix}

    where :math:`\kappa_i` is the harmonic diffusion coefficient in the 
    :math:`i`-th direction.

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
    `name` : `str`, (default="Harmonic Diffusion")
        Name of the module.
    """
    _dynamic_attributes = ["mset", "diffusion_coefficients"]
    def __init__(self, 
                 field_flags: list[str], 
                 diffusion_coefficients: list[float | fr.FieldVariable], 
                 diff_module: fr.grid.DiffModule | None = None, 
                 interp_module: fr.grid.InterpolationModule | None = None,
                 name: str = "Harmonic Diffusion"):
        super().__init__(name)
        self.field_flags = field_flags
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
        return

    @fr.utils.jaxjit
    def diffusion_operator(self, u: fr.FieldVariable) -> fr.FieldVariable:
        r"""
        Applies the harmonic diffusion operator on a scalar field :math:`u`.
        """
        # compute the gradient of the field
        grad_u = list(self.diff_module.grad(u))
        # multiply the gradient with the diffusion coefficients
        for i, coeff in enumerate(self.diffusion_coefficients):
            if isinstance(coeff, fr.FieldVariable):
                # interpolate the diffusion coefficient to the position of the field
                c = self.interp_module.interpolate(coeff, grad_u[i].position)
            else:
                c = coeff
            grad_u[i] *= c
        # compute the divergence of the gradient
        div_u = self.diff_module.div(tuple(grad_u))
        return div_u

    @fr.utils.jaxjit
    def diffuse(self, z: fr.StateBase, dz: fr.StateBase) -> fr.StateBase:
        # loop over all fields
        for name, field in z.fields.items():
            if not any([field.flags[flag] for flag in self.field_flags]):
                # skip the field if it does not have any of the field flags
                continue

            # apply the diffusion operator
            dz.fields[name] += self.diffusion_operator(field)
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
        self._diff_module = value
        return

    @property
    def interp_module(self) -> fr.grid.InterpolationModule:
        """The interpolation module to interpolate the diffusion coefficients."""
        return self._interp_module

    @interp_module.setter
    def interp_module(self, value):
        self._interp_module = value
        return

    @property
    def diffusion_coefficients(self) -> list[float | fr.FieldVariable]:
        """A list of diffusion coefficients."""
        return self._diffusion_coefficients
    
    @diffusion_coefficients.setter
    def diffusion_coefficients(self, value):
        self._diffusion_coefficients = value
        return

fr.utils.jaxify_class(HarmonicDiffusion)
