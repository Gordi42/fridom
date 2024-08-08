import fridom.framework as fr
from functools import partial


@partial(fr.utils.jaxify, dynamic=("_diffusion_coefficients",))
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
    `name` : `str`, (default="Harmonic Diffusion")
        Name of the module.
    """
    name = "Harmonic Diffusion"
    def __init__(self, 
                 field_flags: list[str], 
                 diffusion_coefficients: list[float | fr.FieldVariable]):
        super().__init__()
        self.field_flags = field_flags
        self.diffusion_coefficients = diffusion_coefficients
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
    def diffusion_coefficients(self) -> list[float | fr.FieldVariable]:
        """A list of diffusion coefficients."""
        return self._diffusion_coefficients
    
    @diffusion_coefficients.setter
    def diffusion_coefficients(self, value):
        self._diffusion_coefficients = value
        return
