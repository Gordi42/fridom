"""Harmonic diffusion module."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("_diffusion_coefficients",))
class HarmonicDiffusion(fr.modules.Module):

    r"""
    Harmonic diffusion module.

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
    field_flags : list[str]
        A list of strings that indicate which fields should be diffused.
        For example, if `field_flags=["ENABLE_MIXING"]`, all fields with the
        flag "ENABLE_MIXING" will be diffused. For more information on possible
        flags, see :py:mod:`fridom.framework.ScalarField`.
    diffusion_coefficients : tuple[float | fr.ScalarField]
        A tuple of diffusion coefficients. The length of the tuple must match
        the number of dimensions of the grid.
    name : str, (default="Harmonic Diffusion")
        Name of the module.

    """

    name = "Harmonic Diffusion"
    def __init__(self,
                 field_flags: list[str],
                 diffusion_coefficients: list[float | fr.ScalarField]) -> None:
        super().__init__()
        self.field_flags = field_flags
        self.diffusion_coefficients = diffusion_coefficients

    def diffusion_operator(self, u: fr.ScalarField) -> fr.ScalarField:
        r"""Apply the harmonic diffusion operator on a scalar field :math:`u`."""
        # compute the gradient of the field
        grad_u = list(self.diff_module.grad(u))
        # multiply the gradient with the diffusion coefficients
        for i, coeff in enumerate(self.diffusion_coefficients):
            if isinstance(coeff, fr.ScalarField):
                # interpolate the diffusion coefficient to the position of the field
                c = self.interp_module.interpolate(coeff, grad_u[i].position)
            else:
                c = coeff
            grad_u[i] *= c
            # apply the water mask to the gradient
            grad_u[i] = grad_u[i].apply_water_mask()
        # compute the divergence of the gradient
        div = self.diff_module.div(tuple(grad_u))
        # apply the boundary conditions
        return div.apply_water_mask()

    @fr.utils.jaxjit
    def diffuse(self, z: fr.VectorField, dz: fr.VectorField) -> fr.VectorField:
        # loop over all fields
        #TODO(Silvano): Use new vector field methods
        for name, field in z.fields.items():
            if not any([field.flags[flag] for flag in self.field_flags]):
    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        for f in mz.z:
            if not any(f.flags[flag] for flag in self.field_flags):
                # skip the field if it does not have any of the field flags
                continue

            # apply the diffusion operator
            mz.dz[f.name] += self.diffusion_operator(f)
        return mz

    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------

    @property
    def field_flags(self) -> list[str]:
        """A list of field flags that indicate which fields should be diffused."""
        return self._field_flags

    @field_flags.setter
    def field_flags(self, value: list[str]) -> None:
        self._field_flags = value

    @property
    def diffusion_coefficients(self) -> list[float | fr.ScalarField]:
        """A list of diffusion coefficients."""
        return self._diffusion_coefficients

    @diffusion_coefficients.setter
    def diffusion_coefficients(self, value: list[float | fr.ScalarField]) -> None:
        self._diffusion_coefficients = value
