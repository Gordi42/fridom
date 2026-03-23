"""Biharmonic diffusion module."""
from __future__ import annotations

import fridom.framework as fr


@fr.utils.jaxify
class BiharmonicDiffusion(fr.modules.closures.HarmonicDiffusion):

    r"""
    Biharmonic diffusion module.

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
    field_flags : list[str]
        A list of strings that indicate which fields should be diffused.
        For example, if `field_flags=["ENABLE_MIXING"]`, all fields with the
        flag "ENABLE_MIXING" will be diffused. For more information on possible
        flags, see :py:mod:`fridom.framework.ScalarField`.
    diffusion_coefficients : tuple[float | fr.ScalarField]
        A tuple of diffusion coefficients. The length of the tuple must match
        the number of dimensions of the grid.

    """

    name = "Biharmonic Diffusion"
    def diffusion_operator(self, u: fr.ScalarField) -> fr.ScalarField:
        """Apply the biharmonic diffusion operator on a scalar field :math:`u`."""
        # apply the first harmonic diffusion operator
        div1 = super().diffusion_operator(u)
        # apply the second harmonic diffusion operator
        div2 = super().diffusion_operator(div1)
        return - div2 * self._sign

    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------
    @property
    def diffusion_coefficients(self) -> list[float | fr.ScalarField]:
        """A list of diffusion coefficients."""
        return self._diffusion_coefficients

    @diffusion_coefficients.setter
    def diffusion_coefficients(self, value: tuple[float | fr.ScalarField]) -> None:
        # we need to take the square root of the diffusion coefficients
        ncp = fr.config.ncp
        coeffs = []
        for coeff in value:
            if isinstance(coeff, fr.ScalarField):
                self._sign = ncp.sign(coeff.arr)
                kappa = ncp.sqrt(ncp.abs(coeff.arr))
                kappa = fr.ScalarField(mset=coeff.mset,
                                         arr=kappa,
                                         mdata=coeff.mdata)
            else:
                self._sign = ncp.sign(coeff)
                kappa = ncp.sqrt(ncp.abs(coeff))
            coeffs.append(kappa)
        self._diffusion_coefficients = coeffs
