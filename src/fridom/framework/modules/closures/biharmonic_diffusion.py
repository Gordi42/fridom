"""Biharmonic diffusion module."""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp

import fridom.framework as fr


@partial(fr.utils.jaxify, dynamic=("_original_coefficients", "_sign"))
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

    where we use the biharmonic diffusion coefficient
    :math:`\sqrt{|\kappa_i|}`. The index :math:`i` refers to the direction
    of the diffusion.

    Parameters
    ----------
    field_flags : list[str]
        A list of strings that indicate which fields should be diffused.
        For example, if `field_flags=["ENABLE_MIXING"]`, all fields with the
        flag "ENABLE_MIXING" will be diffused. For more information on possible
        flags, see :py:mod:`fridom.framework.ScalarField`.
    diffusion_coefficients : tuple[float | fr.ScalarField]
        A tuple of diffusion coefficients. The length of the tuple must match
        the number of dimensions of the grid. The coefficients must not have
        mixed signs.

    """

    name = "Biharmonic Diffusion"
    def __init__(self,
                 field_flags: list[str],
                 diffusion_coefficients: list[float | fr.ScalarField],
                 ) -> None:
        super().__init__(field_flags=field_flags,
                         diffusion_coefficients=diffusion_coefficients)
        # the biharmonic operator applies two derivatives in each
        # direction before the fields are synchronized again
        self.required_halo = 2

    def diffusion_operator(self, u: fr.ScalarField) -> fr.ScalarField:
        r"""Apply the biharmonic diffusion operator on a field :math:`u`."""
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
        return self._original_coefficients

    @diffusion_coefficients.setter
    def diffusion_coefficients(
            self, value: tuple[float | fr.ScalarField]) -> None:
        # the harmonic operator is applied twice, hence we store the
        # square root of the diffusion coefficients
        coeffs = []
        sign = 0
        for coeff in value:
            arr = coeff.arr if isinstance(coeff, fr.ScalarField) else coeff
            coeff_sign = jnp.sign(arr)
            if bool(jnp.any(sign * coeff_sign < 0)):
                msg = ("The biharmonic diffusion coefficients must not "
                       "have mixed signs.")
                raise ValueError(msg)
            # zero coefficients do not contribute to the sign
            sign = jnp.where(coeff_sign == 0, sign, coeff_sign)
            kappa = jnp.sqrt(jnp.abs(arr))
            if isinstance(coeff, fr.ScalarField):
                kappa = fr.ScalarField(mset=coeff.mset,
                                       arr=kappa,
                                       mdata=coeff.mdata)
            coeffs.append(kappa)
        self._sign = sign
        self._original_coefficients = list(value)
        self._diffusion_coefficients = coeffs
