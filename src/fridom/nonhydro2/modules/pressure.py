"""The spectral pressure Poisson solve.

Description
-----------
The nonhydrostatic incompressibility constraint is enforced by the
fractional-step projection ``div(grad p) = div(u*)`` then
``u = u* - grad p`` (03 §5.6, project-the-state). Framework2 lands no
Poisson/Symbol operator (``operators/symbol.py`` is a stub,
``SpectralDerivative.eigenvalues`` and ``Fourier.truncation_mask``
raise ``NotImplementedError``), so the inverse is assembled here from
the landed ``Fourier`` transform and the discrete finite-difference
eigenvalue.

The discrete eigenvalue matches the C-grid ``Divergence()@Gradient()``
chain exactly: forward difference (center -> face) has eigenvalue
``(e^{ik dx} - 1)/dx`` and backward difference (face -> center)
``(1 - e^{-ik dx})/dx``; their product is ``-2(1 - cos k dx)/dx^2 =
-khat^2``. Inverting with ``p_hat = -div_hat / khat^2`` therefore drives
the *discrete* divergence to machine zero. ``dsqr`` enters only through
the vertical term ``khat_z^2 / dsqr`` and is read live in-step (it is
not factorable into grid x parameter — D2.4 V-N).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework2.grid.operators.fourier import Fourier
from fridom.framework2.grid.operators.spectral import fourier_wavenumbers
from fridom.framework2.grid.spaces.coefficient import FourierSpace

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


def discrete_laplace_eigenvalues(
    space: SpaceLike, *, vertical: str, dsqr: jax.Array,
) -> jax.Array:
    r"""Assemble ``khat_x^2 + khat_y^2 + khat_z^2 / dsqr`` on ``space``.

    Description
    -----------
    ``space`` is the coefficient (Fourier) space of the transformed
    divergence. Each factor contributes the discrete second-difference
    eigenvalue ``khat^2 = 2(1 - cos k dx)/dx^2`` reshaped onto its axis;
    the vertical factor is weighted by ``1 / dsqr``.

    Parameters
    ----------
    space : SpaceLike
        The coefficient-space of the transformed field.
    vertical : str
        Coordinate name of the vertical (``dsqr``-weighted) axis.
    dsqr : jax.Array
        The live squared-aspect-ratio leaf.

    Returns
    -------
    jax.Array
        The eigenvalue array, broadcastable to the coefficient data.
    """
    bare = space.bare
    names = bare.names
    ndim = len(names)
    total = jnp.zeros((1,) * ndim)
    for index, name in enumerate(names):
        factor = bare.factor(name)
        if not isinstance(factor, FourierSpace):
            raise NotImplementedError(
                "the spectral pressure solve needs a Fourier factor on "
                f"every transformed axis; {name!r} resolved to "
                f"{factor!r} (use a periodic grid, or extend the solve "
                "with the trig transforms)")
        mesh = factor.mesh
        n = factor.origin.shape[0]
        length = mesh.extent[1] - mesh.extent[0]
        dx = length / n
        k = fourier_wavenumbers(factor)
        khat2 = 2.0 * (1.0 - jnp.cos(k * dx)) / dx**2
        shape = [1] * ndim
        shape[index] = khat2.shape[0]
        weight = 1.0 / dsqr if name == vertical else 1.0
        total = total + weight * khat2.reshape(shape)
    return total


class SpectralPressureSolver:

    """Grid-bound spectral solve of ``lap(p) = div`` on a Fourier grid.

    Description
    -----------
    Constructed at trace time inside the projection stage from the
    operand's grid; carries no state and is not a pytree leaf.
    """

    def __init__(self, grid: object) -> None:
        """Bind the all-axes Fourier transform on ``grid``."""
        self._fourier = Fourier(grid)

    def solve(
        self, div: object, *, vertical: str, dsqr: jax.Array,
    ) -> object:
        """Return ``p`` with ``div(grad p) = div`` (discrete, exact).

        Parameters
        ----------
        div : ScalarField
            The (cell-centered, real) divergence of the provisional
            velocity.
        vertical : str
            The vertical coordinate name.
        dsqr : jax.Array
            The live squared-aspect-ratio leaf.

        Returns
        -------
        ScalarField
            The pressure on the same (cell-centered) space as ``div``.
        """
        div_hat = self._fourier.forward(div)
        eig = discrete_laplace_eigenvalues(
            div_hat.function_space, vertical=vertical, dsqr=dsqr)
        # zero-mode mask (grid-static): the mean pressure is gauge
        zero = eig == 0.0
        inv = jnp.where(zero, 0.0, -1.0 / jnp.where(zero, 1.0, eig))
        p_hat = div_hat.with_data(div_hat.data * inv)
        return self._fourier.backward(p_hat).real
