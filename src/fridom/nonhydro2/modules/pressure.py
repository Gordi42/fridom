"""The spectral pressure Poisson solve.

Description
-----------
The nonhydrostatic incompressibility constraint is enforced by the
fractional-step projection ``div(grad p) = div(u*)`` then
``u = u* - grad p`` (03 §5.6, project-the-state). The inverse is the
pseudo-inverse of the discrete Laplacian's diagonal ``Symbol``: the
eigenvalue diagonal is assembled on the transformed coefficient space
and inverted via ``Symbol.inverse`` (the ``k = 0`` nullspace
regularized by its exact structural-zero test — the mean-pressure
gauge), retiring the hand-rolled ``jnp.where`` pseudo-inverse.

The discrete eigenvalue matches the C-grid ``Divergence()@Gradient()``
chain exactly: forward difference (center -> face) has eigenvalue
``(e^{ik dx} - 1)/dx`` and backward difference (face -> center)
``(1 - e^{-ik dx})/dx``; their product is ``-2(1 - cos k dx)/dx^2 =
-khat^2``. Inverting therefore drives the *discrete* divergence to
machine zero. ``dsqr`` enters only through the vertical term
``khat_z^2 / dsqr`` and is read live in-step (it is not factorable
into grid x parameter — D2.4 V-N), so the diagonal is (re)assembled
per solve rather than materialized once by a static ``SpectralSolve``.

The eigenvalue is assembled directly on the transformed coefficient
factors (not via ``Laplacian().eigenvalues``): the composed C-grid FD
symbol tags every axis with an independent half spectrum, incompatible
with the multi-axis rfftn coefficient layout ``(half, full, full)``
this solve transforms into — reconciling the two is a follow-up.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework2.grid.operators.fourier import Fourier
from fridom.framework2.grid.operators.spectral import fourier_wavenumbers
from fridom.framework2.grid.operators.symbol import Symbol
from fridom.framework2.grid.spaces.coefficient import FourierSpace

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


def discrete_laplace_symbol(
    space: SpaceLike, *, vertical: str, dsqr: jax.Array,
) -> Symbol:
    r"""Discrete Laplacian diagonal ``-(khat_x^2 + khat_y^2 + khat_z^2/dsqr)``.

    Description
    -----------
    ``space`` is the coefficient (Fourier) space of the transformed
    divergence. Each factor contributes the discrete second-difference
    eigenvalue ``khat^2 = 2(1 - cos k dx)/dx^2`` reshaped onto its axis;
    the vertical factor is weighted by ``1 / dsqr``. The result is the
    (negative) discrete Laplacian eigenvalue diagonal, wrapped as a
    ``Symbol`` so the caller inverts it through the diagonal-operator
    algebra (``Symbol.inverse``).

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
    Symbol
        The discrete Laplacian diagonal on the bare coefficient space.
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
    return Symbol(bare, -total)


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
        laplace = discrete_laplace_symbol(
            div_hat.function_space, vertical=vertical, dsqr=dsqr)
        # ``Symbol.inverse`` regularizes the ``k = 0`` nullspace (the
        # mean-pressure gauge) via its exact structural-zero test
        p_hat = laplace.inverse()(div_hat)
        return self._fourier.backward(p_hat).real
