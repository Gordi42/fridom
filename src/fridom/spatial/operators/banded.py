"""
Banded (tridiagonal) solve primitives on a device-local axis.

Description
-----------
The grid-free kernel shared by the IMEX implicit vertical-diffusion
solve (``model/implicit.py``) and the mixed Fourier x Chebyshev
spectral banded solve (``operators/spectral_solve.py``, designed-for):
assemble a dense band, apply it along one storage axis (batched over
the off-axis columns), and solve ``system @ x = rhs`` along that axis.

Iteration-1 scope: the band is materialized dense and solved with
``jnp.linalg.solve`` (batched over the off-axis columns); the solve
axis must stay device-local (a tridiagonal is serial along it). A
Thomas / ``jax.lax.linalg.tridiagonal_solve`` kernel is the production
optimization; both respect the same true-shape ``data`` /
``with_data`` halo contract of the caller.
"""
# Wave 9B: lifted out of model/implicit.py (plan section 5, decision C)
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_real

if TYPE_CHECKING:  # pragma: no cover
    import jax


# ================================================================
#  Band assembly
# ================================================================
def second_difference_matrix(coords: jax.Array) -> jax.Array:
    r"""
    Dense ``(N, N)`` Neumann second-difference ``d2/dz2`` on uniform nodes.

    Description
    -----------
    The tridiagonal second-difference stencil with zero-flux (Neumann)
    boundary rows — ``-1`` on the two corner diagonals instead of
    ``-2`` — divided by ``dz^2`` (uniform spacing inferred from the
    first two nodes). Grid-free: it reads only the 1D node line, so the
    same band feeds the diffusion operator (scaled by ``kappa``) and,
    with a per-mode scalar shift, the spectral banded z-solve.

    Parameters
    ----------
    coords : jax.Array
        The evaluation nodes along the solve axis (any shape; flattened
        to the 1D line).

    Returns
    -------
    jax.Array
        The ``(N, N)`` second-difference matrix.
    """
    real = dtype_real()
    line = jnp.reshape(jnp.asarray(coords), (-1,)).astype(real)
    size = line.shape[0]
    dz = line[1] - line[0]
    main = jnp.full((size,), -2.0, dtype=real)
    main = main.at[0].set(-1.0).at[size - 1].set(-1.0)
    off = jnp.ones((size - 1,), dtype=real)
    return (jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)
            ) / (dz * dz)


# ================================================================
#  Batched apply / solve along one storage axis
# ================================================================
def apply_along_axis(
    operator: jax.Array, data: jax.Array, axis_index: int,
) -> jax.Array:
    """
    Apply the ``(N, N)`` band along ``axis_index`` (batched columns).

    Parameters
    ----------
    operator : jax.Array
        The ``(N, N)`` band matrix.
    data : jax.Array
        The operand array; its ``axis_index`` axis has length ``N``.
    axis_index : int
        The storage-frame index of the solve axis.

    Returns
    -------
    jax.Array
        ``operator @ data`` along ``axis_index`` (same shape as ``data``).
    """
    moved = jnp.moveaxis(data, axis_index, -1)
    shape = moved.shape
    flat = moved.reshape(-1, shape[-1])
    out = flat @ operator.T
    return jnp.moveaxis(out.reshape(shape), -1, axis_index)


def solve_along_axis(
    system: jax.Array, data: jax.Array, axis_index: int,
) -> jax.Array:
    """
    Solve ``system @ x = data`` along ``axis_index`` (batched columns).

    Parameters
    ----------
    system : jax.Array
        The ``(N, N)`` band system to invert.
    data : jax.Array
        The right-hand side; its ``axis_index`` axis has length ``N``.
    axis_index : int
        The storage-frame index of the solve axis.

    Returns
    -------
    jax.Array
        The solution ``x`` (same shape as ``data``).
    """
    moved = jnp.moveaxis(data, axis_index, -1)
    shape = moved.shape
    flat = moved.reshape(-1, shape[-1])
    solved = jnp.linalg.solve(system, flat.T).T
    return jnp.moveaxis(solved.reshape(shape), -1, axis_index)
