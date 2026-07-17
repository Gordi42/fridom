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

The multigrid vertical-line smoother (``operators/multigrid.py``,
decision MG-D7) needs a *diagonal-varying* tridiagonal per off-axis
column, so it does not materialize the dense band: the
:func:`tridiagonal_solve_along_axis` Thomas kernel solves one distinct
symmetric tridiagonal per column in a single :func:`jax.lax.scan`
(forward elimination, back substitution), batched over the off-axis
columns. It is reverse-mode differentiable through the ``scan`` (no
``custom_vjp``) and does **no pivoting**: the caller guarantees a
non-singular column (the line smoother substitutes ``diag -> 1`` on
dry/zero-diagonal cells). ``jax.lax.linalg.tridiagonal_solve`` is
deliberately avoided — its autodiff / batching support is
backend-uneven; the ``scan`` Thomas is the portable kernel.
"""
# Wave 9B: lifted out of model/implicit.py (plan section 5, decision C)
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import lax

from fridom.framework.utils import dtype_real

if TYPE_CHECKING:  # pragma: no cover
    import jax


# ================================================================
#  Band assembly
# ================================================================
#: the per-side corner main-diagonal value each boundary condition
#: contributes to :func:`second_difference_matrix`. Neumann (zero-flux,
#: even ghost mirror ``u_{-1} = u_0``) gives ``-1``; Dirichlet (no-slip,
#: odd ghost mirror across the half-cell ``u_{-1} = -u_0``) gives ``-3``
#: (the wall-adjacent row becomes ``(u_1 - 3 u_0) / dz^2``).
_CORNER: dict[str, float] = {"neumann": -1.0, "dirichlet": -3.0}


def validate_boundary_conditions(
    bc: tuple[str, str],
) -> tuple[str, str]:
    """
    Return the ``(low, high)`` boundary-condition pair or raise.

    Description
    -----------
    The single source of truth for the per-side boundary vocabulary of
    :func:`second_difference_matrix` (and the ``VerticalDiffusion``
    operator that carries one): each side must be ``"neumann"`` (the
    zero-flux corner) or ``"dirichlet"`` (the no-slip odd-mirror
    corner). The choice is static (host-side), so it is validated in
    plain Python — never on a traced value.

    Parameters
    ----------
    bc : tuple[str, str]
        The per-side ``(low, high)`` boundary conditions.

    Returns
    -------
    tuple[str, str]
        The validated ``(low, high)`` pair.

    Raises
    ------
    ValueError
        If `bc` is not a length-2 pair of accepted condition names.
    """
    pair = tuple(bc)
    valid = tuple(_CORNER)
    if len(pair) != 2 or any(  # noqa: PLR2004 — a (low, high) pair
            side not in _CORNER for side in pair):
        raise ValueError(
            "boundary conditions must be a (low, high) pair drawn from "
            f"{valid}, got {bc!r}")
    return pair


def second_difference_matrix(
    coords: jax.Array,
    bc: tuple[str, str] = ("neumann", "neumann"),
) -> jax.Array:
    r"""
    Dense ``(N, N)`` second-difference ``d2/dz2`` on uniform nodes.

    Description
    -----------
    The tridiagonal second-difference stencil divided by ``dz^2``
    (uniform spacing inferred from the first two nodes) with per-side
    boundary rows selected by `bc`. A **Neumann** (zero-flux) side puts
    ``-1`` on its corner diagonal (the even ghost mirror ``u_{-1} =
    u_0``); a **Dirichlet** (no-slip) side puts ``-3`` (the odd ghost
    mirror across the half-cell ``u_{-1} = -u_0``, so the wall-adjacent
    row is ``(u_1 - 3 u_0) / dz^2``). The default ``("neumann",
    "neumann")`` reproduces the historical zero-flux band. Grid-free: it
    reads only the 1D node line, so the same band feeds the diffusion
    operator (scaled by ``kappa``) and, with a per-mode scalar shift,
    the spectral banded z-solve; the ``bc`` choice is static (host-side)
    and jit-friendly (it selects Python constants, never a traced
    branch).

    Parameters
    ----------
    coords : jax.Array
        The evaluation nodes along the solve axis (any shape; flattened
        to the 1D line).
    bc : tuple[str, str], optional
        The per-side ``(low, high)`` boundary conditions, each
        ``"neumann"`` or ``"dirichlet"`` (default: ``("neumann",
        "neumann")``).

    Returns
    -------
    jax.Array
        The ``(N, N)`` second-difference matrix.

    Raises
    ------
    ValueError
        If `bc` is not a length-2 pair of accepted condition names.
    """
    low, high = validate_boundary_conditions(bc)
    real = dtype_real()
    line = jnp.reshape(jnp.asarray(coords), (-1,)).astype(real)
    size = line.shape[0]
    dz = line[1] - line[0]
    main = jnp.full((size,), -2.0, dtype=real)
    main = main.at[0].set(_CORNER[low]).at[size - 1].set(_CORNER[high])
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


# ================================================================
#  Per-column Thomas solve (distinct tridiagonal per off-axis column)
# ================================================================
def tridiagonal_solve_along_axis(
    lower: jax.Array,
    diag: jax.Array,
    upper: jax.Array,
    data: jax.Array,
    axis_index: int,
) -> jax.Array:
    r"""
    Solve one symmetric tridiagonal per column along ``axis_index``.

    Description
    -----------
    The batched Thomas algorithm: a distinct tridiagonal ``T`` (the
    per-cell ``lower``/``diag``/``upper`` bands) is solved against
    ``data`` along ``axis_index``, batched over every off-axis column.
    Unlike :func:`solve_along_axis` no dense band is materialized —
    the multigrid vertical-line smoother's ``T`` varies from column to
    column (MG-D7). Implemented as one forward-elimination
    :func:`jax.lax.scan` (the modified super-diagonal and right-hand
    side) and one reverse back-substitution ``scan``, so the whole
    solve is reverse-mode differentiable with no ``custom_vjp``.

    The band arrays broadcast against ``data``: ``lower[i]`` is the
    sub-diagonal coupling of cell ``i`` to ``i - 1`` and ``upper[i]``
    the super-diagonal coupling to ``i + 1``, both along
    ``axis_index``. The Neumann ends ``lower[0]`` and ``upper[N - 1]``
    are unused (the recurrence seeds them with a zero carry), so the
    caller may leave them at zero.

    Precondition: **no pivoting** is performed, so every column's
    tridiagonal must be non-singular (diagonally dominant is
    sufficient). A dry / zero-diagonal column must be sanitized by the
    caller — the vertical-line smoother substitutes ``diag -> 1`` and
    ``rhs -> 0`` there through the double-``jnp.where`` guard, which
    also keeps the reverse pass NaN-free.

    Parameters
    ----------
    lower : jax.Array
        The sub-diagonal band (same shape as ``data``; ``lower[0]``
        along ``axis_index`` unused).
    diag : jax.Array
        The main diagonal band (same shape as ``data``; non-zero on
        every solved cell).
    upper : jax.Array
        The super-diagonal band (same shape as ``data``;
        ``upper[N - 1]`` along ``axis_index`` unused).
    data : jax.Array
        The right-hand side; its ``axis_index`` axis has length ``N``.
    axis_index : int
        The storage-frame index of the solve axis.

    Returns
    -------
    jax.Array
        The per-column solution ``x`` (same shape as ``data``).
    """
    lo = jnp.moveaxis(lower, axis_index, 0)
    di = jnp.moveaxis(diag, axis_index, 0)
    up = jnp.moveaxis(upper, axis_index, 0)
    rhs = jnp.moveaxis(data, axis_index, 0)
    shape = rhs.shape
    size = shape[0]
    lo = jnp.broadcast_to(lo, shape).reshape(size, -1)
    di = jnp.broadcast_to(di, shape).reshape(size, -1)
    up = jnp.broadcast_to(up, shape).reshape(size, -1)
    rhs = rhs.reshape(size, -1)
    batch = rhs.shape[1]
    zero = jnp.zeros((batch,), dtype=rhs.dtype)

    def eliminate(
        carry: tuple[jax.Array, jax.Array],
        row: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]
    ]:
        c_prev, d_prev = carry
        low, dia, upp, right = row
        denom = dia - low * c_prev
        c_new = upp / denom
        d_new = (right - low * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    _, (c_star, d_star) = lax.scan(
        eliminate, (zero, zero), (lo, di, up, rhs))

    def substitute(
        x_next: jax.Array, row: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        c_new, d_new = row
        x_cur = d_new - c_new * x_next
        return x_cur, x_cur

    _, solved = lax.scan(
        substitute, zero, (c_star, d_star), reverse=True)
    solved = solved.reshape(shape)
    return jnp.moveaxis(solved, 0, axis_index)
