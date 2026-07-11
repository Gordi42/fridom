"""
Pure array-level stencil kernels for staggered grids.

Description
-----------
The numerical core wrapped by the Wave-2 operator classes
(``FiniteDifference``, ``LinearInterp``): pure functions on arrays —
no spaces, no fields, no operators. All kernels follow the
slice-based shape-changing stencil rule of
``notes/framework2/02_rules.md`` section 3.5: a ``size``-point kernel
maps length m to m - size + 1 along the stencil axis by summing
weighted slice windows into the (halo-extended) input — never a
``roll``, never a gather. Output entry ``i`` sits at the midpoint of
the input window ``i : i + size``, the half-integer staggering
target.

Periodicity and boundaries live entirely in the caller's halo fill
(section 3.5): the caller slices the halo-extended local array so
that the input window length encodes the target space, and the same
kernel realizes ``Center -> Right``, ``Right -> Center``,
``Outer(n+1) -> Center(n)``, and ``Center(n) -> Inner(n-1)`` — the
staggering direction is pure window alignment, chosen by the caller.

Stencil coefficients are plain Python floats computed exactly with
``fractions`` on the host and baked into the jaxpr as constants
(never ``jnp``); a scalar ``spacing`` folds into the weights at
trace time. Every kernel is one fused arithmetic expression over
static-shape slice views with no data-dependent Python branching, so
the kernels are shard_map-compatible by construction.
"""
from __future__ import annotations

from fractions import Fraction
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

# minimal supported finite-difference order (a two-point stencil)
_MIN_ORDER = 2


# ================================================================
#  Coefficient computation (host-side, exact)
# ================================================================
def _solve_exact(
    matrix: list[list[Fraction]],
    rhs: list[Fraction],
) -> list[Fraction]:
    """
    Solve a linear system exactly over the rationals.

    Description
    -----------
    Gauss-Jordan elimination in ``Fraction`` arithmetic; used at
    kernel-construction time only (tiny systems), so exactness beats
    speed.

    Parameters
    ----------
    matrix : list[list[Fraction]]
        The (nonsingular) coefficient matrix.
    rhs : list[Fraction]
        The right-hand side.

    Returns
    -------
    list[Fraction]
        The exact solution vector.
    """
    n = len(rhs)
    aug = [[*row, b] for row, b in zip(matrix, rhs, strict=True)]
    for col in range(n):
        pivot = next(row for row in range(col, n) if aug[row][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [entry / div for entry in aug[col]]
        for row in range(n):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col]
                aug[row] = [
                    entry - factor * lead
                    for entry, lead in zip(aug[row], aug[col], strict=True)
                ]
    return [aug[row][n] for row in range(n)]


@cache
def staggered_diff_weights(order: int) -> tuple[float, ...]:
    """
    Finite-difference weights for the staggered derivative.

    Description
    -----------
    Weights of the ``order``-point stencil that evaluates the first
    derivative at the midpoint of the stencil, for unit spacing: the
    stencil nodes sit at the half-integer offsets
    ``-(order - 1) / 2, ..., (order - 1) / 2``. By midpoint symmetry
    the ``order``-point stencil is accurate (and, on polynomials,
    exact) to degree ``order`` — the classic staggered
    even-order family (order 2: ``(-1, 1)``; order 4:
    ``(1/24, -27/24, 27/24, -1/24)``).

    The weights are solved exactly over the rationals
    (moment/Vandermonde system) and returned as floats — static
    Python constants baked into the jaxpr, never traced.

    Parameters
    ----------
    order : int
        The (even, >= 2) order of accuracy; also the stencil size.

    Returns
    -------
    tuple[float, ...]
        The ``order`` stencil weights, left to right.

    Raises
    ------
    ValueError
        If order is odd or smaller than 2.
    """
    if order < _MIN_ORDER or order % 2 != 0:
        raise ValueError(
            f"staggered finite differences require an even order >= 2, "
            f"got {order}")
    offsets = [Fraction(2 * j - order + 1, 2) for j in range(order)]
    matrix = [[offset**row for offset in offsets] for row in range(order)]
    rhs = [Fraction(int(row == 1)) for row in range(order)]
    return tuple(float(w) for w in _solve_exact(matrix, rhs))


# ================================================================
#  Slice-window machinery
# ================================================================
def _stencil_windows(
    arr: Array,
    axis: int,
    size: int,
) -> tuple[Array, ...]:
    """
    Slice the ``size`` stencil windows of ``arr`` along ``axis``.

    Description
    -----------
    Window ``j`` is ``arr[..., j : j + m - size + 1, ...]`` along the
    stencil axis (m the input length), so a weighted sum of the
    windows applies the stencil at every output point — the
    slice-based shape-changing stencil of section 3.5.

    Parameters
    ----------
    arr : Array
        The input array (halo-extended by the caller as needed).
    axis : int
        The stencil axis; may be negative.
    size : int
        The stencil size (number of windows).

    Returns
    -------
    tuple[Array, ...]
        The ``size`` windows, each of length m - size + 1 along
        ``axis``.

    Raises
    ------
    ValueError
        If the axis is shorter than the stencil.
    """
    out_len = arr.shape[axis] - size + 1
    if out_len < 1:
        raise ValueError(
            f"axis {axis} of length {arr.shape[axis]} is shorter than "
            f"the {size}-point stencil")
    index = [slice(None)] * arr.ndim
    windows = []
    for offset in range(size):
        index[axis] = slice(offset, offset + out_len)
        windows.append(arr[tuple(index)])
    return tuple(windows)


def apply_stencil(
    arr: Array,
    axis: int,
    weights: tuple[float, ...],
) -> Array:
    """
    Apply a 1D stencil with static weights along an axis.

    Description
    -----------
    The generic slice-based kernel: the weighted sum of the
    ``len(weights)`` slice windows, mapping length m to
    m - len(weights) + 1 along ``axis``. The weights must be static
    Python floats (they are baked into the jaxpr); one fused
    arithmetic expression, no data-dependent branching.

    Parameters
    ----------
    arr : Array
        The input array (halo-extended by the caller as needed).
    axis : int
        The stencil axis; may be negative.
    weights : tuple[float, ...]
        The static stencil weights, left to right.

    Returns
    -------
    Array
        The stencil output; ``axis`` shrinks by len(weights) - 1.
    """
    windows = _stencil_windows(arr, axis, len(weights))
    out = weights[0] * windows[0]
    for weight, window in zip(weights[1:], windows[1:], strict=True):
        out = out + weight * window
    return out


# ================================================================
#  Kernels
# ================================================================
def staggered_diff(
    arr: Array,
    axis: int,
    spacing: float | Array = 1.0,
    order: int = 2,
) -> Array:
    """
    Staggered finite-difference derivative along an axis.

    Description
    -----------
    Differentiates onto the half-integer points: output entry ``i``
    approximates the derivative at the midpoint of the input window
    ``i : i + order``, with the length shrinking by order - 1 along
    ``axis`` (section 3.5). Order 2 is the two-point
    forward/backward difference of the staggering pairs
    ``Center -> Right`` / ``Right -> Center`` (and their bounded
    ``Outer -> Center`` / ``Center -> Inner`` variants); the caller's
    window alignment picks the direction.

    A scalar ``spacing`` is folded into the static weights at trace
    time (a uniform mesh constant-folds completely); an array
    ``spacing`` — the dual center-to-center measure field of a
    nonuniform mesh — divides the unit-spacing stencil output.

    Parameters
    ----------
    arr : Array
        The input array (halo-extended by the caller as needed).
    axis : int
        The stencil axis; may be negative.
    spacing : float | Array, optional
        The grid-spacing denominator: a Python scalar, or a measure
        array broadcastable against the output (default: 1.0).
    order : int, optional
        The even order of accuracy = stencil size (default: 2).

    Returns
    -------
    Array
        The derivative; ``axis`` shrinks by order - 1.
    """
    weights = staggered_diff_weights(order)
    if isinstance(spacing, int | float):
        scaled = tuple(weight / spacing for weight in weights)
        return apply_stencil(arr, axis, scaled)
    return apply_stencil(arr, axis, weights) / spacing


def linear_interp(arr: Array, axis: int) -> Array:
    """
    Two-point linear interpolation onto the staggered points.

    Description
    -----------
    Output entry ``i`` is the mean of input entries ``i`` and
    ``i + 1`` — the second-order interpolation between staggered
    positions, shrinking the axis by one (section 3.5). As with
    :func:`staggered_diff`, the staggering direction
    (``Center <-> Right``, ``Outer -> Center``, ``Center -> Inner``)
    is the caller's window alignment.

    Parameters
    ----------
    arr : Array
        The input array (halo-extended by the caller as needed).
    axis : int
        The stencil axis; may be negative.

    Returns
    -------
    Array
        The interpolated values; ``axis`` shrinks by one.
    """
    return apply_stencil(arr, axis, (0.5, 0.5))
