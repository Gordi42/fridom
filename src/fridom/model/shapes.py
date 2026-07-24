"""Coordinate-named spatial shape builders (envelopes, patterns, ...)."""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax


# ================================================================
#  Gaussian shape (coordinate-named callable)
# ================================================================
def gaussian(
    pos: Mapping[str, float],
    width: float | Mapping[str, float],
) -> Callable[..., jax.Array]:
    r"""
    Build the Gaussian coordinate-named shape callable.

    Description
    -----------
    Returns the coordinate-named callable

    .. math::
        E(\boldsymbol{x}) =
            \prod_{i} \exp\left(-\frac{(x_i - p_i)^2}{w_i^2}\right)

    over the coordinates named in ``pos``. Its signature names exactly
    those coordinates, so the consuming factories (wave-packet
    envelopes, forcing patterns, ...) vary the shape along them and
    stay constant along every other axis.

    ``width`` is either a bare number — broadcast to every axis named
    in ``pos`` — or a mapping naming exactly the ``pos`` keys.

    Parameters
    ----------
    pos : Mapping[str, float]
        Shape centres, keyed by coordinate name.
    width : float | Mapping[str, float]
        Shape widths; a bare number broadcasts to every ``pos`` axis,
        a mapping must name the same keys as ``pos``.

    Returns
    -------
    Callable[..., jax.Array]
        The shape callable (keyword coordinates to values).

    Raises
    ------
    ValueError
        On a ``width`` mapping whose keys differ from ``pos``.
    """
    if isinstance(width, (int, float)):
        widths = {axis: float(width) for axis in pos}
    else:
        if set(pos) != set(width):
            raise ValueError(
                f"pos and width must name the same coordinates; got "
                f"pos keys {tuple(sorted(pos))} and width keys "
                f"{tuple(sorted(width))}")
        widths = dict(width)

    def shape(**coords: jax.Array) -> jax.Array:
        value = jnp.asarray(1.0)
        for axis, centre in pos.items():
            value = value * jnp.exp(
                -((coords[axis] - centre) ** 2) / widths[axis] ** 2)
        return value

    shape.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(
            coordinate, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for coordinate in pos])
    return shape
