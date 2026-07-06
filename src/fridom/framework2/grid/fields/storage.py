"""
Storage-side helpers of the field core.

Description
-----------
Shared plumbing between ``ScalarField``, the ``Grid`` field factory,
and ``RandomFieldFactory``: the space-derived storage dtype (rules
section 3.1), the Hermitian value invariant of real-origin Fourier
factors (section 3.2), and the true-shape -> storage routing through
``decomposition.pad`` + halo sync (sections 3.5, 5).
"""
# Wave 2: field-core storage helpers
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_comp, dtype_real
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.coefficient import FourierSpace

if TYPE_CHECKING:  # pragma: no cover
    import jax
    import numpy as np

    from fridom.framework2.grid.decomposition.decomposition import (
        Decomposition,
    )
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


def storage_dtype(space: SpaceLike) -> type[np.floating]:
    """
    Return the space-derived storage dtype (rules section 3.1).

    Description
    -----------
    dtype is derived, never stored: any complex factor or any Fourier
    factor (a real-origin Fourier space stores the complex Hermitian
    half spectrum) makes the storage complex; sine/cosine/Chebyshev
    coefficients of real origins stay real.

    Parameters
    ----------
    space : SpaceLike
        The (product) function space.

    Returns
    -------
    type[np.floating]
        The derived storage scalar type.
    """
    for factor in space.factors:
        if (factor.scalars is Scalars.COMPLEX
                or isinstance(factor, FourierSpace)):
            return dtype_comp()
    return dtype_real()


def factor_axes(space: SpaceLike) -> tuple[tuple[object, int], ...]:
    """
    Pair each factor with its starting array axis.

    Parameters
    ----------
    space : SpaceLike
        The (product) function space.

    Returns
    -------
    tuple[tuple[FunctionSpace, int], ...]
        ``(factor, axis)`` pairs; multi-name factors span
        ``len(factor.shape)`` axes starting at ``axis``.
    """
    pairs = []
    axis = 0
    for factor in space.factors:
        pairs.append((factor, axis))
        axis += len(factor.shape)
    return tuple(pairs)


def self_conjugate_axis_indices(
    factor: FourierSpace,
) -> tuple[int, ...]:
    """
    Return the self-conjugate mode indices of a half spectrum.

    Description
    -----------
    On a real-origin Fourier factor the stored half spectrum keeps
    ``k = 0`` and, at even origin lengths, the Nyquist mode; realness
    there is a value constraint the shape cannot encode
    (section 3.2).

    Parameters
    ----------
    factor : FourierSpace
        A real-scalars Fourier factor.

    Returns
    -------
    tuple[int, ...]
        The self-conjugate indices along the factor's axis.
    """
    n = factor.origin.shape[0]
    indices = [0]
    if n % 2 == 0:
        indices.append(factor.shape[0] - 1)
    return tuple(indices)


def hermitian_project(
    data: jax.Array, space: SpaceLike,
) -> jax.Array:
    """
    Project the imaginary part at self-conjugate modes to zero.

    Description
    -----------
    Applied per real-origin Fourier factor on construction from raw
    data (``create_field(..., data=...)``); real-linear operators
    preserve the invariant automatically. A no-op on spaces without
    real-origin Fourier factors.

    Parameters
    ----------
    data : jax.Array
        A true-shape array on ``space``.
    space : SpaceLike
        The (product) function space of ``data``.

    Returns
    -------
    jax.Array
        The projected array (``data`` itself when nothing applies).
    """
    for factor, axis in factor_axes(space):
        if not isinstance(factor, FourierSpace):
            continue
        if factor.scalars is not Scalars.REAL:
            continue
        mask_1d = jnp.zeros(factor.shape[0], dtype=bool)
        mask_1d = mask_1d.at[
            jnp.asarray(self_conjugate_axis_indices(factor))].set(
            True)
        shape = [1] * data.ndim
        shape[axis] = factor.shape[0]
        mask = mask_1d.reshape(shape)
        data = jnp.where(mask, data.real.astype(data.dtype), data)
    return data


def store(
    decomposition: Decomposition,
    space: SpaceLike,
    true_data: jax.Array,
) -> jax.Array:
    """
    Route a true-shape array into synced, storage-shaped form.

    Description
    -----------
    The single write path of the storage contract (fields doc):
    ``decomposition.pad`` applies halo and stagger padding, the halo
    sync fills the ghost slots, so stored halos are always valid.

    Parameters
    ----------
    decomposition : Decomposition
        The grid's decomposition.
    space : SpaceLike
        The (laid-out) function space of the field.
    true_data : jax.Array
        The true-shape array (global on one device).

    Returns
    -------
    jax.Array
        The synced storage-shaped array.
    """
    padded = decomposition.pad(true_data, space)
    return decomposition.sync(padded, space)
