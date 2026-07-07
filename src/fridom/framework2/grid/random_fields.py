"""
``RandomFieldFactory``: the ``grid.random`` accessor.

Description
-----------
Owning class doc: ``notes/framework2/classes/grid.md``. Seeded,
sharding-consistent random field generators (rules section 3.10):
values are a pure function of ``(space.shape, seed)`` over the
**global true-DOF index** — per-DOF ``fold_in(seed, global_index)``
keying, drawing directly into the local shard — so draws are
deterministic across device counts by construction. The draw covers
the true shape only, so random values never land in padding.
"""
# Wave 2: RandomFieldFactory
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.grid.fields.storage import (
    factor_axes,
    flat_hermitian_applies,
    self_conjugate_axis_indices,
    storage_dtype,
    store,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.coefficient import FourierSpace

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


class RandomFieldFactory:

    """
    Seeded random fields, deterministic across device layouts.

    Description
    -----------
    Created by the grid itself and reached as ``grid.random``; fully
    static (holds only the grid reference). Draws are **white in
    coefficients**: unit variance per DOF (complex normal on complex
    storage), with real unit-variance draws at the self-conjugate
    modes of real-origin Fourier factors (section 3.2).

    Parameters
    ----------
    grid : Grid
        The grid the factory is bound to.
    """

    def __init__(self, grid: Grid) -> None:
        """Bind the factory to its grid (created by the grid)."""
        self._grid: Grid = grid

    def normal(self, space: SpaceLike, seed: int) -> ScalarField:
        """
        Standard-normal field on ``space``.

        Description
        -----------
        Complex normal (independent real/imaginary parts, unit total
        variance) on complex storage; at the self-conjugate modes of
        real-origin Fourier factors the draw is real with unit
        variance.

        Parameters
        ----------
        space : SpaceLike
            The (product) function space; mandatory.
        seed : int
            The PRNG seed.

        Returns
        -------
        ScalarField
            The random field (default metadata).
        """
        grid = self._grid
        space = grid._laid_out(space)  # noqa: SLF001 — grid-made
        decomposition = grid.decomposition
        slices = decomposition.local_slice(space)
        indices = _global_indices(space.shape, slices)
        key = jax.random.key(seed)
        dtype = storage_dtype(space)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            draws = _keyed_normal(key, indices, 2)
            values = (draws[..., 0] + 1j * draws[..., 1]) / jnp.sqrt(
                jnp.asarray(2.0, dtype=dtype_real()))
            mask = _self_conjugate_mask(space, slices)
            if mask is not None:
                values = jnp.where(mask, draws[..., 0], values)
        else:
            values = _keyed_normal(key, indices, 1)[..., 0]
        values = values.astype(dtype)
        stored = store(decomposition, space, values)
        return ScalarField(grid, space, stored)

    def phase(self, space: SpaceLike, seed: int) -> ScalarField:
        """Unit-modulus field e^{i theta}, theta uniform per DOF."""
        raise NotImplementedError(
            "random.phase (the spectra-IC building block) is not "
            "part of the Wave-2 field core; it lands with the "
            "transform cluster")


# ================================================================
#  Global-index keyed draws
# ================================================================
def _global_indices(
    shape: tuple[int, ...],
    slices: tuple[slice, ...],
) -> jax.Array:
    """
    Compute the global flat true-DOF index per local DOF.

    Parameters
    ----------
    shape : tuple[int, ...]
        The global true shape of the space.
    slices : tuple[slice, ...]
        The local shard's global index ranges
        (``decomposition.local_slice``).

    Returns
    -------
    jax.Array
        Integer array of the local true shape (C-order flat global
        indices).
    """
    strides = []
    acc = 1
    for n in reversed(shape):
        strides.append(acc)
        acc *= n
    strides.reverse()
    ndim = len(shape)
    total = jnp.zeros((1,) * ndim, dtype=jnp.int64)
    for axis, (sl, stride) in enumerate(
            zip(slices, strides, strict=True)):
        idx = jnp.arange(sl.start, sl.stop, dtype=jnp.int64) * stride
        reshape = [1] * ndim
        reshape[axis] = idx.size
        total = total + idx.reshape(reshape)
    return total


def _keyed_normal(
    key: jax.Array,
    indices: jax.Array,
    draws_per_dof: int,
) -> jax.Array:
    """
    Per-DOF fold_in keyed standard-normal draws.

    Parameters
    ----------
    key : jax.Array
        The seed key.
    indices : jax.Array
        Global flat DOF indices (any shape).
    draws_per_dof : int
        Number of independent draws per DOF.

    Returns
    -------
    jax.Array
        Real draws of shape ``(*indices.shape, draws_per_dof)``.
    """
    flat = indices.reshape(-1)
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(flat)
    samples = jax.vmap(
        lambda k: jax.random.normal(k, (draws_per_dof,),
                                    dtype=dtype_real()))(keys)
    return samples.reshape(*indices.shape, draws_per_dof)


def _self_conjugate_mask(
    space: SpaceLike,
    slices: tuple[slice, ...],
) -> jax.Array | None:
    """
    Local mask of the self-conjugate DOFs, or None if none exist.

    Description
    -----------
    A DOF is self-conjugate when its index is a self-conjugate mode
    (k = 0 / Nyquist) along **every** real-origin Fourier axis; the
    draw must be real there (Hermitian value invariant, 3.2). The
    flat real-draw rule shares ``flat_hermitian_applies`` with the
    field factory: on multi-axis coefficient spaces the invariant is
    the conjugate *pairing*, which a plane of forced-real draws
    would corrupt — no mask applies there (the pairing draw is
    designed-for with the spectra ICs).
    """
    if not flat_hermitian_applies(space):
        return None
    mask = None
    ndim = len(space.shape)
    for factor, axis in factor_axes(space):
        if not (isinstance(factor, FourierSpace)
                and factor.scalars is Scalars.REAL):
            continue
        sl = slices[axis]
        idx = jnp.arange(sl.start, sl.stop)
        axis_mask = jnp.isin(
            idx, jnp.asarray(self_conjugate_axis_indices(factor)))
        reshape = [1] * ndim
        reshape[axis] = idx.size
        axis_mask = axis_mask.reshape(reshape)
        mask = axis_mask if mask is None else mask & axis_mask
    return mask
