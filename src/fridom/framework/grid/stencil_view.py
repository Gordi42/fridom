"""StencilView class: a view of a stencil on a given array and axis."""
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Iterator

    import fridom.framework as fr


class StencilView:

    """A view of a stencil on a given array and axis."""

    def __init__(self,
                 stencil: fr.grid.Stencil,
                 arr: jnp.ndarray,
                 axis: int) -> None:
        self.stencil = stencil
        self.arr = arr
        self.axis = axis

    @cache  # noqa: B019
    def __getitem__(self, index: int) -> jnp.ndarray:

        @self.stencil.grid.domain_decomp.shard_map
        def shift(arr: jnp.ndarray, shift: int) -> jnp.ndarray:
            return jnp.roll(arr, shift=shift, axis=self.axis)

        return shift(self.arr, shift=self.stencil.shifts[index])

    def __iter__(self) -> Iterator[jnp.ndarray]:
        for i in range(self.stencil.size):
            yield self[i]
