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
        # the shift must be closed over (and not passed as an argument)
        # because the sharded map only accepts array arguments
        shift_value = self.stencil.shifts[index]

        @self.stencil.grid.domain_decomp.shard_map
        def shift(arr: jnp.ndarray) -> jnp.ndarray:
            return jnp.roll(arr, shift=shift_value, axis=self.axis)

        return shift(self.arr)

    def __iter__(self) -> Iterator[jnp.ndarray]:
        for i in range(self.stencil.size):
            yield self[i]
