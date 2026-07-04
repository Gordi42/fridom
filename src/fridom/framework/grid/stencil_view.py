"""StencilView class representing a view of a stencil on a given array and axis."""
from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import fridom.framework as fr

ncp = fr.config.ncp

class StencilView:

    """A view of a stencil on a given array and axis."""

    def __init__(self, stencil: fr.grid.Stencil, arr: ncp.ndarray, axis: int) -> None:
        self.stencil = stencil
        self.arr = arr
        self.axis = axis

    @cache  # noqa: B019
    def __getitem__(self, index: int) -> ncp.ndarray:

        @self.stencil.grid.domain_decomp.shard_map
        def shift(arr: ncp.ndarray, shift: int) -> ncp.ndarray:
            return ncp.roll(arr, shift=shift, axis=self.axis)

        return shift(self.arr, shift=self.stencil.shifts[index])

    def __iter__(self) -> Iterator[ncp.ndarray]:
        for i in range(self.stencil.size):
            yield self[i]
