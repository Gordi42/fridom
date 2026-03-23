"""Stencil class representing a stencil of a given size and offset."""
from __future__ import annotations

from typing import Iterator

import fridom.framework as fr

ncp = fr.config.ncp

class Stencil:

    """
    Class representing a stencil of a given size and offset.

    Description
    -----------
    A stencil defines the local neighborhood of points used for mapping
    a function from one function space to another. The size of the stencil
    determines how many neighboring points are used, while the offset determines
    the bias.

    We consider two cases, CENTER -> FACE mapping and FACE -> CENTER mapping.

    For the CENTER -> FACE mapping:

    ::

                x'_{n-1}      x'_{n}     x'_{n+1}           (FACE positions)
                    v           v           v
        |  x_{n-1}  |   x_{n}   |  x_{n+1}  |  x_{n+2}  |   (CENTER positions)

                    |-----------|                         size = 1, offset = 0
                    |-----------------------|             size = 2, offset = 0
                    |-----------------------------------| size = 3, offset = 0
        |-----------------------------------|             size = 3, offset = 1

    For the FACE -> CENTER mapping:

    ::

                x'_{n-1}      x'_{n}     x'_{n+1}           (FACE positions)
                    v           v           v
        |  x_{n-1}  |   x_{n}   |  x_{n+1}  |  x_{n+2}  |   (CENTER positions)

                          |-----------|                   size = 1, offset = 0
              |-----------------------|                   size = 2, offset = 0
              |-----------------------------------|       size = 3, offset = 0
        ------------------------------|                   size = 3, offset = 1

    """

    def __init__(
            self,
            grid: fr.grid.GridBase,
            size: int, offset: int,
            destination: fr.grid.AxisPosition) -> None:
        self.grid = grid
        self.size = size
        self.offset = offset
        self.destination = destination

        start = 0 if destination == fr.grid.AxisPosition.FACE else 1
        self.shifts = [start + offset - i for i in range(size)]

    def view(self, arr: ncp.ndarray, axis: int) -> fr.grid.StencilView:
        """Return a view of the stencil on the given array and axis."""
        return fr.grid.StencilView(self, arr, axis)

    def __iter__(self) -> Iterator[int]:
        """Iterate over the fields of the vector field."""
        return iter(self.shifts)
