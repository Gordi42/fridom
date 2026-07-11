"""
``IntervalMesh``: the uniform 1D interval.

Description
-----------
Owning class doc: ``design/specs/grid/classes/meshes.md``. The
iteration-1 workhorse: n equal cells on a physical interval,
periodic or bounded.
"""
# Wave 1: IntervalMesh
from __future__ import annotations

from typing import Self

from fridom.framework2.grid.meshes.structured_1d import StructuredMesh1D


class IntervalMesh(StructuredMesh1D):

    """
    Uniform 1D interval with n equal cells.

    Description
    -----------
    Inherits the full ``StructuredMesh1D`` factory surface and its
    default decomposition traits (GHOST-first for nodal/average
    spaces, ``(LOCAL, TRANSPOSE)`` for coefficient spaces,
    ``(LOCAL,)`` for ``ConstantSpace``). ``refined(factor)`` is
    iteration 1 here: the padded-transform target for dealiased
    products.

    Parameters
    ----------
    shape : int
        The cell count n.
    extent : tuple[float, float]
        The physical interval (x_min, x_max).
    periodic : bool, optional
        Whether the interval is periodic (default: True).
    name : str
        The mandatory coordinate name.
    """

    def __init__(self, shape: int, extent: tuple[float, float],
                 periodic: bool = True, *, name: str) -> None:
        """Store the uniform interval (shape is the cell count)."""
        super().__init__(shape, extent, periodic, name=name)

    @property
    def dx(self) -> float:
        """
        Uniform cell width (extent length / n_cells).

        Description
        -----------
        A descriptor convenience: operators must still obtain
        measures through the grid's metric fields (no operator bakes
        ``dx`` into Python constants); on this mesh those fields are
        constant and XLA folds them.
        """
        return (self._extent[1] - self._extent[0]) / self._n_cells

    def _make_refined(self, n_cells: int) -> Self:
        """Construct the scaled mesh (same extent, topology, name)."""
        return type(self)(n_cells, self._extent, self._periodic,
                          name=self._names[0])
