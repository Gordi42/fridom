"""
Average spaces: cell-mean functionals (the FV representation).

Description
-----------
Owning class doc: ``notes/framework2/classes/spaces.md``
(``AverageSpace`` and friends). A ``CellAvg`` DOF is the functional
``(1/dx) ∫_cell u dx``, not a value at any point — "averages have no
position". The average family mirrors the nodal family: ``CellAvg``
on primal cells, ``FaceAvg`` on dual cells.
"""
# Wave 1: AverageSpace (ABC), CellAvg, FaceAvg
from __future__ import annotations

from fridom.framework2.grid.spaces.function_space import FunctionSpace


class AverageSpace(FunctionSpace):

    """
    Averages over a cell family (primal or dual).

    Description
    -----------
    Distinct from nodal spaces even where the DOF counts agree:
    high-order reconstruction differs between point values and cell
    means, and the coefficient spaces of average origins differ from
    those of nodal origins (the ``sinc(k dx / 2)`` factor).
    """


class CellAvg(AverageSpace):

    """Averages over the n primal cells."""

    @property
    def shape(self) -> tuple[int, ...]:
        """(n,) on periodic and bounded meshes alike."""
        return (self._mesh.n_cells,)


class FaceAvg(AverageSpace):

    """Averages over the dual cells around faces."""

    @property
    def shape(self) -> tuple[int, ...]:
        """(n,) on periodic meshes, (n - 1,) on bounded ones."""
        n = self._mesh.n_cells
        if self._mesh.periodic:
            return (n,)
        return (n - 1,)
