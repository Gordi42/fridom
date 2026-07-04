"""Spectral grid based on the cartesian grid."""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom.framework as fr

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):

    """Spectral grid with spectral differentiation operators."""

    def __init__(self,
                 shape: list[int],
                 domain_size: list[float],
                 periodic_bounds: list[bool] | None = None,
                 ) -> None:
        super().__init__(shape=shape,
                         domain_size=domain_size,
                         periodic_bounds=periodic_bounds,
                         diff_mod=fr.grid.cartesian.SpectralDiff(),
                         interp_mod=fr.grid.DummyInterpolation())
        self.name = "Spectral Grid"
        self.mpi_available = False
        self.spectral_grid = True

    def setup(self, mset: fr.ModelSettingsBase) -> None:
        """Set up the grid (see :py:meth:`fr.grid.GridBase.setup`)."""
        super().setup(mset, req_halo=0)

    def get_mesh(self,
                 position: fr.grid.Position | None = None,  # noqa: ARG002 (interface conformity)
                 spectral: bool = False ) -> tuple[np.ndarray]:
        """Return the meshgrid at the cell center."""
        return super().get_mesh(position=self.cell_center, spectral=spectral)

    def fft(self,
            arr: np.ndarray,
            padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
            bc_types: tuple[fr.grid.BCType] | None = None,
            positions: tuple[fr.grid.AxisPosition] | None = None,  # noqa: ARG002 (interface conformity)
            ) -> np.ndarray:
        """Transform an array from physical to spectral space."""
        return super().fft(arr=arr,
                           padding=padding,
                           bc_types=bc_types,
                           positions=self.cell_center)

    def ifft(self,
             arr: np.ndarray,
             padding: fr.grid.FFTPadding = fr.grid.FFTPadding.NOPADDING,
             bc_types: tuple[fr.grid.BCType] | None = None,
             positions: tuple[fr.grid.AxisPosition] | None = None,  # noqa: ARG002 (interface conformity)
             ) -> np.ndarray:
        """Transform an array from spectral to physical space."""
        return super().ifft(arr=arr,
                            padding=padding,
                            bc_types=bc_types,
                            positions=self.cell_center)
