import fridom.framework as fr
import numpy as np
from functools import partial

@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):
    def __init__(self, 
                 N: list[int],
                 L: list[float],
                 periodic_bounds: list[bool] | None = None,
                 shared_axes=[0],) -> None:
        super().__init__(N=N, 
                         L=L, 
                         periodic_bounds=periodic_bounds,
                         shared_axes=shared_axes, 
                         diff_mod=fr.grid.cartesian.SpectralDiff(),
                         interp_mod=fr.grid.DummyInterpolation())
        self.name = "Spectral Grid"
        self.mpi_available = False
        self.spectral_grid = True

    def setup(self, mset: 'fr.ModelSettingsBase'):
        super().setup(mset, req_halo=0)

    def get_mesh(self, 
                 position: fr.grid.Position | None = None,
                 spectral: bool = False ) -> tuple[np.ndarray]:
        return super().get_mesh(position=self.cell_center, spectral=spectral)

    def fft(self, 
            arr: np.ndarray,
            padding = fr.grid.FFTPadding.NOPADDING,
            bc_types: tuple[fr.grid.BCType] | None = None,
            positions: tuple[fr.grid.AxisPosition] | None = None,
            ) -> np.ndarray:
        return super().fft(arr=arr,
                           padding=padding, 
                           bc_types=bc_types, 
                           positions=self.cell_center)

    def ifft(self, 
             arr: np.ndarray,
             padding = fr.grid.FFTPadding.NOPADDING,
             bc_types: tuple[fr.grid.BCType] | None = None,
             positions: tuple[fr.grid.AxisPosition] | None = None,
             ) -> np.ndarray:
        return super().ifft(arr=arr,
                            padding=padding, 
                            bc_types=bc_types, 
                            positions=self.cell_center)
