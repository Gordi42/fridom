
import numpy as np

import fridom.framework as fr


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):
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

    def setup(self, mset: "fr.ModelSettingsBase") -> None:
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
