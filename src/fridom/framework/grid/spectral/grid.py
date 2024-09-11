import fridom.framework as fr
import numpy as np
from functools import partial


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):
    def __init__(self, 
                 N: list[int],
                 L: list[float],
                 periodic_bounds: list[bool] | None = None) -> None:
        super().__init__(N=N, L=L, periodic_bounds=periodic_bounds,
                         shared_axes=[0], 
                         diff_mod=fr.grid.spectral.SpectralDiff(),
                         interp_mod=fr.grid.InterpolationModule())
        self.name = "Spectral Grid"

    def setup(self, mset: 'fr.ModelSettingsBase'):
        fft = fr.grid.spectral.FFT(self._periodic_bounds)
        super().setup(mset, req_halo=0, fft_module=fft)

    def get_mesh(self, 
                 position: fr.grid.Position | None = None,
                 spectral: bool = False ) -> tuple[np.ndarray]:
        return super().get_mesh(position=self.cell_center, spectral=spectral)

    @partial(fr.utils.jaxjit, static_argnames=["transform_types"])
    def fft(self, 
            arr: np.ndarray,
            transform_types: 'tuple[fr.grid.TransformType] | None' = None
            ) -> np.ndarray:
        f = lambda x, axes: self._fft.forward(x, axes, transform_types)
        return self._pfft.forward_apply(arr, f)

    @partial(fr.utils.jaxjit, static_argnames=["transform_types"])
    def ifft(self, 
             arr: np.ndarray,
             transform_types: 'tuple[fr.grid.TransformType] | None' = None
             ) -> np.ndarray:
        f = lambda x, axes: self._fft.backward(x, axes, transform_types)
        return self._pfft.backward_apply(arr, f)