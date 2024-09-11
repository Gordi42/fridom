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
        self.mpi_available = False

    def setup(self, mset: 'fr.ModelSettingsBase'):
        fft = fr.grid.spectral.FFT(self._periodic_bounds)
        super().setup(mset, req_halo=0, fft_module=fft)

        self.pad_extend_inner: tuple[slice] = None
        self.pad_trim_inner: tuple[slice] = None
        self.pad_extend: tuple[tuple[int]] = None

    def get_mesh(self, 
                 position: fr.grid.Position | None = None,
                 spectral: bool = False ) -> tuple[np.ndarray]:
        return super().get_mesh(position=self.cell_center, spectral=spectral)

    @partial(fr.utils.jaxjit, static_argnames=["transform_types"])
    def fft(self, 
            arr: np.ndarray,
            transform_types: 'tuple[fr.grid.TransformType] | None' = None,
            padding = fr.grid.FFTPadding.NOPADDING,
            ) -> np.ndarray:
        # Forward transform the array
        f = lambda x, axes: self._fft.forward(x, axes, transform_types)
        u_hat = self._pfft.forward_apply(arr, f)
        
        # Apply padding if necessary
        if padding == fr.grid.FFTPadding.EXTEND:
            u_hat = u_hat[self.pad_extend_inner]
        return u_hat

    @partial(fr.utils.jaxjit, static_argnames=["transform_types"])
    def ifft(self, 
             arr: np.ndarray,
             transform_types: 'tuple[fr.grid.TransformType] | None' = None
             ) -> np.ndarray:
        ncp = fr.config.ncp
        # Apply padding if necessary
        match self._padding:
            case fr.grid.FFTPadding.NOPADDING:
                u = arr
            case fr.grid.FFTPadding.TRIM:
                u = ncp.zeros_like(arr)
                u = fr.utils.modify_array(u, self.pad_trim_inner, arr)
            case fr.grid.FFTPadding.EXTEND:
                u = ncp.pad(arr, self.pad_extend, mode='constant')

        f = lambda x, axes: self._fft.backward(x, axes, transform_types)
        return self._pfft.backward_apply(u, f)