import fridom.framework as fr
import numpy as np
from functools import partial


@fr.utils.jaxify
class Grid(fr.grid.cartesian.Grid):
    def __init__(self, 
                 N: list[int],
                 L: list[float],
                 periodic_bounds: list[bool] | None = None) -> None:
        super().__init__(N=N, 
                         L=L, 
                         periodic_bounds=periodic_bounds,
                         shared_axes=[0], 
                         diff_mod=fr.grid.spectral.SpectralDiff(),
                         interp_mod=fr.grid.InterpolationModule())
        self.name = "Spectral Grid"
        self.mpi_available = False

    def setup(self, mset: 'fr.ModelSettingsBase'):
        fft = fr.grid.spectral.FFT(self._periodic_bounds)
        super().setup(mset, req_halo=0, fft_module=fft)

        # ----------------------------------------------------------------
        #  Prepare padding for FFT
        # ----------------------------------------------------------------
        # first the outer padding of trim option
        trim_zero_slice = []
        for i in range(self.n_dims):
            if self._periodic_bounds[i]:
                new_kmax = int(2/3 * int(self.N[i]/2))
                trim_zero_slice.append(slice(new_kmax+1,-new_kmax))
            else:
                new_kmax = int(2/3 * (self.N[i]-1))
                trim_zero_slice.append(slice(new_kmax+1, None))

        # extend option
        extend_first_halfs = []
        extend_second_halfs = []
        for i in range(self.n_dims):
            first_half = [slice(None)] * self.n_dims
            first_half[i] = slice(0, int((self.N[i]+1)/2))
            second_half = [slice(None)] * self.n_dims
            second_half[i] = slice(-int(self.N[i]/2), None)
            extend_first_halfs.append(tuple(first_half))
            extend_second_halfs.append(tuple(second_half))

        self.pad_trim_zero_slice: tuple[slice] = tuple(trim_zero_slice)
        self.extend_first_halfs: tuple[tuple[slice]] = tuple(extend_first_halfs)
        self.extend_second_halfs: tuple[tuple[slice]] = tuple(extend_second_halfs)


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
                u = fr.utils.modify_array(arr, self.pad_trim_zero_slice, 0)
            case fr.grid.FFTPadding.EXTEND:
                u = ncp.pad(arr, self.pad_extend, mode='constant')

        f = lambda x, axes: self._fft.backward(x, axes, transform_types)
        return self._pfft.backward_apply(u, f)

    def _pad_extend_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        ncp = fr.config.ncp
        N = self.N[axis]
        if self._periodic_bounds[axis]:
            sl_first_half = [slice(None)] * self.n_dims
            sl_first_half[axis] = slice(0, int((N+1)/2))
            sl_second_half = [slice(None)] * self.n_dims
            sl_second_half[axis] = slice(-int(N/2), None)
            first_part = arr[tuple(sl_first_half)]
            paddings = [(0,0)] * self.n_dims
            paddings[axis] = (0, int((N+1)/2))
            first_part = ncp.pad(first_part, paddings, mode='constant')
            arr = ncp.concatenate((first_part, arr[tuple(sl_second_half)]), axis=axis)
        else:
            paddings = [(0,0)] * self.n_dims
            paddings[axis] = (0, int((N+1)/2))
            arr = ncp.pad(arr, paddings, mode='constant')
        return arr

    def _unpad_extend_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        ncp = fr.config.ncp
        N = self.N[axis]
        if self._periodic_bounds[axis]:
            sl_first_half = [slice(None)] * self.n_dims
            sl_first_half[axis] = slice(0, int((N+1)/2))
            sl_second_half = [slice(None)] * self.n_dims
            sl_second_half[axis] = slice(-int(N/2), None)
            arr = ncp.concatenate((arr[tuple(sl_first_half)], arr[tuple(sl_second_half)]), axis=axis)
        else:
            sl = [slice(None)] * self.n_dims
            sl[axis] = slice(0, N)
            arr = arr[tuple(sl)]
        return arr

    def pad_extend(self, arr: np.ndarray) -> np.ndarray:
        for axis in range(self.n_dims):
            arr = self._pad_extend_axis(arr, axis)
        return arr

    def unpad_extend(self, arr: np.ndarray) -> np.ndarray:
        for axis in range(self.n_dims):
            arr = self._unpad_extend_axis(arr, axis)
        return arr
