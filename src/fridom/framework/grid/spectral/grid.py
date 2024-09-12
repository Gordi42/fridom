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
            slices = [slice(None)] * self.n_dims
            if self._periodic_bounds[i]:
                new_kmax = int(2/3 * int(self.N[i]/2))
                slices[i] = slice(new_kmax+1, -new_kmax)
            else:
                new_kmax = int(2/3 * (self.N[i]-1))
                slices[i] = slice(new_kmax+1, None)
            trim_zero_slice.append(tuple(slices))

        # extend option
        extend_first_halfs = []
        extend_second_halfs = []
        extend_paddings = []
        extend_unpad_slices = []
        for i in range(self.n_dims):
            first_half = [slice(None)] * self.n_dims
            first_half[i] = slice(0, int((self.N[i]+1)/2))
            extend_first_halfs.append(tuple(first_half))

            second_half = [slice(None)] * self.n_dims
            second_half[i] = slice(-int(self.N[i]/2), None)
            extend_second_halfs.append(tuple(second_half))

            paddings = [(0,0)] * self.n_dims
            paddings[i] = (0, int((self.N[i]+1)/2))
            extend_paddings.append(tuple(paddings))

            sl = [slice(None)] * self.n_dims
            sl[i] = slice(0, self.N[i])
            extend_unpad_slices.append(tuple(sl))

        self.pad_trim_zero_slice: tuple[slice] = tuple(trim_zero_slice)
        self.extend_first_halfs: tuple[tuple[slice]] = tuple(extend_first_halfs)
        self.extend_second_halfs: tuple[tuple[slice]] = tuple(extend_second_halfs)
        self.extend_pad: tuple[tuple[int]] = tuple(extend_paddings)
        self.extend_unpad_slices: tuple[tuple[slice]] = tuple(extend_unpad_slices)


    def get_mesh(self, 
                 position: fr.grid.Position | None = None,
                 spectral: bool = False ) -> tuple[np.ndarray]:
        return super().get_mesh(position=self.cell_center, spectral=spectral)

    @partial(fr.utils.jaxjit, static_argnames=["bc_types", "padding"])
    def fft(self, 
            arr: np.ndarray,
            bc_types: 'tuple[fr.grid.BCTypes] | None' = None,
            padding = fr.grid.FFTPadding.NOPADDING,
            ) -> np.ndarray:
        # Forward transform the array
        f = lambda x, axes: self._fft.forward(x, axes, bc_types)
        u_hat = self._pfft.forward_apply(arr, f)
        
        # Apply padding if necessary
        if padding == fr.grid.FFTPadding.EXTEND:
            u_hat = self.unpad_extend(u_hat)
        return u_hat

    @partial(fr.utils.jaxjit, static_argnames=["bc_types", "padding"])
    def ifft(self, 
             arr: np.ndarray,
             bc_types: 'tuple[fr.grid.BCType] | None' = None,
             padding = fr.grid.FFTPadding.NOPADDING,
             ) -> np.ndarray:
        # Apply padding if necessary
        match padding:
            case fr.grid.FFTPadding.NOPADDING:
                u = arr
            case fr.grid.FFTPadding.TRIM:
                u = self.pad_trim(arr)
            case fr.grid.FFTPadding.EXTEND:
                u = self.pad_extend(arr)

        f = lambda x, axes: self._fft.backward(x, axes, bc_types)
        return self._pfft.backward_apply(u, f)

    def _pad_extend_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        ncp = fr.config.ncp
        if self._periodic_bounds[axis]:
            first_part = arr[self.extend_first_halfs[axis]]
            second_part = arr[self.extend_second_halfs[axis]]
            first_part = ncp.pad(first_part, self.extend_pad[axis], mode='constant')
            arr = ncp.concatenate((first_part, second_part), axis=axis)
        else:
            arr = ncp.pad(arr, self.extend_pad[axis], mode='constant')
        return arr

    def _unpad_extend_axis(self, arr: np.ndarray, axis: int) -> np.ndarray:
        ncp = fr.config.ncp
        if self._periodic_bounds[axis]:
            arr = ncp.concatenate(
                (arr[self.extend_first_halfs[axis]], 
                 arr[self.extend_second_halfs[axis]]), axis=axis)
        else:
            arr = arr[self.extend_unpad_slices[axis]]
        return arr

    def pad_extend(self, arr: np.ndarray) -> np.ndarray:
        for axis in range(self.n_dims):
            arr = self._pad_extend_axis(arr, axis)
        return arr

    def unpad_extend(self, arr: np.ndarray) -> np.ndarray:
        for axis in range(self.n_dims):
            arr = self._unpad_extend_axis(arr, axis)
        return arr

    def pad_trim(self, arr: np.ndarray) -> np.ndarray:
        for axis in range(self.n_dims):
            arr = fr.utils.modify_array(arr, self.pad_trim_zero_slice[axis], 0)
        return arr
