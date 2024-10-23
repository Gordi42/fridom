from functools import partial
from numpy import ndarray
import fridom.framework as fr


class SingleDecomposition(fr.domain_decomposition.DomainDecomposition):
    def __init__(self, shape: tuple[int], 
                 halo: int = 0, 
                 periods: tuple[int] = (0, 0),
                 shared_axes: tuple[int] | None = None, 
                 device_ids: list[int] | None = None):
        super().__init__(shape, halo, periods, shared_axes, device_ids)

        def _make_slice_tuple(slc):
            slice_list = []
            for i in range(self.n_dims):
                full_slice = [slice(None)]*self.n_dims
                full_slice[i] = slc
                slice_list.append(tuple(full_slice))
            return tuple(slice_list)

        # create slices for halo exchange
        self._inner = _make_slice_tuple(slice(halo, -halo))
        self._send_to_next = _make_slice_tuple(slice(-2*halo, -halo))
        self._send_to_prev = _make_slice_tuple(slice(halo, 2*halo))
        self._recv_from_next = _make_slice_tuple(slice(-halo, None))
        self._recv_from_prev = _make_slice_tuple(slice(None, halo))

        # create paddings for halo exchange
        self._pw_periodic = [(halo, halo) if self.periods[i] else (0, 0) 
                             for i in range(self.n_dims)]
        self._pw_nonperiodic = [(0, 0) if self.periods[i] else (halo, halo) 
                                for i in range(self.n_dims)]

    # ================================================================
    #  Halo exchange
    # ================================================================

    @partial(fr.utils.jaxjit, static_argnames='flat_axes')
    def sync(self, arr: ndarray, flat_axes: list[int] | None = None) -> ndarray:
        # nothing to do if there are no halo regions
        if self.halo == 0:
            return arr

        flat_axes = flat_axes or []
        
        # synchronize cpu and gpu
        if fr.config.backend == fr.config.Backend.CUPY:
            fr.config.ncp.cuda.Stream.null.synchronize()

        # synchronize one dimension at a time
        for axis in range(self.n_dims):
            if axis in flat_axes:
                continue
            if self.periods[axis]:
                arr = self._sync_periodic_axis(arr, axis)
            else:
                arr = self._sync_non_periodic_axis(arr, axis)
        return arr

    @partial(fr.utils.jaxjit, static_argnames=['axis'])
    def _sync_periodic_axis(self, arr: ndarray, axis: int,) -> ndarray:
        if self.shape[axis] < self.halo:
            pad = fr.config.ncp.pad
            ics = self._inner[axis]
            pad_width = self._paddings[axis]
            return pad(arr[ics], pad_width, mode='wrap')
        else:
            rfn = self._recv_from_next[axis]
            rfp = self._recv_from_prev[axis]
            stn = self._send_to_next[axis]
            stp = self._send_to_prev[axis]
            if fr.config.backend_is_jax:
                arr = arr.at[rfn].set(arr[stp])
                arr = arr.at[rfp].set(arr[stn])
            else:
                arr[rfn] = arr[stp]
                arr[rfp] = arr[stn]
            return arr

    @partial(fr.utils.jaxjit, static_argnames=['axis'])
    def _sync_non_periodic_axis(self, arr: ndarray, axis: int,) -> ndarray:
        rfn = self._recv_from_next[axis]
        rfp = self._recv_from_prev[axis]
        if fr.config.backend_is_jax:
            arr = arr.at[rfn].set(0)
            arr = arr.at[rfp].set(0)
        else:
            arr[rfn] = arr[0]
            arr[rfp] = arr[0]
        return arr

    # ================================================================
    #  Transpose
    # ================================================================

    @partial(fr.utils.jaxjit, static_argnames=('axes_in', 'axes_out'))
    def transpose(self, arr: ndarray, axes_in: tuple[int], axes_out: tuple[int]) -> ndarray:
        # nothing to do here since we are not parallel
        return arr

    # ================================================================
    #  Padding
    # ================================================================

    @fr.utils.jaxjit
    def pad(self, arr: ndarray) -> ndarray:
        if arr.shape != self.shape:
            raise ValueError(f"Array shape {arr.shape} does not match domain shape {self.shape}")
        ncp = fr.config.ncp
        arr = ncp.pad(arr, self._pw_periodic, mode='wrap')
        arr = ncp.pad(arr, self._pw_nonperiodic, mode='constant')
        return arr

    @fr.utils.jaxjit
    def unpad(self, arr: ndarray) -> ndarray:
        padded_shape = tuple(s + 2*self.halo for s in self.shape)
        if arr.shape != padded_shape:
            raise ValueError(f"Array shape {arr.shape} does not match padded shape {padded_shape}")
        return arr[self._inner]

    # ================================================================
    #  Gather
    # ================================================================

    @partial(fr.utils.jaxjit, static_argnames=('dest_rank', 'slc'))
    def gather(self, 
               arr: ndarray, 
               slc: tuple[slice] | None = None,
               dest_rank: int | None = None) -> ndarray:
        if arr.shape == self.shape:
            return arr[slc]
        else:
            return arr[self._inner][slc]


    # ================================================================
    #  Array creation
    # ================================================================

    def create_array(self, pad: bool = True) -> ndarray:
        arr = fr.config.ncp.zeros(self.shape, dtype=fr.config.dtype_real)
        if pad:
            arr = self.pad(arr)
        return arr

    def create_meshgrid(self, *args: ndarray, pad: bool = True) -> tuple[ndarray]:
        X = fr.config.ncp.meshgrid(*args, indexing='ij')
        if pad:
            X = tuple(self.pad(x) for x in X)
        return X

    # ================================================================
    #  Array operations
    # ================================================================

    def sum(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        return fr.config.ncp.sum(arr, axis=axes)

    def max(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        return fr.config.ncp.max(arr, axis=axes)

    def min(self, arr: ndarray, axes: list[int] | None = None) -> ndarray:
        return fr.config.ncp.min(arr, axis=axes)
