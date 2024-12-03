from functools import partial
from numpy import ndarray
import fridom.framework as fr


@fr.utils.jaxify
class SingleDecomposition(fr.domain_decomposition.DomainDecomposition):
    def __init__(self, shape: tuple[int],
                 halo: int = 0,
                 periods: tuple[bool] | None = None,
                 shared_axes: tuple[int] | None = None,
                 device_ids: list[int] | None = None):
        super().__init__(shape, halo, periods, shared_axes, device_ids)
        self._p_dims = tuple([1]*self.n_dims)

        # ----------------------------------------------------------------
        #  Halo exchange slices and paddings
        # ----------------------------------------------------------------

        def _make_slice_tuple(slc):
            slice_list = []
            for i in range(self.n_dims):
                full_slice = [slice(None)]*self.n_dims
                full_slice[i] = slc
                slice_list.append(tuple(full_slice))
            return tuple(slice_list)

        # create slices for halo exchange
        inner = slice(halo, -halo) if self.halo > 0 else slice(None)
        self._inner_slice = tuple([inner]*self.n_dims)
        self._inner = _make_slice_tuple(inner)
        self._send_to_next = _make_slice_tuple(slice(-2*halo, -halo))
        self._send_to_prev = _make_slice_tuple(slice(halo, 2*halo))
        self._recv_from_next = _make_slice_tuple(slice(-halo, None))
        self._recv_from_prev = _make_slice_tuple(slice(None, halo))

        # create paddings for halo exchange
        self._pw_periodic = [(halo, halo) if self.periods[i] else (0, 0) 
                             for i in range(self.n_dims)]
        self._pw_nonperiodic = [(0, 0) if self.periods[i] else (halo, halo) 
                                for i in range(self.n_dims)]
        paddings = tuple(tuple((halo, halo) if i == j else (0, 0) 
                               for i in range(self.n_dims))
                         for j in range(self.n_dims))
        self._paddings = paddings

        # ----------------------------------------------------------------
        #  Extend slices and paddings
        # ----------------------------------------------------------------

        # paddings for spectral extend
        # first the outer padding of trim option
        trim_zero_slice = []
        for i in range(self.n_dims):
            slices = [slice(None)] * self.n_dims
            if self.periods[i]:
                new_kmax = int(2/3 * int(self.shape[i]/2))
                slices[i] = slice(new_kmax+1, -new_kmax)
            else:
                new_kmax = int(2/3 * (self.shape[i]-1))
                slices[i] = slice(new_kmax+1, None)
            trim_zero_slice.append(tuple(slices))

        # extend option
        extend_first_halfs = []
        extend_second_halfs = []
        extend_paddings = []
        extend_unpad_slices = []
        for i in range(self.n_dims):
            first_half = [slice(None)] * self.n_dims
            first_half[i] = slice(0, int((self.shape[i]+1)/2))
            extend_first_halfs.append(tuple(first_half))

            second_half = [slice(None)] * self.n_dims
            second_half[i] = slice(-int(self.shape[i]/2), None)
            extend_second_halfs.append(tuple(second_half))

            paddings = [(0,0)] * self.n_dims
            paddings[i] = (0, int((self.shape[i]+1)/2))
            extend_paddings.append(tuple(paddings))

            sl = [slice(None)] * self.n_dims
            sl[i] = slice(0, self.shape[i])
            extend_unpad_slices.append(tuple(sl))

        self._pad_trim_zero_slice: tuple[slice] = tuple(trim_zero_slice)
        self._extend_first_halfs: tuple[tuple[slice]] = tuple(extend_first_halfs)
        self._extend_second_halfs: tuple[tuple[slice]] = tuple(extend_second_halfs)
        self._extend_pad: tuple[tuple[int]] = tuple(extend_paddings)
        self._extend_unpad_slices: tuple[tuple[slice]] = tuple(extend_unpad_slices)

    # ================================================================
    #  Halo exchange
    # ================================================================

    @partial(fr.utils.jaxjit, static_argnames='flat_axes')
    def sync(self, arr: ndarray, flat_axes: list[int] | None = None) -> ndarray:
        # nothing to do if there are no halo regions
        if self.halo == 0:
            return arr

        flat_axes = flat_axes or []
        
        # synchronize cpu and gpu on cupy backend
        if fr.config.backend == "cupy":
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
            arr[rfn] = 0
            arr[rfp] = 0
        return arr

    # ================================================================
    #  Padding
    # ================================================================

    @partial(fr.utils.jaxjit, static_argnames='flat_axes')
    def pad(self, arr: ndarray, flat_axes: tuple[int] | None = None) -> ndarray:
        if self.halo == 0:
            return arr
        ncp = fr.config.ncp
        # update the paddings for flat axes
        pw_periodic = list(self._pw_periodic)
        pw_nonperiodic = list(self._pw_nonperiodic)
        for axis in flat_axes or []:
            pw_periodic[axis] = (0, 0)
            pw_nonperiodic[axis] = (0, 0)
        # pad the array
        arr = ncp.pad(arr, tuple(pw_periodic), mode='wrap')
        arr = ncp.pad(arr, tuple(pw_nonperiodic), mode='constant')
        return arr

    @partial(fr.utils.jaxjit, static_argnames='flat_axes')
    def unpad(self, arr: ndarray, flat_axes: tuple[int] | None = None) -> ndarray:
        if self.halo == 0:
            return arr
        # remove the paddings for flat axes
        ics = list(self._inner_slice)
        for axis in flat_axes or []:
            ics[axis] = slice(None)
        return arr[tuple(ics)]

    # ----------------------------------------------------------------
    #  Spectral paddings
    # ----------------------------------------------------------------

    def _pad_extend_axis(self, 
                         arr: ndarray, 
                         axis: int,
                         ) -> ndarray:
        ncp = fr.config.ncp
        if self.periods[axis]:
            first_part = arr[self._extend_first_halfs[axis]]
            second_part = arr[self._extend_second_halfs[axis]]
            first_part = ncp.pad(first_part, self._extend_pad[axis], mode='constant')
            arr = ncp.concatenate((first_part, second_part), axis=axis)
        else:
            arr = ncp.pad(arr, self._extend_pad[axis], mode='constant')
        return arr

    def _unpad_extend_axis(self, 
                           arr: ndarray, 
                           axis: int,
                           ) -> ndarray:
        ncp = fr.config.ncp
        if self.periods[axis]:
            arr = ncp.concatenate(
                (arr[self._extend_first_halfs[axis]], 
                 arr[self._extend_second_halfs[axis]]), axis=axis)
        else:
            arr = arr[self._extend_unpad_slices[axis]]
        return arr

    def pad_extend(self, arr: ndarray) -> ndarray:
        for axis in range(self.n_dims):
            arr = self._pad_extend_axis(arr, axis)
        return arr

    def unpad_extend(self, arr: ndarray) -> ndarray:
        for axis in range(self.n_dims):
            arr = self._unpad_extend_axis(arr, axis)
        return arr

    def pad_trim(self, arr: ndarray) -> ndarray:
        for axis in range(self.n_dims):
            arr = fr.utils.modify_array(arr, self._pad_trim_zero_slice[axis], 0)
        return arr


    # ================================================================
    #  Gather
    # ================================================================

    def gather(self, 
               arr: ndarray, 
               slc: tuple[slice] | None = None,
               dest_rank: int | None = None,
               spectral: bool = False) -> ndarray:
        if arr.shape == self.shape:
            return arr[slc]
        else:
            return arr[self._inner_slice][slc]

    # ================================================================
    #  Array creation
    # ================================================================

    def _get_array_attrs(self, 
                         topo: tuple[bool] | None
                         ) -> tuple[tuple[int], tuple[int]]:
        """
        Returns the shape and the flat axes for the given topology

        Parameters
        ----------
        topo : tuple[bool] | None
            The topology of the array
        
        Returns
        -------
        shape : tuple[int]
            The shape of the array
        flat_axes : tuple[int]
            The flat axes of the array
        """
        shape = self.shape
        flat_axes = [i for i, is_extended in enumerate(topo or []) if not is_extended]
        # we have to adjust the shape for the topology
        if topo is not None:
            shape = list(self.shape)
            # each axis that is not extended has size 1
            for i, is_extended in enumerate(topo):
                if not is_extended:
                    shape[i] = 1
        return tuple(shape), tuple(flat_axes)

    def create_array(self, 
                     pad: bool = True, 
                     spectral: bool = False,
                     topo: tuple[bool] | None = None
                     ) -> ndarray:
        dtype = fr.config.dtype_comp if spectral else fr.config.dtype_real
        shape, flat_axes = self._get_array_attrs(topo)
        # create the array
        arr = fr.config.ncp.zeros(shape, dtype=dtype)
        # pad the array
        if pad and not spectral:
            arr = self.pad(arr, flat_axes)
        return arr

    def create_random_array(self, 
                            seed: int = 1234,
                            pad: bool = True,
                            spectral: bool = False,
                            topo: tuple[bool] | None = None
                            ) -> ndarray:
        dtype = fr.config.dtype_comp if spectral else fr.config.dtype_real
        shape, flat_axes = self._get_array_attrs(None)
        # create the array
        arr = fr.utils.random_array(shape, seed).astype(dtype)
        # add imaginary part if the array is complex
        if spectral:
            imag = fr.utils.random_array(shape, 2*seed+3).astype(dtype)
            arr = arr + 1j*imag
        # pad the array
        if pad and not spectral:
            return self.pad(arr, flat_axes)
        return arr

    def create_meshgrid(self, 
                        *args: ndarray, 
                        pad: bool = True,
                        spectral: bool = False) -> tuple[ndarray]:
        X = fr.config.ncp.meshgrid(*args, indexing='ij')
        if pad:
            X = tuple(self.pad(x) for x in X)
        return X

    # ================================================================
    #  Array operations
    # ================================================================

    def sum(self, 
            arr: ndarray, 
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        arr = self.unpad(arr)
        return fr.config.ncp.sum(arr, axis=axes)

    def max(self,
            arr: ndarray, 
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        arr = self.unpad(arr)
        return fr.config.ncp.max(arr, axis=axes)

    def min(self,
            arr: ndarray, 
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        arr = self.unpad(arr)
        return fr.config.ncp.min(arr, axis=axes)
