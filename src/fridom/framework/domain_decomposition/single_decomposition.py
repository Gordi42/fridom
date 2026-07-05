"""Domain decomposition for single-device (serial) execution."""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom.framework as fr

if TYPE_CHECKING:
    from numpy import ndarray


@fr.utils.jaxify
class SingleDecomposition(fr.domain_decomposition.DomainDecomposition):

    """Domain decomposition where a single process holds the full domain."""

    def __init__(self, shape: tuple[int],
                 halo: int = 0,
                 periods: tuple[bool] | None = None,
                 shared_axes: tuple[int] | None = None,
                 device_ids: list[int] | None = None) -> None:
        super().__init__(shape, halo, periods, shared_axes, device_ids)
        self._p_dims = tuple([1]*self.n_dims)

        # ----------------------------------------------------------------
        #  Halo exchange slices and paddings
        # ----------------------------------------------------------------

        def _make_slice_tuple(slc: slice) -> tuple[tuple[slice, ...], ...]:
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

    # ================================================================
    #  Halo exchange
    # ================================================================

    def sync(
        self, arr: ndarray, flat_axes: list[int] | None = None,
    ) -> ndarray:
        """Synchronize the halo regions of an array."""
        # nothing to do if there are no halo regions
        if self.halo == 0:
            return arr

        flat_axes = flat_axes or []

        # synchronize one dimension at a time
        for axis in range(self.n_dims):
            if axis in flat_axes:
                continue
            if self.periods[axis]:
                arr = self._sync_periodic_axis(arr, axis)
            else:
                arr = self._sync_non_periodic_axis(arr, axis)
        return arr

    def _sync_periodic_axis(self, x: ndarray, axis: int) -> ndarray:
        halo = self.halo
        x = jnp.swapaxes(x, 0, axis)
        x = jnp.concatenate(
            [ x[-2*halo:-halo], x[halo:-halo], x[halo:2*halo] ], axis=0)
        return jnp.swapaxes(x, 0, axis)

    def _sync_non_periodic_axis(self, x: ndarray, axis: int) -> ndarray:
        halo = self.halo
        x = jnp.swapaxes(x, 0, axis)
        halo_region = jnp.zeros_like(x[:halo])
        x = jnp.concatenate([halo_region, x[halo:-halo], halo_region], axis=0)
        return jnp.swapaxes(x, 0, axis)

    # ================================================================
    #  Padding
    # ================================================================

    def pad(
        self, arr: ndarray, flat_axes: tuple[int] | None = None,
    ) -> ndarray:
        """Add halo padding to an array."""
        if self.halo == 0:
            return arr
        # update the paddings for flat axes
        pw_periodic = list(self._pw_periodic)
        pw_nonperiodic = list(self._pw_nonperiodic)
        for axis in flat_axes or []:
            pw_periodic[axis] = (0, 0)
            pw_nonperiodic[axis] = (0, 0)
        # pad the array
        arr = jnp.pad(arr, tuple(pw_periodic), mode="wrap")
        return jnp.pad(arr, tuple(pw_nonperiodic), mode="constant")

    def unpad(
        self, arr: ndarray, flat_axes: tuple[int] | None = None,
    ) -> ndarray:
        """Remove halo padding from an array."""
        if self.halo == 0:
            return arr
        # remove the paddings for flat axes
        ics = list(self._inner_slice)
        for axis in flat_axes or []:
            ics[axis] = slice(None)
        return arr[tuple(ics)]

    # ================================================================
    #  Gather
    # ================================================================

    def gather(self,
               arr: ndarray,
               slc: tuple[slice] | None = None,
               dest_rank: int | None = None,  # noqa: ARG002 (interface conformity)
               spectral: bool = False) -> ndarray:  # noqa: ARG002 (interface conformity)
        """Gather an array to a single process."""
        if arr.shape == self.shape:
            return arr[slc]
        return arr[self._inner_slice][slc]

    # ================================================================
    #  Array creation
    # ================================================================

    def _get_array_attrs(self,
                         topo: tuple[bool] | None
                         ) -> tuple[tuple[int], tuple[int]]:
        """
        Return the shape and the flat axes for the given topology.

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
        flat_axes = [
            i for i, is_extended in enumerate(topo or [])
            if not is_extended]
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
        """Create an array filled with zeros."""
        dtype = fr.utils.dtype_comp() if spectral else fr.utils.dtype_real()
        shape, flat_axes = self._get_array_attrs(topo)
        # create the array
        arr = jnp.zeros(shape, dtype=dtype)
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
        """Create an array filled with random numbers."""
        dtype = fr.utils.dtype_comp() if spectral else fr.utils.dtype_real()
        shape, flat_axes = self._get_array_attrs(topo)
        # create the array
        arr = fr.utils.random_array(
            shape, seed, ignore_warning=True).astype(dtype)
        # add imaginary part if the array is complex
        if spectral:
            imag = fr.utils.random_array(
                shape, 2*seed+3, ignore_warning=True).astype(dtype)
            arr = arr + 1j*imag
        # pad the array
        if pad and not spectral:
            return self.pad(arr, flat_axes)
        return arr

    def create_meshgrid(self,
                        *args: ndarray,
                        pad: bool = True,
                        spectral: bool = False) -> tuple[ndarray]:  # noqa: ARG002 (interface conformity)
        """Create a meshgrid of arrays."""
        mesh = jnp.meshgrid(*args, indexing="ij")
        if pad:
            mesh = tuple(self.pad(x) for x in mesh)
        return mesh

    # ================================================================
    #  Array operations
    # ================================================================

    def sum(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:  # noqa: ARG002 (interface conformity)
        """Sum an array across specified axes."""
        arr = self.unpad(arr)
        return jnp.sum(arr, axis=axes, keepdims=True)

    def max(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:  # noqa: ARG002 (interface conformity)
        """Find the maximum of an array across specified axes."""
        arr = self.unpad(arr)
        return jnp.max(arr, axis=axes, keepdims=True)

    def min(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:  # noqa: ARG002 (interface conformity)
        """Find the minimum of an array across specified axes."""
        arr = self.unpad(arr)
        return jnp.min(arr, axis=axes, keepdims=True)

    def cumsum(self,  # noqa: D102
               arr: ndarray,
               axis: int,
               ) -> ndarray:
        arr = self.unpad(arr)
        cumsum = jnp.cumsum(arr, axis=axis)
        return self.pad(cumsum)

    def inv_cumsum(self,  # noqa: D102
                   arr: ndarray,
                   axis: int,
                   ) -> ndarray:
        arr = self.unpad(arr)
        # reverse the array in the given axis
        arr = jnp.flip(arr, axis=axis)
        # calculate the cumsum
        cumsum = jnp.cumsum(arr, axis=axis)
        # reverse the array back
        cumsum = jnp.flip(cumsum, axis=axis)
        return self.pad(cumsum)

    def roll(self,  # noqa: D102
             arr: ndarray,
             shift: int | tuple[int],
             axis: int | tuple[int]) -> ndarray:
        return jnp.roll(arr, shift, axis)
