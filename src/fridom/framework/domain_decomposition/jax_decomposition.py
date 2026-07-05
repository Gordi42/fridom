"""Domain decomposition based on JAX sharding."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import fridom.framework as fr

if TYPE_CHECKING:
    from numpy import ndarray


@fr.utils.jaxify
class JaxDecomposition(fr.domain_decomposition.DomainDecomposition):

    """Domain decomposition that distributes arrays with JAX sharding."""

    def __init__(self,
                 shape: tuple[int],
                 halo: int = 0,
                 periods: tuple[bool] | None = None,
                 shared_axes: tuple[int] | None = None,
                 device_ids: list[int] | None = None) -> None:
        super().__init__(shape, halo, periods, shared_axes, device_ids)

        self._rank = jax.process_index()
        self.n_ranks = jax.process_count()
        self.n_devices = (
            jax.device_count() if self.n_ranks == 1 else self.n_ranks)

        # the first axis is distributed across the devices
        if shape[0] % self.n_devices != 0:
            msg = (
                f"The first dimension of the shape {shape} must be "
                f"divisible by the number of devices {self.n_devices}.")
            raise ValueError(msg)

        local_shape = list(self.shape)
        local_shape[0] //= self.n_devices
        self._local_shape = tuple(local_shape)

        for dim in range(len(local_shape)):
            if local_shape[dim] < self.halo:
                msg = (
                    f"Local shape {local_shape} is smaller than halo "
                    f"{self.halo} in dimension {dim}")
                raise ValueError(msg)

        self._p_dims = (self.n_devices,) + (1,) * (self.n_dims - 1)
        self._setup_mesh()

        # ----------------------------------------------------------------
        #  Halo exchange slices and paddings
        # ----------------------------------------------------------------
        pw = self.halo
        self._padding = (((pw, pw),) * self.n_dims)

        inner = slice(self.halo, -self.halo) if self.halo > 0 else slice(None)
        self._inner_slice = tuple([inner]*self.n_dims)

    def _setup_mesh(self) -> None:
        """Create the device mesh and the sharding specifications."""
        self.mesh = jax.make_mesh((self.n_devices,), axis_names=("x",))
        if self.n_devices == 1:
            # a single device holds the whole domain: every sharding
            # is fully replicated, shard maps reduce to plain calls,
            # and all array operations stay ordinary local operations
            self._spec_main = P(*(None,) * self.n_dims)
            self._spec_alt = self._spec_main
        elif self.n_dims > 1:
            self._spec_main = P("x", * (self.n_dims - 1) * (None,))
            # spectral arrays are distributed along the second axis, so
            # that fourier transforms along the first axis are local
            self._spec_alt = self._permute_spec(0, 1)
        else:
            self._spec_main = P("x")
            # 1-d domains have no second axis to transpose to; the
            # alternative sharding replicates the array instead
            self._spec_alt = P(None)
        self._shard_main = NamedSharding(self.mesh, self._spec_main)
        self._shard_alt = NamedSharding(self.mesh, self._spec_alt)

    def _permute_spec(self, dim1: int, dim2: int) -> P:
        spec_list = list(self._spec_main)
        spec_list[dim1], spec_list[dim2] = spec_list[dim2], spec_list[dim1]
        return P(*spec_list)

    # ================================================================
    #  Pickling
    # ================================================================

    def __getstate__(self) -> dict:
        """Return the state for pickling.

        Description
        -----------
        The device mesh and the shardings hold device handles that
        cannot be pickled; they are dropped here and rebuilt from the
        remaining state when unpickling.
        """
        state = self.__dict__.copy()
        for attr in ("mesh", "_spec_main", "_spec_alt",
                     "_shard_main", "_shard_alt"):
            state.pop(attr, None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state and rebuild the device mesh."""
        self.__dict__.update(state)
        self._setup_mesh()

    def __to_numpy__(self, memo: dict) -> JaxDecomposition:
        """Return a host-side copy (see :py:func:`fr.utils.to_numpy`).

        Description
        -----------
        The decomposition holds no device arrays, so a deep copy
        (which rebuilds the device mesh) is sufficient. Without this
        hook, the generic attribute walk of ``to_numpy`` would try to
        deep-copy the raw mesh and fail on its device handles.
        """
        return deepcopy(self, memo)

    # ================================================================
    #  Halo exchange
    # ================================================================

    def sync(
        self, arr: ndarray, flat_axes: list[int] | None = None,
    ) -> ndarray:
        """Synchronize the halo regions of an array across all processes."""
        # nothing to do if there are no halo regions
        if self.halo == 0:
            return arr

        flat_axes = flat_axes or []

        # the first axis is distributed across the devices and requires
        # a halo exchange between neighboring devices
        if 0 not in flat_axes:
            arr = self._sync_sharded_axis(arr)

        # all other axes are local to each device
        for axis in range(1, self.n_dims):
            if axis in flat_axes:
                continue
            if self.periods[axis]:
                arr = self._sync_periodic_axis(arr, axis)
            else:
                arr = self._sync_non_periodic_axis(arr, axis)
        return arr

    def _sync_sharded_axis(self, arr: ndarray) -> ndarray:
        if self.n_devices == 1:
            # a single device is its own neighbor: the halo exchange
            # reduces to a periodic wrap (or zero boundaries)
            if self.periods[0]:
                return self._sync_periodic_axis(arr, 0)
            return self._sync_non_periodic_axis(arr, 0)

        halo = self.halo
        n_devices = self.n_devices

        @self.main_shard_map
        def halo_exchange(x: ndarray) -> ndarray:
            left_halo = x[halo : 2 * halo]
            right_halo = x[-(2 * halo) : -halo]

            permutations_forward = [
                (i, (i + 1) % n_devices) for i in range(n_devices)]
            permutations_backward = [
                (i, (i - 1) % n_devices) for i in range(n_devices)]

            if not self.periods[0]:
                # last device has no right neighbor
                permutations_forward = permutations_forward[:-1]
                # first device has no left neighbor
                permutations_backward = permutations_backward[1:]

            received_left_halo = jax.lax.ppermute(
                right_halo,
                axis_name="x",
                perm=permutations_forward,
            )
            received_right_halo = jax.lax.ppermute(
                left_halo,
                axis_name="x",
                perm=permutations_backward,
            )

            return jnp.concatenate(
                [received_left_halo, x[halo:-halo], received_right_halo],
                axis=0)

        return halo_exchange(arr)

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
    #  Apply Transform (e.g. FFT)
    # ================================================================

    def parallel_forward_transform(self, func: callable) -> callable:
        """Wrap a forward transform to work on distributed arrays."""
        if self.n_devices == 1:
            # the whole domain is local: apply the transform directly
            def _local_forward_transform(
                arr: ndarray, axes: list[int] | None = None,
            ) -> ndarray:
                return func(self.unpad(arr), axes=axes)
            return _local_forward_transform

        def _my_forward_transform(
            arr: ndarray, axes: list[int] | None = None,
        ) -> ndarray:
            axes = set(axes or list(range(self.n_dims)))
            # unpad the array
            arr = self.unpad(arr)
            # apply the forward transform in all but x-axis
            possible_axes = set(range(1, self.n_dims))
            matching_axes = possible_axes & axes
            if matching_axes:
                @self.main_shard_map
                def apply_func(arr: ndarray) -> ndarray:
                    return func(arr, axes=tuple(matching_axes))
                arr = apply_func(arr)
            # switch to alternative sharding
            arr = jax.device_put(arr, self._shard_alt)
            # apply the forward transform in the x-axis
            if 0 in axes:
                @self.alt_shard_map
                def apply_func(arr: ndarray) -> ndarray:
                    return func(arr, axes=(0,))
                arr = apply_func(arr)
            return arr

        return _my_forward_transform

    def parallel_backward_transform(self, func: callable) -> callable:
        """Wrap a backward transform to work on distributed arrays."""
        if self.n_devices == 1:
            # the whole domain is local: apply the transform directly
            def _local_backward_transform(
                arr: ndarray, axes: list[int] | None = None,
            ) -> ndarray:
                return self.sync(self.pad(func(arr, axes=axes)))
            return _local_backward_transform

        def _my_backward_transform(
            arr: ndarray, axes: list[int] | None = None,
        ) -> ndarray:
            axes = set(axes or list(range(self.n_dims)))
            # apply the backward transform in the x-axis
            if 0 in axes:
                @self.alt_shard_map
                def apply_func(arr: ndarray) -> ndarray:
                    return func(arr, axes=(0,))
                arr = apply_func(arr)

            # switch to main sharding
            arr = jax.device_put(arr, self._shard_main)

            # apply the forward transform in all but x-axis
            possible_axes = set(range(1, self.n_dims))
            matching_axes = possible_axes & axes
            if matching_axes:
                @self.main_shard_map
                def apply_func(arr: ndarray) -> ndarray:
                    return func(arr, axes=tuple(matching_axes))
                arr = apply_func(arr)
            arr = self.pad(arr)
            return self.sync(arr)

        return _my_backward_transform

    # ================================================================
    #  Padding
    # ================================================================

    def pad(
        self, arr: ndarray, flat_axes: list[int] | None = None,
    ) -> ndarray:
        """Add halo padding to an array."""
        if self.halo == 0:
            return arr

        # update the paddings for flat axes
        paddings = list(self._padding)
        for axis in flat_axes or []:
            paddings[axis] = (0, 0)
        paddings = tuple(paddings)

        if self.n_devices == 1 or 0 in (flat_axes or []):
            # a single device or a flat first axis carries no
            # distributed halo: all padding is local
            return jnp.pad(arr, paddings)

        # each device shard carries its own halo region on the
        # distributed axis
        @self.main_shard_map
        def _pad(arr: ndarray) -> ndarray:
            return jnp.pad(arr, paddings)

        return _pad(arr)

    def unpad(
        self, arr: ndarray, flat_axes: list[int] | None = None,
    ) -> ndarray:
        """Remove halo padding from an array."""
        if self.halo == 0:
            return arr

        # remove the paddings for flat axes
        ics = list(self._inner_slice)
        for axis in flat_axes or []:
            ics[axis] = slice(None)
        ics = tuple(ics)

        if self.n_devices == 1 or 0 in (flat_axes or []):
            # a single device or a flat first axis carries no
            # distributed halo: all slicing is local
            return arr[ics]

        @self.main_shard_map
        def _unpad(arr: ndarray) -> ndarray:
            return arr[ics]

        return _unpad(arr)

    # ================================================================
    #  Gather
    # ================================================================

    def gather(self,
               arr: ndarray,
               slc: tuple[slice] | None = None,
               dest_rank: int | None = None,  # noqa: ARG002 (interface conformity)
               spectral: bool = False) -> ndarray:
        """Gather a distributed array on all processes."""
        if slc is None:
            slc = (slice(None), )*self.n_dims
        if not spectral:
            arr = self.unpad(arr)
        return multihost_utils.process_allgather(arr, tiled=True)[slc]

    # ================================================================
    #  Array creation
    # ================================================================
    def _get_array_attrs(self,
                         topo: tuple[bool] | None
                         ) -> tuple[tuple[int], tuple[int]]:
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
                     topo: tuple[bool] | None = None,
                     ) -> ndarray:
        """Create a sharded array filled with zeros."""
        dtype = fr.utils.dtype_comp() if spectral else fr.utils.dtype_real()
        sharding = self._get_sharding(spectral, topo)
        shape, flat_axes = self._get_array_attrs(topo)

        if self.n_devices == 1:
            # create the array directly (jitted creation would
            # retrace on every call)
            arr = jnp.zeros(shape, dtype=dtype)
        else:
            @partial(jax.jit, out_shardings=sharding)
            def create_zeros() -> ndarray:
                return jnp.zeros(shape, dtype=dtype)
            arr = create_zeros()
            arr = jax.reshard(arr, sharding)

        if pad and not spectral:
            arr = self.pad(arr, flat_axes)
            arr = self.sync(arr, flat_axes)
        return arr

    def create_random_array(self,
                            seed: int = 1234,
                            pad: bool = True,
                            spectral: bool = False,
                            topo: tuple[bool] | None = None,
                            ) -> ndarray:
        """Create a sharded array filled with random numbers."""
        dtype = fr.utils.dtype_comp() if spectral else fr.utils.dtype_real()
        sharding = self._get_sharding(spectral, topo)
        shape, flat_axes = self._get_array_attrs(topo)

        def _random_array() -> ndarray:
            real = jax.random.normal(jax.random.PRNGKey(seed), shape)
            if not spectral:
                return real.astype(dtype)
            imag = jax.random.normal(jax.random.PRNGKey(2*seed+3), shape)
            return jnp.array(real + 1j*imag, dtype=dtype)

        if self.n_devices == 1:
            # create the array directly (jitted creation would
            # retrace on every call)
            arr = _random_array()
        else:
            arr = jax.jit(_random_array, out_shardings=sharding)()
            arr = jax.reshard(arr, sharding)

        if pad and not spectral:
            arr = self.pad(arr, flat_axes)
            arr = self.sync(arr, flat_axes)
        return arr


    def create_meshgrid(self,
                        *args: ndarray,
                        pad: bool = True,
                        spectral: bool = False) -> tuple[ndarray]:
        """Create a sharded meshgrid of arrays."""
        sharding = self._get_sharding(spectral, None)
        if self.n_devices == 1:
            # create the meshgrid directly (jitted creation would
            # retrace on every call)
            arrs = jnp.meshgrid(*args, indexing="ij")
        else:
            shardings = [sharding]*len(args)
            @partial(jax.jit, out_shardings=shardings)
            def create_meshgrid() -> list[ndarray]:
                return jnp.meshgrid(*args, indexing="ij")
            arrs = create_meshgrid()
            arrs = [jax.reshard(arr, sharding) for arr in arrs]

        if pad and not spectral:
            # synchronize so that periodic halo regions carry the
            # wrapped coordinate values instead of zeros
            return tuple(self.sync(self.pad(arr)) for arr in arrs)
        return tuple(arrs)


    # ================================================================
    #  Array operations
    # ================================================================

    def sum(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:  # noqa: ARG002 (interface conformity)
        """Sum an array across specified axes."""
        # the halo cells must not contribute to the sum
        arr = self.unpad(arr)
        return jnp.sum(arr, axis=axes, keepdims=True)

    def max(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:  # noqa: ARG002 (interface conformity)
        """Find the maximum of an array across specified axes."""
        # the halo cells must not contribute to the maximum
        arr = self.unpad(arr)
        return jnp.max(arr, axis=axes, keepdims=True)

    def min(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:  # noqa: ARG002 (interface conformity)
        """Find the minimum of an array across specified axes."""
        # the halo cells must not contribute to the minimum
        arr = self.unpad(arr)
        return jnp.min(arr, axis=axes, keepdims=True)

    def _cumsum_along_axis(self, arr: ndarray, axis: int) -> ndarray:
        """Cumulative sum of an unpadded array along an axis."""
        if axis > 0:
            # the axis is local to each device
            return jnp.cumsum(arr, axis=axis)
        # a cumulative sum along the distributed axis is computed under
        # the alternative sharding, where the first axis is local
        arr = self.to_alterative_sharding(arr)
        cumsum = jnp.cumsum(arr, axis=axis)
        return self.to_main_sharding(cumsum)

    def cumsum(self,  # noqa: D102
               arr: ndarray,
               axis: int,
               ) -> ndarray:
        arr = self.unpad(arr)
        cumsum = self._cumsum_along_axis(arr, axis)
        return self.sync(self.pad(cumsum))

    def inv_cumsum(self,  # noqa: D102
                   arr: ndarray,
                   axis: int,
                   ) -> ndarray:
        arr = self.unpad(arr)
        # reverse the array in the given axis
        arr = jnp.flip(arr, axis=axis)
        # calculate the cumsum
        cumsum = self._cumsum_along_axis(arr, axis)
        # reverse the array back
        cumsum = jnp.flip(cumsum, axis=axis)
        return self.sync(self.pad(cumsum))

    # ================================================================
    #  Helper functions
    # ================================================================
    def _get_sharding(
        self,
        spectral: bool = False,
        topo: tuple[bool] | None = None,
    ) -> NamedSharding:
        shard = self._shard_alt if spectral else self._shard_main
        if topo is not None:
            new_specs = []
            for ax, is_extended in zip(shard.spec, topo, strict=False):
                if is_extended:
                    new_specs.append(ax)
                else:
                    new_specs.append(None)
            shard = NamedSharding(self.mesh, P(*new_specs))
        return shard

    def _shard_map_with(self, func: callable,
                        sharding: NamedSharding) -> callable:
        """Apply a shard map with the given sharding to a function."""
        if self.n_devices == 1:
            # a single shard covers the whole domain: the mapped
            # function can be applied directly (eager shard maps
            # carry a large dispatch overhead)
            return func

        mapped = shard_map(func,
                           mesh=self.mesh,
                           in_specs=sharding.spec,
                           out_specs=sharding.spec)

        def wrapper(*args: ndarray) -> ndarray:
            # shard_map does not reshard its inputs, so arrays that
            # carry a different sharding (e.g. plain arrays assigned
            # directly to a field) are moved to the target sharding
            args = tuple(jax.device_put(arg, sharding) for arg in args)
            return mapped(*args)
        return wrapper

    def main_shard_map(self, func: callable) -> callable:
        """Apply a shard map with the main sharding to a function."""
        return self._shard_map_with(func, self._shard_main)

    def alt_shard_map(self, func: callable) -> callable:
        """Apply a shard map with the alternative sharding to a function."""
        return self._shard_map_with(func, self._shard_alt)

    def shard_map(self, func: callable) -> callable:
        """Decorate a function to apply it to the active processes only."""
        return self.main_shard_map(func)

    def to_alterative_sharding(self, arr: ndarray) -> ndarray:
        """Convert an array to the alternative sharding."""
        return jax.device_put(arr, self._shard_alt)

    def to_main_sharding(self, arr: ndarray) -> ndarray:
        """Convert an array to the main sharding."""
        return jax.device_put(arr, self._shard_main)
