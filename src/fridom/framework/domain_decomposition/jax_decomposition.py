from __future__ import annotations

from functools import cached_property, partial
from typing import TYPE_CHECKING

import jax
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import fridom.framework as fr

if TYPE_CHECKING:
    from numpy import ndarray

MINIMUM_NUMBER_OF_DIMS = 2
ncp = fr.config.ncp

@fr.utils.jaxify
class JaxDecomposition(fr.domain_decomposition.DomainDecomposition):
    def __init__(self,
                 shape: tuple[int],
                 halo: int = 0,
                 periods: tuple[bool] | None = None,
                 p_dims: tuple[int] | None = None,
                 shared_axes: tuple[int] | None = None,
                 device_ids: list[int] | None = None) -> None:
        super().__init__(shape, halo, periods, shared_axes, device_ids)

        # initialize jax distributed
        # jax.distributed.initialize()
        self._rank = jax.process_index()
        self.n_ranks = jax.process_count()
        self.n_devices = jax.device_count() if self.n_ranks == 1 else self.n_ranks

        if len(shape) < MINIMUM_NUMBER_OF_DIMS:
            msg = f"Must have at least {MINIMUM_NUMBER_OF_DIMS} dimensions."
            raise ValueError(msg)


        self._p_dims = (self.n_devices,)
        self.mesh = jax.make_mesh(self._p_dims, axis_names=("x",))
        self._spec_main = P("x", * (len(shape) - 1) * (None,))
        self._spec_alt = self._permute_spec(0, 1)
        self._shard_main = NamedSharding(self.mesh, self._spec_main)
        self._shard_alt = NamedSharding(self.mesh, self._spec_alt)

        # ----------------------------------------------------------------
        #  Halo exchange slices and paddings
        # ----------------------------------------------------------------
        local_shape = list(self.shape)
        local_shape[0] //= self.n_devices
        self._local_shape = tuple(local_shape)

        for dim in range(len(local_shape)):
            if local_shape[dim] < self.halo:
                msg = f"Local shape {local_shape} is smaller than halo {self.halo} in dimension {dim}"
                raise ValueError(msg)


        pw = self.halo
        self._padding = (((pw, pw),) * self.n_dims)

        inner = slice(self.halo, -self.halo) if self.halo > 0 else slice(None)
        self._inner_slice = tuple([inner]*self.n_dims)

    def _permute_spec(self, dim1, dim2):
        spec_list = list(self._spec_main)
        spec_list[dim1], spec_list[dim2] = spec_list[dim2], spec_list[dim1]
        return P(*spec_list)

    # ================================================================
    #  Halo exchange
    # ================================================================

    def sync(self, arr: ndarray, flat_axes: list[int] | None = None) -> ndarray:
        halo = self.halo
        n_devices = self.n_devices

        @self.main_shard_map
        def halo_exchange_across_x(x):
            left_halo = x[halo : 2 * halo]
            right_halo = x[-(2 * halo) : -halo]

            permutations_forward = [(i, (i + 1) % n_devices) for i in range(n_devices)]
            permutations_backward = [(i, (i - 1) % n_devices) for i in range(n_devices)]

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

            return jax.numpy.concatenate(
                [received_left_halo, x[halo:-halo], received_right_halo], axis=0)

        def halo_exchange_axis(x, dim):
            x = ncp.swapaxes(x, 0, dim)
            spec = self._permute_spec(0, dim)
            def halo_exchange(x):
                if not self.periods[dim]:
                    left = right = ncp.zeros_like(x[:halo])
                else:
                    left = x[halo : 2 * halo]
                    right = x[-(2 * halo) : -halo]
                return ncp.concatenate([right, x[halo:-halo], left], axis=0)
            x = shard_map(halo_exchange, mesh=self.mesh, in_specs=spec, out_specs=spec)(x)
            return ncp.swapaxes(x, 0, dim)

        x = halo_exchange_across_x(arr)
        for dim in range(1, self.n_dims):
            if dim in (flat_axes or []):
                continue
            x = halo_exchange_axis(x, dim)
        return x


    # ================================================================
    #  Apply Transform (e.g. FFT)
    # ================================================================

    def parallel_forward_transform(self, func: callable) -> callable:

        def _my_forward_transform(arr: ndarray, axes: list[int] | None = None) -> ndarray:
            axes = set(axes or list(range(self.n_dims)))
            # unpad the array
            arr = self.unpad(arr)
            # apply the forward transform in all but x-axis
            possible_axes = set(range(1, self.n_dims))
            matching_axes = possible_axes & axes
            if matching_axes:
                @self.main_shard_map
                def apply_func(arr):
                    return func(arr, axes=tuple(matching_axes))
                arr = apply_func(arr)
            # switch to alternative sharding
            arr = jax.device_put(arr, self._shard_alt)
            # apply the forward transform in the x-axis
            if 0 in axes:
                @self.alt_shard_map
                def apply_func(arr):
                    return func(arr, axes=(0,))
                arr = apply_func(arr)
            return arr

        return _my_forward_transform

    def parallel_backward_transform(self, func: callable) -> callable:

        def _my_backward_transform(arr, axes: list[int] | None = None):
            axes = set(axes or list(range(self.n_dims)))
            # apply the backward transform in the x-axis
            if 0 in axes:
                @self.alt_shard_map
                def apply_func(arr):
                    return func(arr, axes=(0,))
                arr = apply_func(arr)

            # switch to main sharding
            arr = jax.device_put(arr, self._shard_main)

            # apply the forward transform in all but x-axis
            possible_axes = set(range(1, self.n_dims))
            matching_axes = possible_axes & axes
            if matching_axes:
                @self.main_shard_map
                def apply_func(arr):
                    return func(arr, axes=tuple(matching_axes))
                arr = apply_func(arr)
            arr = self.pad(arr)
            return self.sync(arr)

        return _my_backward_transform

    # ================================================================
    #  Padding
    # ================================================================

    @cached_property
    def pad(self) -> callable:
        def pad(arr: ndarray, flat_axes: list[int] | None = None) -> ndarray:
            if self.halo == 0:
                return arr

            # update the paddings for flat axes
            paddings = list(self._padding)
            for axis in flat_axes or []:
                paddings[axis] = (0, 0)

            @self.main_shard_map
            def _pad(arr: ndarray) -> ndarray:
                ncp = fr.config.ncp
                return ncp.pad(arr, tuple(paddings))

            return _pad(arr)
        return pad

    @cached_property
    def unpad(self) -> callable:
        def unpad(arr: ndarray, flat_axes: list[int] | None = None) -> ndarray:
            if self.halo == 0:
                return arr

            # remove the paddings for flat axes
            ics = list(self._inner_slice)
            for axis in flat_axes or []:
                ics[axis] = slice(None)

            @self.main_shard_map
            def _unpad(arr: ndarray) -> ndarray:
                return arr[tuple(ics)]

            return _unpad(arr)
        return unpad

    # ----------------------------------------------------------------
    #  Spectral paddings
    # ----------------------------------------------------------------

    def pad_extend(self, arr: ndarray) -> ndarray:
        msg = "Spectral padding not supported in JaxDecomposition."
        raise NotImplementedError(msg)

    def unpad_extend(self, arr: ndarray) -> ndarray:
        msg = "Spectral padding not supported in JaxDecomposition."
        raise NotImplementedError(msg)

    def pad_trim(self, arr: ndarray) -> ndarray:
        msg = "Spectral padding not supported in JaxDecomposition."
        raise NotImplementedError(msg)

    # ================================================================
    #  Gather
    # ================================================================

    def gather(self,
               arr: ndarray,
               slc: tuple[slice] | None = None,
               dest_rank: int | None = None,
               spectral: bool = False) -> ndarray:
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
                     topo: tuple[bool] | None = None,
                     ) -> ndarray:
        dtype = fr.config.dtype_comp if spectral else fr.config.dtype_real
        sharding = self._get_sharding(spectral, topo)
        shape, flat_axes = self._get_array_attrs(topo)

        @partial(jax.jit, out_shardings=sharding)
        def create_zeros():
            return jax.numpy.zeros(shape, dtype=dtype)
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
        dtype = fr.config.dtype_comp if spectral else fr.config.dtype_real
        sharding = self._get_sharding(spectral, topo)
        shape, flat_axes = self._get_array_attrs(topo)

        @partial(jax.jit, out_shardings=sharding)
        def create_random_array():
            real = jax.random.normal(jax.random.PRNGKey(seed), shape)
            if not spectral:
                return real.astype(dtype)
            imag = jax.random.normal(jax.random.PRNGKey(2*seed+3), shape)
            return jax.numpy.array(real + 1j*imag, dtype=dtype)
        arr = create_random_array()
        arr = jax.reshard(arr, sharding)

        if pad and not spectral:
            arr = self.pad(arr, flat_axes)
            arr = self.sync(arr, flat_axes)
        return arr


    def create_meshgrid(self,
                        *args: ndarray,
                        pad: bool = True,
                        spectral: bool = False) -> tuple[ndarray]:
        sharding = self._get_sharding(spectral, None)
        shardings = [sharding]*len(args)
        @partial(jax.jit, out_shardings=shardings)
        def create_meshgrid():
            return jax.numpy.meshgrid(*args, indexing="ij")
        arrs = create_meshgrid()
        arrs = [jax.reshard(arr, sharding) for arr in arrs]

        if pad and not spectral:
            return tuple(self.pad(arr) for arr in arrs)
        return arrs


    # ================================================================
    #  Array operations
    # ================================================================

    def sum(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        return jax.numpy.sum(arr, axis=axes)

    def max(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        return jax.numpy.max(arr, axis=axes)

    def min(self,
            arr: ndarray,
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        return jax.numpy.min(arr, axis=axes)


    # ================================================================
    #  Helper functions
    # ================================================================
    def _get_sharding(self, spectral: bool = False, topo: tuple[bool] | None = None) -> NamedSharding:
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

    def main_shard_map(self, func: callable) -> callable:
        return shard_map(func,
                         mesh=self.mesh,
                         in_specs=self._spec_main,
                         out_specs=self._spec_main)

    def alt_shard_map(self, func: callable) -> callable:
        return shard_map(func,
                         mesh=self.mesh,
                         in_specs=self._spec_alt,
                         out_specs=self._spec_alt)

    def shard_map(self, func: callable) -> callable:
        return self.main_shard_map(func)

    def to_alterative_sharding(self, arr: ndarray) -> ndarray:
        return jax.device_put(arr, self._shard_alt)

    def to_main_sharding(self, arr: ndarray) -> ndarray:
        return jax.device_put(arr, self._shard_main)
