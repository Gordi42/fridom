import fridom.framework as fr
import numpy as np
from numpy import ndarray
import jax
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.custom_partitioning import custom_partitioning
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jaxdecomp
import fridom.framework as fr
from functools import partial, cached_property

# ================================================================
#  Custom partitioning to avoid unnecessary communication
# ================================================================

def _supported_sharding(sharding: NamedSharding, shape):
    rank = len(shape.shape)
    max_shared_dims = min(len(sharding.spec), rank-1)
    names = tuple(sharding.spec[:max_shared_dims]) + tuple(None for _ in range(rank - max_shared_dims))
    return NamedSharding(sharding.mesh, P(*names))

def _infer_sharding_from_operands(mesh, arg_shapes, result_shape):
    arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
    return _supported_sharding(arg_shardings[0], arg_shapes[0])

def _partitionate_function(f: callable, 
                          in_shardings: NamedSharding,
                          out_shardings: NamedSharding):
    def partition(mesh, arg_shapes, result_shape):
        arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
        return (mesh, 
                f,               
                _supported_sharding(arg_shardings[0], arg_shapes[0]),               
                (_supported_sharding(arg_shardings[0], arg_shapes[0]),))
    
    my_f = custom_partitioning(f)
    my_f.def_partition(
        infer_sharding_from_operands=_infer_sharding_from_operands,
        partition=partition)
    return jax.jit(my_f, in_shardings=in_shardings, out_shardings=out_shardings)


@fr.utils.jaxify
class JaxDecomposition(fr.domain_decomposition.DomainDecomposition):
    def __init__(self,
                 shape: tuple[int],
                 halo: int = 0,
                 periods: tuple[bool] | None = None,
                 p_dims: tuple[int] | None = None,
                 shared_axes: tuple[int] | None = None,
                 device_ids: list[int] | None = None):
        super().__init__(shape, halo, periods, shared_axes, device_ids)

        # create a device array
        device_ids = device_ids or jax.devices()
        size = len(device_ids)

        if len(shape) != 3:
            raise ValueError("Only 3D domains are supported.")

        self._p_dims = p_dims or (size, 1)
        if self.size != size:
            raise ValueError(f"Number of available devices: {size} does not match to the processor grid: {self.p_dims}")

        self._shared_axes = [i for i, x in enumerate(self.p_dims) if x == 1]
        all_axes = set(range(3))
        self._z_ffts = self._shared_axes
        self._y_ffts = list(all_axes - set(self._shared_axes))
        self._x_ffts = list(all_axes - set(self._shared_axes) - set(self._y_ffts))

        # ----------------------------------------------------------------
        #  Create the device mesh
        # ----------------------------------------------------------------
        devices = mesh_utils.create_device_mesh(self.p_dims, devices=device_ids)
        self.mesh = Mesh(devices, axis_names=('x', 'y'))
        self.p_phys = P('x', 'y', None)
        self.p_spec = P('y', None, 'x')
        self.shard_phys = NamedSharding(self.mesh, self.p_phys)
        self.shard_spec = NamedSharding(self.mesh, self.p_spec)

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
        self._inner_slice = tuple([slice(halo, -halo)]*self.n_dims)
        self._inner = _make_slice_tuple(slice(halo, -halo))
        self._send_to_next = _make_slice_tuple(slice(-2*halo, -halo))
        self._send_to_prev = _make_slice_tuple(slice(halo, 2*halo))
        self._recv_from_next = _make_slice_tuple(slice(-halo, None))
        self._recv_from_prev = _make_slice_tuple(slice(None, halo))
        
        pw = self.halo
        self._padding = ((pw, pw), (pw, pw), (pw, pw))
        self._halo_extents = (pw, pw)

    # ================================================================
    #  Halo exchange
    # ================================================================

    @partial(fr.utils.jaxjit, static_argnames=['flat_axes'])
    def sync(self, arr: ndarray, flat_axes: list[int] | None = None) -> ndarray:
        if flat_axes is not None:
            raise NotImplementedError("Flat axes not supported in JaxDecomposition.")
        arr = jaxdecomp.halo_exchange(
            arr, halo_extents=self._halo_extents, halo_periods=self._periods[:-1])

        for axis in self.shared_axes:
            arr = self._sync_shared_axis(arr, axis)
        return arr

    @partial(fr.utils.jaxjit, static_argnames=['axis'])
    def _sync_shared_axis(self, arr: ndarray, axis: int,) -> ndarray:
        @partial(shard_map, mesh=self.mesh, in_specs=self.p_phys, out_specs=self.p_phys)
        def _sync(arr: ndarray) -> ndarray:
            rfn = self._recv_from_next[axis]
            rfp = self._recv_from_prev[axis]
            stn = self._send_to_next[axis]
            stp = self._send_to_prev[axis]
            arr = arr.at[rfn].set(arr[stp])
            arr = arr.at[rfp].set(arr[stn])
            return arr
        return _sync(arr)

    # ================================================================
    #  Apply Transform (e.g. FFT)
    # ================================================================

    # def parallel_forward_transform(self, func: callable) -> callable:
    #     @partial(jax.jit, static_argnames=['axes'])
    #     def my_transform(arr, axes: tuple[int] | None = None):
    #         # unpad the array
    #         arr = self.unpad(arr)

    #         arr = func(arr, axes=(0, ))
    #         arr = func(arr, axes=(1, ))
    #         arr = func(arr, axes=(2, ))
    #         return arr
    #     return my_transform

    #     return self._xy_pencil_forward(func)

    # def parallel_backward_transform(self, func: callable) -> callable:
    #     @partial(jax.jit, static_argnames=['axes'])
    #     def my_transform(arr, axes: tuple[int] | None = None):
    #         arr = func(arr, axes=(0, ))
    #         arr = func(arr, axes=(1, ))
    #         arr = func(arr, axes=(2, ))
    #         return self.pad(arr)
    #     return my_transform

    #     return self._xy_pencil_backward(func)

    def _xy_pencil_forward(self, func: callable) -> callable:
        # operation in the z-axis
        @partial(_partitionate_function,
                in_shardings=NamedSharding(self.mesh, P('x', 'y', None)),
                out_shardings=NamedSharding(self.mesh, P('x', 'y', None)))
        def func_axis_2(x):
            return func(x, axes=(2,))

        # operation in the x-axis
        @partial(_partitionate_function, 
                in_shardings=NamedSharding(self.mesh, P('y', 'x', None)), 
                out_shardings=NamedSharding(self.mesh, P('y', 'x', None)))
        def func_axis_0(x):
            x = jax.numpy.transpose(x, (2, 0, 1))
            x = func(x, axes=(0,))
            x = jax.numpy.transpose(x, (1, 2, 0))
            return x

        # operation in the y-pencil
        @partial(_partitionate_function,
                in_shardings=NamedSharding(self.mesh, P('x', 'y', None)),
                out_shardings=NamedSharding(self.mesh, P('x', 'y', None)))
        def func_axis_1(x):
            x = jax.numpy.transpose(x, (1, 2, 0))
            x = func(x, axes=(1,))
            x = jax.numpy.transpose(x, (2, 0, 1))
            return x

        @partial(jax.jit, static_argnames=['axes'])
        def my_forward_transform(arr, axes: list[int] | None = None):
            axes = axes or list(range(self.n_dims))
            # unpad the array
            arr = self.unpad(arr)
            # apply the forward transform in the z-axis
            if 2 in axes:
                arr = func_axis_2(arr)
            # make the x-axis the shared axis
            arr = jaxdecomp.transposeZtoY(arr)
            # apply the forward transform in the x-axis
            if 0 in axes:
                arr = func_axis_0(arr)
            # make the y-axis the shared axis
            arr = jaxdecomp.transposeYtoX(arr)
            # apply the forward transform in the y-axis
            if 1 in axes:
                arr = func_axis_1(arr)
            # transpose the array back to the original shape
            arr = jax.numpy.transpose(arr, (1, 2, 0))
            return arr
        
        return my_forward_transform

    def _xy_pencil_backward(self, func: callable) -> callable:
        # operation in the z-axis
        @partial(_partitionate_function,
                in_shardings=NamedSharding(self.mesh, P('x', 'y', None)),
                out_shardings=NamedSharding(self.mesh, P('x', 'y', None)))
        def func_axis_2(x):
            return func(x, axes=(2,))

        # operation in the x-axis
        @partial(_partitionate_function, 
                in_shardings=NamedSharding(self.mesh, P('y', 'x', None)), 
                out_shardings=NamedSharding(self.mesh, P('y', 'x', None)))
        def func_axis_0(x):
            x = jax.numpy.transpose(x, (2, 0, 1))
            x = func(x, axes=(0,))
            x = jax.numpy.transpose(x, (1, 2, 0))
            return x

        # operation in the y-pencil
        @partial(_partitionate_function,
                in_shardings=NamedSharding(self.mesh, P('x', 'y', None)),
                out_shardings=NamedSharding(self.mesh, P('x', 'y', None)))
        def func_axis_1(x):
            x = jax.numpy.transpose(x, (1, 2, 0))
            x = func(x, axes=(1,))
            x = jax.numpy.transpose(x, (2, 0, 1))
            return x

        @partial(jax.jit, static_argnames=['axes'])
        def my_backward_transform(arr, axes: list[int] | None = None):
            axes = axes or list(range(self.n_dims))
            # transpose the array to match jaxdecomp
            arr = jax.numpy.transpose(arr, (2, 0, 1))
            # apply the forward transform in the y-axis
            if 1 in axes:
                arr = func_axis_1(arr)
            # make the x-axis the shared axis
            arr = jaxdecomp.transposeXtoY(arr)
            # apply the forward transform in the x-axis
            if 0 in axes:
                arr = func_axis_0(arr)
            # make the z-axis the shared axis
            arr = jaxdecomp.transposeYtoZ(arr)
            # apply the forward transform in the z-axis
            if 2 in axes:
                arr = func_axis_2(arr)
            # restore the padding
            arr = self.pad(arr)
            arr = self.sync(arr)
            return arr
        
        return my_backward_transform

    # ================================================================
    #  Padding
    # ================================================================

    @cached_property
    def pad(self) -> callable:

        @partial(shard_map, mesh=self.mesh, in_specs=self.p_phys, out_specs=self.p_phys)
        def _pad(arr: ndarray) -> ndarray:
            ncp = fr.config.ncp

            if self.halo == 0:
                return arr

            arr = ncp.pad(arr, pad_width=self._padding, mode="wrap")
            return arr

        @partial(jax.jit, static_argnames=['flat_axes'])
        def pad(arr: ndarray, flat_axes: list[int] | None = None) -> ndarray:
            if flat_axes is not None:
                raise NotImplementedError("Flat axes not supported in JaxDecomposition.")

            return _pad(arr)
        return pad


    @cached_property
    def unpad(self) -> callable:
        @partial(jax.jit, static_argnames=['flat_axes'])
        @partial(shard_map, mesh=self.mesh, in_specs=self.p_phys, out_specs=self.p_phys)
        def unpad(arr: ndarray, flat_axes: tuple[int] | None = None) -> ndarray:
            if flat_axes is not None:
                raise NotImplementedError("Flat axes not supported in JaxDecomposition.")

            if self.halo == 0:
                return arr

            return arr[self._inner_slice]
        return unpad

    # ----------------------------------------------------------------
    #  Spectral paddings
    # ----------------------------------------------------------------

    def pad_extend(self, arr: ndarray) -> ndarray:
        raise NotImplementedError("Spectral padding not supported in JaxDecomposition.")

    def unpad_extend(self, arr: ndarray) -> ndarray:
        raise NotImplementedError("Spectral padding not supported in JaxDecomposition.")

    def pad_trim(self, arr: ndarray) -> ndarray:
        raise NotImplementedError("Spectral padding not supported in JaxDecomposition.")

    # ================================================================
    #  Gather
    # ================================================================

    def gather(self, 
               arr: ndarray, 
               slc: tuple[slice] | None = None,
               dest_rank: int | None = None,
               spectral: bool = False) -> ndarray:
        # first we unpad the array
        if not spectral:
            arr = self.unpad(arr)
        # gather the array
        slc = slc or (slice(None), )*self.n_dims
        if jax.process_count() == 1:
            # single process
            return jax.device_get(arr[slc])
        # multiple processes
        if dest_rank is not None:
            raise ValueError("dest_rank is not supported in JaxDecomposition.")
        return multihost_utils.process_allgather(arr[slc], tiled=True)

    # ================================================================
    #  Array creation
    # ================================================================

    @partial(jax.jit, static_argnames=['pad', 'spectral'])
    def create_array(self, 
                     pad: bool = True, 
                     spectral: bool = False) -> ndarray:
        dtype = fr.config.dtype_comp if spectral else fr.config.dtype_real
        sharding = self.shard_spec if spectral else self.shard_phys

        @partial(jax.jit, out_shardings=sharding)
        def create_zeros():
            return jax.numpy.zeros(self.shape, dtype=dtype)
        arr = create_zeros()

        if pad and not spectral:
            arr = self.pad(arr)
            arr = self.sync(arr)
        return arr

    def create_random_array(self, 
                            seed: int = 1234,
                            pad: bool = True,
                            spectral: bool = False) -> ndarray:
        dtype = fr.config.dtype_comp if spectral else fr.config.dtype_real
        sharding = self.shard_spec if spectral else self.shard_phys
        @partial(jax.jit, out_shardings=sharding)
        def create_random_array():
            real = jax.random.normal(jax.random.PRNGKey(seed), self.shape)
            if not spectral:
                return real.astype(dtype)
            imag = jax.random.normal(jax.random.PRNGKey(2*seed+3), self.shape)
            return jax.numpy.array(real + 1j*imag, dtype=dtype)
        arr = create_random_array()

        if pad and not spectral:
            arr = self.pad(arr)
            arr = self.sync(arr)
        return arr


    def create_meshgrid(self, 
                        *args: ndarray, 
                        pad: bool = True,
                        spectral: bool = False) -> tuple[ndarray]:
        sharding = self.shard_spec if spectral else self.shard_phys
        shardings = [sharding]*len(args)
        @partial(jax.jit, out_shardings=shardings)
        def create_meshgrid():
            return jax.numpy.meshgrid(*args, indexing='ij')
        arrs = create_meshgrid()
        
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
        arr = self.unpad(arr)
        return jax.numpy.sum(arr, axis=axes)

    def max(self,
            arr: ndarray, 
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        arr = self.unpad(arr)
        return jax.numpy.max(arr, axis=axes)

    def min(self,
            arr: ndarray, 
            axes: list[int] | None = None,
            spectral: bool = False) -> ndarray:
        arr = self.unpad(arr)
        return jax.numpy.min(arr, axis=axes)

    # ================================================================
    #  Helper functions
    # ================================================================
    def shard_map(self, func: callable) -> callable:
        return shard_map(func, 
                         mesh=self.mesh, 
                         in_specs=self.p_phys, 
                         out_specs=self.p_phys)