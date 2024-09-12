import fridom.framework as fr
# Import external modules
import numpy as np
from functools import partial
# Import internal modules
from fridom.framework import config, utils


def _create_kn_mesh(N: int):
    ncp = config.ncp
    n = ncp.arange(0, N)
    n, k = ncp.meshgrid(n, n, indexing="ij")
    return n, k

def _apply_weights(x, weights, axis):
    ncp = config.ncp
    y = ncp.tensordot(x, weights, axes=([axis], [0]))
    y = ncp.moveaxis(y, -1, axis)
    return y

@partial(utils.jaxjit, static_argnames=['axis', 'N'])
def dct_type2(x, axis, N):
    ncp = config.ncp
    n, k = _create_kn_mesh(N)
    weights = 2 * ncp.cos((ncp.pi / N) * k * (n + 0.5))
    return _apply_weights(x, weights, axis)

@partial(utils.jaxjit, static_argnames=['axis', 'N'])
def idct_type2(x, axis, N):
    ncp = config.ncp
    k, n = _create_kn_mesh(N)
    weights = 2 * ncp.cos((ncp.pi / N) * k * (n + 0.5))
    weights = utils.modify_array(weights, (0, slice(None)), 1)
    return _apply_weights(x, weights, axis) / (2 * N)

@partial(utils.jaxjit, static_argnames=['axis', 'N'])
def dst_type1(x, axis, N):
    # we assume that the position of the variable is at the cell edges
    # |-----x-----|-----x-----|-----x-----|-----x-----|
    #             ^           ^           ^           ^
    #            x0          x1          x2          x(N-1)
    # A function f with frequency k is given by:
    # f(xi) = sin(k*(xi+dx/2))
    #       = -i/2 * (exp(i*k*(xi+dx/2)) - exp(-i*k*(xi+dx/2)))
    # we only consider positive frequencies in the sine transform
    # f(xi) = -i/2 * exp(i*k*(xi+dx/2))
    #       = -i/2 * exp(i*k*dx/2) * exp(i*k*xi)
    # the factor 1/2 does not matter, but we need the rotation by -i*exp(i*k*dx/2)
    # so that the sine transform is consistent with fourier transforms
    # Note that dx is given by pi/N
    ncp = config.ncp
    n, k = _create_kn_mesh(N)
    weights = 2 * ncp.sin(ncp.pi * k * (n+1) / N)
    # apply the rotation factor
    weights *= -1j * ncp.exp(1j*k*ncp.pi/(2*N))
    return _apply_weights(x, weights, axis)

@partial(utils.jaxjit, static_argnames=['axis', 'N'])
def idst_type1(x, axis, N):
    ncp = config.ncp
    k, n = _create_kn_mesh(N)
    weights = 2 * ncp.sin(ncp.pi * k * (n+1) / N)
    # similar as the dst1, we need to apply the inverse rotation factor
    weights *= 1j * ncp.exp(-1j*k*ncp.pi/(2 * N))
    return _apply_weights(x, weights, axis) / (2 * N)

@partial(utils.jaxjit, static_argnames=['axis', 'N'])
def dst_type2(x, axis, N):
    ncp = config.ncp
    n, k = _create_kn_mesh(N)
    weights = -2j * ncp.sin(ncp.pi * k * (2*n+1) / (2*N))
    return _apply_weights(x, weights, axis)

@partial(utils.jaxjit, static_argnames=['axis', 'N'])
def idst_type2(x, axis, N):
    ncp = config.ncp
    k, n = _create_kn_mesh(N)
    weights = 2j * ncp.sin(ncp.pi * k * (2*n+1) / (2*N))
    return _apply_weights(x, weights, axis) / (2 * N)


@utils.jaxify
class FFT:
    """
    Class for performing fourier transforms on a cartesian grid.
    
    Description
    -----------
    Model grids that have periodic boundary conditions in some directions, 
    and non-periodic boundary conditions in other directions, require a
    combination of fast fourier transforms and discrete cosine transforms.
    This class provides a method to transform an array from physical space to
    spectral space and back. For the discrete cosine transform, the type 2
    transform is used. This means that the variable must be located at the 
    cell centers in that direction.
        
    Parameters
    ----------
    `periodic` : `tuple[bool]`
        A list of booleans that indicate whether the axis is periodic.
        If True, the axis is periodic, if False, the axis is non-periodic.
    
    Examples
    --------
    .. code-block:: python

        import numpy as np
        from fridom.framework.grid.cartesian import FFT
        fft = FFT(periodic=(True, True, False))
        u = np.random.rand(*(32, 32, 8))
        v = fft.forward(u)
        w = fft.backward(v).real
        assert np.allclose(u, w)
    """
    def __init__(self, 
                 periodic: tuple[bool]) -> None:
        
        # --------------------------------------------------------------
        #  Check which axis to apply fft, dct
        # --------------------------------------------------------------
        fft_axes = []  # Periodic axes (fast fourier transform)
        dct_axes = []  # Non-periodic axes (discrete cosine transform)
        for i in range(len(periodic)):
            if periodic[i]:
                fft_axes.append(i)
            else:
                dct_axes.append(i)

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------

        # private attributes
        self._periodic = periodic
        self._fft_axes = fft_axes
        self._dct_axes = dct_axes
        return

    def get_freq(self, shape: tuple[int], 
                 dx: tuple[float], ) -> tuple[np.ndarray]:
        """
        Get the frequencies for the given shape and dx.
        
        Description
        -----------
        This method calculates the frequencies for the given shape and dx. The 
        returned frequencies could be used to construct wavenumber meshgrids.
        
        Parameters
        ----------
        `shape` : `tuple[int]`
            The global shape (number of grid points in each direction).
        `dx` : `tuple[float]`
            The grid spacing in each direction.
        
        Returns
        -------
        `tuple[np.ndarray]`
            The frequencies in each direction.
        
        Examples
        --------
        .. code-block:: python

            import numpy as np
            from fridom.framework.grid.cartesian import FFT
            fft = FFT(periodic=(True, True, False))
            shape = (32, 32, 8)  # Number of grid points in x,y,z
            dx = (0.1, 0.1, 0.1)  # Grid spacing in x,y,z
            kx, ky, kz = fft.get_freq(shape, dx)
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        """
        ncp = config.ncp
        k = []
        for i in range(len(shape)):
            if self._periodic[i]:
                k.append(ncp.fft.fftfreq(shape[i], dx[i]/(2*ncp.pi)))
            else:
                k.append(ncp.linspace(0, ncp.pi/dx[i], shape[i], endpoint=False))
        return tuple(k)

    @partial(utils.jaxjit, static_argnames=['axes', 'bc_types', 'positions'])
    def forward(self, 
                u: np.ndarray, 
                axes: list[int] | None = None,
                bc_types: tuple[fr.grid.BCType] | None = None,
                positions: tuple[fr.grid.AxisPosition] | None = None,
                ) -> np.ndarray:
        """
        Forward transform from physical space to spectral space.
        
        Parameters
        ----------
        `u` : `np.ndarray`
            The array to transform from physical space to spectral space.
        `axes` : `list[int] | None`
            The axes to transform. If None, all axes are transformed.
        `bc_types` : `tuple[fr.grid.BCType] | None`
            The type of boundary conditions for each axis.
        `positions` : `tuple[fr.grid.AxisPosition] | None`
            The position of the variable in each direction.
        
        Returns
        -------
        `np.ndarray`
            The transformed array in spectral space. If all dimensions are
            periodic, the obtained array is real, else it is complex.
        """
        ncp = config.ncp; scp = config.scp
        # Get the axes to apply fft, dct
        if axes is None:
            fft_axes = self._fft_axes
            dct_axes = self._dct_axes
        else:
            fft_axes = list(set(axes) & set(self._fft_axes))
            dct_axes = list(set(axes) & set(self._dct_axes))

        u_hat = u
        if bc_types is None:
            bc_types = tuple(fr.grid.BCType.NEUMANN for _ in range(u.ndim))

        if positions is None:
            positions = tuple(fr.grid.AxisPosition.CENTER for _ in range(u.ndim))
        
        # discrete cosine transform
        for axis in dct_axes:
            if bc_types[axis] == fr.grid.BCType.NEUMANN:
                if config.backend_is_jax:
                    u_hat = dct_type2(u_hat, axis, u_hat.shape[axis])
                else:
                    u_hat = scp.fft.dct(u_hat, axis=axis)
            
            if bc_types[axis] == fr.grid.BCType.DIRICHLET:
                if positions[axis] == fr.grid.AxisPosition.CENTER:
                    u_hat = dst_type2(u_hat, axis, u_hat.shape[axis])
                if positions[axis] == fr.grid.AxisPosition.FACE:
                    u_hat = dst_type1(u_hat, axis, u_hat.shape[axis])

        # fourier transform for periodic boundary conditions
        u_hat = ncp.fft.fftn(u_hat, axes=fft_axes)

        return u_hat

    @partial(utils.jaxjit, static_argnames=['axes', 'bc_types', 'positions'])
    def backward(self, u_hat: np.ndarray, 
                 axes: list[int] | None = None,
                 bc_types: tuple[fr.grid.BCType] | None = None,
                 positions: tuple[fr.grid.AxisPosition] | None = None,
                 ) -> np.ndarray:
        """
        Backward transform from spectral space to physical space.
        
        Parameters
        ----------
        `u_hat` : `np.ndarray`
            The array to transform from spectral space to physical space.
        `axes` : `list[int] | None`
            The axes to transform. If None, all axes are transformed.
        `bc_types` : `tuple[fr.grid.BCType] | None`
            The type of boundary conditions for each axis.
        `positions` : `tuple[fr.grid.AxisPosition] | None`
            The position of the variable in each direction.
        
        Returns
        -------
        `np.ndarray`
            The transformed array in physical space.
        """
        ncp = config.ncp; scp = config.scp
        if axes is None:
            fft_axes = self._fft_axes
            dct_axes = self._dct_axes
        else:
            fft_axes = list(set(axes) & set(self._fft_axes))
            dct_axes = list(set(axes) & set(self._dct_axes))

        # fourier transform for periodic boundary conditions
        u = ncp.fft.ifftn(u_hat, axes=fft_axes)

        if bc_types is None:
            bc_types = tuple(fr.grid.BCType.NEUMANN for _ in range(u.ndim))

        if positions is None:
            positions = tuple(fr.grid.AxisPosition.CENTER for _ in range(u.ndim))
        
        # discrete cosine transform
        for axis in dct_axes:
            if bc_types[axis] == fr.grid.BCType.NEUMANN:
                if config.backend_is_jax:
                    u = idct_type2(u, axis, u.shape[axis])
                else:
                    u = scp.fft.idct(u, axis=axis)
            
            if bc_types[axis] == fr.grid.BCType.DIRICHLET:
                if positions[axis] == fr.grid.AxisPosition.CENTER:
                    u = idst_type2(u, axis, u.shape[axis])
                if positions[axis] == fr.grid.AxisPosition.FACE:
                    u = idst_type1(u, axis, u.shape[axis])

        return u
