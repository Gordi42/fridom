# Import external modules
import numpy as np
# Import internal modules
from fridom.framework import config, utils
from functools import partial

@partial(utils.jaxjit, static_argnames=['axis', 'N'])
def dct_type2(x, axis, N):
    ncp = config.ncp
    n = ncp.arange(0, N)
    n, k = ncp.meshgrid(n, n, indexing="ij")
    weights = 2 * ncp.cos((ncp.pi / N) * k * (n + 0.5))

    y = ncp.tensordot(x, weights, axes=([axis], [0]))
    y = ncp.moveaxis(y, -1, axis)
    return y

@partial(utils.jaxjit, static_argnames=['axis', 'N'])
def dct_type3(x, axis, N):
    ncp = config.ncp
    n = ncp.arange(0, N)
    n, k = ncp.meshgrid(n, n, indexing="ij")
    weights = 2 * ncp.cos((ncp.pi / N) * n * (k + 0.5))
    weights = utils.modify_array(weights, (0, slice(None)), 1)

    y = ncp.tensordot(x, weights, axes=([axis], [0]))
    y = ncp.moveaxis(y, -1, axis)
    return y / (2 * N)

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
    
    Methods
    -------
    `get_freq(shape: tuple, dx: tuple)`
        Get (kx, ky, ...) frequencies for the given shape and dx.
    `forward(u: np.ndarray, axes: list[int] | None = None) -> np.ndarray`
        Forward transform from physical space to spectral space.
    `backward(u_hat: np.ndarray, axes: list[int] | None = None) -> np.ndarray`
        Backward transform from spectral space to physical space.
    
    Examples
    --------
    >>> import numpy as np
    >>> from fridom.framework.grid.cartesian import FFT
    >>> fft = FFT(periodic=(True, True, False))
    >>> u = np.random.rand(*(32, 32, 8))
    >>> v = fft.forward(u)
    >>> w = fft.backward(v).real
    >>> assert np.allclose(u, w)

    """
    _dynamic_attributes = [ ]
    def __init__(self, periodic: tuple[bool]) -> None:
        
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
        >>> import numpy as np
        >>> from fridom.framework.grid.cartesian import FFT
        >>> fft = FFT(periodic=(True, True, False))
        >>> shape = (32, 32, 8)  # Number of grid points in x,y,z
        >>> dx = (0.1, 0.1, 0.1)  # Grid spacing in x,y,z
        >>> kx, ky, kz = fft.get_freq(shape, dx)
        >>> KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        """
        ncp = config.ncp
        k = []
        for i in range(len(shape)):
            if self._periodic[i]:
                k.append(ncp.fft.fftfreq(shape[i], dx[i]/(2*ncp.pi)))
            else:
                k.append(ncp.linspace(0, ncp.pi/dx[i], shape[i], endpoint=False))
        return tuple(k)

    @partial(utils.jaxjit, static_argnames=['axes',])
    def forward(self, u: np.ndarray, axes: list[int] | None = None) -> np.ndarray:
        """
        Forward transform from physical space to spectral space.
        
        Parameters
        ----------
        `u` : `np.ndarray`
            The array to transform from physical space to spectral space.
        `axes` : `list[int] | None`
            The axes to transform. If None, all axes are transformed.
        
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
        
        # discrete cosine transform
        for axis in dct_axes:
            if config.backend_is_jax:
                u_hat = dct_type2(u_hat, axis, u_hat.shape[axis])
                # u_hat = scp.fft.dct(u_hat, type=2, axis=axis)
            else:
                u_hat = scp.fft.dct(u_hat, type=2, axis=axis)

        # fourier transform for periodic boundary conditions
        u_hat = ncp.fft.fftn(u_hat, axes=fft_axes)
        

        return u_hat

    @partial(utils.jaxjit, static_argnames=['axes',])
    def backward(self, u_hat: np.ndarray, 
                 axes: list[int] | None = None) -> np.ndarray:
        """
        Backward transform from spectral space to physical space.
        
        Parameters
        ----------
        `u_hat` : `np.ndarray`
            The array to transform from spectral space to physical space.
        
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
        
        # discrete cosine transform
        for axis in dct_axes:
            if config.backend_is_jax:
                # u = scp.fft.idct(u, type=2, axis=axis)
                u = dct_type3(u, axis, u.shape[axis])
            else:
                u = scp.fft.idct(u, type=2, axis=axis)

        return u

utils.jaxify_class(FFT)
