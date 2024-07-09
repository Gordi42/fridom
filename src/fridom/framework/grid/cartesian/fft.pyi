# Import external modules
import numpy as np

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
    `periodic` : `list[bool]`
        A list of booleans that indicate whether the axis is periodic.
        If True, the axis is periodic, if False, the axis is non-periodic.
    
    Methods
    -------
    `get_freq(shape, dx, periodic)`
        Get (kx, ky, ...) frequencies for the given shape and dx.
    `forward(u: np.ndarray, axes: list[int] | None = None) -> np.ndarray`
        Forward transform from physical space to spectral space.
    `backward(u_hat: np.ndarray, axes: list[int] | None = None) -> np.ndarray`
        Backward transform from spectral space to physical space.
    
    Examples
    --------
    >>> import numpy as np
    >>> from fridom.framework.grid.cartesian import FFT
    >>> fft = FFT(periodic=[True, True, False])
    >>> u = np.random.rand(*[32, 32, 8])
    >>> v = fft.forward(u)
    >>> w = fft.backward(v).real
    >>> assert np.allclose(u, w)

    """
    def __init__(self, periodic: list[bool]) -> None: ...

    def get_freq(self, 
                 shape: list[int], 
                 dx: list[float], ) -> tuple[np.ndarray]:
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
        >>> fft = FFT(periodic=[True, True, False])
        >>> shape = [32, 32, 8]  # Number of grid points in x,y,z
        >>> dx = [0.1, 0.1, 0.1]  # Grid spacing in x,y,z
        >>> kx, ky, kz = fft.get_freq(shape, dx)
        >>> KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        """

    def forward(self, 
                u: np.ndarray, 
                axes: list[int] | None = None) -> np.ndarray:
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

    def backward(self, 
                 u_hat: np.ndarray, 
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