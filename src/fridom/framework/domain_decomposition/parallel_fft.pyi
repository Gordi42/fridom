# Import external modules
import numpy as np
# Import internal modules
from .domain_decomposition import DomainDecomposition

class ParallelFFT:
    """
    A class to perform fourier transforms on decomposed domains.
    
    Description
    -----------
    This class provides methods to perform forward and backward fourier transforms
    on decomposed domains. Besides the classical fftn and ifftn functions, this
    class also provides the possibility to apply a custom transformation 
    function. For example if one wants to apply cosine transforms along some axes
    and fft along others.

    Parameters
    ----------
    `domain_in` : `DomainDecomposition`
        The domain decomposition of the input data
    `shared_axes_out` : `list[int]`, optional (default=None)
        The axes that should be shared in the output domain
    `halo_out` : `int`, optional (default=0)
        The halo size of the output domain
    
    Attributes
    ----------
    `domain_in` : `DomainDecomposition`
        The domain decomposition of the input data (usually the physical domain)
    `domain_out` : `DomainDecomposition`
        The domain decomposition of the output data (usually the spectral domain)
    
    Methods
    -------
    `forward(u: np.ndarray) -> np.ndarray`
        Forward Fourier Transform
    `backward(u_hat: np.ndarray) -> np.ndarray`
        Backward Fourier Transform (Inverse Transform)
    `forward_apply(u: np.ndarray, apply_fun: callable) -> np.ndarray`
        Custom Forward Transform
    `backward_apply(u: np.ndarray, apply_fun: callable) -> np.ndarray`
        Custom Backward Transform
    
    Examples
    --------
    >>> import numpy as np
    >>> from fridom.framework \\
    ...     .domain_decomposition import DomainDecomposition, ParallelFFT
    >>> # Create a 3D physical domain that is decomposed along the z-axis and
    >>> # two halo points at each side of the domain
    >>> domain = DomainDecomposition(n_global=[64]*3, shared_axes=[0,1], halo=2)
    >>> 
    >>> # Create a ParallelFFT with a spectral domain that is decomposed along 
    >>> # the kx-axes and zero halo points
    >>> pfft = ParallelFFT(domain, shared_axes_out=[1,2], halo_out=0)
    >>> 
    >>> # Get the spectral domain
    >>> domain_spectral = pfft.domain_out
    >>> 
    >>> # Create a random field in the physical domain
    >>> u = np.random.rand(*domain.my_subdomain.shape)
    >>> domain.sync(u)
    >>> 
    >>> # Perform a forward transform
    >>> u_hat = pfft.forward(u)
    >>> 
    >>> # Perform a backward transform
    >>> v = pfft.backward(u_hat)
    >>> 
    >>> # Check if the data is the same
    >>> assert np.allclose(u, v)
    """
    def __init__(self, 
                 domain_in: DomainDecomposition, 
                 shared_axes_out : 'list[int]' = None,
                 halo_out: int = 0) -> None: ...

    def forward(self, arr: np.ndarray) -> np.ndarray:
        """
        Perform a forward fourier transform on the input data.
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The input data to be fourier transformed (can be real or complex)
        
        Returns
        -------
        `np.ndarray`
            The fourier transformed data (complex)
        
        Examples
        --------
        >>> import numpy as np
        >>> from fridom.framework \\
        ...     .domain_decomposition import DomainDecomposition, ParallelFFT
        >>> # Create the domain decomposition and the ParallelFFT object
        >>> domain = DomainDecomposition(n_global=[64]*3, shared_axes=[0,1])
        >>> pfft = ParallelFFT(domain)
        >>> 
        >>> # Create a random field in the physical domain
        >>> u = np.random.rand(*domain.my_subdomain.shape)
        >>> domain.sync(u)
        >>> 
        >>> # Perform a forward transform
        >>> u_hat = pfft.forward(u)
        """

    def backward(self, arr: np.ndarray) -> np.ndarray:
        """
        Perform a backward fourier transform on the input data.
        
        Parameters
        ----------
        `u_hat` : `np.ndarray`
            The fourier transformed data to be transformed back to the physical
            domain (complex)
        
        Returns
        -------
        `np.ndarray`
            The transformed data in the physical domain (complex)
        
        Examples
        --------

        >>> import numpy as np
        >>> from fridom.framework \\
        ...     .domain_decomposition import DomainDecomposition, ParallelFFT
        >>> # Create the domain decomposition and the ParallelFFT object
        >>> domain = DomainDecomposition(n_global=[64]*3, shared_axes=[0,1])
        >>> pfft = ParallelFFT(domain)
        >>>
        >>> # Get the spectral domain
        >>> domain_spectral = pfft.domain_out
        >>>
        >>> # Create a random field in the spectral domain
        >>> u = np.random.rand(*domain_spectral.my_subdomain.shape)
        >>>
        >>> # Perform a backward transform
        >>> u = pfft.backward(u)  # will be complex
        """

    def forward_apply(self, arr: np.ndarray, apply_fun: callable) -> np.ndarray:
        """
        Do a forward transform while applying a function to the data at each step.
        
        Description
        -----------
        This function can be used to create a custom forward transform that works
        on decomposed domains. For example, a normal fft along the x and y axis 
        and a cosine transform along the z axis. For that one would create a 
        function my_custom_transform(u, axes) that applies the given transform
        along the given axes. This custom function would then be passed to this
        method.
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The input data to be transformed
        `apply_fun` : `callable`
            The function that will be applied to the data at each step, should have
            the signature `apply_fun(arr, axes) -> np.ndarray`

        Returns
        -------
        `np.ndarray`
            The transformed data (lives in the output domain)
        
        Examples
        --------
        >>> from fridom.framework \\
        ...     .domain_decomposition import DomainDecomposition, ParallelFFT
        >>> # Create a custom transform that applies a normal fft along the x axis
        >>> # and a cosine transform along the y axis
        >>> import numpy as np
        >>> from scipy.fftpack import dct
        >>> 
        >>> def my_custom_transform(u, axes):
        ...     if 0 in axes:
        ...         u = np.fft.fftn(u, axes=(0,))
        ...     if 1 in axes:
        ...         u = dct(u, axis=1)
        ...     return u
        >>> 
        >>> # Create the domain decomposition and the ParallelFFT object
        >>> domain = DomainDecomposition(n_global=[64]*2, shared_axes=[0])
        >>> pfft = ParallelFFT(domain)
        >>> 
        >>> # Create a random field in the physical domain
        >>> u = np.random.rand(*domain.my_subdomain.shape)
        >>> domain.sync(u)
        >>> 
        >>> # Perform a custom forward transform
        >>> u_hat = pfft.forward_apply(u, my_custom_transform)
        """

    def backward_apply(self, arr: np.ndarray, apply_fun: callable) -> np.ndarray:
        """
        The same as the forward_apply method but for the backward transform.
        For more information see the forward_apply method.
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The input data to be transformed (lives in the output domain)
        `apply_fun` : `callable`
            The function that will be applied to the data at each step, should have
            the signature `apply_fun(arr, axes) -> np.ndarray`
        
        Returns
        -------
        `np.ndarray`
            The transformed data (lives in the input domain)
        """

    @property
    def domain_in(self) -> DomainDecomposition:
        """The domain decomposition of the input data"""

    @property
    def domain_out(self) -> DomainDecomposition:
        """The domain decomposition of the output data"""
