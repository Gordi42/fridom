import numpy as np
# Import internal modules
from .domain_decomposition import DomainDecomposition


class Transformer:
    """
    Transformation module that transforms data from one domain to another.
    
    Description
    -----------
    Let's assume we have two domain decompositions, one that has pencils
    in the x-direction and one that has pencils in the y-direction. The
    input array is a x-pencil array and we want to transform it to a
    y-pencil array. Hence, one processor has to send different parts of
    the input array to different processors and receive different parts 
    of the output array from different processors. This transformer
    function does exactly that. It is not limited to pencils and can be
    used for any domain decomposition.
    
    Parameters
    ----------
    `domain_in` : `DomainDecomposition`
        The input domain.
    `domain_out` : `DomainDecomposition`
        The output domain.
        
    Attributes
    ----------
    `domain_in` : `DomainDecomposition`
        The input domain.
    `domain_out` : `DomainDecomposition`
        The output domain.
    
    Methods
    -------
    `forward(arr_in, arr_out=None)` : `np.ndarray` or `cupy.ndarray`
        Transform an array from the input domain to the output domain.
    `backward(arr_in, arr_out=None)` : `np.ndarray` or `cupy.ndarray`
        Transform an array from the output domain to the input domain.
    
    Examples
    --------

    >>> from fridom.framework import config
    >>> from fridom.framework \\
    ...     .domain_decomposition import DomainDecomposition, Transformer
    >>> # create two domains where one shares the x-axis and the other the y-axis
    >>> domain_x = DomainDecomposition(n_global=[128]*2, shared_axes=[0])
    >>> domain_y = DomainDecomposition(n_global=[128]*2, shared_axes=[1])
    >>> 
    >>> # create a random array on the local domain
    >>> u = config.ncp.random.rand(*domain_x.my_subdomain.shape)
    >>> domain_x.sync(u)
    >>> 
    >>> # construct transformers between the domains
    >>> transformer = Transformer(domain_x, domain_y)
    >>> 
    >>> # transform the array from domain_x to domain_y
    >>> v = transformer.forward(u)
    >>> assert v.shape == domain_y.my_subdomain.shape
    >>> 
    >>> # transform the array back from domain_y to domain_x
    >>> w = transformer.backward(v)
    >>> assert config.ncp.allclose(u, w)
    """
    def __init__(self, 
                 domain_in: 'DomainDecomposition', 
                 domain_out: 'DomainDecomposition') -> None: ...

    def forward(self, 
                arr_in: np.ndarray, 
                arr_out: np.ndarray | None = None
                ) -> np.ndarray:
        """
        Transform an array from the input domain to the output domain.
        
        Parameters
        ----------
        `arr_in` : `np.ndarray` or `cupy.ndarray`
            The array to be transformed (lives in the input domain).
        `arr_out` : `np.ndarray` or `cupy.ndarray`
            The transformed array (lives in the output domain). If None, a new
            array will be created.
        
        Returns
        -------
        `np.ndarray` or `cupy.ndarray`
            The transformed array. If `arr_out` is not None, the output array
            will be the same as `arr_out`.
        """
    
    def backward(self, 
                 arr_in: np.ndarray, 
                 arr_out: np.ndarray | None = None
                 ) -> np.ndarray:
        """
        Transform an array from the output domain to the input domain.
        
        Parameters
        ----------
        `arr_in` : `np.ndarray` or `cupy.ndarray`
            The array to be transformed (lives in the output domain).
        `arr_out` : `np.ndarray` or `cupy.ndarray`
            The transformed array (lives in the input domain). If None, a new
            array will be created.
        
        Returns
        -------
        `np.ndarray` or `cupy.ndarray`
            The transformed array. If `arr_out` is not None, the output array
            will be the same as `arr_out`.
        """

    @property
    def domain_in(self) -> DomainDecomposition:
        """The domain decomposition of the input domain."""

    @property
    def domain_out(self) -> DomainDecomposition:
        """The domain decomposition of the output domain."""