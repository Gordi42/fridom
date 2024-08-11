# Import external modules
import numpy as np
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from .domain_decomposition import DomainDecomposition
from .transformer import Transformer

@partial(utils.jaxjit, static_argnames=['transform_funs', 'apply_fun', 'fft_axes'])
def transform(arr_in: np.ndarray, 
              domain_in: DomainDecomposition, 
              domain_out: DomainDecomposition, 
              transform_funs: 'list[callable]',
              fft_axes: 'list[list[int]]',
              apply_fun: callable) -> np.ndarray:
    """
    Perform a transform from one domain to another, while applying a function
    to the data at each step.
    
    Description
    -----------
    Let's say we want to apply a transformation to a field but the transformation
    requires the data of the full domain along the axes that are not shared. This
    function will perform the transformation along the shared axes and then
    transform the data to the next domain, where the remaining axes are transformed.
    
    Parameters
    ----------
    `arr_in` : `np.ndarray`
        A numpy array containing the data to be transformed
    `domain_in` : `DomainDecomposition`
        The domain decomposition of the input data
    `domain_out` : `DomainDecomposition`
        The domain decomposition of the output data
    `transform_funs` : `list[callable]`
        The domain transformations that should be applied at each step
    `fft_axes` : `list[list[int]]`
        A list of lists containing the axes along the apply_fun should be applied
        at each step
    `apply_fun` : `callable`
        The function that will be applied to the data at each step, should have
        the signature `apply_fun(arr, axes) -> np.ndarray`
    
    Returns
    -------
    `np.ndarray`
        The transformed data
    
    Examples
    --------
    See the implementation in the forward and backward methods of the ParallelFFT
    class.
    """
    # first remove the halo cells of the input array and make the array complex
    arr_out = arr_in[domain_in.my_subdomain.inner_slice].astype(config.dtype_comp)
    for i, transform in enumerate(transform_funs):
        # apply the function to the given axes
        arr_out  = apply_fun(arr_out , axes=fft_axes[i])
        # arr_out  = config.ncp.fft.fftn(arr_out , axes=fft_axes[i])
        # transform the data to the next domain
        arr_out  = transform(arr_out )

    # check if there are any axes left to apply the function to
    if fft_axes[-1]:
        # function should only be applied to the inner slice without halo cells
        in_s = domain_out.my_subdomain.inner_slice
        arr_out = utils.modify_array(arr_out, in_s,
                        apply_fun(arr_out[in_s], axes=fft_axes[-1]))
        # sync the data to the full domain
        arr_out = domain_out.sync(arr_out)
    return arr_out 


@partial(utils.jaxify, dynamic=('_domain_in', '_domain_out'))
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
    
    Examples
    --------
    .. code-block:: python

        import numpy as np
        from fridom.framework.domain_decomposition import DomainDecomposition, ParallelFFT
        # Create a 3D physical domain that is decomposed along the z-axis and
        # two halo points at each side of the domain
        domain = DomainDecomposition(n_global=(64, 64, 64), shared_axes=[0,1], halo=2)
        
        # Create a ParallelFFT with a spectral domain that is decomposed along 
        # the kx-axes and zero halo points
        pfft = ParallelFFT(domain, shared_axes_out=[1,2], halo_out=0)
        
        # Get the spectral domain
        domain_spectral = pfft.domain_out
        
        # Create a random field in the physical domain
        u = np.random.rand(*domain.my_subdomain.shape)
        domain.sync(u)
        
        # Perform a forward transform
        u_hat = pfft.forward(u)
        
        # Perform a backward transform
        v = pfft.backward(u_hat)
        
        # Check if the data is the same
        assert np.allclose(u, v)
    """
    def __init__(self, 
                 domain_in: DomainDecomposition, 
                 shared_axes_out : 'list[int]' = None,
                 halo_out: int = 0) -> None:
        n_dims = domain_in.n_dims
        shared_axes_out = shared_axes_out or []

        # Find out which axes are shared in the input domain
        shared_axes_in = domain_in.shared_axes
        n_shared_axes = len(shared_axes_in)
        
        # --------------------------------------------------------------
        #  Construct the domain decomposition for the output
        # --------------------------------------------------------------

        # check if the number of shared axes is valid
        if n_shared_axes == 0:
            raise ValueError("The input domain must have at least one shared \
                             axis")
        if n_shared_axes < len(shared_axes_out):
            raise ValueError("The number of shared axes in the output domain \
                             must be less than or equal to the number of shared \
                             axes in the input domain")

        # Find out which axes are missing
        ax_set_all = set(range(n_dims))
        ax_set_in = set(shared_axes_in)
        ax_set_out = set(shared_axes_out)

        axes_missing = list(ax_set_all - ax_set_in - ax_set_out)

        # Fill the shared_axes_out with the missing axes until it has the same
        # number of elements as the number of shared axes in the input domain
        axes = shared_axes_in + axes_missing
        while len(shared_axes_out) < n_shared_axes:
            shared_axes_out.append(axes.pop())

        domain_out = DomainDecomposition(
            n_global=domain_in.n_global,
            halo=halo_out,
            shared_axes=shared_axes_out)
        
        # --------------------------------------------------------------
        #  Construct the domain decompositions for the intermediate domains
        # --------------------------------------------------------------

        # update the missing axes
        ax_set_out = set(shared_axes_out)
        axes_missing = list(ax_set_all - ax_set_in - ax_set_out)

        # split the missing axes into groups of n_shared_axes
        # and fill up the last group with the shared axes in the output domain
        if len(axes_missing) > 0:  # if there are missing axes
            n = len(axes_missing)
            shared_axes_mids = [axes_missing[i:i+n] 
                                for i in range(0, len(axes_missing), n)]
            shared_axes_mids[-1] += shared_axes_out
            shared_axes_mids[-1] = shared_axes_mids[-1][:n_shared_axes]
        else:  # if there are no missing axes
            shared_axes_mids = []

        # create all missing domains
        # we also recreate the input and output domain without halo cells
        # to avoid halo cells in the fft
        all_shared_axes = [shared_axes_in] + shared_axes_mids + [shared_axes_out]
        domain_list_all = [DomainDecomposition(
            n_global=domain_in.n_global,
            halo=0,
            shared_axes=shared_axes)
            for shared_axes in all_shared_axes]

        # the final domain in the domain list should include the halo cells
        domain_list_forward = domain_list_all[:-1] + [domain_out]
        domain_list_backward = [domain_in] + domain_list_all[1:]

        # -------------------------------------------------------------
        #  Create the transformers
        # -------------------------------------------------------------

        number_of_transforms = len(shared_axes_mids) + 1
        forward_transforms = [Transformer(domain_list_forward[i], 
                                          domain_list_forward[i+1]).forward
                              for i in range(number_of_transforms)]
        backward_transforms = [Transformer(domain_list_backward[i],
                                           domain_list_backward[i+1]).backward
                               for i in range(number_of_transforms)][::-1]

        # -------------------------------------------------------------
        #  Find the axes to be fourier transformed (avoid double ffts)
        # -------------------------------------------------------------

        # axes to be fourier transformed at each domain
        missing_fft_axes = set(range(n_dims))
        fft_axes = [tuple(shared_axes_in)]
        missing_fft_axes -= set(shared_axes_in)
        for shared_axes_mid in shared_axes_mids:
            missing_axes = list(set(shared_axes_mid) & missing_fft_axes)
            fft_axes.append(tuple(missing_axes))
            missing_fft_axes -= set(missing_axes)

        missing_axes = list(set(shared_axes_out) & missing_fft_axes)
        fft_axes.append(tuple(missing_axes))

        # -------------------------------------------------------------
        #  Set the attributes
        # -------------------------------------------------------------
        # public readable attributes
        self._domain_in = domain_in
        self._domain_out = domain_out

        # private attributes
        self._forward_transforms = tuple(forward_transforms)
        self._backward_transforms = tuple(backward_transforms)
        self._fft_axes = fft_axes
        return

    @utils.jaxjit
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
        .. code-block:: python

            import numpy as np
            from fridom.framework.domain_decomposition import DomainDecomposition, ParallelFFT
            # Create the domain decomposition and the ParallelFFT object
            domain = DomainDecomposition(n_global=(64, 64, 64), shared_axes=[0,1])
            pfft = ParallelFFT(domain)
            
            # Create a random field in the physical domain
            u = np.random.rand(*domain.my_subdomain.shape)
            domain.sync(u)
            
            # Perform a forward transform
            u_hat = pfft.forward(u)
        """
        return self.forward_apply(arr, config.ncp.fft.fftn)

    @utils.jaxjit
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
        .. code-block:: python

            import numpy as np
            from fridom.framework.domain_decomposition import DomainDecomposition, ParallelFFT
            # Create the domain decomposition and the ParallelFFT object
            domain = DomainDecomposition(n_global=(64, 64, 64), shared_axes=[0,1])
            pfft = ParallelFFT(domain)
           
            # Get the spectral domain
            domain_spectral = pfft.domain_out
           
            # Create a random field in the spectral domain
            u = np.random.rand(*domain_spectral.my_subdomain.shape)
           
            # Perform a backward transform
            u = pfft.backward(u)  # will be complex
        """
        return self.backward_apply(arr, config.ncp.fft.ifftn)

    @partial(utils.jaxjit, static_argnames=['apply_fun'])
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
        .. code-block:: python

            from fridom.framework.domain_decomposition import DomainDecomposition, ParallelFFT
            # Create a custom transform that applies a normal fft along the x axis
            # and a cosine transform along the y axis
            import numpy as np
            from scipy.fftpack import dct
            
            def my_custom_transform(u, axes):
                if 0 in axes:
                    u = np.fft.fftn(u, axes=(0,))
                if 1 in axes:
                    u = dct(u, axis=1)
                return u
            
            # Create the domain decomposition and the ParallelFFT object
            domain = DomainDecomposition(n_global=(64, 64), shared_axes=[0])
            pfft = ParallelFFT(domain)
            
            # Create a random field in the physical domain
            u = np.random.rand(*domain.my_subdomain.shape)
            domain.sync(u)
            
            # Perform a custom forward transform
            u_hat = pfft.forward_apply(u, my_custom_transform)
        """
        return transform(arr, self._domain_in, self._domain_out,
                        self._forward_transforms, tuple(self._fft_axes),
                        apply_fun)

    @partial(utils.jaxjit, static_argnames=['apply_fun'])
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
        return transform(arr, self._domain_out, self._domain_in,
                        self._backward_transforms, tuple(self._fft_axes[::-1]),
                        apply_fun)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def domain_in(self) -> DomainDecomposition:
        """The domain decomposition of the input data"""
        return self._domain_in

    @property
    def domain_out(self) -> DomainDecomposition:
        """The domain decomposition of the output data"""
        return self._domain_out
