# cython: language_level=3
from .domain_decomposition cimport DomainDecomposition
from .transformer import Transformer
from fridom.framework import config

cdef object transform(object arr_in, 
                      DomainDecomposition domain_in, 
                      DomainDecomposition domain_out, 
                      list transform_funs,
                      list fft_axes,
                      object apply_fun):
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
    cdef object arr_out, transform
    cdef int i
    arr_out = arr_in[domain_in.my_subdomain.inner_slice].astype(config.dtype_comp)
    for i, transform in enumerate(transform_funs):
        # apply the function to the given axes
        arr_out  = apply_fun(arr_out , axes=fft_axes[i])
        # transform the data to the next domain
        arr_out  = transform(arr_out )

    # check if there are any axes left to apply the function to
    cdef tuple in_s
    if fft_axes[-1]:
        # function should only be applied to the inner slice without halo cells
        in_s = domain_out.my_subdomain.inner_slice
        arr_out[in_s] = apply_fun(arr_out[in_s], axes=fft_axes[-1])
        # sync the data to the full domain
        domain_out.sync(arr_out)
    return arr_out 

cdef class ParallelFFT:
    def __init__(self, 
                 DomainDecomposition domain_in, 
                 list shared_axes_out = None,
                 int halo_out = 0):
        cdef int n_dims = domain_in.n_dims
        shared_axes_out = shared_axes_out or []

        # Find out which axes are shared in the input domain
        cdef list shared_axes_in = domain_in.shared_axes
        cdef int n_shared_axes = len(shared_axes_in)
        
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
        cdef set ax_set_all = set(range(n_dims))
        cdef set ax_set_in = set(shared_axes_in)
        cdef set ax_set_out = set(shared_axes_out)

        cdef list axes_missing = list(ax_set_all - ax_set_in - ax_set_out)

        # Fill the shared_axes_out with the missing axes until it has the same
        # number of elements as the number of shared axes in the input domain
        cdef list axes = shared_axes_in + axes_missing
        while len(shared_axes_out) < n_shared_axes:
            shared_axes_out.append(axes.pop())

        cdef DomainDecomposition domain_out = DomainDecomposition(
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
        cdef int n
        cdef list shared_axes_mids
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
        cdef list all_shared_axes
        all_shared_axes = [shared_axes_in] + shared_axes_mids + [shared_axes_out]
        cdef list domain_list_all = [DomainDecomposition(
            n_global=domain_in.n_global,
            halo=0,
            shared_axes=shared_axes)
            for shared_axes in all_shared_axes]

        # the final domain in the domain list should include the halo cells
        cdef list domain_list_forward = domain_list_all[:-1] + [domain_out]
        cdef list domain_list_backward = [domain_in] + domain_list_all[1:]

        # -------------------------------------------------------------
        #  Create the transformers
        # -------------------------------------------------------------

        cdef int number_of_transforms = len(shared_axes_mids) + 1
        cdef list forward_transforms = [Transformer(domain_list_forward[i], 
                                          domain_list_forward[i+1]).forward
                                        for i in range(number_of_transforms)]
        cdef list backward_transforms = [Transformer(domain_list_backward[i],
                                           domain_list_backward[i+1]).backward
                                         for i in range(number_of_transforms)]
        backward_transforms = backward_transforms[::-1]  # reverse the list

        # -------------------------------------------------------------
        #  Find the axes to be fourier transformed (avoid double ffts)
        # -------------------------------------------------------------

        # axes to be fourier transformed at each domain
        cdef set missing_fft_axes = set(range(n_dims))
        cdef list fft_axes = [shared_axes_in]
        cdef list missing_axes
        missing_fft_axes -= set(shared_axes_in)
        for shared_axes_mid in shared_axes_mids:
            missing_axes = list(set(shared_axes_mid) & missing_fft_axes)
            fft_axes.append(missing_axes)
            missing_fft_axes -= set(missing_axes)


        missing_axes = list(set(shared_axes_out) & missing_fft_axes)
        fft_axes.append(missing_axes)

        # -------------------------------------------------------------
        #  Set the attributes
        # -------------------------------------------------------------

        self.domain_in = domain_in
        self.domain_out = domain_out

        # private attributes
        self._forward_transforms = forward_transforms
        self._backward_transforms = backward_transforms
        self._fft_axes = fft_axes
        return

    cpdef object forward(self, object arr):
        return self.forward_apply(arr, config.ncp.fft.fftn)

    cpdef object backward(self, object arr):
        return self.backward_apply(arr, config.ncp.fft.ifftn)

    cpdef object forward_apply(self, object arr, object apply_func):
        return transform(arr, self.domain_in, self.domain_out,
                        self._forward_transforms, self._fft_axes,
                        apply_func)

    cpdef object backward_apply(self, object arr, object apply_func):
        return transform(arr, self.domain_out, self.domain_in,
                        self._backward_transforms, self._fft_axes[::-1],
                        apply_func)

    # ================================================================
    #  Properties
    # ================================================================
    property domain_in:
        def __get__(self):
            return self.domain_in

    property domain_out:
        def __get__(self):
            return self.domain_out