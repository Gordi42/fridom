# cython: language_level=3
from mpi4py cimport MPI
from .domain_decomposition cimport DomainDecomposition
from .subdomain cimport Subdomain
from fridom.framework import config

cdef dict get_overlap_info(DomainDecomposition domain_in, 
                           DomainDecomposition domain_out):
    """
    Get information about which processors of the output domain overlaps with
    the subdomain of this processor in the input domain.
    
    Description
    -----------
    In case we want to send data from one domain to another, we need to know
    which processors of the output domain overlap with the subdomain of this
    processor in the input domain. This method provides this information.
    The method returns a dictionary containing a list of processors and slices.
    The slice s=slices[i] of an array arr has to be sent to the processor
    processors[i] of the output domain. For more information about the usage of
    this method, see the transformer function.
    
    Parameters
    ----------
    `domain_in` : `DomainDecomposition`
        The input domain.
    `domain_out` : `DomainDecomposition`
        The output domain.
    
    Returns
    -------
    `dict`
        A dictionary containing the overlap information. The dictionary has the
        following keys:
        - `overlap_slices` : list[slice]
            A list of slices that represent the overlap between the input and
            output domains.
        - `processors` : list[int]
            A list of the processors of the output domain that overlap with the
            input domain.
        - `slice_same_proc` : slice
            The slice of the input domain that overlaps with the output domain
            of the same processor.
    
    Examples
    --------
    ```
    # create two domains
    domain1 = DomainDecomposition(n_global=[128]*3, shared_axes=[0])
    domain2 = DomainDecomposition(n_global=[128]*3, shared_axes=[1])

    # get the overlap information
    overlap_info = get_overlap_info(domain1, domain2)

    print(f"The slice {overlap_info['overlap_slices'][0]} of domain1 \
    shares an overlap with processor {overlap_info['processors'][0]} \
    of the domain2.")
    ```
    """
    cdef list overlap_slices = []
    cdef list processors = []
    cdef tuple slice_same_proc = None
    cdef int i
    cdef Subdomain in_subdomain, out_subdomain

    in_subdomain = domain_in.my_subdomain
    for i in range(domain_in.comm.Get_size()):
        # get the domain of the output
        out_subdomain = domain_out.all_subdomains[i]
        # check if there is an overlap between the two domains
        if not in_subdomain.has_overlap(out_subdomain):
            continue
        # get the overlap slice
        overlap_slice = in_subdomain.get_overlap_slice(out_subdomain)
        if i == domain_in.rank:
            slice_same_proc = overlap_slice
        else:
            overlap_slices.append(overlap_slice)
            processors.append(i)
    cdef dict overlap_info
    overlap_info = dict(overlap_slices=overlap_slices,
                        processors=processors,
                        slice_same_proc=slice_same_proc)
    return overlap_info

cdef object transform(DomainDecomposition domain_in, 
                      DomainDecomposition domain_out, 
                      bint same_domain,
                      dict overlap_info_in,
                      dict overlap_info_out,
                      object arr_in,
                      object arr_out = None):
    """
    Transform data from an array in the input domain to an array in the output
    domain. This function is called by the transformer class.
    
    Description
    -----------
    
    Parameters
    ----------
    `domain_in` : `DomainDecomposition`
        The input domain.
    `domain_out` : `DomainDecomposition`
        The output domain.
    `same_domain` : `bool`
        A boolean that indicates if the input and output domains are the same.
    `overlap_info_in` : `dict`
        The overlap information of the input domain. This information can be
        obtained by calling the get_overlap_info method.
    `overlap_info_out` : `dict`
        The overlap information of the output domain.
    `arr_in` : `np.ndarray` or `cupy.ndarray`
        The input array.
    `arr_out` : `np.ndarray` or `cupy.ndarray`
        The output array. If None, a new array will be created.
    
    Returns
    -------
    `np.ndarray` or `cupy.ndarray`
        Output array after transformation. If `arr_out` is not None, the output
        array will be the same as `arr_out`.
    
    Examples
    --------
    See the transformer class for an example.
    """
    if same_domain:
        if arr_out is None:
            return arr_in
        else:
            arr_out[:] = arr_in
            return arr_out
        
    # first synchronize the gpu and cpu
    domain_in.sync_with_device()

    # create new array
    if arr_out is None:
        arr_out = config.ncp.zeros(
            domain_out.my_subdomain.shape, dtype=arr_in.dtype)

    # send the data
    cdef list destinations = overlap_info_in['processors']
    cdef list send_slices = overlap_info_in['overlap_slices']
    cdef list bufs = [config.ncp.ascontiguousarray(arr_in[s]) for s in send_slices]
    domain_in.sync_with_device()
    cdef list reqs
    reqs = [domain_in.comm.Isend(buf, dest=dest, tag=0) 
            for buf, dest in zip(bufs, destinations)]

    # receive the data
    cdef list sources = overlap_info_out['processors']
    cdef list recv_slices = overlap_info_out['overlap_slices']
    bufs = [config.ncp.empty(arr_out[s].shape, dtype=arr_out.dtype) 
            for s in recv_slices]

    reqs += [domain_in.comm.Irecv(buf, source=source, tag=0)
            for buf, source in zip(bufs, sources)]

    # wait for all non-blocking operations to complete
    MPI.Request.Waitall(reqs)

    # copy the received data to the new array
    cdef slice recv_slice
    cdef object buf
    for recv_slice, buf in zip(recv_slices, bufs):
        arr_out[recv_slice] = buf

    # copy the matching slice
    cdef tuple send_same_proc = overlap_info_in['slice_same_proc']
    cdef tuple recv_same_proc = overlap_info_out['slice_same_proc']
    if send_same_proc is not None:
        arr_out[recv_same_proc] = arr_in[send_same_proc]

    # synchronize halo regions
    domain_out.sync(arr_out)

    return arr_out

cdef class Transformer:
    def __init__(self,
                 DomainDecomposition domain_in,
                 DomainDecomposition domain_out):
        # First check if the domains are the same
        cdef bint same_domain = True
        cdef int my_proc, other_proc
        for my_proc, other_proc in zip(domain_in.n_procs, domain_out.n_procs):
            if my_proc != other_proc:
                same_domain = False
                break
        if domain_in.halo != domain_out.halo:
            same_domain = False

        # Get the overlap information
        cdef dict overlap_info_in = get_overlap_info(domain_in, domain_out)
        cdef dict overlap_info_out = get_overlap_info(domain_out, domain_in)

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------
        self.domain_in = domain_in
        self.domain_out = domain_out

        # private attributes
        self._same_domain = same_domain
        self._overlap_info_in = overlap_info_in
        self._overlap_info_out = overlap_info_out

    cpdef object forward(self, arr_in, arr_out = None):
        return transform(
            self.domain_in, self.domain_out, self._same_domain,
            self._overlap_info_in, self._overlap_info_out, arr_in, arr_out)

    cpdef object backward(self, arr_in, arr_out = None):
        return transform(
            self.domain_out, self.domain_in, self._same_domain,
            self._overlap_info_out, self._overlap_info_in, arr_in, arr_out)

    # ================================================================
    #  Properties
    # ================================================================

    property domain_in:
        def __get__(self):
            return self._domain_in

    property domain_out:
        def __get__(self):
            return self._domain_out