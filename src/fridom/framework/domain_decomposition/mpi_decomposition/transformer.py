# Import internal modules
from fridom.framework import config, utils
# Import external modules
from typing import TYPE_CHECKING
import numpy as np
from functools import partial
# Import type information
if TYPE_CHECKING:
    from .domain_decomposition import DomainDecomposition


@utils.jaxify
class OverlapInfo:
    """
    Get information about which processors of the output domain overlaps with
    the subdomain of this processor in the input domain.
    
    Description
    -----------
    In case we want to send data from one domain to another, we need to know
    which processors of the output domain overlap with the subdomain of this
    processor in the input domain. This class provides this information.
    
    Parameters
    ----------
    `domain_in` : `DomainDecomposition`
        The input domain.
    `domain_out` : `DomainDecomposition`
        The output domain.

    Examples
    --------

    .. code-block:: python

        # create two domains
        domain1 = DomainDecomposition(n_global=tuple([128]*3), shared_axes=[0])
        domain2 = DomainDecomposition(n_global=tuple([128]*3), shared_axes=[1])

        # get the overlap information
        overlap_info = OverlapInfo(domain1, domain2)

        print(f"The slice {OverlapInfo.overlap_slices[0]} of domain1 \
        shares an overlap with processor {OverlapInfo.processors[0]} \
        of the domain2.")
    """
    def __init__(self, 
                 domain_in: 'DomainDecomposition', 
                 domain_out: 'DomainDecomposition') -> None:
        overlap_slices = []
        processors = []
        slice_same_proc = None
        in_subdomain = domain_in.my_subdomain
        if utils.MPI_AVAILABLE:
            n_procs = domain_in.comm.Get_size()
        else:
            n_procs = 1
        for i in range(n_procs):
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

        # ----------------------------------------------------------------
        #  Set the attributes
        # ----------------------------------------------------------------
        self.overlap_slices = overlap_slices
        self.processors = processors
        self.slice_same_proc = slice_same_proc
        return


@partial(utils.jaxjit, static_argnames=["same_domain"])
def transform(domain_in: 'DomainDecomposition', 
              domain_out: 'DomainDecomposition', 
              same_domain: bool,
              overlap_info_in: OverlapInfo,
              overlap_info_out: OverlapInfo,
              arr_in: np.ndarray) -> np.ndarray:
    """
    Transform data from an array in the input domain to an array in the output
    domain. This function is called by the transformer class.
    
    Parameters
    ----------
    `domain_in` : `DomainDecomposition`
        The input domain.
    `domain_out` : `DomainDecomposition`
        The output domain.
    `same_domain` : `bool`
        A boolean that indicates if the input and output domains are the same.
    `overlap_info_in` : `OverlapInfo`
        The overlap information of the input domain. 
    `overlap_info_out` : `OverlapInfo`
        The overlap information of the output domain.
    `arr_in` : `np.ndarray`
        The input array.
    
    Returns
    -------
    `np.ndarray`
        Output array after transformation. If `arr_out` is not None, the output
        array will be the same as `arr_out`.
    
    Examples
    --------
    See the transformer class for an example.
    """
    if same_domain:
        return arr_in
        
    # first synchronize the gpu and cpu
    domain_in.sync_with_device()

    # create new array
    arr_out = config.ncp.zeros(
        domain_out.my_subdomain.shape, dtype=arr_in.dtype)

    # send the data
    destinations = overlap_info_in.processors
    send_slices = overlap_info_in.overlap_slices
    bufs = [config.ncp.ascontiguousarray(arr_in[s]) for s in send_slices]
    domain_in.sync_with_device()
    reqs = [domain_in.comm.Isend(buf, dest=dest, tag=0) 
            for buf, dest in zip(bufs, destinations)]

    # receive the data
    sources = overlap_info_out.processors
    recv_slices = overlap_info_out.overlap_slices
    bufs = [config.ncp.empty(arr_out[s].shape, dtype=arr_out.dtype) 
            for s in recv_slices]

    reqs += [domain_in.comm.Irecv(buf, source=source, tag=0)
            for buf, source in zip(bufs, sources)]

    # wait for all non-blocking operations to complete
    if utils.MPI_AVAILABLE:
        utils.MPI.Request.Waitall(reqs)

    # copy the received data to the new array
    for recv_slice, buf in zip(recv_slices, bufs):
        arr_out = utils.modify_array(arr_out, recv_slice, buf)

    # copy the matching slice
    send_same_proc = overlap_info_in.slice_same_proc
    recv_same_proc = overlap_info_out.slice_same_proc
    if send_same_proc is not None:
        arr_out = utils.modify_array(
            arr_out, recv_same_proc, arr_in[send_same_proc])

    # synchronize halo regions
    arr_out = domain_out.sync(arr_out)

    return arr_out


@partial(utils.jaxify, dynamic=("_domain_in", "_domain_out", 
                                "_overlap_info_in", "_overlap_info_out"))
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
    
    Examples
    --------
    .. code-block:: python

        from fridom.framework import config
        from fridom.framework.domain_decomposition import DomainDecomposition, Transformer
        # create two domains where one shares the x-axis and the other the y-axis
        domain_x = DomainDecomposition(n_global=(128, 128), shared_axes=[0])
        domain_y = DomainDecomposition(n_global=(128, 128), shared_axes=[1])
        
        # create a random array on the local domain
        u = config.ncp.random.rand(*domain_x.my_subdomain.shape)
        domain_x.sync(u)
        
        # construct transformers between the domains
        transformer = Transformer(domain_x, domain_y)
        
        # transform the array from domain_x to domain_y
        v = transformer.forward(u)
        assert v.shape == domain_y.my_subdomain.shape
        
        # transform the array back from domain_y to domain_x
        w = transformer.backward(v)
        assert config.ncp.allclose(u, w)
    """
    def __init__(self, 
                 domain_in: 'DomainDecomposition', 
                 domain_out: 'DomainDecomposition') -> None:
        # First check if the domains are the same
        same_domain = True
        for my_proc, other_proc in zip(domain_in.n_procs, domain_out.n_procs):
            if my_proc != other_proc:
                same_domain = False
                break
        if domain_in.halo != domain_out.halo:
            same_domain = False

        # Get the overlap information
        overlap_info_in = OverlapInfo(domain_in, domain_out)
        overlap_info_out = OverlapInfo(domain_out, domain_in)

        # --------------------------------------------------------------
        #  Set the attributes
        # --------------------------------------------------------------
        # readable public attributes
        self._domain_in = domain_in
        self._domain_out = domain_out

        # private attributes for internal use
        self._same_domain = same_domain
        self._overlap_info_in = overlap_info_in
        self._overlap_info_out = overlap_info_out
        return

    @utils.jaxjit
    def forward(self, arr: np.ndarray) -> np.ndarray:
        """
        Transform an array from the input domain to the output domain.
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The array to be transformed (lives in the input domain).
        
        Returns
        -------
        `np.ndarray`
            The transformed array. 
        """
        return transform(
            self._domain_in, self._domain_out, self._same_domain,
            self._overlap_info_in, self._overlap_info_out, arr)
    
    @utils.jaxjit
    def backward(self, arr: np.ndarray) -> np.ndarray:
        """
        Transform an array from the output domain to the input domain.
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The array to be transformed (lives in the output domain).
        
        Returns
        -------
        `np.ndarray`
            The transformed array.
        """
        return transform(
            self._domain_out, self._domain_in, self._same_domain,
            self._overlap_info_out, self._overlap_info_in, arr)

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def domain_in(self) -> 'DomainDecomposition':
        """The domain decomposition of the input domain."""
        return self._domain_in

    @property
    def domain_out(self) -> 'DomainDecomposition':
        """The domain decomposition of the output domain."""
        return self._domain_out
