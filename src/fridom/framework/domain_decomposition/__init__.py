"""
Domain Decomposition
===
Decomposing the domain into subdomains for parallel computation.

Classes
-------
`DomainDecomposition`
    The main class for domain decomposition.
`Subdomain`
    Store the information (position, size, etc) of a subdomain in the 
    global domain
`Transformer`
    To transform arrays that are distributed on one domain to another domain.
`ParallelFFT`
    Perform fourier transform on the distributed arrays.
"""

from .domain_decomposition import DomainDecomposition
from .subdomain import Subdomain
from .transformer import Transformer
from .parallel_fft import ParallelFFT