"""
Domain Decomposition
====================
Decomposing the domain into subdomains for parallel computation.
"""
from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from .domain_decomposition import DomainDecomposition
    from .subdomain import Subdomain
    from .transformer import Transformer
    from .parallel_fft import ParallelFFT

# ================================================================
#  Setup lazy loading
# ================================================================
all_modules_by_origin = { }

dom_path = "fridom.framework.domain_decomposition"
all_imports_by_origin = { 
    f"{dom_path}.domain_decomposition": ["DomainDecomposition"], 
    f"{dom_path}.subdomain": ["Subdomain"],
    f"{dom_path}.transformer": ["Transformer"],
    f"{dom_path}.parallel_fft": ["ParallelFFT"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)