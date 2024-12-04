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
    from .domain_decomposition import DomainDecomposition, get_default_domain_decomposition
    from .single_decomposition import SingleDecomposition
    from .jax_decomposition import JaxDecomposition

# ================================================================
#  Setup lazy loading
# ================================================================
all_modules_by_origin = { }

dom_path = "fridom.framework.domain_decomposition"
all_imports_by_origin = {
    f"{dom_path}.domain_decomposition": ["DomainDecomposition", 
                                         "get_default_domain_decomposition"], 
    f"{dom_path}.single_decomposition": ["SingleDecomposition"],
    f"{dom_path}.jax_decomposition": ["JaxDecomposition"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)