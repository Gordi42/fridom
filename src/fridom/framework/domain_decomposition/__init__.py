"""
Domain Decomposition.

=====================

Decomposing the domain into subdomains for parallel computation.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from .domain_decomposition import (
        DomainDecomposition,
        get_default_domain_decomposition,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
all_modules_by_origin = { }

dom_path = "fridom.framework.domain_decomposition"
all_imports_by_origin = {
    f"{dom_path}.domain_decomposition": ["DomainDecomposition",
                                         "get_default_domain_decomposition"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
