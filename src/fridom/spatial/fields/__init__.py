"""
Field types of the grid cluster.

Description
-----------
Owning class doc: ``design/specs/grid/classes/fields.md``.
Waves 2/3 re-export ``FieldMetadata``, ``ScalarField``,
``VectorField``, and ``TensorField`` here.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import (
        metadata,
        scalar_field,
        storage,
        tensor_field,
        vector_field,
    )

    # import all classes
    from .metadata import FieldMetadata
    from .scalar_field import ScalarField
    from .vector_field import VectorField

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.spatial.fields"

all_modules_by_origin = {
    base: [
        "metadata",
        "scalar_field",
        "storage",
        "vector_field",
        "tensor_field",
    ],
}

all_imports_by_origin = {
    f"{base}.metadata": ["FieldMetadata"],
    f"{base}.scalar_field": ["ScalarField"],
    f"{base}.vector_field": ["VectorField"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
