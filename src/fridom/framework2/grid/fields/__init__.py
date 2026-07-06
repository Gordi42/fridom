"""
Field types of the grid cluster.

Description
-----------
Owning class doc: ``notes/framework2/classes/fields.md``.
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
        tensor_field,
        vector_field,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.framework2.grid.fields"

all_modules_by_origin = {
    base: [
        "metadata",
        "scalar_field",
        "vector_field",
        "tensor_field",
    ],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
