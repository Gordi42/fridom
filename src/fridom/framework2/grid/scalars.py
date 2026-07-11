"""
Static scalar markers of the grid cluster.

Description
-----------
Owning class doc: ``design/specs/grid/classes/meshes.md`` (static
markers). ``Scalars`` is the field of scalars (Körper) marker with the
module-level aliases ``Real`` / ``Complex`` (re-exported at top level
as ``fr.Real`` / ``fr.Complex``); ``Variance`` is the designed-for
component-variance marker for vector components on metric meshes.
"""
# Wave 0: Scalars, Real, Complex, Variance
from __future__ import annotations

from enum import Enum, auto


class Scalars(Enum):

    """
    The field of scalars (Körper) a function space is over.

    Description
    -----------
    The marker denotes the Körper of the *represented function*, not
    the storage dtype: an ``fr.Real`` Fourier space stores a complex
    Hermitian half-spectrum. It participates in the interned identity
    of every space, so real and complex variants are distinct
    dispatch keys.
    """

    REAL = auto()
    COMPLEX = auto()


class Variance(Enum):

    """
    Component variance of a vector component on a metric mesh.

    Description
    -----------
    Covariant and contravariant components live on distinct spaces so
    the strict algebra catches variance mixing (designed-for; the
    default ``None`` on spaces means scalar/no-variance).
    """

    COVARIANT = auto()
    CONTRAVARIANT = auto()


# module-level aliases, re-exported at top level as fr.Real / fr.Complex
Real: Scalars = Scalars.REAL
Complex: Scalars = Scalars.COMPLEX
