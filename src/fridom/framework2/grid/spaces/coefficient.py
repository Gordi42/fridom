"""
Coefficient spaces: modal coefficients relative to a basis.

Description
-----------
Owning class doc: ``notes/framework2/classes/spaces.md``
(``CoefficientSpace`` and friends). A coefficient space is defined
by (basis, origin): the origin space is constitutive, fixing the
inverse transform and the shape. The class names carry a ``Space``
suffix to avoid colliding with the transform operators; users never
type them — the factory spellings ``mx.fourier(origin=...)`` are the
API, and reprs print the concept-note form ``Fourier(x,
origin=Center)``.
"""
# Wave 1: CoefficientSpace (ABC), FourierSpace, SineSpace,
#    CosineSpace, ChebyshevSpace
from __future__ import annotations

from typing import TYPE_CHECKING, Self

from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.function_space import (
    _FACTORY_TOKEN,
    FunctionSpace,
    space_key,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.decomposition.layout import Layout
    from fridom.framework2.grid.meshes.mesh import Mesh


class CoefficientSpace(FunctionSpace):

    """
    Modal coefficients of a basis, tied to an origin space.

    Description
    -----------
    The origin is constitutive: coefficient spaces of distinct
    origins are distinct interned spaces, related only by exact
    operators. Scalars follow the origin (the Körper of the
    *represented* function), and the BC structure is the origin's
    (``self.bc is self.origin.bc``).

    Parameters
    ----------
    mesh : Mesh
        The owning mesh factor.
    origin : FunctionSpace
        The origin space (constitutive; always explicit).
    layout : Layout | None, optional
        The negotiated device layout (default: None).
    _token : object
        The private factory token (construction only through mesh
        factories).
    """

    def __init__(self, mesh: Mesh, origin: FunctionSpace,
                 *, layout: Layout | None = None,
                 _token: object = None) -> None:
        """Guarded constructor; see the class docstring."""
        super().__init__(mesh, origin.scalars, origin.bc,
                         layout=layout, _token=_token)
        self._origin: FunctionSpace = origin

    @property
    def origin(self) -> FunctionSpace:
        """The origin space (constitutive of this space).

        It fixes the inverse transform and the shape.
        """
        return self._origin

    # ------------------------------------------------------------
    #  Variant interning (scalar variants route through the origin)
    # ------------------------------------------------------------
    def _origin_with(self, scalars: Scalars) -> FunctionSpace:
        """Return the interned origin variant of the given Körper."""
        if scalars is Scalars.COMPLEX:
            return self._origin.as_complex()
        return self._origin.as_real()

    def _variant_key(self, scalars: Scalars,
                     layout: Layout | None) -> tuple:
        """Return the interning key of a (scalars, layout) variant.

        The key is (type, origin identity), plus the layout when
        set; the scalar variant swaps the origin.
        """
        return space_key(type(self), self._origin_with(scalars),
                         layout=layout)

    def _construct(self, scalars: Scalars,
                   layout: Layout | None) -> Self:
        """Build (not intern) the variant on the swapped origin.

        ``as_complex`` on a coefficient space is never a dtype flag
        flip — it changes the origin and hence generally the shape.
        """
        return type(self)(self._mesh, self._origin_with(scalars),
                          layout=layout, _token=_FACTORY_TOKEN)

    def _repr_details(self) -> tuple[str, ...]:
        """Append the origin, e.g. ``origin=Center``."""
        origin_cls = type(self._origin)
        label = origin_cls.__name__
        return (f"origin={label}",)


class FourierSpace(CoefficientSpace):

    """
    Fourier coefficients of a periodic origin.

    Description
    -----------
    A real-origin Fourier space stores the complex Hermitian half
    spectrum (rfft layout): shape (n // 2 + 1,). A complex-origin
    one stores the full spectrum: shape (n,).
    """

    _repr_label = "Fourier"

    @property
    def shape(self) -> tuple[int, ...]:
        """DOF count of the spectrum.

        (n // 2 + 1,) for a real origin (Hermitian half spectrum),
        (n,) for a complex one (full spectrum).
        """
        n = self._origin.shape[0]
        if self._scalars is Scalars.REAL:
            return (n // 2 + 1,)
        return (n,)


class SineSpace(CoefficientSpace):

    """
    DST coefficients of a Dirichlet-structured bounded origin.

    Description
    -----------
    One coefficient per origin DOF: DST-I of Dirichlet ``Inner``
    has shape (n - 1,), DST-II of Dirichlet ``Center`` has (n,).
    """

    _repr_label = "Sine"

    @property
    def shape(self) -> tuple[int, ...]:
        """One mode per origin DOF."""
        return self._origin.shape


class CosineSpace(CoefficientSpace):

    """
    DCT coefficients of a Neumann-structured bounded origin.

    Description
    -----------
    One coefficient per origin DOF: DCT-II of Neumann ``Center``
    has shape (n,), DCT-I of Neumann ``Outer`` has (n + 1,) —
    Neumann never reduces the origin shape (owner decision
    2026-07-07, ``NodalSpace`` shape note).
    """

    _repr_label = "Cosine"

    @property
    def shape(self) -> tuple[int, ...]:
        """One mode per origin DOF."""
        return self._origin.shape


class ChebyshevSpace(CoefficientSpace):

    """Chebyshev coefficients of a Gauss-Lobatto origin."""

    _repr_label = "Chebyshev"

    @property
    def shape(self) -> tuple[int, ...]:
        """One mode per Lobatto point: (n + 1,)."""
        return self._origin.shape
