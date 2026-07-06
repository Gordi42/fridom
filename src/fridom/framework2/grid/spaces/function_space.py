"""
The ``FunctionSpace`` ABC: interned static space descriptors.

Description
-----------
Owning class doc: ``notes/framework2/classes/spaces.md``. A
``FunctionSpace`` describes how a continuous field is represented on
one mesh. Spaces are static, hashable, identity-compared, and
interned in their owning mesh's registry; construction is guarded by
a private factory token so the only construction path is the mesh
factory surface (``mx.center``, ``mx.fourier(origin=...)``, ...).
"""
# Wave 1: FunctionSpace (ABC)
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

from fridom.framework2.grid.scalars import Scalars

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.bc import BCStructure
    from fridom.framework2.grid.decomposition.layout import Layout
    from fridom.framework2.grid.meshes.mesh import Mesh
    from fridom.framework2.grid.scalars import Variance
    from fridom.framework2.grid.spaces.tensor_product import (
        TensorProductSpace,
    )

# The private factory token: mesh factories pass it to space
# constructors; direct construction without it raises, because it
# would silently void the identity-equality guarantee of the strict
# algebra (cluster rules in notes/framework2/classes/meshes.md).
_FACTORY_TOKEN: object = object()

# Sentinel for "keep the current layout" in variant lookups.
_KEEP: object = object()


def space_key(
    cls: type[FunctionSpace],
    *descriptors: object,
    layout: Layout | None = None,
) -> tuple:
    """
    Build the mesh-registry interning key of a space.

    Description
    -----------
    The key is the tuple of the space's defining static descriptors
    — ``(type, node set / basis / origin, bc, scalars)`` per the
    class doc — plus the layout *when set*, so bare (layout-free)
    keys are unchanged by the layout protocol. Mesh identity is not
    part of the key: the registry is mesh-owned.

    Parameters
    ----------
    cls : type[FunctionSpace]
        The concrete space class.
    *descriptors : object
        The remaining value-hashable defining descriptors.
    layout : Layout | None, optional
        The device layout; appended to the key only when not None
        (default: None).

    Returns
    -------
    tuple
        The value-hashable interning key.
    """
    key = (cls, *descriptors)
    if layout is None:
        return key
    return (*key, layout)


class FunctionSpace(ABC):

    """
    How a continuous field is represented on one mesh.

    Description
    -----------
    Static, hashable, interned descriptor of one discrete
    representation on one mesh. Spaces hold no jax arrays and no
    grid reference; they enter field pytrees as static aux data and
    serve as jit-cache and dispatch keys. Interning turns value
    equality into identity, so the strict-algebra check is
    ``a is b``.

    Parameters
    ----------
    mesh : Mesh
        The owning mesh factor.
    scalars : Scalars
        The field of scalars (Körper) the space is defined over.
    bc : BCStructure
        The homogeneous BC structure baked into the space.
    layout : Layout | None, optional
        The negotiated device layout; None for a bare space
        (default: None).
    _token : object
        The owning mesh's private factory token; raises unless it is
        the module-private token (construction only through mesh
        factories).
    """

    # subclasses may override the repr label (e.g. "Fourier" for
    # FourierSpace); None means the class name
    _repr_label: ClassVar[str | None] = None

    def __init__(self, mesh: Mesh, scalars: Scalars, bc: BCStructure,
                 *, layout: Layout | None = None,
                 _token: object = None) -> None:
        """Guarded constructor; see the class docstring."""
        if _token is not _FACTORY_TOKEN:
            raise TypeError(
                "function spaces are constructed through their mesh's "
                "factory attributes (mx.center, mx.fourier(origin=...),"
                " ...), never directly")
        self._mesh: Mesh = mesh
        self._scalars: Scalars = scalars
        self._bc: BCStructure = bc
        self._layout: Layout | None = layout

    # ================================================================
    #  Identity (explicit, see cluster rules)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity: return ``self is other``.

        Explicit so fridom's structural-equality walk falls through
        to plain ``==``.
        """
        return self is other

    def __hash__(self) -> int:
        """Identity hash, matching ``__eq__``."""
        return id(self)

    # ================================================================
    #  Defining attributes
    # ================================================================
    @property
    def mesh(self) -> Mesh:
        """The owning mesh factor.

        A static back-reference; spaces hold no *grid* reference.
        """
        return self._mesh

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """True DOF count per storage axis.

        One entry for 1D factors, more for 2D structured factors.
        """
        ...

    @property
    def scalars(self) -> Scalars:
        """The Körper the space is defined over."""
        return self._scalars

    @property
    def bc(self) -> BCStructure:
        """Homogeneous BC structure baked into the space.

        The free structure for BC-free spaces.
        """
        return self._bc

    @property
    def variance(self) -> Variance | None:
        """Component variance on metric meshes (designed-for).

        None means scalar/no-variance; iteration 1 never sets it.
        """
        return None

    # ================================================================
    #  Product protocol (shared with TensorProductSpace)
    # ================================================================
    @property
    def factors(self) -> tuple[FunctionSpace, ...]:
        """The flat factor tuple of a lone factor.

        Bare-normalized, like the product's stored factors.
        """
        return (self.bare,)

    @property
    def names(self) -> tuple[str, ...]:
        """The owning mesh's coordinate names."""
        return self._mesh.names

    def factor(self, name: str) -> FunctionSpace:
        """
        Return the (bare) factor contributing coordinate ``name``.

        Parameters
        ----------
        name : str
            One of the owning mesh's coordinate names.

        Returns
        -------
        FunctionSpace
            The bare variant of this space (dispatch keys never see
            layouts).
        """
        if name in self._mesh.names:
            return self.bare
        raise KeyError(
            f"no factor contributes coordinate {name!r}; this space's "
            f"names are {self._mesh.names}")

    # ================================================================
    #  Layout protocol (section 5.1)
    # ================================================================
    @property
    def layout(self) -> Layout | None:
        """Negotiated device layout, or None for a bare space."""
        return self._layout

    @property
    def bare(self) -> FunctionSpace:
        """The layout-free interned variant (self if bare)."""
        return self._variant(layout=None)

    def with_layout(self, layout: Layout | None) -> Self:
        """
        Return the interned variant carrying ``layout``.

        Description
        -----------
        Laid-out variants are grid-minted after negotiation; mesh
        factories mint bare spaces only. The layout enters the
        interning key only when set, so bare keys are unchanged.

        Parameters
        ----------
        layout : Layout | None
            The device layout; None returns the bare variant.

        Returns
        -------
        Self
            The interned variant with the requested layout.
        """
        return self._variant(layout=layout)

    # ================================================================
    #  Scalar variants (interning lookups, not relational properties)
    # ================================================================
    def as_complex(self) -> Self:
        """Return the interned ``fr.Complex`` variant of this space.

        Idempotent on complex spaces.
        """
        return self._variant(scalars=Scalars.COMPLEX)

    def as_real(self) -> Self:
        """Return the interned ``fr.Real`` variant of this space.

        The codomain of ``f.real``; idempotent on real spaces.
        """
        return self._variant(scalars=Scalars.REAL)

    # ================================================================
    #  Tensor product
    # ================================================================
    def __mul__(
        self, other: FunctionSpace | TensorProductSpace,
    ) -> TensorProductSpace:
        """Tensor product (flat, associative)."""
        from fridom.framework2.grid.spaces.tensor_product import (  # noqa: PLC0415
            TensorProductSpace,
        )
        return TensorProductSpace.of(self, other)

    # ================================================================
    #  Variant interning machinery (internal)
    # ================================================================
    def _variant(self, *, scalars: Scalars | None = None,
                 layout: object = _KEEP) -> Self:
        """
        Return the interned variant with the given overrides.

        Description
        -----------
        Routes back through the owning mesh's registry, so scalar
        and layout variants are interned exactly like factory-made
        spaces (value-equal requests return the identical object).

        Parameters
        ----------
        scalars : Scalars | None, optional
            The Körper of the variant; None keeps the current one
            (default: None).
        layout : object, optional
            The layout of the variant; the module sentinel keeps the
            current one (default: keep).

        Returns
        -------
        Self
            The interned variant (self when nothing changes).
        """
        scalars = self._scalars if scalars is None else scalars
        layout = self._layout if layout is _KEEP else layout
        if scalars is self._scalars and layout == self._layout:
            return self
        key = self._variant_key(scalars, layout)
        return self._mesh._intern(  # noqa: SLF001 — interning seam
            key, lambda: self._construct(scalars, layout))

    def _variant_key(self, scalars: Scalars,
                     layout: Layout | None) -> tuple:
        """Return the interning key of a (scalars, layout) variant.

        Must match the owning mesh factory's key format.
        """
        return space_key(type(self), self._bc, scalars, layout=layout)

    def _construct(self, scalars: Scalars,
                   layout: Layout | None) -> Self:
        """Build (not intern) the (scalars, layout) variant."""
        return type(self)(self._mesh, scalars, self._bc,
                          layout=layout, _token=_FACTORY_TOKEN)

    # ================================================================
    #  Repr
    # ================================================================
    def _repr_details(self) -> tuple[str, ...]:
        """Extra repr entries of the concrete space family."""
        return ()

    def __repr__(self) -> str:
        """E.g. ``Center(x)``, ``Fourier(x, origin=Center)``."""
        label = self._repr_label or type(self).__name__
        parts = [self._mesh._label,  # noqa: SLF001 — repr seam
                 *self._repr_details()]
        if not self._bc.is_free:
            bc_names = ", ".join(
                component.name for component in self._bc.components)
            parts.append(f"bc=({bc_names})")
        if self._scalars is Scalars.COMPLEX:
            parts.append("complex")
        if self._layout is not None:
            parts.append(f"layout={self._layout!r}")
        return f"{label}({', '.join(parts)})"
