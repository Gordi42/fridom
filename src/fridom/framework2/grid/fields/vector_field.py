"""
``VectorField``: a named collection of scalar fields (no metric).

Description
-----------
Owning class doc: ``notes/framework2/classes/fields.md``. Thin means
thin (section 2.4): no metric, no axis semantics, no inner product —
the coordinate association and variance of a component are carried by
its *space*. Consequently there is no ``dot`` and no ``div`` method
(divergence is an operator, ``fr.operators.Divergence()``).

Pytree contract: the component mapping is flattened keyed, in
component-declaration order — the fields tuple is the dynamic leaf
container, the name tuple lives in the static treedef — so renaming
or re-keying a component changes the treedef (retrace), while
componentwise arithmetic *preserves* per-component metadata (names
are structural, the scan-stability rule):
``tree_structure(z + dt * dz) == tree_structure(z)``.
"""
# Wave 3: VectorField
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

import jax.numpy as jnp

from fridom.framework.utils import jaxify
from fridom.framework2.grid.errors import GridMismatchError
from fridom.framework2.grid.fields.scalar_field import ScalarField

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Iterator

    import jax

    from fridom.framework2.grid.grid import Grid

# Python scalars entering componentwise arithmetic
_SCALAR_TYPES = int | float | complex


@partial(jaxify, dynamic=("_fields",))
class VectorField:

    """
    Thin container of ScalarFields; carries no metric.

    Description
    -----------
    Component spaces are unconstrained (components live on different,
    related spaces — the C-grid staggering); all components must
    share one grid. Iterable input takes names from each field's
    ``metadata.name``; mapping input takes the mapping keys.

    Parameters
    ----------
    components : Mapping[str, ScalarField] | Iterable[ScalarField]
        A name -> field mapping (in declaration order) or an
        iterable of named fields.
    """

    def __init__(
        self,
        components: (Mapping[str, ScalarField]
                     | Iterable[ScalarField]),
    ) -> None:
        """Build the container; see the class docstring."""
        if isinstance(components, Mapping):
            names = tuple(components.keys())
            fields = tuple(components.values())
        else:
            fields = tuple(components)
            names = tuple(f.metadata.name for f in fields)
        if not fields:
            raise ValueError(
                "a VectorField needs at least one component")
        duplicates = tuple(
            name for i, name in enumerate(names)
            if name in names[:i])
        if duplicates:
            if "unnamed" in duplicates:
                raise ValueError(
                    "several components carry the default name "
                    "'unnamed'; name the fields before collecting "
                    "them (FieldMetadata.create(name=...) / "
                    "f.with_metadata(name=...)) or pass a "
                    "name -> field mapping")
            raise ValueError(
                f"duplicate component names {duplicates}; component "
                "names key the container and must be unique")
        grid = fields[0].grid
        for name, field in zip(names, fields, strict=True):
            if field.grid is not grid:
                raise GridMismatchError(
                    f"component {name!r} lives on a different grid; "
                    "all components of a VectorField must be "
                    "created on the identical grid object",
                    left=grid, right=field.grid,
                    operation="VectorField")
        self._names: tuple[str, ...] = names
        self._fields: tuple[ScalarField, ...] = fields

    # ================================================================
    #  Identity (matches ScalarField: no value equality)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity: return ``self is other`` (pytree friendly)."""
        return self is other

    def __hash__(self) -> int:
        """Identity hash, matching ``__eq__``."""
        return id(self)

    def __bool__(self) -> bool:
        """Raise TypeError: vector fields have no truth value."""
        raise TypeError(
            "vector fields have no truth value; compare component "
            "arrays explicitly via vec[name].data")

    # ================================================================
    #  Component access
    # ================================================================
    @property
    def components(self) -> Mapping[str, ScalarField]:
        """Read-only name -> field view, in declaration order."""
        return MappingProxyType(
            dict(zip(self._names, self._fields, strict=True)))

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in declaration order."""
        return self._names

    @property
    def grid(self) -> Grid:
        """The common grid of all components."""
        return self._fields[0].grid

    def __getitem__(self, key: str | int) -> ScalarField:
        """
        Return a component by name or positional index.

        Parameters
        ----------
        key : str | int
            The component name or its declaration-order index.

        Returns
        -------
        ScalarField
            The component field.
        """
        if isinstance(key, str):
            try:
                index = self._names.index(key)
            except ValueError:
                raise KeyError(
                    f"no component named {key!r}; components are "
                    f"{self._names}") from None
            return self._fields[index]
        if isinstance(key, int) and not isinstance(key, bool):
            return self._fields[key]
        raise TypeError(
            f"components are indexed by name or position, got "
            f"{key!r}")

    def __iter__(self) -> Iterator[ScalarField]:
        """Iterate over component fields in declaration order."""
        return iter(self._fields)

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self._fields)

    def __contains__(self, name: str) -> bool:
        """Whether a component of that name exists."""
        return name in self._names

    # ================================================================
    #  Functional surface
    # ================================================================
    def map(
        self,
        fn: Callable[[ScalarField], ScalarField],
    ) -> Self:
        """
        Apply ``fn`` to each component on its own space (2.4).

        Description
        -----------
        ``fn`` receives each component field and must return a
        ``ScalarField``; the result keeps names and order. ``map``
        never inspects spaces — per-component space changes (e.g.
        transforms) land in the returned collection.

        Parameters
        ----------
        fn : Callable[[ScalarField], ScalarField]
            The per-component function.

        Returns
        -------
        Self
            The mapped collection.
        """
        mapped = tuple(fn(field) for field in self._fields)
        return type(self)(
            dict(zip(self._names, mapped, strict=True)))

    def replace(self, **components: ScalarField) -> Self:
        """
        Return the collection with named components updated.

        Parameters
        ----------
        **components : ScalarField
            New fields for existing component names.

        Returns
        -------
        Self
            The updated collection; ``self`` is unchanged.
        """
        unknown = tuple(name for name in components
                        if name not in self._names)
        if unknown:
            raise KeyError(
                f"no components named {unknown}; components are "
                f"{self._names}")
        updated = {
            name: components.get(name, field)
            for name, field in zip(self._names, self._fields,
                                   strict=True)}
        return type(self)(updated)

    # ================================================================
    #  Arithmetic (componentwise delegation; join rule per
    #  component; metadata preserved — the scan-stability rule)
    # ================================================================
    def _componentwise(
        self,
        other: object,
        fn: Callable[[ScalarField, object], ScalarField],
        operation: str,
    ) -> Self:
        """Delegate ``fn`` per component, preserving metadata."""
        if isinstance(other, VectorField):
            if self._names != other.component_names:
                raise ValueError(
                    f"componentwise {operation!r} needs identical "
                    f"component-name tuples, got {self._names} vs "
                    f"{other.component_names}")
            pairs = zip(self._fields, tuple(other), strict=True)
            results = tuple(
                _keep_metadata(fn(a, b), a) for a, b in pairs)
        else:
            results = tuple(
                _keep_metadata(fn(a, other), a)
                for a in self._fields)
        return type(self)(
            dict(zip(self._names, results, strict=True)))

    def __add__(self, other: Self | complex) -> Self:
        """Componentwise sum (join rule per component)."""
        if isinstance(other, VectorField | _SCALAR_TYPES):
            return self._componentwise(
                other, lambda a, b: a + b, "+")
        return NotImplemented

    def __radd__(self, other: complex) -> Self:
        """Scalar + vector."""
        if isinstance(other, _SCALAR_TYPES):
            return self._componentwise(
                other, lambda a, b: b + a, "+")
        return NotImplemented

    def __sub__(self, other: Self | complex) -> Self:
        """Componentwise difference."""
        if isinstance(other, VectorField | _SCALAR_TYPES):
            return self._componentwise(
                other, lambda a, b: a - b, "-")
        return NotImplemented

    def __rsub__(self, other: complex) -> Self:
        """Scalar - vector."""
        if isinstance(other, _SCALAR_TYPES):
            return self._componentwise(
                other, lambda a, b: b - a, "-")
        return NotImplemented

    def __neg__(self) -> Self:
        """Componentwise negation (metadata preserved)."""
        return self._componentwise(
            None, lambda a, _: -a, "-")

    def __pos__(self) -> Self:
        """Identity."""
        return self

    def __mul__(
        self, other: Self | ScalarField | complex,
    ) -> Self:
        """
        Componentwise product.

        Description
        -----------
        Scalar scaling, ``ScalarField`` broadcast product (the
        classic ``f_cor * velocity``), or component-by-component
        product (each pair follows the ScalarField join rule).

        Parameters
        ----------
        other : Self | ScalarField | complex
            The multiplier.

        Returns
        -------
        Self
            The componentwise product.
        """
        if isinstance(other,
                      VectorField | ScalarField | _SCALAR_TYPES):
            return self._componentwise(
                other, lambda a, b: a * b, "*")
        return NotImplemented

    def __rmul__(self, other: ScalarField | complex) -> Self:
        """ScalarField/scalar * vector (broadcast product)."""
        if isinstance(other, ScalarField | _SCALAR_TYPES):
            return self._componentwise(
                other, lambda a, b: b * a, "*")
        return NotImplemented

    def __truediv__(
        self, other: Self | ScalarField | complex,
    ) -> Self:
        """Componentwise quotient (dispatched per component)."""
        if isinstance(other,
                      VectorField | ScalarField | _SCALAR_TYPES):
            return self._componentwise(
                other, lambda a, b: a / b, "/")
        return NotImplemented

    def __pow__(self, exponent: float) -> Self:
        """Componentwise physical power."""
        if isinstance(exponent, int | float):
            return self._componentwise(
                exponent, lambda a, b: a ** b, "**")
        return NotImplemented

    # ================================================================
    #  Diagnostics and export
    # ================================================================
    def has_nan(self) -> jax.Array:
        """0-d boolean array: any NaN in any component."""
        return jnp.stack(
            [field.has_nan() for field in self._fields]).any()

    def block_until_ready(self) -> Self:
        """Wait for async device work on all components."""
        for field in self._fields:
            field.block_until_ready()
        return self

    @property
    def xr(self) -> object:
        """Export to an xarray Dataset (Wave-4 export cluster)."""
        raise NotImplementedError(
            "xarray export arrives with the Wave-4 export cluster")

    def __repr__(self) -> str:
        """Names and component spaces summary."""
        parts = ", ".join(
            f"{name}: {field.function_space.bare!r}"
            for name, field in zip(self._names, self._fields,
                                   strict=True))
        return f"VectorField({parts})"


def _keep_metadata(
    result: ScalarField, source: ScalarField,
) -> ScalarField:
    """
    Re-attach the source component's metadata to an op result.

    Description
    -----------
    Bare ``ScalarField`` arithmetic returns default metadata (new
    quantity); componentwise ops keep each component's metadata
    because component names are structural (they key the pytree) —
    dropping them would change the treedef of ``z + dt * dz`` and
    break ``lax.scan``/``jit`` round trips.

    Parameters
    ----------
    result : ScalarField
        The freshly computed component.
    source : ScalarField
        The component whose metadata is preserved.

    Returns
    -------
    ScalarField
        The result with the source's metadata.
    """
    if result.metadata is source.metadata:
        return result
    return type(result)(
        result.grid, result.function_space,
        result._data,  # noqa: SLF001 — plumbing-constructor seam
        source.metadata)
