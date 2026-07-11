"""
``VectorField``: a named collection of scalar fields (no metric).

Description
-----------
Owning class doc: ``design/specs/grid/classes/fields.md``. Thin means
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
from fridom.framework2.grid.errors import (
    GridMismatchError,
    MissingComponentError,
)
from fridom.framework2.grid.fields.scalar_field import ScalarField

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Iterator

    import jax
    import xarray as xr

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
                raise MissingComponentError(
                    f"no component named {key!r}; components are "
                    f"{self._names}") from None
            return self._fields[index]
        if isinstance(key, int) and not isinstance(key, bool):
            return self._fields[key]
        raise TypeError(
            f"components are indexed by name or position, got "
            f"{key!r}")

    def require(self, name: str, *, hint: str) -> ScalarField:
        """
        Return component ``name`` or raise a hinted error.

        Description
        -----------
        The lookup for callers that can suggest a fix: on a miss the
        ``MissingComponentError`` names the missing component, the
        caller's ``hint`` (what was expected / how to supply it), and
        the present component names.

        Parameters
        ----------
        name : str
            The component name to fetch.
        hint : str
            A caller-supplied remediation hint included in the
            message.

        Returns
        -------
        ScalarField
            The requested component.
        """
        if name in self._names:
            return self._fields[self._names.index(name)]
        raise MissingComponentError(
            f"no component named {name!r}: {hint}; present "
            f"components are {self._names}")

    def select(self, *names: str) -> VectorField:
        """
        Return a new VectorField of only the named components.

        Description
        -----------
        Subsets and re-keys the container in the given ``names`` order
        (preserving it); a name that is not a present component raises
        ``MissingComponentError``.

        Parameters
        ----------
        *names : str
            The component names to keep, in the desired order.

        Returns
        -------
        VectorField
            The subset collection; ``self`` is unchanged.
        """
        missing = tuple(name for name in names
                        if name not in self._names)
        if missing:
            raise MissingComponentError(
                f"no components named {missing}; components are "
                f"{self._names}")
        return type(self)(
            {name: self._fields[self._names.index(name)]
             for name in names})

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
        ``ScalarField``; the result keeps names and order, and the
        incumbent component's metadata is re-attached to each result
        (2026-07-08 amendment). ``map`` never inspects spaces —
        per-component space changes (e.g. transforms) land in the
        returned collection.

        Parameters
        ----------
        fn : Callable[[ScalarField], ScalarField]
            The per-component function.

        Returns
        -------
        Self
            The mapped collection.
        """
        mapped = tuple(
            _keep_metadata(fn(field), field)
            for field in self._fields)
        return type(self)(
            dict(zip(self._names, mapped, strict=True)))

    def replace(self, **components: ScalarField) -> Self:
        """
        Return the collection with named components updated.

        Description
        -----------
        The incumbent component's metadata is re-attached to each
        incoming field, keyed by component name (2026-07-08
        amendment) — component annotation stays authoritative even
        when the replacement was built by default-metadata scalar
        arithmetic. Untouched components pass through unchanged.

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
            name: (_keep_metadata(components[name], field)
                   if name in components else field)
            for name, field in zip(self._names, self._fields,
                                   strict=True)}
        return type(self)(updated)

    def add(self, **contributions: ScalarField) -> Self:
        """
        Functional accumulate of named contributions.

        Description
        -----------
        Each keyword names an existing component; that component is
        replaced by ``self[name] + contribution`` through the
        metadata-preserving path (the incumbent component's metadata
        is re-attached, 2026-07-08 amendment). Every other component
        passes through unchanged (the same object). The result keeps
        the vector's component order, never the keyword order. This
        is the composer's primitive for summing tendency-contribution
        dicts (model design D1.5).

        Parameters
        ----------
        **contributions : ScalarField
            Per-component addends; each pair follows the ScalarField
            join rule on its own space.

        Returns
        -------
        Self
            The accumulated collection; ``self`` is unchanged.
        """
        unknown = tuple(name for name in contributions
                        if name not in self._names)
        if unknown:
            raise KeyError(
                f"no components named {unknown}; components are "
                f"{self._names}")
        updated = {
            name: (_keep_metadata(field + contributions[name], field)
                   if name in contributions else field)
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
        if isinstance(other, VectorField) or _is_scalar(other):
            return self._componentwise(
                other, lambda a, b: a + b, "+")
        return NotImplemented

    def __radd__(self, other: complex) -> Self:
        """Scalar + vector."""
        if _is_scalar(other):
            return self._componentwise(
                other, lambda a, b: b + a, "+")
        return NotImplemented

    def __sub__(self, other: Self | complex) -> Self:
        """Componentwise difference."""
        if isinstance(other, VectorField) or _is_scalar(other):
            return self._componentwise(
                other, lambda a, b: a - b, "-")
        return NotImplemented

    def __rsub__(self, other: complex) -> Self:
        """Scalar - vector."""
        if _is_scalar(other):
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
        if (isinstance(other, VectorField | ScalarField)
                or _is_scalar(other)):
            return self._componentwise(
                other, lambda a, b: a * b, "*")
        return NotImplemented

    def __rmul__(self, other: ScalarField | complex) -> Self:
        """ScalarField/scalar * vector (broadcast product)."""
        if isinstance(other, ScalarField) or _is_scalar(other):
            return self._componentwise(
                other, lambda a, b: b * a, "*")
        return NotImplemented

    def __truediv__(
        self, other: Self | ScalarField | complex,
    ) -> Self:
        """Componentwise quotient (dispatched per component)."""
        if (isinstance(other, VectorField | ScalarField)
                or _is_scalar(other)):
            return self._componentwise(
                other, lambda a, b: a / b, "/")
        return NotImplemented

    def __pow__(self, exponent: float) -> Self:
        """Componentwise physical power."""
        if isinstance(exponent, int | float) or _is_0d_array(exponent):
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
    def xr(self) -> xr.Dataset:
        """
        Export to an ``xarray.Dataset`` of the components' exports.

        Returns
        -------
        xr.Dataset
            One data variable per component, keyed by component
            name (label/gather rules:
            ``fridom.framework2.grid.export``).
        """
        from fridom.framework2.grid.export import (  # noqa: PLC0415 — deferred: keeps optional xarray off the field-core import path
            vector_to_dataset,
        )
        return vector_to_dataset(self)

    def __repr__(self) -> str:
        """Names and component spaces summary."""
        parts = ", ".join(
            f"{name}: {field.function_space.bare!r}"
            for name, field in zip(self._names, self._fields,
                                   strict=True))
        return f"VectorField({parts})"


def _is_0d_array(value: object) -> bool:
    """
    Whether ``value`` is a 0-d array (a scalar, not a field).

    Description
    -----------
    A raw 0-d ``jax.Array`` (e.g. a traced ``ctx.params`` leaf) is a
    scalar coefficient broadcast across every component. An array with
    ``ndim >= 1`` is not a scalar (mirrors ``scalar_field._is_scalar``).
    """
    return (not isinstance(value, VectorField | ScalarField)
            and getattr(value, "ndim", None) == 0)


def _is_scalar(value: object) -> bool:
    """Whether ``value`` enters componentwise arithmetic as scalar."""
    return isinstance(value, _SCALAR_TYPES) or _is_0d_array(value)


def _keep_metadata(
    result: ScalarField, source: ScalarField,
) -> ScalarField:
    """
    Re-attach the source component's metadata to an op result.

    Description
    -----------
    Bare ``ScalarField`` arithmetic returns default metadata (new
    quantity); componentwise ops and the functional surface
    (``map``/``replace``/``add``) re-attach each incumbent
    component's metadata by key (2026-07-08 amendment). Under the
    annotation-exempt aux rule this is no longer load-bearing for
    the treedef — it keeps component annotation authoritative.

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
