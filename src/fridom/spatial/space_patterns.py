"""
Grid-free space descriptors.

Description
-----------
Owning class spec: ``design/specs/model/classes/declarations.md``
("Dof, SpacePattern, and the sugar constructors"; "SpaceRule").
``SpacePattern`` is the name-keyed semantic space tag of model D1.2:
declarations stay grid-free, and the model resolves each pattern per
mesh factor through the grid-level ``("declared_space", mesh)``
resolver rows at assembly step 1. Patterns are frozen, value-hashable
descriptors — they double as dispatch-merge key components
(``(kind, SpacePattern)``). ``SpaceRule`` is the full-power per-field
escape hatch: it wraps a pure ``fn(grid) -> space`` callable, is
identity-hashed, and is never a dispatch key.
"""
# Wave 2 A: Dof, SpacePattern, Collocated, Staggered, Profile,
#    SpaceRule
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import TYPE_CHECKING

from fridom.spatial.bc import BC, BCStructure
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Mapping

    from fridom.spatial.grid import Grid
    from fridom.spatial.meshes.mesh import Mesh
    from fridom.spatial.spaces.tensor_product import (
        SpaceLike,
    )

class Dof(Enum):

    """
    Per-coordinate semantic tag, resolved per mesh at assembly.

    Description
    -----------
    The closed tag vocabulary of model D1.2: what a declaration says
    about one coordinate, without naming a concrete space. The
    grid-level ``("declared_space", mesh)`` resolver row of each mesh
    maps ``COLLOCATED``/``STAGGERED`` to that mesh's default cell /
    dual representation; ``CONSTANT`` resolves to the universal
    ``mesh.constant`` broadcast factor (the old ``topo=False``).
    """

    COLLOCATED = auto()
    STAGGERED = auto()
    CONSTANT = auto()


@dataclass(frozen=True)
class SpacePattern:

    """
    Name-keyed semantic tags + default + per-coordinate BCs.

    Description
    -----------
    A frozen, value-hashable descriptor (model D1.2): two patterns
    built from equal data compare and hash equal — load-bearing for
    the ``(kind, SpacePattern)`` dispatch-merge keys. Construction
    normalizes the mapping-valued fields to sorted tuples of pairs,
    so insertion order never leaks into equality. Names absent from
    a grid are simply unmatched (dimension generality); the
    ``require=`` names assert a grid match at resolution.

    Parameters
    ----------
    default : Dof, optional
        The tag of every coordinate no ``tags`` entry names
        (default: Dof.COLLOCATED).
    tags : tuple[tuple[str, Dof], ...], optional
        Per-coordinate tag overrides, keyed by coordinate name
        (default: ()).
    bc : tuple[tuple[str, BC | BCStructure], ...], optional
        Per-coordinate homogeneous BC structure, keyed by coordinate
        name; enters the resolved space's interning key
        (default: ()).
    wall_bc : tuple[tuple[str, BC | BCStructure], ...], optional
        Topology-conditional per-coordinate BCs: an entry applies
        only where the matched mesh factor is *bounded*
        (non-periodic) and is ignored on periodic factors, so a
        fully periodic grid resolves to the identical interned
        spaces as without it. This is the wall-derivation channel
        of the topology-driven-walls decision (C8): grid
        periodicity is the only user-facing switch. A name may not
        appear in both ``bc`` and ``wall_bc`` (default: ()).
    require : tuple[str, ...], optional
        Names that must match a mesh factor at resolution — the
        typo mitigation adopted at D4 sign-off (default: ()).
    scalars : Scalars | None, optional
        Requested Körper of the resolved space (``REAL`` /
        ``COMPLEX``, no width axis — CS-17 is global-precision
        only); None keeps the resolvers' choice (default: None).
    """

    default: Dof = Dof.COLLOCATED
    tags: tuple[tuple[str, Dof], ...] = ()
    bc: tuple[tuple[str, BC | BCStructure], ...] = ()
    wall_bc: tuple[tuple[str, BC | BCStructure], ...] = ()
    require: tuple[str, ...] = ()
    scalars: Scalars | None = None

    def __post_init__(self) -> None:
        """Canonicalize and validate the field values."""
        if not isinstance(self.default, Dof):
            raise TypeError(
                f"default must be a Dof member, got {self.default!r}")
        object.__setattr__(
            self, "tags", _normalize_pairs(
                self.tags, _check_dof, label="tags"))
        object.__setattr__(
            self, "bc", _normalize_pairs(
                self.bc, _check_bc, label="bc"))
        object.__setattr__(
            self, "wall_bc", _normalize_pairs(
                self.wall_bc, _check_bc, label="wall_bc"))
        object.__setattr__(
            self, "require", _normalize_names(self.require))
        overlap = ({name for name, _ in self.bc}
                   & {name for name, _ in self.wall_bc})
        if overlap:
            raise ValueError(
                f"coordinates {tuple(sorted(overlap))} appear in "
                "both bc and wall_bc; a coordinate carries either "
                "an unconditional BC (bc) or a topology-conditional "
                "one (wall_bc), never both")
        if self.scalars is not None and not isinstance(
                self.scalars, Scalars):
            raise TypeError(
                f"scalars must be a Scalars member or None, got "
                f"{self.scalars!r}")

    @classmethod
    def create(
        cls,
        default: Dof = Dof.COLLOCATED,
        tags: Mapping[str, Dof] | None = None,
        bc: Mapping[str, BC | BCStructure] | None = None,
        wall_bc: Mapping[str, BC | BCStructure] | None = None,
        require: Iterable[str] = (),
        scalars: Scalars | None = None,
    ) -> SpacePattern:
        """
        Build a pattern from mappings (canonical tuple form).

        Parameters
        ----------
        default : Dof, optional
            The tag of every unnamed coordinate
            (default: Dof.COLLOCATED).
        tags : Mapping[str, Dof] | None, optional
            Per-coordinate tag overrides (default: None).
        bc : Mapping[str, BC | BCStructure] | None, optional
            Per-coordinate BC structure (default: None).
        wall_bc : Mapping[str, BC | BCStructure] | None, optional
            Topology-conditional BCs, applied only on bounded mesh
            factors (default: None).
        require : Iterable[str], optional
            Names that must match a mesh factor (default: ()).
        scalars : Scalars | None, optional
            Requested Körper of the resolved space (default: None).

        Returns
        -------
        SpacePattern
            The normalized, value-hashable pattern.
        """
        return cls(
            default=default,
            tags=() if tags is None else tuple(tags.items()),
            bc=() if bc is None else tuple(bc.items()),
            wall_bc=(() if wall_bc is None
                     else tuple(wall_bc.items())),
            require=tuple(require),
            scalars=scalars)

    # ================================================================
    #  Resolution (model assembly step 1)
    # ================================================================
    def resolve(self, grid: Grid) -> TensorProductSpace:
        """
        Resolve the pattern to a bare interned product space.

        Description
        -----------
        Per mesh factor: pick the tag (``default`` when no
        coordinate name of the mesh matches) and the BC, and call
        the grid-level resolver row
        ``grid.dispatch[("declared_space", mesh)](tag, bc)``.
        A ``wall_bc`` entry substitutes for an absent ``bc`` entry
        only where the matched mesh is bounded (topology-driven
        walls, C8); on periodic factors it is ignored, so periodic
        grids resolve to the identical interned spaces as without
        it. ``Dof.CONSTANT`` resolves directly to the universal
        ``mesh.constant`` (no resolver row consulted). The result is
        the flat interned product — bare, pre-layout. Pure: spaces
        are interned, so repeated resolution returns the identical
        object.

        Parameters
        ----------
        grid : Grid
            The assembled grid whose mesh factors resolve the tags.

        Returns
        -------
        TensorProductSpace
            The bare interned product (a lone factor space on
            single-factor grids).

        Raises
        ------
        ValueError
            If a ``require`` name matched no mesh factor, if a BC
            entry names a constant-resolved coordinate, or if the
            resolved Körper cannot satisfy ``scalars``.
        DispatchError
            If a non-constant factor's mesh has no
            ``("declared_space", mesh)`` resolver row (hinted).
        """
        self._check_require(grid)
        tags = dict(self.tags)
        bcs = dict(self.bc)
        walls = dict(self.wall_bc)
        factors = []
        for mesh in grid.factors:
            tag = _match(mesh, tags, self.default)
            bc = _match(mesh, bcs, None)
            if bc is None and not getattr(mesh, "periodic", False):
                # topology-conditional walls: a wall_bc entry bites
                # only on a bounded mesh factor (C8)
                bc = _match(mesh, walls, None)
            if tag is Dof.CONSTANT:
                if bc is not None:
                    raise ValueError(
                        f"pattern {self!r} pins a BC on a "
                        f"coordinate of {mesh!r}, which resolves to "
                        "the constant factor; constant factors "
                        "carry no BC structure")
                factors.append(mesh.constant)
                continue
            factors.append(_resolver_row(grid, mesh)(tag, bc))
        space = TensorProductSpace.of(*factors)
        return self._request_scalars(space)

    def _check_require(self, grid: Grid) -> None:
        """Raise if a required name matched no mesh factor."""
        missing = tuple(
            name for name in self.require
            if name not in grid.names)
        if missing:
            raise ValueError(
                f"pattern {self!r} requires the coordinate names "
                f"{missing}, which match no mesh factor of the grid "
                f"(coordinates: {grid.names}); require= asserts the "
                "name-matched-a-factor property (D4 sign-off, typo "
                "mitigation)")

    def _request_scalars(self, space: SpaceLike) -> SpaceLike:
        """Apply the requested Körper to the resolved space."""
        if self.scalars is None:
            return space
        if self.scalars is Scalars.COMPLEX:
            return space.as_complex()
        if space.scalars is not Scalars.REAL:
            raise ValueError(
                f"pattern {self!r} requests REAL scalars but the "
                f"resolver rows produced the complex {space!r}")
        return space

    def __repr__(self) -> str:
        """Round-tripping repr, default-valued fields omitted."""
        parts = []
        for field in fields(self):
            value = getattr(self, field.name)
            if value == field.default:
                continue
            parts.append(f"{field.name}={_render(value)}")
        return f"SpacePattern({', '.join(parts)})"


# ================================================================
#  Sugar constructors (PascalCase factory functions, per spec)
# ================================================================
def Collocated(  # noqa: N802 — constructor-like factory (spec)
    *, bc: Mapping[str, BC | BCStructure] | None = None,
    wall_bc: Mapping[str, BC | BCStructure] | None = None,
    require: Iterable[str] = (),
    scalars: Scalars | None = None,
) -> SpacePattern:
    """
    Collocated everywhere: ``SpacePattern()``.

    Parameters
    ----------
    bc : Mapping[str, BC | BCStructure] | None, optional
        Per-coordinate BC structure (default: None).
    wall_bc : Mapping[str, BC | BCStructure] | None, optional
        Topology-conditional BCs, applied only on bounded mesh
        factors (default: None).
    require : Iterable[str], optional
        Names that must match a mesh factor (default: ()).
    scalars : Scalars | None, optional
        Requested Körper of the resolved space (default: None).

    Returns
    -------
    SpacePattern
        The all-collocated pattern.
    """
    return SpacePattern.create(bc=bc, wall_bc=wall_bc,
                               require=require, scalars=scalars)


def Staggered(  # noqa: N802 — constructor-like factory (spec)
    *names: str,
    bc: Mapping[str, BC | BCStructure] | None = None,
    wall_bc: Mapping[str, BC | BCStructure] | None = None,
    require: Iterable[str] = (),
    scalars: Scalars | None = None,
) -> SpacePattern:
    """
    Staggered along the named coordinates, collocated elsewhere.

    Description
    -----------
    B-grid velocities name two coordinates; at least one name is
    required. A name absent from the grid is simply unmatched:
    ``Staggered("y")`` on an (x, z) grid resolves fully collocated —
    exactly right for the transverse ``v`` of a 2D rotating slice
    (paired with the V-N1 transverse-component rule).

    Parameters
    ----------
    *names : str
        The coordinates to stagger (at least one).
    bc : Mapping[str, BC | BCStructure] | None, optional
        Per-coordinate BC structure (default: None).
    wall_bc : Mapping[str, BC | BCStructure] | None, optional
        Topology-conditional BCs, applied only on bounded mesh
        factors (default: None).
    require : Iterable[str], optional
        Names that must match a mesh factor (default: ()).
    scalars : Scalars | None, optional
        Requested Körper of the resolved space (default: None).

    Returns
    -------
    SpacePattern
        The staggered pattern.
    """
    if not names:
        raise ValueError(
            "Staggered() needs at least one coordinate name; "
            "'collocated everywhere' is spelled Collocated()")
    return SpacePattern.create(
        tags=dict.fromkeys(names, Dof.STAGGERED),
        bc=bc, wall_bc=wall_bc, require=require, scalars=scalars)


def Profile(  # noqa: N802 — constructor-like factory (spec)
    *names: str,
    bc: Mapping[str, BC | BCStructure] | None = None,
    wall_bc: Mapping[str, BC | BCStructure] | None = None,
    require: Iterable[str] = (),
    scalars: Scalars | None = None,
) -> SpacePattern:
    """
    ConstantSpace on all axes except the named ones (old ``topo``).

    Description
    -----------
    ``Profile("z")`` is the classic background profile (``n2(z)``);
    ``Profile()`` is all-constant — the one-DOF R2 parameter field.
    Unmatched names degrade gracefully: ``Profile("y")`` on an
    (x, z) grid resolves to all-constant.

    Parameters
    ----------
    *names : str
        The coordinates the field varies along.
    bc : Mapping[str, BC | BCStructure] | None, optional
        Per-coordinate BC structure (default: None).
    wall_bc : Mapping[str, BC | BCStructure] | None, optional
        Topology-conditional BCs, applied only on bounded mesh
        factors (default: None).
    require : Iterable[str], optional
        Names that must match a mesh factor (default: ()).
    scalars : Scalars | None, optional
        Requested Körper of the resolved space (default: None).

    Returns
    -------
    SpacePattern
        The profile pattern.
    """
    return SpacePattern.create(
        default=Dof.CONSTANT,
        tags=dict.fromkeys(names, Dof.COLLOCATED),
        bc=bc, wall_bc=wall_bc, require=require, scalars=scalars)


# ================================================================
#  SpaceRule: the full-power per-field escape hatch
# ================================================================
class SpaceRule:

    """
    Wraps ``fn(grid) -> space``; same resolve protocol.

    Description
    -----------
    The escape hatch of model D1.2: full mesh-factory power for the
    ~5% of cases the semantic tags do not cover (coefficient-space
    AUX fields, unstructured factors until the tag vocabulary
    grows). The callable must be pure and static-only: it runs at
    assembly step 1, before ``negotiate``, may not consult the
    decomposition, and returns a bare (pre-layout) space built from
    the grid's own mesh factories. Unlike ``SpacePattern`` a rule is
    identity-hashed (it wraps behavior, not value data) and is NEVER
    a dispatch-merge key; the restart fingerprint hashes the
    *resolved* bare space, so a rule needs no token of its own.

    Parameters
    ----------
    fn : Callable[[Grid], TensorProductSpace]
        The pure per-grid space rule.
    """

    def __init__(
        self,
        fn: Callable[[Grid], TensorProductSpace],
    ) -> None:
        """Store the space rule; reject non-callables."""
        if not callable(fn):
            raise TypeError(
                f"SpaceRule wraps a callable fn(grid) -> space, "
                f"got {fn!r}")
        self._fn: Callable[[Grid], TensorProductSpace] = fn

    @property
    def fn(self) -> Callable[[Grid], TensorProductSpace]:
        """The wrapped pure space rule."""
        return self._fn

    def resolve(self, grid: Grid) -> TensorProductSpace:
        """
        Call ``fn(grid)``; debug mode checks purity by identity.

        Description
        -----------
        Under ``__debug__`` (i.e. unless Python runs with ``-O``)
        the rule is resolved twice and the results are compared by
        identity — cheap, because spaces are interned: a pure rule
        returns the identical object both times.

        Parameters
        ----------
        grid : Grid
            The assembled grid to resolve against.

        Returns
        -------
        TensorProductSpace
            The bare interned space the rule built.

        Raises
        ------
        ValueError
            If the debug double-resolve returns a distinct object
            (the rule is impure or bypasses the interned factories).
        """
        space = self._fn(grid)
        if __debug__ and self._fn(grid) is not space:
            raise ValueError(
                f"{self!r} is impure: two resolves returned "
                "distinct spaces. Rules must be pure and build "
                "bare spaces from the grid's own mesh factories "
                "(interning makes repeated resolution return the "
                "identical object)")
        return space

    def __repr__(self) -> str:
        """Render as ``SpaceRule(<fn name>)``."""
        name = getattr(self._fn, "__qualname__",
                       repr(self._fn))
        return f"SpaceRule({name})"


# ================================================================
#  Normalization / rendering helpers
# ================================================================
def _normalize_pairs(
    pairs: object,
    check_value: Callable[[str, object, str], object],
    *,
    label: str,
) -> tuple[tuple[str, object], ...]:
    """Sort name-keyed pairs; validate names and values."""
    normalized: dict[str, object] = {}
    for pair in tuple(pairs):
        try:
            name, value = pair
        except (TypeError, ValueError):
            raise TypeError(
                f"{label} entries are (name, value) pairs, got "
                f"{pair!r}") from None
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{label} keys are non-empty coordinate names, got "
                f"{name!r}")
        if name in normalized:
            raise ValueError(
                f"duplicate {label} entry for coordinate {name!r}")
        normalized[name] = check_value(name, value, label)
    return tuple(sorted(normalized.items()))


def _check_dof(name: str, value: object, label: str) -> Dof:
    """Validate one tags value."""
    if not isinstance(value, Dof):
        raise TypeError(
            f"{label}[{name!r}] must be a Dof member, got "
            f"{value!r}")
    return value


def _check_bc(
    name: str, value: object, label: str,
) -> BC | BCStructure:
    """Validate one bc / wall_bc value; normalize component tuples."""
    if isinstance(value, BC | BCStructure):
        return value
    if isinstance(value, tuple):
        return BCStructure(value)
    raise TypeError(
        f"{label}[{name!r}] must be a BC member, a BCStructure, or "
        f"a per-component BC tuple, got {value!r}")


def _normalize_names(names: object) -> tuple[str, ...]:
    """Sort and deduplicate the require names."""
    names = tuple(names)
    for name in names:
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"require entries are non-empty coordinate names, "
                f"got {name!r}")
    return tuple(sorted(set(names)))


def _render(value: object) -> str:
    """Render one repr component in round-tripping spelling."""
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, tuple):
        inner = ", ".join(_render(item) for item in value)
        if len(value) == 1:
            inner += ","
        return f"({inner})"
    return repr(value)


def _match(
    mesh: Mesh, entries: dict[str, object], default: object,
) -> object:
    """
    Pick a mesh factor's entry from name-keyed pattern data.

    Description
    -----------
    The name-keyed premise of D1.2: a mesh factor's entry is the one
    matching any of its coordinate names (1D factors have exactly
    one). Distinct values matched through different names of one
    multi-name factor are contradictory and raise.

    Parameters
    ----------
    mesh : Mesh
        The mesh factor being resolved.
    entries : dict[str, object]
        The pattern's name-keyed data (tags or bc).
    default : object
        The value when no name matches.

    Returns
    -------
    object
        The matched value, or ``default``.
    """
    matched = [entries[name] for name in mesh.names
               if name in entries]
    if not matched:
        return default
    first = matched[0]
    if any(value != first for value in matched[1:]):
        raise ValueError(
            f"contradictory pattern entries {matched!r} match the "
            f"coordinate names of the single mesh factor {mesh!r}; "
            "a factor resolves to one representation")
    return first


def _resolver_row(
    grid: Grid, mesh: Mesh,
) -> Callable[[Dof, BC | BCStructure | None], object]:
    """
    Look up the mesh's ``("declared_space", mesh)`` resolver row.

    Parameters
    ----------
    grid : Grid
        The grid whose dispatch registry holds the resolver table.
    mesh : Mesh
        The mesh factor being resolved.

    Returns
    -------
    Callable
        The resolver ``(tag, bc) -> factor space``.

    Raises
    ------
    DispatchError
        If the mesh has no resolver row (hinted: rows are seeded
        grid-level, never module-mergeable).
    """
    try:
        return grid.dispatch[("declared_space", mesh)]
    except KeyError:
        raise DispatchError(
            f"no ('declared_space', {mesh!r}) resolver row is "
            "registered on this grid, so declared space patterns "
            "cannot resolve along it. Resolver rows are grid-level "
            "only (model D1.2): seed one per mesh at grid "
            "construction via grid.dispatch[('declared_space', "
            "mesh)] = resolver, where resolver(tag, bc) returns "
            "the mesh's factor space for a Dof tag; they are never "
            "module-mergeable") from None
