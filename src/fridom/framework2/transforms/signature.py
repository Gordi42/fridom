"""
Transform signatures: what compose/call checks compare (task 2.8).

Description
-----------
``StateSignature`` is the compared value behind a transform's
domain/codomain: grid **object identity** plus the mapped
``(name, space)`` subset (names by equality, spaces by identity —
bare spaces are interned, so ``==`` is identity — order included),
plus a call-time ``rest`` policy for extra input components. Owning
class spec: ``notes/framework2/model/classes/transforms.md``
§"StateSignature"; design source
``notes/framework2/model/08_state_transforms.md`` §10.3 law 2 /
§10.7.2 (mapped subset + rest) / S6 (signature != treedef).

``rest`` is **excluded** from equality/hash (spec completion 4): it
is call-time behavior of the owning transform, not type identity —
including it would break ``Identity`` polymorphism and the
mapped-subset interop law 2 exists to provide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from fridom.framework2.transforms.errors import SignatureMismatchError

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


@dataclass(frozen=True, eq=False)
class StateSignature:

    """
    Grid identity + ordered mapped ``(name, space)`` subset + rest.

    Description
    -----------
    Not a pytree: it lives in a Tier-1 transform's statics
    (identity-hashed grid/spaces inside) and on Tier-2 host objects.
    Signature != treedef (S6): an accumulator-augmented twin and an
    explicit-params Tier-1 transform both interoperate with
    model-built transforms because comparison sees only the mapped
    PROGNOSTIC subset on the same grid object.

    Parameters
    ----------
    grid : Grid
        The grid the mapped components live on (compared by
        object identity).
    components : tuple[tuple[str, SpaceLike], ...]
        The ordered mapped ``(name, bare space)`` subset.
    rest : {"zero", "pass"}, optional
        Policy for extra input components at call time; **excluded**
        from equality/hash (default: "zero").
    """

    grid: Grid
    components: tuple[tuple[str, SpaceLike], ...]
    rest: Literal["zero", "pass"] = field(default="zero")

    # ================================================================
    #  Constructors
    # ================================================================
    @classmethod
    def of_prognostic(
        cls, source: object, *, rest: Literal["zero", "pass"] = "zero",
    ) -> StateSignature:
        """
        Build the full ordered PROGNOSTIC ``(name, space)`` table.

        Description
        -----------
        The Tier-2 preset signature: from an assembled model (its
        field table's PROGNOSTIC rows) or from a State/VectorField
        (all its components). Spaces are taken as the bare
        (layout-free, interned) space of each component.

        Parameters
        ----------
        source : Model | VectorField
            The model or state whose PROGNOSTIC table is read.
        rest : {"zero", "pass"}, optional
            The extra-component policy (default: "zero").

        Returns
        -------
        StateSignature
            The signature over the source's PROGNOSTIC components.
        """
        table = getattr(source, "field_table", None)
        if table is not None:
            grid = source.grid
            names = table.prognostic
            components = tuple(
                (name, table[name].space.bare) for name in names)
            return cls(grid=grid, components=components, rest=rest)
        # a State / VectorField: read every component
        grid = source.grid
        components = tuple(
            (name, source[name].function_space.bare)
            for name in source.component_names)
        return cls(grid=grid, components=components, rest=rest)

    # ================================================================
    #  Identity (grid by object identity; rest excluded)
    # ================================================================
    @property
    def names(self) -> tuple[str, ...]:
        """The mapped component names, in order."""
        return tuple(name for name, _ in self.components)

    def _key(self) -> tuple:
        """Return the comparison key: grid id + ordered name/space pairs."""
        return (id(self.grid), self.components)

    def __eq__(self, other: object) -> bool:
        """Grid by identity, names by equality, spaces by identity."""
        if not isinstance(other, StateSignature):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Identity hash, matching ``__eq__`` (rest excluded)."""
        return hash(self._key())

    # ================================================================
    #  The call-time check (law 2, §10.7.2)
    # ================================================================
    def validate_input(self, state: object, *, path: str = "") -> None:
        """
        Require the input to contain the mapped components.

        Description
        -----------
        The call-time check (trace-time under jit — zero steady
        cost): the input must contain every mapped component (name +
        bare space) in the mapped order; extra PROGNOSTIC components
        are legal and handled per ``rest`` by the owning transform.
        Raises ``SignatureMismatchError`` with the composition-tree
        path, a componentwise diff, and the grid-identity verdict.

        Parameters
        ----------
        state : VectorField
            The input state (duck-typed component container).
        path : str, optional
            The composition-tree path for attribution (default: "").
        """
        where = f" at {path!r}" if path else ""
        grid = getattr(state, "grid", None)
        if grid is not self.grid:
            hint = ("; one grid, many models: build both on one grid"
                    if grid is not None else "")
            raise SignatureMismatchError(
                f"transform input{where} lives on a different grid "
                f"object than the transform's signature{hint}")
        present = set(state.component_names)
        missing, mismatched = [], []
        for name, space in self.components:
            if name not in present:
                missing.append(name)
                continue
            got = state[name].function_space.bare
            if got != space:
                mismatched.append((name, space, got))
        ordered = tuple(name for name in state.component_names
                        if name in self.names)
        if missing or mismatched or ordered != self.names:
            raise SignatureMismatchError(
                self._diff(where, state, missing, mismatched, ordered))

    def _diff(
        self,
        where: str,
        state: object,
        missing: list,
        mismatched: list,
        ordered: tuple,
    ) -> str:
        """Assemble the componentwise mismatch report."""
        lines = [f"transform input{where} does not match the "
                 "signature's mapped components:"]
        lines.append(f"  expected (in order): {self.names}")
        lines.append(f"  input components:    "
                     f"{tuple(state.component_names)}")
        if missing:
            lines.append(f"  missing: {tuple(missing)}")
        for name, want, got in mismatched:
            lines.append(
                f"  space mismatch for {name!r}: expected {want!r}, "
                f"got {got!r}")
        if not missing and not mismatched and ordered != self.names:
            lines.append(f"  wrong order: {ordered} vs {self.names}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Grid name/id + the component table + rest policy."""
        grid_name = getattr(self.grid, "name", None) or hex(id(
            self.grid))
        comps = ", ".join(
            f"{name}:{space!r}" for name, space in self.components)
        return (f"StateSignature(grid={grid_name}, [{comps}], "
                f"rest={self.rest!r})")
