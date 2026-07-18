"""
The data-movement operators: ``Reshard`` and ``Sync``.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_base.md``
(section "Reshard / Sync and requirements-driven lowering"; concepts
in ``04_decomposition.md`` section 5.1). ``Reshard`` is the
user-reachable explicit layout change (``f.reshard(...)`` sugar) —
identity on the bare space, kernel ``Decomposition.redistribute``,
targets restricted to the negotiated layout vocabulary, halo-trace
rule: reset accumulated depth on the moved axes. ``Sync`` is the
internal-only halo-exchange node the operator base appends after
every kernel (iteration-1 contract); users and kernel authors never
spell it. The requirements-driven lowering pass over composites is
designed-for; iteration 1 places ``Sync`` through the base and
elides ``Reshard`` at the application seam.
"""
# Wave 3: Reshard, Sync
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from fridom.spatial.errors import (
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.spatial.interning import InternTable
from fridom.spatial.operators.base import (
    FieldLike,
    Operator,
    UnaryOperator,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.decomposition.layout import Layout
    from fridom.spatial.spaces.tensor_product import SpaceLike

# bound (grid, target) variants are interned so structurally-equal
# requests return the identical object (identity-hash invariant, D6)
_MOVEMENT_TABLE = InternTable()


@final
class Reshard(UnaryOperator):

    """
    Explicit layout change: identity on the bare space.

    Description
    -----------
    Grid-bound (like transforms): the target must be in the grid's
    negotiated layout vocabulary (section 5.1's closed set). The
    kernel is ``Decomposition.redistribute``; the post-kernel sync of
    the operator base refills the re-blocked ghost slots. Applying to
    a field already in the target layout skips the kernel (identity
    elision; the ``f.reshard`` sugar elides the whole application).
    Field arithmetic never reshards — this operator is the explicit
    conversion the layout-strict algebra demands.

    Parameters
    ----------
    grid : object
        The grid whose decomposition realizes the movement.
    target : Layout
        The target layout; must be negotiated vocabulary.
    """

    dispatch_kind: ClassVar[str | None] = None

    def __new__(cls, grid: object, target: Layout) -> Reshard:
        """Return the interned (grid, target) variant."""
        vocabulary = grid.decomposition.layouts
        if target not in vocabulary:
            raise ValueError(
                f"{target} is not in the negotiated layout "
                f"vocabulary {vocabulary} (section 5.1: the set is "
                "closed; declare new layouts at negotiation)")

        def build() -> Reshard:
            obj = super(Reshard, cls).__new__(cls)
            obj._grid = grid
            obj._target = target
            return obj

        return _MOVEMENT_TABLE.intern(
            ("reshard", grid, target), build)

    def __init__(self, grid: object, target: Layout) -> None:
        """No-op: attributes are set in the interning ``__new__``."""

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def grid(self) -> object:
        """The bound grid."""
        return self._grid

    @property
    def target(self) -> Layout:
        """The target layout."""
        return self._target

    # ------------------------------------------------------------
    #  Signature and trace rule
    # ------------------------------------------------------------
    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """
        Return ``domain.bare.with_layout(target)``.

        Description
        -----------
        The one layout-transition resolver: the shared application
        path keeps a laid-out codomain instead of re-attaching the
        domain's layout. (The class doc's "bare domains raise" is
        enforced at application time — the shared path hands
        resolvers stripped spaces, so bareness is checked in
        ``_apply``.)
        """
        return domain.bare.with_layout(self._target)

    def _trace_reset_names(
        self, domain: SpaceLike,
    ) -> tuple[str, ...]:
        """
        Halo-trace rule: the moved axes reset accumulated depth.

        Description
        -----------
        A redistribute is a global data movement, at least as strong
        as a sync on every axis whose device assignment changes
        between the operand's layout and the target.
        """
        source = domain.layout
        before = {} if source is None else dict(source.device_axes)
        after = dict(self._target.device_axes)
        return tuple(sorted(
            name for name in set(before) | set(after)
            if before.get(name) != after.get(name)))

    # ------------------------------------------------------------
    #  Kernel
    # ------------------------------------------------------------
    def _apply(self, f: FieldLike) -> FieldLike:
        """Redistribute the storage onto the target layout."""
        if f.grid is not self._grid:
            raise GridMismatchError(
                "Reshard is grid-bound: the operand lives on a "
                "different grid", left=self._grid, right=f.grid,
                operation="reshard")
        space = f.function_space
        if space.layout is None:
            raise SpaceMismatchError(
                "cannot reshard a field on a bare (layout-free) "
                f"space {space!r}", left=space, operation="reshard")
        if space.layout == self._target:
            return f  # identity elision
        decomposition = self._grid.decomposition
        data = decomposition.redistribute(
            f._data,  # noqa: SLF001 — documented storage seam
            space, space.layout, self._target)
        # halo-validity claim (task 1.8, stage B): re-blocking leaves
        # the moved axes' ghost slots stale, unmoved axes carry over
        valid = f.halo_valid
        for name in self._trace_reset_names(space):
            if name in valid:
                valid = valid.reset(name)
        return type(f)(f.grid, space.bare.with_layout(self._target),
                       data, f.metadata, halo_valid=valid)


@final
class Sync(Operator):

    """
    Halo-exchange node; inserted by the base/lowering only.

    Description
    -----------
    Internal-only (a user-facing sync would leak the storage layer
    into the semantic layer): it realizes the consumption-side
    contract (task 1.8) — the operator base inserts this node
    *before* a kernel whose operand's ghost validity is below the
    application's requirement — by delegating to ``grid.sync``,
    which resolves the per-axis fill modes (periodic wrap /
    BC-structured fill / designed-for ``ghost_fill``) and is a
    structural no-op where the negotiated width is 0. Identity on
    the space: halo validity is storage bookkeeping, not space
    identity. A singleton.
    """

    _instance: ClassVar[Sync | None] = None

    def __new__(cls) -> Sync:
        """Return the interned singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Identity — halo validity is storage, not space identity."""
        return domains[0] if len(domains) == 1 else domains

    def __call__(self, f: FieldLike) -> FieldLike:
        """Exchange/fill the operand's halos through its grid."""
        return f.grid.sync(f)
