r"""
Mixed per-component transforms: composition + resolution front door.

Description
-----------
A walled component's coefficient space is a **mixed** product —
``Fourier(x) x Fourier(y) x Sine/Cosine(z)`` — and the registry's
form-2 product resolution deliberately raises on it (``registry.py``:
a genuinely mixed product resolves to *different* operators across
its factors). The per-family grid-bound transforms *are* resolvable
per factor, each carrying its own family axes, and their composition
round-trips machine-exactly. :class:`ComposedTransform` packages that
composition behind the exact transform surface ``BoundTransform`` /
``SpectralSolve`` / ``GridSymbols`` consume (``forward`` /
``backward`` / ``codomain`` / ``backward_space`` /
``requirements``); :func:`resolve_transform` is the resolution front
door — the registry's own instance on a homogeneous product
(identical behavior), the composed transform on a mixed one.

The forward order is **Hermitian-first**: the real-to-complex
(Fourier) family runs before the real-to-real trig families, so the
Hermitian half spectrum lands on the first real transformed axis —
the single-device planner convention of ``transform.py``. The trig
kernels accept the complex intermediate; ``backward`` is the exact
reverse, running the Hermitian family last.
"""
# Phase 2 C5: ComposedTransform + resolve_transform (mixed products)
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, final

from fridom.framework2.grid.errors import GridMismatchError
from fridom.framework2.grid.operators.base import OperatorRequirements
from fridom.framework2.grid.operators.registry import DispatchError
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.operators.base import FieldLike
    from fridom.framework2.grid.operators.transform import Transform
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


# ================================================================
#  The composed transform
# ================================================================
@final
class ComposedTransform:

    r"""
    Per-family transforms of a mixed product, composed in order.

    Description
    -----------
    Holds the (deduplicated) per-family grid-bound transforms in
    forward execution order — the Hermitian (Fourier) family first —
    and implements the transform surface by threading the operand
    through them: ``forward`` runs the parts in order, ``backward``
    is the exact reverse, and ``codomain`` / ``backward_space``
    thread the bare space resolutions the same way. Each part only
    touches its own family axes (foreign factors pass through its
    planner untouched), so the composition is exactly the manual
    per-family application.

    Parameters
    ----------
    parts : tuple[Transform, ...]
        The per-family grid-bound transforms, in forward execution
        order. Build through :func:`resolve_transform` instead of
        this plumbing constructor.
    """

    def __init__(self, parts: tuple[Transform, ...]) -> None:
        """Validate and store the per-family forward chain."""
        parts = tuple(parts)
        if not parts:
            raise ValueError(
                "a composed transform composes at least one "
                "per-family transform, got an empty chain")
        grid = parts[0].grid
        for part in parts[1:]:
            if part.grid is not grid:
                raise GridMismatchError(
                    "the per-family transforms of a composed "
                    "transform must share one grid",
                    left=grid, right=part.grid,
                    operation="ComposedTransform")
        self._parts: tuple[Transform, ...] = parts

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def parts(self) -> tuple[Transform, ...]:
        """The per-family transforms, in forward execution order."""
        return self._parts

    @property
    def grid(self) -> Grid:
        """The shared bound grid of every part."""
        return self._parts[0].grid

    # ================================================================
    #  Application
    # ================================================================
    def forward(self, f: FieldLike) -> FieldLike:
        """
        Nodal -> mixed coefficient: run the parts in order.

        Parameters
        ----------
        f : FieldLike
            The field on the mixed nodal domain.

        Returns
        -------
        FieldLike
            The field on the mixed coefficient product (metadata
            preserved by each part).
        """
        for part in self._parts:
            f = part.forward(f)
        return f

    def backward(self, f: FieldLike) -> FieldLike:
        """
        Mixed coefficient -> nodal: run the parts in reverse.

        Parameters
        ----------
        f : FieldLike
            The field on the mixed coefficient product.

        Returns
        -------
        FieldLike
            The field on the mixed nodal domain (real for a real
            origin: the Hermitian family runs last).
        """
        for part in reversed(self._parts):
            f = part.backward(f)
        return f

    # ================================================================
    #  Space resolution
    # ================================================================
    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """
        Forward target: thread the parts' codomains in order.

        Parameters
        ----------
        domain : SpaceLike
            The bare mixed nodal domain.

        Returns
        -------
        SpaceLike
            The bare mixed coefficient product.
        """
        space = domain.bare
        for part in self._parts:
            space = part.codomain(space)
        return space

    def backward_space(self, domain: SpaceLike) -> SpaceLike:
        """
        Backward target: thread the parts' targets in reverse.

        Parameters
        ----------
        domain : SpaceLike
            The bare mixed coefficient product.

        Returns
        -------
        SpaceLike
            The bare mixed nodal domain.
        """
        space = domain.bare
        for part in reversed(self._parts):
            space = part.backward_space(space)
        return space

    def requirements(
        self,
        domain: SpaceLike,  # noqa: ARG002 — uniform declaration
    ) -> OperatorRequirements:
        """
        Declare halo = 0, layout "transpose" (the transform record).

        Parameters
        ----------
        domain : SpaceLike
            The (factor) space the transform is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record, matching every
            part's own declaration.
        """
        return OperatorRequirements(halo=0, layout="transpose")


# ================================================================
#  The resolution front door
# ================================================================
#: per-grid memo of resolved transforms, keyed on the interned bare
#: operand space (the ``base.py`` ``_SYNC_CACHE`` idiom: the
#: WeakKeyDictionary auto-evicts a dropped grid with its memo)
_RESOLVED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def resolve_transform(
    grid: Grid, space: SpaceLike,
) -> Transform | ComposedTransform:
    """
    Resolve the transform of ``space``, composing mixed products.

    Description
    -----------
    The front door replacing direct ``grid.dispatch.resolve``
    ("transform", ...) calls: a homogeneous space returns the
    registry's own grid-bound instance (identical behavior); a
    genuinely mixed product — the walled ``Fourier x Fourier x
    Sine/Cosine`` case the registry deliberately raises on — resolves
    per non-Constant factor, deduplicates the per-family instances,
    orders them Hermitian-first, and wraps them in a
    :class:`ComposedTransform`. Resolutions are memoized per
    ``(grid, bare space)`` identity (spaces are interned).

    Parameters
    ----------
    grid : Grid
        The grid whose dispatch registry mediates the resolution.
    space : SpaceLike
        The (possibly laid-out) operand space.

    Returns
    -------
    Transform | ComposedTransform
        The registry's transform, or the composed mixed one.
    """
    bare = space.bare
    memo = _RESOLVED.setdefault(grid, {})
    cached = memo.get(bare)
    if cached is None:
        cached = _resolve(grid, bare)
        memo[bare] = cached
    return cached


def _resolve(
    grid: Grid, bare: SpaceLike,
) -> Transform | ComposedTransform:
    """Uncached resolution: registry first, mixed composition after."""
    try:
        return grid.dispatch.resolve("transform", bare)
    except DispatchError:
        if not isinstance(bare, TensorProductSpace):
            raise
        parts: list[Transform] = []
        for factor in bare.factors:
            if isinstance(factor, ConstantSpace):
                continue
            part = grid.dispatch.resolve("transform", factor)
            if part not in parts:  # identity: operators hash by id
                parts.append(part)
        if not parts:
            raise  # all-Constant: the registry's error stands
        # Hermitian-first forward order (the planner convention):
        # stable sort keeps grid coordinate order within each group
        parts.sort(
            key=lambda p: not p._hermitian)  # noqa: SLF001 — family classvar
        return ComposedTransform(tuple(parts))
