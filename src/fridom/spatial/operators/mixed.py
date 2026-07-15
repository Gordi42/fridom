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

from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.errors import GridMismatchError
from fridom.spatial.operators.base import OperatorRequirements
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.operators.transform import (
    _MIN_STAGES,
    TransformPlan,
    TransformStage,
    _complex_storage,
    _paddable,
    _rebuild,
)
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.grid import Grid
    from fridom.spatial.operators.base import FieldLike
    from fridom.spatial.operators.transform import Transform
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


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
        #: memoized joint distributed plans; stores None for
        #: ineligible spaces, so membership (not ``.get``) is the
        #: cache test (the ``Transform._dist_plans`` idiom)
        self._dist_plans: dict[SpaceLike, TransformPlan | None] = {}

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
    #  Distributed planning (multi-device; layout-annotated stages)
    # ================================================================
    def distributed_forward_plan(
        self, domain: SpaceLike,
    ) -> TransformPlan | None:
        """
        Return the joint layout-annotated forward schedule, or None.

        Description
        -----------
        The mixed-product counterpart of
        ``Transform.distributed_forward_plan`` (section 1.1 of
        ``design/plans/active/distributed_transform_plan.md``),
        planned jointly across the per-family parts: the
        initially-local axes transform first under the operand's
        nodal layout, one reshard makes the sharded axis local and
        shards the transpose partner, and the sharded axis
        transforms last. The joint geometry is family-aware where
        the single-family planner need not be: the Hermitian
        half-spectrum stage ``h`` must belong to the Hermitian
        (Fourier) part, so the transpose partner ``b`` prefers a
        divisible axis that leaves a Fourier axis local — the rfft
        on that local axis halves every downstream stage. Returns
        None when the operand is single-device or ineligible (a
        padded part, a non-1-D device mapping, an indivisible
        sharded extent, or no divisible partner axis); callers then
        keep the replicated composite, leaving the single-device
        program unchanged.

        Parameters
        ----------
        domain : SpaceLike
            The (nodal) operand space.

        Returns
        -------
        TransformPlan | None
            The distributed plan, or None when ineligible.
        """
        bare = domain.bare
        if bare not in self._dist_plans:
            self._dist_plans[bare] = (
                self._build_distributed_forward(bare))
        return self._dist_plans[bare]

    def _build_distributed_forward(
        self, bare: SpaceLike,
    ) -> TransformPlan | None:
        """Uncached build of the joint distributed plan (or None)."""
        geometry = self._joint_geometry(bare)
        if geometry is None:
            return None
        name_a, name_b, name_h, stage_names, part_of = geometry
        default_layout = self.grid.decomposition.default_layout
        axis_name = default_layout.device_axes[0][1]
        spectral_layout = Layout({name_b: axis_name})
        ordered = ((name_h,) if name_h is not None else ()) + tuple(
            n for n in stage_names if n not in (name_a, name_h))
        ordered += (name_a,)
        stages: list[TransformStage] = []
        mapping: dict[str, FunctionSpace] = {}
        for axis in ordered:
            part = part_of[axis]
            factor = bare.factor(axis)
            origin = part._coarse_origin(factor, axis)  # noqa: SLF001
            half = axis == name_h
            coeff = part._coefficient_factor(  # noqa: SLF001 — planner seam
                origin, half=half)
            layout = (spectral_layout if axis == name_a
                      else default_layout)
            stages.append(TransformStage(
                axis=axis, index=bare.names.index(axis), half=half,
                nodal=factor, coeff=coeff, layout=layout))
            mapping[axis] = coeff
        codomain = _rebuild(bare, mapping).with_layout(
            spectral_layout)
        return TransformPlan(
            bare.with_layout(default_layout), codomain,
            tuple(stages))

    def _joint_geometry(
        self, bare: SpaceLike,
    ) -> tuple[str, str, str | None, tuple[str, ...],
               dict[str, Transform]] | None:
        """
        Choose the joint slab axes, or None (ineligible).

        Description
        -----------
        The family-aware counterpart of
        ``Transform._distributed_geometry``: ``a`` is the single
        default-layout-sharded coordinate, ``b`` the transpose partner
        that leaves a Hermitian (Fourier) axis local when one exists —
        preferring a **divisible** partner (the byte-identical fast
        path), else a **paddable** one (the padded transpose) — and
        ``h`` the last such local Hermitian axis of a real domain (None
        otherwise). Returns the axes together with the stage-name order
        and the axis -> owning-part map, or None when ``a`` / the only
        partner would pad too heavily (a trailing shard empties).
        """
        decomposition = self.grid.decomposition
        mapped = dict(decomposition.default_layout.device_axes)
        if (any(part.pad is not None for part in self._parts)
                or getattr(decomposition, "device_count", 1) <= 1
                or len(mapped) != 1):
            return None
        (name_a,) = mapped
        part_of: dict[str, Transform] = {}
        for part in self._parts:
            for axis in part._stage_axes(bare):  # noqa: SLF001 — planner seam
                part_of[axis] = part
        stage_names = tuple(n for n in bare.names if n in part_of)
        shards = decomposition.device_count
        if (len(stage_names) < _MIN_STAGES
                or name_a not in stage_names
                or any(len(bare.factor(n).shape) != 1
                       for n in stage_names)
                or not _paddable(bare.factor(name_a).shape[0], shards)):
            return None
        partners = tuple(n for n in stage_names if n != name_a)
        divisible = tuple(
            n for n in partners
            if bare.factor(n).shape[0] % shards == 0)
        # prefer divisible partners (byte-identical fast path); else the
        # paddable ones (the padded balanced all-to-all)
        candidates = divisible or tuple(
            n for n in partners
            if _paddable(bare.factor(n).shape[0], shards))
        if not candidates:
            return None
        hermitian: tuple[str, ...] = ()
        if not _complex_storage(bare):
            hermitian = tuple(
                n for n in stage_names
                if n != name_a
                and part_of[n]._hermitian)  # noqa: SLF001 — family classvar
        name_b = next(
            (n for n in candidates
             if any(m != n for m in hermitian)),
            candidates[0])
        remaining = tuple(n for n in hermitian if n != name_b)
        name_h = remaining[-1] if remaining else None
        return name_a, name_b, name_h, stage_names, part_of


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
