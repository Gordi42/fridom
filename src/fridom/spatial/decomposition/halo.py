"""
Halo machinery: negotiated ghost-layer widths and the halo trace.

Description
-----------
Owning class doc: ``design/specs/grid/classes/decomposition.md``.
``HaloSpec`` is the negotiated replacement of the global halo
integer: per-coordinate-name ghost widths, keyed by name because
names are the stable addressing scheme of the flat product. It is a
static, hashable value that enters jit cache keys through the
decomposition. ``HaloTracer`` and ``trace_halo`` are the automatic
halo-accounting trace: a data-free ``ScalarField`` stand-in dry-runs
the tendency, generically intercepted at the shared operator
application path (setup phase only, never inside jit).
"""
# Wave 1: HaloSpec -- Wave 3: HaloTracer, trace_halo
#    (GhostFill is designed-for)
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from fridom.spatial.operators.base import (
    Composite,
    Dispatched,
    SeparableOperator,
    _resolve_axis,
    resolve_codomain,
)
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
    join,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator, Mapping

    from fridom.spatial.operators.base import Operator
    from fridom.spatial.spaces.tensor_product import SpaceLike


@dataclass(frozen=True, init=False)
class HaloSpec:

    """
    Negotiated ghost-layer widths, one per coordinate name.

    Description
    -----------
    Storage is a sorted ``tuple[tuple[str, int], ...]``, not a
    mapping: frozen dataclasses used as jit-cache-key components must
    be hashable. The constructor accepts any ``Mapping[str, int]``
    and normalizes, so value-equal mappings produce equal (and
    equally hashing) specs. ``grow`` and ``merge_max`` are the two
    accumulation rules of the halo-accounting trace: sequential
    un-synced applications *add*, parallel expression branches *max*.

    Parameters
    ----------
    widths : Mapping[str, int]
        Per-coordinate-name ghost widths; all widths must be >= 0.
    """

    widths: tuple[tuple[str, int], ...]

    def __init__(self, widths: Mapping[str, int]) -> None:
        """Normalize the mapping to sorted tuple storage (hashable)."""
        items = []
        for name in sorted(widths):
            width = widths[name]
            if width < 0:
                raise ValueError(
                    f"halo width along {name!r} must be >= 0, "
                    f"got {width}")
            items.append((name, int(width)))
        object.__setattr__(self, "widths", tuple(items))

    @classmethod
    def zero(cls, names: tuple[str, ...]) -> HaloSpec:
        """
        Build a spec with width 0 on every name.

        Parameters
        ----------
        names : tuple[str, ...]
            The coordinate names the spec covers.

        Returns
        -------
        HaloSpec
            The all-zero spec over `names`.
        """
        return cls(dict.fromkeys(names, 0))

    def __getitem__(self, name: str) -> int:
        """
        Return the width along `name`.

        Parameters
        ----------
        name : str
            A coordinate name covered by this spec.

        Returns
        -------
        int
            The ghost width along `name`.

        Raises
        ------
        KeyError
            If `name` is not covered by this spec.
        """
        for key, width in self.widths:
            if key == name:
                return width
        raise KeyError(name)

    def grow(self, name: str, by: int) -> HaloSpec:
        """
        Return a new spec with `name` widened by `by`.

        Description
        -----------
        The *sequential* accumulation rule: un-synced composition
        chains add their per-operator widths.

        Parameters
        ----------
        name : str
            A coordinate name covered by this spec.
        by : int
            The additional width; must be >= 0.

        Returns
        -------
        HaloSpec
            The widened spec; `self` is unchanged.
        """
        if by < 0:
            raise ValueError(f"grow amount must be >= 0, got {by}")
        merged = dict(self.widths)
        merged[name] = self[name] + by
        return HaloSpec(merged)

    def consume(self, name: str, by: int) -> HaloSpec:
        """
        Return a new spec with `name` lowered by `by` (floor 0).

        Description
        -----------
        The *validity* counterpart of ``grow`` (task 1.8): a stencil
        application of reach ``by`` along ``name`` leaves ``by``
        fewer valid ghost layers on its result. Names other than
        `name` carry over.

        Parameters
        ----------
        name : str
            A coordinate name covered by this spec.
        by : int
            The consumed depth; must be >= 0.

        Returns
        -------
        HaloSpec
            The lowered spec; `self` is unchanged.
        """
        if by < 0:
            raise ValueError(
                f"consume amount must be >= 0, got {by}")
        merged = dict(self.widths)
        merged[name] = max(self[name] - by, 0)
        return HaloSpec(merged)

    def merge_max(self, other: HaloSpec) -> HaloSpec:
        """
        Return the pointwise maximum of two specs.

        Description
        -----------
        The *parallel* accumulation rule: independent tendency terms
        contribute their maximum, not their sum. The result covers
        the union of the two name sets; a name missing from one spec
        counts as width 0.

        Parameters
        ----------
        other : HaloSpec
            The spec to merge with.

        Returns
        -------
        HaloSpec
            The pointwise-maximum spec over the union of names.
        """
        merged = dict(self.widths)
        for name, width in other.widths:
            merged[name] = max(merged.get(name, 0), width)
        return HaloSpec(merged)

    def over(self, names: tuple[str, ...]) -> HaloSpec:
        """
        Return the spec restricted to exactly `names`.

        Description
        -----------
        Names missing from this spec count as width 0. Used to stamp
        a field's halo validity from the decomposition-wide
        negotiated widths (task 1.8): validity specs canonically
        cover exactly the field's space names.

        Parameters
        ----------
        names : tuple[str, ...]
            The coordinate names the result covers.

        Returns
        -------
        HaloSpec
            The restricted spec over `names`.
        """
        widths = dict(self.widths)
        return HaloSpec({name: widths.get(name, 0) for name in names})

    def merge_min(self, other: HaloSpec) -> HaloSpec:
        """
        Return the pointwise minimum of two specs.

        Description
        -----------
        The combination rule of halo *validity* (task 1.8): a field
        built from several operands can only claim ghost layers every
        operand had. The result covers the union of the two name
        sets; a name missing from one spec counts as width 0.

        Parameters
        ----------
        other : HaloSpec
            The spec to merge with.

        Returns
        -------
        HaloSpec
            The pointwise-minimum spec over the union of names.
        """
        mine = dict(self.widths)
        theirs = dict(other.widths)
        return HaloSpec({
            name: min(mine.get(name, 0), theirs.get(name, 0))
            for name in mine.keys() | theirs.keys()})


# ================================================================
#  Halo-accounting trace (HaloTracer, trace_halo)
# ================================================================
# Python scalars entering tracer arithmetic (mirrors ScalarField)
_SCALAR_TYPES = int | float | complex


class _TraceRecorder:

    """Accumulates the maximal observed halo depth of one trace."""

    def __init__(self) -> None:
        self.spec: HaloSpec = HaloSpec({})

    def observe(self, depth: HaloSpec) -> None:
        """Merge one observed accumulated depth into the maximum."""
        self.spec = self.spec.merge_max(depth)


class _TracerGrid:

    """
    Minimal grid stand-in a ``HaloTracer`` presents as ``.grid``.

    Description
    -----------
    Exposes exactly the registry (as ``dispatch``) so the seeded
    ``Dispatched`` verbs resolve against tracers; everything else a
    real grid offers is deliberately absent (tracers never reach the
    decomposition).

    ``chart_coords`` is the one geometry attribute term bodies may
    branch on at trace time, and it is always ``None`` here: a module
    whose grid *is* chart-coupled declares ``extra_halo`` and is
    halo-trace exempt, so only flat-gated bodies ever run against
    this stub — for them ``None`` is the true answer.
    """

    #: tracers only reach flat-gated module bodies (class docstring)
    chart_coords: None = None

    def __init__(self, dispatch: object) -> None:
        self.dispatch = dispatch


def _laid_out_like(space: SpaceLike, reference: SpaceLike) -> SpaceLike:
    """Mimic the application path's layout re-attachment."""
    if space.layout is not None:
        return space  # layout-transition operator (Reshard)
    if reference.layout is not None:
        return space.with_layout(reference.layout)
    return space


class HaloTracer:

    """
    Data-free ``ScalarField`` stand-in carrying space + halo depth.

    Description
    -----------
    Presents the iteration-1 ``ScalarField`` surface (arithmetic,
    ``.diff``, ``.to``, ``.grid``/``.function_space``) so operators
    run unchanged; the shared application path intercepts tracer
    operands through the ``_trace_apply`` / ``_trace_apply_nary``
    hooks, records ``requirements(domain).halo`` and returns the
    codomain-space tracer without touching kernel code. ``.data``
    raises ``TypeError``: a module that drops to raw arrays escapes
    the accounting, so the escape must be declared
    (``Module.extra_halo``, Phase 2). The trace mirrors the
    consumption-side runtime (task 1.8): each application records
    its accumulated depth — the recorder's maximum is the step's
    *sync-free* width demand — and the tracer's depth then follows
    the kernels' validity claims (accumulate along periodic stencil
    chains, reset at bounded stencils and every other re-store,
    which are free re-sync points at any width). Elementwise
    arithmetic also resets the traced depth — a deliberate
    under-negotiation: the runtime combine keeps the operands'
    minimum claim (storage-frame arithmetic, 2026-07-12), which
    elides downstream syncs wherever the negotiated width already
    covers them, but no *width demand* accrues through arithmetic
    (a sync there stays correct at any width, so the negotiation
    does not widen the halo for it).

    Parameters
    ----------
    function_space : SpaceLike
        The space this tracer pretends to live on.
    registry : object
        The (duck-typed) operator registry used by the mimicked
        dispatch surface (``.diff``, ``.to``, products).
    depth : HaloSpec | None, optional
        Accumulated per-name depth; None means zero on every name
        (default: None).
    """

    def __init__(
        self,
        function_space: SpaceLike,
        registry: object,
        depth: HaloSpec | None = None,
        *,
        recorder: _TraceRecorder | None = None,
    ) -> None:
        """Create a tracer on `function_space` (zero default depth)."""
        self._space = function_space
        self._registry = registry
        self._depth = (HaloSpec.zero(tuple(function_space.names))
                       if depth is None else depth)
        self._recorder = recorder

    # ================================================================
    #  Properties (the mimicked field surface)
    # ================================================================
    @property
    def function_space(self) -> SpaceLike:
        """The space this tracer pretends to live on."""
        return self._space

    @property
    def depth(self) -> HaloSpec:
        """Accumulated per-name halo depth since the last sync."""
        return self._depth

    @property
    def data(self) -> NoReturn:
        """Raise TypeError: bypasses must declare Module.extra_halo."""
        raise TypeError(
            "HaloTracer has no data: dropping to raw arrays escapes "
            "the halo accounting — declare the bypass via "
            "Module.extra_halo instead")

    @property
    def grid(self) -> _TracerGrid:
        """Registry-bearing grid stand-in (verb dispatch only)."""
        return _TracerGrid(self._registry)

    @property
    def shape(self) -> tuple[int, ...]:
        """Global true DOF shape, ``function_space.shape``."""
        return self._space.shape

    def __repr__(self) -> str:
        """Space and accumulated depth summary."""
        return (f"HaloTracer(space={self._space!r}, "
                f"depth={self._depth.widths})")

    # ================================================================
    #  Trace plumbing
    # ================================================================
    def _child(self, space: SpaceLike, depth: HaloSpec) -> HaloTracer:
        """Build the successor tracer (registry/recorder shared)."""
        return HaloTracer(space, self._registry, depth,
                          recorder=self._recorder)

    def _record(self, depth: HaloSpec) -> None:
        """Report one accumulated depth to the trace recorder."""
        if self._recorder is not None:
            self._recorder.observe(depth)

    def _grown(self, op: Operator) -> HaloSpec:
        """
        Compute the depth after `op`.

        Description
        -----------
        Per-axis for separable kernels; every bindable axis for
        whole-space operators (conservative).
        """
        space = self._space.bare
        if isinstance(op, SeparableOperator):
            axis = _resolve_axis(op, space)
            factor = (space if not hasattr(space, "factor")
                      else space.factor(axis))
            if isinstance(factor, ConstantSpace):
                return self._depth  # identity application
            return self._depth.grow(axis, op.requirements(factor).halo)
        halo = op.requirements(space).halo
        depth = self._depth
        if halo:
            for factor in space.factors:
                if factor.collapses_axis:
                    continue
                for name in factor.names:
                    depth = depth.grow(name, halo)
        return depth

    def _trace_apply(self, op: Operator) -> HaloTracer:
        """
        Unary interception hook of the shared application path.

        Description
        -----------
        Reshard-style operators declare ``_trace_reset_names``: a
        redistribute is at least as strong as a sync on the moved
        axes, so their accumulated depth resets (doc 04 section 5.1).
        ``Composite`` chains re-enter the hook factor by factor.
        Everything else records the grown depth — the recorder's
        maximum is the *sync-free* width demand of the step (task
        1.8) — and returns the codomain tracer carrying the depth
        the runtime kernel claims leave behind: accumulated on
        periodic axes (the kernel keeps ghost slots valid, chains
        elide), reset on a bounded applied axis (the kernel's output
        ghosts are not the BC-consistent fill, the runtime re-syncs
        there at any width, so no width demand accrues).
        """
        if isinstance(op, Composite):
            result: HaloTracer = self
            for factor in reversed(op.factors):
                result = factor(result)
            return result
        reset = getattr(op, "_trace_reset_names", None)
        if reset is not None:
            widths = dict(self._depth.widths)
            for name in reset(self._space):
                widths[name] = 0
            codomain = _laid_out_like(
                resolve_codomain(op, self._space), self._space)
            return self._child(codomain, HaloSpec(widths))
        grown = self._grown(op)
        self._record(grown)
        codomain = _laid_out_like(
            resolve_codomain(op, self._space), self._space)
        return self._child(codomain, self._claimed(op, grown))

    def _claimed(
        self, op: Operator, grown: HaloSpec,
    ) -> HaloSpec:
        """
        Return the depth the kernel's validity claim leaves behind.

        Description
        -----------
        Mirrors the stage-B construction-seam claims with depths
        measured from a (virtual) entry sync: consume-claiming
        kernels keep the accumulated depth; a bounded applied axis
        claims zero validity, which is a *free* re-sync point (the
        runtime refills there at any width), so its depth resets.
        Whole-space applications keep the conservative grown depth.
        """
        space = self._space.bare
        if not isinstance(op, SeparableOperator):
            return grown
        axis = _resolve_axis(op, space)
        factor = (space if not hasattr(space, "factor")
                  else space.factor(axis))
        if isinstance(factor, ConstantSpace):
            return grown
        if getattr(factor.mesh, "periodic", False):
            return grown
        widths = dict(grown.widths)
        if axis in widths and grown[axis] != self._depth[axis]:
            widths[axis] = 0  # the stencil consumed a bounded axis
        return HaloSpec(widths)

    def _trace_apply_nary(
        self, op: Operator, operands: tuple[object, ...],
    ) -> HaloTracer:
        """
        N-ary interception hook (binary/elementwise operators).

        Description
        -----------
        Depths of the tracer operands max-merge (parallel branches);
        real-field operands contribute zero. The codomain tracer
        keeps the merged depth (task 1.8): the storage-frame
        elementwise kernels claim the pointwise minimum of their
        operands' validity, which is the maximum of their depths.
        """
        depth = HaloSpec({})
        spaces = []
        for operand in operands:
            if isinstance(operand, HaloTracer):
                depth = depth.merge_max(operand._depth)  # noqa: SLF001
            spaces.append(operand.function_space)
        codomain = op.codomain(*(space.bare for space in spaces))
        halo = op.requirements(codomain).halo
        if halo:
            for factor in codomain.factors:
                if factor.collapses_axis:
                    continue
                for name in factor.names:
                    depth = depth.merge_max(
                        HaloSpec({name: 0})).grow(name, halo)
        self._record(depth)
        codomain = _laid_out_like(codomain, self._space)
        return self._child(codomain,
                           depth.over(tuple(codomain.names)))

    # ================================================================
    #  Dispatch sugar (mirrors ScalarField's thin forwarders)
    # ================================================================
    def diff(self, name: str) -> HaloTracer:
        """Default derivative along ``name`` (traced generically)."""
        return Dispatched("diff")[name](self)

    def to(self, target: object) -> HaloTracer:
        """
        Convert per axis onto the target's space (traced).

        Description
        -----------
        Mirrors the eager ``ScalarField.to`` exactly, including the
        BC-sibling seam: where the registered operator's codomain is
        a BC-sibling of the requested factor (nodal operator outputs
        are BC-free; owner decision), the eager path adopts the
        requested tag via ``retag`` — so the tracer relabels its
        space the same way (``_retag_factor``), keeping the traced
        space identical to the runtime one. Any other codomain
        disagreement raises, as it does eagerly.
        """
        from fridom.spatial.fields.scalar_field import (  # noqa: PLC0415 — fields import the operator base
            _bc_siblings,
            _conversion_kind,
            _target_space,
        )
        space = _target_space(self._space, target)
        src_bare = self._space.bare
        dst_bare = space.bare
        if dst_bare is src_bare:
            return self
        result: HaloTracer = self
        for name in dst_bare.names:
            src = result.function_space.bare.factor(name)
            dst = dst_bare.factor(name)
            if src is dst:
                continue
            if isinstance(src, ConstantSpace):
                result = result._broadcast_factor(name, dst)
                continue
            op = self._registry.resolve(
                _conversion_kind(src, dst), src)[name]
            resolved = resolve_codomain(
                op, result.function_space).factor(name)
            if resolved is not dst:
                if _bc_siblings(resolved, dst):
                    # the eager path retags onto the requested
                    # sibling; mirror it so the traced space agrees
                    applied: HaloTracer = op(result)
                    result = applied._retag_factor(name, dst)
                    continue
                from fridom.spatial.errors import (  # noqa: PLC0415 — keep errors off the module import path
                    SpaceMismatchError,
                )
                raise SpaceMismatchError(
                    f"the registered operator lands on {resolved!r},"
                    f" not the requested {dst!r}; use an explicit "
                    "operator instance or a registry override",
                    left=src, right=dst, operation="to")
            result = op(result)
        return result

    def retag(self, target: object) -> HaloTracer:
        """
        Relabel onto a BC-sibling space (traced ``retag``).

        Description
        -----------
        Mirrors the eager ``ScalarField.retag`` exactly: per axis the
        target factor must differ from the source factor **only** in
        BC structure — same mesh, same node-set class, same shape,
        same scalars — anything else raises ``SpaceMismatchError``.
        The accumulated depth resets on the retagged axes (the eager
        ``retag`` resets halo validity there — the ghost policy
        changed with the tag, a free re-sync point) and carries over
        on the others, so a module's traced retag (e.g. adopting a
        velocity's wall tag onto a BC-free stencil output) keeps the
        traced space identical to the runtime one.

        Parameters
        ----------
        target : object
            A field/tracer, a full product space, or a single factor
            space (shorthand: retag that factor, keep the rest).

        Returns
        -------
        HaloTracer
            The retagged tracer (``self`` when already on target).
        """
        from fridom.spatial.errors import (  # noqa: PLC0415 — keep errors off the module import path
            SpaceMismatchError,
        )
        from fridom.spatial.fields.scalar_field import (  # noqa: PLC0415 — fields import the operator base
            _bc_siblings,
            _target_space,
        )
        space = _target_space(self._space, target)
        src_bare = self._space.bare
        dst_bare = space.bare
        if dst_bare is src_bare:
            return self
        if src_bare.names != dst_bare.names:
            raise SpaceMismatchError(
                f"cannot retag {src_bare!r} onto {dst_bare!r}: "
                "the coordinate names differ",
                left=src_bare, right=dst_bare, operation="retag")
        result: HaloTracer = self
        for name in dst_bare.names:
            src = src_bare.factor(name)
            dst = dst_bare.factor(name)
            if src is dst:
                continue
            if not _bc_siblings(src, dst):
                raise SpaceMismatchError(
                    f"retag changes BC structure only: at {name!r} "
                    f"the factors {src!r} and {dst!r} differ beyond "
                    "their BC tags (mesh, node-set class, shape and "
                    "scalars must match); use .to for a conversion",
                    left=src, right=dst, operation="retag",
                    mismatched_names=(name,))
            result = result._retag_factor(name, dst)
        return result

    def _retag_factor(self, name: str, dst: SpaceLike) -> HaloTracer:
        """
        Trace-side twin of the eager BC-sibling ``retag`` adoption.

        Description
        -----------
        Replaces the factor ``name`` with its requested BC-sibling
        ``dst`` and resets the accumulated depth along that axis:
        the eager ``retag`` resets halo validity on retagged axes
        (the ghost policy changed with the tag), which is a free
        re-sync point — no width demand accrues, mirroring the
        bounded-axis reset of ``_claimed``.
        """
        bare = self._space.bare
        if isinstance(bare, TensorProductSpace):
            target = bare.replace(**{name: dst})
        else:
            target = dst
        target = _laid_out_like(target, self._space)
        widths = dict(self._depth.over(tuple(target.names)).widths)
        widths[name] = 0
        return self._child(target, HaloSpec(widths))

    def _broadcast_factor(
        self, name: str, dst: SpaceLike,
    ) -> HaloTracer:
        """
        Traced constant broadcast of one factor (``.to`` broadcast row).

        Description
        -----------
        The trace-side twin of the eager ``_broadcast_factor``: replace
        the ``ConstantSpace`` factor ``name`` with ``dst`` and carry the
        depth over. A constant broadcast is halo 0, so no ghost demand
        accrues and nothing is recorded.
        """
        bare = self._space.bare
        if isinstance(bare, TensorProductSpace):
            target = bare.replace(**{name: dst})
        else:
            target = dst
        target = _laid_out_like(target, self._space)
        return self._child(
            target, self._depth.over(tuple(target.names)))

    def integrate(self, *names: str) -> HaloTracer:
        """Weighted integral; mirrors the ScalarField stub."""
        raise NotImplementedError(
            "f.integrate forwards to the seeded 'integrate' verb "
            "once the Integral operator rows land in Wave 3")

    # ================================================================
    #  Arithmetic (mixed-operand reflected ops survive by design:
    #  ScalarField dunders return NotImplemented on tracers)
    # ================================================================
    def _linear(self, other: object) -> HaloTracer:
        """
        Join spaces; the result's depth resets (for +/-).

        Description
        -----------
        Deliberate under-negotiation (class docstring): the runtime
        ``+``/``-`` keeps the operands' minimum ghost claim
        (storage-frame combine), but a consumer syncing after
        arithmetic is correct at any width, so no width demand
        accrues through it — the traced depth resets and the
        negotiation never widens the halo for arithmetic chains.
        """
        if isinstance(other, HaloTracer):
            joined = join(self._space.bare, other._space.bare)  # noqa: SLF001
        elif isinstance(other, _SCALAR_TYPES):
            joined = self._space.bare
        else:
            space = getattr(other, "function_space", None)
            if space is None:
                return NotImplemented
            joined = join(self._space.bare, space.bare)
        return self._child(
            _laid_out_like(joined, self._space),
            HaloSpec.zero(tuple(joined.names)))

    def _reset_child(self) -> HaloTracer:
        """Return a same-space, zero-depth child (under-negotiation)."""
        return self._child(
            self._space, HaloSpec.zero(tuple(self._space.names)))

    def _dispatch_product(
        self, other: object, kind: str, *, reflected: bool = False,
    ) -> HaloTracer:
        """
        Resolve a product kind on the join and trace through it.

        Description
        -----------
        Mirrors the eager dunder (scalar_field ``_dispatched_product``):
        both operands are lifted onto the join *before* dispatch, so the
        product operator's codomain resolver sees the pre-lifted common
        space — a ``ConstantSpace`` operand (an ``fr.Profile()`` field)
        broadcasts to the full factor exactly as it does eagerly. The
        lift is a constant broadcast: halo 0 (it reads the single DOF and
        adds no ghost demand), so the traced depth carries over unchanged.
        """
        if isinstance(other, HaloTracer):
            other_space = other._space  # noqa: SLF001
        elif isinstance(other, _SCALAR_TYPES):
            other_space = self._space
            other = self._child(
                self._space,
                HaloSpec.zero(tuple(self._space.names)))
        else:
            other_space = getattr(other, "function_space", None)
            if other_space is None:
                return NotImplemented
        joined = _laid_out_like(
            join(self._space.bare, other_space.bare), self._space)
        op = self._registry.resolve(kind, joined)
        left = self._broadcast_to(self, joined)
        right = self._broadcast_to(other, joined)
        return op(right, left) if reflected else op(left, right)

    def _broadcast_to(
        self, operand: object, joined: SpaceLike,
    ) -> object:
        """
        Lift a tracer/field operand onto ``joined`` (constant broadcast).

        Description
        -----------
        The trace-side twin of the eager ``_lift_field``: the sanctioned
        constant broadcast (rules 3.3) changes only the space (a
        ``ConstantSpace`` factor becomes the full factor), never the
        ghost demand. Tracer operands carry their depth over unchanged
        (halo 0); a real captured field operand (an ``fr.Profile()``
        constant) reuses the eager broadcast so eager and traced agree.
        """
        space = operand.function_space
        if space is joined:
            return operand
        if isinstance(operand, HaloTracer):
            return operand._child(  # noqa: SLF001
                joined,
                operand._depth.over(tuple(joined.names)))  # noqa: SLF001
        from fridom.spatial.fields.scalar_field import (  # noqa: PLC0415 — trace-side reuse of the eager broadcast
            _lift_field,
        )
        return _lift_field(operand, joined)

    def __add__(self, other: object) -> HaloTracer:
        """Linear; join rule, depths max-merge."""
        return self._linear(other)

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__

    def __neg__(self) -> HaloTracer:
        """Negation: depth resets (deliberate under-negotiation)."""
        return self._reset_child()

    def __pos__(self) -> HaloTracer:
        """Identity."""
        return self

    def __mul__(self, other: object) -> HaloTracer:
        """Scalar: depth resets (conservative). Field/tracer: product."""
        if isinstance(other, _SCALAR_TYPES):
            return self._reset_child()
        return self._dispatch_product(other, "multiply")

    def __rmul__(self, other: object) -> HaloTracer:
        """Reflected multiply (field * tracer)."""
        if isinstance(other, _SCALAR_TYPES):
            return self._reset_child()
        return self._dispatch_product(other, "multiply",
                                      reflected=True)

    def __truediv__(self, other: object) -> HaloTracer:
        """Scalar: depth resets (conservative). Field/tracer: divide."""
        if isinstance(other, _SCALAR_TYPES):
            return self._reset_child()
        return self._dispatch_product(other, "divide")

    def __rtruediv__(self, other: object) -> HaloTracer:
        """Reflected divide (scalar or field over the tracer)."""
        return self._dispatch_product(other, "divide", reflected=True)

    def __pow__(self, exponent: float) -> HaloTracer:
        """Physical power through the ("power", space) row."""
        if not isinstance(exponent, int | float):
            return NotImplemented
        return self._dispatch_product(exponent, "power")

    def __abs__(self) -> HaloTracer:
        """Pointwise modulus through the ("abs", space) row."""
        op = self._registry.resolve("abs", self._space.bare)
        return op(self)


class VectorTracer:

    """
    ``VectorField``/State-mimicking stand-in over tracer components.

    Description
    -----------
    Performs **no grid validation** — component mapping (``.map``)
    and composed operators trace through it. Built by ``trace_halo``;
    arithmetic and functional updates mirror the ``VectorField``
    surface componentwise.

    Parameters
    ----------
    components : Mapping[str, HaloTracer] | Iterable[HaloTracer]
        The component tracers; unnamed iterables are keyed
        positionally (``c0``, ``c1``, ...).
    """

    def __init__(
        self,
        components: Mapping[str, HaloTracer] | tuple[HaloTracer, ...],
    ) -> None:
        """Normalize the components to a name-keyed mapping."""
        if hasattr(components, "items"):
            self._components = dict(components.items())
        else:
            self._components = {
                f"c{i}": tracer
                for i, tracer in enumerate(components)}

    @property
    def components(self) -> Mapping[str, HaloTracer]:
        """Read-only name -> tracer view, in declaration order."""
        return dict(self._components)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in declaration order."""
        return tuple(self._components)

    def __getitem__(self, key: str | int) -> HaloTracer:
        """Component by name or positional index."""
        if isinstance(key, int):
            return tuple(self._components.values())[key]
        return self._components[key]

    def __iter__(self) -> Iterator[HaloTracer]:
        """Iterate over component tracers in declaration order."""
        return iter(self._components.values())

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self._components)

    def __contains__(self, name: str) -> bool:
        """Whether a component of that name exists."""
        return name in self._components

    def map(
        self, fn: Callable[[HaloTracer], HaloTracer],
    ) -> VectorTracer:
        """Apply ``fn`` to each component on its own space."""
        return VectorTracer({name: fn(tracer)
                             for name, tracer in
                             self._components.items()})

    def replace(self, **components: HaloTracer) -> VectorTracer:
        """Functional update of named components."""
        merged = dict(self._components)
        for name, tracer in components.items():
            if name not in merged:
                raise KeyError(name)
            merged[name] = tracer
        return VectorTracer(merged)

    def _combine(self, other: object) -> VectorTracer:
        """Componentwise linear combination (depths max-merge)."""
        if isinstance(other, VectorTracer):
            return VectorTracer({
                name: tracer + other[name]
                for name, tracer in self._components.items()})
        return VectorTracer({name: tracer + other
                             for name, tracer in
                             self._components.items()})

    def __add__(self, other: object) -> VectorTracer:
        """Componentwise addition."""
        return self._combine(other)

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__

    def __neg__(self) -> VectorTracer:
        """Negation: componentwise (spaces and depths unchanged)."""
        return self

    def __pos__(self) -> VectorTracer:
        """Identity."""
        return self

    def __mul__(self, other: object) -> VectorTracer:
        """Componentwise scaling/product."""
        return VectorTracer({name: tracer * other
                             for name, tracer in
                             self._components.items()})

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> VectorTracer:
        """Componentwise division."""
        return VectorTracer({name: tracer / other
                             for name, tracer in
                             self._components.items()})


def trace_halo(
    tendency: Callable[..., object],
    state_spaces: Mapping[str, SpaceLike] | tuple[SpaceLike, ...],
    registry: object,
) -> HaloSpec:
    """
    Dry-run the tendency on tracers; return the max accumulated depth.

    Description
    -----------
    Feeds the tendency a ``VectorTracer`` whose components are
    ``HaloTracer``s on `state_spaces`. A name-keyed mapping
    propagates its names into the components, so traced demands and
    error messages are name-addressed (the model layer's
    name-addressed tendencies); a positional sequence keys them
    ``c0``, ``c1``, ..., and a lone positional space is passed as
    its bare tracer. The tendency must be traceable **without** jit
    — plain Python over fields. The accounting runs over `registry`
    as merged, so overrides are honored; under the iteration-1
    sync-after-every-operator contract the result reproduces the
    per-operator maximum, while the tracer's grow/merge rules carry
    the designed-for sync-elision semantics.

    Parameters
    ----------
    tendency : Callable[..., object]
        The tendency callable (takes the state stand-in).
    state_spaces : Mapping[str, SpaceLike] | tuple[SpaceLike, ...]
        The spaces of the model's state fields, name-keyed or
        positional.
    registry : object
        The (duck-typed) operator registry, as merged.

    Returns
    -------
    HaloSpec
        The maximal accumulated per-name depth over the trace.
    """
    if not state_spaces:
        raise ValueError("trace_halo needs at least one state space")
    recorder = _TraceRecorder()
    if hasattr(state_spaces, "items"):
        state: HaloTracer | VectorTracer = VectorTracer({
            name: HaloTracer(space, registry, recorder=recorder)
            for name, space in state_spaces.items()})
    else:
        tracers = tuple(
            HaloTracer(space, registry, recorder=recorder)
            for space in state_spaces)
        state = (tracers[0] if len(tracers) == 1
                 else VectorTracer(tracers))
    tendency(state)
    return recorder.spec
