"""
The ``Transform`` ABC: grid-bound changes of representation.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_transforms.md``
(planner rules: section 5.1 of ``04_decomposition.md``). Transforms
are the deliberate exception to grid-freedom: they bind the grid at
construction (plans, layouts, refined meshes are static structure)
and are applied through ``forward``/``backward``; the inherited
``__call__`` (via ``_apply`` delegating to ``forward``) exists for
registry uniformity only.

The multi-axis schedule is a **static plan object**
(``TransformPlan``): ``axes`` is a set, and the planner is free to
reorder the 1D stages for speed. The deterministic **single-device
convention implemented here is grid coordinate order** (first-listed
in ``grid.names``); on one device every axis is local, so no reshard
stages exist and the codomain keeps the operand's layout. The
multi-device planner slots into ``forward_plan``/``backward_plan``
later without changing the plan's shape. For real fields the first
stage of the forward schedule is the real-to-complex one: its factor
keeps the real origin and the Hermitian half spectrum; every later
stage targets the complexified origin (full spectrum). The backward
schedule runs the complex stages first and the half-spectrum stage
last.
"""
# Wave 3: Transform (ABC), TransformStage, TransformPlan
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp

from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.errors import (
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.spatial.fields.storage import storage_dtype, store
from fridom.spatial.operators.base import (
    FieldLike,
    OperatorRequirements,
    UnaryOperator,
)
from fridom.spatial.operators.dealias import PadFactor
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import CellAvg, FaceAvg
from fridom.spatial.spaces.coefficient import (
    CoefficientSpace,
    FourierSpace,
)
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.function_space import FunctionSpace
from fridom.spatial.spaces.nodal import NodalSpace
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import jax

    from fridom.spatial.meshes.mesh import Mesh
    from fridom.spatial.spaces.tensor_product import SpaceLike


#: a distributed (slab) plan needs a sharded stage axis and a
#: transpose partner — the minimum multi-axis transform
_MIN_STAGES = 2


def _paddable(n: int, shards: int) -> bool:
    """
    Whether an extent shards over ``shards`` devices via padding.

    Description
    -----------
    The padded balanced all-to-all pads a split axis to
    ``shards * ceil(n / shards)`` before the transpose; the last shard
    must keep at least one true slot (heavy padding that would empty a
    trailing shard is rejected, mirroring the decomposition's
    ``_cells_per_shard`` and the negotiation shardability guard). An
    extent already divisible by ``shards`` is trivially paddable.

    Parameters
    ----------
    n : int
        The (nodal) extent along the axis.
    shards : int
        The device count.

    Returns
    -------
    bool
        True iff the extent pads to ``shards`` non-empty blocks.
    """
    cells = -(-n // shards)
    return (shards - 1) * cells < n


# ================================================================
#  Axis-generic array helpers (shared by the transform kernels)
# ================================================================
def axis_slice(arr: jax.Array, axis: int, start: int,
               stop: int) -> jax.Array:
    """
    Slice ``arr[start:stop]`` along one axis.

    Parameters
    ----------
    arr : jax.Array
        The array to slice.
    axis : int
        The array axis to slice along.
    start : int
        Slice start (static).
    stop : int
        Slice stop (static).

    Returns
    -------
    jax.Array
        The sliced array.
    """
    index: list[slice] = [slice(None)] * arr.ndim
    index[axis] = slice(start, stop)
    return arr[tuple(index)]


def axis_concat(parts: Sequence[jax.Array], axis: int) -> jax.Array:
    """
    Concatenate array parts along one axis.

    Parameters
    ----------
    parts : Sequence[jax.Array]
        The (dtype-promoted) parts to join.
    axis : int
        The array axis to join along.

    Returns
    -------
    jax.Array
        The concatenated array.
    """
    return jnp.concatenate(parts, axis=axis)


def axis_zeros(arr: jax.Array, axis: int, count: int) -> jax.Array:
    """
    Zeros shaped like ``arr`` with ``count`` slots along ``axis``.

    Parameters
    ----------
    arr : jax.Array
        The template array (shape and dtype source).
    axis : int
        The array axis whose extent is replaced.
    count : int
        The extent of the zeros along ``axis``.

    Returns
    -------
    jax.Array
        The zero-filled array.
    """
    shape = list(arr.shape)
    shape[axis] = count
    return jnp.zeros(shape, arr.dtype)


def axis_vector(values: jax.Array, ndim: int,
                axis: int) -> jax.Array:
    """
    Reshape a 1D vector to broadcast along one array axis.

    Parameters
    ----------
    values : jax.Array
        The 1D vector (e.g. per-mode weights).
    ndim : int
        The rank of the array the vector multiplies.
    axis : int
        The array axis the vector varies along.

    Returns
    -------
    jax.Array
        The reshaped vector (singleton on every other axis).
    """
    shape = [1] * ndim
    shape[axis] = values.shape[0]
    return values.reshape(shape)


def embed_tail(arr: jax.Array, axis: int, count: int) -> jax.Array:
    """
    Zero-pad ``arr`` at the tail of ``axis`` up to ``count`` slots.

    Description
    -----------
    The generic coefficient embedding of padded sine/cosine/Chebyshev
    backward transforms: modes are stored ascending, so refining adds
    zero modes at the end.

    Parameters
    ----------
    arr : jax.Array
        The coefficient array.
    axis : int
        The mode axis.
    count : int
        The target extent (>= the current extent).

    Returns
    -------
    jax.Array
        The embedded array (``arr`` itself when nothing is added).
    """
    missing = count - arr.shape[axis]
    if missing == 0:
        return arr
    return axis_concat((arr, axis_zeros(arr, axis, missing)), axis)


# ================================================================
#  The static plan objects
# ================================================================
@dataclass(frozen=True)
class TransformStage:

    """
    One 1D stage of a planned multi-axis transform.

    Description
    -----------
    A static record consumed by the concrete kernels: which axis to
    transform, whether this is the Hermitian real-to-complex stage,
    and the two factor spaces bracketing the stage. ``nodal`` is the
    stage's nodal-side factor — the domain factor on ``forward``
    (the *refined* origin for padded transforms) and the target
    factor on ``backward``; ``coeff`` is always the (coarse)
    coefficient factor.

    Parameters
    ----------
    axis : str
        The coordinate name of the stage.
    index : int
        The array axis of the coordinate in the plan's domain.
    half : bool
        Whether this is the Hermitian half-spectrum (rfft) stage.
    nodal : FunctionSpace
        The nodal-side factor of the stage.
    coeff : FunctionSpace
        The (coarse) coefficient factor of the stage.
    layout : Layout | None, optional
        The device layout this stage executes under (the multi-device
        planner's per-stage pencil); None on the single-device plan
        (every axis local, no reshard stages) — default: None.
    """

    axis: str
    index: int
    half: bool
    nodal: FunctionSpace
    coeff: FunctionSpace
    layout: Layout | None = None


@dataclass(frozen=True)
class TransformPlan:

    """
    The static schedule of one transform application.

    Description
    -----------
    Planner output (section 5.1): the stages, in execution order, and
    the resolved bare codomain. The single-device planner orders the
    forward stages in **grid coordinate order** (first-listed
    convention, documented in the module docstring), which fixes the
    half-spectrum factor of multi-axis real transforms; the backward
    schedule is the reverse with the half-spectrum stage last. On one
    device all axes are local, so the plan carries no reshard stages;
    the multi-device planner extends this object without changing the
    transforms.

    Parameters
    ----------
    domain : SpaceLike
        The bare domain the plan was built for.
    codomain : SpaceLike
        The resolved bare codomain.
    stages : tuple[TransformStage, ...]
        The 1D stages, in execution order.
    """

    domain: SpaceLike
    codomain: SpaceLike
    stages: tuple[TransformStage, ...]


# ================================================================
#  The Transform ABC
# ================================================================
class Transform(UnaryOperator, ABC):

    """
    Grid-bound change of representation: nodal <-> coefficient.

    Description
    -----------
    The deliberate exception to grid-freedom (rules section 2.5):
    transforms bind the grid at construction because they need the
    decomposition and the transform plan up front, and they are
    applied through ``forward``/``backward``. ``codomain`` maps each
    named-axis factor to its **per-origin** coefficient space (the
    origin is constitutive, section 3.2), so ``backward`` needs no
    target argument. With ``pad`` set, ``backward`` lands in the
    finer nodal space on the ``refined(pad.factor)`` mesh and
    ``forward`` takes that finer space and trims back to the
    **coarse** coefficient space (the documented padded-``forward``
    codomain exception). The inherited ``__call__`` delegates to
    ``forward`` for registry uniformity (deviation recorded in the
    class doc); user code writes ``forward``/``backward``.

    Parameters
    ----------
    grid : Grid
        The grid to bind (decomposition + plans).
    axes : tuple[str, ...] | str | None, optional
        Coordinate names to transform along, a *set* the planner
        orders (default: None, all grid coordinates).
    pad : PadFactor | None, optional
        Dealiasing pad factor, or None for the plain transform
        (default: None).
    """

    dispatch_kind: ClassVar[str | None] = "transform"

    #: whether the transform family is real-to-complex (rfft-style):
    #: the first forward stage on real data keeps the real origin and
    #: produces the Hermitian half spectrum (Fourier only)
    _hermitian: ClassVar[bool] = False

    #: the coefficient-space family of this transform's codomains
    _space_family: ClassVar[type[CoefficientSpace]] = CoefficientSpace

    def __init__(
        self,
        grid: object,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid; transform along ``axes`` (default: all)."""
        names = grid.names
        if axes is None:
            requested: tuple[str, ...] = tuple(names)
        elif isinstance(axes, str):
            requested = (axes,)
        else:
            requested = tuple(axes)
        for axis in requested:
            if axis not in names:
                raise ValueError(
                    f"unknown transform axis {axis!r}; the grid's "
                    f"coordinate names are {names}")
        if len(set(requested)) != len(requested):
            raise ValueError(
                f"duplicate transform axes in {requested}")
        if pad is not None and not isinstance(pad, PadFactor):
            raise TypeError(
                "pad= takes a PadFactor (fr.operators.dealias."
                f"degree(p)), got {pad!r}")
        self._grid: object = grid
        # planner normalization: axes are a set; the deterministic
        # single-device stage order is grid coordinate order
        self._axes: tuple[str, ...] = tuple(
            name for name in names if name in requested)
        self._pad: PadFactor | None = pad
        self._fine_meshes: dict[str, Mesh] = {}
        if pad is not None:
            for axis in self._axes:
                mesh = next(m for m in grid.factors
                            if axis in m.names)
                self._fine_meshes[axis] = mesh.refined(pad.factor)
        #: memoized static plans, keyed on the interned bare domain
        self._plans: dict[tuple[str, SpaceLike], TransformPlan] = {}
        #: memoized distributed (layout-annotated) plans; stores None
        #: for ineligible spaces, so membership (not ``.get``) is the
        #: cache test
        self._dist_plans: dict[
            tuple[str, SpaceLike], TransformPlan | None] = {}

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def grid(self) -> object:
        """The bound grid (decomposition + plans)."""
        return self._grid

    @property
    def axes(self) -> tuple[str, ...]:
        """Coordinate names this transform acts along.

        Normalized to grid coordinate order (the planner's
        deterministic single-device stage order).
        """
        return self._axes

    @property
    def pad(self) -> PadFactor | None:
        """Dealiasing pad factor, or None for the plain transform."""
        return self._pad

    # ================================================================
    #  Application
    # ================================================================
    def forward(self, f: FieldLike) -> FieldLike:
        """
        Nodal/average -> coefficient (trims if padded).

        Parameters
        ----------
        f : FieldLike
            The field on the (per-origin) domain space; on a padded
            transform, on the refined mesh's finer nodal space.

        Returns
        -------
        FieldLike
            The coefficient-space field (metadata preserved: a
            transform is a representation change, not a new
            quantity).
        """
        self._check_grid(f, "forward")
        self._reject_sharded_transform(f, "forward")
        self._reject_replicating_transform(f, "forward")
        plan = self.forward_plan(f.function_space)
        data = jnp.asarray(f.data)
        fused = self._forward_fused_kernel(data, plan)
        if fused is not None:
            data = fused
        else:
            for stage in plan.stages:
                data = self._forward_kernel(data, stage)
        data = self._project(data, plan.codomain)
        return self._deliver(f, data, plan.codomain)

    def backward(self, f: FieldLike) -> FieldLike:
        """
        Coefficient -> origin nodal space (padded if ``pad``).

        Parameters
        ----------
        f : FieldLike
            The field on this transform family's coefficient space.

        Returns
        -------
        FieldLike
            The nodal-space field (on the refined mesh's finer nodal
            space when padded; metadata preserved).
        """
        self._check_grid(f, "backward")
        self._reject_sharded_transform(f, "backward")
        self._reject_replicating_transform(f, "backward")
        plan = self.backward_plan(f.function_space)
        data = jnp.asarray(f.data)
        fused = self._backward_fused_kernel(data, plan)
        if fused is not None:
            data = fused
        else:
            for stage in plan.stages:
                data = self._backward_kernel(data, stage)
        return self._deliver(f, data, plan.codomain)

    def _apply(self, f: FieldLike) -> FieldLike:
        """Delegate to ``forward`` (registry-uniform application)."""
        return self.forward(f)

    def apply_diagonal(
        self,
        f: FieldLike,
        symbol_factory: object,
    ) -> FieldLike:
        r"""
        Fused ``backward(symbol(forward(f)))`` -- distributed-safe.

        Description
        -----------
        The consumer surface for a spectral operator applied in the
        nodal frame: a forward transform, a per-mode diagonal ``symbol``
        multiply, and a backward transform, as **one** operation. On a
        grid whose default layout shards a transform axis the naive
        ``backward(symbol(forward(f)))`` re-gathers (the standalone
        ``forward`` materializes a coefficient field, which the storage
        contract replicates -- the Tier-1 taught error), so this routes
        through the fused ``jax.shard_map`` lowering
        (``distributed_transform.DistributedTransform``) whenever
        ``resolve_distributed_transform`` serves the space: the
        coefficient frame stays internal and no axis is gathered. On a
        single device (or a replicated operand) it runs the plain
        sandwich, bit-for-bit unchanged.

        Because the two frames differ (the single-device codomain halves
        the first axis; the distributed internal frame halves the local
        Hermitian axis, or runs fully complex), the symbol is built
        **per frame** by ``symbol_factory``: it receives the bare
        coefficient space the multiply runs on and returns the matching
        :class:`~fridom.spatial.operators.symbol.Symbol`
        (``lambda coeff_bare: op.eigenvalues(grid, coeff_bare)``). The
        symbol must be endo on that frame (its codomain equals its
        domain), so the backward returns to the operand's own nodal
        space.

        Parameters
        ----------
        f : FieldLike
            The nodal operand.
        symbol_factory : object
            A callable ``coeff_bare -> Symbol`` building the diagonal on
            the coefficient frame the multiply runs on.

        Returns
        -------
        FieldLike
            The nodal result on the operand's own layout.

        Raises
        ------
        NotImplementedError
            If the distributed route resolves but the symbol retags
            (no layout-preserving distributed form).
        """
        self._check_grid(f, "apply_diagonal")
        dt = self._distributed_diagonal_route(f)
        if dt is not None:
            symbol = symbol_factory(dt.coeff.bare)
            if symbol.codomain is not symbol.space:
                raise NotImplementedError(
                    f"{type(self).__name__}.apply_diagonal cannot run a "
                    "retagging symbol on a sharded transform axis: the "
                    "distributed fused route is layout-preserving and "
                    "needs an endo diagonal (codomain == domain). "
                    f"Got {symbol.space!r} -> {symbol.codomain!r}.")
            return dt.apply_diagonal(f, symbol.data)
        coeff = self.forward(f)
        symbol = symbol_factory(coeff.function_space.bare)
        return self.backward(symbol(coeff))

    def _distributed_diagonal_route(
        self, f: FieldLike,
    ) -> object | None:
        """
        Resolve the distributed fused route for ``f``, or None.

        Description
        -----------
        Returns the
        :class:`~fridom.spatial.operators.distributed_transform.DistributedTransform`
        only when the operand's own layout actually shards one of this
        transform's stage axes **and** the space is servable; otherwise
        None (the caller keeps the plain, layout-preserving sandwich --
        a single-device or replicated operand needs no reshard).
        """
        layout = f.function_space.layout
        if layout is None:
            return None
        bare = f.function_space.bare
        if all(layout.is_local(axis)
               for axis in self._stage_axes(bare)):
            return None
        from fridom.spatial.operators.distributed_transform import (  # noqa: PLC0415 — deferred: distributed_transform imports this module
            resolve_distributed_transform,
        )
        return resolve_distributed_transform(self, self._grid, bare)

    # ================================================================
    #  Space resolution
    # ================================================================
    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """
        Forward target: the per-origin coefficient space.

        Parameters
        ----------
        domain : SpaceLike
            The bare domain space (the refined finer nodal space on
            padded transforms).

        Returns
        -------
        SpaceLike
            The bare coefficient codomain (always on the coarse
            meshes: the padded-``forward`` codomain exception).
        """
        return self.forward_plan(domain).codomain

    def backward_space(self, domain: SpaceLike) -> SpaceLike:
        """
        Backward target: origin space, refined by ``pad`` if set.

        Parameters
        ----------
        domain : SpaceLike
            The bare coefficient space.

        Returns
        -------
        SpaceLike
            The bare nodal target space.
        """
        return self.backward_plan(domain).codomain

    def requirements(
        self,
        domain: SpaceLike,  # noqa: ARG002 — uniform declaration
    ) -> OperatorRequirements:
        """
        Declare halo = 0, layout "transpose".

        Description
        -----------
        Distributed operation is transpose-based (jaxDecomp-style);
        the grid's negotiation reads this per factor. On a single
        device every axis is local and the declaration is trivially
        met.

        Parameters
        ----------
        domain : SpaceLike
            The (factor) space the transform is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=0, layout="transpose")

    # ================================================================
    #  Planning (static; memoized per interned bare domain)
    # ================================================================
    def forward_plan(self, domain: SpaceLike) -> TransformPlan:
        """
        Return the static forward schedule for ``domain``.

        Description
        -----------
        Stages run in grid coordinate order (single-device planner
        convention); for ``_hermitian`` transforms on real-storage
        domains the first stage is the real-to-complex one — its
        factor keeps the real origin (half spectrum), all later
        stages target ``as_complex()`` origins (full spectrum).
        ``ConstantSpace`` factors along transform axes contribute no
        stage (identity, rules section 3.3).

        Parameters
        ----------
        domain : SpaceLike
            The (possibly laid-out) domain space.

        Returns
        -------
        TransformPlan
            The memoized static plan.
        """
        bare = domain.bare
        cached = self._plans.get(("forward", bare))
        if cached is not None:
            return cached
        axes = self._stage_axes(bare)
        half_pending = self._hermitian and not _complex_storage(bare)
        stages: list[TransformStage] = []
        mapping: dict[str, FunctionSpace] = {}
        for axis in axes:
            factor = bare.factor(axis)
            origin = self._coarse_origin(factor, axis)
            half = half_pending
            half_pending = False
            coeff = self._coefficient_factor(origin, half=half)
            stages.append(TransformStage(
                axis=axis, index=bare.names.index(axis), half=half,
                nodal=factor, coeff=coeff))
            mapping[axis] = coeff
        plan = TransformPlan(bare, _rebuild(bare, mapping),
                             tuple(stages))
        self._plans[("forward", bare)] = plan
        return plan

    def backward_plan(self, domain: SpaceLike) -> TransformPlan:
        """
        Return the static backward schedule for ``domain``.

        Description
        -----------
        The reverse of the forward schedule: full-spectrum stages run
        first (reversed grid order), the half-spectrum stage — if any
        — last. When a half-spectrum factor is present the
        represented field is real, so every transformed factor lands
        on the ``as_real()`` variant of its origin (the honest
        codomain: the inverse of the Hermitian stage produces real
        data).

        Parameters
        ----------
        domain : SpaceLike
            The (possibly laid-out) coefficient space.

        Returns
        -------
        TransformPlan
            The memoized static plan.
        """
        bare = domain.bare
        cached = self._plans.get(("backward", bare))
        if cached is not None:
            return cached
        axes = self._stage_axes(bare)
        stages: list[TransformStage] = []
        halves = 0
        for axis in axes:
            coeff = bare.factor(axis)
            if not isinstance(coeff, self._space_family):
                raise SpaceMismatchError(
                    f"backward of {type(self).__name__} takes "
                    f"{self._space_family.__name__} factors along "
                    f"its axes, got {coeff!r} along {axis!r}",
                    left=bare, operation="backward")
            half = (isinstance(coeff, FourierSpace)
                    and coeff.scalars is Scalars.REAL)
            halves += half
            target = self._target_origin(coeff, axis)
            stages.append(TransformStage(
                axis=axis, index=bare.names.index(axis), half=half,
                nodal=target, coeff=coeff))
        if halves > 1:
            raise SpaceMismatchError(
                "a coefficient space carries at most one Hermitian "
                f"half-spectrum factor, got {halves} on {bare!r}",
                left=bare, operation="backward")
        if halves:
            stages = [TransformStage(
                axis=s.axis, index=s.index, half=s.half,
                nodal=s.nodal.as_real(), coeff=s.coeff)
                for s in stages]
        ordered = tuple(reversed(
            [s for s in stages if not s.half]))
        ordered += tuple(s for s in stages if s.half)
        mapping = {s.axis: s.nodal for s in stages}
        plan = TransformPlan(bare, _rebuild(bare, mapping), ordered)
        self._plans[("backward", bare)] = plan
        return plan

    # ================================================================
    #  Distributed planning (multi-device; layout-annotated stages)
    # ================================================================
    def distributed_forward_plan(
        self, domain: SpaceLike,
    ) -> TransformPlan | None:
        """
        Return the layout-annotated forward schedule, or None.

        Description
        -----------
        The multi-device schedule for a sharded operand (section 1.1 of
        ``design/plans/active/distributed_transform_plan.md``): the
        initially-local axes transform first under the operand's nodal
        layout, one reshard makes the sharded axis local and shards the
        transpose partner, and the sharded axis transforms last. Each
        stage carries the ``Layout`` it executes under; the codomain
        carries the final pencil. Returns None when the operand is
        single-device or ineligible (a padded transform, a non-1-D
        mesh, or a layout ``_distributed_geometry`` declines), in
        which case callers keep the replicated ``forward_plan`` path —
        leaving the single-device program unchanged.

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
        key = ("forward", bare)
        if key not in self._dist_plans:
            self._dist_plans[key] = self._build_distributed_forward(
                bare)
        return self._dist_plans[key]

    def distributed_backward_plan(
        self, domain: SpaceLike,
    ) -> TransformPlan | None:
        """
        Return the layout-annotated backward schedule, or None.

        Description
        -----------
        The mirror of :meth:`distributed_forward_plan`: the sharded-axis
        inverse stage runs first (operand sharded on the transpose
        partner), one reshard restores the nodal layout, and the
        remaining inverse stages (the Hermitian half stage last) run
        local. The per-stage layouts are identical to the forward plan's
        (each stage runs under the same pencil in both directions); the
        schedule is simply reversed. Returns None when ineligible.

        Parameters
        ----------
        domain : SpaceLike
            The (coefficient) operand space.

        Returns
        -------
        TransformPlan | None
            The distributed plan, or None when ineligible.
        """
        bare = domain.bare
        key = ("backward", bare)
        if key not in self._dist_plans:
            self._dist_plans[key] = self._build_distributed_backward(
                bare)
        return self._dist_plans[key]

    def _build_distributed_forward(
        self, bare: SpaceLike,
    ) -> TransformPlan | None:
        """Uncached build of the distributed forward plan (or None)."""
        decomposition = self._grid.decomposition
        if (self._pad is not None
                or getattr(decomposition, "device_count", 1) <= 1):
            return None
        geometry = self._distributed_geometry(bare)
        if geometry is None:
            return None
        name_a, name_b, name_h, stage_names = geometry
        default_layout, spectral_layout = self._distributed_layouts(
            name_b)
        # local axes first (rfft on the half axis), sharded axis last
        ordered = ((name_h,) if name_h is not None else ()) + tuple(
            n for n in stage_names if n not in (name_a, name_h))
        ordered += (name_a,)
        stages: list[TransformStage] = []
        mapping: dict[str, FunctionSpace] = {}
        for axis in ordered:
            factor = bare.factor(axis)
            origin = self._coarse_origin(factor, axis)
            half = axis == name_h
            coeff = self._coefficient_factor(origin, half=half)
            layout = (spectral_layout if axis == name_a
                      else default_layout)
            stages.append(TransformStage(
                axis=axis, index=bare.names.index(axis), half=half,
                nodal=factor, coeff=coeff, layout=layout))
            mapping[axis] = coeff
        codomain = _rebuild(bare, mapping).with_layout(spectral_layout)
        return TransformPlan(
            bare.with_layout(default_layout), codomain, tuple(stages))

    def _build_distributed_backward(
        self, coeff_bare: SpaceLike,
    ) -> TransformPlan | None:
        """
        Uncached build of the distributed backward plan (or None).

        Description
        -----------
        Recovers the nodal domain through the (unchanged) single-device
        ``backward_plan`` — its codomain is the nodal target — then
        reverses the distributed forward stages. The per-stage layouts
        carry over unchanged (a stage runs under the same pencil in both
        directions), so the reversed order alone flips the reshard.
        """
        nodal_bare = self.backward_plan(coeff_bare).codomain
        forward = self.distributed_forward_plan(nodal_bare)
        if forward is None:
            return None
        default_layout = self._grid.decomposition.default_layout
        stages = tuple(reversed(forward.stages))
        codomain = nodal_bare.with_layout(default_layout)
        return TransformPlan(forward.codomain, codomain, stages)

    def _distributed_geometry(
        self, bare: SpaceLike,
    ) -> tuple[str, str, str | None, tuple[str, ...]] | None:
        """
        Choose the slab axes ``(a, b, h, stage axes)``, or None.

        Description
        -----------
        The slab axis roles are self-contained here (the fused kernel
        in ``operators/distributed_solve.py`` consumes them): ``a`` is
        the single coordinate the default layout shards, ``b`` the
        transpose partner (the first other stage coordinate whose
        extent **divides** the device count — the byte-identical fast
        path — else the first **paddable** other stage coordinate, run
        through the padded balanced all-to-all), ``h`` the last
        remaining stage coordinate (the local Hermitian half axis of a
        real domain under a ``_hermitian`` family; None otherwise — the
        trig families are real-to-real and run their pipelines fully
        complex). None on a non-1-D or fewer-than-two-stage layout, or
        when ``a`` / the only partner would pad too heavily (a trailing
        shard empties); the padded / single-device guards live in the
        caller. Indivisible-but-paddable ``a`` and ``b`` are accepted
        (the padded transpose), so a prime domain keeps the distributed
        solve.
        """
        decomposition = self._grid.decomposition
        mapped = dict(decomposition.default_layout.device_axes)
        if len(mapped) != 1:
            return None
        (name_a,) = mapped
        stage_names = self._stage_axes(bare)
        if (len(stage_names) < _MIN_STAGES
                or name_a not in stage_names):
            return None
        if any(len(bare.factor(n).shape) != 1 for n in stage_names):
            return None
        shards = decomposition.device_count
        if not _paddable(bare.factor(name_a).shape[0], shards):
            return None
        partners = tuple(n for n in stage_names if n != name_a)
        # prefer a divisible partner (keeps the byte-identical fast
        # path); else the first paddable one (the padded transpose)
        name_b = next(
            (n for n in partners
             if bare.factor(n).shape[0] % shards == 0), None)
        if name_b is None:
            name_b = next(
                (n for n in partners
                 if _paddable(bare.factor(n).shape[0], shards)), None)
        if name_b is None:
            return None
        real = not _complex_storage(bare)
        local = tuple(n for n in stage_names
                      if n not in (name_a, name_b))
        name_h = (local[-1] if (real and local and self._hermitian)
                  else None)
        return name_a, name_b, name_h, stage_names

    def _distributed_layouts(
        self, name_b: str,
    ) -> tuple[Layout, Layout]:
        """
        Return the ``(nodal default, spectral pencil)`` layout pair.

        Description
        -----------
        The nodal layout is the decomposition's default (the operand's
        pencil); the spectral pencil shards the transpose partner
        ``name_b`` on the same device-mesh axis.
        """
        default_layout = self._grid.decomposition.default_layout
        axis_name = default_layout.device_axes[0][1]
        return default_layout, Layout({name_b: axis_name})

    # ================================================================
    #  Extension contract (subclass hooks)
    # ================================================================
    @abstractmethod
    def _coefficient_factor(self, origin: FunctionSpace, *,
                            half: bool) -> FunctionSpace:
        """
        Resolve the coefficient factor of one (coarse) origin.

        Parameters
        ----------
        origin : FunctionSpace
            The bare origin factor (validated by the subclass).
        half : bool
            Whether this is the Hermitian half-spectrum stage.

        Returns
        -------
        FunctionSpace
            The per-origin coefficient factor.
        """
        ...

    @abstractmethod
    def _forward_kernel(self, data: jax.Array,
                        stage: TransformStage) -> jax.Array:
        """
        Run one forward 1D stage on true-shape data.

        Parameters
        ----------
        data : jax.Array
            The true-shape array.
        stage : TransformStage
            The static stage record.

        Returns
        -------
        jax.Array
            The transformed (and trimmed, if padded) array.
        """
        ...

    @abstractmethod
    def _backward_kernel(self, data: jax.Array,
                         stage: TransformStage) -> jax.Array:
        """
        Run one backward 1D stage on true-shape data.

        Parameters
        ----------
        data : jax.Array
            The true-shape coefficient array.
        stage : TransformStage
            The static stage record.

        Returns
        -------
        jax.Array
            The synthesized (padded, if ``pad``) array.
        """
        ...

    def _forward_fused_kernel(
        self,
        data: jax.Array,  # noqa: ARG002 — hook signature
        plan: TransformPlan,  # noqa: ARG002 — hook signature
    ) -> jax.Array | None:
        """
        Return the whole-plan forward result, or None (default).

        Description
        -----------
        Fusion hook consulted by ``forward`` before the per-stage
        loop: a subclass may run the entire schedule in one call
        (``Fourier``'s all-axis ``rfftn``/``fftn``) and return the
        result; returning None runs the staged 1D kernels. A fused
        kernel must reproduce the staged result exactly — same
        layout, half-spectrum axis, dtype and normalization — up to
        floating-point rounding.

        Parameters
        ----------
        data : jax.Array
            The true-shape operand array.
        plan : TransformPlan
            The static forward schedule.

        Returns
        -------
        jax.Array | None
            The fully transformed array, or None to run the stages.
        """
        return None

    def _backward_fused_kernel(
        self,
        data: jax.Array,  # noqa: ARG002 — hook signature
        plan: TransformPlan,  # noqa: ARG002 — hook signature
    ) -> jax.Array | None:
        """
        Return the whole-plan backward result, or None (default).

        Description
        -----------
        The ``backward`` counterpart of ``_forward_fused_kernel``:
        one call for the entire schedule (``Fourier``'s all-axis
        ``irfftn``/``ifftn``), or None to run the staged 1D kernels.

        Parameters
        ----------
        data : jax.Array
            The true-shape coefficient array.
        plan : TransformPlan
            The static backward schedule.

        Returns
        -------
        jax.Array | None
            The fully synthesized array, or None to run the stages.
        """
        return None

    def _project(
        self,
        data: jax.Array,
        codomain: SpaceLike,  # noqa: ARG002 — hook signature
    ) -> jax.Array:
        """
        Post-forward value projection hook (identity by default).

        Parameters
        ----------
        data : jax.Array
            The forward-transformed true-shape array.
        codomain : SpaceLike
            The bare coefficient codomain.

        Returns
        -------
        jax.Array
            The projected array (``Fourier`` enforces the Hermitian
            value invariant at self-conjugate modes here).
        """
        return data

    # ================================================================
    #  Internal helpers
    # ================================================================
    def _check_grid(self, f: FieldLike, operation: str) -> None:
        """Reject fields created on a different grid."""
        if f.grid is not self._grid:
            raise GridMismatchError(
                "transforms are grid-bound; the operand was created "
                "on a different grid",
                left=self._grid, right=f.grid, operation=operation)

    def _reject_replicating_transform(self, f: FieldLike,
                                      operation: str) -> None:
        r"""
        Taught skip when a *non-transform* axis is sharded (Tier 2).

        Description
        -----------
        The sibling of :meth:`_reject_sharded_transform` (which owns
        Tier 1 — a sharded *stage* axis — and runs first). Here every
        stage axis is device-local but the operand shards at least one
        **non-transform** axis. The naive ``forward`` still materializes
        a coefficient field, and the storage contract
        (``decomposition.sharding`` in
        ``spatial.decomposition.tensor``) replicates every
        ``CoefficientSpace`` factor unconditionally, so GSPMD silently
        **all-gathers** the sharded non-stage axis onto every device:
        numerically correct, but the whole global array is materialized
        per device — the change of representation does not scale. This
        naive GSPMD transform path is illegal by design (the
        owner-approved campaign,
        ``design/research/gspmd_naive_transform_illegality.md``).

        Like the Tier-1 guard the predicate reads the **operand's own**
        function-space layout (not the grid default), so a deliberately
        gathered / replicated field passes — an *explicit*
        replicate-then-compute stays legal, only the *silent* gather is
        rejected. A ``device_ids=(0,)`` grid and a grid too small to
        shard both collapse to one device and are exempt; a sharded
        stage axis is Tier 1's domain (already raised), so this guard is
        reached only with every stage axis local. The check is host-side
        Python on static layout metadata — a plain ``if`` outside any
        traced value.

        Parameters
        ----------
        f : FieldLike
            The operand field (carries its function-space layout).
        operation : str
            The calling direction (``"forward"`` / ``"backward"``),
            for the error message.

        Raises
        ------
        NotImplementedError
            When the decomposition spans several devices, no stage axis
            is sharded, and the operand's layout shards at least one
            non-transform axis.
        """
        decomposition = self._grid.decomposition
        if getattr(decomposition, "device_count", 1) <= 1:
            return
        layout = f.function_space.layout
        if layout is None:
            return
        bare = f.function_space.bare
        stage = self._stage_axes(bare)
        if not stage:
            return  # nothing transformed: an identity, no coefficient
        if any(not layout.is_local(axis) for axis in stage):
            return  # a sharded stage axis is Tier 1's domain
        sharded = tuple(
            name for name in bare.names if not layout.is_local(name))
        if not sharded:
            return  # fully replicated operand: an explicit gather
        raise NotImplementedError(
            f"{type(self).__name__}.{operation} cannot run on this "
            f"grid that shards the non-transform axis/axes {sharded!r} "
            f"across devices while transforming only the local axes "
            f"{stage!r}: the forward transform's codomain is a "
            "CoefficientSpace, which the storage contract "
            "(decomposition.sharding in spatial.decomposition.tensor) "
            "always replicates, so the naive GSPMD path would silently "
            "all-gather the sharded axis onto every device "
            "(numerically correct but unscalable — the whole global "
            "array is materialized per device). This Tier-2 naive "
            "transform path is illegal by design. For a spectral solve "
            "pass SpectralSolve(..., allow_replicated=True), which "
            "performs an explicit, honest replicate-then-compute "
            "(gather the operand to the replicated layout, apply, "
            "reshard back). For a standalone transform gather the "
            "operand first yourself "
            "(field.reshard(fr.spatial.decomposition.layout.Layout({})))"
            " or build the grid on a single device "
            "(Grid(..., device_ids=(0,)), which leaves every axis "
            "local); see "
            "design/research/gspmd_naive_transform_illegality.md.")

    def _reject_sharded_transform(self, f: FieldLike,
                                  operation: str) -> None:
        r"""
        Taught skip when a transform axis is sharded (Tier 1).

        Description
        -----------
        ``forward``/``backward`` always run the single-device plan;
        the safe distributed lowerings (``SlabPlan`` in
        ``spatial.operators.distributed_solve``, ``ContractPlan`` in
        ``spatial.operators.distributed_contract``) run their FFTs
        inside their own ``jax.shard_map`` and bypass this seam, so
        this guard fires only for a *naive* consumer that would run a
        transform axis through the plain GSPMD path. That path would
        silently all-gather the sharded axis (CPU) or crash in XLA's
        distributed-FFT lowering (GPU, jaxlib 0.10.2 — ``complex64``
        twiddle constants multiplied against ``complex128`` cuFFT
        data, rejected by the HLO verifier). The predicate reads the
        **operand's own** function-space layout (not the grid's
        default), so a deliberately gathered/replicated field passes;
        a sharded **non-transform** axis (Tier 2) is left legal (the
        FFT axes are local, GSPMD needs no reshard). The check is
        host-side Python on static layout metadata — a plain ``if``
        outside any traced value. A ``device_ids=(0,)`` grid and a
        grid too small to shard both collapse to one device and are
        exempt. See ``design/research/multidevice_test_faults.md``.

        Parameters
        ----------
        f : FieldLike
            The operand field (carries its function-space layout).
        operation : str
            The calling direction (``"forward"`` / ``"backward"``),
            for the error message.

        Raises
        ------
        NotImplementedError
            When the decomposition spans several devices and the
            operand's layout shards at least one of this transform's
            stage axes.
        """
        decomposition = self._grid.decomposition
        if getattr(decomposition, "device_count", 1) <= 1:
            return
        layout = f.function_space.layout
        if layout is None:
            return
        sharded = tuple(
            axis for axis in self._stage_axes(f.function_space.bare)
            if not layout.is_local(axis))
        if not sharded:
            return
        # narrow the blunt error: a plain (unpadded) Fourier transform
        # on a 1-D device mesh has a fused forward->diagonal->backward
        # route that keeps the sharded axis local per shard -- served
        # either as a standalone Fourier transform
        # (DistributedTransform.apply_diagonal, via
        # Transform.apply_diagonal) or as the Fourier part of a walled
        # ComposedTransform (ComposedTransform.apply_diagonal, via the
        # slab pipeline). Only a *standalone* forward/backward (which
        # materializes a coefficient field the storage contract
        # replicates) and the homogeneous-trig / non-1-D remainder have
        # no distributed route and land here.
        route = (
            " A plain forward->diagonal->backward apply (a spectral "
            "operator on the sharded field) does have a distributed "
            "route: resolve the transform "
            "(resolve_transform(grid, space.bare)) and call its "
            "apply_diagonal(f, symbol_factory) instead of the "
            "standalone forward/backward -- on a plain-Fourier grid "
            "the fused transform runs inside a jax.shard_map, and on "
            "a mixed (walled) grid the ComposedTransform routes the "
            "periodic axes through the same slab pipeline the walled "
            "spectral solve rides."
            if (self._space_family is FourierSpace
                and self._pad is None)
            else "")
        raise NotImplementedError(
            f"{type(self).__name__}.{operation} cannot run on this "
            f"grid that shards the transform axis/axes {sharded!r} "
            "across devices: the naive change-of-representation path "
            "would silently all-gather the sharded axis (CPU) or "
            "crash in XLA's distributed-FFT lowering (GPU, jaxlib "
            "0.10.2 — complex64 twiddle constants multiplied against "
            "complex128 cuFFT data, rejected by the HLO verifier). "
            "The distributed spectral solve "
            "(spatial.operators.distributed_solve) and the fused "
            "channel projection (spatial.operators.distributed_"
            f"contract) run their transforms inside a jax.shard_map "
            f"and are unaffected.{route} For host-side spectral "
            "analysis features build the grid on a single device "
            "(Grid(..., device_ids=(0,)), which leaves every axis "
            "local); see design/research/multidevice_test_faults.md.")

    def _stage_axes(self, bare: SpaceLike) -> tuple[str, ...]:
        """Transform axes present on ``bare``, constants dropped."""
        names = bare.names
        for axis in self._axes:
            if axis not in names:
                raise SpaceMismatchError(
                    f"transform axis {axis!r} is not a coordinate "
                    f"of {bare!r}; its names are {names}",
                    left=bare, operation="transform")
        return tuple(
            axis for axis in self._axes
            if not isinstance(bare.factor(axis), ConstantSpace))

    def _coarse_origin(self, factor: FunctionSpace,
                       axis: str) -> FunctionSpace:
        """
        Map a forward domain factor to its coarse origin.

        Description
        -----------
        Unpadded transforms take the factor itself. Padded ones
        require the factor to live on the refined mesh and rebuild
        the same nodal space on the coarse parent (the codomain
        exception: the coefficient target's origin is the coarse
        space).
        """
        if self._pad is None:
            return factor
        fine = self._fine_meshes[axis]
        if factor.mesh is not fine:
            raise SpaceMismatchError(
                f"padded forward along {axis!r} takes the refined "
                f"mesh's finer nodal space (on {fine!r}), got "
                f"{factor!r}", left=factor, operation="forward")
        return _sibling_origin(factor, fine.refined_from,
                               operation="forward")

    def _target_origin(self, coeff: CoefficientSpace,
                       axis: str) -> FunctionSpace:
        """
        Map a coefficient factor to its backward target.

        Description
        -----------
        The origin is constitutive: unpadded transforms land on
        ``coeff.origin``; padded ones land on the same nodal space
        rebuilt on the refined mesh.
        """
        origin = coeff.origin
        if self._pad is None:
            return origin
        return _sibling_origin(origin, self._fine_meshes[axis],
                               operation="backward")

    def _deliver(self, f: FieldLike, data: jax.Array,
                 codomain: SpaceLike) -> FieldLike:
        """
        Build the (laid-out, synced) result field from true data.

        Description
        -----------
        The tail of ``forward``/``backward``: cast to the codomain's
        derived storage dtype (dropping the exactly-zero imaginary
        part of real targets), re-attach the operand's layout (the
        single-device plan's final pencil is the operand's layout),
        and route through the storage contract.
        """
        dtype = storage_dtype(codomain)
        if (jnp.iscomplexobj(data)
                and not jnp.issubdtype(dtype, jnp.complexfloating)):
            data = data.real
        data = data.astype(dtype)
        layout = f.function_space.layout
        space = (codomain if layout is None
                 else codomain.with_layout(layout))
        stored = store(f.grid.decomposition, space, data)
        return type(f)(f.grid, space, stored, f.metadata)


# ================================================================
#  Module helpers
# ================================================================
def _complex_storage(space: SpaceLike) -> bool:
    """Whether the space's derived storage dtype is complex."""
    return jnp.issubdtype(storage_dtype(space), jnp.complexfloating)


def _rebuild(bare: SpaceLike,
             mapping: dict[str, FunctionSpace]) -> SpaceLike:
    """
    Substitute factors by coordinate name, allowing mesh changes.

    Description
    -----------
    Unlike ``TensorProductSpace.replace`` this permits replacements
    on *different* meshes: padded transforms swap a coarse-mesh
    factor for its refined-mesh sibling (same coordinate name).
    """
    if not mapping:
        return bare
    if isinstance(bare, FunctionSpace):
        return next(iter(mapping.values()))
    factors = tuple(
        next((new for name, new in mapping.items()
              if name in factor.names), factor)
        for factor in bare.factors)
    return TensorProductSpace.of(*factors)


def _sibling_origin(origin: FunctionSpace, mesh: Mesh,
                    operation: str) -> FunctionSpace:
    """
    Rebuild a nodal or average origin on a sibling mesh.

    Description
    -----------
    Padded transforms carry a physical-space origin from the coarse
    mesh to its refined sibling (or back). Nodal origins keep their
    node set and BC structure; average origins (``CellAvg`` /
    ``FaceAvg``, Fourier only) keep their cell family. The average
    families carry a **width-dependent** ``sinc(k dx / 2)`` factor
    that differs between the two meshes, so the padded Fourier
    kernels rescale by the ``sinc`` ratio; here we only name the
    sibling space (the numerics live in ``operators.fourier``).

    Parameters
    ----------
    origin : FunctionSpace
        The (bare) nodal or average origin on the source mesh.
    mesh : Mesh
        The sibling (refined or parent) mesh.
    operation : str
        The calling direction, for the error message.

    Returns
    -------
    FunctionSpace
        The same origin rebuilt on ``mesh`` (scalars preserved).
    """
    if isinstance(origin, NodalSpace):
        sibling: FunctionSpace = mesh.nodal(
            origin.node_set, bc=origin.bc)
    elif isinstance(origin, CellAvg):
        sibling = mesh.cell_avg
    elif isinstance(origin, FaceAvg):
        sibling = mesh.face_avg
    else:
        raise NotImplementedError(
            f"padded {operation} is defined for nodal and average "
            f"origins, got {origin!r}")
    if origin.scalars is Scalars.COMPLEX:
        sibling = sibling.as_complex()
    return sibling
