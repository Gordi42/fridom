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
    """

    axis: str
    index: int
    half: bool
    nodal: FunctionSpace
    coeff: FunctionSpace


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
        plan = self.forward_plan(f.function_space)
        data = jnp.asarray(f.data)
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
        plan = self.backward_plan(f.function_space)
        data = jnp.asarray(f.data)
        for stage in plan.stages:
            data = self._backward_kernel(data, stage)
        return self._deliver(f, data, plan.codomain)

    def _apply(self, f: FieldLike) -> FieldLike:
        """Delegate to ``forward`` (registry-uniform application)."""
        return self.forward(f)

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
        return _sibling_nodal(factor, fine.refined_from,
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
        return _sibling_nodal(origin, self._fine_meshes[axis],
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


def _sibling_nodal(origin: FunctionSpace, mesh: Mesh,
                   operation: str) -> FunctionSpace:
    """
    Rebuild a nodal origin on a sibling (refined/parent) mesh.

    Description
    -----------
    Padded transforms of **nodal** origins only in iteration 1:
    average origins carry a width-dependent ``sinc`` relation to
    their nodal siblings, so zero-padding their spectra onto a finer
    average space is not the exact refinement.
    """
    if not isinstance(origin, NodalSpace):
        raise NotImplementedError(
            f"padded {operation} is defined for nodal origins in "
            f"iteration 1, got {origin!r} (average origins change "
            "their sinc factor under refinement)")
    sibling = mesh.nodal(origin.node_set, bc=origin.bc)
    if origin.scalars is Scalars.COMPLEX:
        sibling = sibling.as_complex()
    return sibling
