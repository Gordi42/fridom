"""
``Fallback``: graded near-boundary order reduction (interior DOFs only).

Description
-----------
Owning plan: ``design/plans/active/fallback_operator_plan.md`` (stage
F1). ``Fallback`` runs a wide high-order reconstruction kernel in the
interior and, at the ``K`` output faces adjacent to each physical
wall, substitutes progressively narrower interior-only stencils
(order ``p -> p-2 -> ... -> 1``). It **reads no exterior values and
needs no boundary condition**, so it makes a wide bounded
reconstruction legal *by construction* and retires WENO's
periodic-only restriction (``weno.py`` ``codomain`` raises on bounded;
``Fallback`` owns the bounded ``CellAvg -> Inner`` signature instead).

Primary target: ``WenoReconstruction(order=5)`` wide reconstruction on
a bounded axis, falling back to WENO-3 then 1st-order upwind at the two
faces adjacent to each wall while keeping full order in the interior.

Ladder / boundary convention
-----------------------------
:func:`graded_ladder` builds the canonical, widest-first ladder

    ``WenoReconstruction(order) -> WenoReconstruction(order-2)
      -> ... -> WenoReconstruction(3) -> UpwindOne()``

whose head (``ladder[0]``) is the interior kernel and whose tail
(``ladder[1:]``) is the ordered ``boundary`` tuple passed to
``Fallback``. The ``boundary`` tuple is **widest-first, wall-adjacent
LAST**: ``boundary[0]`` is the widest reduced rung (order
``order-2``), applied to the reduced face **furthest** from the wall;
``boundary[-1]`` is the narrowest (1st-order upwind), applied to the
face **adjacent** to the wall. ``K = len(boundary)`` is the number of
reduced faces per side, a compile-time constant.

Index arithmetic (derived, static)
-----------------------------------
For an interior kernel of odd order ``O`` on a bounded mesh of ``n``
cells the reconstruction is ``CellAvg -> Inner`` (``n - 1`` interior
faces, physical face ``F = 1 .. n-1``). The interior halo is
``H = O // 2 + 1``; the wide window already reaches ``boundary_slack``
wall-adjacent interior faces (always ``1`` for the canonical WENO
ladder), so

    ``K = H - boundary_slack = (O - 1) // 2``.

A reduced face at distance ``d = 1 .. K`` from a wall (``d = 1`` is
wall-adjacent) is reconstructed by a stencil of order ``2*d - 1`` --
exactly ``boundary[K - d]``. That order is the widest odd stencil
whose biased window stays inside ``[0, n-1]``: the interior-vs-boundary
split is a STATIC index partition (never a data-dependent
``jnp.where``), matching the slice-window rules the WENO kernel
follows.

The ladder arithmetic, the wall-window builder and the multi-device
``patch_physical_ends`` seam are shared with the nodal C-grid graded
closure (``model.modules.advection``) and live in
:mod:`~fridom.spatial.operators.graded`; the average family is that
module's ``shift = 0`` cell frame (the cells *are* the operand's true
DOFs, so a BC-free bounded ``CellAvg`` offers nothing outside them --
``K = O // 2 + 0``, matching the derivation above).
"""
# Wave 4.x (F1): Fallback, UpwindOne, graded_ladder,
#    graded_reconstruction
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, final

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
)
from fridom.spatial.operators.graded import (
    Rung,
    apply_graded_walls,
    biased_offset,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.operators.reconstruct import (
    apply_fv_staggered,
)
from fridom.spatial.operators.weno import (
    WenoReconstruction,
    require_uniform_mesh,
    weno_reconstruct,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import CellAvg

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from jax import Array

    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )


# ================================================================
#  UpwindOne: the 1st-order (single-cell) reconstruction rung
# ================================================================
@final
@interned
class UpwindOne(SeparableOperator):

    """
    1st-order upwind reconstruction: the single wall-side cell value.

    Description
    -----------
    The innermost rung of the graded ladder (``_shu_row(1, face)`` --
    a one-cell reconstruction whose coefficient is unity, so the face
    value is just the upwind cell average). ``bias="left"`` takes the
    cell to the left of the face, ``bias="right"`` the cell to the
    right. Constructed as a standalone reconstruction operator too so
    the ladder is a homogeneous tuple of reconstruction kernels; its
    codomain matches ``WenoReconstruction`` on periodic
    (``CellAvg -> Right``) and bounded (``CellAvg -> Inner``) alike.
    Interned on ``bias`` via ``@interned`` (D6) -- structurally-equal
    rungs are the same object, the identity-hash invariant the graded
    ``Fallback`` keys on.

    Deliberately **not** guarded against stretched (mapped) factors,
    unlike its wider ladder siblings: the one-cell row carries no
    offsets at all (its single coefficient is unity on any mesh), so
    it is exact on constants and 1st-order accurate on a mapped mesh
    just as on a uniform one -- its design order survives the
    stretching. Only the wider uniform-offset rows lose order, and
    those refuse through ``weno.require_uniform_mesh``.

    Parameters
    ----------
    bias : Literal["left", "right"], optional
        The upwind side (default: "left").
    """

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __init__(
        self, bias: Literal["left", "right"] = "left",
    ) -> None:
        """Validate and store the upwind bias (interned on it, D6)."""
        if bias not in ("left", "right"):
            raise ValueError(
                f"bias must be 'left' or 'right', got {bias!r}")
        self._bias = bias

    def _intern_key(self) -> tuple:
        """Structural key: the upwind bias side (D6)."""
        return (self._bias,)

    @property
    def order(self) -> int:
        """Formal order (always 1)."""
        return 1

    @property
    def bias(self) -> Literal["left", "right"]:
        """Upwind bias side."""
        return self._bias

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """CellAvg -> Right (periodic) | Inner (bounded)."""
        if not isinstance(domain, CellAvg):
            raise SpaceMismatchError(
                "UpwindOne reconstructs primal cell averages onto "
                f"faces (CellAvg -> face), got {domain!r}",
                left=domain, operation="reconstruct")
        if domain.scalars is Scalars.COMPLEX:
            raise SpaceMismatchError(
                "UpwindOne has no complex iteration-1 signature, got "
                f"{domain!r}", left=domain, operation="reconstruct")
        mesh = domain.mesh
        return mesh.right if mesh.periodic else mesh.inner

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — fixed one-cell reach
    ) -> OperatorRequirements:
        """Declare halo = 1 (single-cell shift), layout "any"."""
        return OperatorRequirements(halo=1)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Reconstruct along ``axis`` as the single upwind cell value.

        Description
        -----------
        Kernel output ``t`` is window cell ``t`` (left bias) or
        ``t + 1`` (right bias); the face lands on output slot
        ``t + m0`` with ``m0 = 1`` (left) / ``0`` (right).
        """
        m0 = 1 if self._bias == "left" else 0

        def kernel(arr: Array, axis_index: int) -> Array:
            index = [slice(None)] * arr.ndim
            length = arr.shape[axis_index] - 1
            start = 0 if self._bias == "left" else 1
            index[axis_index] = slice(start, start + length)
            return arr[tuple(index)]

        return apply_fv_staggered(self, f, axis, 2, kernel,
                                  metadata=f.metadata, align=m0)


# ================================================================
#  Graded ladder builder
# ================================================================
#: smallest WENO order carried by the graded ladder before it drops
#: to the 1st-order upwind rung (the reduced tables ground orders 3+)
_MIN_WENO_ORDER = 3


def graded_ladder(
    order: int, bias: Literal["left", "right"] = "left",
) -> tuple[SeparableOperator, ...]:
    """
    Build the canonical widest-first graded reconstruction ladder.

    Description
    -----------
    ``WenoReconstruction(order) -> WenoReconstruction(order-2) -> ...
    -> WenoReconstruction(3) -> UpwindOne(bias)``. The head is the
    interior kernel; ``ladder[1:]`` is the ``boundary`` tuple of
    :class:`Fallback` (widest reduced rung first, 1st-order upwind
    last).

    Parameters
    ----------
    order : int
        The interior odd formal order (iteration 1: 3 or 5).
    bias : Literal["left", "right"], optional
        The shared upwind bias of every rung (default: "left").

    Returns
    -------
    tuple[SeparableOperator, ...]
        The ladder, head (interior) first.
    """
    rungs: list[SeparableOperator] = []
    current = order
    while current >= _MIN_WENO_ORDER:
        rungs.append(WenoReconstruction(current, bias=bias))
        current -= 2
    rungs.append(UpwindOne(bias))
    return tuple(rungs)


def graded_reconstruction(
    order: int = 5, bias: Literal["left", "right"] = "left",
) -> Fallback:
    """
    Build the graded ``Fallback`` for a WENO interior of ``order``.

    Description
    -----------
    Convenience factory (the ``boundary="graded"`` knob of F2 dispatches
    here): ``ladder = graded_ladder(order, bias)`` and returns
    ``Fallback(ladder[0], ladder[1:])``.

    Parameters
    ----------
    order : int, optional
        The interior odd formal order (default: 5).
    bias : Literal["left", "right"], optional
        The shared upwind bias (default: "left").

    Returns
    -------
    Fallback
        The interned graded operator.
    """
    ladder = graded_ladder(order, bias=bias)
    return Fallback(ladder[0], ladder[1:])


# ================================================================
#  Fallback
# ================================================================
@final
@interned
class Fallback(SeparableOperator):

    """
    Graded near-boundary order reduction over interior DOFs only.

    Description
    -----------
    Sibling of ``SeparableComposite`` (a ``SeparableOperator`` built
    directly, not by ``@``), interned on ``(interior,
    tuple(boundary))`` via ``@interned`` (D6). Runs ``interior`` in the
    interior and the ``boundary`` rungs at the ``K`` faces adjacent to
    each wall; see the module docstring for the ladder convention and
    the static index arithmetic.

    Parameters
    ----------
    interior : WenoReconstruction
        The wide interior reconstruction kernel.
    boundary : Sequence[SeparableOperator]
        The ordered reduced-order ladder rungs, widest-first,
        wall-adjacent last (``graded_ladder(order, bias)[1:]``).
    """

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __init__(
        self,
        interior: WenoReconstruction,
        boundary: Sequence[SeparableOperator],
    ) -> None:
        """Validate the ladder and cache the static structure (D6)."""
        self._setup(interior, tuple(boundary))

    def _intern_key(self) -> tuple:
        """Structural key: the interior kernel and the ladder (D6)."""
        return (self._interior, self._boundary)

    def _setup(
        self,
        interior: WenoReconstruction,
        boundary: tuple[SeparableOperator, ...],
    ) -> None:
        """Validate the ladder and cache the static structure."""
        if not isinstance(interior, WenoReconstruction):
            raise TypeError(
                "Fallback's interior kernel must be a "
                f"WenoReconstruction, got {interior!r}")
        order = interior.order
        bias = interior.bias
        boundary_slack = 1  # canonical WENO ladder: one interior face
        expected_k = order // 2 + 1 - boundary_slack
        if len(boundary) != expected_k:
            raise ValueError(
                f"a WENO-{order} interior needs {expected_k} boundary "
                f"rungs (halo {order // 2 + 1} - boundary_slack "
                f"{boundary_slack}), got {len(boundary)}")
        for j, rung in enumerate(boundary):
            expected_order = order - 2 * (j + 1)
            rung_order = getattr(rung, "order", None)
            rung_bias = getattr(rung, "bias", None)
            if rung_order != expected_order:
                raise ValueError(
                    f"boundary rung {j} must have order "
                    f"{expected_order} (graded ladder), got "
                    f"{rung_order!r}")
            if rung_bias != bias:
                raise ValueError(
                    f"boundary rung {j} must share the interior bias "
                    f"{bias!r}, got {rung_bias!r}")
        if boundary[-1].order != 1:
            raise ValueError(
                "the wall-adjacent (last) boundary rung must be "
                f"1st-order upwind, got order {boundary[-1].order}")
        self._interior: WenoReconstruction = interior
        self._boundary: tuple[SeparableOperator, ...] = boundary
        self._k: int = expected_k

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def interior(self) -> WenoReconstruction:
        """The wide interior reconstruction kernel."""
        return self._interior

    @property
    def boundary(self) -> tuple[SeparableOperator, ...]:
        """The reduced-order ladder rungs (widest-first)."""
        return self._boundary

    @property
    def reduced_rows(self) -> int:
        """``K``: reduced faces per side (compile-time constant)."""
        return self._k

    # ------------------------------------------------------------
    #  Signature and requirements
    # ------------------------------------------------------------
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve the reconstruction codomain (periodic AND bounded).

        Description
        -----------
        ``CellAvg -> Right`` on a periodic mesh (delegates to the
        interior), ``CellAvg -> Inner`` on a bounded mesh -- the
        interior alone raises on bounded, so ``Fallback`` owns the
        bounded interior-face signature. Every ladder rung shares this
        signature by construction (validated in ``_setup``: same kind,
        same bias, odd reconstruction orders).

        The graded ladder retires WENO's *periodic*-only restriction,
        not its *uniform*-mesh one: every rung of order >= 3 is a
        uniform-offset Shu row (the interior kernel and the reduced
        wall rungs alike), so a stretched (mapped) factor raises here
        exactly as it does on the bare interior kernel
        (``weno.require_uniform_mesh``).

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space.

        Returns
        -------
        FunctionSpace
            The face point-value codomain factor.
        """
        if not isinstance(domain, CellAvg):
            raise SpaceMismatchError(
                "Fallback reconstructs primal cell averages onto "
                f"faces (CellAvg -> face), got {domain!r}",
                left=domain, operation="reconstruct")
        require_uniform_mesh(
            domain, "the graded Fallback (its interior and reduced "
                    "rungs are WENO/Shu rows)")
        if domain.scalars is Scalars.COMPLEX:
            raise SpaceMismatchError(
                "the WENO smoothness indicators are real quadratic "
                "forms; complex operands have no iteration-1 "
                f"signature, got {domain!r}",
                left=domain, operation="reconstruct")
        mesh = domain.mesh
        return mesh.right if mesh.periodic else mesh.inner

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """
        Declare the interior halo; ``layout="any"`` on every axis.

        Description
        -----------
        The interior kernel dominates the halo demand. The bounded axis
        may be sharded: ``sync`` fills each block's halos so the wide
        interior pass is already correct on a blocked global array
        (identical to the sharded-periodic case), and the two wall
        blocks' reduced rows are patched behind the decomposition-layer
        seam ``patch_physical_ends`` (stage F6), so no axis needs to be
        kept undistributed -- ``layout="any"`` on periodic and bounded
        alike.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        halo = self._interior.requirements(domain).halo
        return OperatorRequirements(halo=halo)

    # ------------------------------------------------------------
    #  Kernel application (interior pass + wall-slot assembly)
    # ------------------------------------------------------------
    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Interior WENO pass, then overwrite the ``K`` wall faces/side.

        Description
        -----------
        (1) Run the interior kernel over the halo-extended storage via
        the shared ``apply_fv_staggered`` tail (resolving the bounded
        ``CellAvg -> Inner`` codomain through ``self.codomain``); its
        near-wall output slots are computed from exterior cells and are
        discarded. (2) Hand the two physical-wall ends to the shared
        graded tail ``graded.apply_graded_walls`` (the ``shift = 0``
        cell frame: the ``CellAvg`` DOFs *are* the lattice cells), which
        drives the decomposition seam ``patch_physical_ends`` and, per
        wall, overwrites the ``K`` reduced faces of one local block from
        that block's wall-side interior cells -- indexed off the
        **local** true count, never the global ``n``, so it is correct
        whether the axis is undistributed or genuinely sharded. (3) The
        bounded halo-validity reset is inherited from the interior tail.
        On a periodic axis there are no walls and the interior pass is
        returned unchanged.

        Parameters
        ----------
        f : FieldLike
            The operand field (storage-shaped ``_data``).
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The reconstructed field (metadata kept: same quantity).
        """
        space = f.function_space.bare
        factor = space.factor(axis)
        mesh = factor.mesh
        order = self._interior.order
        bias = self._interior.bias
        m0 = biased_offset(order, bias)

        def kernel(arr: Array, axis_index: int) -> Array:
            return weno_reconstruct(arr, axis_index, order=order,
                                    bias=bias)

        interior = apply_fv_staggered(self, f, axis, order, kernel,
                                      metadata=f.metadata, align=m0)
        if getattr(mesh, "periodic", False):
            return interior

        rungs = tuple(
            _weno_rung(rung.order, bias) for rung in self._boundary)
        return apply_graded_walls(f, axis, interior, rungs, shift=0)


def _weno_rung(
    order: int, bias: Literal["left", "right"],
) -> Rung:
    """
    Build the graded rung of one WENO ladder step.

    Description
    -----------
    Order 1 is the single upwind cell (``_shu_row(1, .)`` = unit
    coefficient), so its kernel is the identity on the size-1 window;
    order ``p >= 3`` runs ``weno_reconstruct`` on the length-``p``
    window, which yields exactly the single face value.

    Parameters
    ----------
    order : int
        The rung's odd formal order.
    bias : Literal["left", "right"]
        The shared upwind bias side.

    Returns
    -------
    Rung
        The (size, offset, kernel) triple of the rung.
    """
    offset = biased_offset(order, bias)
    if order == 1:
        return Rung(1, offset, lambda arr, _axis: arr)

    def kernel(arr: Array, axis_index: int) -> Array:
        return weno_reconstruct(arr, axis_index, order=order,
                                bias=bias)

    return Rung(order, offset, kernel)
