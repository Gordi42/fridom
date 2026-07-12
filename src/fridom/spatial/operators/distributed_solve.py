r"""
Transform-plan-driven distributed spectral solve (reconciliation S2).

Description
-----------
Owning plan: ``design/plans/active/distributed_transform_plan.md``
(Stage 2). This is the seam that lets :class:`SpectralSolve` obtain
distribution by composing the *ordinary* transform: it resolves the
fused slab pipeline's geometry from the transform's
``distributed_forward_plan`` (the Stage-1 planner) instead of
``slab_fft``'s bespoke ``_slab_geometry`` / ``_internal_coeff``, then
reuses the proven slab ``shard_map`` kernel
(:class:`~fridom.spatial.operators.slab_fft.SlabPlan` /
:class:`~fridom.spatial.operators.slab_fft.SlabSolve`) as the lowering
target.

Because the Stage-1 planner reproduces the slab axis roles
(``a``/``b``/``h``) and internal coefficient space byte-for-byte on the
supported (divisible, unpadded, all-Fourier) grids, the resolved solve
is identical to ``resolve_slab_plan`` + ``SlabSolve`` — same kernel,
same two ``all_to_all``, no gather — only the *resolution* now flows
through the transform. Stage 4 relocates the kernel bodies here and
retires the slab-specific resolution; Stage 3 switches ``SpectralSolve``
onto :func:`resolve_distributed_solve`.
"""
# S2: distributed solve resolved from the transform's layout plan
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.storage import storage_dtype
from fridom.spatial.operators.base import EigenbasisError
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.slab_fft import (
    SlabPlan,
    SlabSolve,
    symbol_fits,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.operators.transform import Transform
    from fridom.spatial.spaces.tensor_product import SpaceLike


#: per-grid memo of resolved plans, keyed on the interned bare space
#: (the ``mixed.py``/``slab_fft.py`` ``WeakKeyDictionary`` idiom: a
#: dropped grid auto-evicts its memo)
_PLANS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def build_distributed_plan(
    transform: Transform, grid: object, bare: SpaceLike,
) -> SlabPlan | None:
    """
    Build the fused solve plan from the transform's layout plan.

    Description
    -----------
    Derives the slab kernel geometry (mesh, sharded axis ``a``,
    transpose partner ``b``, local Hermitian axis ``h``, transformed
    axes, internal coefficient space) from
    ``transform.distributed_forward_plan(bare)`` rather than
    ``slab_fft._slab_geometry``/``_internal_coeff``. Returns None when
    the transform is not a plain unpadded :class:`Fourier`, the mesh is
    not 1-D, or the operand is single-device / ineligible (the planner
    returns None) — the caller then keeps the replicated composite.

    Parameters
    ----------
    transform : Transform
        The transform resolved for ``bare`` (only a plain unpadded
        ``Fourier`` is eligible).
    grid : object
        The grid carrying the decomposition.
    bare : SpaceLike
        The bare nodal operand space.

    Returns
    -------
    SlabPlan | None
        The reusable slab pipeline, or None when ineligible.
    """
    if not isinstance(transform, Fourier) or transform.pad is not None:
        return None
    forward = transform.distributed_forward_plan(bare)
    if forward is None:
        return None
    decomposition = grid.decomposition
    mesh = decomposition.device_mesh
    if len(mesh.axis_names) != 1:
        return None
    names = bare.names
    ((name_b, _),) = forward.codomain.layout.device_axes
    half = next((s for s in forward.stages if s.half), None)
    return SlabPlan(
        mesh=mesh,
        axis_name=mesh.axis_names[0],
        domain=bare,
        coeff=forward.codomain.bare,
        layout=decomposition.default_layout,
        a=forward.stages[-1].index,
        b=names.index(name_b),
        h=None if half is None else half.index,
        fft_axes=tuple(sorted(stage.index for stage in forward.stages)),
        real=not jnp.issubdtype(storage_dtype(bare),
                                jnp.complexfloating),
    )


def resolve_distributed_plan(
    transform: Transform, grid: object, bare: SpaceLike,
) -> SlabPlan | None:
    """
    Resolve (and memoize) the distributed plan of ``bare``, or None.

    Parameters
    ----------
    transform : Transform
        The transform resolved for ``bare``.
    grid : object
        The grid carrying the decomposition and dispatch registry.
    bare : SpaceLike
        The bare nodal operand space.

    Returns
    -------
    SlabPlan | None
        The memoized plan, or None when ineligible.
    """
    memo = _PLANS.setdefault(grid, {})
    if bare not in memo:
        memo[bare] = build_distributed_plan(transform, grid, bare)
    return memo[bare]


def resolve_distributed_solve(
    elliptic: Operator,
    transform: Transform,
    grid: object,
    bare: SpaceLike,
    where_zero: complex,
) -> SlabSolve | None:
    """
    Resolve the transform-driven distributed solve, or None.

    Description
    -----------
    The Stage-3 entry point: pairs the resolved plan with the inverse
    eigenvalue diagonal on the plan's internal coefficient space, as
    ``SpectralSolve._resolve_slab`` did — but through the transform's
    layout plan. Returns None (fall back to the replicated composite)
    when the plan is ineligible, the eigenvalues do not materialize on
    the internal space, or the diagonal is not a broadcast-shaped
    endomorphism there.

    Parameters
    ----------
    elliptic : Operator
        The elliptic operator to invert (an :class:`Operator` recipe; a
        pre-assembled ``Symbol`` is not eligible — it is bound to the
        replicated codomain).
    transform : Transform
        The transform resolved for ``bare``.
    grid : object
        The grid.
    bare : SpaceLike
        The bare nodal operand space.
    where_zero : complex
        The inverse value at structural zeros (the nullspace gauge).

    Returns
    -------
    SlabSolve | None
        The distributed solve, or None when ineligible.
    """
    plan = resolve_distributed_plan(transform, grid, bare)
    if plan is None:
        return None
    try:
        symbol = elliptic.eigenvalues(grid, plan.coeff)
    except (EigenbasisError, SpaceMismatchError):
        return None
    if not symbol_fits(plan, symbol):
        return None
    return SlabSolve(plan, symbol.inverse(where_zero))
