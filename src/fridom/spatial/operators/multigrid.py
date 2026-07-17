r"""
``MultigridVCycle``: the fixed-count symmetric V-cycle preconditioner.

Description
-----------
Owning design: ``design/plans/active/multigrid_pathway_plan.md`` §B2
(decisions MG-D7/D8). A model-agnostic geometric-multigrid engine, the
sibling of :mod:`~fridom.spatial.operators.krylov`: it consumes a
static tuple of :class:`MultigridLevel`\ s (finest first) — each an
SPD operator ``A_l``, a symmetric fixed-sweep :class:`Smoother`, an
optional nullspace ``projection``, and a :class:`GridTransfer` to the
next-coarser level — and applies as a field-to-field callable
``M_inv(residual) -> correction`` suitable as the ``preconditioner=``
of :class:`~fridom.spatial.operators.krylov.ConjugateGradient`
(MG-D8).

Trace-time object (CS-D2 discipline)
------------------------------------
Like ``ConjugateGradient`` and unlike the field operators, the cycle
is **not** a jax pytree: it is a plain object built once at trace time,
never a leaf. All of its structure — the level tuple, the sweep counts
— is Python-static, and the cycle body is a **Python recursion over
the level tuple, fully unrolled at trace time**. There is no
``lax`` control flow over levels and no tolerance branching, so the
HLO is flat in both the level count and the sweep counts, and the
whole cycle jit-compiles once across right-hand-side values.

Why it is a valid CG preconditioner (MG-D7)
-------------------------------------------
The smoothers are symmetric (damped point / vertical-line Jacobi) and
the transfer pair is adjoint (``R = P†`` in the measure-weighted
product, MG-D2), so a V-cycle with ``pre_sweeps == post_sweeps`` and a
fixed coarse sweep count is a **fixed symmetric linear operator** —
SPD in the finest level's measure-weighted product. That keeps plain
``ConjugateGradient`` valid (no FGMRES / flexible CG). The construction
therefore **forbids** ``pre_sweeps != post_sweeps`` and any inner CG at
the coarsest level (a fixed-iteration CG is still a nonlinear map of
its input and would silently break the outer CG): the coarsest solve
is a fixed number of extra smoother sweeps.

Per-level nullspace consistency
-------------------------------
Every level's operator inherits the finest problem's singularity (the
constants, or the immersed wet-region constant). The restricted
residual handed down is projected onto the coarser level's range with
that level's ``projection`` (mean removal, never point pinning), and
each level projects its own returned correction — the same pluggable
``projection=`` discipline ``ConjugateGradient`` uses (IP-D6).
"""
# Multigrid pathway plan, phase B (B2): the V-cycle engine (MG-D7/D8)
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from fridom.spatial.operators.banded import tridiagonal_solve_along_axis

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.operators.transfer import GridTransfer

    #: a field-to-field callable (an operator ``A_l`` or a projection)
    FieldOp = Callable[[ScalarField], ScalarField]


# ================================================================
#  Vertical-line tridiagonal bands
# ================================================================
class VerticalBands(NamedTuple):

    """
    The per-column symmetric tridiagonal ``T`` of a line smoother.

    Description
    -----------
    The three bands (as fields on the pressure space) and the storage
    axis they run along. ``diag`` is the operator's full diagonal (so
    the horizontal stiffness enters the column solve as a stronger
    diagonal, MG-D7); ``lower``/``upper`` are the operator's vertical
    off-diagonals (``lower[c]`` couples cell ``c`` to ``c - 1``,
    ``upper[c]`` to ``c + 1``, both along ``axis``), with the Neumann
    ends ``lower[0]`` and ``upper[N - 1]`` left at zero. On a uniform
    column ``T`` is symmetric per column by construction (``lower[c] ==
    upper[c - 1]``); on a stretched column (the measure-weighted
    operator, N3) it is instead **self-adjoint under the physical cell
    measure** — ``m_cell[c - 1] * upper[c - 1] == m_cell[c] *
    lower[c]``, i.e. ``diag(m_cell) T`` is symmetric — which is CG's
    inner product, so the symmetric V-cycle argument (MG-D7) carries
    over unchanged. The batched Thomas solve
    (:func:`~fridom.spatial.operators.banded.tridiagonal_solve_along_axis`)
    needs no per-column symmetry.

    Parameters
    ----------
    lower : ScalarField
        The sub-diagonal band.
    diag : ScalarField
        The main diagonal band (the operator's full diagonal).
    upper : ScalarField
        The super-diagonal band.
    axis : int
        The storage-frame index of the vertical (column) axis.
    """

    lower: ScalarField
    diag: ScalarField
    upper: ScalarField
    axis: int


# ================================================================
#  Smoothers (symmetric, coloring-free; pure sweep(x, b, operator))
# ================================================================
class Smoother(ABC):

    """
    A symmetric stationary smoother: one damped fixed-sweep update.

    Description
    -----------
    The cycle drives smoothing through :meth:`sweep`, passing the
    level's operator ``A`` (so the smoother holds only its coefficients
    and damping, not a second copy of ``A``). Every concrete smoother
    is symmetric and coloring-free (MG-D7), so ``pre == post`` sweeps
    keep the V-cycle symmetric.
    """

    @abstractmethod
    def sweep(
        self, x: ScalarField, b: ScalarField, operator: FieldOp,
    ) -> ScalarField:
        """
        Return ``x`` after one damped smoothing sweep toward ``A x = b``.

        Parameters
        ----------
        x : ScalarField
            The current iterate.
        b : ScalarField
            The right-hand side on the same space.
        operator : FieldOp
            The level's SPD operator ``A``.

        Returns
        -------
        ScalarField
            The smoothed iterate.
        """


class DampedJacobi(Smoother):

    r"""
    Damped point-Jacobi smoother ``x <- x + omega (b - A x) / d``.

    Description
    -----------
    The isotropic smoother (MG-D7): a diagonal-scaled damped
    correction, symmetric because ``d`` is the operator's diagonal.
    The division is guarded with the double-``jnp.where`` pattern so a
    zero-diagonal (dry / padded) cell takes a zero update and the
    reverse-mode gradient stays NaN-free (the differentiability
    policy; ``krylov._guarded_ratio`` is the house pattern).

    Parameters
    ----------
    diagonal : ScalarField
        The operator's diagonal, on the operand's space.
    omega : float, optional
        The damping factor (default: 2/3, the smoothing-optimal value
        for the point Laplacian).
    """

    def __init__(self, diagonal: ScalarField,
                 omega: float = 2.0 / 3.0) -> None:
        """Store the diagonal field and the damping factor."""
        self._diagonal: ScalarField = diagonal
        self._omega: float = float(omega)

    @property
    def diagonal(self) -> ScalarField:
        """The operator's diagonal field."""
        return self._diagonal

    @property
    def omega(self) -> float:
        """The damping factor."""
        return self._omega

    def sweep(
        self, x: ScalarField, b: ScalarField, operator: FieldOp,
    ) -> ScalarField:
        """Apply one damped-Jacobi sweep (guarded diagonal division)."""
        residual = b - operator(x)
        d = self._diagonal.data
        zero = d == 0.0
        safe = jnp.where(zero, 1.0, d)
        ratio = jnp.where(zero, 0.0, residual.data / safe)
        return x.with_data(x.data + self._omega * ratio)


class VerticalLineJacobi(Smoother):

    r"""
    Damped vertical-line Jacobi ``x <- x + omega T^{-1}(b - A x)``.

    Description
    -----------
    The anisotropy smoother (MG-D4/D7): every column is relaxed at
    once through the per-column symmetric tridiagonal ``T`` (the
    :class:`VerticalBands`), solved by the batched Thomas kernel
    :func:`~fridom.spatial.operators.banded.tridiagonal_solve_along_axis`.
    Because ``T`` carries the full diagonal, its columns are strictly
    diagonally dominant on wet cells, so the pivot-free Thomas solve is
    stable. Dry / zero-diagonal columns are sanitized with the
    double-``jnp.where`` guard (``diag -> 1``, ``rhs -> 0``, output
    forced to zero), which both keeps a dry column a no-op and keeps
    the reverse-mode gradient NaN-free.

    Parameters
    ----------
    bands : VerticalBands
        The per-column tridiagonal ``T`` and its column axis.
    omega : float, optional
        The damping factor (default: 0.8, the B0-spike optimum for the
        steep mapped column).
    """

    def __init__(self, bands: VerticalBands, omega: float = 0.8) -> None:
        """Store the tridiagonal bands and the damping factor."""
        self._bands: VerticalBands = bands
        self._omega: float = float(omega)

    @property
    def bands(self) -> VerticalBands:
        """The per-column tridiagonal bands."""
        return self._bands

    @property
    def omega(self) -> float:
        """The damping factor."""
        return self._omega

    def sweep(
        self, x: ScalarField, b: ScalarField, operator: FieldOp,
    ) -> ScalarField:
        """Apply one damped vertical-line sweep (guarded Thomas solve)."""
        residual = b - operator(x)
        diag = self._bands.diag.data
        zero = diag == 0.0
        safe_diag = jnp.where(zero, 1.0, diag)
        safe_rhs = jnp.where(zero, 0.0, residual.data)
        correction = tridiagonal_solve_along_axis(
            self._bands.lower.data, safe_diag, self._bands.upper.data,
            safe_rhs, self._bands.axis)
        correction = jnp.where(zero, 0.0, correction)
        return x.with_data(x.data + self._omega * correction)


# ================================================================
#  The hierarchy level and the V-cycle engine
# ================================================================
@dataclass(frozen=True)
class MultigridLevel:

    """
    One level of the multigrid hierarchy (finest to coarsest).

    Description
    -----------
    Bundles the level's SPD operator, its symmetric smoother, the
    optional nullspace projection, and the transfer down to the next
    coarser level (``None`` on the coarsest level). All four are
    Python-static structure the cycle unrolls over at trace time.

    Parameters
    ----------
    operator : FieldOp
        The level's SPD operator ``A_l`` (field to field).
    smoother : Smoother
        The level's symmetric fixed-sweep smoother.
    projection : FieldOp | None
        The nullspace projection for this level (mean removal), or
        ``None`` when the level operator is non-singular.
    transfer : GridTransfer | None
        The transfer to the next-coarser level; ``None`` marks the
        coarsest level.
    """

    operator: FieldOp
    smoother: Smoother
    projection: FieldOp | None
    transfer: GridTransfer | None


class MultigridVCycle:

    r"""
    Fixed-count symmetric V-cycle preconditioner (MG-D7/D8).

    Description
    -----------
    See the module docstring. Applied as ``vcycle(residual) ->
    correction`` — the ``preconditioner=`` callable of
    :class:`~fridom.spatial.operators.krylov.ConjugateGradient`. Built
    once at trace time from a finest-first level tuple; the cycle body
    is a Python recursion unrolled into the trace (no ``lax`` control
    flow over levels).

    Parameters
    ----------
    levels : tuple[MultigridLevel, ...]
        The hierarchy, finest first; at least one level. Only the last
        level may carry ``transfer=None``.
    pre_sweeps : int, optional
        Pre-smoothing sweeps before the coarse correction; must equal
        ``post_sweeps`` for symmetry (default: 1).
    post_sweeps : int, optional
        Post-smoothing sweeps after the coarse correction (default: 1).
    coarse_sweeps : int, optional
        Smoother sweeps standing in for the coarsest solve (no inner
        CG — forbidden, MG-D7) (default: 8).
    """

    def __init__(
        self,
        levels: tuple[MultigridLevel, ...],
        *,
        pre_sweeps: int = 1,
        post_sweeps: int = 1,
        coarse_sweeps: int = 8,
    ) -> None:
        """Validate the hierarchy and the symmetric sweep budget."""
        if not levels:
            raise ValueError(
                "MultigridVCycle needs at least one level")
        for level in levels[:-1]:
            if level.transfer is None:
                raise ValueError(
                    "only the coarsest (last) level may have "
                    "transfer=None; an interior level needs a transfer "
                    "to the next-coarser level")
        if levels[-1].transfer is not None:
            raise ValueError(
                "the coarsest (last) level must have transfer=None "
                "(it is solved by fixed smoother sweeps, MG-D7)")
        if pre_sweeps != post_sweeps:
            raise ValueError(
                "pre_sweeps must equal post_sweeps for a symmetric "
                f"V-cycle (MG-D7), got {pre_sweeps} and {post_sweeps}")
        if pre_sweeps < 1 or coarse_sweeps < 1:
            raise ValueError(
                "pre_sweeps and coarse_sweeps must be >= 1, got "
                f"{pre_sweeps} and {coarse_sweeps}")
        self._levels: tuple[MultigridLevel, ...] = tuple(levels)
        self._pre_sweeps: int = pre_sweeps
        self._post_sweeps: int = post_sweeps
        self._coarse_sweeps: int = coarse_sweeps

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def levels(self) -> tuple[MultigridLevel, ...]:
        """The hierarchy levels (finest first)."""
        return self._levels

    @property
    def pre_sweeps(self) -> int:
        """The pre-smoothing sweep count."""
        return self._pre_sweeps

    @property
    def post_sweeps(self) -> int:
        """The post-smoothing sweep count."""
        return self._post_sweeps

    @property
    def coarse_sweeps(self) -> int:
        """The coarsest-level smoother sweep count."""
        return self._coarse_sweeps

    # ================================================================
    #  Application
    # ================================================================
    def __call__(self, r: ScalarField) -> ScalarField:
        """
        Return the V-cycle correction for the residual ``r``.

        Parameters
        ----------
        r : ScalarField
            The residual on the finest level's space.

        Returns
        -------
        ScalarField
            The preconditioner correction on the same space.
        """
        return self._cycle(0, r)

    @staticmethod
    def _project(level: MultigridLevel, f: ScalarField) -> ScalarField:
        """Apply the level's nullspace projection (identity if None)."""
        if level.projection is None:
            return f
        return level.projection(f)

    def _cycle(self, index: int, b: ScalarField) -> ScalarField:
        r"""
        Return the coarse-grid correction at ``levels[index]``.

        Description
        -----------
        The recursive V-cycle body (module docstring): zero initial
        guess, ``pre_sweeps`` smoothing, restrict-and-project the
        residual onto the coarser level, recurse, prolong and correct,
        ``post_sweeps`` smoothing; the coarsest level is
        ``coarse_sweeps`` smoothing from zero. Each level projects its
        own returned correction (nullspace consistency).

        Parameters
        ----------
        index : int
            The level index (0 = finest).
        b : ScalarField
            The right-hand side on this level's space.

        Returns
        -------
        ScalarField
            The correction on this level's space.
        """
        level = self._levels[index]
        operator = level.operator
        smoother = level.smoother
        x = b.with_data(0.0 * b.data)
        if level.transfer is None:
            for _ in range(self._coarse_sweeps):
                x = smoother.sweep(x, b, operator)
            return self._project(level, x)
        for _ in range(self._pre_sweeps):
            x = smoother.sweep(x, b, operator)
        residual = b - operator(x)
        child = self._levels[index + 1]
        coarse_b = self._project(
            child, level.transfer.restrict(residual))
        coarse_correction = self._cycle(index + 1, coarse_b)
        x = x + level.transfer.prolong(coarse_correction)
        for _ in range(self._post_sweeps):
            x = smoother.sweep(x, b, operator)
        return self._project(level, x)
