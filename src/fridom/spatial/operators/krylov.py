r"""
``ConjugateGradient``: the matrix-free preconditioned CG solver core.

Description
-----------
Owning design: ``design/plans/active/coordinate_systems_plan.md``
decision **CS-D2** (stage C3). Non-separable elliptic operators (the
pressure Poisson problem on a coordinate-mapped grid, where the axes
couple and the exact spectral
:class:`~fridom.spatial.operators.spectral_solve.SpectralSolve` no
longer applies) are solved by **matrix-free conjugate gradients**,
preconditioned by the separable spectral inverse.

This module is the *solver core* only: it consumes the system operator
``A`` and the preconditioner ``M_inv`` as opaque field-to-field
callables (an :class:`~fridom.spatial.operators.base.Operator`, a
``SpectralSolve``, or any Python callable on the same space). The
mapped pressure operator and the non-hydrostatic wiring live
elsewhere.

Two CS-D2 requirements shape the implementation:

- **Fixed iteration count (static trace).** The recurrence runs a
  Python-static number of iterations with *no* tolerance break — it is
  a plain unrolled loop, so the whole solve is one straight-line
  jaxpr: it jit-compiles once across right-hand-side *values* and is
  reverse-mode differentiable (``jax.grad`` flows through the chain)
  without a ``custom_vjp``. The alternative fixed-length ``lax``
  loops carry the field iterates as a pytree; the unrolled form sides
  steps the treedef-stability bookkeeping that
  ``ScalarField.halo_valid`` would otherwise impose on a loop carry,
  at the documented cost of a larger trace for large iteration counts.
- **Everything pure.** No Python-side state is mutated and nothing
  branches on a traced value, so the solver is safe inside a
  jit-compiled tendency container.

Inner products
--------------
CG is correct only in the inner product under which ``A`` is
symmetric positive definite. For the flux-form mapped Laplacian that
is the **measure-weighted** :math:`L^2` product
:math:`\langle a, b\rangle = \int a\,b \,\mathrm{d}V` (the mimetic
``div``/``grad`` adjointness holds under the
:math:`\sqrt{g}`-weighted product; SPD-ness is what licenses CG).
Every inner product here is that product, evaluated through the
field's own ``integrate`` machinery — which sums the **true DOFs**
(``f.data``, halo/pad stripped) weighted by ``grid.measure`` and
handles the cross-shard reduction. Routing the dot products through
``integrate`` is therefore what keeps them halo-clean under domain
decomposition: no halo or stagger-pad slot ever enters a sum (a leak
would silently converge the solver to the wrong answer on several
devices). On a uniform flat grid the measure is a constant XLA folds,
so the weighting reduces to the plain Euclidean product up to a factor
that cancels in the CG ratios.

Nullspace
---------
``project_mean`` handles the singular Neumann/periodic Poisson problem
whose nullspace is the constants: the (measure-weighted) mean is
removed from the right-hand side, from every preconditioned residual,
and from the returned solution, pinning the mean-free gauge — the same
``k = 0`` gauge ``SpectralSolve``'s ``where_zero=0`` selects.

The initial guess is zero unless an explicit ``x0`` is passed.

Exact convergence under fixed iterations
----------------------------------------
Because there is no tolerance break, the recurrence keeps running
after the residual reaches exact zero (a zero right-hand side, or an
exact preconditioner such as the flat spectral inverse on a
constant-metric mapped grid, stage C3). The scalar ratios
``alpha = rz / <p, Ap>`` and ``beta = rz_new / rz`` then divide zero
by zero; both are computed through a guarded division that returns
**zero** when the denominator is exactly zero, which turns every
post-convergence iteration into an exact no-op (``x`` and ``r``
unchanged) instead of poisoning the solve with NaNs. For nonzero
denominators the guard is bitwise-neutral.
"""
# CS-D2 (stage C3): matrix-free preconditioned CG, fixed iterations
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.spatial.operators.base import FieldLike


def _guarded_ratio(num: jax.Array, den: jax.Array) -> jax.Array:
    """
    Return ``num / den``, or exact zero for a zero denominator.

    Description
    -----------
    The post-convergence guard of the fixed-iteration recurrence
    (module docstring): a zero denominator only arises when the
    iteration has already converged exactly, and a zero ratio makes
    the remaining iterations exact no-ops. The zero branch is
    selected through the double-``where`` pattern so reverse-mode
    gradients stay NaN-free.

    Parameters
    ----------
    num : jax.Array
        The 0-d numerator.
    den : jax.Array
        The 0-d denominator.

    Returns
    -------
    jax.Array
        The 0-d guarded ratio.
    """
    zero = den == 0.0
    safe = jnp.where(zero, 1.0, den)
    return jnp.where(zero, 0.0, num / safe)


class ConjugateGradient:

    r"""
    Matrix-free preconditioned conjugate-gradient solve of ``A x = b``.

    Description
    -----------
    A setup / trace-time object (constructed once, carrying no mutable
    state): it captures the system operator, an optional
    preconditioner, and the static iteration count, then applies to a
    right-hand-side field. ``cg(rhs)`` returns the solution;
    ``cg.solve(rhs)`` returns ``(solution, info)`` with the final
    (measure-weighted) residual norm and the iteration count.

    ``A`` must be symmetric positive definite in the measure-weighted
    :math:`L^2` product (module docstring); the preconditioner
    ``M_inv``, when given, should approximate ``A``'s inverse and be
    SPD in the same product (the exact
    :class:`~fridom.spatial.operators.spectral_solve.SpectralSolve`
    inverse converges the iteration in one step). Both are consumed as
    field-to-field callables on the operand's space.

    Parameters
    ----------
    operator : Callable[[FieldLike], FieldLike]
        The SPD system operator ``A`` (an ``Operator``, or any callable
        mapping a field to a field on the same space).
    preconditioner : Callable[[FieldLike], FieldLike] | None, optional
        The preconditioner ``M_inv`` approximating ``A`` inverse; None
        runs unpreconditioned CG (identity preconditioner)
        (default: None).
    iterations : int
        The fixed number of CG iterations (static; ``>= 1``). There is
        no tolerance break (CS-D2).
    project_mean : bool, optional
        Subtract the measure-weighted mean from the right-hand side,
        the preconditioned residuals, and the solution — the constants
        nullspace projection for a singular (Neumann/periodic) Poisson
        problem (default: False).
    """

    def __init__(
        self,
        operator: Callable[[FieldLike], FieldLike],
        *,
        preconditioner: Callable[[FieldLike], FieldLike] | None = None,
        iterations: int,
        project_mean: bool = False,
    ) -> None:
        """Validate and store the operator, preconditioner, budget."""
        if not callable(operator):
            raise TypeError(
                f"operator must be a field-to-field callable, got "
                f"{operator!r}")
        if preconditioner is not None and not callable(preconditioner):
            raise TypeError(
                "preconditioner must be a field-to-field callable or "
                f"None, got {preconditioner!r}")
        if isinstance(iterations, bool) or not isinstance(
                iterations, int):
            raise TypeError(
                f"iterations must be an int, got {iterations!r}")
        if iterations < 1:
            raise ValueError(
                f"iterations must be >= 1 (fixed count, CS-D2), got "
                f"{iterations}")
        self._operator: Callable[[FieldLike], FieldLike] = operator
        self._preconditioner: (
            Callable[[FieldLike], FieldLike] | None) = preconditioner
        self._iterations: int = iterations
        self._project_mean: bool = bool(project_mean)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def operator(self) -> Callable[[FieldLike], FieldLike]:
        """The SPD system operator ``A``."""
        return self._operator

    @property
    def preconditioner(
        self,
    ) -> Callable[[FieldLike], FieldLike] | None:
        """The preconditioner ``M_inv`` (None = unpreconditioned)."""
        return self._preconditioner

    @property
    def iterations(self) -> int:
        """The fixed CG iteration count (static)."""
        return self._iterations

    @property
    def project_mean(self) -> bool:
        """Whether the constants nullspace is projected out."""
        return self._project_mean

    # ================================================================
    #  Recurrence helpers
    # ================================================================
    def _dot(self, a: FieldLike, b: FieldLike) -> jax.Array:
        r"""
        Measure-weighted inner product :math:`\int a\,b\,\mathrm{d}V`.

        Description
        -----------
        Evaluated through the field's ``integrate`` (true DOFs times
        ``grid.measure``, cross-shard sum included), so it is
        halo-clean under decomposition and the product in which ``A``
        is SPD (module docstring). Returns the 0-d scalar.

        Parameters
        ----------
        a : FieldLike
            The left operand field.
        b : FieldLike
            The right operand field.

        Returns
        -------
        jax.Array
            The 0-d weighted inner product.
        """
        return jnp.sum((a * b).integrate().data)

    def _project(self, f: FieldLike) -> FieldLike:
        """Remove the weighted mean when projecting the nullspace."""
        if self._project_mean:
            return f - f.mean()
        return f

    def _precondition(self, r: FieldLike) -> FieldLike:
        """Apply the preconditioner (identity when unpreconditioned)."""
        if self._preconditioner is None:
            return r
        return self._preconditioner(r)

    # ================================================================
    #  Application
    # ================================================================
    def __call__(
        self, rhs: FieldLike, x0: FieldLike | None = None,
    ) -> FieldLike:
        """
        Solve ``A x = rhs`` and return the solution field.

        Parameters
        ----------
        rhs : FieldLike
            The right-hand-side field on the operator's space.
        x0 : FieldLike | None, optional
            The initial guess; None starts from zeros (default: None).

        Returns
        -------
        FieldLike
            The solution field on the same space.
        """
        return self.solve(rhs, x0)[0]

    def solve(
        self, rhs: FieldLike, x0: FieldLike | None = None,
    ) -> tuple[FieldLike, dict[str, object]]:
        r"""
        Solve ``A x = rhs`` for ``x`` (fixed-iteration PCG).

        Description
        -----------
        The standard preconditioned conjugate-gradient recurrence over
        scalar fields, run for exactly ``iterations`` steps with no
        tolerance break (CS-D2). Inner products are the measure-weighted
        :math:`L^2` product (module docstring); with ``project_mean``
        the constants nullspace is projected out of the right-hand
        side, the preconditioned residuals, and the solution.

        Parameters
        ----------
        rhs : FieldLike
            The right-hand-side field on the operator's space.
        x0 : FieldLike | None, optional
            The initial guess; None starts from zeros (default: None).

        Returns
        -------
        tuple[FieldLike, dict[str, object]]
            The solution field and an ``info`` mapping with the final
            weighted residual norm (``"residual_norm"``, a 0-d
            ``jax.Array``) and the ``"iterations"`` count.
        """
        b = self._project(rhs)
        if x0 is None:
            x = 0.0 * b
            r = b
        else:
            x = self._project(x0)
            r = self._project(b - self._operator(x))
        z = self._project(self._precondition(r))
        p = z
        rz = self._dot(r, z)
        for _ in range(self._iterations):
            ap = self._operator(p)
            alpha = _guarded_ratio(rz, self._dot(p, ap))
            x = x + alpha * p
            r = r - alpha * ap
            z = self._project(self._precondition(r))
            rz_new = self._dot(r, z)
            beta = _guarded_ratio(rz_new, rz)
            p = z + beta * p
            rz = rz_new
        x = self._project(x)
        info: dict[str, object] = {
            "residual_norm": jnp.sqrt(self._dot(r, r)),
            "iterations": self._iterations,
        }
        return x, info
