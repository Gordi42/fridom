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

- **Static trace, either mode.** The recurrence runs a Python-static
  number of iterations, so the solve jit-compiles once across
  right-hand-side *values* and is reverse-mode differentiable
  (``jax.grad`` flows through it) without a ``custom_vjp``. The
  **default** is a convergence break at ``tolerance = 1e-8``
  (*The convergence break*, below), under which ``iterations`` is the
  *maximum* budget. The explicit opt-out ``tolerance=None`` runs a
  fixed static count with no break; both modes keep the static trace
  and the exact gradient.
- **Everything pure.** No Python-side state is mutated and nothing
  branches on a traced value, so the solver is safe inside a
  jit-compiled tendency container.

The scanned recurrence (ROADMAP 3.6)
------------------------------------
The iteration was originally a plain Python loop, fully unrolled into
the trace. That made tracing and XLA compilation **O(iterations)**
(measured: 245 k HLO lines and 48 s of compile at 300 iterations,
while warm execution was ~0.05 % of the cost), and a jitted
multi-device mapped solve never finished compiling at all. The loop
is now a single :func:`jax.lax.scan` body, so trace and compile are
**O(1)** in the iteration count.

``lax.scan`` — not ``lax.fori_loop`` — because reverse-mode
differentiation must keep flowing through the solve; ``scan`` is
differentiable, ``fori_loop`` is not.

**The carry is raw arrays, not fields.** A ``ScalarField``'s
``halo_valid`` is static aux data that *participates in the pytree
treedef* (it drives sync placement, so it must key the jit cache), so
carrying fields through a ``scan`` would impose treedef stability on
a quantity that operators legitimately change. The carry is therefore
a flat tuple of true-shape :class:`jax.Array`\ s (whose treedef is
trivially stable) plus the 0-d ``rz``; the body rebuilds the fields
through ``with_data``, which declares the *canonical* halo state:
zero valid ghost layers, synced at first consumption. (Storage-frame
arithmetic makes iterate claims propagate, so the rebuild *is* a
claim reset — always sound, and it keeps the body's sync placement
independent of the claim history.)

**True shape, not the storage frame — measured, not assumed.** The
zero-copy alternative (carrying ``_data`` and rebuilding via
``with_storage``) removes six pad/unpad copies per iteration, and
was tried 2026-07-14: the standalone solve improved, but inside the
model's chunked step the mapped projection regressed +21-24 % per
step at 256^3 on ONE A100 (4-GPU runs unchanged, peak memory
*lower*) — the padded carry couples the iterates' buffer lifetimes
across the scan boundary that the pad/unpad copies decouple, and
XLA:GPU's single-device buffer assignment loses more than the copies
cost. Re-attempt only with the step benchmark suite
(``benchmarks/model``, nh_mapped cases) green on one device.

**The first iteration is peeled** out of the scan and runs unrolled.
The operator and preconditioner are opaque closures that may perform
trace-time bookkeeping on their first application — resolving
registry rows, memoizing a halo exchange, or filling a caller's
per-solve metric memo (the mapped pressure solver does exactly this).
Such an entry, first created *inside* a scan body, would hold a
body-level tracer and leak out of the loop. Peeling forces every
first-application side effect to happen at the enclosing trace level,
where its residuals are ordinary closure constants that the scan
hoists. The cost is a trace of two iteration bodies instead of one —
still O(1) in the iteration count.

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

``projection=`` generalizes this to any singular operator whose
nullspace is not the global constants: the caller supplies a
field-to-field projection installed at the same three sites (IP-D6:
the immersed cut-cell operator's nullspace is the **wet-region**
constant, projected as the wet-volume-weighted mean
``p - (int theta p dV)/(int theta dV)``). ``project_mean=True`` is the
all-wet special case ``projection = lambda f: f - f.mean()``; passing
both is a construction error.

The initial guess is zero unless an explicit ``x0`` is passed.

Exact convergence under fixed iterations
----------------------------------------
Under the ``tolerance=None`` opt-out there is no tolerance break, so
the recurrence keeps running
after the residual reaches exact zero (a zero right-hand side, or an
exact preconditioner such as the flat spectral inverse on a
constant-metric mapped grid, stage C3). The scalar ratios
``alpha = rz / <p, Ap>`` and ``beta = rz_new / rz`` then divide zero
by zero; both are computed through a guarded division that returns
**zero** when the denominator is exactly zero, which turns every
post-convergence iteration into an exact no-op (``x`` and ``r``
unchanged) instead of poisoning the solve with NaNs. For nonzero
denominators the guard is bitwise-neutral.

The convergence break (the default ``tolerance``)
-------------------------------------------------
The default ``tolerance = 1e-8`` adds an early stop: the solve refines
only until the measure-weighted *true* relative residual satisfies

.. math::

    \sqrt{\langle r, r\rangle} \le
    \texttt{tolerance}\,\sqrt{\langle b, b\rangle}

(``b`` the projected right-hand side). The comparison is evaluated
squared — ``rr <= tolerance**2 * bb`` — with the threshold formed
once outside the loop; a zero right-hand side (``bb == 0``) converges
immediately without a NaN. The explicit opt-out ``tolerance=None`` is
exactly the fixed-iteration recurrence above, unchanged bit for bit —
same scan, no extra dot products, no conditional — for a caller that
needs a deterministic fixed count.

**Why 1e-8.** The default is :math:`\sqrt{\varepsilon}` for
``float64`` (the same relative-residual default Oceananigans' PCG
uses), and it sits 5-6 decades above the measured preconditioned
residual floor (``~4.5e-14`` on a strong f64 mapping), so the
tolerance always fires *before* the floor — the (T4) NaN-at-floor trap
below cannot engage in ``float64``. Solution differences against the
full fixed budget are at the ``1e-8`` relative level, far below
truncation error. The design record carries the full contraction
study.

**Masked scan, not** ``while_loop``. The stop is *not* a
``lax.while_loop`` with a dynamic trip count. The scan keeps its full
static ``iterations`` length, but each body step wraps the real CG
step in ``lax.cond(converged, identity, real_step)``, so once the
residual clears the threshold every remaining step is a no-op. Three
properties motivate this over the shorter
``while_loop`` + ``lax.custom_linear_solve`` form:

- **Exact** ``jax.grad`` **(the repo invariant).** ``scan`` + ``cond``
  differentiate the *actual truncated algorithm*, so the gradient is
  exact to finite-difference precision, matching the fixed-iteration
  path — and with no transpose machinery. A ``while_loop`` with
  ``custom_linear_solve`` instead substitutes
  implicit-function-theorem gradients whose error is proportional to
  the tolerance, and carries a silent-wrong-reverse-gradient trap
  (``symmetric=True`` gives a badly wrong ``grad`` on any operator
  that is only *measure-weighted* self-adjoint — correct today only
  because the current computational measure happens to be uniform).
- **O(1) compile.** The body is still traced once (plus the peel);
  the ``cond`` adds a bounded amount of HLO, not per-iteration
  growth.
- **Runtime skip.** ``lax.cond`` lowers to a real ``stablehlo``
  conditional (not a compute-both ``select``), so on a scalar
  predicate the skipped steps cost nothing at run time.

**Two caveats.**

- *(T4) Keep the tolerance above the residual floor.* A
  preconditioned ``float64`` solve bottoms out near ``~1e-14``
  (conservative); a tolerance *below* the achievable floor never
  fires, so the solve runs the full budget and the post-floor
  iterations divide a tiny residual by a tiny residual.
  ``_guarded_ratio`` guards only an *exact* zero, so those iterations
  keep the forward value finite but can NaN the *reverse* gradient — a
  pre-existing fixed-iteration hazard. A tolerance that actually fires
  *removes* this hazard for gradient users by stopping before the
  floor.
- *(T5) Do not* ``vmap`` *the tolerance solve.* Under ``vmap`` the
  scalar ``lax.cond`` predicate becomes a batched one and ``cond``
  degrades to a compute-both ``select`` — correct, but it forfeits
  the runtime skip and can re-enter the T4 floor on lanes that have
  already converged. Map the fixed-iteration path, or run the
  tolerance solve unbatched.
"""
# CS-D2 (stage C3): matrix-free preconditioned CG, fixed iterations
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import lax

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.spatial.operators.base import FieldLike

    #: the scan carry: the true-shape ``x``, ``r`` and ``p`` arrays
    #: plus the 0-d ``rz = <r, z>`` (module docstring)
    Carry = tuple[jax.Array, jax.Array, jax.Array, jax.Array]

    #: the ``tolerance``-mode scan carry: the fixed ``Carry`` plus the
    #: 0-d ``rr = <r, r>`` and the 0-d integer step counter ``k``
    TolCarry = tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
        jax.Array]


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
        The CG iteration count (static; ``>= 1``). Under the default
        ``tolerance`` it is the *maximum* budget and the convergence
        break stops earlier; with the ``tolerance=None`` opt-out all
        ``iterations`` steps run (CS-D2).
    tolerance : float | None, optional
        The convergence break on the measure-weighted true relative
        residual: the recurrence stops refining once
        :math:`\sqrt{\langle r, r\rangle} \le \texttt{tolerance}\,
        \sqrt{\langle b, b\rangle}` (``b`` the projected right-hand
        side) and every later scanned step is a no-op (*The convergence
        break* in the module docstring: masked scan, exact gradient,
        O(1) compile, the T4 floor and T5 no-``vmap`` caveats). The
        default ``1e-8`` is :math:`\sqrt{\varepsilon}` for ``float64``
        (Oceananigans' PCG relative-residual precedent); it sits 5-6
        decades above the measured preconditioned residual floor
        (``~4.5e-14`` on a strong f64 mapping), so it fires before the
        floor and the T4 NaN-at-floor trap cannot engage in
        ``float64``, and the resulting solution differs from the full
        fixed budget only at the ``1e-8`` relative level (far below
        truncation error). ``None`` is the explicit opt-out — the
        fixed-iteration recurrence, bit-for-bit unchanged — for a
        caller that needs a deterministic fixed count
        (default: 1e-8).
    project_mean : bool, optional
        Subtract the measure-weighted mean from the right-hand side,
        the preconditioned residuals, and the solution — the constants
        nullspace projection for a singular (Neumann/periodic) Poisson
        problem (default: False).
    projection : Callable[[FieldLike], FieldLike] | None, optional
        A custom nullspace projection applied at exactly the same
        three sites as ``project_mean`` (right-hand side, every
        preconditioned residual, the solution) — the generalization
        for a singular operator whose nullspace is not the global
        constants (IP-D6: the immersed **wet-volume-weighted mean**).
        Mutually exclusive with ``project_mean`` (default: None).
    """

    def __init__(
        self,
        operator: Callable[[FieldLike], FieldLike],
        *,
        preconditioner: Callable[[FieldLike], FieldLike] | None = None,
        iterations: int,
        tolerance: float | None = 1e-8,
        project_mean: bool = False,
        projection: Callable[[FieldLike], FieldLike] | None = None,
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
        if tolerance is not None:
            if isinstance(tolerance, bool) or not isinstance(
                    tolerance, int | float):
                raise TypeError(
                    "tolerance must be an int or float, or None for a "
                    f"fixed iteration count, got {tolerance!r}")
            if tolerance <= 0:
                raise ValueError(
                    "tolerance must be > 0 (or None for a fixed "
                    f"iteration count), got {tolerance}")
        if projection is not None and not callable(projection):
            raise TypeError(
                "projection must be a field-to-field callable or "
                f"None, got {projection!r}")
        if project_mean and projection is not None:
            raise ValueError(
                "project_mean=True and projection= are mutually "
                "exclusive: both install a nullspace projection at "
                "the same three sites — pass one (project_mean is the "
                "global-mean special case of projection)")
        self._operator: Callable[[FieldLike], FieldLike] = operator
        self._preconditioner: (
            Callable[[FieldLike], FieldLike] | None) = preconditioner
        self._iterations: int = iterations
        self._tolerance: float | None = (
            None if tolerance is None else float(tolerance))
        self._project_mean: bool = bool(project_mean)
        self._projection: (
            Callable[[FieldLike], FieldLike] | None) = projection

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
        """The CG iteration count (static; the max under a tolerance)."""
        return self._iterations

    @property
    def tolerance(self) -> float | None:
        """The convergence break (None = fixed iteration count)."""
        return self._tolerance

    @property
    def project_mean(self) -> bool:
        """Whether the constants nullspace is projected out."""
        return self._project_mean

    @property
    def projection(
        self,
    ) -> Callable[[FieldLike], FieldLike] | None:
        """The custom nullspace projection (None = no custom hook)."""
        return self._projection

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
        """Project out the nullspace (custom hook or weighted mean)."""
        if self._projection is not None:
            return self._projection(f)
        if self._project_mean:
            return f - f.mean()
        return f

    def _precondition(self, r: FieldLike) -> FieldLike:
        """Apply the preconditioner (identity when unpreconditioned)."""
        if self._preconditioner is None:
            return r
        return self._preconditioner(r)

    def _step(
        self, x: FieldLike, r: FieldLike, p: FieldLike, rz: jax.Array,
    ) -> tuple[FieldLike, FieldLike, FieldLike, jax.Array]:
        r"""
        Advance one PCG iteration (the shared recurrence body).

        Description
        -----------
        The single definition of a CG step: the peeled first
        iteration and the scanned remainder both call it, so the two
        are bitwise the same arithmetic. ``z`` is local to the step
        (it only builds the new ``p``) and is therefore not carried.
        The guarded ratios make a post-convergence step an exact
        no-op (module docstring).

        Parameters
        ----------
        x : FieldLike
            The current iterate.
        r : FieldLike
            The current residual.
        p : FieldLike
            The current search direction.
        rz : jax.Array
            The 0-d :math:`\langle r, z\rangle` of the current step.

        Returns
        -------
        tuple[FieldLike, FieldLike, FieldLike, jax.Array]
            The advanced ``(x, r, p, rz)``.
        """
        ap = self._operator(p)
        alpha = _guarded_ratio(rz, self._dot(p, ap))
        x = x + alpha * p
        r = r - alpha * ap
        z = self._project(self._precondition(r))
        rz_new = self._dot(r, z)
        beta = _guarded_ratio(rz_new, rz)
        return x, r, z + beta * p, rz_new

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
        Solve ``A x = rhs`` for ``x`` (PCG).

        Description
        -----------
        The standard preconditioned conjugate-gradient recurrence over
        scalar fields, run until the default ``tolerance`` convergence
        break fires, or for a fixed ``iterations`` steps under the
        ``tolerance=None`` opt-out (CS-D2; module docstring). Inner
        products are the
        measure-weighted :math:`L^2` product (module docstring); with
        ``project_mean`` the constants nullspace is projected out of the
        right-hand side, the preconditioned residuals, and the solution.

        The first iteration is peeled and the remaining
        ``iterations - 1`` run inside one :func:`jax.lax.scan` body
        over a raw-array carry, so the trace is O(1) in the iteration
        count (module docstring). A single-iteration solve is the
        peel alone (the scan then has length zero). Under a
        ``tolerance`` the scan still runs its full static length, but
        each step past convergence is a ``lax.cond`` no-op (the carry
        also threads the squared residual ``rr`` and the step counter).

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
            ``jax.Array``) and the ``"iterations"`` count — the static
            Python int on the fixed path, or the traced 0-d
            ``jax.Array`` step count on the ``tolerance`` path (the
            break makes it data-dependent).
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

        # ``bb`` seeds the tolerance relative-residual threshold; it
        # reads the projected right-hand side and is formed before the
        # peel (which leaves ``b`` untouched). ``None`` on the fixed
        # path pays no extra dot product
        bb = None if self._tolerance is None else self._dot(b, b)

        # the peel: one unrolled iteration, so every first-application
        # side effect of the opaque operator/preconditioner closures
        # lands in *this* trace and not in the scan body
        x, r, p, rz = self._step(x, r, p, rz)

        # the iterates are now in the canonical halo state (zero valid
        # ghosts) that field arithmetic always produces, so they can be
        # torn down to raw arrays and rebuilt inside the body from
        # these templates (grid, space and metadata are static there)
        t_x, t_r, t_p = x, r, p

        if self._tolerance is not None:
            # the tolerance convergence break (module docstring): the
            # scan keeps its full static length, but each step skips the
            # real CG step through ``lax.cond`` once the measure-weighted
            # true relative residual clears the threshold. ``k`` counts
            # the steps actually taken; the peel is step 1
            rr = self._dot(r, r)
            threshold = self._tolerance ** 2 * bb

            def real_step(s: TolCarry) -> TolCarry:
                x_d, r_d, p_d, rz_s, _rr, k_s = s
                n_x, n_r, n_p, n_rz = self._step(
                    t_x.with_data(x_d), t_r.with_data(r_d),
                    t_p.with_data(p_d), rz_s)
                return (n_x.data, n_r.data, n_p.data, n_rz,
                        self._dot(n_r, n_r), k_s + 1)

            def tol_body(
                state: TolCarry, _: None,
            ) -> tuple[TolCarry, None]:
                # state == (x, r, p, rz, rr, k); break on the residual
                _x, _r, _p, _rz, rr_c, _k = state
                converged = rr_c <= threshold
                new_state = lax.cond(
                    converged, lambda s: s, real_step, state)
                return new_state, None

            state: TolCarry = (
                x.data, r.data, p.data, rz, rr, jnp.asarray(1))
            state, _ = lax.scan(
                tol_body, state, None, length=self._iterations - 1)
            x_d, r_d, _p_d, _rz, rr_f, k_f = state
            x = self._project(t_x.with_data(x_d))
            return x, {
                "residual_norm": jnp.sqrt(rr_f),
                "iterations": k_f,
            }

        def body(carry: Carry, _: None) -> tuple[Carry, None]:
            x_d, r_d, p_d, rz_c = carry
            new_x, new_r, new_p, new_rz = self._step(
                t_x.with_data(x_d), t_r.with_data(r_d),
                t_p.with_data(p_d), rz_c)
            return (new_x.data, new_r.data, new_p.data, new_rz), None

        carry: Carry = (x.data, r.data, p.data, rz)
        carry, _ = lax.scan(
            body, carry, None, length=self._iterations - 1)
        x_d, r_d, _p_d, _rz = carry
        x = self._project(t_x.with_data(x_d))
        r = t_r.with_data(r_d)
        info: dict[str, object] = {
            "residual_norm": jnp.sqrt(self._dot(r, r)),
            "iterations": self._iterations,
        }
        return x, info
