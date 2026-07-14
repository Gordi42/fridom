r"""
Exponential (ETD) time steppers.

Description
-----------
``ETDRK4`` — fourth-order exponential Runge-Kutta (Cox-Matthews) on an
eigenbasis-diagonalized linear operator — and the ``phi``-function
kernel it is built on. Design source:
``design/research/d7_1_exponential_stepper.md``.

The split is ``dX/dt = L X + N(X)``: ``L`` is the model's LINEAR
operator (every ``linear=True`` term — gravity, Coriolis, background
advection) and ``N`` the nonlinear remainder. ``L`` is diagonalized
once, host-side, by the model's own eigenbasis, so its exact
propagator is a per-mode scalar: the columns satisfy
:math:`L q = -i \omega q`, hence
:math:`e^{L h} q = e^{-i \omega h} q`.

The step is the exact variation-of-constants formula

.. math::
    X^{n+1} = e^{L h} X^n
              + \int_0^h e^{L(h-s)} N(t_n + s)\, ds

with the integral evaluated EXACTLY per mode against an RK4 stage
interpolant of ``N``. The linear part therefore carries **no**
time-discretization error at all:

- no gravity-wave CFL — ``dt`` is set by the nonlinear (advective)
  scale, not by :math:`c\,k_{\max}`;
- no damping — :math:`|e^{-i\omega h}| = 1` exactly, unlike every
  semi-implicit theta-method, which must damp the waves to stay
  stable (see the design record);
- :meth:`ETDRK4.time_discretization_effect` is the **identity**: the
  scheme propagates every linear eigenmode at exactly its
  discrete-spatial frequency, so the eigenanalysis needs no
  correction.

**The model contract.** Because the stepper supplies ``L`` itself,
the model must NOT also carry the linear terms in its tendency, or
they are counted twice. Assemble the model with the linear terms
filtered out and hand the stepper the eigenbasis of the UNFILTERED
model:

.. code-block:: python

    import fridom as fr
    import fridom.shallowwater2 as sw
    from fridom.model import term_predicates as terms

    full = sw.Model(grid=grid, csqr=1.0, coriolis=..., advection=True,
                    time_stepper=fr.model.time_steppers.AdamBashforth(dt))
    basis = sw.eigenbasis(full)          # the linear operator L
    model = sw.Model(grid=grid, csqr=1.0, coriolis=..., advection=True,
                     time_stepper=fr.model.time_steppers.ETDRK4(dt, basis),
                     term_filter=~terms.linear)   # N only

Forgetting ``term_filter`` is silently wrong physics, so the stepper
verifies it at trace time and raises :class:`LinearTermInTendencyError`
naming the offending terms.

**Time-dependent parameters.** ``L`` is frozen: the eigenbasis is a
snapshot. A time-dependent parameter that lives in ``N`` (a ``Ramp``
on ``scaling.rossby`` — the optimal-balance ramp) is fully correct,
and the RK stages are evaluated at their own clock times
(``clock.shifted(c_i * dt)``) so the scheme keeps fourth order
through it. A parameter that lives in ``L`` itself is NOT supported:
``L(t_1)`` and ``L(t_2)`` do not commute, so ``exp(L dt)`` stops
being the propagator. Keep the stiff, time-INDEPENDENT part
(gravity) in the eigenbasis and leave the time-dependent part in the
tendency; ``dt`` is then capped by that part's own frequency instead
of the gravity CFL. See the design record.
"""
from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Final

import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_real, jaxify
from fridom.model.errors import LinearTermInTendencyError
from fridom.model.stages import StageKind
from fridom.model.terms import Treatment
from fridom.model.time_steppers.base import TimeStepper
from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.model.clock import Clock
    from fridom.model.schedule import BoundSchedule

#: ``|z|`` below which the phi-functions are Taylor-summed. Their
#: closed forms cancel catastrophically near the origin (``phi3`` has a
#: ``z^3`` denominator), and the vortical modes sit at exactly z = 0.
_TAYLOR_CUT: Final[float] = 0.5
_TAYLOR_TERMS: Final[int] = 18

#: The Cox-Matthews stage nodes. ``N`` is evaluated at ``t_n``,
#: ``t_n + h/2``, ``t_n + h/2``, ``t_n + h`` — evaluating all four at
#: ``t_n`` would silently drop the scheme to first order for any
#: time-dependent parameter (the per-stage eval_params rule, 03 5.4).
_STAGE_C: Final[tuple[float, ...]] = (0.0, 0.5, 0.5, 1.0)


# ================================================================
#  The phi-function kernel
# ================================================================
def phi_functions(
    z: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Evaluate ``(phi1, phi2, phi3)`` stably at `z`.

    Description
    -----------
    The exponential-integrator kernel

    .. math::
        \varphi_1 = \frac{e^z - 1}{z}, \quad
        \varphi_2 = \frac{e^z - 1 - z}{z^2}, \quad
        \varphi_3 = \frac{e^z - 1 - z - z^2/2}{z^3}

    with limits ``1``, ``1/2``, ``1/6`` at the origin. The closed
    forms lose all significance for small ``|z|`` (at ``|z| = 1e-3``
    the ``phi3`` numerator cancels to ten digits), and the vortical
    branch sits at exactly ``z = 0``, so they are Taylor-summed
    (:math:`\varphi_k = \sum_j z^j / (j+k)!`) below
    :data:`_TAYLOR_CUT` and evaluated closed-form above it.

    Parameters
    ----------
    z : jax.Array
        The per-mode argument ``L*dt`` (here ``-i*omega*dt``).

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        ``(phi1, phi2, phi3)``, same shape as `z`.
    """
    small = jnp.abs(z) < _TAYLOR_CUT
    # a safe denominator: the unselected closed-form branch must not
    # produce a NaN that could contaminate the select
    safe = jnp.where(small, jnp.ones_like(z), z)
    exp_safe = jnp.exp(safe)
    closed = (
        (exp_safe - 1.0) / safe,
        (exp_safe - 1.0 - safe) / safe**2,
        (exp_safe - 1.0 - safe - 0.5 * safe**2) / safe**3,
    )
    near = jnp.where(small, z, jnp.zeros_like(z))
    taylor = []
    for k in (1, 2, 3):
        acc = jnp.zeros_like(near)
        for j in range(_TAYLOR_TERMS - 1, -1, -1):
            acc = acc * near + 1.0 / float(math.factorial(j + k))
        taylor.append(acc)
    return tuple(
        jnp.where(small, term, form)
        for term, form in zip(taylor, closed, strict=True))


# ================================================================
#  ETDRK4
# ================================================================
@partial(jaxify, dynamic=("dt", "_columns", "_metric", "_omega"))
class ETDRK4(TimeStepper):

    r"""
    Fourth-order exponential Runge-Kutta (Cox-Matthews).

    Description
    -----------
    Integrates the linear operator EXACTLY through its eigenbasis and
    the nonlinear remainder with an RK4 stage structure whose
    quadrature weights are the ``phi``-functions of ``z = L dt``:

    .. math::
        Q &= \tfrac{h}{2}\,\varphi_1(z/2) \\
        a &= e^{z/2} X^n + Q\, N(X^n) \\
        b &= e^{z/2} X^n + Q\, N(a) \\
        c &= e^{z/2} a + Q\,(2 N(b) - N(X^n)) \\
        X^{n+1} &= e^{z} X^n + f_1 N(X^n)
                   + 2 f_2 (N(a) + N(b)) + f_3 N(c)

    with :math:`f_1 = h(\varphi_1 - 3\varphi_2 + 4\varphi_3)`,
    :math:`f_2 = h(\varphi_2 - 2\varphi_3)` and
    :math:`f_3 = h(4\varphi_3 - \varphi_2)` — each tending to
    :math:`h/6` as :math:`z \to 0`, so the scheme degenerates to
    classical RK4 exactly when the linear operator is absent.

    A Runge-Kutta stage structure is load-bearing here, not a
    stylistic choice. The exact propagator of an OSCILLATORY ``L`` is
    neutral (:math:`|e^{z}| = 1`), so it leaves no damping margin; a
    *multistep* exponential scheme (ETD-Adams-Bashforth) must
    extrapolate ``N`` across a fast rotating phase and resonates —
    measured unstable near ``omega*dt ~ 2-3``, which capped it at
    2-7x. ETDRK4 has no cross-step phase to extrapolate and reaches
    14-255x (design record). Self-starting: the carry is the unit
    pytree ``()``.

    Parameters
    ----------
    dt : float | np.timedelta64
        The signed step size (converted once; see ``TimeStepper``).
    eigenbasis : ChannelEigenmodesBase
        The model's eigenbasis (``sw.eigenbasis(model)``), supplying
        the columns ``q``, the frequencies ``omega``, the energy
        metric and the per-plane segment layout. Walls, ``f(y)`` and
        ``c^2(y)`` are all admissible — the channel engine
        diagonalizes the bounded axis densely.

    Raises
    ------
    LinearTermInTendencyError
        At trace time, if the model still carries ``linear=True``
        terms in its tendency (they would be counted twice — once by
        the tendency and once by ``exp(L dt)``). Assemble the model
        with ``term_filter=~fr.model.term_predicates.linear``.
    """

    supported_treatments: ClassVar[frozenset[Treatment]] = (
        frozenset({Treatment.EXPLICIT}))

    def __init__(
        self, dt: float | np.timedelta64, eigenbasis: object,
    ) -> None:
        """Bind the eigenbasis arrays and its segment layout."""
        super().__init__(dt)
        from fridom.model._eigenbasis import (  # noqa: PLC0415 — deferred: avoid an eigenbasis import cycle at module load
            fourier_ops,
        )
        self._columns = jnp.asarray(eigenbasis.q)
        self._metric = jnp.asarray(eigenbasis.metric)
        self._omega = jnp.real(jnp.asarray(eigenbasis.omega))
        self._ops = fourier_ops(eigenbasis)
        self._components = tuple(eigenbasis.components)
        self._segments = tuple(
            (name, eigenbasis.slices[name].start,
             eigenbasis.slices[name].stop)
            for name in self._components)
        self._bounded = eigenbasis.grid.names.index(
            eigenbasis.bounded_axis)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def components(self) -> tuple[str, ...]:
        """The PROGNOSTIC components the eigenbasis spans."""
        return self._components

    @property
    def omega(self) -> jax.Array:
        """The per-mode linear frequencies of the eigenbasis."""
        return self._omega

    def _statics(self) -> tuple:
        """Return the fingerprint rows: the segment layout."""
        return (("components", self._components),)

    # ================================================================
    #  The spectral seam (physical <-> eigenbasis amplitudes)
    # ================================================================
    def _forward(
        self, vector: VectorField,
    ) -> tuple[jax.Array, dict]:
        """Contract a physical vector onto the eigenbasis columns."""
        coefficients = {}
        for name in self._components:
            field = vector[name]
            for op in self._ops:
                field = op.forward(field)
            coefficients[name] = field
        stacked = jnp.concatenate(
            [jnp.moveaxis(coefficients[name].data, self._bounded, -1)
             for name in self._components], axis=-1)
        amplitudes = jnp.einsum(
            "...dj,d,...d->...j", jnp.conj(self._columns),
            self._metric, stacked)
        return amplitudes, coefficients

    def _backward(
        self,
        amplitudes: jax.Array,
        coefficients: dict,
        template: VectorField,
    ) -> VectorField:
        """Synthesize a physical vector from eigen amplitudes."""
        stacked = jnp.einsum(
            "...dj,...j->...d", self._columns, amplitudes)
        result = {}
        for name, start, stop in self._segments:
            field = coefficients[name].with_data(jnp.moveaxis(
                stacked[..., start:stop], -1, self._bounded))
            for op in reversed(self._ops):
                field = op.backward(field)
            result[name] = template[name].with_data(
                jnp.real(field.data))
        return VectorField(result)

    # ================================================================
    #  The scan-body protocol
    # ================================================================
    def init(self, tendency_template: VectorField) -> tuple:
        """Return the unit carry: ETDRK4 is self-starting."""
        _ = tendency_template
        return ()

    def step(
        self,
        stepper_state: tuple[()],  # noqa: ARG002
        state: VectorField,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[tuple, VectorField, Clock]:
        """
        Advance one ETDRK4 step.

        Description
        -----------
        Four nonlinear evaluations, each at its OWN clock time
        (:data:`_STAGE_C`), combined with the ``phi``-function
        quadrature weights of ``z = -i omega dt``; the linear
        propagation rides entirely on ``exp(z)`` / ``exp(z/2)``. The
        post-advance stage groups (S3' ADVANCE, S4 CONSTRAINT) run at
        the ticked clock, as in every other stepper.

        Parameters
        ----------
        stepper_state : tuple[()]
            The unit carry ``()`` (ETDRK4 is self-starting).
        state : VectorField
            The full assembled state vector.
        stages : BoundSchedule
            The per-step stage-group view.
        clock : Clock
            The pre-step clock.

        Returns
        -------
        tuple[tuple, VectorField, Clock]
            The advanced carry entries.
        """
        _check_no_linear_terms(stages)
        dt = self.dt
        names = stages.schedule.prognostic

        z = -1j * self._omega.astype(dtype_real()) * dt
        phi1, phi2, phi3 = phi_functions(z)
        half_phi1, _, _ = phi_functions(0.5 * z)
        exp_half, exp_full = jnp.exp(0.5 * z), jnp.exp(z)
        quad = 0.5 * dt * half_phi1
        weight_1 = dt * (phi1 - 3.0 * phi2 + 4.0 * phi3)
        weight_2 = dt * (phi2 - 2.0 * phi3)
        weight_3 = dt * (4.0 * phi3 - phi2)

        state = stages.prepare(
            state, self._stage_context(stages, clock, dt, 0))
        prognostic = VectorField({name: state[name] for name in names})
        amp_x, coefficients = self._forward(prognostic)

        def at(amplitudes: jax.Array) -> VectorField:
            """Re-enter the full state at a stage's amplitudes."""
            physical = self._backward(
                amplitudes, coefficients, prognostic)
            return state.replace(**dict(physical.components))

        def tendency(stage_state: VectorField, index: int) -> jax.Array:
            """Return a stage's nonlinear tendency as amplitudes."""
            ctx = self._stage_context(stages, clock, dt, index)
            amp, _ = self._forward(stages.tendency(
                stage_state, ctx).explicit)
            return amp

        n_0 = tendency(state, 0)
        amp_a = exp_half * amp_x + quad * n_0
        n_a = tendency(at(amp_a), 1)
        amp_b = exp_half * amp_x + quad * n_a
        n_b = tendency(at(amp_b), 2)
        amp_c = exp_half * amp_a + quad * (2.0 * n_b - n_0)
        n_c = tendency(at(amp_c), 3)

        amp_next = (exp_full * amp_x + weight_1 * n_0
                    + 2.0 * weight_2 * (n_a + n_b) + weight_3 * n_c)
        state = at(amp_next)                                   # S3
        clock = clock.tick(dt)

        ctx = stages.context(clock, dt=dt, stage_dt=dt)
        if stages.schedule.kind_entries(StageKind.ADVANCE):
            state = stages.advance_stages(state, ctx)           # S3'
        state = stages.constrain(state, ctx)                    # S4
        return (), state, clock

    def _stage_context(
        self,
        stages: BoundSchedule,
        clock: Clock,
        dt: jax.Array,
        index: int,
    ) -> object:
        """Return the stage-`index` context, at its own clock time."""
        return stages.context(
            clock.shifted(_STAGE_C[index] * dt), dt=dt, stage_dt=dt)

    # ================================================================
    #  Host-side analysis
    # ================================================================
    def time_discretization_effect(
        self,
        omega: np.ndarray,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Return `omega` unchanged: ``L`` is integrated EXACTLY.

        Description
        -----------
        With ``N = 0`` the scheme collapses to
        ``X^{n+1} = e^{L dt} X^n``, so every linear eigenmode advances
        by exactly ``exp(-i omega dt)``: the discrete frequency IS the
        continuous one, and there is no correction to apply. This is
        the whole point of the scheme — an eigenanalysis built on the
        spatial symbols describes the time-stepped dynamics exactly,
        with no ``AdamBashforth``-style dispersion offset.

        Parameters
        ----------
        omega : np.ndarray
            Continuous frequencies (rad/s); any shape.
        dt : float | None, optional
            Accepted for protocol symmetry and ignored — the result
            does not depend on the step size (default: None).

        Returns
        -------
        np.ndarray
            ``omega``, as complex; same shape.
        """
        _ = dt
        return np.asarray(omega, dtype=np.complex128)

    # ================================================================
    #  Introspection
    # ================================================================
    def __repr__(self) -> str:
        """Compact host-side summary."""
        return (f"ETDRK4(dt={self.dt!r}, "
                f"modes={self._omega.shape})")


# ================================================================
#  Internals
# ================================================================
def _check_no_linear_terms(stages: BoundSchedule) -> None:
    """
    Raise if the tendency still carries ``linear=True`` terms.

    Description
    -----------
    The double-counting guard. ``ETDRK4`` supplies the linear operator
    itself (through ``exp(L dt)``), so a model that ALSO evaluates its
    linear terms in the tendency integrates them twice — silently
    wrong physics, not a crash. Runs host-side at trace time (the
    schedule is static), so it fires on the first compile.
    """
    offenders = tuple(
        entry.key for entry in stages.schedule.entries
        if entry.is_term and entry.linear)
    if offenders:
        raise LinearTermInTendencyError(terms=offenders)
