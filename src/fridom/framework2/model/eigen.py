r"""
Numeric eigenmodes of a linearized model via ``eigh(iML, M)``.

Description
-----------
The **numeric probe** for a linearized model's eigenmodes: the
model-agnostic, host-side spectral path (analysis, never the traced
step). It derives the per-mode system matrix ``L(k)`` directly from
the numeric tendency, so it needs no symbolic operator assembly.

For a constant-coefficient periodic model the linearized tendency
:math:`L` is translation-invariant, so per Fourier wavenumber ``k`` it
is a small :math:`m\times m` matrix :math:`L(k)` (``m`` = prognostic
components). We obtain it by a **transfer-function probe**: apply the
linearized ``z \mapsto Lz`` (``fr.linearize(model).tendency``) to
unit-impulse basis fields (one nonzero grid point per component, whose
DFT is unity at every mode) and read
:math:`L(k)[c',c] = \mathrm{FFT}(\text{response}_{c'})[k]` per mode.

Because ``L`` is ``M``-skew-adjoint under the energy metric
(``fr.EnergyMetric``), :math:`H := iML` is Hermitian and
:math:`L q = i\omega q \Leftrightarrow H q = -\omega M q`. We solve the
**generalized Hermitian eigenproblem** :math:`H q = \mu M q` batched
over modes: Cholesky-whiten the (diagonal, positive) metric
:math:`M = R^{H}R`, run a standard ``eigh`` on
:math:`R^{-H} H R^{-1}`, back-substitute :math:`q = R^{-1}\tilde q`,
and set :math:`\omega = -\mu`. This returns **real** frequencies and
**M-orthonormal** eigenvectors (including an orthonormal basis of every
degenerate eigenspace) for free.

Constraints (nonhydro pressure)
-------------------------------
The nonhydro pressure is eliminated by the Leray projection ``P``,
carried by a ``CONSTRAINT`` stage. ``model.tendency(constraints=True)``
does **not** currently expose it as a matvec — it writes the diagnostic
``p`` into the prognostic-only tendency accumulator and raises (a
framework wall recorded for H1). So this probe assembles the raw
operator ``S`` with ``constraints=False`` and probes the linear Leray
projector ``P`` separately (through the assembled schedule's
``constrain`` stage), forming the constrained, energy-conserving
operator :math:`P S P`. Its ``eigh`` spectrum reproduces the analytic
discrete inertia-gravity frequencies to machine precision (the
divergence-free nullspace shows up as exact extra zeros).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_comp, dtype_real
from fridom.framework2.model.context import StepContext
from fridom.framework2.model.energy import EnergyMetric
from fridom.framework2.model.stages import StageKind
from fridom.framework2.model.term_predicates import linearize

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.model.model import Model


class NumericEigenmodes:

    r"""
    Per-mode numeric eigenpairs of a linearized model's operator.

    Description
    -----------
    Holds the batched spectrum and eigenvectors produced by
    :func:`numeric_eigenpairs`. Both are indexed by the model's Fourier
    mode grid (the leading axes) followed by the ``m``-sized component
    axes, with ``components`` naming the row/eigenvector order.

    Parameters
    ----------
    omega : jax.Array
        Real frequencies, shape ``(*modes, m)``, sorted ascending per
        mode (``L q = i omega q``).
    q : jax.Array
        Eigenvectors, shape ``(*modes, m, m)``; ``q[..., :, j]`` is the
        M-orthonormal eigenvector for ``omega[..., j]``.
    components : tuple[str, ...]
        The prognostic component names, the ``m`` row order of ``q``.
    weights : tuple[float, ...]
        The energy-metric diagonal ``diag(M)`` in ``components`` order.
    """

    def __init__(
        self,
        omega: jax.Array,
        q: jax.Array,
        components: tuple[str, ...],
        weights: tuple[float, ...],
    ) -> None:
        """Store the batched spectrum, eigenvectors and metric."""
        self.omega = omega
        self.q = q
        self.components = components
        self.weights = weights

    # ================================================================
    #  Self-consistency diagnostics (validation helpers)
    # ================================================================
    def orthonormality_error(self) -> jax.Array:
        r"""
        Return the max deviation of :math:`q^{H} M q` from the identity.

        Description
        -----------
        The eigenvectors are M-orthonormal by construction; this is a
        residual on that property (a 0-d real array), near machine
        epsilon on a well-conditioned metric.

        Returns
        -------
        jax.Array
            ``max |q^H M q - I|`` over all modes (a 0-d array).
        """
        w = jnp.asarray(self.weights, dtype=dtype_comp())
        gram = jnp.einsum(
            "...ij,i,...ik->...jk", jnp.conj(self.q), w, self.q)
        identity = jnp.eye(len(self.components), dtype=gram.dtype)
        return jnp.max(jnp.abs(gram - identity))


def numeric_eigenpairs(
    model: Model, *, at_time: float = 0.0,
) -> NumericEigenmodes:
    r"""
    Solve ``eigh(iML, M)`` for a linearized model's numeric eigenpairs.

    Description
    -----------
    The Phase-H0 numeric probe: builds the linear variant
    ``fr.linearize(model)``, probes its per-mode operator ``L(k)`` (and
    the Leray projector, if the model carries a ``CONSTRAINT`` stage)
    via unit-impulse transfer functions, reads the energy metric ``M``
    from ``fr.EnergyMetric.from_model``, and solves the batched
    generalized Hermitian eigenproblem. Restricted to constant-
    coefficient periodic models (the ``EnergyMetric`` /
    ``Eigenmodes.from_model`` gate); a variable-coefficient or non-
    periodic block axis is out of H0 scope.

    Parameters
    ----------
    model : Model
        An assembled, Fourier-diagonalizable model (nonhydro or shallow
        water in iteration 1).
    at_time : float, optional
        Evaluation time for time-dependent parameters (default: 0.0).

    Returns
    -------
    NumericEigenmodes
        The per-mode spectrum and M-orthonormal eigenvectors.
    """
    metric = EnergyMetric.from_model(model, at_time=at_time)
    lin = linearize(model)
    prog, base0 = _rest_background(lin, at_time)
    weights = _metric_weights(metric, prog)

    t = jnp.asarray(at_time, dtype=dtype_real())
    symbol = _probe_symbol(
        lambda z: lin.tendency(z, t=at_time, constraints=False),
        base0, prog)
    projector = _leray_projector(lin, base0, prog, t)
    if projector is not None:
        symbol = projector @ symbol @ projector

    omega, q = _generalized_eigh(symbol, weights)
    return NumericEigenmodes(omega, q, prog, weights)


# ================================================================
#  Probe machinery
# ================================================================
def _rest_background(
    lin: Model, at_time: float,
) -> tuple[tuple[str, ...], VectorField]:
    """Return the prognostic names and the rest (zeroed) background."""
    prog = lin.tendency(
        lin.state, t=at_time, constraints=False).component_names
    zeroed = {
        name: lin.state[name].with_data(
            jnp.zeros_like(lin.state[name].data))
        for name in prog}
    return prog, lin.state.replace(**zeroed)


def _probe_symbol(
    apply_fn: Callable[[VectorField], VectorField],
    base0: VectorField,
    prog: tuple[str, ...],
) -> jax.Array:
    r"""
    FFT-probe a linear ``state -> state`` map into per-mode symbols.

    Description
    -----------
    Applies ``apply_fn`` to a unit-impulse basis field in each input
    component ``c`` (one nonzero grid point, whose DFT is unity at every
    mode) and reads the response's DFT, so
    ``S[..., c', c] = FFT(apply_fn(e_c)_{c'})``. Returns the batched
    ``(*modes, m_out, m_in)`` symbol tensor.
    """
    shape = base0[prog[0]].data.shape
    axes = tuple(range(len(shape)))
    columns = []
    for name in prog:
        field = base0[name]
        impulse = jnp.zeros(
            field.data.shape, dtype=dtype_real()).at[
                (0,) * field.data.ndim].set(1.0)
        response = apply_fn(base0.replace(**{name: field.with_data(
            impulse)}))
        column = jnp.stack(
            [jnp.fft.fftn(response[out].data, axes=axes)
             for out in prog], axis=-1)
        columns.append(column)
    return jnp.stack(columns, axis=-1)


def _leray_projector(
    lin: Model,
    base0: VectorField,
    prog: tuple[str, ...],
    t: jax.Array,
) -> jax.Array | None:
    r"""
    Probe the linear CONSTRAINT (Leray) projector, or ``None``.

    Description
    -----------
    ``model.tendency(constraints=True)`` cannot serve the constrained
    matvec (it writes the diagnostic ``p`` into the prognostic-only
    accumulator and raises), so the projector is probed directly through
    the assembled schedule's ``constrain`` stage — a read-only use of
    the linear variant's internals until a public ``constrain`` matvec
    (H1) exists. Returns ``None`` for an unconstrained model (e.g.
    shallow water).
    """
    schedule = lin._artifacts.schedule  # noqa: SLF001 — host-side probe
    if not schedule.kind_entries(StageKind.CONSTRAINT):
        return None
    modules = lin._carry.modules  # noqa: SLF001 — host-side probe
    stepper = lin._stepper  # noqa: SLF001 — host-side probe
    table = schedule.binding_table
    params = (table.eval_params(modules, stepper, t)
              if table is not None else {})
    ctx = StepContext(params=params, clock=t, dt=stepper.dt,
                      stage_dt=stepper.dt)
    bound = schedule.bind(modules)

    def constrain(state: VectorField) -> VectorField:
        """Embed the prognostic perturbation and run CONSTRAINT."""
        full = base0.replace(**{name: state[name] for name in prog})
        return bound.constrain(full, ctx)

    return _probe_symbol(constrain, base0, prog)


# ================================================================
#  Generalized Hermitian eigensolve
# ================================================================
def _generalized_eigh(
    symbol: jax.Array, weights: tuple[float, ...],
) -> tuple[jax.Array, jax.Array]:
    r"""
    Solve ``H q = mu M q`` for ``H = iML``, ``M = diag(weights)``.

    Description
    -----------
    Whitens the diagonal positive metric (:math:`R = \mathrm{diag}
    \sqrt{w}`), runs a batched ``eigh`` on :math:`R^{-H} H R^{-1}`,
    back-substitutes :math:`q = R^{-1}\tilde q`, sets
    :math:`\omega = -\mu`, and sorts each mode ascending.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The frequencies ``(*modes, m)`` and eigenvectors
        ``(*modes, m, m)``.
    """
    w = jnp.asarray(weights, dtype=dtype_real())
    inv_sqrt = 1.0 / jnp.sqrt(w)
    hamiltonian = 1j * w[:, None] * symbol
    whitened = inv_sqrt[:, None] * hamiltonian * inv_sqrt[None, :]
    mu, q_white = jnp.linalg.eigh(whitened)

    omega = -mu
    order = jnp.argsort(omega, axis=-1)
    omega = jnp.take_along_axis(omega, order, axis=-1)
    q_white = jnp.take_along_axis(q_white, order[..., None, :], axis=-1)
    q = inv_sqrt[:, None] * q_white
    return omega, q


def _metric_weights(
    metric: EnergyMetric, prog: tuple[str, ...],
) -> tuple[float, ...]:
    """Return the metric diagonal in prognostic-component order."""
    weights = metric.weights
    missing = [name for name in prog if name not in weights]
    if missing:
        raise ValueError(
            "the energy metric does not weight every prognostic "
            f"component; missing {missing!r} (H0 needs an energy metric "
            "covering all prognostic components)")
    return tuple(float(weights[name]) for name in prog)
