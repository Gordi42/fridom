r"""
The energy metric ``M`` and its inner product on ``State``.

Description
-----------
The model conserves a quadratic energy
:math:`E = \tfrac12\langle z, z\rangle_M` with the Hermitian,
positive-definite **energy inner product**

.. math::
    \langle a, b\rangle_M
        = \sum_c \int \overline{a_c}\,(w_c\,b_c)\; \mathrm{d}V ,

a per-component weight :math:`w_c` times the quadrature measure. This
is the missing load-bearing surface for the energy-metric projections
(``notes/framework2/projection_eigenmode_plan.md`` §0-1): under
:math:`M` the linearized operator is skew-adjoint, so eigenmodes are
:math:`M`-orthogonal and every spectral projector reads
:math:`P_s z = q_s\,\langle q_s, z\rangle_M / \langle q_s, q_s\rangle_M`
(no hand-written left vector ``p`` — it is the derived :math:`M q`).

``EnergyMetric`` is model-agnostic: it holds only the per-component
weight map (``diag(1, 1, dsqr, 1/N^2)`` on ``(u,v,w,b)`` for nonhydro;
``diag(1, 1, 1/c^2)`` on ``(u,v,p)`` for shallow water — the very
``ekin``/``epot`` factors). :meth:`EnergyMetric.from_model` sources
those weights from an assembled model's parameters.

Reduction (iteration 1): :meth:`inner` returns a single scalar (a 0-d
``jax`` array — a global inner product is inherently a number, and the
choice is uniform across the two node families). A **physical/nodal**
state reduces through ``ScalarField.integrate`` (the ``grid.measure``
trapezoidal quadrature); a **coefficient-space** state reduces by the
Parseval identity (``ScalarField.integrate`` refuses coefficient
factors), the per-mode sum times the transformed-axis volume — exact
for the ``norm="forward"`` amplitude convention on a full (complex)
spectrum. A mixed physical/coefficient state is out of iteration-1
scope.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework2.grid.spaces.coefficient import CoefficientSpace
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.framework2.model.time_dependent import resolve_at

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.model.model import Model

# ----------------------------------------------------------------
#  Model-owned energy parameter names. framework2 core cannot import
#  the nonhydro2 / shallowwater2 packages (they depend on it, not the
#  other way round), so ``from_model`` dispatches on these dotted keys
#  directly — a deliberate iteration-1 coupling until a per-model
#  energy-weights hook exists.
# ----------------------------------------------------------------
_DSQR = "nonhydro.dsqr"
_CSQR = "shallowwater.csqr"

# A component weight is a scalar or a one-DOF constant field.
Weight = float | int | complex


# ================================================================
#  Per-model energy-weight builders (the single source of truth)
# ================================================================
def nonhydro_energy_weights(
    dsqr: float, inv_n2: float,
) -> dict[str, float]:
    r"""Assemble the nonhydro energy weights ``diag(1, 1, dsqr, 1/N^2)``.

    Description
    -----------
    The canonical nonhydro energy metric ``M`` on ``(u, v, w, b)``. The
    caller passes the **already-computed** reciprocal ``inv_n2`` so the
    degenerate ``N^2 = 0`` path (which the eigenmode classes permit,
    falling back to ``1``) never divides here.

    Parameters
    ----------
    dsqr : float
        The squared aspect ratio (the ``w`` weight).
    inv_n2 : float
        The reciprocal squared buoyancy frequency ``1/N^2`` (the ``b``
        weight), computed by the caller.

    Returns
    -------
    dict[str, float]
        The ``(u, v, w, b)`` energy weights.
    """
    return {"u": 1.0, "v": 1.0, "w": dsqr, "b": inv_n2}


def shallowwater_energy_weights(inv_csqr: float) -> dict[str, float]:
    r"""Assemble the shallow-water weights ``diag(1, 1, 1/c^2)``.

    Description
    -----------
    The canonical shallow-water energy metric ``M`` on ``(u, v, p)``.
    The caller passes the **already-computed** reciprocal ``inv_csqr``
    so the degenerate ``c^2 = 0`` path (which the eigenmode class
    permits, falling back to ``1``) never divides here.

    Parameters
    ----------
    inv_csqr : float
        The reciprocal squared phase speed ``1/c^2`` (the ``p``
        weight), computed by the caller.

    Returns
    -------
    dict[str, float]
        The ``(u, v, p)`` energy weights.
    """
    return {"u": 1.0, "v": 1.0, "p": inv_csqr}


class EnergyMetric:

    r"""
    Diagonal, self-adjoint, positive energy metric ``M`` on a State.

    Description
    -----------
    Holds a per-component weight map; :meth:`apply` realizes
    :math:`M z` (each component scaled by its weight), :meth:`inner`
    the energy inner product :math:`\langle a, b\rangle_M`, and
    :meth:`norm` the induced norm. Components absent from the weight
    map are ignored (not part of the energy); a weighted component
    missing from an operand raises on access.

    Parameters
    ----------
    weights : Mapping[str, float | ScalarField]
        Per-component energy weights; scalars or one-DOF constant
        fields (the ``diag(M)`` entries).
    """

    def __init__(
        self, weights: Mapping[str, Weight | ScalarField],
    ) -> None:
        """Store a copy of the per-component weight map."""
        if not weights:
            raise ValueError(
                "an energy metric needs at least one weighted "
                "component")
        self._weights = dict(weights)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def weights(self) -> Mapping[str, Weight | ScalarField]:
        """Read-only per-component weight map (``diag(M)``)."""
        return MappingProxyType(self._weights)

    @property
    def component_names(self) -> tuple[str, ...]:
        """The weighted component names, in declaration order."""
        return tuple(self._weights)

    # ================================================================
    #  M z  — the metric applied
    # ================================================================
    def apply(self, state: VectorField) -> VectorField:
        r"""
        Return ``M z``: each weighted component scaled by its weight.

        Parameters
        ----------
        state : VectorField
            The state (or any component collection) to weight.

        Returns
        -------
        VectorField
            ``self.apply(state)``; the same container type, weighted
            components replaced, others passed through unchanged.
        """
        scaled = {
            name: weight * state[name]
            for name, weight in self._weights.items()
            if name in state}
        return state.replace(**scaled)

    def __call__(self, state: VectorField) -> VectorField:
        """Alias for :meth:`apply` (the ``M(z)`` spelling)."""
        return self.apply(state)

    # ================================================================
    #  The energy inner product and norm
    # ================================================================
    def inner(
        self, a: VectorField, b: VectorField,
    ) -> jax.Array:
        r"""
        Return the energy inner product :math:`\langle a, b\rangle_M`.

        Description
        -----------
        :math:`\sum_c \mathrm{reduce}(\overline{a_c}\,(w_c\,b_c))`
        over the weighted components. A physical/nodal state reduces
        through ``integrate`` (``grid.measure`` quadrature); a
        coefficient-space state reduces by Parseval (the per-mode sum
        times the transformed-axis volume, ``norm="forward"``). The
        result is a single scalar (0-d ``jax`` array).

        Parameters
        ----------
        a : VectorField
            The left operand (conjugated).
        b : VectorField
            The right operand (weighted).

        Returns
        -------
        jax.Array
            The (generally complex) inner product, a 0-d array.
        """
        spectral = self._is_spectral(a[self.component_names[0]])
        total = jnp.asarray(0.0 + 0.0j)
        for name, weight in self._weights.items():
            a_c, b_c = a[name], b[name]
            if spectral:
                volume = _spectral_volume(a_c)
                contrib = weight * volume * jnp.sum(
                    jnp.conj(a_c.data) * b_c.data)
            else:
                term = a_c.conj() * (weight * b_c)
                contrib = jnp.sum(term.integrate().data)
            total = total + contrib
        return total

    def norm(self, a: VectorField) -> jax.Array:
        r"""
        Return the energy norm :math:`\sqrt{\langle a, a\rangle_M}`.

        Parameters
        ----------
        a : VectorField
            The state to measure.

        Returns
        -------
        jax.Array
            The non-negative norm (a 0-d real array).
        """
        return jnp.sqrt(jnp.real(self.inner(a, a)))

    # ================================================================
    #  Construction from a model
    # ================================================================
    @classmethod
    def from_model(
        cls, model: Model, *, at_time: float = 0.0,
    ) -> EnergyMetric:
        r"""
        Build the energy metric from an assembled model's parameters.

        Description
        -----------
        Reads the constant energy weights from ``model.parameters``
        with the same Fourier-diagonalizability gate as
        ``Eigenmodes.from_model`` (the metric feeds that projector): a
        beta-plane core provides no constant ``coriolis.f0`` and is
        rejected; ``Ramp``-valued parameters are frozen at ``at_time``;
        a variable-coefficient (field-valued) weight is rejected.
        Nonhydro (``nonhydro.dsqr`` present) yields
        ``diag(1, 1, dsqr, 1/N^2)`` on ``(u,v,w,b)``; shallow water
        (``shallowwater.csqr`` present) yields ``diag(1, 1, 1/c^2)``
        on ``(u,v,p)``.

        Parameters
        ----------
        model : Model
            An assembled nonhydro or shallow-water model.
        at_time : float, optional
            Evaluation time for time-dependent parameters
            (default: 0.0).

        Returns
        -------
        EnergyMetric
            The metric with the model's constant energy weights.
        """
        params = model.parameters
        if CORIOLIS_F0 not in params:
            raise ValueError(
                "the energy metric needs a Fourier-diagonalizable "
                "model: no constant 'coriolis.f0' (a beta-plane f(y) "
                "is not supported); assemble with an f-plane Coriolis "
                "module")
        if _DSQR in params:
            dsqr = _read_scalar(params, _DSQR, at_time)
            n2 = _read_scalar(params, STRATIFICATION_N2, at_time)
            if n2 == 0.0:
                raise ValueError(
                    "the nonhydro energy weight 1/N^2 needs a nonzero "
                    "stratification 'stratification.n2'")
            weights = nonhydro_energy_weights(dsqr, 1.0 / n2)
        elif _CSQR in params:
            csqr = _read_scalar(params, _CSQR, at_time)
            if csqr == 0.0:
                raise ValueError(
                    "the shallow-water energy weight 1/c^2 needs a "
                    "nonzero phase speed 'shallowwater.csqr'")
            weights = shallowwater_energy_weights(1.0 / csqr)
        else:
            raise ValueError(
                "unrecognized model energy: expected a "
                f"{_DSQR!r} (nonhydro) or {_CSQR!r} (shallow water) "
                "provider on this model")
        return cls(weights)

    # ================================================================
    #  Internals
    # ================================================================
    @staticmethod
    def _is_spectral(field: ScalarField) -> bool:
        """Whether a field's reduction is Parseval (else physical)."""
        factors = field.function_space.factors
        coefficient = any(
            isinstance(f, CoefficientSpace) for f in factors)
        physical = any(
            not isinstance(f, CoefficientSpace | ConstantSpace)
            for f in factors)
        if coefficient and physical:
            raise NotImplementedError(
                "a mixed coefficient/physical state has no "
                "iteration-1 energy reduction; transform the "
                "coefficient axes back first (Parseval on the "
                "spectral axes plus quadrature on the rest is "
                "roadmap Phase I)")
        return coefficient


def _spectral_volume(field: ScalarField) -> float:
    r"""Transformed-axis volume :math:`\prod L` (the Parseval factor)."""
    volume = 1.0
    for factor in field.function_space.factors:
        if isinstance(factor, CoefficientSpace):
            extent = factor.mesh.extent
            volume *= float(extent[1] - extent[0])
    return volume


def _read_scalar(
    params: Mapping[str, object], name: str, at_time: float,
) -> float:
    """Read a constant scalar parameter, freezing a Ramp at ``at_time``."""
    if name not in params:
        raise ValueError(
            f"the energy metric needs a constant {name!r}; the model "
            "does not provide it")
    value = resolve_at(params[name], at_time)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"the energy weight from {name!r} must be a constant "
            "scalar (a variable-coefficient / field-valued parameter "
            "is not supported in iteration 1)") from exc
