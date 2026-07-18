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
(``design/plans/active/projection_eigenmode_plan.md`` §0-1): under
:math:`M` the linearized operator is skew-adjoint, so eigenmodes are
:math:`M`-orthogonal and every spectral projector reads
:math:`P_s z = q_s\,\langle q_s, z\rangle_M / \langle q_s, q_s\rangle_M`
(no hand-written left vector ``p`` — it is the derived :math:`M q`).

``EnergyMetric`` is model-agnostic: it holds only the per-component
weight map (``diag(1, 1, dsqr, 1/N^2)`` on ``(u,v,w,b)`` for nonhydro;
``diag(1, 1, 1/c^2)`` on ``(u,v,p)`` for shallow water — the very
``ekin``/``epot`` factors). :meth:`EnergyMetric.from_model` sources
those weights from an assembled model's parameters.

A weight may also be a **profile field** (a ``ScalarField`` on a
meridional ``fr.Profile``): the varying-coefficient metrics of the
channel engine — ``diag(c^2, c^2, 1)`` on ``(u,v,p)`` for the
variable-depth shallow water (the ``c^2`` weight moves onto the
velocities because :math:`c^2` sits inside the divergence flux;
for constant :math:`c^2` the two conventions differ by the overall
factor :math:`c^2` only) and ``diag(1, 1, dsqr, 1/N^2(y))`` for the
meridionally stratified nonhydro. A field weight is **sampled on
the component's own node set** through ``.to`` — exactly the way
the tendency samples the coefficient — wherever the metric is
applied. ``from_model`` assembles field weights only when the
caller opts in (``allow_field_weights=True``, the channel engine);
translation-invariant consumers keep the taught rejection.

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

from fridom.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.model.time_dependent import resolve_at
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.spaces.coefficient import CoefficientSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.model.model import Model
    from fridom.spatial.fields.vector_field import VectorField

# ----------------------------------------------------------------
#  Model-owned energy parameter names. The model core cannot import
#  the nonhydro2 / shallowwater2 packages (they depend on it, not the
#  other way round), so ``from_model`` dispatches on these dotted keys
#  directly and builds the ``diag(M)`` dicts inline — a deliberate
#  iteration-1 coupling until a per-model energy-weights hook exists.
#  The canonical per-model builders live beside their models
#  (``nonhydro2.energy`` / ``shallowwater2.energy``).
# ----------------------------------------------------------------
_DSQR = "nonhydro.dsqr"
_CSQR = "shallowwater.csqr"
_HYDRO_CSQR = "hydrostatic.csqr"

# A component weight is a scalar; ScalarField widens it to a
# (profile) field, sampled per component through ``.to``.
Weight = float | int | complex


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

        Description
        -----------
        A field-valued (profile) weight is sampled on the
        component's own node set through ``.to`` before the
        pointwise scaling — the same sampling the tendency uses for
        the coefficient.

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
            name: _weigh(weight, state[name])
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
        through the seeded ``integrate`` verb, so on a grid whose
        mapping derives a volume element the reduction is the
        **physical** (Jacobian-weighted) energy — the intended metric
        for the ``u`` / ``v`` / ``b`` legs on a terrain-following grid;
        on a flat grid it is the plain ``grid.measure`` quadrature. A
        coefficient-space state reduces by Parseval (the per-mode sum
        times the transformed-axis volume, ``norm="forward"``). The
        result is a single scalar (0-d ``jax`` array). (Known gap: the
        depth-integrated ``ps`` leg still wants an explicit ``H`` weight
        and the Parseval leg an ``sqrt_g`` factor on mapped grids — a
        separate follow-up, not this change.)

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
                if isinstance(weight, ScalarField):
                    raise NotImplementedError(
                        "a coefficient-space state has no Parseval "
                        "reduction under a varying (field-valued) "
                        "energy weight — the weight is not diagonal "
                        "in the transformed basis; reduce the "
                        "physical state instead")
                volume = _spectral_volume(a_c)
                contrib = weight * volume * jnp.sum(
                    jnp.conj(a_c.data) * b_c.data)
            else:
                term = a_c.conj() * _weigh(weight, b_c)
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
        cls,
        model: Model,
        *,
        at_time: float = 0.0,
        require_constant_coriolis: bool = True,
        allow_field_weights: bool = False,
    ) -> EnergyMetric:
        r"""
        Build the energy metric from an assembled model's parameters.

        Description
        -----------
        Reads the constant energy weights from ``model.parameters``
        with the same Fourier-diagonalizability gate as
        ``Eigenmodes.from_model`` (the metric feeds that projector): a
        beta-plane core provides no constant ``coriolis.f0`` and is
        rejected; ``Ramp``-valued parameters are frozen at ``at_time``.
        Nonhydro (``nonhydro.dsqr`` present) yields
        ``diag(1, 1, dsqr, 1/N^2)`` on ``(u,v,w,b)``; shallow water
        (``shallowwater.csqr`` present) yields ``diag(1, 1, 1/c^2)``
        on ``(u,v,p)``.

        A **varying** coefficient — the absent scalar provide with
        the profile field present (``csqr`` on the shallow-water
        model, ``n2`` on the nonhydro model;
        provides-implies-constancy) — is rejected with a taught
        error unless the caller opts in with
        ``allow_field_weights=True`` (the dense-column channel
        engine, whose bounded axis needs no translation invariance).
        The varying assemblies are ``diag(c^2, c^2, 1)`` on
        ``(u,v,p)`` and ``diag(1, 1, dsqr, 1/N^2(y))`` on
        ``(u,v,w,b)``, with
        the field weights sampled per component wherever the metric
        is applied.

        The weights are baked once here and never re-read from the
        live state: a ``Ramp`` scalar is frozen at ``at_time``
        (default 0.0), and a profile field weight is captured as the
        ``model.state`` snapshot at build time (whatever ``csqr`` /
        ``N^2`` held then — the ``t = 0`` materialized profile).
        :meth:`apply` / :meth:`inner` reuse those baked weights, so a
        time-dependent weight (a ramped scalar, or a ``time_dependent``
        ``csqr`` / ``N^2`` profile) does NOT track its stage-time
        values — the metric is a fixed-time analysis surface, not an
        evaluation-time read (TDF-D6).

        The weights themselves never involve the Coriolis parameter
        (rotation does no work), so a consumer that tolerates a
        varying ``f`` — the dense-column channel probe — passes
        ``require_constant_coriolis=False`` to skip that gate while
        keeping the genuine weight gates.

        Parameters
        ----------
        model : Model
            An assembled nonhydro or shallow-water model.
        at_time : float, optional
            Evaluation time for time-dependent parameters
            (default: 0.0).
        require_constant_coriolis : bool, optional
            Whether to require the constant ``coriolis.f0`` provide
            (the Fourier-diagonalizability proxy); pass ``False``
            for consumers that support a spatially varying ``f``
            (default: True).
        allow_field_weights : bool, optional
            Whether a varying coefficient (a ``csqr`` / ``n2``
            profile field without the constant scalar provide) may
            enter the metric as a field weight; ``False`` keeps the
            taught rejection for translation-invariant consumers
            (default: False).

        Returns
        -------
        EnergyMetric
            The metric with the model's energy weights.
        """
        params = model.parameters
        if require_constant_coriolis and CORIOLIS_F0 not in params:
            raise ValueError(
                "the energy metric needs a Fourier-diagonalizable "
                "model: no constant 'coriolis.f0' (a beta-plane f(y) "
                "is not supported); assemble with an f-plane Coriolis "
                "module")
        if _DSQR in params:
            dsqr = _read_scalar(params, _DSQR, at_time)
            if STRATIFICATION_N2 in params:
                n2 = _read_scalar(params, STRATIFICATION_N2, at_time)
                if n2 == 0.0:
                    raise ValueError(
                        "the nonhydro energy weight 1/N^2 needs a "
                        "nonzero stratification 'stratification.n2'")
                weights = {"u": 1.0, "v": 1.0, "w": dsqr, "b": 1.0 / n2}
            else:
                n2_field = _profile_field(
                    model, "n2", str(STRATIFICATION_N2),
                    allowed=allow_field_weights)
                weights = {
                    "u": 1.0, "v": 1.0, "w": dsqr,
                    "b": 1.0 / n2_field}
        elif _CSQR in params:
            csqr = _read_scalar(params, _CSQR, at_time)
            if csqr == 0.0:
                raise ValueError(
                    "the shallow-water energy weight 1/c^2 needs a "
                    "nonzero phase speed 'shallowwater.csqr'")
            weights = {"u": 1.0, "v": 1.0, "p": 1.0 / csqr}
        elif _HYDRO_CSQR in params:
            weights = _hydrostatic_weights(params, at_time)
        elif _state_field(model, "csqr") is not None:
            csqr_field = _profile_field(
                model, "csqr", _CSQR, allowed=allow_field_weights)
            weights = {"u": csqr_field, "v": csqr_field, "p": 1.0}
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
            not (isinstance(f, CoefficientSpace) or f.collapses_axis)
            for f in factors)
        if coefficient and physical:
            raise NotImplementedError(
                "a mixed coefficient/physical state has no "
                "iteration-1 energy reduction; transform the "
                "coefficient axes back first (Parseval on the "
                "spectral axes plus quadrature on the rest is "
                "roadmap Phase I)")
        return coefficient


def _hydrostatic_weights(
    params: Mapping[str, object], at_time: float,
) -> dict[str, Weight | ScalarField]:
    r"""Assemble ``diag(1, 1, 1/N^2, 1/c^2)`` on ``(u, v, b, ps)``.

    Description
    -----------
    The hydrostatic energy metric: unit weight on the horizontal
    velocities, ``1/N^2`` on the buoyancy tracer and ``1/c^2`` on the
    surface pressure ``ps`` (the barotropic phase speed
    ``hydrostatic.csqr``). The ``ps`` weight is depth-integrated to
    ``H/c^2`` where the metric is applied (the ``ps`` node's
    bounded-axis measure is the full depth), the factor that pairs
    ``-grad ps`` with the depth-mean divergence into a skew-adjoint
    operator.
    """
    csqr = _read_scalar(params, _HYDRO_CSQR, at_time)
    if csqr == 0.0:
        raise ValueError(
            "the hydrostatic energy weight 1/c^2 needs a nonzero "
            "barotropic phase speed 'hydrostatic.csqr'")
    n2 = _read_scalar(params, STRATIFICATION_N2, at_time)
    if n2 == 0.0:
        raise ValueError(
            "the hydrostatic energy weight 1/N^2 needs a nonzero "
            "stratification 'stratification.n2'")
    return {"u": 1.0, "v": 1.0, "b": 1.0 / n2, "ps": 1.0 / csqr}


def _weigh(
    weight: Weight | ScalarField, field: ScalarField,
) -> ScalarField:
    r"""Scale ``field`` by a weight, sampling a profile through ``.to``.

    Description
    -----------
    A field-valued weight is lifted onto the component's own node
    set first (``weight.to(field)`` — the ConstantSpace/Profile
    broadcast plus the staggered interpolation), exactly the way the
    tendency samples the coefficient; a scalar weight scales
    directly.
    """
    if isinstance(weight, ScalarField):
        return weight.to(field) * field
    return weight * field


def _state_field(model: Model, name: str) -> ScalarField | None:
    """Read a named field off the model state, or ``None``."""
    try:
        state = model.state
    except AttributeError:
        return None
    if state is None or name not in state:
        return None
    return state[name]


def _profile_field(
    model: Model, name: str, param: object, *, allowed: bool,
) -> ScalarField:
    r"""Fetch a varying coefficient field; teach the rejection.

    Description
    -----------
    The varying-coefficient detection of ``from_model``: the scalar
    provide is absent (provides-implies-constancy), so the
    coefficient must exist as the model's profile *field*. Without
    the caller's ``allow_field_weights`` opt-in the varying case is
    a taught error — a coefficient profile breaks translation
    invariance along periodic axes, so only the dense-column channel
    engine (``fr.channel_eigenpairs`` on a single-walled grid, via
    ``sw.eigenbasis`` / ``nh.eigenbasis``) can serve it.
    """
    field = _state_field(model, name)
    if field is None:
        raise ValueError(
            f"the energy metric needs a constant {param!r}; the "
            f"model provides neither the scalar nor a {name!r} "
            "profile field")
    if not allowed:
        raise ValueError(
            f"the model carries a varying {name!r} profile (no "
            f"constant {param!r} provide): this consumer needs "
            "constant coefficients — a profile breaks translation "
            "invariance along periodic axes. On a single-walled "
            "channel use the dense-column engine instead "
            "(fr.channel_eigenpairs / sw.eigenbasis / nh.eigenbasis)")
    return field


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
