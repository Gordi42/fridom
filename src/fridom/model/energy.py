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

A field weight whose source field is declared ``time_dependent`` (a
``ProfileFunction`` ``csqr(y, t)`` / ``N^2(y, t)``) is stored not as
the ``t = 0`` snapshot but as a **state-sourced weight** descriptor
(:class:`StateSourcedWeight`): "read component ``csqr`` off the
operand, weight is that field (or its reciprocal)". :meth:`apply` /
:meth:`inner` resolve it from the operand — which, post-TDF, carries
its own stage-time values — so the metric is evaluated at the
measured state's own time with no clock plumbing (TDF-D10). The
frozen-snapshot spelling is ``from_model(..., snapshot=True)`` (the
eigen/channel family, which needs the metric matching a frozen
basis).

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

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.model.time_dependent import resolve_at
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.spaces.coefficient import CoefficientSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

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


# ----------------------------------------------------------------
#  State-sourced weight (TDF-D10): a field weight whose source is a
#  ``time_dependent`` AUXILIARY field is not baked at build time but
#  resolved off the operand's own stage-time state at apply time.
#  The transforms are module-level (hashable by identity), so the
#  descriptor stays a frozen, static host object — never a pytree.
# ----------------------------------------------------------------
def _identity(field: ScalarField) -> ScalarField:
    """Identity weight transform (the ``csqr`` velocity weight)."""
    return field


def _reciprocal(field: ScalarField) -> ScalarField:
    """Reciprocal weight transform (the ``1/N^2`` buoyancy weight)."""
    return 1.0 / field


@dataclass(frozen=True)
class StateSourcedWeight:

    r"""
    A field weight read off the operand state at apply time (TDF-D10).

    Description
    -----------
    The stored form of a field weight whose source field is declared
    ``time_dependent`` (a ``ProfileFunction`` ``csqr(y, t)`` /
    ``N^2(y, t)``). Rather than baking the ``t = 0`` snapshot into the
    metric, the descriptor names the source component and the transform
    that turns it into the weight; :meth:`EnergyMetric.apply` /
    :meth:`EnergyMetric.inner` read ``operand[field]`` and apply ``fn``,
    so the metric tracks the measured state's own stage time with no
    clock plumbing. Frozen (hashable, not a pytree) so it composes with
    the static host-side metric.

    Parameters
    ----------
    field : str
        The source component read off the operand (``"csqr"`` / ``"n2"``).
    fn : Callable
        The weight transform applied to the resolved source field
        (:func:`_identity` for ``csqr``, :func:`_reciprocal` for
        ``1/N^2``).
    """

    field: str
    fn: Callable[[ScalarField], ScalarField]


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
    weights : Mapping[str, float | ScalarField | StateSourcedWeight]
        Per-component energy weights; scalars, one-DOF constant fields
        (the ``diag(M)`` entries), or a state-sourced descriptor
        resolved off the operand at apply time.
    """

    def __init__(
        self,
        weights: Mapping[str, Weight | ScalarField | StateSourcedWeight],
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
    def weights(
        self,
    ) -> Mapping[str, Weight | ScalarField | StateSourcedWeight]:
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
        the coefficient. A **state-sourced** weight (TDF-D10) is first
        resolved off ``state`` itself: its source field must be present
        (the ``if name in state`` skip keeps unweighted components out,
        then the source read is gated on it) or a taught error names
        the snapshot spelling.

        Parameters
        ----------
        state : VectorField
            The state (or any component collection) to weight; also the
            operand a state-sourced weight resolves from.

        Returns
        -------
        VectorField
            ``self.apply(state)``; the same container type, weighted
            components replaced, others passed through unchanged.
        """
        scaled = {
            name: _weigh(self._resolve(weight, state), state[name])
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
        result is a single scalar (0-d ``jax`` array). A ``z``-constant
        barotropic leg (the hydrostatic ``ps``) carries its physical
        column depth ``H`` in the **weight** (``H/c^2``), since
        ``integrate`` gives a ``ConstantSpace`` factor no depth; on a
        terrain grid that ``H(x, y)`` is a field weight sampled through
        ``_weigh`` like any other. (Known gap: the Parseval leg still
        wants an ``sqrt_g`` factor on mapped grids — a separate
        follow-up, not this change.)

        A **state-sourced** weight (TDF-D10) resolves off ``b`` — the
        operand the weight multiplies (``= a`` for :meth:`norm`) — so
        the energy is measured at ``b``'s own stage time. A cross-time
        inner product (``a`` and ``b`` at different times) is not an
        energy and is out of scope; ``b`` alone sources the weight.

        Parameters
        ----------
        a : VectorField
            The left operand (conjugated).
        b : VectorField
            The right operand (weighted); also the operand a
            state-sourced weight resolves from.

        Returns
        -------
        jax.Array
            The (generally complex) inner product, a 0-d array.
        """
        spectral = self._is_spectral(a[self.component_names[0]])
        total = jnp.asarray(0.0 + 0.0j)
        for name, weight in self._weights.items():
            # the weight multiplies b, so a state-sourced weight
            # resolves off b (= a for norm); a cross-time inner product
            # is not an energy and is out of scope.
            resolved = self._resolve(weight, b)
            a_c, b_c = a[name], b[name]
            if spectral:
                if isinstance(resolved, ScalarField):
                    raise NotImplementedError(
                        "a coefficient-space state has no Parseval "
                        "reduction under a varying (field-valued) "
                        "energy weight — the weight is not diagonal "
                        "in the transformed basis; reduce the "
                        "physical state instead")
                volume = _spectral_volume(a_c)
                contrib = resolved * volume * jnp.sum(
                    jnp.conj(a_c.data) * b_c.data)
            else:
                term = a_c.conj() * _weigh(resolved, b_c)
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
        snapshot: bool = False,
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

        Scalar weights are baked once here and never re-read from the
        live state: a ``Ramp`` scalar is frozen at ``at_time``
        (default 0.0) — the deliberate constancy of the adiabatic-
        ramping reference metrics (a clock is not readable off a bare
        component bundle). A **static** profile field weight is
        likewise captured as the ``model.state`` snapshot at build
        time.

        A field weight whose source field is declared
        ``time_dependent`` (a ``ProfileFunction`` ``csqr`` / ``N^2``,
        marked on its ``FieldRecord``) is stored **state-sourced** by
        default (TDF-D10): the metric holds a :class:`StateSourcedWeight`
        descriptor, and :meth:`apply` / :meth:`inner` read the source
        component off the operand — which, post-TDF, carries its own
        stage-time values — so the metric is automatically evaluated at
        the measured state's own time. Pass ``snapshot=True`` for the
        frozen-analysis spelling: it reproduces today's build-time
        baking everywhere (whatever ``csqr`` / ``N^2`` held at
        ``at_time``), the metric a frozen eigen/channel basis needs
        (TDF-D6, no re-diagonalization). The hydrostatic terrain depth
        weight ``H(x, y)`` is grid-metric-derived, not state-resident,
        and keeps snapshot semantics regardless.

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
        snapshot : bool, optional
            Whether to freeze a ``time_dependent`` field weight at its
            build-time value (``at_time``) instead of storing the
            state-sourced descriptor; the eigen/channel family passes
            ``True`` so a frozen basis keeps its matching frozen metric
            (TDF-D6, default: False).

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
                b_weight = _weight_or_source(
                    model, "n2", 1.0 / n2_field, _reciprocal,
                    snapshot=snapshot)
                weights = {
                    "u": 1.0, "v": 1.0, "w": dsqr, "b": b_weight}
        elif _CSQR in params:
            csqr = _read_scalar(params, _CSQR, at_time)
            if csqr == 0.0:
                raise ValueError(
                    "the shallow-water energy weight 1/c^2 needs a "
                    "nonzero phase speed 'shallowwater.csqr'")
            weights = {"u": 1.0, "v": 1.0, "p": 1.0 / csqr}
        elif _HYDRO_CSQR in params:
            weights = _hydrostatic_weights(
                model, at_time, allowed=allow_field_weights)
        elif _state_field(model, "csqr") is not None:
            csqr_field = _profile_field(
                model, "csqr", _CSQR, allowed=allow_field_weights)
            csqr_weight = _weight_or_source(
                model, "csqr", csqr_field, _identity, snapshot=snapshot)
            weights = {
                "u": csqr_weight, "v": csqr_weight, "p": 1.0}
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
    def _resolve(
        weight: Weight | ScalarField | StateSourcedWeight,
        operand: VectorField,
    ) -> Weight | ScalarField:
        r"""Resolve a state-sourced weight off ``operand``; else pass through.

        Description
        -----------
        The TDF-D10 resolution seam: a :class:`StateSourcedWeight`
        reads its source component off ``operand`` and applies its
        transform, yielding a ``ScalarField`` that flows into the
        existing ``_weigh`` / Parseval branches exactly like a baked
        field weight. A scalar or already-baked field weight passes
        through unchanged. A missing source component (an eigenmode
        basis vector, a bare ``(u, v, p)`` bundle) is a taught error
        naming the snapshot spelling.
        """
        if not isinstance(weight, StateSourcedWeight):
            return weight
        if weight.field not in operand:
            raise ValueError(
                f"the state-sourced energy weight reads component "
                f"{weight.field!r} off the operand, but the operand "
                f"does not carry it (an eigenmode basis vector or a "
                f"bare component bundle has no such field). Build the "
                f"metric with EnergyMetric.from_model(..., "
                f"snapshot=True) to freeze the weight at build time")
        return weight.fn(operand[weight.field])

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
    model: Model, at_time: float, *, allowed: bool,
) -> dict[str, Weight | ScalarField]:
    r"""Assemble ``diag(1, 1, 1/N^2, H/c^2)`` on ``(u, v, b, ps)``.

    Description
    -----------
    The hydrostatic energy metric: unit weight on the horizontal
    velocities, ``1/N^2`` on the buoyancy tracer and the
    **depth-weighted** ``H/c^2`` on the surface pressure ``ps`` (the
    barotropic phase speed ``hydrostatic.csqr``). The ``ps`` field is
    ``z``-constant (a ``fr.Profile``), so ``integrate`` gives it no
    depth (the ``ConstantSpace`` reduction is the identity — the
    physical-integral ruling); its depth must therefore ride the
    **weight**. ``H`` is the physical column depth: the vertical axis
    physical extent (a scalar) on a flat or stretched-only mesh, and
    the column-Jacobian integral ``\int J\,\mathrm{d}z`` on a ``maps=``
    vertical column — a scalar on a horizontally-uniform (stretched-z)
    map, a field ``H(x, y)`` on a terrain-following one. That ``H``
    factor pairs ``-grad ps`` with the depth-mean divergence into an
    exactly skew-adjoint operator (the dense-column channel engine
    keeps ``H`` in this weight too and reduces the ``ps`` bounded-axis
    measure to unity accordingly).

    A terrain depth weight ``H(x, y)`` is field-valued and breaks
    translation invariance along the periodic axes, so — like the
    ``csqr(y)`` / ``N^2(y)`` profile weights — it enters only when the
    caller opts in (``allow_field_weights=True``); otherwise the
    varying case is a taught error.
    """
    params = model.parameters
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
    ps_weight = _ps_depth_weight(model, csqr, allowed=allowed)
    return {"u": 1.0, "v": 1.0, "b": 1.0 / n2, "ps": ps_weight}


def _ps_depth_weight(
    model: Model, csqr: float, *, allowed: bool,
) -> Weight | ScalarField:
    r"""Return the depth-weighted surface-pressure weight ``H/c^2``.

    Description
    -----------
    ``H`` is the physical column depth: the vertical axis physical
    extent on a flat / stretched-only mesh (a scalar), and the
    column-Jacobian integral on a ``maps=`` vertical column — a scalar
    when the map is horizontally uniform (stretched-z), a field
    ``H(x, y)`` on a terrain-following grid. A terrain depth field is
    gated behind ``allowed`` (the ``allow_field_weights`` opt-in), the
    same rejection the ``csqr(y)`` / ``N^2(y)`` profile weights carry.
    """
    grid = model.grid
    vertical = _vertical_axis(model)
    column = _vertical_column(grid, vertical)
    if column is None:
        return _vertical_extent(grid, vertical) / csqr
    depth = _physical_column_depth(model, vertical, column)
    scalar = _uniform_scalar(depth)
    if scalar is not None:
        return scalar / csqr
    if not allowed:
        raise ValueError(
            "the hydrostatic surface-pressure depth weight varies with "
            "horizontal position (a terrain-following column depth "
            "H(x, y)): a field weight breaks translation invariance "
            "along the periodic axes, so a translation-invariant "
            "consumer cannot serve it. Reduce the physical state "
            "directly (EnergyMetric.inner integrates the depth) with "
            "allow_field_weights=True")
    return depth * (1.0 / csqr)


def _vertical_axis(model: Model) -> str:
    """Return the ``ps`` depth axis (its ``ConstantSpace`` factor).

    Description
    -----------
    The vertical is the axis ``ps`` is constant along — read off the
    barotropic ``ConstantSpace`` factor of its own function space, so
    a grid with several bounded axes (a walled horizontal channel)
    resolves correctly. Without a ``ps`` state field the single
    bounded (non-periodic) axis is the fallback.
    """
    ps = _state_field(model, "ps")
    if ps is not None:
        constant = [
            name for name in ps.function_space.bare.names
            if getattr(ps.function_space.factor(name), "is_constant",
                       False)]
        if len(constant) == 1:
            return constant[0]
    bounded = [
        name for mesh in model.grid.factors
        if not getattr(mesh, "periodic", True)
        for name in mesh.names]
    if len(bounded) != 1:
        raise ValueError(
            "the hydrostatic energy metric needs exactly one bounded "
            "(vertical) axis for the ps depth weight when no "
            f"z-constant 'ps' names it; this grid bounds {bounded!r}")
    return bounded[0]


def _vertical_extent(grid: object, vertical: str) -> float:
    """Physical extent ``H`` of the vertical mesh axis (a scalar)."""
    for mesh in grid.factors:
        if vertical in mesh.names:
            lo, hi = mesh.extent
            return float(hi - lo)
    raise ValueError(  # pragma: no cover — vertical is a grid axis
        f"the vertical axis {vertical!r} is not a grid factor")


def _vertical_column(
    grid: object, vertical: str,
) -> tuple[str, str] | None:
    """Return the ``(mapped, base)`` vertical column, or ``None``.

    Description
    -----------
    ``None`` off a ``maps=`` grid (a flat mesh, or a stretched-only
    ``MappedIntervalMesh`` whose stretching already rides
    ``grid.measure`` — its ``mesh.extent`` is the physical depth). A
    single-base analytic column whose base is the vertical axis
    (``zp = z * H(x, y)``) returns its ``(mapped, base)`` pair.
    """
    mapping = getattr(grid, "mapping", None)
    if mapping is None:
        return None
    entry = mapping.column_corrections.get(vertical)
    if entry is None or entry[1] != vertical:
        return None
    return entry


def _physical_column_depth(
    model: Model, vertical: str, column: tuple[str, str],
) -> ScalarField:
    r"""Return ``H = \int J\,\mathrm{d}z`` on the ``ps`` cell.

    Description
    -----------
    The plain vertical integral of the column Jacobian
    ``J = d<mapped>_d<base>`` (rules 2.7) at the buoyancy cell's
    horizontal staggering (which ``ps`` shares) — the physical column
    depth, landing on the barotropic ``ConstantSpace`` z-factor.
    """
    ref = _state_field(model, "b")
    if ref is None:
        raise ValueError(
            "the hydrostatic ps depth weight on a mapped (terrain / "
            "stretched-z) grid integrates the column Jacobian and needs "
            "the buoyancy field 'b' in the model state")
    mapped, base = column
    jac = ref.grid.metric(ref.function_space.bare, f"d{mapped}_d{base}")
    return Integral()[vertical](jac)


def _uniform_scalar(field: ScalarField) -> float | None:
    """Collapse a ``z``-constant depth field to a scalar if uniform.

    Description
    -----------
    The physical column depth is a ``z``-constant field; when it does
    not vary along the horizontal axes (a flat-bottom or stretched-z
    map) it is a single scalar depth, returned as a float. A genuine
    terrain column varies horizontally and returns ``None`` (the
    field-valued case).
    """
    data = field.data
    spread = float(jnp.max(data) - jnp.min(data))
    scale = float(jnp.max(jnp.abs(data)))
    if spread <= 1e-12 * max(scale, 1.0):
        return float(jnp.max(data))
    return None


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


def _weight_or_source(
    model: Model,
    name: str,
    baked: Weight | ScalarField,
    fn: Callable[[ScalarField], ScalarField],
    *,
    snapshot: bool,
) -> Weight | ScalarField | StateSourcedWeight:
    r"""Return a state-sourced descriptor for a marked field, else ``baked``.

    Description
    -----------
    The TDF-D10 build-time switch: a field weight whose source field is
    declared ``time_dependent`` (the ``FieldRecord.time_dependent``
    flag, read off ``model.field_table[name]`` — the composed record,
    not the ``model.state`` field, which carries no such attribute) is
    stored as a :class:`StateSourcedWeight` so the metric tracks the
    operand's stage time. ``snapshot=True``, or a non-marked (static)
    profile, keeps the pre-baked weight bit-identically.
    """
    if not snapshot and getattr(
            model.field_table[name], "time_dependent", False):
        return StateSourcedWeight(name, fn)
    return baked


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
