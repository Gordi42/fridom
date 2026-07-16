"""
The default stopping norm and the idempotency check (2.8).

Description
-----------
``relative_l2`` is the old ``norm_of_diff`` formula relocated to the
transforms layer (respecting D1's eviction of norms from fields;
§10.2): relative, dimensionless, parameter-free — which is exactly
why it is the explicit-kwarg default and the energy norm cannot be.
``assert_idempotent`` is idempotency's third declared consumer
(§10.2/§10.3). ``relative_imbalance`` (§10.9, paper eq. 5.2) is the
relative-imbalance helper ``eta(z) = ||(I - P) z|| / ||z||`` — the
same default norm, with an ``EnergyMetric`` one kwarg away. Owning
class spec: ``design/specs/model/classes/transforms.md`` §"relative_l2,
assert_idempotent". ``norm=nh.diagnostics.energy_norm(model)`` is one
kwarg away for dimensional stratified runs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from typing import Protocol

    from fridom.model.transforms.base import StateTransform
    from fridom.spatial.fields.vector_field import VectorField

    class Metric(Protocol):

        """A norm-provider: any object exposing ``norm(state)``."""

        def norm(self, state: VectorField) -> object:
            """Return the (real, non-negative) norm of a state."""


def _l2_norm(state: VectorField) -> jnp.ndarray:
    r"""Volume-weighted :math:`\ell_2` norm over the components."""
    total = jnp.asarray(0.0)
    for field in state:
        squared = (field.conj() * field).integrate()
        total = total + jnp.sum(jnp.real(squared.data))
    return jnp.sqrt(total)


def relative_l2(a: VectorField, b: VectorField) -> float:
    r"""
    Return the relative :math:`\ell_2` distance of two states.

    Description
    -----------
    :math:`2\,\lVert a - b\rVert_2 / (\lVert a\rVert_2 +
    \lVert b\rVert_2)`, volume-weighted (``grid.measure`` quadrature)
    over the components. Returns a host ``float`` (FixedPoint is
    host-level in iteration 1). A zero denominator (both operands
    numerically zero) returns ``0.0``.

    Parameters
    ----------
    a : VectorField
        The first state.
    b : VectorField
        The second state (identical component tuple to ``a``).

    Returns
    -------
    float
        The relative distance (dimensionless, non-negative).
    """
    diff = a - b
    numerator = 2.0 * _l2_norm(diff)
    denominator = _l2_norm(a) + _l2_norm(b)
    result = jnp.where(denominator == 0.0, 0.0,
                       numerator / denominator)
    return float(result)


def relative_imbalance(
    z: VectorField,
    projection: StateTransform,
    *,
    metric: Metric | None = None,
) -> float:
    r"""
    Return the relative imbalance :math:`\eta` of a state.

    Description
    -----------
    :math:`\eta(z) = \lVert (I - P)\,z\rVert / \lVert z\rVert` (paper
    eq. 5.2): the fraction of ``z`` **not** captured by the projection
    ``P`` — the relative imbalance when ``P`` is a slow/balanced
    projector. The residual is computed directly as
    ``z - projection(z)`` (equivalently ``(Identity - projection)(z)``,
    with ``projection``'s ``rest`` policy completing any extra
    components).

    The default norm is the volume-weighted :math:`\ell_2` underlying
    :func:`relative_l2` (dimensionless, parameter-free). Pass ``metric``
    — any object exposing ``norm(state) -> real scalar``, e.g. an
    :class:`~fridom.model.energy.EnergyMetric` — for the energy norm of
    a dimensional stratified run.

    A zero-norm input (``z`` numerically zero) returns ``0.0`` — the
    same exact-zero guard :func:`relative_l2` uses; the imbalance of
    nothing is nothing. A genuinely tiny (but nonzero) ``z`` yields a
    correspondingly ill-conditioned ratio by design.

    Parameters
    ----------
    z : VectorField
        The state whose imbalance is measured.
    projection : StateTransform
        The projector ``P`` (typically a slow/balanced projection); its
        signature must accept ``z``.
    metric : Metric | None, optional
        A norm-provider (``norm(state) -> real scalar``); ``None`` uses
        the volume-weighted :math:`\ell_2` norm (default: None).

    Returns
    -------
    float
        The relative imbalance :math:`\eta` (dimensionless,
        non-negative).
    """
    residual = z - projection(z)
    if metric is None:
        numerator = _l2_norm(residual)
        denominator = _l2_norm(z)
    else:
        numerator = jnp.asarray(metric.norm(residual))
        denominator = jnp.asarray(metric.norm(z))
    result = jnp.where(denominator == 0.0, 0.0,
                       numerator / denominator)
    return float(result)


def assert_idempotent(
    transform: StateTransform,
    state: VectorField,
    *,
    norm: Callable[[VectorField, VectorField], float] = relative_l2,
    tol: float = 1e-9,
) -> None:
    r"""
    Assert ``norm(T(T(s)), T(s)) <= tol`` on a probe state.

    Description
    -----------
    The generic idempotency validation (one of the three declared
    consumers): applies ``T`` once and twice on ``state`` and checks
    that the second application is a no-op to ``tol``. Raises a taught
    ``AssertionError`` on failure, reporting the measured distance.

    Parameters
    ----------
    transform : StateTransform
        The transform whose idempotency is validated.
    state : VectorField
        The probe state.
    norm : Callable[[VectorField, VectorField], float], optional
        The distance used (default: :func:`relative_l2`).
    tol : float, optional
        The tolerance (default: 1e-9).
    """
    once = transform(state)
    twice = transform(once)
    distance = norm(twice, once)
    if distance > tol:
        raise AssertionError(
            f"{type(transform).__name__} is not idempotent: "
            f"norm(T(T(s)), T(s)) = {distance:.3e} > tol={tol:.3e} "
            "(the declared idempotent flag does not hold on this "
            "probe state)")
