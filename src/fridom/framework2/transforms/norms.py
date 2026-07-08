"""
The default stopping norm and the idempotency check (2.8).

Description
-----------
``relative_l2`` is the old ``norm_of_diff`` formula relocated to the
transforms layer (respecting D1's eviction of norms from fields;
§10.2): relative, dimensionless, parameter-free — which is exactly
why it is the explicit-kwarg default and the energy norm cannot be.
``assert_idempotent`` is idempotency's third declared consumer
(§10.2/§10.3). Owning class spec:
``notes/framework2/model/classes/transforms.md`` §"relative_l2,
assert_idempotent". ``norm=nh.diagnostics.energy_norm(model)`` is one
kwarg away for dimensional stratified runs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.transforms.base import StateTransform


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
