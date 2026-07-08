"""Parameterful shallow-water diagnostics.

Description
-----------
Diagnostics that carry parameters are pure package-level functions
(D2.3), bound by the model into ``model.diagnostics.*`` with parameters
resolved through ``model.parameters``. Each takes ``(state, params)``
and returns a ``ScalarField``. Parameter-free diagnostics (``rel_vort``,
``divergence``) live on ``sw.State`` instead.

``ekin`` / ``epot`` are the linearized (quadratic) energies consistent
with the energy metric ``M = diag(1, 1, 1/c^2)`` on ``(u, v, p)``
(``fr.EnergyMetric``) — a single source of truth for the metric
weights. The ``DIAGNOSTICS`` mapping is contributed by
``sw.DynamicalCore`` (the diagnostics-namespace channel).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.shallowwater2.params import CSQR

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.fields.vector_field import VectorField


def _detach(state: VectorField) -> VectorField:
    """Rebuild every component to a fresh field (halo-validity reset).

    Description
    -----------
    ``model.diagnostics.X()`` evaluates on the **live carry** state,
    and interpolating a carry field mutates its halo-validity in place
    (a framework halo-bookkeeping side effect). Rebuilding through
    ``with_data`` detaches the diagnostic from the carry so a
    diagnostic call never corrupts the next ``advance``.
    """
    return state.map(lambda f: f.with_data(f.data))


def ekin(
    state: VectorField,
    params: Mapping[str, object],  # noqa: ARG001 — diagnostic protocol
) -> ScalarField:
    """Kinetic energy ``0.5 (u^2 + v^2)`` at cell center.

    Description
    -----------
    The linearized (quadratic) kinetic energy matching the energy
    metric weights ``1`` on ``u`` and ``v``. Velocities are
    interpolated onto the pressure cell.
    """
    state = _detach(state)
    center = state["p"].function_space
    u = state["u"].to(center).data
    v = state["v"].to(center).data
    return state["p"].with_data(0.5 * (u**2 + v**2))


def epot(
    state: VectorField, params: Mapping[str, object],
) -> ScalarField:
    """Potential energy ``0.5 p^2 / c^2`` at cell center.

    Description
    -----------
    The linearized (quadratic) potential energy consistent with the
    energy metric weight ``1/c^2`` on ``p``. Carries ``c^2``.
    """
    state = _detach(state)
    csqr = params[CSQR]
    p = state["p"]
    return p.with_data(0.5 * p.data**2 / csqr)


DIAGNOSTICS = {
    "ekin": ekin,
    "epot": epot,
}
