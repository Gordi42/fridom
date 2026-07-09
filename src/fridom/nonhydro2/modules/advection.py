"""Centered, Rossby-scaled advection.

Description
-----------
``CenteredAdvection`` transports every ``ADVECTED`` component in
flux form ``A(v, q) = -div(v q) = -sum_i d_i( interp(v_i) interp(q) )``
(divergence-free velocity assumed), scaled by the Rossby number
``scaling.rossby`` (a defaulted reference, so the module stays
Ro-ignorant — D2 reconciliation 4). The flux for axis ``i`` lives on
``q``'s control-volume face in direction ``i`` (``q`` toggled along
``i``); both ``v_i`` and ``q`` are interpolated there, multiplied
(a genuine field-times-field product), and differenced back onto
``q``'s space. The chained interpolation/difference stencils are
declared through ``extra_halo``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom.framework2 as fr
from fridom.framework2.grid.decomposition.halo import HaloSpec

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.model.context import StepContext


class CenteredAdvection(fr.Module):

    """Flux-form centered advection of every ADVECTED component."""

    parameter_references = (
        fr.ParameterReference(
            fr.params.SCALING_ROSSBY, default=1.0,
            hint="Rossby number (nh.DynamicalCore)"),
    )

    def __init__(self) -> None:
        """No numeric leaves; targets resolve at bind."""
        self._advected: tuple[str, ...] = ()
        self._axis_velocity: tuple[tuple[str, str], ...] = ()
        self._coords: tuple[str, ...] = ()

    def bind(self, table: object) -> None:
        """Freeze the advected set and the axis -> velocity mapping."""
        self._advected = table.select(fr.roles.ADVECTED)
        selector = table.velocity()
        # selector.labels pairs each velocity name with its axis
        self._coords = tuple(axis for _, axis in selector.labels)
        self._axis_velocity = tuple(
            (axis, name) for name, axis in selector.labels)

    @property
    def extra_halo(self) -> HaloSpec:
        """The chained interp/difference stencils (raw-``.data`` scale)."""
        return HaloSpec(dict.fromkeys(self._coords, 2))

    def tendency_terms(self) -> tuple[fr.TendencyTerm, ...]:
        """One term advancing (and transporting) every advected field."""
        return (
            fr.TendencyTerm(
                name="advection", fn=self._advect,
                treatment=fr.Treatment.EXPLICIT,
                advances=self._advected, transports=self._advected),
        )

    def _advect(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """Flux-form transport of every advected component (Ro-scaled)."""
        ro = ctx.params[fr.params.SCALING_ROSSBY]
        out: dict[str, ScalarField] = {}
        for qname in self._advected:
            q = state[qname]
            res = None
            for axis, vname in self._axis_velocity:
                v = state[vname]
                flux_space = q.diff(axis).function_space
                flux = v.to(flux_space) * q.to(flux_space)
                divergence = flux.diff(axis)
                res = -divergence if res is None else res - divergence
            out[qname] = ro * res
        return out
