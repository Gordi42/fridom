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
halo-traced numerically (pure field arithmetic), so the module
declares no ``extra_halo``.

**Walled grids are future work**: the advective flux stencils near
rigid walls (bounded, non-periodic mesh factors) are not covered
yet, so ``bind`` rejects walled grids with a taught error — build a
linear model (``advection=False`` in ``nh.Model``) instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom.framework2 as fr

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

    def bind(self, table: object) -> None:
        """Freeze the advected set and the axis -> velocity mapping.

        Raises
        ------
        NotImplementedError
            On a walled grid (any bounded mesh factor): the
            advective flux stencils near rigid walls are future
            work, and the natural downstream failure (an operator
            dispatch mismatch deep in the flux chain) would be
            cryptic.
        """
        factors = getattr(table.grid, "factors", ())
        walled = tuple(
            name for mesh in factors for name in mesh.names
            if not getattr(mesh, "periodic", True))
        if walled:
            raise NotImplementedError(
                f"CenteredAdvection does not support walled grids "
                f"yet (bounded coordinates: {walled}); the "
                "advective flux stencils near rigid walls are "
                "future work. Build a linear model instead "
                "(advection=False in nh.Model) or drop the "
                "advection module")
        self._advected = table.select(fr.roles.ADVECTED)
        selector = table.velocity()
        # selector.labels pairs each velocity name with its axis
        self._axis_velocity = tuple(
            (axis, name) for name, axis in selector.labels)

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
