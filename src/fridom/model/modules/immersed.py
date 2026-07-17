r"""
Shared immersed-domain helper module: keep dead DOFs dead.

Description
-----------
The model-agnostic home of the immersed-boundary weighting idioms
(IP-D5). Iteration 1 ships ``MaskState``: a CONSTRAINT-stage module,
auto-added by each model factory when ``grid.immersed is not None``,
that multiplies every PROGNOSTIC field of the state by its boolean
per-space mask ``grid.immersed.mask(field.space)`` (the domain's
default slip rule). This keeps the dry DOFs zero against the modules
that legitimately do not consult the mask (Coriolis, wave makers,
pressure-gradient tendencies) without touching them — the old stack's
sync-mask mechanism, done once and model-agnostically.

The mask per prognostic space is a pure function of the grid, so the
stage materializes it in-trace (the immersed descriptor memoizes
concrete-only, like ``grid._measures``); ``bind`` precomputes only the
prognostic name tuple and validates that the grid carries an immersed
domain. Genuine cut-cell **fraction** weighting of the flux-form terms
(IP-D4) is the sibling work of the model stages, not this module.
"""
from __future__ import annotations

from fridom.model.module import Module
from fridom.model.stages import Stage, StageKind
from fridom.spatial.decomposition.halo import HaloSpec

#: the intra-kind order of the masking stage: a large sentinel so it
#: sorts **after** every physics CONSTRAINT stage (the pressure
#: projection at order 0, any diagnostic constraints in between) — the
#: dry DOFs are zeroed once, last, after the wet region is settled.
_MASK_ORDER = 1_000_000


class MaskState(Module):

    """
    Zero every prognostic field on its dry (masked) DOFs (IP-D5).

    Description
    -----------
    A CONSTRAINT-stage (S4) module masking the whole prognostic state
    against the grid's immersed domain: for each PROGNOSTIC field
    ``phi`` it writes ``phi <- phi * mask(phi.space)`` with the
    boolean per-space wet mask derived under the domain's default
    slip rule (velocity-role faces take the slip combination, cell
    fields the plain cell mask — both fall out of
    ``ImmersedDomain.mask``). It advances nothing (a pure correction,
    ``advances=()``); the coverage lint sees no claim.

    Add one instance per model when ``grid.immersed is not None``; on
    an unmasked grid the module refuses to bind (a clear assembly
    error) rather than installing a silent no-op stage.
    """

    def bind(self, table: object) -> None:
        """
        Freeze the prognostic name tuple; require an immersed grid.

        Parameters
        ----------
        table : FieldTable
            The resolved field table (its ``grid`` must carry an
            immersed domain; ``prognostic`` names the masked fields).

        Raises
        ------
        ValueError
            If the grid has no immersed domain (nothing to mask).
        """
        grid = table.grid
        if grid is None or getattr(grid, "immersed", None) is None:
            raise ValueError(
                "MaskState needs an immersed grid: attach an "
                "ImmersedDomain (Grid(..., immersed=...)) or drop the "
                "module — it masks the prognostic state against the "
                "wet region, and there is none here")
        self._names: tuple[str, ...] = tuple(table.prognostic)
        # capture the (real-grid) immersed descriptor and coordinate
        # names at bind: the masking runs a concrete mask field the
        # halo tracer cannot follow (it drops to raw arrays), so the
        # stage is halo-trace exempt (`extra_halo`) and reads the
        # descriptor from here, never from the (tracer) field grid
        self._immersed: object = grid.immersed
        self._coords: tuple[str, ...] = tuple(grid.names)

    @property
    def extra_halo(self) -> HaloSpec:
        """Exempt the masking from the halo trace (a local multiply).

        Description
        -----------
        The mask is a concrete per-space field (materialized from the
        immersed descriptor) multiplied pointwise onto each prognostic
        — a zero-stencil operation the ``HaloTracer`` cannot follow (it
        drops to ``.data``). The stage therefore declares its (zero)
        FD-stencil halo here rather than being traced, exactly like the
        mapped projection's metric multiplies (V-N2).
        """
        return HaloSpec(dict.fromkeys(self._coords, 0))

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The state-masking CONSTRAINT stage (runs last, advances none).

        The high ``order`` (:data:`_MASK_ORDER`) sorts the masking after
        every other same-kind CONSTRAINT stage — the pressure
        projection writes the prognostic velocities, so the mask must
        follow it (the composer's overlap lint requires the explicit
        order; correctness never rides list position).
        """
        return (
            Stage(kind=StageKind.CONSTRAINT,
                  fn="_mask_state", name="mask_state",
                  order=_MASK_ORDER),
        )

    def _mask_state(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Multiply every prognostic by its boolean per-space mask.

        The mask is derived on each field's own space (so a
        velocity-role face takes the slip combination and a cell
        field the cell mask), then applied as a pure array product on
        the true DOFs — dry DOFs go to zero, wet DOFs are untouched.
        """
        if not self._names:
            return {}
        immersed = self._immersed
        out: dict = {}
        for name in self._names:
            field = state[name]
            mask = immersed.mask(field.function_space)
            out[name] = field.with_data(field.data * mask.data)
        return out
