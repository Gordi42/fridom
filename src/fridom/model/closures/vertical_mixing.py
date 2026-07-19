r"""
Implicit vertical mixing (``fr.closures.VerticalMixing``).

Description
-----------
A user-wirable vertical-diffusion closure on the horizontal velocities
and the buoyancy tracer, backed by the framework's mergeable
tridiagonal :class:`~fridom.model.implicit.VerticalDiffusion` operator.
The operator is

.. math::

    \partial_t q = \partial_z (\kappa\, \partial_z q),
    \qquad q \in \{u, v, b\},

with a per-leg coefficient (vertical viscosity ``kv`` on the
velocities, vertical diffusivity ``kb`` on the buoyancy) and the
per-side wall rows ``VerticalDiffusion`` builds: zero-flux (Neumann) for
the tracer always and for the free-slip velocity default, the no-slip
(Dirichlet) wall row when ``bottom``/``top`` ask for it. That
well-posed bounded-vertical column solve is exactly what the explicit
harmonic closures (``fr.closures.HarmonicDiffusion``) cannot offer on a
walled grid. The column is measure-aware — a stretched mesh contributes
its true non-uniform spacing and a static terrain-following grid its
column Jacobian (the along-``sigma`` convention; see the class
docstring's caveat) — so no uniform-``dz`` restriction applies.

Treatment is **author-declared** on the module (the spec §5.1 override
pattern): ``treatment=fr.model.IMPLICIT`` (the default) makes the
operator an IMEX solve (CNAB2 / SBDF2 — unconditionally stable for the
stiff vertical column ``kappa dt / dz^2 >> 1``), while
``treatment=fr.model.EXPLICIT`` derives the tendency from the operator's
own ``apply`` (write-once — the same ``L @ q`` the CNAB right-hand side
uses, so flipping the treatment cannot desynchronize the two) and runs
under any explicit stepper. An IMPLICIT closure under a purely explicit
stepper is an assembly error, never a silent demotion.

The velocity leg and the buoyancy leg are two ``VerticalDiffusion``
terms of the **same** merge family (axis ``z``): the composer merges
them into one tridiagonal solve set (the Oceananigans coefficient-merge
precedent), and they merge equally with any other same-axis
``VerticalDiffusion`` a model carries (kappa-summed on a shared field).
The coefficients are provided parameters (``mixing.vertical_nu`` /
``mixing.vertical_kappa``), so ``model.update_parameters`` sweeps them
without a recompile.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

from fridom.framework.utils import jaxify
from fridom.model.errors import AssemblyError
from fridom.model.implicit import VerticalDiffusion
from fridom.model.module import Module
from fridom.model.parameters import (
    ParameterDeclaration,
    leaf,
)
from fridom.model.params import ParamName
from fridom.model.roles import TRACER, Velocity
from fridom.model.terms import TendencyTerm, Treatment
from fridom.spatial.decomposition.halo import HaloSpec

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.context import StepContext
    from fridom.model.declarations import Lifecycle
    from fridom.model.field_table import FieldTable


# ================================================================
#  Canonical parameter names (one provider per name)
# ================================================================
VERTICAL_VISCOSITY: Final[ParamName] = ParamName(
    "mixing.vertical_nu", units="m^2/s",
    hint="provided by fr.closures.VerticalMixing(kv=...)")

VERTICAL_DIFFUSIVITY: Final[ParamName] = ParamName(
    "mixing.vertical_kappa", units="m^2/s",
    hint="provided by fr.closures.VerticalMixing(kb=...)")


# ================================================================
#  Slip vocabulary (velocity leg only; tracers are always no-flux)
# ================================================================
#: map a per-wall slip choice to the column band's boundary row.
#: ``"free-slip"`` is the zero-stress Neumann row (today's behaviour);
#: ``"no-slip"`` is the Dirichlet row (odd-mirror wall value, ``-3``
#: corner). Applied to the VELOCITY leg only — a tracer has no slip, so
#: the buoyancy leg stays Neumann (no-flux) always.
_SLIP_TO_BC: Final[dict[str, str]] = {
    "free-slip": "neumann",
    "no-slip": "dirichlet",
}


def _slip_bc(side: str, which: str) -> str:
    """Return the band BC for a slip choice, raising a taught error."""
    if side not in _SLIP_TO_BC:
        raise ValueError(
            f"VerticalMixing {which}= must be 'free-slip' (zero wall "
            f"stress, the default) or 'no-slip' (Dirichlet wall), got "
            f"{side!r}")
    return _SLIP_TO_BC[side]


# ================================================================
#  The unbound coefficient callables (read live from ctx.params)
# ================================================================
def _viscosity(
    _module: object, _state: object, ctx: StepContext, _name: str,
) -> object:
    """Return the live vertical viscosity ``kv`` (velocity leg)."""
    return ctx.params[VERTICAL_VISCOSITY]


def _diffusivity(
    _module: object, _state: object, ctx: StepContext, _name: str,
) -> object:
    """Return the live vertical diffusivity ``kb`` (buoyancy leg)."""
    return ctx.params[VERTICAL_DIFFUSIVITY]


# ================================================================
#  Moving-geometry guard (static terrain is fine; a moving one is not)
# ================================================================
def _reject_moving_terrain_column(
    grid: object, axis: str, table: FieldTable,
) -> None:
    r"""
    Reject a moving terrain-following solve column (taught error).

    Description
    -----------
    The measure-aware implicit column reads its terrain Jacobian from
    ``grid.metric(..., params=None)`` — the **static** mapping geometry.
    That is exact for a static stretched or terrain-following grid, but
    ``ImplicitOperator.solve`` receives only ``(rhs, dt_gamma, ctx)`` —
    no state — so the live mapping-parameter fields a
    :class:`~fridom.model.modules.moving_geometry.MovingGeometry` module
    carries are unreachable there. Silently reading the static geometry
    while the terrain moves is exactly the silent-wrong-physics class
    the taught-error doctrine targets, so it is refused. The signal is
    the discovery convention of stage C4: a mapping parameter that also
    rides the field table as a state field (named after the parameter)
    means a moving-geometry module owns it. A static terrain column
    carries no such field and proceeds normally.

    Parameters
    ----------
    grid : object
        The solve column's grid (``mapping`` seam; anything without one
        is treated as unmapped).
    axis : str
        The solve-axis coordinate name.
    table : FieldTable
        The bound field table (queried for the moving-parameter fields).

    Raises
    ------
    NotImplementedError
        If the mapping couples the solve axis and a mapping parameter
        rides the field table as a state field.
    """
    mapping = getattr(grid, "mapping", None)
    corrections = getattr(mapping, "column_corrections", {})
    if axis not in corrections:
        # no mapping, or the mapping does not couple the solve axis: the
        # band builder never reads the terrain metric, so a static read
        # cannot go stale
        return
    moving = tuple(
        name for name in mapping.param_names if name in table)
    if moving:
        raise NotImplementedError(
            f"VerticalMixing cannot solve a MOVING terrain column along "
            f"{axis!r}: the mapping parameter(s) {moving} ride the state "
            "(a MovingGeometry module), but the implicit column solve "
            "reads STATIC geometry (grid.metric(params=None)) and has no "
            "state seam to thread the live parameters through. A static "
            "stretched or terrain column is fully supported; a moving "
            "one needs the params-through-solve seam, not yet built. "
            "Drop the closure, or freeze the geometry.")


# ================================================================
#  VerticalMixing
# ================================================================
@partial(jaxify, dynamic=("kv", "kb"))
class VerticalMixing(Module):

    r"""
    Implicit (or explicit) vertical diffusion of ``u``, ``v`` and ``b``.

    Description
    -----------
    Targets the PROGNOSTIC ``Velocity`` members with the vertical
    viscosity ``kv`` and the ``TRACER`` members with the vertical
    diffusivity ``kb`` (resolved at bind). Contributes one
    ``VerticalDiffusion`` term per leg (both on the ``vertical`` axis,
    the same merge family), whose ``treatment`` is the author-declared
    override.

    The column is measure-aware: a stretched vertical mesh gets its true
    non-uniform spacing and a static terrain-following (``maps=``) grid
    its column Jacobian, both from ``VerticalDiffusion``'s band builder,
    so no uniform-``dz`` restriction applies. **Along-coordinate
    caveat** (§3.6-B, the ROMS-default convention, owner-ratified
    2026-07-19): on a terrain-following grid the solve runs along the
    ``sigma`` coordinate, which is **tilted from the geopotential** where
    the terrain slopes — the documented along-coordinate operator, not a
    rotated / geopotential one. This is accepted practice for viscosity
    (friction is the main terrain consumer); for a tracer it is the
    spurious-diapycnal-mixing hazard of steep slopes, so along-``sigma``
    ``kb`` mixing tilts with the terrain. A **moving** terrain column is
    refused at bind (the solve reads static geometry). The
    geopotential-correct rotated closure is the recorded stage-5
    follow-up.

    Parameters
    ----------
    kv : float | fr.model.Ramp | None, optional
        Vertical viscosity (:math:`\mathrm{m^2/s}`) applied to every
        PROGNOSTIC ``Velocity`` component; ``None`` omits the velocity
        leg (default: None).
    kb : float | fr.model.Ramp | None, optional
        Vertical diffusivity (:math:`\mathrm{m^2/s}`) applied to every
        ``TRACER`` component; ``None`` omits the buoyancy leg
        (default: None).
    vertical : str, optional
        The mixing (solve) coordinate name (default: ``"z"``).
    treatment : fr.model.Treatment, optional
        The author-declared integration treatment: ``fr.model.IMPLICIT``
        (the default — an IMEX tridiagonal solve) or
        ``fr.model.EXPLICIT`` (the write-once ``apply``-derived path)
        (default: ``fr.model.IMPLICIT``).
    bottom : str, optional
        The slip condition at the bottom (low-side) wall of the
        VELOCITY leg: ``"free-slip"`` (the default — zero wall stress, a
        Neumann row) or ``"no-slip"`` (a Dirichlet wall row). Ignored by
        the buoyancy leg, which is always no-flux (default:
        ``"free-slip"``).
    top : str, optional
        The slip condition at the top (high-side) wall of the VELOCITY
        leg, same values as `bottom` (default: ``"free-slip"``).

    Raises
    ------
    ValueError
        If both ``kv`` and ``kb`` are ``None``, or ``bottom`` / ``top``
        is not a recognized slip condition.
    TypeError
        If ``treatment`` is not a ``Treatment`` member.
    """

    def __init__(
        self,
        *,
        kv: float | None = None,
        kb: float | None = None,
        vertical: str = "z",
        treatment: Treatment = Treatment.IMPLICIT,
        bottom: str = "free-slip",
        top: str = "free-slip",
    ) -> None:
        """Store the coefficient leaves, geometry, treatment and slip."""
        if kv is None and kb is None:
            raise ValueError(
                "VerticalMixing needs at least one coefficient: kv= "
                "(vertical viscosity on the velocities) or kb= "
                "(vertical diffusivity on the buoyancy tracer)")
        if not isinstance(treatment, Treatment):
            raise TypeError(
                "treatment is an author-declared fr.model.Treatment "
                f"(IMPLICIT or EXPLICIT), got {treatment!r}")
        # validate the slip choices at construction (taught error)
        self._velocity_bc: tuple[str, str] = (
            _slip_bc(bottom, "bottom"), _slip_bc(top, "top"))
        self.kv = None if kv is None else leaf(kv)
        self.kb = None if kb is None else leaf(kb)
        self._vertical = vertical
        self._treatment = treatment
        self._velocity_targets: tuple[str, ...] = ()
        self._tracer_targets: tuple[str, ...] = ()

    # ================================================================
    #  Halo-trace exemption (the EXPLICIT write-once path)
    # ================================================================
    @property
    def extra_halo(self) -> HaloSpec | None:
        """Exempt the raw-``.data`` ``apply`` from the halo trace (V-N2).

        Description
        -----------
        Under ``EXPLICIT`` treatment the tendency is the operator's own
        ``VerticalDiffusion.apply`` — a dense full-column solve on
        ``field.data`` (the vertical axis is device-local, so no ghost
        widths are consumed), which the numeric halo tracer's stand-in
        grid cannot run. Declaring an ``extra_halo`` marks the module
        halo-trace exempt (its ``apply`` is validated in the real-field
        dry run instead), the same raw-``.data`` bypass the implicit
        free surface uses for its spectral solve. Zero ghost width — the
        column op needs none — merged into the model demand. Under
        ``IMPLICIT`` treatment the term is skipped by the trace's
        treatment gate already, so no exemption is declared.
        """
        if self._treatment is Treatment.EXPLICIT:
            return HaloSpec({})
        return None

    # ================================================================
    #  References and published parameters
    # ================================================================
    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """Publish ``kv`` / ``kb`` (only the legs that are set)."""
        declarations: tuple[ParameterDeclaration, ...] = ()
        if self.kv is not None:
            declarations += (
                ParameterDeclaration(
                    VERTICAL_VISCOSITY, attr="kv", units="m^2/s",
                    doc="vertical viscosity (implicit mixing)"),)
        if self.kb is not None:
            declarations += (
                ParameterDeclaration(
                    VERTICAL_DIFFUSIVITY, attr="kb", units="m^2/s",
                    doc="vertical diffusivity (implicit mixing)"),)
        return declarations

    # ================================================================
    #  Bind: resolve the role targets (PROGNOSTIC), reject empties
    # ================================================================
    def bind(self, table: FieldTable) -> None:
        """Freeze the velocity / tracer targets from the field table.

        Raises
        ------
        AssemblyError
            If a coefficient leg resolves zero target fields (a ``kv``
            with no PROGNOSTIC velocity, or a ``kb`` with no tracer).
        NotImplementedError
            On an immersed grid, or a **moving** terrain-following solve
            column (a static stretched or terrain column is supported by
            the measure-aware band; a moving one needs the params-
            through-solve seam the implicit ``solve`` cannot reach).
        """
        from fridom.model.declarations import (  # noqa: PLC0415 — avoid an import cycle at module load
            Lifecycle,
        )
        grid = getattr(table, "grid", None)
        if getattr(grid, "immersed", None) is not None:
            raise NotImplementedError(
                "VerticalMixing does not support immersed (cut-cell) "
                "grids: its implicit column solve assumes a uniform dz "
                "(the tridiagonal band), so a partial bottom cell would "
                "silently solve the wrong operator. A wet-aware "
                "variable-dz tridiagonal is a §5 deferral "
                "(immersed_closures_sadourny_plan) — the intersection "
                "of two implicit-machinery generalizations, its own "
                "roadmap item when picked up. Drop the closure on an "
                "immersed grid.")
        _reject_moving_terrain_column(grid, self._vertical, table)
        if self.kv is not None:
            self._velocity_targets = self._prognostic(
                table, table.select(Velocity), Lifecycle)
            if not self._velocity_targets:
                raise AssemblyError(
                    "VerticalMixing got kv= but no PROGNOSTIC "
                    "Velocity field to apply it to; drop kv= or add "
                    "a velocity-carrying core")
        if self.kb is not None:
            self._tracer_targets = tuple(table.select(TRACER))
            if not self._tracer_targets:
                raise AssemblyError(
                    "VerticalMixing got kb= but no TRACER field to "
                    "apply it to; drop kb= or add a stratification "
                    "module (the buoyancy tracer)")

    @staticmethod
    def _prognostic(
        table: FieldTable,
        names: tuple[str, ...],
        lifecycle: type[Lifecycle],
    ) -> tuple[str, ...]:
        """Keep the PROGNOSTIC members of a role selection (V-H2)."""
        return tuple(
            name for name in names
            if table[name].lifecycle is lifecycle.PROGNOSTIC)

    # ================================================================
    #  The tendency terms (one VerticalDiffusion per leg)
    # ================================================================
    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """One mergeable ``VerticalDiffusion`` term per active leg."""
        terms: tuple[TendencyTerm, ...] = ()
        if self._velocity_targets:
            terms += (TendencyTerm(
                name="friction",
                treatment=self._treatment,
                implicit=VerticalDiffusion(
                    axis=self._vertical,
                    fields=self._velocity_targets,
                    kappa=_viscosity,
                    bc=self._velocity_bc),
                advances=self._velocity_targets,
                linear=True),)
        if self._tracer_targets:
            terms += (TendencyTerm(
                name="mixing",
                treatment=self._treatment,
                implicit=VerticalDiffusion(
                    axis=self._vertical,
                    fields=self._tracer_targets,
                    kappa=_diffusivity),
                advances=self._tracer_targets,
                linear=True),)
        return terms
