"""The hydrostatic preset factory.

Description
-----------
``hy.Model(grid=..., free_surface=..., ...)`` is a thin **factory
function** (never a class — D1.3 commitment 2): it builds the module
tuple and delegates to a plain ``fr.model.Model``. Preset assembly and
explicit assembly produce identical carry treedefs (the D4 preset
test). The default stepper is ``AdamBashforth(dt, order=3)``.

**Advection (stage H2b).** The diagnosed vertical velocity ``w`` lives
on the both-boundary vertical face set ``Outer`` (required for the
machine-exact continuity fundamental theorem and the machine-exact
linear energy conservation — see ``hy.modules.HydrostaticCore``). The
shared flux-form advection family (``fr.model.modules.CenteredAdvection``
et al.) transports a cell-centred tracer through the **interior**
vertical faces (``Inner``) and resolves the advecting velocity there
(``w.to(Inner)``). Stage H2b seeds the spatial-layer
``fr.operators.Restriction`` row for that hop: ``Outer ⊃ Inner`` on a
bounded mesh, so ``Outer -> Inner`` is the exact drop of the two
boundary faces — physically the standard fixed-domain closure, **zero
advective flux through the top and bottom boundary faces** under a
linear free surface (the surface volume flux is carried by the ``ps``
equation, not by advection). The surface velocity ``w(0)`` therefore
never enters an advective flux, so a transported tracer's mass is
conserved to roundoff.

The factory default is now ``CenteredAdvection()`` (the common-
denominator scheme); ``advection=False`` recovers the linear model
(what the dispersion, geostrophic-balance and energy-conservation
gates validate), and ``UpwindAdvection`` / ``WENOAdvection`` are
accepted on the uniform unmapped grids the hydrostatic preset targets
(the biased schemes self-reject stretched / mapped geometry at bind).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr
from fridom.hydrostatic.modules.core import HydrostaticCore
from fridom.hydrostatic.modules.free_surface import ExplicitFreeSurface
from fridom.hydrostatic.modules.stratification import (
    ConstantStratification,
)
from fridom.model.modules.advection import CenteredAdvection

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from fridom.model.model import Model as _Model
    from fridom.model.time_steppers.base import TimeStepper
    from fridom.spatial.grid import Grid


def Model(  # noqa: N802 — a factory that mirrors fr.model.Model's surface
    *,
    grid: Grid,
    dt: float = 1.0,
    free_surface: fr.model.Module | None = None,
    csqr: float | fr.model.Ramp = 1.0,
    rossby_number: float | fr.model.Ramp = 1.0,
    coriolis: fr.model.Module | None = None,
    stratification: fr.model.Module | None = None,
    advection: fr.model.Module | bool = True,
    surface_advective_flux: bool = False,
    time_stepper: TimeStepper | None = None,
    modules_extra: Sequence[fr.model.Module] = (),
    name: str | None = None,
    **kwargs: object,
) -> _Model:
    """Assemble a hydrostatic model (preset over ``fr.model.Model``).

    Parameters
    ----------
    grid : Grid
        The grid to assemble on: doubly-periodic horizontal, bounded
        vertical (the ``CumulativeIntegral`` needs a bounded z to seed
        the running integral).
    dt : float, optional
        Time step for the default stepper (default: 1.0).
    free_surface : fr.model.Module | None, optional
        The barotropic (surface-pressure) module
        (default: ``ExplicitFreeSurface()``). Owns the ``ps``
        declaration and its evolution.
    csqr : float | fr.model.Ramp, optional
        The squared barotropic phase speed ``c^2 = g H``, forwarded to
        the core and read by the free surface (default: 1.0).
    rossby_number : float | fr.model.Ramp, optional
        The Rossby number scaling the (separate) advection term
        (default: 1.0).
    coriolis : fr.model.Module | None, optional
        The Coriolis module. ``None`` — the default — means **no
        rotation at all**: no Coriolis module is installed, so the
        model carries no ``f_coriolis`` field and no rotation term.
        Rotation is opt-in: pass ``hy.FPlaneCoriolis(f0=...)`` /
        ``hy.BetaPlaneCoriolis(...)`` (default: None).
    stratification : fr.model.Module | None, optional
        The stratification module
        (default: ``ConstantStratification(n2=1.0)``).
    advection : fr.model.Module | bool, optional
        Nonlinear advection of ``u``/``v``/``b``. ``True`` (the
        default) installs ``CenteredAdvection()``; ``False`` omits
        advection (the linear hydrostatic model); a module instance
        (``UpwindAdvection`` / ``WENOAdvection`` / a configured
        ``CenteredAdvection``) is installed as given. The vertical leg
        consumes the diagnosed ``w`` on the ``Outer`` faces through the
        seeded ``Outer -> Inner`` restriction (module docstring); the
        boundary-face flux is a structural zero (default: True).
    surface_advective_flux : bool, optional
        The constancy-preserving **surface closure** for the default
        advection: advect **through** the top/bottom boundary faces with
        the one-sided (top-cell) face value instead of dropping the
        surface velocity ``w(0)``. It removes the surface-cell constancy
        violation ``A(q=const) ~ q*w(0)/dz`` (the spurious source the
        default fixed-domain closure leaves in the top cell — the
        Oceananigans-equivalent linear-free-surface treatment). Tracer
        content is then exchanged with the moving surface rather than
        conserved to roundoff. Applies **only** when ``advection`` is
        left at its default (``True`` installs ``CenteredAdvection``);
        pass a configured ``CenteredAdvection(surface_flux=True)`` to
        combine it with a non-default scheme (default: False).
    time_stepper : TimeStepper | None, optional
        Override the default ``AdamBashforth(dt, order=3)``.
    modules_extra : Sequence[fr.model.Module], optional
        Additional modules (default: ()).
    name : str | None, optional
        Model name (default: None).
    **kwargs : object
        Forwarded to ``fr.model.Model`` (e.g. ``chunk_size``).

    Returns
    -------
    fr.model.Model
        The assembled model.
    """
    if free_surface is None:
        free_surface = ExplicitFreeSurface()
    if stratification is None:
        stratification = ConstantStratification(n2=1.0)
    if advection is True:
        advection = CenteredAdvection(surface_flux=surface_advective_flux)
    if time_stepper is None:
        time_stepper = fr.model.time_steppers.AdamBashforth(dt, order=3)

    modules: list[fr.model.Module] = [
        HydrostaticCore(csqr=csqr, rossby_number=rossby_number),
    ]
    # rotation is opt-in: coriolis=None installs no module at all
    if coriolis is not None:
        modules.append(coriolis)
    modules.append(stratification)
    modules.append(free_surface)
    if advection is not False:
        modules.append(advection)
    modules.extend(modules_extra)
    # immersed (cut-cell) grid: one shared CONSTRAINT-stage MaskState
    # keeps every 3D prognostic's dry DOFs dead against the modules that
    # do not consult the mask (Coriolis, the p_hyd pressure gradient,
    # the thermal-wind terms) — the masked continuity, the fraction-
    # weighted advection and the masked barotropic solve handle the wet
    # region themselves; the 2D barotropic prognostics (ps, U, V) are
    # masked by the free-surface module, which owns them (IP-D9). The
    # hydrostatic model stays on the grid's default (nodal) family: the
    # mask enters through explicit fraction arithmetic, not a family-
    # dispatched solver, so there is no unmasked "nodal path" to reject.
    # Appended last so its masking runs after the physics stages.
    if getattr(grid, "immersed", None) is not None:
        modules.append(fr.model.modules.MaskState())

    return fr.model.Model(grid=grid, modules=tuple(modules),
                          time_stepper=time_stepper, name=name, **kwargs)
