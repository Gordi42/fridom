"""The hydrostatic preset factory.

Description
-----------
``hy.Model(grid=..., free_surface=..., ...)`` is a thin **factory
function** (never a class — D1.3 commitment 2): it builds the module
tuple and delegates to a plain ``fr.model.Model``. Preset assembly and
explicit assembly produce identical carry treedefs (the D4 preset
test). The default stepper is ``AdamBashforth(dt, order=3)``.

**Advection (stage-H2 limitation).** The diagnosed vertical velocity
``w`` lives on the both-boundary vertical face set ``Outer`` (required
for the machine-exact continuity fundamental theorem and the
machine-exact linear energy conservation — see
``hy.modules.HydrostaticCore``). The shared flux-form advection
family (``fr.model.modules.CenteredAdvection`` et al.) transports a
cell-centred tracer through the **interior** vertical faces
(``Inner``) and interpolates the advecting velocity there
(``w.to(Inner)``); there is no registered interpolation between the
``Outer`` and ``Inner`` vertical face sets, so the shared advection's
vertical leg cannot consume a ``w``-on-``Outer`` velocity. This is a
spatial-layer gap (a vertical ``Center``<->``Outer`` /
``Outer``->``Inner`` interpolation, or a shared-advection enhancement
restricting an ``Outer`` velocity to the interior flux faces) tracked
for a follow-up. Until then the factory default is a **linear**
hydrostatic model (``advection=False``); requesting advection raises a
taught error rather than the cryptic dispatch failure it would hit
deep in the flux chain. The linear model is exactly what the
dispersion, geostrophic-balance and energy-conservation gates
validate.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr
from fridom.hydrostatic.modules.core import HydrostaticCore
from fridom.hydrostatic.modules.free_surface import ExplicitFreeSurface
from fridom.hydrostatic.modules.stratification import (
    ConstantStratification,
)

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
    advection: fr.model.Module | bool = False,
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
        Nonlinear advection. **Currently only ``False`` (a linear
        model) is supported** — the shared flux-form advection cannot
        consume the diagnosed ``w`` on the ``Outer`` vertical faces
        (see the module docstring). A truthy value raises
        ``NotImplementedError`` (default: False).
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

    Raises
    ------
    NotImplementedError
        If ``advection`` is truthy (the stage-H2 vertical-advection
        limitation).
    """
    if advection is not False and advection is not None:
        raise NotImplementedError(
            "hy.Model does not yet support nonlinear advection: the "
            "diagnosed vertical velocity w lives on the both-boundary "
            "vertical face set (Outer), and the shared flux-form "
            "advection cannot interpolate it onto the interior flux "
            "faces (no Outer<->Inner vertical interpolation is "
            "registered). Use advection=False (the linear hydrostatic "
            "model); the advective path awaits a spatial-layer "
            "vertical-interpolation follow-up (stage H2 record).")
    if free_surface is None:
        free_surface = ExplicitFreeSurface()
    if stratification is None:
        stratification = ConstantStratification(n2=1.0)
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
    modules.extend(modules_extra)

    return fr.model.Model(grid=grid, modules=tuple(modules),
                          time_stepper=time_stepper, name=name, **kwargs)
