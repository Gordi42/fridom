"""The nonhydrostatic preset factory.

Description
-----------
``nh.Model(grid=..., coriolis=..., stratification=..., ...)`` is a thin
**factory function** (never a class — D1.3 commitment 2): it builds the
module tuple and delegates to a plain ``fr.Model``. Preset assembly and
explicit assembly produce identical carry treedefs (the D4 preset
test). The default stepper is ``AdamBashforth(order=3)`` (the nh
cutover default, V-N3).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.model.model import Model as _Model
from fridom.framework2.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.framework2.modules.coriolis import FPlaneCoriolis
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.grid import Grid


def Model(  # noqa: N802 — a factory that mirrors fr.Model's surface
    *,
    grid: Grid,
    dsqr: float = 1.0,
    rossby_number: object = 1.0,
    coriolis: object | None = None,
    stratification: object | None = None,
    advection: object | None = None,
    modules_extra: tuple[object, ...] = (),
    time_stepper: object | None = None,
    dt: float = 1.0,
    name: str | None = None,
) -> _Model:
    """Assemble a nonhydrostatic model (preset over ``fr.Model``).

    Parameters
    ----------
    grid : fr.grid.Grid
        The grid to assemble on.
    dsqr : float, optional
        Squared aspect ratio for the dynamical core (default: 1.0).
    rossby_number : float | fr.Ramp, optional
        Rossby number (default: 1.0).
    coriolis : fr.Module, optional
        The Coriolis module (default: ``FPlaneCoriolis(f0=1.0)``).
    stratification : fr.Module, optional
        The stratification module
        (default: ``ConstantStratification(n2=1.0)``).
    advection : fr.Module | None, optional
        The advection module (default: ``CenteredAdvection()``; pass
        ``False`` — any falsy non-None — to omit advection for a
        linear model).
    modules_extra : tuple[fr.Module, ...], optional
        Additional modules (tracers, closures) (default: ()).
    time_stepper : fr.time_steppers.TimeStepper, optional
        Override the default ``AdamBashforth(dt, order=3)``.
    dt : float, optional
        Time step for the default stepper (default: 1.0).
    name : str, optional
        Model name (default: None).

    Returns
    -------
    fr.Model
        The assembled model.
    """
    if coriolis is None:
        coriolis = FPlaneCoriolis(f0=1.0)
    if stratification is None:
        stratification = ConstantStratification(n2=1.0)
    if advection is None:
        advection = CenteredAdvection()
    if time_stepper is None:
        time_stepper = AdamBashforth(dt, order=3)

    modules: list[object] = [
        DynamicalCore(dsqr=dsqr, rossby_number=rossby_number),
        coriolis,
        stratification,
    ]
    if advection:
        modules.append(advection)
    modules.extend(modules_extra)

    return _Model(grid=grid, modules=tuple(modules),
                  time_stepper=time_stepper, name=name)
