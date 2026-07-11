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

import fridom.framework2 as fr
from fridom.framework2.modules.coriolis import FPlaneCoriolis
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.model.model import Model as _Model
    from fridom.framework2.model.time_steppers.base import TimeStepper


def Model(  # noqa: N802 — a factory that mirrors fr.Model's surface
    *,
    grid: Grid,
    dsqr: float = 1.0,
    rossby_number: float | fr.Ramp = 1.0,
    coriolis: fr.Module | None = None,
    stratification: fr.Module | None = None,
    advection: fr.Module | bool = True,
    modules_extra: Sequence[fr.Module] = (),
    time_stepper: TimeStepper | None = None,
    dt: float = 1.0,
    name: str | None = None,
) -> _Model:
    """Assemble a nonhydrostatic model (preset over ``fr.Model``).

    Parameters
    ----------
    grid : Grid
        The grid to assemble on.
    dsqr : float, optional
        Squared aspect ratio for the dynamical core (default: 1.0).
    rossby_number : float | fr.Ramp, optional
        Rossby number (default: 1.0).
    coriolis : fr.Module | None, optional
        The Coriolis module (default: ``FPlaneCoriolis(f0=1.0)``).
    stratification : fr.Module | None, optional
        The stratification module
        (default: ``ConstantStratification(n2=1.0)``).
    advection : fr.Module | bool, optional
        The advection module: ``True`` uses the default
        ``CenteredAdvection()``, ``False`` omits advection (a linear
        model), and a module instance is used as given (default: True).
    modules_extra : Sequence[fr.Module], optional
        Additional modules (tracers, closures) (default: ()).
    time_stepper : TimeStepper | None, optional
        Override the default ``AdamBashforth(dt, order=3)``.
    dt : float, optional
        Time step for the default stepper (default: 1.0).
    name : str | None, optional
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
    if advection is True:
        advection = CenteredAdvection()
    if time_stepper is None:
        time_stepper = fr.time_steppers.AdamBashforth(dt, order=3)

    modules: list[fr.Module] = [
        DynamicalCore(dsqr=dsqr, rossby_number=rossby_number),
        coriolis,
        stratification,
    ]
    if advection is not False:
        modules.append(advection)
    modules.extend(modules_extra)

    return fr.Model(grid=grid, modules=tuple(modules),
                    time_stepper=time_stepper, name=name)
