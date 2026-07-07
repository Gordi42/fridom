"""
The shallow-water preset factory (framework2 port).

Description
-----------
``sw.Model(grid=..., coriolis=..., ...)`` is a **thin factory
function** (never a subclass, D1.3 commitment 2 / model.md section 6):
it builds the shallow-water module tuple and delegates to a plain
``fr.Model``. Preset and explicit assembly produce **identical carry
treedefs** — the D4 preset test. The factory may only build the
module tuple, forward kwargs, and pick default modules/stepper; it
holds no parameters or fields and never mutates the model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom.framework2 as fr
from fridom.shallowwater2.modules.core import ShallowWaterCore
from fridom.shallowwater2.modules.coriolis import FPlaneCoriolis
from fridom.shallowwater2.modules.sadourny import SadournyAdvection

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.model.model import Model as _Model
    from fridom.framework2.model.time_steppers.base import TimeStepper


def Model(  # noqa: N802 — constructor-like factory (D1.3)
    *,
    grid: Grid,
    csqr: float = 1.0,
    rossby_number: float = 1.0,
    coriolis: fr.Module | None = None,
    advection: bool = True,
    time_stepper: TimeStepper | None = None,
    modules_extra: Sequence[fr.Module] = (),
    name: str | None = None,
    **kwargs: object,
) -> _Model:
    """
    Assemble a shallow-water model (thin preset over ``fr.Model``).

    Parameters
    ----------
    grid : Grid
        The (periodic, 2-D) grid.
    csqr : float, optional
        Squared gravity-wave phase speed :math:`c^2` (default: 1.0).
    rossby_number : float, optional
        Rossby number scaling the advection (default: 1.0).
    coriolis : fr.Module | None, optional
        The Coriolis field provider; default
        ``FPlaneCoriolis(f0=1.0)``.
    advection : bool, optional
        Include the Sadourny nonlinear advection (default: True).
    time_stepper : TimeStepper | None, optional
        The stepper; default ``AdamBashforth(dt=1.0, order=3)`` (the
        cutover package default — pass an explicit one for a real
        run).
    modules_extra : Sequence[fr.Module], optional
        Additional modules (tracers, closures) appended after the
        core physics (default: ()).
    name : str | None, optional
        Model name (default: None).
    **kwargs : object
        Forwarded to ``fr.Model`` (e.g. ``chunk_size``).

    Returns
    -------
    fr.Model
        The assembled model.
    """
    core = ShallowWaterCore(csqr=csqr, rossby_number=rossby_number)
    cor = FPlaneCoriolis(f0=1.0) if coriolis is None else coriolis
    modules: tuple[fr.Module, ...] = (core, cor)
    if advection:
        modules += (SadournyAdvection(),)
    modules += tuple(modules_extra)
    if time_stepper is None:
        time_stepper = fr.time_steppers.AdamBashforth(dt=1.0, order=3)
    return fr.Model(grid=grid, modules=modules,
                    time_stepper=time_stepper, name=name, **kwargs)
