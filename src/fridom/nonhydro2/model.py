"""The nonhydrostatic preset factory.

Description
-----------
``nh.Model(grid=..., coriolis=..., stratification=..., ...)`` is a thin
**factory function** (never a class — D1.3 commitment 2): it builds the
module tuple and delegates to a plain ``fr.model.Model``. Preset assembly and
explicit assembly produce identical carry treedefs (the D4 preset
test). The default stepper is ``AdamBashforth(order=3)`` (the nh
cutover default, V-N3).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
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
    dsqr: float = 1.0,
    rossby_number: float | fr.model.Ramp = 1.0,
    coriolis: fr.model.Module | None = None,
    stratification: fr.model.Module | None = None,
    advection: fr.model.Module | bool = True,
    pressure_iterations: int = 30,
    modules_extra: Sequence[fr.model.Module] = (),
    time_stepper: TimeStepper | None = None,
    dt: float = 1.0,
    single_precision_solve: bool = False,
    name: str | None = None,
    **kwargs: object,
) -> _Model:
    """Assemble a nonhydrostatic model (preset over ``fr.model.Model``).

    Parameters
    ----------
    grid : Grid
        The grid to assemble on.
    dsqr : float, optional
        Squared aspect ratio for the dynamical core (default: 1.0).
    rossby_number : float | fr.model.Ramp, optional
        Rossby number (default: 1.0).
    coriolis : fr.model.Module | None, optional
        The Coriolis module. ``None`` — the argument omitted, the
        default — means **no rotation at all**: no Coriolis module
        is installed, so the model carries no ``f_coriolis`` field,
        no rotation term and no ``coriolis.f0`` provide. Rotation is
        opt-in: pass ``nh.FPlaneCoriolis(f0=...)`` /
        ``nh.BetaPlaneCoriolis(...)`` on a flat grid, or
        ``fr.modules.RotationCoriolis(omega=(0.0, 0.0, Omega),
        coords=...)`` on a chart-coupled grid (default: None).

        Note that a non-rotating **linear** nonhydrostatic model
        (``advection=False``) leaves ``u``/``v`` advanced by no term
        at all and is rejected by the D1.4 coverage lint — a linear
        run needs a Coriolis module.
    stratification : fr.model.Module | None, optional
        The stratification module
        (default: ``ConstantStratification(n2=1.0)``).
    advection : fr.model.Module | bool, optional
        The advection module: ``True`` uses the default
        ``CenteredAdvection()``, ``False`` omits advection (a linear
        model), and a module instance is used as given (default: True).
    pressure_iterations : int, optional
        The fixed PCG iteration budget of the mapped pressure solve,
        forwarded to the dynamical core; consumed only on a
        coordinate-mapped grid (the flat spectral solve is exact and
        iterates nothing) (default: 30).
    modules_extra : Sequence[fr.model.Module], optional
        Additional modules (tracers, closures) (default: ()).
    time_stepper : TimeStepper | None, optional
        Override the default ``AdamBashforth(dt, order=3)``.
    dt : float, optional
        Time step for the default stepper (default: 1.0).
    single_precision_solve : bool, optional
        Run the spectral pressure projection in single precision
        (``float32``/``complex64``) while the state stays
        ``float64`` — a performance option forwarded to
        ``DynamicalCore`` (see its ``single_precision_solve`` doc).
        Off by default (default: False).
    name : str | None, optional
        Model name (default: None).
    **kwargs : object
        Forwarded to ``fr.model.Model`` (e.g. ``chunk_size``).

    Returns
    -------
    fr.model.Model
        The assembled model.
    """
    if stratification is None:
        stratification = ConstantStratification(n2=1.0)
    if advection is True:
        advection = CenteredAdvection()
    if time_stepper is None:
        time_stepper = fr.model.time_steppers.AdamBashforth(dt, order=3)

    modules: list[fr.model.Module] = [
        DynamicalCore(dsqr=dsqr, rossby_number=rossby_number,
                      single_precision_solve=single_precision_solve,
                      pressure_iterations=pressure_iterations),
    ]
    # rotation is opt-in: coriolis=None installs no module at all
    if coriolis is not None:
        modules.append(coriolis)
    modules.append(stratification)
    if advection is not False:
        modules.append(advection)
    modules.extend(modules_extra)

    return fr.model.Model(grid=grid, modules=tuple(modules),
                          time_stepper=time_stepper, name=name, **kwargs)
