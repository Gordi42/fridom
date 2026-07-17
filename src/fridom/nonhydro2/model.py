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
from fridom.model.modules.advection import CenteredAdvection
from fridom.model.modules.moving_geometry import MovingGeometry
from fridom.nonhydro2.modules.core import (
    DynamicalCore,
    resolve_model_family,
)
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
    pressure_tolerance: float | None = None,
    pressure_preconditioner: str = "spectral",
    multigrid_levels: int = 5,
    modules_extra: Sequence[fr.model.Module] = (),
    time_stepper: TimeStepper | None = None,
    dt: float = 1.0,
    single_precision_solve: bool = False,
    family: str | None = None,
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
        The fixed PCG iteration budget of the fixed-iteration pressure
        solve, forwarded to the dynamical core; consumed on a
        coordinate-mapped grid *and* on an immersed (cut-cell) grid
        (both run the fixed-iteration PCG). The flat spectral solve is
        exact and iterates nothing (default: 30).
    pressure_tolerance : float | None, optional
        An optional PCG convergence break forwarded to the dynamical
        core (the measure-weighted true relative residual; masked scan,
        exact gradient — see ``DynamicalCore`` and
        :class:`ConjugateGradient`). ``pressure_iterations`` becomes the
        maximum budget; ``None`` runs the fixed count (default: None).
    pressure_preconditioner : str, optional
        The PCG preconditioner of the fixed-iteration pressure solve
        (B4), forwarded to the dynamical core: ``"spectral"`` (the flat
        separable spectral inverse) or ``"multigrid"`` (the
        semicoarsened geometric-multigrid V-cycle). Consumed on a mapped
        or immersed grid; a flat grid uses the exact spectral solve and
        ignores it (default: ``"spectral"``).
    multigrid_levels : int, optional
        The maximum multigrid level count when
        ``pressure_preconditioner="multigrid"`` (floored on small
        grids); ignored otherwise (default: 5).
    modules_extra : Sequence[fr.model.Module], optional
        Additional modules (tracers, closures) (default: ()).
    time_stepper : TimeStepper | None, optional
        Override the default ``AdamBashforth(dt, order=3)``.
    dt : float, optional
        Time step for the default stepper (default: 1.0).
    single_precision_solve : bool, optional
        Run the spectral machinery of the pressure projection in
        single precision (``float32``/``complex64``) while the state
        stays ``float64`` — the whole solve on a flat grid, the PCG
        preconditioner on a mapped one. A performance option
        forwarded to ``DynamicalCore`` (see its
        ``single_precision_solve`` doc). Off by default
        (default: False).
    family : str | None, optional
        The discretization family of the whole model (FV-D3, stage
        F3): ``"fv"`` is the finite-volume C-grid (scalars on
        ``CellAvg``, velocities on the faces — FV-D2 option A),
        ``"nodal"`` the point-value C-grid. ``None`` is the auto
        default: **``"fv"`` on any periodic, walled, static mapped
        (terrain-following) or immersed grid** — so a plain nonhydro
        model is finite-volume by default (owner ruling 2026-07-17:
        FV wherever capable, no surprising family changes by grid
        type). It is at bitwise parity with the nodal model on
        flat/walled grids (scoping study §1; the walled solve is
        eager-bitwise, ≤1.2e-14 jitted, §11), the mapped FV pressure
        operator is likewise bit-identical to nodal (§13), and an
        immersed grid runs the masked FV path (stage I2). The family
        threads to every field (``u, v, w, p`` and the default
        stratification's ``b``) and seeds the FV C-grid ``diff``
        profile — on a walled grid the pressure DCT-II runs on the
        Neumann ``CellAvg`` origin (stage F4), on a mapped grid the
        projection routes to the family-aware ``MappedPressureSolver``
        (stage F5), on an immersed grid to the masked
        ``ImmersedPressureSolver`` (stage I2). One carve-out keeps
        the auto default nodal: a **moving geometry** — a
        ``MovingGeometry`` in ``modules_extra`` that drives the
        mapping in time — because the ALE mesh-velocity correction is
        nodal-only (an explicit ``"fv"`` with a
        ``MeshVelocityCorrection`` is a taught error at bind). An
        immersed grid rejects an explicit ``"nodal"`` (the mask is
        FV-only, IP-D7), and a grid with both a mapped column and an
        immersed domain rejects an explicit ``"fv"`` (the mapped and
        masked PCGs are not yet composed)
        (default: None).
    name : str | None, optional
        Model name (default: None).
    **kwargs : object
        Forwarded to ``fr.model.Model`` (e.g. ``chunk_size``).

    Returns
    -------
    fr.model.Model
        The assembled model.
    """
    # resolve the model family against the grid and adopt it as the
    # grid's default (auto-flip: any periodic, walled, static-mapped
    # or immersed grid promotes None -> "fv"; owner ruling 2026-07-17,
    # FV wherever capable). A MovingGeometry module drives the mapping
    # in time and the ALE mesh-velocity correction is nodal-only, so a
    # dynamically driven mapping keeps the auto default nodal; dynamism
    # is a module property the grid cannot self-report, so the factory
    # supplies it here (an explicit family="fv" with the ALE module is a
    # taught error at MeshVelocityCorrection.bind, not a silent
    # fallback). Every family=None field of the model — u/v/w/p, the
    # default b, and any user tracer — then follows uniformly, so an FV
    # model has no accidental nodal field (only an explicit
    # family="nodal" is the documented mixed corner). Explicit "fv" is
    # served on periodic, walled, mapped (F4, F5) and immersed (I2)
    # grids; only a grid with both a mapped column and an immersed
    # domain is a taught error, and explicit "nodal" on an immersed
    # grid is a taught error too (mask is FV-only, IP-D7).
    dynamic_geometry = any(
        isinstance(module, MovingGeometry) for module in modules_extra)
    resolved = resolve_model_family(
        family, grid, dynamic_geometry=dynamic_geometry)
    grid.set_default_family(resolved)
    if stratification is None:
        stratification = ConstantStratification(n2=1.0)
    if advection is True:
        advection = CenteredAdvection()
    if time_stepper is None:
        time_stepper = fr.model.time_steppers.AdamBashforth(dt, order=3)

    modules: list[fr.model.Module] = [
        DynamicalCore(dsqr=dsqr, rossby_number=rossby_number,
                      single_precision_solve=single_precision_solve,
                      pressure_iterations=pressure_iterations,
                      pressure_tolerance=pressure_tolerance,
                      pressure_preconditioner=pressure_preconditioner,
                      multigrid_levels=multigrid_levels),
    ]
    # rotation is opt-in: coriolis=None installs no module at all
    if coriolis is not None:
        modules.append(coriolis)
    modules.append(stratification)
    if advection is not False:
        modules.append(advection)
    modules.extend(modules_extra)
    # immersed (cut-cell) grid: one shared CONSTRAINT-stage MaskState
    # keeps every prognostic's dry DOFs dead against the modules that
    # do not consult the mask (Coriolis, wave makers, pressure-gradient
    # tendencies) — the masked pressure solve and the fraction-weighted
    # advection handle the wet region themselves (IP-D5). Appended last
    # so its masking runs after the physics stages of the step.
    if getattr(grid, "immersed", None) is not None:
        modules.append(fr.model.modules.MaskState())

    return fr.model.Model(grid=grid, modules=tuple(modules),
                          time_stepper=time_stepper, name=name, **kwargs)
