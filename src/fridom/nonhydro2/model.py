"""The nonhydrostatic preset factory.

Description
-----------
``nh.Model(grid=..., core=..., time_stepper=..., ...)`` is a thin
**factory function** (never a class — D1.3 commitment 2): it builds
the module tuple and delegates to a plain ``fr.model.Model``. Preset
assembly and explicit assembly produce identical carry treedefs (the
D4 preset test).

The physics lives on the **core** (``nh.Core``, which also carries
the solver/family knobs; ``core=None`` — the default — is a plain
``nh.Core()``), the physics modules (Coriolis / buoyancy, whose kwarg
sets fix the scaling variant) and the **scaling** policy
(``fr.scaling``): ``scaling=`` names the reference time frame
(default: ``fr.scaling.Dimensional()`` — a dimensional assembly needs
no scaling argument at all). The retired preset kwargs (``dsqr=``,
``rossby_number=``, the solver knobs, ``dt=``, the renamed
``stratification=``) raise taught TypeErrors naming the new spelling.
Buoyancy is **opt-in**: ``buoyancy=None`` (the default) installs no
buoyancy module at all.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr
from fridom._sequences import as_tuple
from fridom.nonhydro2.modules.core import Core, resolve_model_family

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from fridom.model.model import Model as _Model
    from fridom.model.time_steppers.base import TimeStepper
    from fridom.spatial.grid import Grid

#: retired preset kwargs -> the taught replacement spelling
_RETIRED_KWARGS = {
    "dsqr": (
        "dsqr= is retired: the core carries the aspect ratio — pass "
        "core=nh.Core(aspect_ratio=...) (squared at the use sites, "
        "so aspect_ratio=sqrt(dsqr))"),
    "rossby_number": (
        "rossby_number= is retired: the nonlinearity number is the "
        "scaling mechanism's own regime number — pass a "
        "nondimensional Coriolis module "
        "(FPlaneCoriolis(rossby_number=...)) with "
        "scaling=fr.scaling.Rotational() (epsilon = Ro), or a "
        "nondimensional buoyancy module "
        "(buoyancy=ConstantStratification(froude_number=...)) with "
        "scaling=fr.scaling.InternalWave() (epsilon = Fr)"),
    "dt": (
        "dt= is retired on the preset: pass the stepper explicitly, "
        "time_stepper=fr.model.time_steppers.AdamBashforth(dt, "
        "order=3)"),
    "stratification": (
        "stratification= was renamed to buoyancy= — the slot holds "
        "the buoyancy formulation, which need not carry a background "
        "stratification: pass buoyancy=nh.ConstantStratification("
        "n2=...) / nh.MeridionalStratification(...) / "
        "nh.BuoyancyTracer() (a bare buoyancy tracer, no N^2)"),
}

#: solver/family kwargs that moved onto the core (one taught message)
_CORE_KWARGS = (
    "family", "single_precision_solve", "pressure_iterations",
    "pressure_tolerance", "pressure_preconditioner",
    "multigrid_levels", "multigrid_tridiagonal_method",
    "multigrid_coarsen_vertical", "multigrid_agglomerate", "vertical",
    "coords",
)


def _refuse_boolean_advection(advection: object) -> None:
    """Raise the taught TypeError on ``advection=True`` / ``False``."""
    if isinstance(advection, bool):
        spelled = ("advection=None (no advection, the default)"
                   if not advection else
                   "the module that names the scheme, e.g. "
                   "advection=nh.CenteredAdvection()")
        raise TypeError(
            "nh.Model advection= takes a module or None, not "
            f"{advection!r}: the preset installs no scheme unasked and "
            f"a boolean cannot say which one is meant — pass {spelled}")


def _refuse_retired_kwargs(kwargs: dict) -> None:
    """Raise the taught TypeError on a retired/renamed preset kwarg."""
    for retired, message in _RETIRED_KWARGS.items():
        if retired in kwargs:
            raise TypeError(f"nh.Model {message}")
    moved = [key for key in _CORE_KWARGS if key in kwargs]
    if moved:
        raise TypeError(
            f"nh.Model kwargs {moved} moved onto the core: pass "
            "them to nh.Core(...) — e.g. core=nh.Core("
            "aspect_ratio=..., pressure_iterations=..., "
            "family=...)")


def Model(  # noqa: N802 — a factory that mirrors fr.model.Model's surface
    *,
    grid: Grid,
    time_stepper: TimeStepper,
    core: fr.model.Module | None = None,
    scaling: object | None = None,
    coriolis: fr.model.Module | None = None,
    buoyancy: fr.model.Module | None = None,
    advection: fr.model.Module | None = None,
    modules_extra: fr.model.Module | Sequence[fr.model.Module] = (),
    name: str | None = None,
    **kwargs: object,
) -> _Model:
    """Assemble a nonhydrostatic model (preset over ``fr.model.Model``).

    Parameters
    ----------
    grid : Grid
        The grid to assemble on.
    time_stepper : TimeStepper
        The time stepper (e.g.
        ``fr.model.time_steppers.AdamBashforth(dt, order=3)``).
    core : fr.model.Module | None, optional
        The dynamical core, ``nh.Core``: carries the aspect ratio
        (``aspect_ratio=``, squared at the use sites) and the
        pressure-solver / discretization-family knobs. The preset
        resolves the family auto-flip against the grid through
        ``core.family``. ``None`` — the default — is a plain
        ``nh.Core()`` (aspect ratio 1, default solver knobs)
        (default: None).
    scaling : object | None, optional
        The ``fr.scaling`` policy naming the reference time frame;
        ``None`` defaults to ``fr.scaling.Dimensional()`` — a
        dimensional assembly needs no scaling argument. A
        nondimensional Coriolis/buoyancy kwarg set needs the
        matching nondimensional policy (``fr.scaling.Rotational()`` /
        ``InternalWave()`` / ``Advective()``); the assembly refuses a
        mismatch with a taught error (default: None).
    coriolis : fr.model.Module | None, optional
        The Coriolis module. ``None`` — the argument omitted, the
        default — means **no rotation at all**: no Coriolis module
        is installed, so the model carries no ``f_coriolis`` field,
        no rotation term and no rotation provide. Rotation is
        opt-in: pass ``nh.FPlaneCoriolis(f0=...)`` (dimensional) /
        ``FPlaneCoriolis(rossby_number=...)`` (nondimensional) /
        ``nh.BetaPlaneCoriolis(...)``. Those are the whole menu here:
        the nonhydrostatic model is **Cartesian-only** — its pressure
        projection and flux-form advection are metric-blind, so
        ``nh.Core`` refuses a chart-coupled grid at bind — which
        leaves the chart rotation
        ``fr.model.modules.RotationCoriolis`` with no grid to act on
        in this package. Use ``fridom.shallowwater2`` for a model on
        a chart (default: None).

        Note that a non-rotating **linear** nonhydrostatic model
        (``advection=None``) with no buoyancy module leaves
        ``u``/``v`` advanced by no term at all and is rejected by
        the D1.4 coverage lint.
    buoyancy : fr.model.Module | None, optional
        The buoyancy formulation: the module that declares the
        buoyancy variable ``b`` and couples it into the momentum
        equation. ``None`` — the default — installs **no buoyancy at
        all** (explicit opt-in; there is no surprising default
        physics): pass ``nh.ConstantStratification(n2=...)``
        (dimensional) / ``ConstantStratification(froude_number=...)``
        (nondimensional) / ``nh.MeridionalStratification(...)`` for a
        background stratification, or ``nh.BuoyancyTracer()`` for a
        bare buoyancy tracer without one (default: None).
    advection : fr.model.Module | None, optional
        The advection module. ``None`` (the default) omits
        advection, a linear model: a scheme is never installed
        unasked, and a boolean is refused with a taught error. Pass
        the module that names the scheme, ``nh.CenteredAdvection()``
        / ``nh.UpwindAdvection(order=...)`` /
        ``nh.WENOAdvection(...)``. The module is scaling-neutral and
        adopts the assembly's variant at bind (default: None).
    modules_extra : fr.model.Module | Sequence[fr.model.Module], optional
        Additional modules (tracers, closures). A list or a tuple is
        the module collection, anything else a single module, so one
        extra module is ``modules_extra=wave_maker`` (default: ()).
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
    TypeError
        On the retired kwargs ``dsqr=`` / ``rossby_number=`` /
        ``dt=``, the renamed ``stratification=`` (now ``buoyancy=``)
        and the solver/family kwargs that moved onto the core
        (taught messages naming the new spelling).
    """
    _refuse_retired_kwargs(kwargs)
    if core is None:
        core = Core()
    if scaling is None:
        scaling = fr.scaling.Dimensional()
    # resolve the model family against the grid and adopt it as the
    # grid's default (auto-flip: every grid promotes None -> "fv";
    # owner ruling 2026-07-17, FV wherever capable). The requested
    # family lives on the core (core.family); every family=None field
    # of the model then follows the grid default uniformly.
    resolved = resolve_model_family(
        getattr(core, "family", None), grid)
    grid.set_default_family(resolved)
    _refuse_boolean_advection(advection)

    modules: list[fr.model.Module] = [core]
    # rotation is opt-in: coriolis=None installs no module at all
    if coriolis is not None:
        modules.append(coriolis)
    # buoyancy is opt-in too (no surprising default physics)
    if buoyancy is not None:
        modules.append(buoyancy)
    if advection is not None:
        modules.append(advection)
    modules.extend(as_tuple(modules_extra))
    # immersed (cut-cell) grid: one shared CONSTRAINT-stage MaskState
    # keeps every prognostic's dry DOFs dead against the modules that
    # do not consult the mask (Coriolis, wave makers, pressure-gradient
    # tendencies) — the masked pressure solve and the fraction-weighted
    # advection handle the wet region themselves (IP-D5). Appended last
    # so its masking runs after the physics stages of the step.
    if getattr(grid, "immersed", None) is not None:
        modules.append(fr.model.modules.MaskState())

    return fr.model.Model(grid=grid, modules=tuple(modules),
                          time_stepper=time_stepper, name=name,
                          scaling=scaling, **kwargs)
