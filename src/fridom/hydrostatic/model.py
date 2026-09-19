"""The hydrostatic preset factory.

Description
-----------
``hy.Model(grid=..., core=..., time_stepper=..., ...)`` is a thin
**factory function** (never a class — D1.3 commitment 2): it builds
the module tuple and delegates to a plain ``fr.model.Model``. Preset
assembly and explicit assembly produce identical carry treedefs (the
D4 preset test).

The physics lives on the **core** (``hy.Core``, gravity-first: the
physical constant centralizes there), the physics modules (``core=``
and ``free_surface=`` are required, ``buoyancy=`` optional — no
surprising default physics), and the **scaling**
policy (``fr.scaling``): ``scaling=`` names the reference time frame
(default: ``fr.scaling.Dimensional()`` — a dimensional assembly needs
no scaling argument at all). The retired preset kwargs (``csqr=``,
``rossby_number=``, ``dt=``, the renamed ``stratification=``) raise
taught TypeErrors naming the new spelling.

**Advection (stage H2b).** The diagnosed vertical velocity ``w`` lives
on the both-boundary vertical face set ``Outer`` (required for the
machine-exact continuity fundamental theorem and the machine-exact
linear energy conservation — see ``hy.Core``). The shared flux-form
advection family (``fr.model.modules.CenteredAdvection`` et al.)
transports a cell-centred tracer through the **interior** vertical
faces (``Inner``) and resolves the advecting velocity there
(``w.to(Inner)``). Stage H2b seeds the spatial-layer
``fr.operators.Restriction`` row for that hop: ``Outer ⊃ Inner`` on a
bounded mesh, so ``Outer -> Inner`` is the exact drop of the two
boundary faces — physically the standard fixed-domain closure, **zero
advective flux through the top and bottom boundary faces** under a
linear free surface (the surface volume flux is carried by the ``ps``
equation, not by advection). The surface velocity ``w(0)`` therefore
never enters an advective flux, so a transported tracer's mass is
conserved to roundoff.

The factory installs **no advection unless asked** (owner rulings
2026-08-22, ``design/decisions/no_default_advection.md`` and
``object_or_none_keywords.md``): the default ``advection=None`` is the
linear model (what the dispersion, geostrophic-balance and
energy-conservation gates validate), a boolean is refused, and the
scheme is named as a module — ``CenteredAdvection()`` (the
common-denominator scheme) or ``UpwindAdvection`` / ``WENOAdvection``,
accepted on the flat grids the hydrostatic preset targets, a
**stretched** vertical column included — their reconstruction rows are
built from the factor's own cell widths there (route (ii); an
average-family tracer then keeps the design order, a nodal one drops to
2nd, see ``fr.model.modules.advection``). Only a terrain-following
**mapped column** (a ``CoordinateMapping``) is still self-rejected by
the biased schemes at bind.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr
from fridom._sequences import as_tuple
from fridom.hydrostatic.modules.core import resolve_model_family

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from fridom.model.model import Model as _Model
    from fridom.model.time_steppers.base import TimeStepper
    from fridom.spatial.grid import Grid

#: retired preset kwargs -> the taught replacement spelling
_RETIRED_KWARGS = {
    "surface_advective_flux": (
        "surface_advective_flux= is retired: it configured the "
        "advection module the preset used to install unasked. Name "
        "the scheme and its closure yourself — "
        "advection=fr.model.modules.CenteredAdvection("
        "surface_flux=...)"),
    "csqr": (
        "csqr= is retired: gravity is the physical constant and it "
        "centralizes on the core — pass core=hy.Core(gravity=...) "
        "(dimensional; the barotropic update is -g T*, every column "
        "depth is genuine geometry) or a nondimensional free surface "
        "(free_surface=hy.ExplicitFreeSurface(froude_number=...)) "
        "with scaling=fr.scaling.ExternalWave()"),
    "rossby_number": (
        "rossby_number= is retired: the nonlinearity number is the "
        "scaling mechanism's own regime number — pass a "
        "nondimensional Coriolis module "
        "(FPlaneCoriolis(rossby_number=...)) with "
        "scaling=fr.scaling.Rotational() (epsilon = Ro), or a "
        "nondimensional free surface "
        "(hy.ExplicitFreeSurface(froude_number=...)) with "
        "scaling=fr.scaling.ExternalWave() (epsilon = Fr_ext)"),
    "dt": (
        "dt= is retired on the preset: pass the stepper explicitly, "
        "time_stepper=fr.model.time_steppers.AdamBashforth(dt, "
        "order=3)"),
    "stratification": (
        "stratification= was renamed to buoyancy= — the slot holds "
        "the buoyancy formulation: pass "
        "buoyancy=hy.ConstantStratification(n2=...) (dimensional) / "
        "ConstantStratification(froude_number=...) (nondimensional)"),
}


def _refuse_boolean_advection(advection: object, preset: str) -> None:
    """Raise the taught TypeError on ``advection=True`` / ``False``."""
    if isinstance(advection, bool):
        spelled = ("advection=None (no advection, the default)"
                   if not advection else
                   "the module that names the scheme, e.g. "
                   "advection=fr.model.modules.CenteredAdvection()")
        raise TypeError(
            f"{preset}.Model advection= takes a module or None, not "
            f"{advection!r}: the preset installs no scheme unasked and "
            f"a boolean cannot say which one is meant — pass {spelled}")


def Model(  # noqa: N802 — a factory that mirrors fr.model.Model's surface
    *,
    grid: Grid,
    core: fr.model.Module,
    free_surface: fr.model.Module,
    time_stepper: TimeStepper,
    buoyancy: fr.model.Module | None = None,
    scaling: object | None = None,
    coriolis: fr.model.Module | None = None,
    advection: fr.model.Module | None = None,
    modules_extra: fr.model.Module | Sequence[fr.model.Module] = (),
    name: str | None = None,
    **kwargs: object,
) -> _Model:
    """Assemble a hydrostatic model (preset over ``fr.model.Model``).

    Parameters
    ----------
    grid : Grid
        The grid to assemble on: periodic or walled horizontal, bounded
        vertical (the ``CumulativeIntegral`` needs a bounded z to seed
        the running integral). A thin-shell **sphere**
        (``fr.spatial.spherical.Grid(..., vertical=<z mesh>)``) — or
        any orthogonal two-coordinate chart extruded along a flat
        vertical — is supported with ``hy.Core(horizontal=("lon",
        "lat"))``, ``hy.ExplicitFreeSurface(horizontal=("lon",
        "lat"))``, ``fr.model.modules.RotationCoriolis`` and
        ``CenteredAdvection`` (spherical-models plan S0-S2); land and
        bathymetry ride an ``ImmersedDomain`` staircase as on a flat
        grid. The implicit / split-explicit free surfaces and the
        biased advection schemes are taught refusals on a chart.
    core : fr.model.Module
        The dynamical core, ``hy.Core``: the gravity-first physical
        constant lives here (``gravity=``, dimensional) — the
        free-surface family references it; the nondimensional core
        takes no kwarg at all. It also carries the discretization
        family (``hy.Core(family="fv")``), which the factory resolves
        against the grid and adopts as the grid-level default so every
        other module's ``family=None`` declaration follows.
    free_surface : fr.model.Module
        The barotropic (surface-pressure) module — REQUIRED, no
        default: ``hy.ExplicitFreeSurface(...)`` /
        ``hy.ImplicitFreeSurface(...)`` /
        ``hy.SplitExplicitFreeSurface(...)``. Owns the ``ps``
        declaration and its evolution.
    time_stepper : TimeStepper
        The time stepper (e.g.
        ``fr.model.time_steppers.AdamBashforth(dt, order=3)``).
    buoyancy : fr.model.Module | None, optional
        The buoyancy formulation. ``None`` — the default — installs
        **no buoyancy module**: the model carries no ``b`` field and
        the core diagnoses ``p_hyd = 0``, a **constant-density**
        (barotropic) flow driven by the surface pressure alone. Pass
        ``hy.ConstantStratification(n2=...)`` (dimensional) /
        ``ConstantStratification(froude_number=...)`` (nondimensional)
        for a stratified run, or ``hy.BuoyancyTracer()`` for a
        buoyancy tracer with no background stratification. The renamed
        ``stratification=`` spelling raises the taught TypeError
        (default: None).
    scaling : object | None, optional
        The ``fr.scaling`` policy naming the reference time frame;
        ``None`` defaults to ``fr.scaling.Dimensional()`` — a
        dimensional assembly needs no scaling argument. A
        nondimensional module kwarg set needs the matching
        nondimensional policy (``fr.scaling.Rotational()`` /
        ``InternalWave()`` / ``ExternalWave()`` / ``Advective()``);
        the assembly refuses a mismatch with a taught error
        (default: None).
    coriolis : fr.model.Module | None, optional
        The Coriolis module. ``None`` — the default — means **no
        rotation at all**: no Coriolis module is installed, so the
        model carries no ``f_coriolis`` field and no rotation term.
        Rotation is opt-in: pass ``hy.FPlaneCoriolis(f0=...)``
        (dimensional) / ``FPlaneCoriolis(rossby_number=...)``
        (nondimensional) / ``hy.BetaPlaneCoriolis(...)``
        (default: None).
    advection : fr.model.Module | None, optional
        Nonlinear advection of ``u``/``v``/``b``. ``None`` (the
        default) omits advection, the linear hydrostatic model: a
        scheme is never installed unasked, and a boolean is refused
        with a taught error. Pass the module that names the scheme,
        ``fr.model.modules.CenteredAdvection()`` (``surface_flux=``
        configures its constancy-preserving surface closure),
        ``UpwindAdvection(order=...)`` or ``WENOAdvection(...)``. The
        module is scaling-neutral and adopts the assembly's variant
        at bind. The vertical leg consumes the diagnosed ``w`` on the
        ``Outer`` faces through the seeded ``Outer -> Inner``
        restriction (module docstring); the boundary-face flux is a
        structural zero (default: None).
    modules_extra : fr.model.Module | Sequence[fr.model.Module], optional
        Additional modules. A list or a tuple is the module
        collection, anything else a single module, so one extra
        module is ``modules_extra=tracer`` (default: ()).
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
        On a missing ``core=`` / ``free_surface=`` (the two required
        modules), the retired kwargs ``csqr=`` / ``rossby_number=`` /
        ``dt=``, and the renamed ``stratification=`` (now
        ``buoyancy=``) (taught messages naming the new spelling).
    """
    for retired, message in _RETIRED_KWARGS.items():
        if retired in kwargs:
            raise TypeError(f"hy.Model {message}")
    if core is None or free_surface is None:
        raise TypeError(
            "hy.Model has no default physics (owner-ratified): "
            "core= and free_surface= are REQUIRED modules — pass "
            "core=hy.Core(gravity=...) and "
            "free_surface=hy.ExplicitFreeSurface() (or the implicit "
            "/ split-explicit variants); None is not a module")
    if scaling is None:
        scaling = fr.scaling.Dimensional()
    # resolve the discretization family against the grid and adopt it
    # as the grid's default (stage F3). There is no auto flip: None
    # follows the grid, so a plain grid stays on the nodal point-value
    # C-grid and every pre-existing assembly is bitwise unchanged. The
    # requested family lives on the core (core.family); adopting it as
    # the grid default is what makes every family=None field of the
    # model -- b, ps, the split-explicit U/V, a z* eta -- follow
    # uniformly, so an FV model has no accidental nodal field.
    grid.set_default_family(
        resolve_model_family(getattr(core, "family", None), grid))
    _refuse_boolean_advection(advection, "hy")

    modules: list[fr.model.Module] = [core]
    # rotation is opt-in: coriolis=None installs no module at all
    if coriolis is not None:
        modules.append(coriolis)
    # buoyancy is opt-in too: buoyancy=None installs no module, so the
    # model carries no b field and the core diagnoses p_hyd = 0 (a
    # constant-density, barotropic flow)
    if buoyancy is not None:
        modules.append(buoyancy)
    modules.append(free_surface)
    if advection is not None:
        modules.append(advection)
    modules.extend(as_tuple(modules_extra))
    # immersed (cut-cell) grid: one shared CONSTRAINT-stage MaskState
    # keeps every 3D prognostic's dry DOFs dead against the modules that
    # do not consult the mask (Coriolis, the p_hyd pressure gradient,
    # the thermal-wind terms) — the masked continuity, the fraction-
    # weighted advection and the masked barotropic solve handle the wet
    # region themselves; the 2D barotropic prognostics (ps, U, V) are
    # masked by the free-surface module, which owns them (IP-D9). An
    # immersed grid stays NODAL: the mask enters through explicit
    # fraction arithmetic on point-valued C-grid faces, not through a
    # family-dispatched solver, so family='fv' on a cut-cell grid is a
    # taught refusal (resolve_model_family) rather than a silent
    # mis-weighting. Appended last so its masking runs after the
    # physics stages.
    if getattr(grid, "immersed", None) is not None:
        modules.append(fr.model.modules.MaskState())

    return fr.model.Model(grid=grid, modules=tuple(modules),
                          time_stepper=time_stepper, name=name,
                          scaling=scaling, **kwargs)
