"""The hydrostatic preset factory.

Description
-----------
``hy.Model(grid=..., core=..., time_stepper=..., ...)`` is a thin
**factory function** (never a class — D1.3 commitment 2): it builds
the module tuple and delegates to a plain ``fr.model.Model``. Preset
assembly and explicit assembly produce identical carry treedefs (the
D4 preset test).

The physics lives on the **core** (``hy.Core``, gravity-first: the
physical constant centralizes there), the required physics modules
(``stratification=`` and ``free_surface=`` have **no defaults** —
owner-ratified, no surprising default physics) and the **scaling**
policy (``fr.scaling``): ``scaling=`` names the reference time frame
(default: ``fr.scaling.Dimensional()`` — a dimensional assembly needs
no scaling argument at all). The retired preset kwargs (``csqr=``,
``rossby_number=``, ``dt=``) raise taught TypeErrors naming the new
spelling.

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

The factory default is ``CenteredAdvection()`` (the common-
denominator scheme); ``advection=False`` recovers the linear model
(what the dispersion, geostrophic-balance and energy-conservation
gates validate), and ``UpwindAdvection`` / ``WENOAdvection`` are
accepted on the uniform unmapped grids the hydrostatic preset targets
(the biased schemes self-reject stretched / mapped geometry at bind).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr
from fridom.model.modules.advection import CenteredAdvection

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from fridom.model.model import Model as _Model
    from fridom.model.time_steppers.base import TimeStepper
    from fridom.spatial.grid import Grid

#: retired preset kwargs -> the taught replacement spelling
_RETIRED_KWARGS = {
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
}


def Model(  # noqa: N802 — a factory that mirrors fr.model.Model's surface
    *,
    grid: Grid,
    core: fr.model.Module,
    stratification: fr.model.Module,
    free_surface: fr.model.Module,
    time_stepper: TimeStepper,
    scaling: object | None = None,
    coriolis: fr.model.Module | None = None,
    advection: fr.model.Module | bool = True,
    surface_advective_flux: bool | None = None,
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
    core : fr.model.Module
        The dynamical core, ``hy.Core``: the gravity-first physical
        constant lives here (``gravity=``, dimensional) — the
        free-surface family references it; the nondimensional core
        takes no kwarg at all.
    stratification : fr.model.Module
        The stratification module — REQUIRED, no default physics:
        pass ``hy.ConstantStratification(n2=...)`` (dimensional) /
        ``ConstantStratification(froude_number=...)``
        (nondimensional).
    free_surface : fr.model.Module
        The barotropic (surface-pressure) module — REQUIRED, no
        default: ``hy.ExplicitFreeSurface(...)`` /
        ``hy.ImplicitFreeSurface(...)`` /
        ``hy.SplitExplicitFreeSurface(...)``. Owns the ``ps``
        declaration and its evolution.
    time_stepper : TimeStepper
        The time stepper (e.g.
        ``fr.model.time_steppers.AdamBashforth(dt, order=3)``).
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
    advection : fr.model.Module | bool, optional
        Nonlinear advection of ``u``/``v``/``b``. ``True`` (the
        default) installs ``CenteredAdvection()``; ``False`` omits
        advection (the linear hydrostatic model); a module instance
        (``UpwindAdvection`` / ``WENOAdvection`` / a configured
        ``CenteredAdvection``) is installed as given. The module is
        scaling-neutral and adopts the assembly's variant at bind.
        The vertical leg consumes the diagnosed ``w`` on the
        ``Outer`` faces through the seeded ``Outer -> Inner``
        restriction (module docstring); the boundary-face flux is a
        structural zero (default: True).
    surface_advective_flux : bool | None, optional
        Tri-state control of the constancy-preserving **surface
        closure** on the default advection module. The default ``None``
        is the closure: advect **through** the top/bottom boundary faces
        with the one-sided (top-cell) face value (the Oceananigans-
        equivalent linear-free-surface treatment), so ``A(q=const)`` is
        machine-zero in every cell and tracer content is exchanged with
        the moving surface. ``False`` restores the legacy fixed-domain
        closure — it drops the surface velocity ``w(0)`` and so conserves
        tracer content to roundoff, at the price of the surface-cell
        constancy violation ``A(q=const) ~ q*w(0)/dz`` (the source that
        makes the implicit free surface unstable). Only shapes the
        default-constructed advection; a user-passed module carries its
        own ``surface_flux`` (whose ``None`` auto-resolves to the same
        closure on the hydrostatic ``Outer``-``w`` grid) (default: None).
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
    TypeError
        On the retired kwargs ``csqr=`` / ``rossby_number=`` /
        ``dt=`` (taught messages naming the new spelling).
    """
    for retired, message in _RETIRED_KWARGS.items():
        if retired in kwargs:
            raise TypeError(f"hy.Model {message}")
    if scaling is None:
        scaling = fr.scaling.Dimensional()
    if advection is True:
        advection = CenteredAdvection(surface_flux=surface_advective_flux)

    modules: list[fr.model.Module] = [core]
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
                          time_stepper=time_stepper, name=name,
                          scaling=scaling, **kwargs)
