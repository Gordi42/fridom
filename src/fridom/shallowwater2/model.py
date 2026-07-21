"""
The shallow-water preset factory.

Description
-----------
``sw.Model(grid=..., core=..., ...)`` is a **thin factory
function** (never a subclass, D1.3 commitment 2 / model.md section 6):
it builds the shallow-water module tuple and delegates to a plain
``fr.model.Model``. Preset and explicit assembly produce **identical
carry treedefs** — the D4 preset test. The factory may only build the
module tuple, forward kwargs, and pick default modules; it holds no
parameters or fields and never mutates the model.

The physics lives on the **core** (``sw.Core``) and the **scaling**
policy (``fr.scaling``): the core's kwarg set fixes the variant
(dimensional ``gravity=`` + ``depth=`` XOR nondimensional
``froude_number=``), and ``scaling=`` names the reference time frame
(default: ``fr.scaling.Dimensional()`` — a dimensional assembly needs
no scaling argument at all). The retired preset kwargs ``csqr=``,
``rossby_number=`` and ``coords=`` raise taught TypeErrors naming the
new spelling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr
from fridom.shallowwater2.modules.coriolis import (
    carries_linear_rotation,
    check_rotation_modules,
)
from fridom.shallowwater2.modules.sadourny import SadournyAdvection

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from fridom.model.model import Model as _Model
    from fridom.model.time_steppers.base import TimeStepper
    from fridom.spatial.grid import Grid

#: retired preset kwargs -> the taught replacement spelling
_RETIRED_KWARGS = {
    "csqr": (
        "csqr= is retired: the physics lives on the core — pass "
        "core=sw.Core(gravity=..., depth=...) (dimensional, "
        "csqr = g*D) or core=sw.Core(froude_number=..., depth=...) "
        "with scaling=fr.scaling.GravityWave() (nondimensional, "
        "the depth ratio)"),
    "rossby_number": (
        "rossby_number= is retired: the nonlinearity number is the "
        "scaling mechanism's own regime number — pass "
        "core=sw.Core(froude_number=...) with "
        "scaling=fr.scaling.GravityWave() (epsilon = Fr), or a "
        "nondimensional Coriolis module "
        "(FPlaneCoriolis(rossby_number=...)) with "
        "scaling=fr.scaling.Rotational() (epsilon = Ro)"),
    "coords": (
        "coords= is retired on the preset: the coordinate names "
        "live on the core — pass core=sw.Core(..., "
        "coords=('lon', 'lat')); the advection module adopts them "
        "from the core"),
}


def Model(  # noqa: N802 — constructor-like factory (D1.3)
    *,
    grid: Grid,
    core: fr.model.Module,
    time_stepper: TimeStepper,
    scaling: object | None = None,
    coriolis: fr.model.Module | None = None,
    advection: bool = True,
    modules_extra: Sequence[fr.model.Module] = (),
    name: str | None = None,
    **kwargs: object,
) -> _Model:
    r"""
    Assemble a shallow-water model (thin preset over ``fr.model.Model``).

    Description
    -----------
    Works on flat Cartesian grids and on chart-coupled grids
    (coordinate-systems plan, stage C2). The one-line spherical
    assemble is ``fr.spatial.spherical.Grid`` — periodic-lon x
    bounded-lat interval meshes under the orthogonal lat-lon sphere
    chart:

    .. code-block:: python

        grid = fr.spatial.spherical.Grid(
            (nlon, nlat), radius=a, lat_extent=(-lat_max, lat_max))
        model = sw.Model(
            grid=grid,
            core=sw.Core(gravity=g, depth=H,
                         coords=("lon", "lat")),
            coriolis=sw.modules.RotationCoriolis(
                omega=(0.0, 0.0, omega), coords=("lon", "lat"),
                metric_weight="csqr"),
            time_stepper=...)

    Prognostic velocities are the **physical** (m/s) components on
    every grid, chart grids included (``physical_state_components.md``
    ruling (c)) — so a spherical IC is set in physical m/s. The
    chart-native coordinate velocities are exposed read-only via
    ``state.chart`` (ruling (d)).

    Parameters
    ----------
    grid : Grid
        The (periodic, walled, or chart-coupled) 2-D grid.
    core : fr.model.Module
        The dynamical core, ``sw.Core``: its kwarg set fixes the
        scaling variant — dimensional ``gravity=`` + ``depth=``
        (physical parameters, zero scaling operations in the trace)
        XOR nondimensional ``froude_number=`` (+ optional depth
        ratio) under a nondimensional ``scaling=``. Coordinate names
        (``coords=``) live on the core; the advection adopts them.
    time_stepper : TimeStepper
        The time stepper (e.g.
        ``fr.model.time_steppers.AdamBashforth(dt, order=3)``).
    scaling : object | None, optional
        The ``fr.scaling`` policy naming the reference time frame;
        ``None`` defaults to ``fr.scaling.Dimensional()`` — a
        dimensional assembly needs no scaling argument. A
        nondimensional core/Coriolis kwarg set needs the matching
        nondimensional policy (``fr.scaling.GravityWave()`` /
        ``Rotational()`` / ``Advective()``); the assembly refuses a
        mismatch with a taught error (default: None).
    coriolis : fr.model.Module | None, optional
        The Coriolis module. ``None`` — the argument omitted, the
        default — means **no rotation at all**: no ``f_coriolis``
        field, no rotation term. Rotation is opt-in: pass
        ``sw.modules.FPlaneCoriolis(f0=...)`` (dimensional) /
        ``FPlaneCoriolis(rossby_number=...)`` (nondimensional) /
        ``BetaPlaneCoriolis(...)`` on a flat grid, or
        ``sw.modules.RotationCoriolis(omega=(0.0, 0.0, Omega),
        coords=...)`` on a chart-coupled grid (default: None).

        A Coriolis module combined with a variable-depth core must
        carry ``metric_weight="csqr"`` itself — the
        thickness-weighted rotation, exactly M-skew for any ``f``
        and any positive depth profile; the preset raises otherwise
        (without it the rotation does work against the
        :math:`c^2`-weighted energy metric).

        **Exact energy conservation.** The linear rotation is skew
        under the *linearized* metric, not under the
        thickness-weighted energy the nonlinear scheme conserves
        (``sw.diagnostics.etot_full``), which it therefore produces
        at :math:`O(\varepsilon)`. Two ways to fix that, both exact
        (``sw.modules.coriolis``): add ``modules_extra=(
        sw.modules.CoriolisEnergyCorrection(coords=...),)`` next to
        the linear module — the linear operator ``L`` stays
        bit-for-bit unchanged, so eigenmodes / projections / balance
        keep working — or pass the conserving module itself
        (``coriolis=sw.modules.NonlinearFPlaneCoriolis(...)``),
        which is cheaper but leaves ``L`` without any rotation (no
        eigenmodes, projections, balance).
    advection : bool, optional
        Include the Sadourny nonlinear advection; the module is
        scaling-neutral and adopts the assembly's variant at bind
        (default: True).
    modules_extra : Sequence[fr.model.Module], optional
        Additional modules (tracers, closures) appended after the
        core physics (default: ()).
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
        ``coords=`` (taught messages naming the new spelling).
    ValueError
        A variable-depth core combined with a linear Coriolis module
        whose ``metric_weight`` is unset; or a module tuple that
        counts the rotation twice
        (``sw.modules.coriolis.check_rotation_modules``).
    """
    for retired, message in _RETIRED_KWARGS.items():
        if retired in kwargs:
            raise TypeError(f"sw.Model {message}")
    if scaling is None:
        scaling = fr.scaling.Dimensional()
    modules: tuple[fr.model.Module, ...] = (core,)
    # rotation is opt-in: coriolis=None installs no module at all
    if coriolis is not None:
        # the conserving (route B) modules weight the rotation by the
        # thickness itself — exact for any depth profile, so the
        # metric_weight requirement does not apply to them
        if (getattr(core, "variable_depth", False)
                and carries_linear_rotation(coriolis)
                and coriolis.metric_weight is None):
            raise ValueError(
                "a variable-depth shallow-water model (a callable / "
                "law depth on the core) needs the thickness-weighted"
                " rotation: construct the Coriolis module with "
                "metric_weight='csqr' (without it the rotation "
                "does work against the c^2-weighted energy metric "
                "and the model no longer conserves energy exactly)")
        modules += (coriolis,)
    if advection:
        modules += (SadournyAdvection(
            coords=getattr(core, "coords", ("x", "y"))),)
    modules += tuple(modules_extra)
    # immersed (cut-cell) grid: one shared CONSTRAINT-stage MaskState
    # keeps every prognostic's dry DOFs dead against the modules that
    # do not consult the mask (Coriolis) — the fraction-weighted core
    # continuity / Sadourny transport handle the wet region themselves
    # (IP-D5). Appended last so its masking runs after the physics.
    if getattr(grid, "immersed", None) is not None:
        modules += (fr.model.modules.MaskState(),)
    # the rotation must be counted exactly once (route A: linear +
    # correction; route B: the conserving module alone)
    check_rotation_modules(modules)
    return fr.model.Model(grid=grid, modules=modules,
                    time_stepper=time_stepper, name=name,
                    scaling=scaling, **kwargs)
