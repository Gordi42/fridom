"""
The shallow-water preset factory.

Description
-----------
``sw.Model(grid=..., coriolis=..., ...)`` is a **thin factory
function** (never a subclass, D1.3 commitment 2 / model.md section 6):
it builds the shallow-water module tuple and delegates to a plain
``fr.model.Model``. Preset and explicit assembly produce **identical carry
treedefs** — the D4 preset test. The factory may only build the
module tuple, forward kwargs, and pick default modules/stepper; it
holds no parameters or fields and never mutates the model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import fridom as fr
from fridom.model.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
    RotationCoriolis,
)
from fridom.shallowwater2.modules.core import DynamicalCore
from fridom.shallowwater2.modules.coriolis import (
    carries_linear_rotation,
    check_rotation_modules,
)
from fridom.shallowwater2.modules.sadourny import SadournyAdvection

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from fridom.model.model import Model as _Model
    from fridom.model.time_steppers.base import TimeStepper
    from fridom.spatial.grid import Grid


def Model(  # noqa: N802 — constructor-like factory (D1.3)
    *,
    grid: Grid,
    csqr: float | Callable = 1.0,
    rossby_number: float = 1.0,
    coriolis: fr.model.Module | None = None,
    advection: bool = True,
    coords: tuple[str, str] = ("x", "y"),
    time_stepper: TimeStepper | None = None,
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
    chart (the metric is diagonal, so the grid drops the cross-term
    index moves a bounded axis cannot interpolate):

    .. code-block:: python

        grid = fr.spatial.spherical.Grid(
            (nlon, nlat), radius=a, lat_extent=(-lat_max, lat_max))
        model = sw.Model(
            grid=grid, coords=("lon", "lat"), csqr=gh0,
            rossby_number=1.0,
            coriolis=sw.modules.RotationCoriolis(
                omega=(0.0, 0.0, omega), coords=("lon", "lat"),
                metric_weight="csqr"),
            time_stepper=...)

    On the lat-lon chart a polar ``omega=(0, 0, Omega)`` makes the
    derived ``f = 2 Omega . n_hat`` the familiar
    ``2 Omega sin(lat)``.

    Prognostic velocities on chart grids are the contravariant
    components (see ``sw.modules.DynamicalCore``); convert to
    physical m/s components via ``state.u_physical`` /
    ``state.v_physical``.

    Parameters
    ----------
    grid : Grid
        The (periodic, walled, or chart-coupled) 2-D grid.
    csqr : float | Callable, optional
        Squared gravity-wave phase speed :math:`c^2`: a float for
        constant depth, or a callable ``csqr(y)`` for variable
        depth (materialized into a meridional ``csqr`` profile
        field; no constant ``shallowwater.csqr`` provide)
        (default: 1.0).
    rossby_number : float, optional
        Rossby number scaling the advection (default: 1.0).
    coriolis : fr.model.Module | None, optional
        The Coriolis field provider. ``None`` — the argument
        omitted, the default — means **no rotation at all**: no
        Coriolis module is installed, so the model carries no
        ``f_coriolis`` field, no rotation term and no
        ``coriolis.f0`` provide. Rotation is opt-in: pass
        ``sw.modules.FPlaneCoriolis(f0=...)`` /
        ``sw.modules.BetaPlaneCoriolis(...)`` on a flat grid, or
        ``sw.modules.RotationCoriolis(omega=(0.0, 0.0, Omega),
        coords=...)`` on a chart-coupled grid (default: None).

        A Coriolis module combined with a callable ``csqr`` must
        carry ``metric_weight="csqr"`` itself — the
        thickness-weighted rotation, exactly M-skew for any ``f``
        and any positive depth profile; the preset raises otherwise
        (without it the rotation does work against the
        :math:`c^2`-weighted energy metric).

        **Exact energy conservation.** The linear rotation is skew
        under the *linearized* metric, not under the
        thickness-weighted energy the nonlinear scheme conserves
        (``sw.diagnostics.etot_full``), which it therefore produces
        at :math:`O(\mathrm{Ro})`. Two ways to fix that, both
        exact (``sw.modules.coriolis``): add
        ``modules_extra=(sw.modules.CoriolisEnergyCorrection(
        coords=coords),)`` next to the linear module — the linear
        operator ``L`` stays bit-for-bit unchanged, so eigenmodes /
        projections / balance keep working — or pass the conserving
        module itself (``coriolis=sw.modules.NonlinearFPlaneCoriolis(
        f0=...)``), which is cheaper but leaves ``L`` without any
        rotation (no eigenmodes, projections, balance).
    advection : bool, optional
        Include the Sadourny nonlinear advection (default: True).
    coords : tuple[str, str], optional
        The (zonal, meridional) coordinate names in the grid's
        factor order, forwarded to the core and the advection —
        ``("lon", "lat")`` on the standard sphere chart
        (default: ``("x", "y")``).
    time_stepper : TimeStepper | None, optional
        The stepper; default ``AdamBashforth(dt=1.0, order=3)`` (the
        cutover package default — pass an explicit one for a real
        run).
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
    ValueError
        A callable ``csqr`` combined with a linear Coriolis module
        whose ``metric_weight`` is unset; or a module tuple that
        counts the rotation twice
        (``sw.modules.coriolis.check_rotation_modules``).
    """
    core = DynamicalCore(csqr=csqr, rossby_number=rossby_number,
                         coords=coords)
    modules: tuple[fr.model.Module, ...] = (core,)
    # rotation is opt-in: coriolis=None installs no module at all
    if coriolis is not None:
        # the conserving (route B) modules weight the rotation by the
        # thickness itself — exact for any depth profile, so the
        # metric_weight requirement does not apply to them
        if (callable(csqr)
                and isinstance(coriolis,
                               FPlaneCoriolis | BetaPlaneCoriolis
                               | RotationCoriolis)
                and carries_linear_rotation(coriolis)
                and coriolis.metric_weight is None):
            raise ValueError(
                "a variable-depth shallow-water model (callable "
                "csqr) needs the thickness-weighted rotation: "
                "construct the Coriolis module with "
                "metric_weight='csqr' (without it the rotation "
                "does work against the c^2-weighted energy metric "
                "and the model no longer conserves energy exactly)")
        modules += (coriolis,)
    if advection:
        modules += (SadournyAdvection(coords=coords),)
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
    if time_stepper is None:
        time_stepper = fr.model.time_steppers.AdamBashforth(dt=1.0, order=3)
    return fr.model.Model(grid=grid, modules=modules,
                    time_stepper=time_stepper, name=name, **kwargs)
