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
    NoCoriolis,
    RotationCoriolis,
    require_flat_grid_for_the_default,
)
from fridom.shallowwater2.modules.core import DynamicalCore
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
    coriolis: fr.model.Module | bool | None = None,
    advection: bool = True,
    coords: tuple[str, str] = ("x", "y"),
    time_stepper: TimeStepper | None = None,
    modules_extra: Sequence[fr.model.Module] = (),
    name: str | None = None,
    **kwargs: object,
) -> _Model:
    """
    Assemble a shallow-water model (thin preset over ``fr.model.Model``).

    Description
    -----------
    Works on flat Cartesian grids and on chart-coupled grids
    (coordinate-systems plan, stage C2). The spherical recipe —
    periodic-lon x bounded-lat interval meshes under an embedding
    chart, with the diagonal (orthogonal-metric) index-move
    overrides:

    .. code-block:: python

        mlon = fr.spatial.meshes.IntervalMesh(
            nlon, (0.0, 2 * np.pi), name="lon")
        mlat = fr.spatial.meshes.IntervalMesh(
            nlat, (-lat_max, lat_max), periodic=False, name="lat")
        mapping = fr.spatial.CoordinateMapping(chart={
            "X": lambda lon, lat: (
                a * jnp.cos(lat) * jnp.cos(lon),
                a * jnp.cos(lat) * jnp.sin(lon),
                a * jnp.sin(lat))})
        grid = fr.spatial.Grid((mlon, mlat), mapping=mapping)
        grid.merge_overrides({
            "raise_index": fr.spatial.operators.RaiseIndex(
                ("lon", "lat"), diagonal=True),
            "lower_index": fr.spatial.operators.LowerIndex(
                ("lon", "lat"), diagonal=True)})
        model = sw.Model(
            grid=grid, coords=("lon", "lat"), csqr=gh0,
            rossby_number=1.0,
            coriolis=sw.modules.SphericalCoriolis(
                omega=omega, metric_weight="csqr"),
            time_stepper=...)

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
    coriolis : fr.model.Module | bool | None, optional
        The Coriolis field provider. ``None`` (the default) installs
        ``FPlaneCoriolis(f0=1.0, metric_weight="csqr")`` on **flat**
        grids — the nondimensional f-plane the existing setups rely
        on — and **raises** on chart grids, where a metric-blind
        rotation would be silently wrong physics (name
        ``RotationCoriolis`` / ``SphericalCoriolis`` there).
        ``False`` is the explicit no-rotation option (sugar for
        ``fr.modules.NoCoriolis()``). The default rotation is
        **always** thickness-weighted (it is exactly M-skew for any
        ``f`` and any positive depth profile, and coincides with the
        unweighted form for constant depth to rounding — bitwise for
        power-of-two ``csqr``). An explicit framework Coriolis module combined
        with a callable ``csqr`` must carry
        ``metric_weight="csqr"`` itself; the preset raises
        otherwise.
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
        A callable ``csqr`` combined with an explicit framework
        Coriolis module whose ``metric_weight`` is unset, or
        ``coriolis=None`` on a chart-coupled grid.
    """
    core = DynamicalCore(csqr=csqr, rossby_number=rossby_number,
                         coords=coords)
    if coriolis is False:
        cor = NoCoriolis()
    elif coriolis is None:
        require_flat_grid_for_the_default(grid)
        # always the thickness-weighted rotation: exactly M-skew for
        # any f and any depth profile, and identical to the unweighted
        # form for constant depth (to rounding; bitwise for
        # power-of-two csqr)
        cor = FPlaneCoriolis(f0=1.0, metric_weight="csqr")
    else:
        cor = coriolis
        if (callable(csqr)
                and isinstance(cor, FPlaneCoriolis | BetaPlaneCoriolis
                               | RotationCoriolis)
                and cor.metric_weight is None):
            raise ValueError(
                "a variable-depth shallow-water model (callable "
                "csqr) needs the thickness-weighted rotation: "
                "construct the Coriolis module with "
                "metric_weight='csqr' (without it the rotation "
                "does work against the c^2-weighted energy metric "
                "and the model no longer conserves energy exactly)")
    modules: tuple[fr.model.Module, ...] = (core, cor)
    if advection:
        modules += (SadournyAdvection(coords=coords),)
    modules += tuple(modules_extra)
    if time_stepper is None:
        time_stepper = fr.model.time_steppers.AdamBashforth(dt=1.0, order=3)
    return fr.model.Model(grid=grid, modules=modules,
                    time_stepper=time_stepper, name=name, **kwargs)
