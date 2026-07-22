r"""
Shallow-water dimensional-factor tables (``model.units`` rows).

Description
-----------
The §D amplitude table of the shallow-water core (the
``gravity_wave`` mechanism owner), contributed to ``model.units``
through :attr:`fridom.shallowwater2.modules.core.Core.unit_factors`.
Every factor follows the one rule :math:`T_\mathrm{ref} =
\varepsilon\,L/U`:

- velocities ``u`` / ``v``: :math:`U` [m/s];
- pressure ``p``: :math:`U^2/\varepsilon` [m²/s²] (the
  simplest-linear-operator normalization);
- ``thickness`` / ``csqr``: :math:`(U/\mathrm{Fr})^2` [m²/s²] (the
  geopotential scale :math:`c^2 = g\,D`);
- curated ``h``: :math:`U^2/(\varepsilon\,g)` [m] — the surface
  displacement in meters, ``factor("h") * p``; on a **dimensional**
  model it keeps its meaning as :math:`1/g` from the bound
  ``shallowwater.gravity`` (owner ruling 2026-07-21), so the same
  spelling yields meters in both variants;
- derived constants :math:`c_\mathrm{dim} = U/\mathrm{Fr}` [m/s]
  and :math:`D = U^2/(\mathrm{Fr}^2\,g)` [m] (the paper's
  :math:`H = D\,\mathrm{Fr}` reproduces as
  ``factor("h") == D * Fr`` under ``GravityWave``); dimensional
  models report :math:`\sqrt{g\,D}` and the bound depth instead;
- coordinates: :math:`L` [m] (keyed by the core's ``coords=``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.params import SCALING_NONLINEARITY
from fridom.model.units import UnitFactor
from fridom.shallowwater2.params import DEPTH, FROUDE, GRAVITY

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


# ================================================================
#  Row resolvers (fn(values) over the resolved symbols)
# ================================================================
def _velocity(values: Mapping[str, float]) -> float:
    """Return the velocity amplitude ``U``."""
    return values["U"]


def _pressure(values: Mapping[str, float]) -> float:
    """Return the pressure amplitude ``U^2/eps``."""
    return values["U"] ** 2 / values["eps"]


def _geopotential(values: Mapping[str, float]) -> float:
    """Return the geopotential scale ``(U/Fr)^2 = g*D``."""
    return (values["U"] / values["Fr"]) ** 2


def _height(values: Mapping[str, float]) -> float:
    """Return the curated height factor ``U^2/(eps*g)``."""
    return values["U"] ** 2 / (values["eps"] * values["g"])


def _height_dim(values: Mapping[str, float]) -> float:
    """Return the dimensional height factor ``1/g`` (bound g)."""
    return 1.0 / values["g"]


def _phase_speed(values: Mapping[str, float]) -> float:
    """Return the dimensional phase speed ``c_dim = U/Fr``."""
    return values["U"] / values["Fr"]


def _phase_speed_dim(values: Mapping[str, float]) -> float:
    """Return the bound phase speed ``sqrt(g*D)``."""
    return (values["g"] * values["D"]) ** 0.5


def _depth_scale(values: Mapping[str, float]) -> float:
    """Return the dimensional depth ``D = U^2/(Fr^2*g)``."""
    return values["U"] ** 2 / (values["Fr"] ** 2 * values["g"])


def _depth_dim(values: Mapping[str, float]) -> float:
    """Return the bound water depth ``D``."""
    return values["D"]


def _length(values: Mapping[str, float]) -> float:
    """Return the coordinate scale ``L``."""
    return values["L"]


# ================================================================
#  The tables
# ================================================================
_VELOCITY_FACTOR = UnitFactor(
    unit="m/s", expr="U", kind="component", scales=("U",),
    fn=_velocity)

_GEOPOTENTIAL_FACTOR = UnitFactor(
    unit="m^2/s^2", expr="(U/Fr)^2", kind="component",
    scales=("U",), params={"Fr": FROUDE}, fn=_geopotential)

#: the per-component (and curated / derived-constant) rows
COMPONENT_FACTORS: dict[str, UnitFactor] = {
    "u": _VELOCITY_FACTOR,
    "v": _VELOCITY_FACTOR,
    "p": UnitFactor(
        unit="m^2/s^2", expr="U^2/eps", kind="component",
        scales=("U",), params={"eps": SCALING_NONLINEARITY},
        fn=_pressure),
    "thickness": _GEOPOTENTIAL_FACTOR,
    "csqr": _GEOPOTENTIAL_FACTOR,
    "h": UnitFactor(
        unit="m", expr="U^2/(eps*g)", kind="curated",
        scales=("U", "g"), params={"eps": SCALING_NONLINEARITY},
        fn=_height, dim_expr="1/g", dim_params={"g": GRAVITY},
        dim_fn=_height_dim),
    "c_dim": UnitFactor(
        unit="m/s", expr="U/Fr", kind="constant", scales=("U",),
        params={"Fr": FROUDE}, fn=_phase_speed,
        dim_expr="sqrt(g*D)", dim_params={"g": GRAVITY, "D": DEPTH},
        dim_fn=_phase_speed_dim),
    "D": UnitFactor(
        unit="m", expr="U^2/(Fr^2*g)", kind="constant",
        scales=("U", "g"), params={"Fr": FROUDE}, fn=_depth_scale,
        dim_expr="D", dim_params={"D": DEPTH}, dim_fn=_depth_dim),
}


def coordinate_factors(
    zonal: str, meridional: str,
) -> dict[str, UnitFactor]:
    """
    Return the two coordinate rows, keyed by the core's coords.

    Description
    -----------
    Both horizontal coordinates scale with the reference length
    ``L``; the keys follow the core's ``coords=`` renaming (e.g.
    ``("lon", "lat")`` on the standard sphere chart).

    Parameters
    ----------
    zonal : str
        The zonal coordinate name.
    meridional : str
        The meridional coordinate name.

    Returns
    -------
    dict[str, UnitFactor]
        The two ``L``-valued coordinate rows.
    """
    row = UnitFactor(unit="m", expr="L", kind="coordinate",
                     scales=("L",), fn=_length)
    return {zonal: row, meridional: row}
