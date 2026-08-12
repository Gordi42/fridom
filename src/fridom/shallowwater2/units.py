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
    target_unit="m/s", expr="U", kind="component", scales=("U",),
    fn=_velocity)

_GEOPOTENTIAL_FACTOR = UnitFactor(
    target_unit="m^2/s^2", expr="(U/Fr)^2", kind="component",
    scales=("U",), params={"Fr": FROUDE}, fn=_geopotential)

def _vorticity(values: Mapping[str, float]) -> float:
    """Return the vorticity amplitude ``U/L``."""
    return values["U"] / values["L"]


def _energy(values: Mapping[str, float]) -> float:
    """Return the specific-energy amplitude ``U^2``."""
    return values["U"] ** 2


def _energy_full(values: Mapping[str, float]) -> float:
    """Return the thickness-weighted amplitude ``(U^2/eps)^2``."""
    return (values["U"] ** 2 / values["eps"]) ** 2


_VORTICITY_FACTOR = UnitFactor(
    target_unit="1/s", expr="U/L", kind="derived",
    scales=("L", "U"), fn=_vorticity)

_SPECIFIC_ENERGY_FACTOR = UnitFactor(
    target_unit="m^2/s^2", expr="U^2", kind="derived",
    scales=("U",), fn=_energy)

#: the diagnosed quantities' rows (5.6). ``ekin_full``,
#: ``etot_full`` and ``pot_vort`` are DELIBERATELY absent: their
#: factors turn on the geopotential-thickness convention (the
#: ``thickness`` row is ``(U/Fr)^2`` while ``p`` is ``U^2/eps``),
#: which is an open owner call. An absent row leaves the quantity
#: honestly unconvertible; a guessed one would repeat the lat-lon
#: failure of reading a row as something it is not.
DERIVED_FACTORS: dict[str, UnitFactor] = {
    "rel_vort": _VORTICITY_FACTOR,
    "divergence": _VORTICITY_FACTOR,
    "ekin": _SPECIFIC_ENERGY_FACTOR,
    # epot = 0.5*p^2/c^2_eff: the nondimensional c^2_eff carries
    # (eps/Fr)^2*D, which cancels the p amplitude's eps and the
    # depth alike, leaving ekin's factor -- so etot is well defined
    "epot": _SPECIFIC_ENERGY_FACTOR,
    # epot_full = 0.5*p^2 (no 1/c^2), so it is simply the pressure
    # amplitude squared and needs no thickness convention
    "epot_full": UnitFactor(
        target_unit="m^4/s^4", expr="(U^2/eps)^2", kind="derived",
        scales=("U",), params={"eps": SCALING_NONLINEARITY},
        fn=_energy_full),
}

#: the per-component (and curated / derived-constant) rows
COMPONENT_FACTORS: dict[str, UnitFactor] = {
    "u": _VELOCITY_FACTOR,
    "v": _VELOCITY_FACTOR,
    "p": UnitFactor(
        target_unit="m^2/s^2", expr="U^2/eps", kind="component",
        scales=("U",), params={"eps": SCALING_NONLINEARITY},
        fn=_pressure),
    "thickness": _GEOPOTENTIAL_FACTOR,
    "csqr": _GEOPOTENTIAL_FACTOR,
    "h": UnitFactor(
        target_unit="m", expr="U^2/(eps*g)", kind="curated",
        scales=("U", "g"), params={"eps": SCALING_NONLINEARITY},
        fn=_height, dim_expr="1/g", dim_params={"g": GRAVITY},
        dim_fn=_height_dim),
    "c_dim": UnitFactor(
        target_unit="m/s", expr="U/Fr", kind="constant", scales=("U",),
        params={"Fr": FROUDE}, fn=_phase_speed,
        dim_expr="sqrt(g*D)", dim_params={"g": GRAVITY, "D": DEPTH},
        dim_fn=_phase_speed_dim),
    "D": UnitFactor(
        target_unit="m", expr="U^2/(Fr^2*g)", kind="constant",
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
    row = UnitFactor(target_unit="m", expr="L", kind="coordinate",
                     scales=("L",), fn=_length)
    return {zonal: row, meridional: row}
