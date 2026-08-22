r"""
Nonhydrostatic dimensional-factor tables (``model.units`` rows).

Description
-----------
The §D amplitude table of the nonhydrostatic core (the aspect-ratio
owner), contributed to ``model.units`` through
:attr:`fridom.nonhydro2.modules.core.Core.unit_factors`, plus the
stratification family's derived buoyancy frequency. Every factor
follows the one rule :math:`T_\mathrm{ref} = \varepsilon\,L/U`, with
the vertical scale :math:`\delta L` read live off the core's
``nonhydro.aspect_ratio`` provide:

- coordinates: horizontals :math:`L` [m], the vertical
  :math:`\delta\,L` [m] (keyed by the core's ``coords=`` /
  ``vertical=`` names);
- velocities ``u`` / ``v``: :math:`U` [m/s]; ``w``:
  :math:`\delta\,U` [m/s];
- pressure ``p``: :math:`U^2/\varepsilon` [m²/s²] (the
  simplest-linear-operator normalization);
- buoyancy ``b``: :math:`U^2/(\varepsilon\,\delta\,L)` [m/s²];
- derived constant :math:`N_\mathrm{dim} =
  U/(\mathrm{Fr}_\mathrm{int}\,\delta\,L)` [1/s] (the stratification
  family's row); a **dimensional** model reports :math:`\sqrt{N^2}`
  from the bound ``stratification.n2`` instead (the profile-valued
  ``MeridionalStratification`` binds no constant, so its row is
  marked unresolvable rather than claiming a false constant — the
  beta-plane ``f_dim`` precedent).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.model.params import (
    SCALING_NONLINEARITY,
    STRATIFICATION_FROUDE,
    STRATIFICATION_N2,
)
from fridom.model.units import UnitFactor
from fridom.nonhydro2.params import ASPECT_RATIO

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


# ================================================================
#  Row resolvers (fn(values) over the resolved symbols)
# ================================================================
def _velocity(values: Mapping[str, float]) -> float:
    """Return the horizontal velocity amplitude ``U``."""
    return values["U"]


def _vertical_velocity(values: Mapping[str, float]) -> float:
    """Return the vertical velocity amplitude ``delta*U``."""
    return values["delta"] * values["U"]


def _pressure(values: Mapping[str, float]) -> float:
    """Return the pressure amplitude ``U^2/eps``."""
    return values["U"] ** 2 / values["eps"]


def _buoyancy(values: Mapping[str, float]) -> float:
    """Return the buoyancy amplitude ``U^2/(eps*delta*L)``."""
    return values["U"] ** 2 / (
        values["eps"] * values["delta"] * values["L"])


def _frequency(values: Mapping[str, float]) -> float:
    """Return the dimensional frequency ``N_dim = U/(Fr*delta*L)``."""
    return values["U"] / (
        values["Fr"] * values["delta"] * values["L"])


def _frequency_dim(values: Mapping[str, float]) -> float:
    """Return the bound frequency ``sqrt(n2)``."""
    return values["n2"] ** 0.5


def _length(values: Mapping[str, float]) -> float:
    """Return the horizontal coordinate scale ``L``."""
    return values["L"]


def _height(values: Mapping[str, float]) -> float:
    """Return the vertical coordinate scale ``delta*L``."""
    return values["delta"] * values["L"]


# ================================================================
#  The tables
# ================================================================
_VELOCITY_FACTOR = UnitFactor(
    target_unit="m/s", expr="U", kind="component", scales=("U",),
    fn=_velocity)

#: the core's per-component rows (u, v, w, p, b)
COMPONENT_FACTORS: dict[str, UnitFactor] = {
    "u": _VELOCITY_FACTOR,
    "v": _VELOCITY_FACTOR,
    "w": UnitFactor(
        target_unit="m/s", expr="delta*U", kind="component",
        scales=("U",), params={"delta": ASPECT_RATIO},
        fn=_vertical_velocity),
    "p": UnitFactor(
        target_unit="m^2/s^2", expr="U^2/eps", kind="component",
        scales=("U",), params={"eps": SCALING_NONLINEARITY},
        fn=_pressure),
    "b": UnitFactor(
        target_unit="m/s^2", expr="U^2/(eps*delta*L)", kind="component",
        scales=("L", "U"),
        params={"eps": SCALING_NONLINEARITY, "delta": ASPECT_RATIO},
        fn=_buoyancy),
}

def _vorticity(values: Mapping[str, float]) -> float:
    """Return the vorticity amplitude ``U/L``."""
    return values["U"] / values["L"]


def _energy(values: Mapping[str, float]) -> float:
    """Return the specific-energy amplitude ``U^2``."""
    return values["U"] ** 2


def _linear_pot_vort(values: Mapping[str, float]) -> float:
    """Return the linear-PV amplitude ``U/(eps*L)``."""
    return values["U"] / (values["eps"] * values["L"])


_ENERGY_FACTOR = UnitFactor(
    target_unit="m^2/s^2", expr="U^2", kind="derived",
    scales=("U",), fn=_energy)

#: the diagnosed quantities' rows, so a nondimensional store can be
#: converted back (5.6). Absent rows are deliberate, never sloppy:
#: a quantity whose factor is not derived is better left unstamped
#: than stamped wrong.
DERIVED_FACTORS: dict[str, UnitFactor] = {
    "rel_vort_z": UnitFactor(
        target_unit="1/s", expr="U/L", kind="derived",
        scales=("L", "U"), fn=_vorticity),
    "ekin": _ENERGY_FACTOR,
    # epot = 0.5*b^2/N^2_eff: the amplitude ratio is
    # U^2*(Fr/eps)^2 and the nondimensional N^2_eff = (eps/Fr)^2
    # cancels it, so epot shares ekin's factor and etot is well
    # defined -- it only LOOKS scaling-dependent
    "epot": _ENERGY_FACTOR,
    "etot": _ENERGY_FACTOR,
    # NOT U/L despite sharing the unit 1/s: the nondimensional
    # definition multiplies through by eps (diagnostics.py)
    "linear_pot_vort": UnitFactor(
        target_unit="1/s", expr="U/(eps*L)", kind="derived",
        scales=("L", "U"), params={"eps": SCALING_NONLINEARITY},
        fn=_linear_pot_vort),
}

#: the stratification family's derived-constant row
STRATIFICATION_FACTORS: dict[str, UnitFactor] = {
    "N_dim": UnitFactor(
        target_unit="1/s", expr="U/(Fr_int*delta*L)", kind="constant",
        scales=("L", "U"),
        params={"Fr": STRATIFICATION_FROUDE, "delta": ASPECT_RATIO},
        fn=_frequency, dim_expr="sqrt(n2)",
        dim_params={"n2": STRATIFICATION_N2}, dim_fn=_frequency_dim),
}


#: the constant-stratification family's rows: ``N_dim`` plus the
#: derived ``b_total`` (``nh.diagnostics.b_total``), which converts
#: like the anomaly ``b`` itself
CONSTANT_STRATIFICATION_FACTORS: dict[str, UnitFactor] = {
    **STRATIFICATION_FACTORS,
    "b_total": UnitFactor(
        target_unit="m/s^2", expr="U^2/(eps*delta*L)", kind="derived",
        scales=("L", "U"),
        params={"eps": SCALING_NONLINEARITY, "delta": ASPECT_RATIO},
        fn=_buoyancy),
}


def coordinate_factors(
    coords: tuple[str, ...], vertical: str,
) -> dict[str, UnitFactor]:
    """
    Return the coordinate rows, keyed by the core's coords.

    Description
    -----------
    The horizontal coordinates scale with the reference length ``L``;
    the vertical coordinate scales with the vertical scale
    ``delta*L`` (the aspect ratio read live off
    ``nonhydro.aspect_ratio``). The keys follow the core's
    ``coords=`` / ``vertical=`` renaming.

    Parameters
    ----------
    coords : tuple[str, ...]
        The grid coordinate names (the core's ``coords=``).
    vertical : str
        The vertical coordinate name (the core's ``vertical=``).

    Returns
    -------
    dict[str, UnitFactor]
        One coordinate row per name.
    """
    horizontal = UnitFactor(target_unit="m", expr="L", kind="coordinate",
                            scales=("L",), fn=_length)
    height = UnitFactor(target_unit="m", expr="delta*L", kind="coordinate",
                        scales=("L",), params={"delta": ASPECT_RATIO},
                        fn=_height)
    return {name: (height if name == vertical else horizontal)
            for name in coords}
