r"""
Hydrostatic dimensional-factor tables (``model.units`` rows).

Description
-----------
The §D amplitude table of the hydrostatic model, contributed to
``model.units`` through the core, the stratification and the
free-surface family. Every factor follows the one rule
:math:`T_\mathrm{ref} = \varepsilon\,L/U`.

**The vertical convention.** The hydrostatic model carries no aspect
ratio :math:`\delta` (there is no vertical momentum equation to
normalize); its vertical scale is the **vertical mesh extent**
:math:`H` — the flat column's depth, the one sanctioned geometry
read of the free-surface family (``self._depth`` at bind) and the
energy re-key's analytic ``H_ref`` fold. The vertical rows form the
self-consistent scale family of hydrostatic balance, continuity and
the buoyancy restoring for that :math:`H` (the nonhydrostatic table
with :math:`\delta L \to H`):

- coordinates: horizontals :math:`L` [m], the vertical :math:`H`
  [m] (keyed by the core's ``horizontal=`` / ``vertical=`` names);
- velocities ``u`` / ``v``: :math:`U` [m/s]; the diagnosed ``w``:
  :math:`U H/L` [m/s];
- pressures ``p_hyd`` / ``ps``: :math:`U^2/\varepsilon` [m²/s²];
- buoyancy ``b``: :math:`U^2/(\varepsilon\,H)` [m/s²];
- derived constants :math:`N_\mathrm{dim} =
  U/(\mathrm{Fr}_\mathrm{int}\,H)` [1/s] (the stratification row;
  dimensional models report :math:`\sqrt{N^2}` from the bound
  ``stratification.n2``) and the external phase speed
  :math:`c_\mathrm{dim} = U/\mathrm{Fr}_\mathrm{ext}` [m/s] (the
  free-surface row; dimensional models report
  :math:`\sqrt{g\,H_\mathrm{ref}}` with :math:`H_\mathrm{ref}` the
  physical vertical extent — the flat-only reporting convention,
  matching the energy re-key).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.hydrostatic.params import FROUDE, GRAVITY
from fridom.model.params import (
    SCALING_NONLINEARITY,
    STRATIFICATION_FROUDE,
    STRATIFICATION_N2,
)
from fridom.model.units import UnitFactor

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


# ================================================================
#  Row resolvers (fn(values) over the resolved symbols)
# ================================================================
def _velocity(values: Mapping[str, float]) -> float:
    """Return the horizontal velocity amplitude ``U``."""
    return values["U"]


def _pressure(values: Mapping[str, float]) -> float:
    """Return the pressure amplitude ``U^2/eps``."""
    return values["U"] ** 2 / values["eps"]


def _length(values: Mapping[str, float]) -> float:
    """Return the horizontal coordinate scale ``L``."""
    return values["L"]


def _frequency_dim(values: Mapping[str, float]) -> float:
    """Return the bound frequency ``sqrt(n2)``."""
    return values["n2"] ** 0.5


# ================================================================
#  The geometry-free rows
# ================================================================
_VELOCITY_FACTOR = UnitFactor(
    unit="m/s", expr="U", kind="component", scales=("U",),
    fn=_velocity)

_PRESSURE_FACTOR = UnitFactor(
    unit="m^2/s^2", expr="U^2/eps", kind="component",
    scales=("U",), params={"eps": SCALING_NONLINEARITY},
    fn=_pressure)

#: the core's geometry-free component rows (u, v, p_hyd)
COMPONENT_FACTORS: dict[str, UnitFactor] = {
    "u": _VELOCITY_FACTOR,
    "v": _VELOCITY_FACTOR,
    "p_hyd": _PRESSURE_FACTOR,
}

#: the free-surface family's surface-pressure row
SURFACE_PRESSURE_FACTOR: UnitFactor = _PRESSURE_FACTOR


# ================================================================
#  The H-closed rows (the bind-captured vertical extent)
# ================================================================
def vertical_extent(grid: object, vertical: str) -> float:
    """
    Return the vertical mesh extent ``H`` (the flat-column depth).

    Description
    -----------
    The one sanctioned depth read (the free-surface family's
    ``self._depth`` / the energy re-key's ``H_ref``): the physical
    extent of the vertical mesh factor, read once at bind.

    Parameters
    ----------
    grid : Grid
        The bound grid (duck-typed: ``factors``).
    vertical : str
        The vertical coordinate name.

    Returns
    -------
    float
        The vertical extent ``H``.
    """
    for mesh in grid.factors:
        if vertical in mesh.names:
            lo, hi = mesh.extent
            return float(hi - lo)
    # a z-less grid fails at the core's w declaration first
    raise ValueError(  # pragma: no cover
        f"the vertical axis {vertical!r} is not a grid factor")


def coordinate_factors(
    horizontal: tuple[str, str], vertical: str, height: float,
) -> dict[str, UnitFactor]:
    """
    Return the coordinate rows, keyed by the core's names.

    Description
    -----------
    The horizontal coordinates scale with the reference length
    ``L``; the vertical coordinate scales with the vertical mesh
    extent ``H`` (the flat-only convention above). The keys follow
    the core's ``horizontal=`` / ``vertical=`` renaming.

    Parameters
    ----------
    horizontal : tuple[str, str]
        The (zonal, meridional) coordinate names.
    vertical : str
        The vertical coordinate name.
    height : float
        The bind-captured vertical extent ``H``.

    Returns
    -------
    dict[str, UnitFactor]
        One coordinate row per name.
    """
    def _height(values: Mapping[str, float]) -> float:  # noqa: ARG001
        """Return the vertical coordinate scale ``H``."""
        return height

    row = UnitFactor(unit="m", expr="L", kind="coordinate",
                     scales=("L",), fn=_length)
    return {
        horizontal[0]: row,
        horizontal[1]: row,
        vertical: UnitFactor(unit="m", expr="H", kind="coordinate",
                             fn=_height),
    }


def vertical_velocity_factor(height: float) -> UnitFactor:
    """
    Return the diagnosed-``w`` row ``U*H/L`` (continuity).

    Parameters
    ----------
    height : float
        The bind-captured vertical extent ``H``.

    Returns
    -------
    UnitFactor
        The ``w`` component row.
    """
    def _vertical_velocity(values: Mapping[str, float]) -> float:
        """Return the vertical velocity amplitude ``U*H/L``."""
        return values["U"] * height / values["L"]

    return UnitFactor(unit="m/s", expr="U*H/L", kind="component",
                      scales=("L", "U"), fn=_vertical_velocity)


def buoyancy_factor(height: float) -> UnitFactor:
    """
    Return the buoyancy row ``U^2/(eps*H)`` (hydrostatic balance).

    Parameters
    ----------
    height : float
        The bind-captured vertical extent ``H``.

    Returns
    -------
    UnitFactor
        The ``b`` component row.
    """
    def _buoyancy(values: Mapping[str, float]) -> float:
        """Return the buoyancy amplitude ``U^2/(eps*H)``."""
        return values["U"] ** 2 / (values["eps"] * height)

    return UnitFactor(unit="m/s^2", expr="U^2/(eps*H)",
                      kind="component", scales=("U",),
                      params={"eps": SCALING_NONLINEARITY},
                      fn=_buoyancy)


def stratification_factor(height: float) -> UnitFactor:
    """
    Return the derived-constant row ``N_dim = U/(Fr_int*H)``.

    Description
    -----------
    The stratification family's row: the internal Froude number
    ``Fr_int = U/(N H)`` inverted for ``N`` with ``H`` the vertical
    scale above; a dimensional model reports ``sqrt(n2)`` from the
    bound ``stratification.n2`` instead.

    Parameters
    ----------
    height : float
        The bind-captured vertical extent ``H``.

    Returns
    -------
    UnitFactor
        The ``N_dim`` constant row.
    """
    def _frequency(values: Mapping[str, float]) -> float:
        """Return the dimensional frequency ``U/(Fr_int*H)``."""
        return values["U"] / (values["Fr"] * height)

    return UnitFactor(unit="1/s", expr="U/(Fr_int*H)",
                      kind="constant", scales=("U",),
                      params={"Fr": STRATIFICATION_FROUDE},
                      fn=_frequency, dim_expr="sqrt(n2)",
                      dim_params={"n2": STRATIFICATION_N2},
                      dim_fn=_frequency_dim)


def phase_speed_factor(depth: float) -> UnitFactor:
    """
    Return the derived-constant row ``c_dim = U/Fr_ext``.

    Description
    -----------
    The free-surface family's external phase speed; a dimensional
    model reports ``sqrt(g*H_ref)`` from the bound
    ``hydrostatic.gravity`` and the flat column's physical depth
    (the flat-only reporting convention, matching the energy
    re-key).

    Parameters
    ----------
    depth : float
        The bind-captured vertical extent ``H_ref``.

    Returns
    -------
    UnitFactor
        The ``c_dim`` constant row.
    """
    def _phase_speed(values: Mapping[str, float]) -> float:
        """Return the dimensional phase speed ``U/Fr_ext``."""
        return values["U"] / values["Fr"]

    def _phase_speed_dim(values: Mapping[str, float]) -> float:
        """Return the bound phase speed ``sqrt(g*H_ref)``."""
        return (values["g"] * depth) ** 0.5

    return UnitFactor(unit="m/s", expr="U/Fr_ext", kind="constant",
                      scales=("U",), params={"Fr": FROUDE},
                      fn=_phase_speed, dim_expr="sqrt(g*H_ref)",
                      dim_params={"g": GRAVITY},
                      dim_fn=_phase_speed_dim)
