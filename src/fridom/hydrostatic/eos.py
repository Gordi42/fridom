r"""Equations of state for seawater: density from ``T``, ``S`` and depth.

Description
-----------
The small swappable objects ``hy.TemperatureSalinity(eos=...)`` takes.
An equation of state (EOS) maps the conservative temperature
:math:`\Theta` [degC], the absolute salinity :math:`S_A` [g/kg] and the
geopotential **depth** :math:`d \ge 0` [m] (positive downwards, zero at
the resting surface) onto the in-situ density
:math:`\rho(\Theta, S_A, d)` [kg/m^3]. Three are shipped:

- :class:`LinearEOS` —
  :math:`\rho = \rho_0\,[1 - \alpha(\Theta - \Theta_0)
  + \beta(S_A - S_0)]`, depth independent. Its two coefficients are
  **tunable**: the buoyancy module binds them as the live parameters
  ``eos.alpha`` / ``eos.beta`` (sweepable, differentiable).
- :class:`RoquetEOS` — the simplified second-order polynomial family
  of Roquet et al. (2015b), with cabbeling and thermobaricity.
- :class:`TEOS10EOS` — the 55-term polynomial fit of the TEOS-10
  standard of Roquet et al. (2015a), the ``polyTEOS10`` expression
  NEMO and Oceananigans ship.

Every EOS is a pure ``jax.numpy`` function of its arguments: it takes
floats or arrays (any broadcastable shapes), is jit- and
autodiff-safe, and is usable on its own, outside any model:

.. code-block:: python

    eos = hy.TEOS10EOS()
    eos.density(10.0, 30.0, 1000.0)      # 1027.45140 kg/m^3

**The dynamic density anomaly.** A Boussinesq model only ever needs
horizontal differences of density at a fixed height, so the part of
:math:`\rho` that depends on depth alone (the bulk compressibility,
several kg/m^3 per kilometre) is dynamically inert — and numerically
harmful on a terrain-following grid, where it feeds the sigma
pressure-gradient error. :meth:`EquationOfState.density_anomaly`
therefore measures the density against a **reference parcel at the
same depth**,

.. math::

    \rho'(\Theta, S_A, d) = \rho(\Theta, S_A, d)
        - \rho(\Theta_\mathrm{ref}, S_\mathrm{ref}, d) ,

which removes every pure-depth contribution identically (the NEMO
``r0(z)`` split, done EOS-agnostically) while keeping the thermobaric
dependence of the *differences* exact. The buoyancy the hydrostatic
pressure integrates is :math:`b = -g\,\rho'/\rho_0`.

References
----------
- Roquet, F., Madec, G., McDougall, T. J., Barker, P. M. (2015a):
  Accurate polynomial expressions for the density and specific volume
  of seawater using the TEOS-10 standard. *Ocean Modelling* 90,
  29-43, doi:10.1016/j.ocemod.2015.04.002.
- Roquet, F., Madec, G., Brodeau, L., Nycander, J. (2015b): Defining a
  simplified yet "realistic" equation of state for seawater.
  *J. Phys. Oceanogr.* 45, 2564-2579, doi:10.1175/JPO-D-15-0080.1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence


def _horner(coefficients: Sequence[object], x: object) -> object:
    """Evaluate a polynomial (lowest order first) by Horner's rule."""
    acc = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        acc = acc * x + coefficient
    return acc


# ================================================================
#  The interface
# ================================================================
class EquationOfState(ABC):

    r"""
    Interface of an equation of state :math:`\rho(\Theta, S_A, d)`.

    Description
    -----------
    A host-side, immutable configuration object with pure
    ``jax.numpy`` evaluation methods. Subclasses implement
    :meth:`density` and :attr:`uses_depth`; the dynamic anomaly, the
    buoyancy and the expansion coefficients follow from them.

    ``tunable`` names the coefficients the buoyancy module binds as
    **live parameters** (dynamic leaves ``eos.<name>``): the module
    passes their current values back through the ``coefficients=``
    mapping of every evaluation method, so a sweep or a gradient
    reaches the EOS without re-assembly. The vocabulary is closed —
    ``"alpha"`` and ``"beta"`` — and only a depth-independent EOS may
    be tunable.

    Parameters
    ----------
    rho0 : float
        The Boussinesq reference density :math:`\rho_0` [kg/m^3].
    reference : tuple[float, float]
        The reference parcel :math:`(\Theta_\mathrm{ref},
        S_\mathrm{ref})` [degC, g/kg] the dynamic anomaly is measured
        against.
    """

    def __init__(
        self, *, rho0: float, reference: tuple[float, float],
    ) -> None:
        """Store the reference density and the reference parcel."""
        rho0 = float(rho0)
        if not rho0 > 0.0:
            raise ValueError(
                f"the reference density rho0 must be positive "
                f"[kg/m^3], got {rho0!r}")
        self._rho0: float = rho0
        temperature, salinity = reference
        self._reference: tuple[float, float] = (
            float(temperature), float(salinity))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def rho0(self) -> float:
        """The Boussinesq reference density [kg/m^3]."""
        return self._rho0

    @property
    def reference(self) -> tuple[float, float]:
        """The reference parcel ``(T_ref, S_ref)`` [degC, g/kg]."""
        return self._reference

    @property
    @abstractmethod
    def uses_depth(self) -> bool:
        """Whether the density depends on depth (thermobaricity)."""

    @property
    def tunable(self) -> dict[str, float]:
        """The tunable coefficients by name (default: none)."""
        return {}

    # ================================================================
    #  Value semantics (an EOS is static data of a jit-compiled module)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Equal configurations compare equal (one jit-cache entry)."""
        return type(other) is type(self) and repr(other) == repr(self)

    def __hash__(self) -> int:
        """Hash of the configuration (consistent with ``__eq__``)."""
        return hash((type(self).__name__, repr(self)))

    # ================================================================
    #  Evaluation
    # ================================================================
    @abstractmethod
    def density(
        self,
        temperature: object,
        salinity: object,
        depth: object = 0.0,
        coefficients: Mapping[str, object] | None = None,
    ) -> object:
        r"""
        In-situ density :math:`\rho(\Theta, S_A, d)` [kg/m^3].

        Parameters
        ----------
        temperature : object
            Conservative temperature [degC] (float or array).
        salinity : object
            Absolute salinity [g/kg] (float or array).
        depth : object, optional
            Geopotential depth, positive downwards [m] (default: 0.0).
        coefficients : Mapping[str, object] | None, optional
            Live values of the :attr:`tunable` coefficients; ``None``
            uses the constructor values (default: None).

        Returns
        -------
        object
            The in-situ density [kg/m^3].
        """

    def density_anomaly(
        self,
        temperature: object,
        salinity: object,
        depth: object = 0.0,
        coefficients: Mapping[str, object] | None = None,
    ) -> object:
        r"""
        Dynamic density anomaly against the reference parcel [kg/m^3].

        Description
        -----------
        :math:`\rho' = \rho(\Theta, S_A, d) -
        \rho(\Theta_\mathrm{ref}, S_\mathrm{ref}, d)`: the density
        relative to the reference parcel **at the same depth**, free of
        every pure-depth contribution (module docstring).

        Parameters
        ----------
        temperature : object
            Conservative temperature [degC].
        salinity : object
            Absolute salinity [g/kg].
        depth : object, optional
            Geopotential depth, positive downwards [m] (default: 0.0).
        coefficients : Mapping[str, object] | None, optional
            Live tunable coefficients (default: None).

        Returns
        -------
        object
            The dynamic density anomaly [kg/m^3].
        """
        t_ref, s_ref = self._reference
        return (self.density(temperature, salinity, depth, coefficients)
                - self.density(t_ref, s_ref, depth, coefficients))

    def buoyancy(
        self,
        temperature: object,
        salinity: object,
        depth: object = 0.0,
        *,
        gravity: object,
        coefficients: Mapping[str, object] | None = None,
    ) -> object:
        r"""
        Buoyancy :math:`b = -g\,\rho'/\rho_0` [m/s^2].

        Parameters
        ----------
        temperature : object
            Conservative temperature [degC].
        salinity : object
            Absolute salinity [g/kg].
        depth : object, optional
            Geopotential depth, positive downwards [m] (default: 0.0).
        gravity : object
            The gravitational acceleration [m/s^2].
        coefficients : Mapping[str, object] | None, optional
            Live tunable coefficients (default: None).

        Returns
        -------
        object
            The buoyancy of the dynamic density anomaly [m/s^2].
        """
        anomaly = self.density_anomaly(
            temperature, salinity, depth, coefficients)
        return -(gravity / self._rho0) * anomaly

    def thermal_expansion(
        self, temperature: object, salinity: object,
        depth: object = 0.0,
    ) -> object:
        r"""
        Thermal expansion :math:`\alpha = -\rho_0^{-1}\,
        \partial\rho/\partial\Theta` [1/K] (Boussinesq form).

        Parameters
        ----------
        temperature : object
            Conservative temperature [degC].
        salinity : object
            Absolute salinity [g/kg].
        depth : object, optional
            Geopotential depth, positive downwards [m] (default: 0.0).

        Returns
        -------
        object
            The thermal expansion coefficient [1/K].
        """
        slope = jnp.vectorize(jax.grad(self.density, argnums=0))
        return -slope(*_as_real(temperature, salinity, depth)) / self._rho0

    def haline_contraction(
        self, temperature: object, salinity: object,
        depth: object = 0.0,
    ) -> object:
        r"""
        Haline contraction :math:`\beta = \rho_0^{-1}\,
        \partial\rho/\partial S_A` [kg/g] (Boussinesq form).

        Parameters
        ----------
        temperature : object
            Conservative temperature [degC].
        salinity : object
            Absolute salinity [g/kg].
        depth : object, optional
            Geopotential depth, positive downwards [m] (default: 0.0).

        Returns
        -------
        object
            The haline contraction coefficient [kg/g].
        """
        slope = jnp.vectorize(jax.grad(self.density, argnums=1))
        return slope(*_as_real(temperature, salinity, depth)) / self._rho0


def _as_real(*values: object) -> tuple[object, ...]:
    """Coerce the arguments to real arrays (``jax.grad`` needs floats)."""
    return tuple(jnp.asarray(value, dtype=float) for value in values)


# ================================================================
#  Linear
# ================================================================
class LinearEOS(EquationOfState):

    r"""
    The linear equation of state (depth independent, tunable).

    Description
    -----------
    .. math::

        \rho = \rho_0\,[\,1 - \alpha\,(\Theta - \Theta_0)
               + \beta\,(S_A - S_0)\,] ,
        \qquad
        b = g\,[\,\alpha\,(\Theta - \Theta_0) - \beta\,(S_A - S_0)\,] .

    The reference parcel is :math:`(\Theta_0, S_0)`, so the dynamic
    anomaly is the bracket itself and is formed **without** the
    :math:`\rho_0` offset (no cancellation: with a constant salinity
    the buoyancy is an exact affine image of the temperature, the
    ``hy.BuoyancyTracer`` equivalence gate). ``alpha`` and ``beta``
    are tunable: ``hy.TemperatureSalinity`` binds them as the live
    parameters ``eos.alpha`` / ``eos.beta``.

    The defaults are the leading coefficients of the simplified
    Roquet et al. (2015b) equation of state at
    :math:`(10\,^\circ\mathrm{C}, 35\,\mathrm{g/kg})` as NEMO ships
    them (``a0 = 1.6550e-1``, ``b0 = 7.6554e-1`` kg/m^3 per unit,
    :math:`\rho_0 = 1026`): a reasonable global-mean pair.

    Parameters
    ----------
    alpha : float, optional
        Thermal expansion coefficient [1/K] (default: 1.6131e-4).
    beta : float, optional
        Haline contraction coefficient [kg/g] (default: 7.4614e-4).
    temperature0 : float, optional
        Reference temperature :math:`\Theta_0` [degC] (default: 10.0).
    salinity0 : float, optional
        Reference salinity :math:`S_0` [g/kg] (default: 35.0).
    rho0 : float, optional
        Boussinesq reference density [kg/m^3] (default: 1026.0).
    """

    def __init__(
        self,
        alpha: float = 1.6131e-4,
        beta: float = 7.4614e-4,
        *,
        temperature0: float = 10.0,
        salinity0: float = 35.0,
        rho0: float = 1026.0,
    ) -> None:
        """Store the two coefficients and the reference state."""
        super().__init__(rho0=rho0,
                         reference=(temperature0, salinity0))
        self._alpha: float = float(alpha)
        self._beta: float = float(beta)

    @property
    def uses_depth(self) -> bool:
        """``False``: the linear density carries no depth dependence."""
        return False

    @property
    def tunable(self) -> dict[str, float]:
        """``alpha`` [1/K] and ``beta`` [kg/g]."""
        return {"alpha": self._alpha, "beta": self._beta}

    def density_anomaly(
        self,
        temperature: object,
        salinity: object,
        depth: object = 0.0,  # noqa: ARG002 — depth independent
        coefficients: Mapping[str, object] | None = None,
    ) -> object:
        r"""``rho0 * (-alpha (T - T0) + beta (S - S0))``, offset free."""
        live = self.tunable if coefficients is None else coefficients
        t_ref, s_ref = self._reference
        return self._rho0 * (live["beta"] * (salinity - s_ref)
                             - live["alpha"] * (temperature - t_ref))

    def density(
        self,
        temperature: object,
        salinity: object,
        depth: object = 0.0,
        coefficients: Mapping[str, object] | None = None,
    ) -> object:
        """In-situ density ``rho0 + density_anomaly`` [kg/m^3]."""
        return self._rho0 + self.density_anomaly(
            temperature, salinity, depth, coefficients)

    def __repr__(self) -> str:
        """Render the coefficients and the reference state."""
        t_ref, s_ref = self._reference
        return (f"LinearEOS(alpha={self._alpha!r}, beta={self._beta!r}, "
                f"temperature0={t_ref!r}, salinity0={s_ref!r}, "
                f"rho0={self._rho0!r})")


# ================================================================
#  Roquet et al. (2015b): the simplified second-order family
# ================================================================
#: Table 3 of Roquet et al. (2015b): the optimized coefficient sets of
#: rho' = R100 S + R010 T + R020 T^2 - R011 T Z + R200 S^2 - R101 S Z
#:        + R110 S T  (absolute T [degC], S [g/kg]; Z = -depth [m]).
ROQUET_COEFFICIENTS: dict[str, dict[str, float]] = {
    "linear": {
        "R010": -1.775e-1, "R100": 7.718e-1},
    "cabbeling": {
        "R010": -0.844e-1, "R100": 7.718e-1, "R020": -4.561e-3},
    "cabbeling_thermobaricity": {
        "R010": -0.651e-1, "R100": 7.718e-1, "R020": -5.027e-3,
        "R011": -2.5681e-5},
    "freezing": {
        "R010": -0.491e-1, "R100": 7.718e-1, "R020": -5.027e-3,
        "R011": -2.5681e-5},
    "second_order": {
        "R010": 0.182e-1, "R100": 8.078e-1, "R020": -4.937e-3,
        "R011": -2.4677e-5, "R200": -1.115e-4, "R101": -8.241e-6,
        "R110": -2.446e-3},
}

_ROQUET_NAMES = ("R100", "R010", "R020", "R011", "R200", "R101", "R110")


class RoquetEOS(EquationOfState):

    r"""
    The simplified second-order polynomial EOS of Roquet et al. (2015b).

    Description
    -----------
    .. math::

        \rho = \rho_\mathrm{r} + R_{100} S_A + R_{010}\Theta
             + R_{020}\Theta^2 - R_{011}\Theta Z + R_{200} S_A^2
             - R_{101} S_A Z + R_{110} S_A \Theta ,
        \qquad Z = -d ,

    in the absolute :math:`\Theta` [degC] and :math:`S_A` [g/kg] and
    the height :math:`Z \le 0` [m]: the smallest family that carries
    **cabbeling** (:math:`R_{020}, R_{200}, R_{110}`) and
    **thermobaricity** (:math:`R_{011}, R_{101}`). The five optimized
    coefficient sets of the paper's Table 3
    (:data:`ROQUET_COEFFICIENTS`) are selected by name, as in
    Oceananigans' ``RoquetSeawaterPolynomial``; at
    :math:`(10\,^\circ\mathrm{C}, 35\,\mathrm{g/kg}, 0)` the
    ``"second_order"`` set has :math:`-\partial_\Theta\rho = 0.1662`
    and :math:`\partial_S\rho = 0.7755` kg/m^3 per unit, the ``a0`` /
    ``b0`` of the NEMO "S-EOS" spelling of the same fit.

    The fit constrains density *differences* only, so the constant
    :math:`\rho_\mathrm{r}` is a convention: it is anchored such that
    :math:`\rho(\Theta_\mathrm{ref}, S_\mathrm{ref}, 0) = \rho_0`. The
    family has no bulk compressibility term; its in-situ density is a
    dynamic quantity, not a measurement of :math:`\rho(p)`.

    A pure polynomial: no divide, no root, no masked singularity.

    Parameters
    ----------
    coefficient_set : str | Mapping[str, float], optional
        A Table 3 set name — ``"linear"``, ``"cabbeling"``,
        ``"cabbeling_thermobaricity"``, ``"freezing"``,
        ``"second_order"`` — or a mapping of ``R...`` coefficients
        (absent entries are zero) (default: ``"second_order"``).
    rho0 : float, optional
        Boussinesq reference density [kg/m^3]; the paper's fits use
        1024.6 (default: 1024.6).
    reference : tuple[float, float], optional
        The reference parcel of the dynamic anomaly [degC, g/kg]
        (default: (10.0, 35.0)).
    """

    def __init__(
        self,
        coefficient_set: str | Mapping[str, float] = "second_order",
        *,
        rho0: float = 1024.6,
        reference: tuple[float, float] = (10.0, 35.0),
    ) -> None:
        """Resolve the coefficient set (taught error on a bad name)."""
        super().__init__(rho0=rho0, reference=reference)
        if isinstance(coefficient_set, str):
            if coefficient_set not in ROQUET_COEFFICIENTS:
                raise ValueError(
                    f"unknown RoquetEOS coefficient set "
                    f"{coefficient_set!r}; the Table 3 sets are "
                    f"{sorted(ROQUET_COEFFICIENTS)} (or pass a mapping "
                    f"of {_ROQUET_NAMES} coefficients)")
            self._set_name: str = coefficient_set
            given: Mapping[str, float] = ROQUET_COEFFICIENTS[
                coefficient_set]
        else:
            self._set_name = "custom"
            given = dict(coefficient_set)
            unknown = sorted(set(given) - set(_ROQUET_NAMES))
            if unknown:
                raise ValueError(
                    f"unknown RoquetEOS coefficient(s) {unknown}; the "
                    f"polynomial carries {_ROQUET_NAMES}")
        self._r: dict[str, float] = {
            name: float(given.get(name, 0.0)) for name in _ROQUET_NAMES}

    @property
    def coefficients(self) -> dict[str, float]:
        """The seven polynomial coefficients by name."""
        return dict(self._r)

    @property
    def uses_depth(self) -> bool:
        """Whether a thermobaric / halibaric coefficient is nonzero."""
        return self._r["R011"] != 0.0 or self._r["R101"] != 0.0

    def _polynomial(
        self, temperature: object, salinity: object, depth: object,
    ) -> object:
        """Evaluate the polynomial part (``Z = -depth``)."""
        r = self._r
        # -R011 T Z - R101 S Z with Z = -depth
        return (
            (r["R100"] + r["R200"] * salinity + r["R110"] * temperature
             + r["R101"] * depth) * salinity
            + (r["R010"] + r["R020"] * temperature
               + r["R011"] * depth) * temperature)

    def density(
        self,
        temperature: object,
        salinity: object,
        depth: object = 0.0,
        coefficients: Mapping[str, object] | None = None,  # noqa: ARG002
    ) -> object:
        r"""In-situ density, anchored at ``rho(T_ref, S_ref, 0) = rho0``."""
        t_ref, s_ref = self._reference
        return (self._rho0
                + self._polynomial(temperature, salinity, depth)
                - self._polynomial(t_ref, s_ref, 0.0))

    def __repr__(self) -> str:
        """Render the set name and the reference state."""
        shown = (self._set_name if self._set_name != "custom"
                 else {k: v for k, v in self._r.items() if v != 0.0})
        return (f"RoquetEOS({shown!r}, rho0={self._rho0!r}, "
                f"reference={self._reference!r})")


# ================================================================
#  Roquet et al. (2015a): the 55-term TEOS-10 polynomial
# ================================================================
#: reference scales of the polynomial's reduced variables
_TEOS10_SALINITY_UNIT = 40.0 * 35.16504 / 35.0   # S_Au [g/kg]
_TEOS10_TEMPERATURE_UNIT = 40.0                  # Theta_u [degC]
_TEOS10_DEPTH_UNIT = 1.0e4                       # Z_u [m]
_TEOS10_SALINITY_SHIFT = 32.0                    # delta_S [g/kg]

#: the vertical reference profile r0(zeta) (orders 1..6 in zeta)
_TEOS10_R0 = (
    4.6494977072e+01, -5.2099962525e+00, 2.2601900708e-01,
    6.4326772569e-02, 1.5616995503e-02, -1.7243708991e-03)

#: the density fit r'(s, tau, zeta): key "ijk" is the coefficient of
#: s^i tau^j zeta^k (Roquet et al. 2015a, appendix A / polyTEOS10.py)
_TEOS10_R = {
    "000": 8.0189615746e+02, "100": 8.6672408165e+02,
    "200": -1.7864682637e+03, "300": 2.0375295546e+03,
    "400": -1.2849161071e+03, "500": 4.3227585684e+02,
    "600": -6.0579916612e+01, "010": 2.6010145068e+01,
    "110": -6.5281885265e+01, "210": 8.1770425108e+01,
    "310": -5.6888046321e+01, "410": 1.7681814114e+01,
    "510": -1.9193502195e+00, "020": -3.7074170417e+01,
    "120": 6.1548258127e+01, "220": -6.0362551501e+01,
    "320": 2.9130021253e+01, "420": -5.4723692739e+00,
    "030": 2.1661789529e+01, "130": -3.3449108469e+01,
    "230": 1.9717078466e+01, "330": -3.1742946532e+00,
    "040": -8.3627885467e+00, "140": 1.1311538584e+01,
    "240": -5.3563304045e+00, "050": 5.4048723791e-01,
    "150": 4.8169980163e-01, "060": -1.9083568888e-01,
    "001": 1.9681925209e+01, "101": -4.2549998214e+01,
    "201": 5.0774768218e+01, "301": -3.0938076334e+01,
    "401": 6.6051753097e+00, "011": -1.3336301113e+01,
    "111": -4.4870114575e+00, "211": 5.0042598061e+00,
    "311": -6.5399043664e-01, "021": 6.7080479603e+00,
    "121": 3.5063081279e+00, "221": -1.8795372996e+00,
    "031": -2.4649669534e+00, "131": -5.5077101279e-01,
    "041": 5.5927935970e-01, "002": 2.0660924175e+00,
    "102": -4.9527603989e+00, "202": 2.5019633244e+00,
    "012": 2.0564311499e+00, "112": -2.1311365518e-01,
    "022": -1.2419983026e+00, "003": -2.3342758797e-02,
    "103": -1.8507636718e-02, "013": 3.7969820455e-01,
}


def _teos10_table() -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Nest the fit as ``table[k][j][i]`` for the triple Horner rule."""
    table = []
    for k in range(4):
        rows = []
        for j in range(7):
            row = [_TEOS10_R.get(f"{i}{j}{k}", 0.0) for i in range(7)]
            while row and row[-1] == 0.0:
                row.pop()
            rows.append(tuple(row))
        while rows and not rows[-1]:
            rows.pop()
        table.append(tuple(rows))
    return tuple(table)


_TEOS10_TABLE = _teos10_table()


class TEOS10EOS(EquationOfState):

    r"""
    The 55-term polynomial TEOS-10 EOS of Roquet et al. (2015a).

    Description
    -----------
    The Boussinesq ``polyTEOS10`` density,

    .. math::

        \rho(\Theta, S_A, d) = r_0(\zeta)
            + \sum_{ijk} R_{ijk}\, s^i \tau^j \zeta^k ,
        \quad
        s = \sqrt{\frac{S_A + 32}{S_{Au}}},\;
        \tau = \frac{\Theta}{40},\;
        \zeta = \frac{d}{10^4} ,

    a 52-term fit plus the 6-term vertical reference profile
    :math:`r_0`, accurate to the TEOS-10 standard within the
    uncertainty of the underlying measurements over the oceanographic
    funnel. It is the nonlinear EOS of NEMO (``ln_teos10``) and
    Oceananigans (``TEOS10EquationOfState``), evaluated here by a
    triple Horner rule. ``density(10, 30, 1000)`` reproduces the
    paper's check value ``1027.45140`` kg/m^3 (``r0 = 4.59763035``,
    ``r' = 1022.85377``).

    The root is taken of :math:`S_A + 32`, positive for every
    salinity a model can hold (a dry cell's ``S = 0`` included), so
    the reverse-mode derivative has no masked singularity.

    Parameters
    ----------
    rho0 : float, optional
        Boussinesq reference density [kg/m^3]; the paper fits with
        1020 (default: 1020.0).
    reference : tuple[float, float], optional
        The reference parcel of the dynamic anomaly [degC, g/kg]
        (default: (10.0, 35.0)).
    """

    def __init__(
        self,
        *,
        rho0: float = 1020.0,
        reference: tuple[float, float] = (10.0, 35.0),
    ) -> None:
        """Store the reference density and the reference parcel."""
        super().__init__(rho0=rho0, reference=reference)

    @property
    def uses_depth(self) -> bool:
        """``True``: the fit is pressure (depth) dependent."""
        return True

    @staticmethod
    def reference_profile(depth: object) -> object:
        r"""The vertical reference profile :math:`r_0(\zeta)` [kg/m^3]."""
        zeta = depth / _TEOS10_DEPTH_UNIT
        return _horner(_TEOS10_R0, zeta) * zeta

    @staticmethod
    def fit(temperature: object, salinity: object, depth: object) -> object:
        r"""The 52-term fit :math:`r'(s, \tau, \zeta)` [kg/m^3]."""
        tau = temperature / _TEOS10_TEMPERATURE_UNIT
        root = jnp.sqrt((salinity + _TEOS10_SALINITY_SHIFT)
                        / _TEOS10_SALINITY_UNIT)
        zeta = depth / _TEOS10_DEPTH_UNIT
        return _horner(
            [_horner([_horner(row, root) for row in rows], tau)
             for rows in _TEOS10_TABLE],
            zeta)

    def density(
        self,
        temperature: object,
        salinity: object,
        depth: object = 0.0,
        coefficients: Mapping[str, object] | None = None,  # noqa: ARG002
    ) -> object:
        r"""In-situ density :math:`r_0(\zeta) + r'(s, \tau, \zeta)`."""
        return (self.reference_profile(depth)
                + self.fit(temperature, salinity, depth))

    def density_anomaly(
        self,
        temperature: object,
        salinity: object,
        depth: object = 0.0,
        coefficients: Mapping[str, object] | None = None,  # noqa: ARG002
    ) -> object:
        r"""``r'(T, S, d) - r'(T_ref, S_ref, d)``: ``r_0`` never formed."""
        t_ref, s_ref = self._reference
        return (self.fit(temperature, salinity, depth)
                - self.fit(t_ref, s_ref, depth))

    def __repr__(self) -> str:
        """Render the reference density and parcel."""
        return (f"TEOS10EOS(rho0={self._rho0!r}, "
                f"reference={self._reference!r})")
