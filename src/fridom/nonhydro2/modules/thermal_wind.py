r"""Thermal-wind background: the mean-flow terms of a lateral front.

Description
-----------
``ThermalWindBackground`` carries the two linearized mean-flow
interaction terms of a **lateral** (front-in-x) thermal-wind-balanced
basic state on the f-plane. The basic state is

.. math::

    U = 0, \qquad V(z) = \frac{M^2}{f_0}\,(z - z_0), \qquad W = 0,

    B(x, z) = N^2 z + M^2 x ,

with :math:`M^2 = \partial_x B` the (constant) horizontal buoyancy
gradient. Geostrophy plus hydrostasy of the mean state give thermal
wind, :math:`f_0\,\partial_z V = \partial_x B = M^2`, so the single
number :math:`M^2` and the assembly's own :math:`f_0` fix the whole
state — the balance holds by construction, never by the caller
supplying two consistent numbers. It is the classical symmetric- /
baroclinic-instability setup: a constant lateral buoyancy gradient
over a constant stratification :math:`N^2` (the
``ConstantStratification`` restoring).

**The nonhydrostatic sibling of**
:class:`fridom.hydrostatic.modules.thermal_wind.ThermalWindBackground`,
rotated 90 degrees: there the mean flow is zonal (``U(z)``) over a
meridional front (``B(y)``) and the terms land on ``u`` and ``b``;
here the mean flow is meridional (``V(z)``) over a **zonal** front
(``B(x)``) and they land on ``v`` and ``b``. The sign of the buoyancy
term flips with the rotation (:math:`\partial_x B = +f_0\partial_z V`
against :math:`\partial_y B = -f_0\partial_z U`), which is exactly why
the two are separate, explicit modules rather than one axis-generic
one.

Signs (derived from the package conventions, not assumed)
---------------------------------------------------------
The core's Coriolis is :math:`\partial_t u = +f v`,
:math:`\partial_t v = -f u`; the buoyancy coupling is
:math:`\partial_t w = b/\delta^2`, :math:`\partial_t b = -N^2 w`.
Writing the total fields as mean plus perturbation, linearizing the
material derivative about the state above and dropping the primes
(using :math:`\partial_x U = \partial_z U = 0`,
:math:`\partial_x V = 0`) leaves, beyond the terms the core and the
stratification already carry:

.. math::

    \partial_t v \mathrel{+}= -\,w\,\partial_z V
                            = -\frac{M^2}{f_0}\,w , \\
    \partial_t b \mathrel{+}= -\,u\,\partial_x B = -M^2\,u .

The :math:`-N^2 w` half of the buoyancy advection is the
``ConstantStratification`` restoring and is **not** repeated here.

**No Doppler term, deliberately.** The mean flow is meridional and the
front is zonal, so the mean transport of a perturbation is
:math:`-V\,\partial_y q'`: it vanishes identically for the
y-independent (2-D x-z) problem this state is posed for — the
symmetric-instability geometry — and there is nothing to contribute.
A **y-resolved** run is the full baroclinic problem and does need it;
it is the shared advection's ``background=`` seam, not this module's,
and :meth:`background_velocity` returns the matching :math:`V(z)`
profile so the pairing stays thermal-wind-consistent:
``CenteredAdvection(background={"v": tw.background_velocity(f0)})``.

Energetics (what is, and is not, conserved)
-------------------------------------------
Under the model metric :math:`M = \mathrm{diag}(1, 1, \delta^2, 1/N^2)`
the two terms contribute to the perturbation energy rate

.. math::

    \frac{\mathrm{d}E}{\mathrm{d}t}\bigg|_{\rm tw}
      = -\frac{M^2}{f_0}\,\langle v, w\rangle
        - \frac{M^2}{N^2}\,\langle b, u\rangle ,

while every other assembled term is exactly M-skew. So **no
sign-definite quadratic form of the perturbation state is conserved**:
the balanced mean state is a genuine energy reservoir, which is the
instability. The honest gates are therefore the term identities, the
thermal-wind relation, and the growth rate — never a conserved norm.

The terms are declared ``linear=True`` (they carry no Rossby factor:
the mean-flow interactions are O(1), consistent with Coriolis and the
stratification restoring), so they are visible to
``fr.model.linearize`` and to the linear-operator consumers, and a
``advection=None`` model is the exact linear stability problem.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.time_dependent import resolve_at
from fridom.nonhydro2.params import CORIOLIS_F0, THERMAL_WIND_M2

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


@partial(jaxify, dynamic=("m2",))
class ThermalWindBackground(fr.model.Module):

    r"""The two mean-flow interaction terms of a lateral front.

    Contributes :math:`\partial_t v \mathrel{+}= -(M^2/f_0)\,w` (the
    mean-shear tilting) and :math:`\partial_t b \mathrel{+}= -M^2 u`
    (the horizontal buoyancy advection), reading the constant Coriolis
    parameter :math:`f_0` from the model (provided by
    ``nh.FPlaneCoriolis``) so the thermal wind
    :math:`f_0\,\partial_z V = \partial_x B = M^2` is enforced by
    construction. Pair with ``nh.ConstantStratification(n2=...)``,
    which supplies the remaining :math:`-N^2 w`.

    Parameters
    ----------
    m2 : float | fr.model.Ramp, optional
        The horizontal buoyancy gradient
        :math:`M^2 = \partial_x B` [1/s^2] of the thermal-wind state
        (default: 1.0); may be an ``fr.model.Ramp`` for a spun-up
        front. The implied mean flow is
        :math:`V(z) = M^2 (z - z_0)/f_0`, so the assembly's Coriolis
        parameter must be nonzero.
    reference_height : float, optional
        The height :math:`z_0` at which the background meridional
        velocity vanishes, :math:`V(z) = M^2 (z - z_0)/f_0`
        (default: 0.0). It does not enter the tendency terms at all
        (only the *gradients* of the mean state do); it sets the
        profile :meth:`background_velocity` returns, where mid-depth
        gives a depth-mean-zero mean flow.
    """

    def __init__(
        self,
        m2: float | fr.model.Ramp = 1.0,
        *,
        reference_height: float = 0.0,
    ) -> None:
        """Store the buoyancy-gradient leaf and the mean-state offset."""
        self.m2 = fr.model.leaf(m2)
        self._reference_height = float(reference_height)

    # ================================================================
    #  Declarations
    # ================================================================
    field_references = (
        fr.model.FieldReference(
            "u", hint="the horizontal buoyancy advection reads the "
                      "zonal velocity, declared by a nonhydrostatic "
                      "core (nh.Core)"),
        fr.model.FieldReference(
            "v", hint="the tilting term advances the meridional "
                      "velocity, declared by a nonhydrostatic core "
                      "(nh.Core)"),
        fr.model.FieldReference(
            "w", hint="the tilting term reads the vertical velocity, "
                      "declared by a nonhydrostatic core (nh.Core)"),
        fr.model.FieldReference(
            "b", hint="the horizontal buoyancy advection advances "
                      "buoyancy, declared by a buoyancy module "
                      "(nh.ConstantStratification / nh.BuoyancyTracer)"),
    )
    parameter_declarations = (
        fr.model.ParameterDeclaration(
            THERMAL_WIND_M2, attr="m2", units="1/s^2",
            doc="horizontal buoyancy gradient dB/dx of the "
                "thermal-wind state"),
    )
    parameter_references = (
        fr.model.ParameterReference(
            CORIOLIS_F0,
            hint="the thermal wind d_z V = M^2/f0 needs a constant "
                 "Coriolis parameter; add nh.FPlaneCoriolis(f0=...)"),
    )

    # ================================================================
    #  Convenience wiring: the implied mean state
    # ================================================================
    def background_velocity(self, f0: float) -> Callable[[float], float]:
        r"""Return the implied :math:`V(z) = M^2 (z - z_0)/f_0` profile.

        Description
        -----------
        The meridional mean flow in thermal-wind balance with this
        module's :math:`M^2`, at the Coriolis parameter the model is
        assembled with (a host-side helper, so :math:`f_0` is passed in
        rather than read from the assembly).

        A y-independent run does **not** need it: the mean transport
        :math:`-V\,\partial_y q'` vanishes identically there, which is
        why this module contributes no Doppler term. A y-resolved run
        does — pass it to the shared advection,
        ``CenteredAdvection(background={"v": tw.background_velocity(f0)})``,
        and the pairing is thermal-wind-consistent by construction. The
        gradient :math:`M^2` is resolved at ``t = 0`` (a ramped front
        ramps the *interaction* terms; the sampled background velocity
        is the ``t = 0`` snapshot — the advection ``background=`` takes
        a static callable).

        Parameters
        ----------
        f0 : float
            The constant Coriolis parameter of the assembly (nonzero).

        Returns
        -------
        Callable[[float], float]
            The ``lambda z: m2 * (z - reference_height) / f0`` profile.
        """
        shear = float(resolve_at(self.m2, 0.0)) / f0
        z_0 = self._reference_height

        def profile(z: float) -> float:
            return shear * (z - z_0)

        return profile

    def stratification_n2(
        self, f0: float, richardson_number: float,
    ) -> float:
        r"""Return the :math:`N^2` giving a balanced Richardson number.

        Description
        -----------
        The balanced Richardson number of a thermal-wind state is

        .. math::

            \mathrm{Ri} = \frac{N^2}{(\partial_z V)^2}
                        = \frac{N^2 f_0^2}{M^4} ,

        so a requested ``Ri`` fixes :math:`N^2 = \mathrm{Ri}\,M^4/f_0^2`
        — the spelling that keeps the caller's arithmetic out of the
        script:

        .. code-block:: python

            tw = nh.ThermalWindBackground(m2=-1e-7)
            buoyancy = nh.ConstantStratification(
                n2=tw.stratification_n2(f0, richardson_number=0.25))

        Symmetric instability of the state is the sign of its Ertel
        potential vorticity :math:`q = f_0 N^2 (1 - 1/\mathrm{Ri})`:
        unstable for ``Ri < 1``, stable for ``Ri > 1``.

        Parameters
        ----------
        f0 : float
            The constant Coriolis parameter of the assembly (nonzero).
        richardson_number : float
            The requested balanced Richardson number
            :math:`\mathrm{Ri} = N^2 f_0^2/M^4`.

        Returns
        -------
        float
            The matching squared buoyancy frequency
            :math:`N^2` [1/s^2].
        """
        m2 = float(resolve_at(self.m2, 0.0))
        return richardson_number * (m2 / f0) ** 2

    # ================================================================
    #  The mean-flow interaction terms (linear)
    # ================================================================
    @fr.model.term(advances=("v", "b"), linear=True,
                   linear_params=(THERMAL_WIND_M2, CORIOLIS_F0))
    def thermal_wind(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``dv/dt += -(m2/f0) w``; ``db/dt += -m2 u``.

        The tilting term interpolates the vertical velocity onto the
        ``v`` faces (``w.to(v)``) and the horizontal buoyancy advection
        interpolates the zonal velocity onto the ``b`` cells
        (``u.to(b)``). Both coefficients are static (the front's
        :math:`M^2` and the provided constant :math:`f_0`), so the term
        is exactly linear in the state; the shared :math:`M^2` and the
        single division by :math:`f_0` are the thermal-wind relation
        :math:`f_0 \partial_z V = \partial_x B`, so the two signs
        cannot drift apart.
        """
        m2 = ctx.params[THERMAL_WIND_M2]
        f0 = ctx.params[CORIOLIS_F0]
        u, v, w, b = state["u"], state["v"], state["w"], state["b"]
        return {
            "v": -((m2 / f0) * w.to(v)),
            "b": -(m2 * u.to(b)),
        }
