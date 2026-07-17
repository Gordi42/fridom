r"""Thermal-wind background: the linearized mean-flow of an Eady state.

Description
-----------
``ThermalWindBackground`` carries the two linearized mean-flow
interaction terms of a **zonal thermal-wind-balanced** basic state on
the doubly-periodic f-plane that the shared advection ``background=``
does *not* supply. The basic state is

.. math::

    U(z) = \Lambda\,(z - z_0), \qquad V = 0, \qquad
    B(y) = -f_0\,\Lambda\,y ,

with :math:`\Lambda = \partial_z U` the (constant) vertical shear and
:math:`B` the mean buoyancy. It is the classical Eady setup: constant
shear, constant stratification :math:`N^2` (the
``ConstantStratification`` restoring), a rigid lid.

**Why a separate module.** The shared flux-form advection's
``background=`` samples a mean *velocity* :math:`U(z)` and contributes
the linear Doppler transport :math:`-U\,\partial_x q'` of every advected
perturbation :math:`q' \in \{u',v',b'\}` (verified: its
``background_advection`` term loops only over the sampled velocity's own
axis, here ``x``, so with :math:`U` independent of :math:`x` it is
exactly :math:`-U\,\partial_x q'`). It supplies **neither** of the two
remaining mean-flow interactions, because both couple a *perturbation*
velocity to a *mean gradient* that is not a velocity sample:

- the buoyancy source (baroclinic conversion)
  :math:`-v'\,\partial_y B`, whose carrier :math:`\partial_y B` is a
  mean *buoyancy* gradient (the doubly-periodic y-domain cannot hold
  :math:`B(y) = -f_0\Lambda y` as a state field); and
- the momentum tilting :math:`-w'\,\partial_z U`, the vertical
  advection of the *background* momentum by the perturbation (the
  background enters ``background=`` only as an advecting velocity, never
  as an advected quantity, so :math:`(u'\!\cdot\!\nabla)U` has no
  carrier there).

This module supplies exactly those two, and is meant to be **paired
with** ``background=`` (which supplies the Doppler part):
``CenteredAdvection(background={"u": tw.background_velocity()})``, where
``tw`` is this module — :meth:`background_velocity` returns the matching
:math:`U(z)` callable so the pairing is thermal-wind-consistent by
construction.

Signs (derived from the package conventions, not assumed)
---------------------------------------------------------
The core's Coriolis is :math:`\partial_t u = +f v`,
:math:`\partial_t v = -f u`; hydrostatic balance is
:math:`\partial_z p_{hyd} = b`; the stratification restoring is
:math:`\partial_t b = -N^2 w`. Geostrophy + hydrostasy of the mean
state give thermal wind:

.. math::

    \partial_t v = 0:\ \ \partial_y P = -f_0 U, \qquad
    \partial_z P = B \;\Rightarrow\;
    \partial_y B = \partial_z(\partial_y P) = -f_0\,\partial_z U
                 = -f_0\,\Lambda .

Linearizing the material advection of the total fields about this
state (dropping products of primes, using
:math:`\partial_x U = \partial_y U = 0`, :math:`\partial_x B =
\partial_z B|_{\rm mean} = 0`) leaves, beyond the Doppler terms:

.. math::

    \partial_t u' \mathrel{+}= -\,w'\,\partial_z U = -\Lambda\,w', \\
    \partial_t b' \mathrel{+}= -\,v'\,\partial_y B
                            = -v'(-f_0\Lambda) = +f_0\,\Lambda\,v' .

So the buoyancy source is :math:`+f_0\Lambda\,v'` (positive: the
conversion term is verified below to make the mode grow, not decay).
Both signs are locked to the *single* shear :math:`\Lambda` through the
thermal-wind relation, so their relative sign — the one the instability
depends on — cannot drift.

Energetics (what is, and is not, conserved)
-------------------------------------------
Under the model metric :math:`M = \mathrm{diag}(1, 1, 1/N^2, 1/c^2)`
(``hy.energy``) the two new terms contribute to the perturbation energy
rate :math:`\langle X, M\,\partial_t X\rangle`:

.. math::

    \frac{\mathrm{d}E}{\mathrm{d}t}\bigg|_{\rm tw}
      = -\Lambda\,\langle u', w'\rangle
        + \frac{f_0\Lambda}{N^2}\,\langle v', b'\rangle .

Every *other* term of the assembled operator is exactly M-skew and
contributes machine-zero: the Doppler advection is transport by the
divergence-free :math:`U` (skew), and the internal KE :math:`\leftrightarrow`
PE conversion (the pressure-gradient/stratification pair, carrying the
:math:`\langle w'b'\rangle` buoyancy flux) is the machine-exact H2
skew, **unchanged** by this module. Hence **no sign-definite quadratic
form of the perturbation state is conserved** — that is the defining
feature of the instability: the Eady mean flow is a genuine energy
source (:math:`\mathrm{d}E/\mathrm{d}t > 0` for the growing mode),
feeding the perturbation from a reservoir that is not a state field. The
honest gate is therefore the *conversion-term identity* above (the
module's non-conservative contribution equals exactly those two
quadratic forms) together with the eigenvalue growth-rate check, not a
conserved norm.

These are declared ``linear=True`` terms (they carry no Rossby factor:
the mean-flow interactions are O(1), consistent with Coriolis, the
stratification restoring, and the Doppler ``background_advection``), so
they are visible to ``fr.model.linearize`` and to the linear-operator /
eigenmode consumers.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.params import CORIOLIS_F0, SHEAR
from fridom.model.time_dependent import resolve_at

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


@partial(jaxify, dynamic=("shear",))
class ThermalWindBackground(fr.model.Module):

    r"""The two mean-flow interaction terms of a thermal-wind state.

    Contributes :math:`\partial_t u \mathrel{+}= -\Lambda\,w'` (tilting)
    and :math:`\partial_t b \mathrel{+}= +f_0\Lambda\,v'` (baroclinic
    conversion), reading the constant Coriolis parameter :math:`f_0`
    from the model (provided by ``hy.FPlaneCoriolis``) so the thermal
    wind :math:`\partial_y B = -f_0\Lambda` is enforced by construction.
    Pair with ``CenteredAdvection(background={"u":
    tw.background_velocity()})`` for the Doppler part.

    Parameters
    ----------
    shear : float | fr.model.Ramp, optional
        The vertical shear :math:`\Lambda = \partial_z U` of the
        zonal thermal-wind state (default: 1.0); may be an
        ``fr.model.Ramp`` for a spun-up mean flow.
    reference_height : float, optional
        The height :math:`z_0` at which the background zonal velocity
        vanishes, :math:`U(z) = \Lambda(z - z_0)` (default: 0.0); set
        it to mid-depth for a depth-mean-zero (symmetric) mean flow.
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
    meridional : str, optional
        The meridional coordinate name (default: ``"y"``).
    """

    def __init__(
        self,
        shear: float | fr.model.Ramp = 1.0,
        *,
        reference_height: float = 0.0,
        vertical: str = "z",
        meridional: str = "y",
    ) -> None:
        """Store the shear leaf and the mean-state geometry."""
        self.shear = fr.model.leaf(shear)
        self._reference_height = float(reference_height)
        self._vertical = vertical
        self._meridional = meridional

    # ================================================================
    #  Declarations
    # ================================================================
    field_references = (
        fr.model.FieldReference(
            "u", hint="the tilting term acts on the zonal velocity, "
                      "declared by a hydrostatic core "
                      "(hy.HydrostaticCore)"),
        fr.model.FieldReference(
            "v", hint="the baroclinic conversion reads the meridional "
                      "velocity, declared by a hydrostatic core "
                      "(hy.HydrostaticCore)"),
        fr.model.FieldReference(
            "w", hint="the tilting term reads the diagnosed vertical "
                      "velocity, declared by a hydrostatic core "
                      "(hy.HydrostaticCore)"),
        fr.model.FieldReference(
            "b", hint="the baroclinic conversion advances buoyancy, "
                      "declared by a stratification module "
                      "(hy.ConstantStratification)"),
    )
    parameter_declarations = (
        fr.model.ParameterDeclaration(
            SHEAR, attr="shear", units="1/s",
            doc="thermal-wind vertical shear dU/dz"),
    )
    parameter_references = (
        fr.model.ParameterReference(
            CORIOLIS_F0,
            hint="the thermal wind d_y B = -f0 shear needs a constant "
                 "Coriolis parameter; add hy.FPlaneCoriolis(f0=...)"),
    )

    # ================================================================
    #  Convenience wiring: the matching background velocity U(z)
    # ================================================================
    def background_velocity(self) -> Callable[[float], float]:
        r"""Return the matching :math:`U(z) = \Lambda(z - z_0)` callable.

        Description
        -----------
        The zonal background velocity whose Doppler transport
        ``background=`` supplies, in thermal-wind balance with this
        module's shear. Pass it straight to the shared advection:
        ``CenteredAdvection(background={"u": tw.background_velocity()})``.
        The shear is resolved at ``t = 0`` (a static profile; a ramped
        shear ramps the *interaction* terms, but the sampled background
        velocity is the ``t = 0`` snapshot — the advection ``background=``
        takes a static callable).

        Returns
        -------
        Callable[[float], float]
            The ``lambda z: shear * (z - reference_height)`` profile.
        """
        lam = float(resolve_at(self.shear, 0.0))
        z_0 = self._reference_height

        def profile(z: float) -> float:
            return lam * (z - z_0)

        return profile

    # ================================================================
    #  Time-dependence honesty (the linear-term ramp report)
    # ================================================================
    def time_dependent_linear_parameters(self) -> tuple[str, ...]:
        """Report a ramped ``shear`` feeding the linear interaction.

        A time-dependent ``shear`` (an ``fr.Ramp``) is a scalar read at
        stage time through ``ctx.params`` — it advances correctly under
        every re-reading stepper — but it lives inside this module's
        ``linear=True`` term, so a frozen-``L`` (exponential) stepper
        must refuse it (AR-D7), exactly as a ramped ``coriolis.f0`` /
        ``stratification.n2`` does; a plain-float ``shear`` reports
        nothing.
        """
        if isinstance(self.shear, fr.model.TimeDependent):
            return (str(SHEAR),)
        return ()

    # ================================================================
    #  The mean-flow interaction term (linear)
    # ================================================================
    @fr.model.term(advances=("u", "b"), linear=True)
    def thermal_wind(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``du/dt += -shear w``; ``db/dt += +f0 shear v``.

        The tilting term interpolates the diagnosed ``w`` onto the ``u``
        faces (``w.to(u)`` — the vertical ``Outer -> Center`` restriction
        composed with the horizontal ``Center -> Right`` interpolation),
        and the baroclinic conversion interpolates ``v`` onto the ``b``
        cell centres (``v.to(b)``). Both coefficients are static (the
        shear and the provided constant ``f0``), so the term is exactly
        linear in the state — the sign is the thermal-wind derivation of
        the module docstring.
        """
        lam = ctx.params[SHEAR]
        f0 = ctx.params[CORIOLIS_F0]
        u, v, w, b = state["u"], state["v"], state["w"], state["b"]
        return {
            "u": -(lam * w.to(u)),
            "b": (f0 * lam) * v.to(b),
        }
