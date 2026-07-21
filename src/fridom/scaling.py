r"""
Scaling policy objects (``fr.scaling``): the model's time frame.

Description
-----------
A scaling object names the **reference time scale** of a model run —
a pure normalization choice, owned by the user, passed as
``Model(scaling=...)`` (and through every preset). With
:math:`\varepsilon = T_\mathrm{ref} / T_\mathrm{adv}` every tendency
coefficient is a live ratio: nonlinear terms carry
:math:`\varepsilon`, rotation :math:`\varepsilon/\mathrm{Ro}`, each
wave mechanism :math:`(\varepsilon/\mathrm{Fr}_\mathrm{mech})^2` —
and the scaling choice only picks **which mechanism's number becomes**
:math:`\varepsilon`:

- :class:`Dimensional` — no nondimensionalization at all: modules
  take physical parameters (``gravity=``, ``f0=``, ``n2=``) and the
  traced step carries **zero** scaling operations.
- :class:`Advective` — advective time frame,
  :math:`T_\mathrm{ref} = T_\mathrm{adv}`, so
  :math:`\varepsilon = 1` (a constant assembly row).
- :class:`Rotational` — rotational frame,
  :math:`\varepsilon = \mathrm{Ro}` (the Coriolis module's
  ``rossby_number`` leaf).
- :class:`GravityWave` — (shallow-water) gravity-wave frame,
  :math:`\varepsilon = \mathrm{Fr}` (the shallow-water core's
  ``froude_number`` leaf).
- :class:`InternalWave` — internal-wave frame,
  :math:`\varepsilon = \mathrm{Fr}_\mathrm{int}` (the
  stratification module's ``froude_number`` leaf).
- :class:`ExternalWave` — external-(surface-)wave frame,
  :math:`\varepsilon = \mathrm{Fr}_\mathrm{ext}` (the free-surface
  module's ``froude_number`` leaf).

Under a nondimensional scaling the assembly injects one **alias
row**: the canonical ``scaling.nonlinearity`` name is bound onto the
designated mechanism module's own nonlinearity leaf (two names, one
leaf), so every consumer reads the same live value and parameter
sweeps/ramps stay zero-recompile. A dimensional assembly binds **no**
row — row presence is exactly the nondimensionality predicate.

The objects are **frozen host-side dataclasses** (never jaxified,
never traced): they carry policy, not values. The optional reference
scales ``L`` / ``U`` / ``g`` are stored for future conversion helpers
and are not consumed by the assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Scaling:

    r"""
    Base scaling policy (pick a concrete subclass).

    Description
    -----------
    Carries the two policy traits as class attributes —
    ``nondimensional`` (whether the assembly injects the
    ``scaling.nonlinearity`` row) and ``mechanism`` (which module
    family's nonlinearity number becomes :math:`\varepsilon`; None
    for :class:`Dimensional` and :class:`Advective`) — plus the
    stored-only reference scales.

    Parameters
    ----------
    L : float | None, optional
        Reference length scale [m]; stored only (default: None).
    U : float | None, optional
        Reference velocity scale [m/s]; stored only (default: None).
    g : float | None, optional
        Reference gravity [m/s^2]; stored only (default: None).
    """

    L: float | None = None
    U: float | None = None
    g: float | None = None

    #: whether the assembly injects the ``scaling.nonlinearity`` row
    nondimensional: ClassVar[bool] = False

    #: the mechanism whose nonlinearity number becomes epsilon
    mechanism: ClassVar[str | None] = None


@dataclass(frozen=True)
class Dimensional(Scaling):

    """Dimensional physics: no scaling row, zero scaling ops."""

    nondimensional: ClassVar[bool] = False
    mechanism: ClassVar[str | None] = None


@dataclass(frozen=True)
class Advective(Scaling):

    r"""Advective time frame: :math:`\varepsilon = 1` (constant)."""

    nondimensional: ClassVar[bool] = True
    mechanism: ClassVar[str | None] = None


@dataclass(frozen=True)
class Rotational(Scaling):

    r"""Rotational frame: :math:`\varepsilon = \mathrm{Ro}`."""

    nondimensional: ClassVar[bool] = True
    mechanism: ClassVar[str | None] = "rotation"


@dataclass(frozen=True)
class GravityWave(Scaling):

    r"""Gravity-wave frame: :math:`\varepsilon = \mathrm{Fr}` (sw)."""

    nondimensional: ClassVar[bool] = True
    mechanism: ClassVar[str | None] = "gravity_wave"


@dataclass(frozen=True)
class InternalWave(Scaling):

    r"""Internal-wave frame: :math:`\varepsilon = \mathrm{Fr_{int}}`."""

    nondimensional: ClassVar[bool] = True
    mechanism: ClassVar[str | None] = "internal_wave"


@dataclass(frozen=True)
class ExternalWave(Scaling):

    r"""External-wave frame: :math:`\varepsilon = \mathrm{Fr_{ext}}`."""

    nondimensional: ClassVar[bool] = True
    mechanism: ClassVar[str | None] = "external_wave"
