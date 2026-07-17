r"""The HY-D6 matched-numerics comparison preset.

Description
-----------
``comparison_model(grid, dt, ...)`` assembles the **common-denominator
configuration** pinned by decision HY-D6 of the hydrostatic-model plan
(``design/plans/active/hydrostatic_model_plan.md`` §1, §6): the single
setup in which ``fridom.hydrostatic`` is meant to agree, in the shared
numerical limit, with pyOM2/3, Veros, and Oceananigans'
``HydrostaticFreeSurfaceModel``. Everything that can differ between the
reference models is fixed here to the one choice they can all be driven
to:

- **quasi-AB2 time stepping** — ``AdamBashforth(order=2, eps=0.1)``:
  pyOM's second-order Adams-Bashforth with its ``eps=0.1``
  computational-mode damper (``robert_filter`` / ``AB_eps``), which is
  also Oceananigans' ``QuasiAdamsBashforth2`` default.
- **backward-Euler linear free surface** —
  ``ImplicitFreeSurface(epsilon=1)``: the 2D Helmholtz surface-pressure
  solve. This matches Oceananigans' ``ImplicitFreeSurface`` (the FFT
  default on a regular grid) *exactly in formulation*, and pyOM's
  ``enable_free_surface`` backward-Euler option (``eps=1``). Setting
  ``epsilon=0`` switches to the **rigid lid** (the singular Poisson with
  the mean gauge) — the Veros streamfunction physics on a
  doubly-periodic domain, and pyOM's rigid-lid option ("identical to
  MITgcm").
- **centered second-order flux-form advection** —
  ``CenteredAdvection()`` for momentum *and* the buoyancy tracer: pyOM's
  and Veros' always-on centered-2 flux form, and Oceananigans'
  ``Centered(order=2)`` (its default momentum scheme is
  vector-invariant, so a matched run must *force* ``Centered(order=2)``
  — HY-D6 §6).
- **explicit f-plane Coriolis** — ``FPlaneCoriolis(f0=...)``.
- **linear buoyancy** — ``ConstantStratification(n2=...)``: the linear
  equation of state (``BuoyancyTracer`` / linearized-Vallis common
  ground) all three references reduce to.

Reference-model knob map (what to set on each side for a matched run):

============================  =======================================
this preset                   reference-model equivalent
============================  =======================================
``eps=0.1``                   pyOM ``AB_eps`` / Oceananigans
                              ``QuasiAdamsBashforth2`` (default 0.1)
``epsilon=1``                 Oceananigans ``ImplicitFreeSurface``
                              (FFT); pyOM ``enable_free_surface`` BE
``epsilon=0``                 Veros / pyOM rigid lid (streamfunction
                              on a doubly-periodic box)
``CenteredAdvection()``       pyOM/Veros centered-2 flux form;
                              Oceananigans ``Centered(order=2)``
``FPlaneCoriolis(f0)``        the f-plane Coriolis of all three
``ConstantStratification``    linear EOS / ``BuoyancyTracer``
============================  =======================================

Bit-level agreement is only expected in this shared limit; away from it
(higher-order advection, split-explicit surface, nonlinear EOS) the
models compare through convergence and physical diagnostics, not bytes
(HY-D6). The physics-validation suite that exercises this preset lives
in ``tests/hydrostatic/test_comparison.py``; a runnable baseline is
``examples/hydrostatic/comparison_baseline.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.hydrostatic.model import Model
from fridom.hydrostatic.modules.free_surface import ImplicitFreeSurface
from fridom.hydrostatic.modules.stratification import ConstantStratification
from fridom.model.modules.advection import CenteredAdvection
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.model import Model as _Model
    from fridom.spatial.grid import Grid


def comparison_model(
    grid: Grid,
    dt: float,
    *,
    csqr: float = 1.0,
    coriolis_f0: float = 1.0,
    n2: float = 1.0,
    rossby_number: float = 1.0,
    epsilon: float = 1.0,
    eps: float = 0.1,
    surface_advective_flux: bool = False,
    name: str | None = None,
    **kwargs: object,
) -> _Model:
    r"""Assemble the HY-D6 matched-numerics comparison preset.

    Description
    -----------
    The pinned common-denominator hydrostatic configuration (see the
    module docstring for the full reference-model map): quasi-AB2
    (``eps``), the backward-Euler implicit linear free surface
    (``epsilon``), centered-2 flux-form advection of momentum and the
    buoyancy tracer, explicit f-plane Coriolis, and linear
    stratification. The *numerical* configuration is fixed; only the
    physical parameters (``csqr``, ``coriolis_f0``, ``n2``,
    ``rossby_number``) and the two comparison knobs (``epsilon``,
    ``eps``) are exposed, so the comparison config cannot drift
    silently (the protocol-pin test in
    ``tests/hydrostatic/test_comparison.py`` asserts it exactly).

    Parameters
    ----------
    grid : Grid
        The grid: doubly-periodic horizontal, bounded (flat-bottom)
        vertical — the HY-D6 ``(P, P, bounded-z)`` box.
    dt : float
        The time step of the quasi-AB2 stepper.
    csqr : float, optional
        The squared barotropic phase speed :math:`c^2 = g H`, the single
        barotropic parameter, set explicitly per experiment
        (default: 1.0).
    coriolis_f0 : float, optional
        The f-plane Coriolis parameter :math:`f_0` (default: 1.0).
    n2 : float, optional
        The constant squared buoyancy frequency :math:`N^2`
        (default: 1.0).
    rossby_number : float, optional
        The Rossby number scaling the nonlinear advection term
        (default: 1.0).
    epsilon : float, optional
        The free-surface knob (static): ``1.0`` is the backward-Euler
        linear free surface (Oceananigans ``ImplicitFreeSurface`` / pyOM
        ``enable_free_surface``); ``0.0`` is the rigid lid (the Veros /
        pyOM streamfunction physics on a doubly-periodic domain). Must
        be ``>= 0`` (default: 1.0).
    eps : float, optional
        The pyOM quasi-AB2 computational-mode damper (Oceananigans'
        ``QuasiAdamsBashforth2`` default) (default: 0.1).
    surface_advective_flux : bool, optional
        Enable the constancy-preserving **surface closure** on the
        centered advection: advect **through** the top/bottom boundary
        faces with the one-sided face value instead of dropping the
        surface velocity ``w(0)`` (the Oceananigans-equivalent
        linear-free-surface treatment; see ``hy.Model``). Off is the
        HY-D6 fixed-domain closure that conserves tracer content to
        roundoff; on, tracer content is exchanged with the moving
        surface (default: False).
    name : str | None, optional
        Model name (default: None).
    **kwargs : object
        Forwarded to ``hy.Model`` (e.g. ``modules_extra``,
        ``chunk_size``).

    Returns
    -------
    fr.model.Model
        The assembled comparison model.
    """
    return Model(
        grid=grid,
        dt=dt,
        csqr=csqr,
        rossby_number=rossby_number,
        free_surface=ImplicitFreeSurface(epsilon=epsilon),
        coriolis=FPlaneCoriolis(f0=coriolis_f0),
        stratification=ConstantStratification(n2=n2),
        advection=CenteredAdvection(surface_flux=surface_advective_flux),
        time_stepper=AdamBashforth(dt, order=2, eps=eps),
        name=name,
        **kwargs,
    )
