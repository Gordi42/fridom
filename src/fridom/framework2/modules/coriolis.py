r"""
Shared Coriolis modules: the f-plane and beta-plane rotation.

Description
-----------
The framework's reusable Coriolis module library (D2.1 module-library
sharing): both the nonhydrostatic and shallow-water ports consume
``fr.modules.FPlaneCoriolis`` / ``fr.modules.BetaPlaneCoriolis`` — one
clean, field-based implementation instead of a per-package copy.

Following R2 (01_concepts D2.2), the Coriolis parameter is
*intrinsically spatial* — a constant on the f-plane, :math:`f(y)` on
the beta-plane — so it is carried as an AUXILIARY field ``f_coriolis``
on a **one-DOF** ``fr.Profile()`` (f-plane, constant everywhere) or a
meridional ``fr.Profile("y")`` (beta-plane, varying in y, broadcast in
x/z). Because the declared space is static, the f-plane and beta-plane
are two module *types*.

Each module also **carries** the linear rotation term

.. math::
    \partial_t u = f\,v , \qquad \partial_t v = -f\,u

written as **pure field arithmetic** in the energy-conserving
staggered form: ``f`` is sampled at the ``v`` faces and the ``u``
equation averages the flux (``(f_at_v * v).to(u)``), which keeps the
rotation exactly M-skew-adjoint for any ``f`` profile (see the term
docstring). Because ``f_coriolis`` is a *field*
(not a traced ``ctx.params`` scalar), the term needs no ``.with_data``
raw-scalar bypass and declares no ``extra_halo`` — the GAP-A
ConstantSpace/Profile broadcast makes the ``Profile -> nodal`` lift
trace cleanly and the ``.to`` interpolations are halo-traced normally.
The term is 2-D (``u``, ``v`` only) and identical for both the 3-D
nonhydrostatic and 2-D shallow-water cores; it never references ``w``
or the pressure/geopotential.

**Weighted velocity metrics** (``metric_weight``): when a model's
energy metric weights the velocities by a spatially varying field —
the variable-depth shallow water, whose conserved energy is
:math:`\tfrac12\int c^2(u^2 + v^2) + p^2` with :math:`c^2(y)` the
``csqr`` field — the unweighted staggered rotation is no longer
exactly M-skew: the ``u`` and ``v`` nodes sit at different ``y``
positions, so the weight cannot cancel across the interpolation.
Passing ``metric_weight="csqr"`` switches the term to the
**thickness-weighted flux form** (the linearized Sadourny/Arakawa
energy-conserving Coriolis)

.. math::
    \partial_t u = f\,\bar v , \qquad
    \partial_t v = -\overline{w\,f\,u}\,/\,w ,

which is exactly M-skew under ``diag(w, w, ...)`` for **any** ``f``
and any strictly positive weight profile, and reduces to the
unweighted form for a constant weight (up to float rounding). The
default (``None``) keeps the v1 form bit-for-bit.

**Provides-implies-constancy** (02_rules): ``FPlaneCoriolis`` provides
the constant ``coriolis.f0`` (analytic consumers such as
``eigenmodes.from_model`` rely on the provide as the constancy check);
``BetaPlaneCoriolis`` provides only ``coriolis.beta`` and must **not**
provide ``coriolis.f0`` (its ``f`` is the ``f(y)`` field, not a
scalar — an f0 provide would be a false constancy claim).
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import jaxify
from fridom.framework2.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.framework2.model.module import Module
from fridom.framework2.model.parameters import ParameterDeclaration, leaf
from fridom.framework2.model.params import CORIOLIS_BETA, CORIOLIS_F0
from fridom.framework2.model.space_patterns import Profile
from fridom.framework2.model.terms import term

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.fields.scalar_field import ScalarField

_U_HINT = ("velocities are declared by a dynamical-core module, "
           "e.g. nh.DynamicalCore or sw.DynamicalCore")


@term(advances=("u", "v"), linear=True, name="coriolis")
def _coriolis(self, state, ctx) -> dict:  # noqa: ANN001, ARG001
    r"""``du/dt = f v``; ``dv/dt = -f u`` as pure field arithmetic.

    The **energy-conserving** staggered form of the v1 framework:
    ``f`` is sampled once, at the ``u`` faces, and the ``v`` equation
    averages the *flux* ``(f u)`` back to the ``v`` faces. Because
    the ``.to`` interpolations between the ``u`` and ``v`` spaces are
    measure-weighted adjoints of each other, the pair
    ``u += f_u (v.to(u))``, ``v -= (f_u u).to(v)`` is exactly
    M-skew-adjoint for **any** ``f`` profile (rotation does no
    work); sampling ``f`` per target face instead would break
    discrete energy conservation for a varying ``f``. For a constant
    ``f`` the two forms coincide bit-for-bit (``.to`` is linear).
    Shared verbatim by the f-plane (constant ``f``) and beta-plane
    (``f(y)``) module types — the only difference between them is
    the *space* of ``f_coriolis``, not the coupling.

    With a ``metric_weight`` field ``w`` (a varying velocity energy
    weight, e.g. the variable-depth ``csqr``) the interpolated flux
    is thickness-weighted, ``v -= (w_u f_u u).to(v) / w_v``: under
    ``diag(w, w, ...)`` the M-adjoint of ``u += f_u (v.to(u))`` is
    exactly that expression (the weight enters the flux at the
    ``u`` nodes and cancels at the ``v`` nodes), so the pair stays
    M-skew for any ``f`` and any positive ``w`` profile — the
    linearized Sadourny/Arakawa pairing. For a constant ``w`` it
    coincides with the unweighted form mathematically (``.to`` is
    linear); in floating point the two round differently by up to
    1 ulp (bitwise-identical only for power-of-two ``w``), so the
    weighted form is the safe default whenever a weight field
    exists.
    """
    u, v, f = state["u"], state["v"], state["f_coriolis"]
    f_u = f.to(u)
    if self._metric_weight is None:
        return {
            "u": f_u * v.to(u),
            "v": -((f_u * u).to(v)),
        }
    w = state[self._metric_weight]
    return {
        "u": f_u * v.to(u),
        "v": -((w.to(u) * f_u * u).to(v)) / w.to(v),
    }


_WEIGHT_HINT = ("the velocity energy-metric weight field (e.g. the "
                "shallow-water csqr, declared by its dynamical core)")


def _rotation_references(
    metric_weight: str | None,
) -> tuple[FieldReference, ...]:
    """Build the u/v references, plus the weight field if named."""
    refs = (
        FieldReference("u", hint=_U_HINT),
        FieldReference("v", hint=_U_HINT),
    )
    if metric_weight is not None:
        refs += (FieldReference(metric_weight, hint=_WEIGHT_HINT),)
    return refs


@partial(jaxify, dynamic=("f0",))
class FPlaneCoriolis(Module):

    r"""
    Constant-rotation Coriolis on the f-plane; provides ``coriolis.f0``.

    Description
    -----------
    Declares the AUXILIARY ``f_coriolis`` field on ``fr.Profile()``
    (one degree of freedom, :math:`f \equiv f_0` everywhere) and
    carries the linear rotation term. Provides the constant
    ``coriolis.f0`` (provides-implies-constancy).

    Parameters
    ----------
    f0 : float, optional
        The constant Coriolis parameter :math:`f_0` (default: 1.0).
    metric_weight : str | None, optional
        Name of a state field weighting the velocity energy metric
        (e.g. the variable-depth shallow-water ``"csqr"``); switches
        the rotation to the thickness-weighted flux form, the
        M-skew pairing under ``diag(w, w, ...)`` (default: None).
    """

    def __init__(
        self, f0: float = 1.0, *, metric_weight: str | None = None,
    ) -> None:
        """Store the Coriolis parameter as a dynamic leaf."""
        self.f0 = leaf(f0)
        self._metric_weight = metric_weight

    parameter_declarations = (
        ParameterDeclaration(CORIOLIS_F0, attr="f0", units="1/s",
                             doc="constant Coriolis parameter"),
    )

    @property
    def metric_weight(self) -> str | None:
        """The velocity energy-metric weight field name (or None)."""
        return self._metric_weight

    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """u/v, plus the metric-weight field when configured."""
        return _rotation_references(self._metric_weight)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The one-DOF constant Coriolis field (``fr.Profile()``)."""
        return (
            FieldDeclaration(
                "f_coriolis", space=Profile(),
                lifecycle=Lifecycle.AUXILIARY,
                default=self._f_default,
                long_name="Coriolis parameter", units="1/s"),
        )

    def _f_default(self, grid: object, space: object) -> ScalarField:
        """Owner-method default: fill the profile with ``f0``.

        No ``grid.sync`` pre-syncing: the GAP-B fix records the
        consumption-side exchange in an external identity cache, so a
        carry-resident AUXILIARY field keeps a stable scan treedef
        without being pre-synced to full halo.
        """
        return grid.create_field(
            space, data=jnp.full(space.shape, self.f0),
            name="f_coriolis")

    #: ``du/dt = f v``; ``dv/dt = -f u`` (shared rotation term).
    coriolis = _coriolis


@partial(jaxify, dynamic=("f0", "beta"))
class BetaPlaneCoriolis(Module):

    r"""
    Beta-plane Coriolis :math:`f(y) = f_0 + \beta y`.

    Description
    -----------
    Declares the AUXILIARY ``f_coriolis`` field on
    ``fr.Profile("y")`` (varies in the meridional coordinate,
    broadcast elsewhere) and carries the linear rotation term.
    Provides ``coriolis.beta`` and deliberately does **not** provide
    ``coriolis.f0`` (its Coriolis parameter is the ``f(y)`` field, not
    a constant — 02_rules).

    Parameters
    ----------
    f0 : float, optional
        Reference Coriolis parameter at ``y = 0`` (default: 1.0).
    beta : float, optional
        Meridional gradient :math:`\beta = \mathrm{d}f/\mathrm{d}y`
        (default: 0.0).
    meridional : str, optional
        The meridional coordinate name (default: ``"y"``).
    metric_weight : str | None, optional
        Name of a state field weighting the velocity energy metric
        (e.g. the variable-depth shallow-water ``"csqr"``); switches
        the rotation to the thickness-weighted flux form, the
        M-skew pairing under ``diag(w, w, ...)`` (default: None).
    """

    def __init__(
        self, f0: float = 1.0, beta: float = 0.0,
        *, meridional: str = "y", metric_weight: str | None = None,
    ) -> None:
        """Store the leaves and the meridional coordinate name."""
        self.f0 = leaf(f0)
        self.beta = leaf(beta)
        self._meridional = meridional
        self._metric_weight = metric_weight

    parameter_declarations = (
        ParameterDeclaration(CORIOLIS_BETA, attr="beta",
                             units="1/(m s)",
                             doc="meridional Coriolis gradient"),
    )

    @property
    def metric_weight(self) -> str | None:
        """The velocity energy-metric weight field name (or None)."""
        return self._metric_weight

    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """u/v, plus the metric-weight field when configured."""
        return _rotation_references(self._metric_weight)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The ``f(y)`` field on a meridional profile."""
        return (
            FieldDeclaration(
                "f_coriolis", space=Profile(self._meridional),
                lifecycle=Lifecycle.AUXILIARY,
                default=self._f_default,
                long_name="Coriolis parameter", units="1/s"),
        )

    def _f_default(self, grid: object, space: object) -> ScalarField:
        """Owner-method default: materialize ``f0 + beta*y``.

        The meridional profile carries a single non-constant
        coordinate, so ``init`` names exactly that coordinate; the
        signature is stamped dynamically to match ``self._meridional``.
        No pre-syncing (GAP-B) — see ``FPlaneCoriolis._f_default``.
        """
        f0, beta, mer = self.f0, self.beta, self._meridional

        def init(**coords: object) -> object:
            return f0 + beta * coords[mer]

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return grid.create_field(space, init=init, name="f_coriolis")

    #: ``du/dt = f(y) v``; ``dv/dt = -f(y) u`` (shared rotation term).
    coriolis = _coriolis
