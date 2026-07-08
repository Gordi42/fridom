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

written as **pure field arithmetic**: ``f`` and the velocities are
interpolated to the target staggered faces with ``.to`` and multiplied
as fields (``f_at_u * v_at_u``). Because ``f_coriolis`` is a *field*
(not a traced ``ctx.params`` scalar), the term needs no ``.with_data``
raw-scalar bypass and declares no ``extra_halo`` — the GAP-A
ConstantSpace/Profile broadcast makes the ``Profile -> nodal`` lift
trace cleanly and the ``.to`` interpolations are halo-traced normally.
The term is 2-D (``u``, ``v`` only) and identical for both the 3-D
nonhydrostatic and 2-D shallow-water cores; it never references ``w``
or the pressure/geopotential.

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

from fridom.framework.utils import dtype_real, jaxify
from fridom.framework2.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.framework2.model.linear_blocks import (
    Coeff,
    Interp,
    LinearBlock,
    apply_linear_blocks,
)
from fridom.framework2.model.module import Module
from fridom.framework2.model.parameters import ParameterDeclaration
from fridom.framework2.model.params import CORIOLIS_BETA, CORIOLIS_F0
from fridom.framework2.model.space_patterns import Profile
from fridom.framework2.model.terms import term

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.model.context import StepContext

_U_HINT = ("velocities are declared by a dynamical-core module, "
           "e.g. nh.DynamicalCore or sw.DynamicalCore")


# The linear rotation blocks, shared verbatim by the f-plane and
# beta-plane terms (the only difference between them is the *space* of
# ``f_coriolis``, not the coupling). The coefficient's runtime source
# is the constant AUX field ``f_coriolis`` (interpolated onto the
# target face and multiplied as fields); its symbolic constant is the
# provided ``CORIOLIS_F0`` (a beta-plane f(y) provides no f0, so
# ``fr.linear_blocks`` declines it — provides-implies-constancy).
_CORIOLIS_BLOCKS = (
    LinearBlock("u", "v", Interp(),
                Coeff(aux="f_coriolis", const=CORIOLIS_F0)),
    LinearBlock("v", "u", Interp(),
                Coeff(aux="f_coriolis", const=CORIOLIS_F0, sign=-1)),
)


def _coriolis_tendency(state: object, ctx: object) -> dict[str, ScalarField]:
    r"""``{u: f v, v: -f u}`` derived from the shared rotation blocks.

    Description
    -----------
    The numeric consumer of :data:`_CORIOLIS_BLOCKS` (single source of
    truth): ``increment[out] += coeff . op(state[src])`` interpolates
    the Coriolis field ``f`` and the velocities to the opposite face
    and multiplies as fields — bit-identical to the pre-block closure.
    """
    return apply_linear_blocks(_CORIOLIS_BLOCKS, state, ctx)


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
    """

    def __init__(self, f0: float = 1.0) -> None:
        """Store the Coriolis parameter as a dynamic leaf."""
        self.f0 = jnp.asarray(f0, dtype=dtype_real())

    field_references = (
        FieldReference("u", hint=_U_HINT),
        FieldReference("v", hint=_U_HINT),
    )
    parameter_declarations = (
        ParameterDeclaration(CORIOLIS_F0, attr="f0", units="1/s",
                             doc="constant Coriolis parameter"),
    )

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The one-DOF constant Coriolis field (``fr.Profile()``)."""
        return (
            FieldDeclaration(
                "f_coriolis", space=Profile(),
                lifecycle=Lifecycle.AUXILIARY,
                default=FPlaneCoriolis._f_default,
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

    @term(advances=("u", "v"), linear=True, blocks=_CORIOLIS_BLOCKS)
    def coriolis(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        r"""``\partial_t u = f v``; ``\partial_t v = -f u``."""
        return _coriolis_tendency(state, ctx)


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
    """

    def __init__(
        self, f0: float = 1.0, beta: float = 0.0,
        *, meridional: str = "y",
    ) -> None:
        """Store the leaves and the meridional coordinate name."""
        self.f0 = jnp.asarray(f0, dtype=dtype_real())
        self.beta = jnp.asarray(beta, dtype=dtype_real())
        self._meridional = meridional

    field_references = (
        FieldReference("u", hint=_U_HINT),
        FieldReference("v", hint=_U_HINT),
    )
    parameter_declarations = (
        ParameterDeclaration(CORIOLIS_BETA, attr="beta",
                             units="1/(m s)",
                             doc="meridional Coriolis gradient"),
    )

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The ``f(y)`` field on a meridional profile."""
        return (
            FieldDeclaration(
                "f_coriolis", space=Profile(self._meridional),
                lifecycle=Lifecycle.AUXILIARY,
                default=BetaPlaneCoriolis._f_default,
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

    @term(advances=("u", "v"), linear=True, blocks=_CORIOLIS_BLOCKS)
    def coriolis(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        r"""``\partial_t u = f(y) v``; ``\partial_t v = -f(y) u``."""
        return _coriolis_tendency(state, ctx)
