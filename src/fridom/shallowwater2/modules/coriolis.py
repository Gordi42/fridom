"""
Coriolis modules: the f-plane and beta-plane rotation fields.

Description
-----------
Following the R2 rule (01_concepts D2.2): the Coriolis parameter is
*intrinsically spatial* — a constant on the f-plane, ``f(y)`` on the
beta-plane, ``f(x, y)`` under curvilinear coordinates — so it is
carried as an AUXILIARY field, the constant member on ``fr.Profile()``
(one degree of freedom, broadcast). Because the declared space is
static, f-plane and beta-plane are two module *types* (a factory
papers over the ergonomics). Each declares the ``f_coriolis`` field
that the core's rotation term reads by ``FieldReference``.

**Provides-implies-constancy** (02_rules): ``FPlaneCoriolis`` provides
``coriolis.f0`` (the whole truth), so ``sw.eigenmodes.from_model`` can
rely on its presence as a constancy check; ``BetaPlaneCoriolis`` does
**not** (its ``f`` is the ``f(y)`` field) and only publishes the
constant ``coriolis.beta``.

.. note::

    These modules are physically shared with the nonhydrostatic port
    (same ``f_coriolis`` field, same ``coriolis.f0``/``coriolis.beta``
    names). They are defined here for the wave-6 shallow-water port
    and **flagged for consolidation** into a shared framework module
    library (``fr.modules.FPlaneCoriolis`` / ``BetaPlaneCoriolis``,
    the spelling the notes fix).
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp

import fridom.framework2 as fr
from fridom.framework.utils import dtype_real, jaxify


@partial(jaxify, dynamic=("f0",))
class FPlaneCoriolis(fr.Module):

    r"""
    Constant-rotation Coriolis field on the f-plane.

    Description
    -----------
    Declares the AUXILIARY ``f_coriolis`` field on ``fr.Profile()``
    (one degree of freedom, ``f \equiv f_0`` everywhere) and provides
    the constant ``coriolis.f0``. Contributes no tendency term — the
    rotation force lives in the core's ``coriolis`` term, which reads
    this field.

    Parameters
    ----------
    f0 : float, optional
        The constant Coriolis parameter :math:`f_0` (default: 1.0).
    """

    def __init__(self, f0: float = 1.0) -> None:
        """Store the Coriolis parameter as a dynamic leaf."""
        self.f0 = jnp.asarray(f0, dtype=dtype_real())

    @property
    def field_declarations(self) -> tuple[fr.FieldDeclaration, ...]:
        """The one-DOF constant Coriolis field."""
        return (fr.FieldDeclaration(
            "f_coriolis", space=fr.Collocated(),
            lifecycle=fr.Lifecycle.AUXILIARY,
            default=FPlaneCoriolis._f_default,
            long_name="Coriolis parameter", units="1/s"),)

    parameter_declarations = (
        fr.ParameterDeclaration(
            fr.params.CORIOLIS_F0, attr="f0", units="1/s"),)

    def _f_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.grid.ScalarField:
        """Owner-method default: fill the field with ``f0``.

        Halo-synced so its ghost validity meets every term's stencil
        requirement (see the note in ``ShallowWaterCore._csqr_default``
        on carry-resident AUXILIARY fields and scan stability).
        """
        field = grid.create_field(
            space, data=jnp.full(space.shape, self.f0),
            name="f_coriolis")
        return grid.sync(field)


@partial(jaxify, dynamic=("f0", "beta"))
class BetaPlaneCoriolis(fr.Module):

    r"""
    Beta-plane Coriolis field :math:`f(y) = f_0 + \beta y`.

    Description
    -----------
    Declares the AUXILIARY ``f_coriolis`` field on
    ``fr.Profile("y")`` (varies in y, broadcast in x) and publishes
    the constant ``coriolis.beta``. It deliberately does **not**
    provide ``coriolis.f0`` — the Coriolis parameter is the ``f(y)``
    field, not a scalar, so an f0 provide would be a false constancy
    claim (02_rules).

    Parameters
    ----------
    f0 : float, optional
        The reference Coriolis parameter at ``y = 0`` (default: 1.0).
    beta : float, optional
        The meridional gradient :math:`\beta = \mathrm{d}f/\mathrm{d}y`
        (default: 0.0).
    """

    def __init__(self, f0: float = 1.0, beta: float = 0.0) -> None:
        """Store f0 and beta as dynamic leaves."""
        self.f0 = jnp.asarray(f0, dtype=dtype_real())
        self.beta = jnp.asarray(beta, dtype=dtype_real())

    @property
    def field_declarations(self) -> tuple[fr.FieldDeclaration, ...]:
        """The meridionally varying Coriolis field."""
        return (fr.FieldDeclaration(
            "f_coriolis", space=fr.Collocated(),
            lifecycle=fr.Lifecycle.AUXILIARY,
            default=BetaPlaneCoriolis._f_default,
            long_name="Coriolis parameter", units="1/s"),)

    parameter_declarations = (
        fr.ParameterDeclaration(
            fr.params.CORIOLIS_BETA, attr="beta", units="1/(m s)"),)

    def _f_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.grid.ScalarField:
        """Owner-method default: ``f(y) = f0 + beta * y`` (halo-synced).

        See ``ShallowWaterCore._csqr_default`` on why carry-resident
        AUXILIARY fields are pre-synced.
        """
        field = grid.create_field(
            space, init=lambda x, y: self.f0 + self.beta * y + 0.0 * x,
            name="f_coriolis")
        return grid.sync(field)
