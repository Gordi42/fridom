r"""
The shallow-water dynamical core module.

Description
-----------
``DynamicalCore`` declares the state vocabulary (``u``, ``v``,
``p``), owns the squared phase speed :math:`c^2` (the AUXILIARY
``csqr`` field and, when constant, the ``shallowwater.csqr`` scalar)
and the Rossby scaling (``scaling.rossby``), and contributes the
single **linear** pressure-gradient / geopotential-divergence term:

.. math::
    \partial_t \boldsymbol{u} = - \nabla p , \qquad
    \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

**Variable depth**: ``csqr`` accepts a callable :math:`c^2(y)` (the
coriolis two-type precedent, folded into one core because the core
also owns the whole state vocabulary): the ``csqr`` field is then
declared on a meridional ``fr.spatial.Profile("y")`` and the constant
``shallowwater.csqr`` scalar is **not** provided
(provides-implies-constancy, 02_rules) — analytic consumers keyed on
the provide reject the model, the dense-column channel engine serves
it. The tendency terms are untouched either way: they read the
``csqr`` *field* (:math:`c^2` sits inside the divergence — the flux
form), which is exactly the sampling the variable-depth energy
metric ``diag(c^2, c^2, 1)`` pairs with. Pair a varying ``csqr``
with a Coriolis module carrying ``metric_weight="csqr"`` (the
thickness-weighted rotation) so the rotation stays energy-conserving
under that metric; the ``sw.Model`` preset wires this automatically.

The rotation :math:`f\,\underset{\neg}{\boldsymbol{u}}` is **not** a
core term: it is carried by the shared Coriolis module
(``fr.model.modules.FPlaneCoriolis`` / ``BetaPlaneCoriolis``), which declares
the ``f_coriolis`` field and the ``+f v`` / ``-f u`` coupling. The
gravity term here is unscaled (the Rossby number multiplies only the
advection, D2.2). The nonlinear Sadourny advection is a separate
module.
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.diagnostics import DIAGNOSTICS
from fridom.shallowwater2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


@partial(jaxify, dynamic=("csqr", "rossby_number"))
class DynamicalCore(fr.model.Module):

    r"""
    Shallow-water core: declares ``u``, ``v``, ``p``; linear physics.

    Parameters
    ----------
    csqr : float | Callable, optional
        The squared gravity-wave phase speed :math:`c^2`. A float is
        the constant depth: published as ``shallowwater.csqr`` and
        materialized into the one-DOF ``csqr`` field. A callable
        ``csqr(y)`` (evaluated on the meridional coordinate) is the
        variable depth: materialized into a ``csqr`` field on
        ``fr.spatial.Profile("y")``, with **no** ``shallowwater.csqr``
        provide (provides-implies-constancy) (default: 1.0).
    rossby_number : float | fr.model.Ramp, optional
        The Rossby number scaling the (separate) advection term;
        published as ``scaling.rossby`` (default: 1.0); may be a
        ``fr.model.Ramp`` for a spun-up nonlinearity.
    meridional : str, optional
        The meridional coordinate name a callable ``csqr`` varies
        along (default: ``"y"``).
    """

    #: The vocabulary class this core supplies (D1.3 commitment 4).
    state_type = State

    #: Bound parameterful diagnostics (the D1.3 commitment-4 channel).
    diagnostics = DIAGNOSTICS

    def __init__(
        self,
        csqr: float | Callable = 1.0,
        rossby_number: float | fr.model.Ramp = 1.0,
        *,
        meridional: str = "y",
    ) -> None:
        """Store the leaves; a callable ``csqr`` stays static."""
        self._csqr_fn = csqr if callable(csqr) else None
        self.csqr = None if callable(csqr) else fr.model.leaf(csqr)
        self.rossby_number = fr.model.leaf(rossby_number)
        self._meridional = meridional

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """U (east face), v (north face), p (centre), csqr (AUX)."""
        if self._csqr_fn is None:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_default,
                long_name="Squared phase speed", units="m^2/s^2")
        else:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_profile_default,
                long_name="Squared phase speed", units="m^2/s^2")
        return (
            fr.model.FieldDeclaration.velocity(
                "u", "x", space=fr.spatial.Staggered("x"),
                long_name="Velocity (x)", units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "v", "y", space=fr.spatial.Staggered("y"),
                long_name="Velocity (y)", units="m/s"),
            fr.model.FieldDeclaration(
                "p", space=fr.spatial.Collocated(),
                long_name="Pressure (g*eta)", units="m^2/s^2"),
            csqr_decl,
        )

    @property
    def parameter_declarations(
        self,
    ) -> tuple[fr.model.ParameterDeclaration, ...]:
        """Rossby always; ``shallowwater.csqr`` only when constant."""
        decls = (
            fr.model.ParameterDeclaration(
                fr.model.params.SCALING_ROSSBY, attr="rossby_number"),
        )
        if self._csqr_fn is None:
            decls += (
                fr.model.ParameterDeclaration(
                    sw_params.CSQR, attr="csqr", units="m^2/s^2"),
            )
        return decls

    def _csqr_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: fill the one-DOF profile with ``csqr``.

        The field is declared on ``fr.spatial.Profile()`` (constant depth is a
        single degree of freedom); the GAP-A ConstantSpace/Profile
        broadcast lifts it to the nodal join wherever a term multiplies
        it (``c.to(u) * u``). No ``grid.sync``
        pre-syncing: the GAP-B fix keeps carry-resident AUXILIARY
        fields scan-treedef-stable without pre-flooding their halos.
        """
        return grid.create_field(
            space, data=jnp.full(space.shape, self.csqr),
            name="csqr")

    def _csqr_profile_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: materialize the ``csqr(y)`` profile.

        The meridional profile carries a single non-constant
        coordinate, so ``init`` names exactly that coordinate; the
        signature is stamped dynamically to match ``self._meridional``
        (the ``BetaPlaneCoriolis._f_default`` precedent). No
        pre-syncing (GAP-B).
        """
        fn, mer = self._csqr_fn, self._meridional

        def init(**coords: object) -> object:
            return fn(coords[mer])

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return grid.create_field(space, init=init, name="csqr")

    # ================================================================
    #  Tendency terms (linear)
    # ================================================================
    @fr.model.term(advances=("u", "v", "p"), linear=True)
    def gravity(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Pressure gradient and geopotential divergence.

        .. math::
            \partial_t \boldsymbol{u} = - \nabla p , \qquad
            \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

        Pure field arithmetic: ``c^2`` sits INSIDE the divergence
        (``(c.to(u) * u).diff("x")``, the flux form) so the discrete
        stencil matches ``diff(c^2 u)``. The ``csqr`` field lifts from
        its one-DOF ``fr.spatial.Profile()`` onto each velocity face via the
        ConstantSpace broadcast in ``.to``.

        The pressure-gradient entries retag onto their velocities:
        nodal stencil outputs are BC-free, but on a walled grid each
        wall-normal velocity carries the derived Dirichlet wall tag
        on its own axis, so the entry adopts it (the nonhydro
        projection precedent) — identity on periodic grids. The flux
        entries need no retag: ``csqr.to(u)`` adopts the velocity's
        tag (BC-sibling adoption) and the divergence lands BC-free,
        which is ``p``'s space.
        """
        u, v, p = state["u"], state["v"], state["p"]
        csqr = state["csqr"]
        return {
            "u": (-p.diff("x")).retag(u),
            "v": (-p.diff("y")).retag(v),
            "p": -(csqr.to(u) * u).diff("x") - (csqr.to(v) * v).diff("y"),
        }
