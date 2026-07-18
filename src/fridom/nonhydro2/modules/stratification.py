r"""Stratification modules: buoyancy and the linear coupling.

Description
-----------
``ConstantStratification`` registers the buoyancy tracer ``b`` and
contributes **both** linear coupling terms (D1's driving example):
``+b/dsqr`` in the w-equation (buoyancy force) and ``-N^2 w`` in the
b-equation (restoring). It owns the constant ``n2`` leaf and provides
``stratification.n2``; ``dsqr`` is read from ``ctx.params``. The
terms are pure field arithmetic (``.to`` interpolation across the
staggered w-b face), so their halo stencils are traced normally and
the module declares no ``extra_halo``.

``MeridionalStratification`` is the varying twin (the
FPlaneCoriolis/BetaPlaneCoriolis two-type precedent): :math:`N^2(y)`
is carried as an AUXILIARY ``n2`` field on a meridional
``fr.spatial.Profile("y")`` and the module does **not** provide the constant
``stratification.n2`` (provides-implies-constancy, 02_rules).
The restoring term samples :math:`N^2` **at the** ``b`` **nodes**
(``n2.to(b)`` — a pure broadcast, since the profile and the
collocated ``b`` share the meridional nodes) so the coupling pair

.. math::
    \partial_t w = b / \delta^2 , \qquad
    \partial_t b = -N^2(y)\, w

stays exactly M-skew-adjoint under the varying energy metric
``diag(1, 1, dsqr, 1/N^2(y))`` by ``.to`` adjointness: the pointwise
:math:`N^2` at ``b`` cancels the ``1/N^2`` metric weight there,
leaving the plain measure-weighted interpolation pair — for **any**
strictly positive profile. Fourier-diagonalizable consumers reject
the model through the missing provide; the dense-column channel
engine serves it.

``b`` is declared BC-free on every grid (topology-driven walls, C8):
walls enter through grid periodicity alone, the buoyancy's trig
parity on a walled grid is derived by the physics layers
(eigenmodes/transforms), never by a declaration knob.
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.nonhydro2.params import DSQR

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


@partial(jaxify, dynamic=("n2",))
class ConstantStratification(fr.model.Module):

    r"""Registers ``b``; contributes both linear coupling terms.

    Parameters
    ----------
    n2 : float | fr.model.Ramp, optional
        The constant squared buoyancy frequency ``N^2`` (default: 1.0);
        may be an ``fr.model.Ramp`` for a spun-up stratification.
    family : str | None, optional
        The discretization family of the buoyancy tracer ``b``
        (FV-D1b): ``"fv"`` declares it on the average family
        (``CellAvg^3``), so its flux-form advection conserves total
        buoyancy to machine zero while the nodal velocity/pressure
        state is untouched; ``"nodal"`` keeps it collocated with the
        pressure cell. None defers to the grid-level default (the
        rest of the model), so on today's nodal grids ``b`` stays
        nodal unless asked otherwise (default: None).
    """

    def __init__(
        self,
        n2: float | fr.model.Ramp = 1.0,
        *,
        family: str | None = None,
    ) -> None:
        """Store the stratification leaf and the ``b`` family."""
        self.n2 = fr.model.leaf(n2)
        self._family = family

    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """The buoyancy tracer ``b`` on the requested family."""
        return (
            fr.model.FieldDeclaration.tracer(
                "b", space=fr.spatial.Collocated(family=self._family),
                long_name="Buoyancy", units="m/s^2"),
        )

    field_references = (
        fr.model.FieldReference(
            "w", hint="buoyancy couples to vertical velocity, "
                      "declared by a dynamical core (nh.DynamicalCore)"),
    )
    parameter_declarations = (
        fr.model.ParameterDeclaration(
            fr.model.params.STRATIFICATION_N2, attr="n2",
            units="1/s^2",
            doc="squared buoyancy frequency N^2"),
    )
    parameter_references = (
        fr.model.ParameterReference(DSQR, hint="declared by nh.DynamicalCore"),
    )

    @fr.model.term(advances=("w",), linear=True,
                   linear_params=(fr.model.params.STRATIFICATION_N2, DSQR))
    def buoyancy_force(self, state, ctx) -> dict:  # noqa: ANN001
        """``dw/dt += b / dsqr`` (buoyancy interpolated onto the w face).

        The two ``linear=True`` coupling terms depend on ``n2`` and the
        core's ``dsqr``; both potentially-time-dependent parameters are
        annotated here (TDF-D4) so the structural frozen-``L`` guard
        reports a ramped ``n2``. ``dsqr`` lives on the core module, so
        this local annotation is inert (the leaf is out of reach); a
        ramped ``dsqr`` is reported by ``DynamicalCore`` itself.
        """
        dsqr = ctx.params[DSQR]
        return {"w": state["b"].to(state["w"]) / dsqr}

    @fr.model.term(advances=("b",), linear=True)
    def restoring(self, state, ctx) -> dict:  # noqa: ANN001
        """``db/dt += -N^2 w`` (w interpolated onto the b cell)."""
        n2 = ctx.params[fr.model.params.STRATIFICATION_N2]
        return {"b": -(n2 * state["w"].to(state["b"]))}


class MeridionalStratification(fr.model.Module):

    r"""Registers ``b``; the linear coupling with :math:`N^2(y)`.

    Description
    -----------
    The varying twin of :class:`ConstantStratification`: declares
    the AUXILIARY ``n2`` field on a meridional ``fr.spatial.Profile("y")``
    (materialized from the callable) and contributes both linear
    coupling terms with :math:`N^2` sampled pointwise at the ``b``
    nodes — the pairing that keeps ``(dw = b/dsqr, db = -N^2 w)``
    exactly M-skew under the ``1/N^2(y)`` energy weight. Provides
    **no** ``stratification.n2`` scalar (its :math:`N^2` is a field,
    not a constant — provides-implies-constancy).

    Parameters
    ----------
    n2 : Callable
        The squared buoyancy frequency profile ``n2(y)``, evaluated
        on the meridional coordinate; must be strictly positive for
        the energy metric.
    meridional : str, optional
        The meridional coordinate name (default: ``"y"``).
    family : str | None, optional
        The discretization family of the buoyancy tracer ``b``
        (FV-D1b); ``"fv"`` declares it on ``CellAvg^3``. The ``n2``
        profile carries no ``family=`` of its own and so **follows the
        grid family** — a ``CellAvg`` cell-average profile on an FV
        model, a nodal ``Profile`` on a nodal one. It is materialized
        at the cell midpoints on both families (the average profile's
        evaluation nodes are the midpoints), so at 2nd order the
        sampled ``N^2(y)`` values are the same numbers and the
        ``.to(b)`` broadcast onto the shared meridional nodes is
        unaffected. None defers to the grid-level default
        (default: None).
    """

    def __init__(
        self, n2: Callable, *, meridional: str = "y",
        family: str | None = None,
    ) -> None:
        """Store the profile callable, coordinate name, and family."""
        if not callable(n2):
            raise TypeError(
                "MeridionalStratification carries a varying "
                f"stratification profile n2(y); got {n2!r} — a "
                "constant N^2 is nh.ConstantStratification(n2=...)")
        self._n2_fn = n2
        self._meridional = meridional
        self._family = family

    field_references = (
        fr.model.FieldReference(
            "w", hint="buoyancy couples to vertical velocity, "
                      "declared by a dynamical core (nh.DynamicalCore)"),
    )
    parameter_references = (
        fr.model.ParameterReference(DSQR, hint="declared by nh.DynamicalCore"),
    )

    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """The ``b`` tracer and the ``n2(y)`` meridional profile."""
        return (
            fr.model.FieldDeclaration.tracer(
                "b", space=fr.spatial.Collocated(family=self._family),
                long_name="Buoyancy", units="m/s^2"),
            fr.model.FieldDeclaration(
                "n2", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._n2_default,
                long_name="Squared buoyancy frequency",
                units="1/s^2"),
        )

    def _n2_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: materialize the ``n2(y)`` profile.

        The meridional profile carries a single non-constant
        coordinate, so ``init`` names exactly that coordinate; the
        signature is stamped dynamically to match
        ``self._meridional`` (the ``BetaPlaneCoriolis._f_default``
        precedent). No pre-syncing (GAP-B).
        """
        fn, mer = self._n2_fn, self._meridional

        def init(**coords: object) -> object:
            return fn(coords[mer])

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return grid.create_field(space, init=init, name="n2")

    @fr.model.term(advances=("w",), linear=True, linear_params=(DSQR,))
    def buoyancy_force(self, state, ctx) -> dict:  # noqa: ANN001
        """``dw/dt += b / dsqr`` (buoyancy interpolated onto w)."""
        dsqr = ctx.params[DSQR]
        return {"w": state["b"].to(state["w"]) / dsqr}

    @fr.model.term(advances=("b",), linear=True, linear_fields=("n2",))
    def restoring(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        """``db/dt += -N^2(y) w``, with ``N^2`` sampled at ``b``.

        The pointwise sampling (``n2.to(b)`` is a broadcast onto the
        shared meridional nodes) keeps the coupling pair exactly
        M-skew under the ``1/N^2(y)`` energy weight for any profile.
        """
        b = state["b"]
        return {"b": -(state["n2"].to(b) * state["w"].to(b))}
