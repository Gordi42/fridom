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

A ``fr.model.ProfileFunction`` law ``n2(y, t, *params)`` (TDF-D7)
selects a **time-dependent** :math:`N^2(y, t)`: the ``n2`` field is
marked ``time_dependent`` (materialized at ``t = 0``) and rewritten
every substage by a SELF_UPDATE stage sampling the law at the stage
clock — the ``BetaPlaneCoriolis`` ``f`` law precedent — so the
coupling terms, the energy metric's stage-time ``1/N^2`` weight
(TDF-D10), I/O and restart all track it. The law must stay strictly
positive at every ``t``. The static (plain-callable) profile path is
untouched: no stage, no marker, no ``extra_halo``.

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
from fridom.model.scheduled_field import ProfileFunction, profile_coords
from fridom.nonhydro2.params import DSQR
from fridom.spatial.decomposition.halo import HaloSpec

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


@partial(jaxify, dynamic=("_n2_law",))
class MeridionalStratification(fr.model.Module):

    r"""Registers ``b``; the linear coupling with :math:`N^2(y[, t])`.

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

    A ``fr.model.ProfileFunction`` law ``n2(y, t, *params)`` (TDF-D7)
    drives a **time-dependent** :math:`N^2(y, t)`: the ``n2`` field is
    marked ``time_dependent`` (materialized at ``t = 0``) and rewritten
    every substage by a SELF_UPDATE stage sampling the law at the stage
    clock (the ``BetaPlaneCoriolis`` ``f`` law precedent), so the
    coupling terms, the energy metric's stage-time ``1/N^2`` weight
    (TDF-D10), I/O and restart all see the fresh value with no clock
    plumbing. The law must stay strictly positive at every ``t`` (the
    energy metric weights buoyancy by ``1/N^2(y, t)``). A frozen-``L``
    (``ETDRK4``) stepper then refuses the model automatically — the
    marker feeds the frozen-``L`` guard through the ``restoring`` term's
    ``linear_fields=("n2",)`` — while ``AdamBashforth`` runs it. The
    static (plain-callable) profile path is untouched — no stage, no
    marker, no ``extra_halo`` (the marker is repr-participating only
    when True, so the assembly fingerprint is bit-identical) — the law
    path activates only when a ``ProfileFunction`` is passed.

    Parameters
    ----------
    n2 : Callable | fr.model.ProfileFunction
        The squared buoyancy frequency profile, evaluated on the
        meridional coordinate. A plain callable ``n2(y)`` is a
        **static** profile, materialized once at assembly. A
        ``fr.model.ProfileFunction`` law
        ``n2=fr.model.ProfileFunction(lambda y, t, *params: ...)`` gives
        a **time-dependent** :math:`N^2(y, t)` rewritten each substage
        (TDF-D7). Either spelling must be strictly positive at every
        ``t`` for the energy metric; a *constant* :math:`N^2` is
        ``nh.ConstantStratification(n2=...)``.
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
        self, n2: Callable | fr.model.ProfileFunction, *,
        meridional: str = "y", family: str | None = None,
    ) -> None:
        r"""Store the profile (static callable or law), coordinate, family.

        A ``fr.model.ProfileFunction`` selects the time-dependent
        :math:`N^2(y, t)` path (``self._n2_law``, the dynamic-leaf law
        whose ``params`` flow as pytree leaves); a plain callable stays
        the static profile in ``self._n2_fn`` (a static slot). A
        ``ProfileFunction`` is deliberately not callable, so the
        ``isinstance`` check comes first.
        """
        if isinstance(n2, ProfileFunction):
            self._n2_law = n2
            self._n2_fn = None
        elif callable(n2):
            self._n2_law = None
            self._n2_fn = n2
        else:
            raise TypeError(
                "MeridionalStratification carries a varying "
                f"stratification profile n2(y[, t]); got {n2!r} — pass a "
                "callable n2(y) for a static profile, an "
                "fr.model.ProfileFunction(lambda y, t, *params: ...) for a "
                "time-dependent N^2(y, t), or "
                "nh.ConstantStratification(n2=...) for a constant N^2")
        self._meridional = meridional
        self._family = family
        #: grid coordinate names for the law-path halo (set at bind)
        self._halo_coords: tuple[str, ...] = ()

    field_references = (
        fr.model.FieldReference(
            "w", hint="buoyancy couples to vertical velocity, "
                      "declared by a dynamical core (nh.DynamicalCore)"),
    )
    parameter_references = (
        fr.model.ParameterReference(DSQR, hint="declared by nh.DynamicalCore"),
    )

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def _profile_active(self) -> bool:
        """Whether a ``ProfileFunction`` drives ``n2(y, t)`` (TDF-D7).

        A host-side structural predicate on the ``n2`` law type; the
        static-callable profile leaves it False and stays bit-identical.
        """
        return self._n2_law is not None

    @property
    def extra_halo(self) -> HaloSpec | None:
        """The law-path halo substitute (V-N2); ``None`` otherwise.

        The SELF_UPDATE rewrites ``n2`` from raw sampled data
        (``with_data``, halo-trace exempt), so on the law path the module
        declares its coupling-term reach itself: one ghost cell per grid
        axis covers the staggered ``.to`` averages of both terms
        (``b.to(w)``, ``w.to(b)``, ``n2.to(b)`` — reach 1), the same
        over-approximation the ``BetaPlaneCoriolis`` law path documents.
        The static path keeps ``None`` and stays halo-traced
        bit-identically.
        """
        if not self._profile_active:
            return None
        return HaloSpec(dict.fromkeys(self._halo_coords, 1))

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """The ``b`` tracer and the ``n2`` meridional profile.

        The static (plain-callable) path declares the ``n2`` profile
        exactly as before (no ``time_dependent`` marker, so the assembly
        fingerprint is untouched). A ``ProfileFunction`` law marks ``n2``
        ``time_dependent`` (materialized at ``t = 0``), rewritten each
        substage by the SELF_UPDATE stage (TDF-D7).
        """
        b = fr.model.FieldDeclaration.tracer(
            "b", space=fr.spatial.Collocated(family=self._family),
            long_name="Buoyancy", units="m/s^2")
        if self._profile_active:
            n2 = fr.model.FieldDeclaration(
                "n2", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._n2_profile_default,
                long_name="Squared buoyancy frequency",
                units="1/s^2", time_dependent=True)
        else:
            n2 = fr.model.FieldDeclaration(
                "n2", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._n2_default,
                long_name="Squared buoyancy frequency",
                units="1/s^2")
        return (b, n2)

    def _n2_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: materialize the static ``n2(y)`` profile.

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

    def _n2_profile_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        r"""Owner-method default: sample the ``n2(y, t)`` law at ``t = 0``.

        Mirrors ``BetaPlaneCoriolis._f_profile_default``: the AUXILIARY
        field is materialized as the ``t = 0`` snapshot so it keeps a
        valid static treedef; the SELF_UPDATE stage then rewrites it with
        the stage-time value each substage, so this frozen value is never
        read at run time. No pre-syncing (GAP-B).
        """
        coords = profile_coords(grid, space, (self._meridional,))
        data = self._n2_law.sample(coords, 0.0, space.shape)
        return grid.create_field(space, data=data, name="n2")

    # ================================================================
    #  The SELF_UPDATE stage (law path only, S1 per substage)
    # ================================================================
    @property
    def stages(self) -> tuple[fr.model.Stage, ...]:
        """The per-substage ``n2`` rewrite (law path only).

        The law path (``ProfileFunction``, TDF-D7) recomputes the full
        ``n2(y, t)`` law from raw sampled data each substage; the static
        profile emits no stage. Mirrors the ``BetaPlaneCoriolis.stages``
        law arm.
        """
        if self._profile_active:
            return (fr.model.Stage(
                kind=fr.model.StageKind.SELF_UPDATE, fn="_update_n2",
                name="stratification_n2", reads=("n2",),
                writes=("n2",)),)
        return ()

    def _update_n2(self, state, ctx) -> dict:  # noqa: ANN001
        """Re-evaluate the ``n2(y, t)`` law at the substage clock (TDF-D7).

        SELF_UPDATE runs first in every substage (S1), so the coupling
        terms and every ``n2`` consumer (the energy metric's ``1/N^2``
        weight, I/O) read the stage-time field, consistent with
        ``eval_params``. Mirrors ``BetaPlaneCoriolis._update_f_coriolis``.
        """
        time = getattr(ctx.clock, "time", ctx.clock)
        field = state["n2"]
        space = field.function_space
        coords = profile_coords(field.grid, space, (self._meridional,))
        value = self._n2_law.sample(coords, time, space.shape)
        return {"n2": field.with_data(value)}

    def bind(self, table) -> None:  # noqa: ANN001
        """Record the law-path halo axes (static path: no-op).

        The law path declares its own coupling-term reach
        (``extra_halo``), so it records every grid axis here (the
        ``BetaPlaneCoriolis.bind`` precedent); the static-callable
        profile stays halo-traced.
        """
        if self._profile_active:
            self._halo_coords = tuple(table.grid.names)

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
