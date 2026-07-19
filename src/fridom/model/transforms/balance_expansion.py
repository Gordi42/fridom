r"""
BalanceExpansion: slaving-expansion balance on the eigenmode tiers.

Description
-----------
``fr.transforms.BalanceExpansion`` is the current form of the
**nonlinear normal mode decomposition / initialization** (Machenhauer
1977; Baer & Tribbia 1977; Warn, Bokhove, Shepherd & Vallis 1995;
Eden, Chouksey & Olbers 2019; Chouksey, Eden, Masur & Oliver 2023):
project a state onto the slow eigenmode family and slave the fast
(wave) content to it order by order in the timescale-separation
parameter. The recursion itself lives in the private array-agnostic
core (:mod:`fridom.model.transforms._slaving`); this module
assembles the four operator ingredients on a model's own eigenmode
tier and wraps the result as a Tier-2 ``State -> State`` transform.

Projector formulation. For :math:`\partial_t\phi = L\phi +
B(\phi,\phi)` the method needs only the slow projector :math:`V`,
the fast projector :math:`W`, the inverse :math:`L_w^{-1}` of the
linear operator on the fast subspace, and the symmetric bilinear
form :math:`B` of the quadratic term — no per-mode bookkeeping. The
projectors and :math:`L_w^{-1} = f(L)` with :math:`f = -1/(i\omega)`
(the eigenmode tiers label their columns ``L q = -i omega q``)
come from the model's eigenmode tier (the ``function(f, sel)``
applicator), :math:`L` from ``fr.linearize(model)``, and :math:`B`
from the polarization of the nonlinear tendency. The slow space is
**not** required to be stationary (:math:`LV \ne 0` is supported):
a beta-plane Rossby band selected by a frequency predicate
qualifies as long as its rates stay well below the fast gap.

Order-consistent closure (the shipping default). The slow tendency
driving the higher slow-time derivatives is closed order
consistently — the wave feedback :math:`VB(v, s(v))` enters the
slow derivative table at matching series order, as in Warn et al.'s
superbalance hierarchy. The leading-order closure (v1 bookkeeping)
demonstrably saturates the residual slope at 3 from order 3 on; it
remains available only in the private core.

Discretization note. Balance is discretization-specific: balancing
a staggered-grid model with *continuum* eigenvectors injects an
O(1) zeroth-order error (Chouksey et al. 2023, Fig. 8). This
implementation uses the model's own **discrete** eigenmodes on
every tier — the operator-sourced analytic symbols of the C-grid on
periodic (and walled-vertical) grids, and the dense-column numeric
eigenbasis of the walled channel — so the balanced state is
consistent with the model's discrete linear operator by
construction.

Dealiasing. The bilinear form evaluates the model's own nonlinear
terms (``model.variant(term_filter=nonlinear)``), so :math:`B`
inherits exactly the model's dealiasing configuration: a spectral
scheme configured with the ``PadFactor`` machinery is dealiased
here too, while the finite-difference advection modules are nodal
(aliased by design). No separate dealiasing switch exists.
"""
from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from fridom.model._eigenbasis import (
    ZERO_FREQUENCY_TOL,
    ChannelEigenmodesBase,
    _family_codes,
    _predicate_mask,
    predicate_projection,
)
from fridom.model.analytic_distributed import analytic_route
from fridom.model.eigenstates import resolve_mode_branches
from fridom.model.energy import EnergyMetric
from fridom.model.term_predicates import linear, linearize
from fridom.model.transforms._slaving import (
    SlavingOps,
    balance_expansion,
)
from fridom.model.transforms.base import StateTransform
from fridom.model.transforms.info import TransformInfo
from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.model.transforms._slaving import BalanceResult
    from fridom.model.transforms.signature import StateSignature

#: amplitude of the random probe state of the quadraticity lint.
_LINT_AMPLITUDE = 1e-4
#: relative homogeneity-defect bound of the quadraticity lint (a
#: linear contamination scores ~1 at this amplitude; the mildly
#: non-polynomial PV division of the Sadourny scheme scores
#: ~ Ro * amplitude, far below).
_LINT_TOL = 1e-2


# ================================================================
#  Eigenmode-tier resolution (package dispatch)
# ================================================================
def _eigenmodes_from_model(model: object, at_time: float) -> object:
    """Resolve the model's eigenmode tier via its owning package."""
    module = type(model.state).__module__
    package, _, _ = module.rpartition(".")
    if (not package.startswith("fridom.")
            or package.startswith(
                ("fridom.framework", "fridom.spatial", "fridom.model"))):
        raise ValueError(
            "BalanceExpansion dispatches on the model package that "
            "owns the state vocabulary; this model's state is "
            f"{type(model.state).__name__!r} from {module!r} — "
            "assemble with a model package core (e.g. sw.Model / "
            "nh.Model)")
    eigen = import_module(f"{package}.eigenmodes")
    builder = getattr(eigen, "from_model", None)
    if builder is None:
        raise ValueError(
            f"the model package {package!r} exposes no "
            "'eigenmodes.from_model' surface; BalanceExpansion "
            "needs the package eigenmode tiers")
    return builder(model, at_time=at_time)


# ================================================================
#  The analytic tier: physical round-trip operator wrappers
# ================================================================
class _AnalyticOperator:

    """
    Physical-space wrapper of an analytic coefficient applicator.

    Description
    -----------
    Carries a physical state to each component's own coefficient
    basis through the eigenmode kit (retagging onto the kit's
    analysis spaces — identity on periodic grids), applies the
    captured coefficient-space map (``em.projector`` /
    ``em.function`` output), and returns through the backward
    transforms taking the real part (the Hermitian closure on the
    rfft half-lattice; exact for conjugation-closed selections).

    On a grid whose default layout shards a transform axis the plain
    per-component round-trip hits the Tier-1 taught error; a fully
    periodic sharded operand instead routes through the fused
    ``jax.shard_map`` matrix apply — the whole ``V`` / ``W`` /
    ``L_w^{-1}`` operator becomes one per-mode ``D x D`` matrix call
    (assembled once, cached), and the transient full-size coefficient
    ``VectorField`` never materializes.
    """

    def __init__(
        self,
        em: object,
        kit: object,
        components: tuple[str, ...],
        apply_fn: Callable,
        *,
        branches: tuple[int, ...],
        f: Callable | None,
    ) -> None:
        """Prebuild the per-component forward transforms."""
        self._em = em
        self._kit = kit
        self._components = components
        self._apply = apply_fn
        self._branches = branches
        self._f = f
        self._forward = {c: kit.forward(c) for c in components}
        self._matrices: dict[int, jax.Array] = {}

    def __call__(self, state: object) -> object:
        """Round-trip the state through the coefficient map."""
        route = analytic_route(self._em, state)
        if route is not None:
            out = route.apply_matrix(
                {c: state[c] for c in self._components},
                self._matrix(route))
            return type(state)({c: out[c] for c in self._components})
        coeff = VectorField({
            c: self._forward[c](
                state[c].retag(self._forward[c].domain))
            for c in self._components})
        out = self._apply(coeff)
        return type(state)({
            c: self._kit.backward(c)(out[c]).real.retag(state[c])
            for c in self._components})

    def _matrix(self, route: object) -> jax.Array:
        """Assemble (once, per grid) the fused per-mode operator matrix."""
        key = id(self._em.grid)
        matrix = self._matrices.get(key)
        if matrix is None:
            matrix = self._em.operator_matrix(
                route.coeff_of, branches=self._branches, f=self._f)
            self._matrices[key] = matrix
        return matrix


def _analytic_operators(
    em: object, slow: object,
) -> tuple[Callable, Callable, Callable, tuple[str, ...],
           StateSignature | None, str]:
    """Assemble (V, W, inv_lw, components, signature, label)."""
    if not isinstance(slow, str) and callable(slow):
        raise TypeError(
            "a predicate slow selection needs the labeled channel "
            "eigenbasis (exactly one bounded horizontal axis); the "
            "analytic tier selects eigenmode branches by value "
            "(slow=0 or slow=(0,)) or the 'vortical' family string")
    if isinstance(slow, str):
        if slow != "vortical":
            raise ValueError(
                f"unknown analytic slow selection {slow!r}: the "
                "analytic tier carries the 'vortical' family "
                "(branch 0) and the inertia-gravity branches "
                "+1/-1; pass 'vortical' or branch values")
        branches = (0,)
        label = slow
    else:
        branches = resolve_mode_branches(slow)
        label = str(branches)
    fast = tuple(b for b in (0, 1, -1) if b not in branches)
    if not fast:
        raise ValueError(
            "the slow selection covers every eigenmode branch: "
            "no fast complement is left to slave")
    components = tuple(em.q(0).component_names)
    kit = getattr(em, "kit", None)
    if kit is None:
        kit = em._kit  # noqa: SLF001 — package-internal kit access

    def inv_lw(w: np.ndarray) -> np.ndarray:
        return -1.0 / (1j * w)

    # V / W are projectors (f = None on the distributed matrix, exactly
    # the ``function(np.ones_like, ...)`` the eager path builds); the
    # slaved-inverse carries the f(omega) weights (the distributed matrix
    # guards the zero-frequency structural set, never floors)
    slow_op = _AnalyticOperator(
        em, kit, components, em.function(np.ones_like, branches),
        branches=branches, f=None)
    fast_op = _AnalyticOperator(
        em, kit, components, em.function(np.ones_like, fast),
        branches=fast, f=None)
    inv_op = _AnalyticOperator(
        em, kit, components, em.function(inv_lw, fast),
        branches=fast, f=inv_lw)
    return slow_op, fast_op, inv_op, components, None, label


# ================================================================
#  The channel tier: labeled family / predicate operators
# ================================================================
def _channel_operators(
    em: ChannelEigenmodesBase, slow: object,
) -> tuple[Callable, Callable, Callable, tuple[str, ...],
           StateSignature | None, str]:
    """Assemble (V, W, inv_lw, components, signature, label)."""
    labels = np.asarray(em.labels)
    omega = np.asarray(em.omega)
    if isinstance(slow, str):
        codes = _family_codes(em, slow)
        slow_mask = np.isin(labels, np.asarray(codes))
        label = slow
    elif callable(slow):
        slow_mask = np.asarray(_predicate_mask(em, slow))
        label = getattr(slow, "__name__", "predicate")
    else:
        raise TypeError(
            "the channel slow selection is a family name (e.g. "
            "'vortical') or a predicate (omega, labels) -> bool "
            f"mask; got {slow!r}")
    # the fast complement is NOT I - V: structural-zero families
    # (the nonhydro constraint/Leray complement) and zero-frequency
    # columns are excluded — anything outside slow + fast is
    # annihilated by the expansion, matching the theory (those
    # columns carry no dynamics and no invertible L_w).
    excluded = np.zeros(labels.shape, dtype=bool)
    for name in em.nonphysical_families:
        excluded |= labels == em.families[name]
    fast_mask = jnp.asarray(
        ~slow_mask & ~excluded
        & (np.abs(omega) >= ZERO_FREQUENCY_TOL))

    def fast_selection(
        omega: jax.Array,  # noqa: ARG001 — the predicate signature
        labels: jax.Array,  # noqa: ARG001 — the predicate signature
    ) -> jax.Array:
        """Return the precomputed fast column mask."""
        return fast_mask

    slow_op = em.projector(slow)
    fast_op = predicate_projection(em, fast_selection, name="P[fast]")
    inv_op = em.function(lambda w: -1.0 / (1j * w), fast_selection)
    return (slow_op, fast_op, inv_op, em.components,
            slow_op.domain, label)


# ================================================================
#  The public transform
# ================================================================
class BalanceExpansion(StateTransform):

    r"""
    Balance a state by the nonlinear-normal-mode slaving expansion.

    Description
    -----------
    The higher-order balance ("NNMD") transform: the slow content
    :math:`v = Vz` is kept as the base point, the fast content is
    discarded, and the balanced fast (wave) field is rebuilt from
    the slaving recursion

    .. math::

        z_b = \sum_{n=0}^{N} \phi_n, \qquad
        \phi_{n+1} = L_w^{-1} W \Big[\dot\phi_n
            - \sum_{j=0}^{n} B(\phi_j, \phi_{n-j})\Big]

    at truncation order ``N = order`` (``order=0`` is the plain slow
    projection; ``order=1`` the Machenhauer state). Slow-time
    derivatives are closed **order-consistently** (Warn et al. 1995;
    module docstring), and the slow space may rotate
    (:math:`LV \ne 0`): a beta-plane Rossby band selected by a
    frequency predicate is a valid slow space as long as its linear
    and advective rates stay well below the fast frequency gap.

    Eigenmode tiers are dispatched from the model: a fully periodic
    (or walled-vertical nonhydro) grid uses the analytic
    operator-sourced discrete eigenmodes; a grid with exactly one
    bounded horizontal axis uses the labeled numeric channel
    eigenbasis. On every tier the fast set is the complement of the
    slow selection **minus** the structural-zero families (the
    nonhydro constraint/Leray columns, zero-frequency columns):
    content outside slow + fast carries no dynamics and is
    annihilated. A singular :math:`1/(i\omega)` meeting a selected
    zero-frequency mode is a taught error, never a floored division.

    The nonlinear tendency :math:`N(z)` is the model's own
    (``model.variant(term_filter=nonlinear)``), evaluated at the
    frozen ``at_time``; the bilinear form is its polarization
    :math:`B(x, y) = (N(x+y) - N(x) - N(y))/2` (one evaluation when
    the arguments coincide). :math:`B` therefore inherits the
    model's advection scheme and its dealiasing configuration
    verbatim. A cheap quadraticity lint at construction warns when
    the ``nonlinear`` filter keeps terms that are not quadratic in
    the state (``lint=False`` skips it).

    Literature: Machenhauer (1977); Baer & Tribbia (1977); Warn,
    Bokhove, Shepherd & Vallis (1995); Eden, Chouksey & Olbers
    (2019); Chouksey, Eden, Masur & Oliver (2023). See the module
    docstring for the discretization note (discrete eigenmodes avoid
    the O(1) zeroth-order error of continuum modes on staggered
    grids).

    Parameters
    ----------
    model : Model
        The assembled model to balance on (never mutated; internal
        linear/nonlinear variants are assembled from it).
    order : int, optional
        The truncation order ``N >= 0`` (default: 3).
    slow : str | int | tuple[int, ...] | Callable, optional
        The slow selection: on the channel tier a family name or a
        ``(omega, labels) -> bool mask`` predicate (the
        ``projector`` grammar); on the analytic tier the
        ``'vortical'`` family or branch value(s) from ``{0, +1,
        -1}`` (default: "vortical").
    nonlinear : Callable | None, optional
        The term predicate selecting the quadratic tendency terms;
        ``None`` uses ``~fr.terms.linear`` (default: None).
    at_time : float, optional
        The clock time freezing time-dependent parameters and
        pinning every internal tendency evaluation (default: 0.0).
    lint : bool, optional
        Run the construction-time quadraticity lint on the
        ``nonlinear`` selection (default: True).
    """

    def __init__(
        self,
        model: object,
        order: int = 3,
        *,
        slow: object = "vortical",
        nonlinear: Callable | None = None,
        at_time: float = 0.0,
        lint: bool = True,
    ) -> None:
        """Resolve the eigenmode tier and assemble the operators."""
        if isinstance(order, bool) or not isinstance(order, int) \
                or order < 0:
            raise ValueError(
                f"order must be a non-negative int, got {order!r}")
        self._model = model
        self._order = order
        self._at_time = float(at_time)
        term_filter = ~linear if nonlinear is None else nonlinear
        em = _eigenmodes_from_model(model, self._at_time)
        build = (_channel_operators
                 if isinstance(em, ChannelEigenmodesBase)
                 else _analytic_operators)
        (self._slow, self._fast, self._inv_lw, self._components,
         self._signature, self._slow_label) = build(em, slow)
        self._em = em
        self._linear = linearize(model)
        self._quadratic = model.variant(
            term_filter=term_filter,
            name="BalanceExpansion/nonlinear")
        # snapshot=True is load-bearing: the balance expansion pins the
        # metric to its frozen eigenbasis (TDF-D6), so a time_dependent
        # field weight must be baked at at_time, not sourced off state.
        self._metric = EnergyMetric.from_model(
            model, at_time=self._at_time,
            require_constant_coriolis=False,
            allow_field_weights=True, snapshot=True)
        if lint:
            self._lint_quadraticity()

    # ================================================================
    #  Declared structure (Tier 2, endo)
    # ================================================================
    @property
    def traceable(self) -> bool:
        """Tier 2: runs model tendencies internally (host only)."""
        return False

    @property
    def domain(self) -> StateSignature | None:
        """The prognostic endo signature (channel tier) or None."""
        return self._signature

    @property
    def codomain(self) -> StateSignature | None:
        """The prognostic endo signature (channel tier) or None."""
        return self._signature

    @property
    def order(self) -> int:
        """The truncation order ``N``."""
        return self._order

    @property
    def eigenmodes(self) -> object:
        """The resolved eigenmode tier the operators act on."""
        return self._em

    # ================================================================
    #  Application
    # ================================================================
    def _evaluate(
        self, state: object,
    ) -> tuple[object, TransformInfo]:
        """Run the slaving expansion at the configured order."""
        result = self._expand(state, self._order)
        return result.state, TransformInfo.EMPTY

    def _expand(self, state: object, order: int) -> BalanceResult:
        """Run the private core on the restricted prognostic state."""
        ops = SlavingOps(
            slow=self._slow, fast=self._fast, inv_lw=self._inv_lw,
            bilinear=self._make_bilinear(), lin=self._lin)
        return balance_expansion(
            self._restrict(state), order=order, ops=ops,
            scheme="direct", closure="consistent")

    def _restrict(self, state: object) -> object:
        """Restrict the input to the eigenmode component set."""
        missing = tuple(
            c for c in self._components if c not in state)
        if missing:
            raise ValueError(
                f"BalanceExpansion needs the prognostic components "
                f"{self._components}; the input state is missing "
                f"{missing}")
        return type(state)(
            {c: state[c] for c in self._components})

    # ================================================================
    #  The injected operator pieces
    # ================================================================
    def _lin(self, state: object) -> object:
        """Return the linearized tendency ``L z`` at the frozen time."""
        return self._linear.tendency(
            state, t=self._at_time, constraints=True)

    def _nonlinear_tendency(self, state: object) -> object:
        """Return the nonlinear tendency ``N(z)`` at the frozen time."""
        return self._quadratic.tendency(
            state, t=self._at_time, constraints=True)

    def _make_bilinear(self) -> Callable:
        """Build the polarization ``B`` with a per-call ``N`` cache."""
        cache: dict[int, tuple[object, object]] = {}

        def n_of(z: object) -> object:
            entry = cache.get(id(z))
            if entry is not None and entry[0] is z:
                return entry[1]
            value = self._nonlinear_tendency(z)
            cache[id(z)] = (z, value)
            return value

        def bilinear(z1: object, z2: object) -> object:
            if z1 is z2:
                return n_of(z1)
            return 0.5 * (self._nonlinear_tendency(z1 + z2)
                          - n_of(z1) - n_of(z2))

        return bilinear

    # ================================================================
    #  Diagnostics
    # ================================================================
    def residual_series(self, state: object) -> float:
        r"""
        Return the next-order series residual of the balanced state.

        Description
        -----------
        :math:`\|\phi_{N+1}\|_M / \|z_b\|_M` — the M-norm of the
        first truncated term relative to the order-``N`` balanced
        state (the epsilon-slope quantity: order ``N`` scales as
        :math:`\varepsilon^{N+1}`).

        Parameters
        ----------
        state : VectorField
            The state to balance and diagnose.

        Returns
        -------
        float
            The relative next-order term norm (0.0 on a vanishing
            balanced state).
        """
        result = self._expand(state, self._order + 1)
        z_b = result.terms[0]
        for term in result.terms[1:-1]:
            z_b = z_b + term
        denom = float(self._metric.norm(z_b))
        if denom == 0.0:
            return 0.0
        return float(self._metric.norm(result.terms[-1])) / denom

    def residual_fast(self, state: object) -> float:
        r"""
        Return the differential fast-tendency residual.

        Description
        -----------
        :math:`\|W F(z_b) - \partial_s W z_b[\dot v]\|_M /
        \|z_b\|_M` with :math:`\dot v = V F(z_b)` and the slaved
        prediction by a central finite difference of the balance map
        along the slow flow (step ``h ~ eps**(1/3)`` scaled by the
        base-point norm) — the :math:`t \to 0` derivative of the
        run-and-rebalance residual, at two balance plus two tendency
        evaluations.

        Parameters
        ----------
        state : VectorField
            The state to balance and diagnose.

        Returns
        -------
        float
            The relative differential residual (0.0 on a vanishing
            balanced state).
        """
        result = self._expand(state, self._order)
        z_b = result.state
        scale_b = float(self._metric.norm(z_b))
        if scale_b == 0.0:
            return 0.0
        f_zb = self._model.tendency(
            z_b, t=self._at_time, constraints=True)
        w_f = self._fast(f_zb)
        vdot = self._slow(f_zb)
        rate = float(self._metric.norm(vdot))
        if rate == 0.0:
            return float(self._metric.norm(w_f)) / scale_b
        direction = (1.0 / rate) * vdot
        v = result.terms[0]
        h = (float(np.finfo(np.float64).eps) ** (1.0 / 3.0)
             * max(float(self._metric.norm(v)), 1.0))
        plus = self._expand(v + h * direction, self._order).state
        minus = self._expand(v - h * direction, self._order).state
        ds = (rate / (2.0 * h)) * (self._fast(plus)
                                   - self._fast(minus))
        return float(self._metric.norm(w_f - ds)) / scale_b

    # ================================================================
    #  The quadraticity lint
    # ================================================================
    def _lint_quadraticity(self) -> None:
        """Warn when the nonlinear selection is not quadratic."""
        rng = np.random.default_rng(2701)
        fields = {}
        for name in self._components:
            field = self._model.state[name]
            data = _LINT_AMPLITUDE * rng.standard_normal(
                np.asarray(field.data).shape)
            fields[name] = field.with_data(
                jnp.asarray(data, dtype=field.data.dtype))
        z = VectorField(fields)
        n_one = self._nonlinear_tendency(z)
        n_two = self._nonlinear_tendency(2.0 * z)
        reference = float(self._metric.norm(n_two))
        if reference == 0.0:
            return
        defect = float(
            self._metric.norm(n_two - 4.0 * n_one)) / reference
        if defect > _LINT_TOL:
            warnings.warn(
                "the nonlinear= term filter keeps terms that are "
                "not quadratic in the state (relative homogeneity "
                f"defect {defect:.2e} on a small random probe): the "
                "polarization B(x, y) = (N(x+y) - N(x) - N(y))/2 "
                "assumes N(z) = B(z, z), so balance orders beyond 1 "
                "are unreliable with this selection. Restrict the "
                "filter to the quadratic (advection) terms, or pass "
                "lint=False to silence this check.",
                stacklevel=3)

    def __repr__(self) -> str:
        """``BalanceExpansion(order=N, slow=...)``."""
        return (f"BalanceExpansion(order={self._order}, "
                f"slow={self._slow_label!r})")
