r"""
Metric-coefficient derivative kinds on mapped grids.

Description
-----------
The operator side of the coordinate-mapping design (rules section
3.8; ``design/specs/operator_algebra/03_api_sketches.md`` sketch
4.4): "derivative at constant physical coordinate" and "derivative
at constant computational coordinate" are *different dispatch
kinds*. The seeded ``"diff"`` rows stay the computational
(constant-sigma) derivative; this module adds the **physical**
(constant-z) derivative under the ``"physical_diff"`` kind, seeded
by the grid exactly when a ``CoordinateMapping`` couples the
relevant coordinates.

The kind expands — like the grad/div builders — at application
against the operand space and the grid registry into the literal
sketch-4.4 algebra

.. math::

    \partial_x|_z
    = \partial_x|_\sigma
    - \frac{\partial z/\partial x}{\partial z/\partial \sigma}
      \,\partial_\sigma ,

an ``OperatorSum`` whose correction term is the registered ``diff``
along the mapped column, interpolated back onto the main term's
codomain and scaled by a :class:`MetricScaled` coefficient — a
quotient of ``grid.metric`` fields derived **at application time**
on the codomain space, so no operator ever caches a metric and
time-dependent parameters trace through (rules 2.3/3.8).

Dynamic geometry (stage C4): :meth:`MappedDerivative.with_params`
binds caller-supplied parameter fields (module-owned state, e.g.
the ``MovingGeometry`` ``H(t)`` field) into a transient builder
whose expanded coefficients derive from the *current* values via
the ``grid.metric`` ``params=`` overload — values enter as traced
arrays, so sweeping geometry through jit compiles once.
"""
# Coordinate-systems plan, stage C1: physical_diff dispatch kind
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    EigenbasisError,
    Operator,
    resolve_codomain,
)
from fridom.spatial.operators.registry import DispatchError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.operators.base import (
        FieldLike,
        OperatorRequirements,
    )
    from fridom.spatial.operators.registry import (
        OperatorRegistry,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


@final
class MetricScaled(Operator):

    """
    Operator scaled by a metric coefficient derived at application.

    Description
    -----------
    The lazy sibling of
    :class:`~fridom.spatial.operators.base.ScaledOperator`: instead
    of holding a coefficient value it holds metric *names* (static
    strings) and derives the coefficient — ``numerator`` over
    ``denominator`` — via ``grid.metric`` on the target's codomain
    space at every application. This keeps the registry entry fully
    static (no arrays in grid-owned structure) and the metrics
    honest under dynamic parameters: nothing is cached, values are
    re-derived from the mapping at trace time (rules 3.8). The
    scale is pointwise, so requirements delegate to the target; a
    field coefficient breaks translation invariance, so there is no
    symbol.

    Parameters
    ----------
    target : Operator
        The operator whose output is scaled.
    numerator : str | None, optional
        The metric name of the coefficient's numerator; None scales
        by the reciprocal of the denominator alone (default: None).
    denominator : str | None, optional
        The metric name of the coefficient's denominator; None
        scales by the numerator alone (default: None).
    params : Mapping[str, ScalarField] | None, optional
        Dynamic mapping-parameter fields threaded into every
        ``grid.metric`` derivation (the ``params=`` overload,
        stage C4). Instances carrying params are **transient** —
        built at application time (e.g. through
        ``MappedDerivative.with_params``) and applied immediately;
        the registry-seeded rows stay array-free (default: None).
    """

    def __init__(self, target: Operator, *,
                 numerator: str | None = None,
                 denominator: str | None = None,
                 params: Mapping[str, ScalarField] | None = None,
                 ) -> None:
        """Store the target and the static metric names."""
        if not isinstance(target, Operator):
            raise TypeError(
                f"target must be an Operator, got {target!r}")
        if numerator is None and denominator is None:
            raise ValueError(
                "a MetricScaled coefficient names at least one of "
                "numerator= or denominator=")
        if (numerator is not None
                and not isinstance(numerator, str)) or (
                denominator is not None
                and not isinstance(denominator, str)):
            raise TypeError(
                "metric coefficients are named by strings, got "
                f"{numerator!r} / {denominator!r}")
        self._target: Operator = target
        self._numerator: str | None = numerator
        self._denominator: str | None = denominator
        self._params: dict[str, ScalarField] | None = (
            None if params is None else dict(params))

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def target(self) -> Operator:
        """The scaled operator (static structure)."""
        return self._target

    @property
    def numerator(self) -> str | None:
        """The metric name of the numerator, or None."""
        return self._numerator

    @property
    def denominator(self) -> str | None:
        """The metric name of the denominator, or None."""
        return self._denominator

    @property
    def params(self) -> dict[str, ScalarField] | None:
        """Dynamic parameter fields of the derivations (a copy)."""
        return None if self._params is None else dict(self._params)

    # ------------------------------------------------------------
    #  Signature and requirements (pointwise scale: delegate)
    # ------------------------------------------------------------
    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Resolve the target's codomain (the scale is pointwise)."""
        if len(domains) == 1:
            return resolve_codomain(self._target, domains[0])
        return self._target.codomain(*domains)

    def requirements(self, domain: SpaceLike) -> OperatorRequirements:
        """Return the target's requirements (pointwise scale)."""
        return self._target.requirements(domain)

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — capability declined here
        space: SpaceLike,  # noqa: ARG002 — capability declined here
    ) -> object:
        """Raise: a metric field coefficient has no symbol."""
        raise EigenbasisError(
            "a metric coefficient breaks translation invariance; "
            "the scaled operator has no symbol")

    # ------------------------------------------------------------
    #  Application
    # ------------------------------------------------------------
    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Apply the target, then scale by the derived coefficient.

        Description
        -----------
        The coefficient is ``grid.metric(codomain, numerator)``
        (divided by the denominator metric when declared; the pure
        reciprocal when only a denominator is named), derived fresh
        at every application — never cached (rules 3.8). Halo
        tracers pass the target's traced result through unscaled:
        the metric scale is pointwise (halo-neutral) and trace-time
        grids expose no ``metric`` accessor, exactly like the
        generic interception hook skips kernels.

        Parameters
        ----------
        f : FieldLike
            The operand field (or halo tracer).

        Returns
        -------
        FieldLike
            The scaled result on the target's codomain.
        """
        out = self._target(f)
        if getattr(f, "_trace_apply", None) is not None:
            return out

        def metric(name: str) -> FieldLike:
            return f.grid.metric(out.function_space, name,
                                 params=self._params)

        # These reciprocal divides (H4/H5, plan §4) are a masked
        # singularity: on a bounded (walled/immersed) axis the metric
        # denominator ``sqrt_g`` is an exact zero in the never-valid
        # padding, so the reverse VJP (``-num/den**2`` at ``den == 0``)
        # is ``0 * inf = NaN``. The exposure condition IS reachable — a
        # ``jax.grad`` w.r.t. an initial-condition field on a walled
        # chart model (e.g. shallow water on the lat-lon sphere) NaNs
        # here. The seal is the same double-``jnp.where`` as the coriolis
        # metric divides (``model/modules/coriolis._safe_metric_divide``)
        # but is left off pending the owner cost decision (this divide is
        # in the every-step pressure solve; guard §4 H4/H5 / owner D4).
        if self._numerator is None:
            return out / metric(self._denominator)
        coeff = metric(self._numerator)
        if self._denominator is not None:
            coeff = coeff / metric(self._denominator)
        return out * coeff


@final
class MappedDerivative(Operator):

    """
    The constant-physical-coordinate derivative builder.

    Description
    -----------
    Seeded kind-only by the grid when its ``CoordinateMapping``
    carries a single-base analytic map (a mapped column). Like the
    grad/div builders it expands at application against the operand
    space and ``f.grid.dispatch``, so module overrides of the inner
    ``"diff"``/``"interpolate"`` rows propagate; ``expand`` is the
    public seam for tests and the halo trace. Per bound coordinate
    ``q`` the expansion is:

    - ``q`` uncorrected by the mapping: the plain registered
      ``diff`` (the physical and computational derivatives agree);
    - ``q`` the mapped column's base (``sigma``): the derivative
      along the *physical* mapped coordinate,
      ``d<base>_d<mapped> * diff[q]``;
    - ``q`` coupled through parameters (``x``): the sketch-4.4 sum
      ``diff[q] - (d<mapped>_d<q> / d<mapped>_d<base>) *
      (interp @ diff[base])``, the correction interpolated onto the
      main term's codomain so the sum shares one signature.

    The expanded object is ordinary operator algebra
    (``OperatorSum`` / ``ScaledOperator`` / composites), so halo
    accounting flows through the existing tracer machinery term by
    term with no special cases.

    Parameters
    ----------
    corrections : Mapping[str, tuple[str, str]]
        Coordinate name -> (mapped physical name, base coordinate),
        the coupling table derived by the mapping.
    axis : str | None, optional
        The bound coordinate; bind via ``op[axis]`` (default:
        None).
    params : Mapping[str, ScalarField] | None, optional
        Dynamic mapping-parameter fields threaded into the expanded
        ``MetricScaled`` coefficients (stage C4); bind via
        :meth:`with_params`. The grid-seeded registry row carries
        None — static defaults — so callers without a dynamic
        geometry get exactly the C1 behavior (default: None).
    """

    dispatch_kind: ClassVar[str | None] = "physical_diff"

    def __init__(
        self,
        corrections: Mapping[str, tuple[str, str]],
        axis: str | None = None,
        params: Mapping[str, ScalarField] | None = None,
    ) -> None:
        """Store the coupling table and the optional bound axis."""
        corrections = dict(corrections)
        for coord, entry in corrections.items():
            if (not isinstance(coord, str)
                    or not isinstance(entry, tuple)
                    or len(entry) != 2  # noqa: PLR2004
                    or not all(isinstance(n, str) for n in entry)):
                raise TypeError(
                    "corrections map coordinate names to (mapped, "
                    f"base) name pairs, got {coord!r}: {entry!r}")
        self._corrections: dict[str, tuple[str, str]] = corrections
        self._axis: str | None = axis
        self._params: dict[str, ScalarField] | None = (
            None if params is None else dict(params))

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def corrections(self) -> dict[str, tuple[str, str]]:
        """Coordinate -> (mapped, base) coupling table (a copy)."""
        return dict(self._corrections)

    @property
    def bound_axis(self) -> str | None:
        """The bound coordinate name, or None."""
        return self._axis

    @property
    def params(self) -> dict[str, ScalarField] | None:
        """Bound dynamic parameter fields, or None (a copy)."""
        return None if self._params is None else dict(self._params)

    # ------------------------------------------------------------
    #  Axis and parameter binding
    # ------------------------------------------------------------
    def with_params(
        self, params: Mapping[str, ScalarField] | None,
    ) -> MappedDerivative:
        """
        Bind dynamic mapping-parameter fields (stage C4).

        Description
        -----------
        Returns a **transient** builder whose expansion threads
        ``params`` into every ``MetricScaled`` coefficient (and the
        ``grid.metric`` derivations behind them), so the physical
        derivative reads the *current* geometry instead of the
        static declaration defaults. The caller — typically a
        tendency term holding module-owned geometry state — builds
        it at application time and applies it immediately; nothing
        params-bound is ever registered (registries hold no
        arrays).

        Parameters
        ----------
        params : Mapping[str, ScalarField] | None
            Parameter fields by name; None/empty returns a builder
            on the static defaults.

        Returns
        -------
        MappedDerivative
            The params-bound builder (axis binding preserved).
        """
        if not params:
            params = None
        return MappedDerivative(self._corrections, axis=self._axis,
                                params=params)

    def __getitem__(self, axis: str) -> MappedDerivative:
        """
        Bind the coordinate to differentiate along.

        Description
        -----------
        Binding is by **base-coordinate** name: coordinates the
        mapping does not couple give the plain derivative, the
        column's base coordinate gives the derivative along its
        physical image (``physical_diff["sigma"]`` is d/dz).

        Parameters
        ----------
        axis : str
            The coordinate name to bind.

        Returns
        -------
        MappedDerivative
            The bound builder.
        """
        if not isinstance(axis, str):
            raise TypeError(
                f"axes are bound by coordinate name, got {axis!r}")
        if self._axis is not None:
            raise TypeError(
                f"operator is already bound to {self._axis!r}; "
                "rebinding is not allowed")
        return MappedDerivative(self._corrections, axis=axis,
                                params=self._params)

    # ------------------------------------------------------------
    #  Signature (builders expand before they have one)
    # ------------------------------------------------------------
    def codomain(
        self,
        *domains: SpaceLike,  # noqa: ARG002 — always raises
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Unexpanded builders have no signature: raise."""
        raise DispatchError(
            "the 'physical_diff' builder expands against a grid's "
            "registry; call expand(domain, grid) (or a registry) "
            "or apply it to a field")

    # ------------------------------------------------------------
    #  Expansion and application
    # ------------------------------------------------------------
    def expand(
        self,
        domain: SpaceLike,
        registry: OperatorRegistry | object,
    ) -> Operator:
        """
        Expand into the concrete sketch-4.4 algebra.

        Parameters
        ----------
        domain : SpaceLike
            The bare operand space.
        registry : OperatorRegistry | object
            The dispatch registry resolving the inner entries, or
            any object carrying a ``dispatch`` attribute (a grid).

        Returns
        -------
        Operator
            The expanded operator (plain ``diff`` where the
            mapping does not couple the axis).
        """
        registry = getattr(registry, "dispatch", registry)
        axis = self._resolved_axis(domain)
        fd_q = registry.resolve(
            "diff", domain.factor(axis))[axis]
        correction = self._corrections.get(axis)
        if correction is None:
            return fd_q
        mapped, base = correction
        if axis == base:
            # the derivative along the column's physical image:
            # d/d<mapped> = d<base>_d<mapped> * d/d<base>
            return MetricScaled(
                fd_q, numerator=f"d{base}_d{mapped}",
                params=self._params)
        target = resolve_codomain(fd_q, domain)
        fd_b = registry.resolve(
            "diff", domain.factor(base))[base]
        chain: Operator = fd_b
        mid = resolve_codomain(fd_b, domain)
        for name in (base, axis):
            src = mid.factor(name)
            dst = target.factor(name)
            if src is dst:
                continue
            interp = registry.resolve("interpolate", src)[name]
            chain = interp @ chain
            mid = resolve_codomain(interp, mid)
        if mid is not target:
            raise SpaceMismatchError(
                "the physical-derivative correction lands on "
                f"{mid!r}, not the main term's codomain "
                f"{target!r}; register interpolate rows joining "
                "them", left=mid, right=target,
                operation="physical_diff")
        return fd_q - MetricScaled(
            chain,
            numerator=f"d{mapped}_d{axis}",
            denominator=f"d{mapped}_d{base}",
            params=self._params)

    def _resolved_axis(self, domain: SpaceLike) -> str:
        """Return the bound axis, inferred when the domain is 1D."""
        if self._axis is not None:
            return self._axis
        axes = domain.active_axis_names
        if len(axes) != 1:
            raise ValueError(
                "cannot pick an axis for 'physical_diff' on "
                f"{domain!r}; bind explicitly via op[axis]")
        return axes[0]

    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Expand against the operand's grid and apply.

        Parameters
        ----------
        f : FieldLike
            The operand field (or halo tracer).

        Returns
        -------
        FieldLike
            The expanded algebra's result.
        """
        return self.expand(f.function_space.bare,
                           f.grid.dispatch)(f)
