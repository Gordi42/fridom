r"""
Composed vector calculus: the grad/div/curl/laplacian builders.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_composed.md``.
The factories are algebra-derived builders (D1/B1/B2), not bespoke
classes: registered under the generic kinds ``"grad"`` / ``"div"`` /
``"curl"`` / ``"laplacian"``, they expand against a domain's axis
family into a ``BlockMatrix`` whose entries are per-axis ``"diff"``
resolutions from the dispatch registry (so module overrides
propagate), and ``Laplacian`` is the block matmul ``div @ grad``
collapsing to the ``OperatorSum`` of per-axis second-derivative
chains — div∘grad = laplacian holds by construction.

Iteration-1 note: ``operators.base.Block`` is a Wave-2 stub owned by
another cluster (read-only here), so the block subset the factories
need — rectangular entries with ``Zero`` structural zeros, tuple
codomains, row-sum application to ``ScalarField``/``VectorField``
operands, and block matmul — lives here as ``BlockMatrix``. Builders
expand at application against ``f.grid.dispatch`` (the model-assembly
merge moment is Phase 2); ``expand`` is public for tests and the
halo-accounting trace.

Metric-aware vector calculus (coordinate-systems plan, stage C2):
on a chart-coupled grid (a ``CoordinateMapping`` embedding chart,
CS-D1) the *same kinds* hold the metric-aware builders seeded by the
grid — registered entries, not compositions the modules hand-build
(rules 3.4, validation 6.3):

- ``MetricGradient``: scalar -> **covariant** components
  :math:`(\partial_i f)` — the registered staggered ``"diff"`` rows
  per axis, variance-tagged;
- ``MetricDivergence``: **contravariant** components -> scalar in
  flux form,
  :math:`\nabla\!\cdot u = \tfrac{1}{\sqrt{g}}\,
  \partial_i(\sqrt{g}\, u^i)`,
  with :math:`\sqrt{g}` entering as :class:`MetricScaled`
  coefficients derived on the component (flux-point) spaces and the
  outer :math:`1/\sqrt{g}` on the scalar codomain — so the global
  ``integrate`` (which weighs by the same :math:`\sqrt{g}`)
  telescopes the flux differences to zero exactly on closed
  (periodic) charts;
- ``MetricCurl`` (2D): the scalar vorticity
  :math:`\tfrac{1}{\sqrt{g}}(\partial_u v_{cov} -
  \partial_v u_{cov})` from covariant components;
- ``MetricLaplacian`` (Laplace-Beltrami): the honest composition
  ``div ∘ raise ∘ grad`` resolved *through the kinds*, so module
  overrides of any leg propagate;
- ``RaiseIndex`` / ``LowerIndex``: explicit metric-consuming index
  maps contracting components with ``inv_g_<u><v>`` / ``g_<u><v>``
  from ``grid.metric`` at application, retagging variance.

Staggering/variance convention (recorded per the C2 task): covariant
and contravariant components share the C-grid face staggering (the
u-component on ``Right(u) ⊗ Center(v)``, ...); the variance tag alone
distinguishes them. Diagonal raise/lower terms are pointwise scalings
on the component's own space; off-diagonal (cross) terms interpolate
the source component onto the **target component's** space through
the registered per-axis ``"interpolate"`` rows and derive the metric
coefficient there. All metric coefficients are derived at
application (never cached, rules 2.3/3.8); with the 2nd-order
staggered ``"diff"`` family this keeps the discrete
``div`` and ``grad`` negative adjoints under the
:math:`\sqrt{g}`-weighted inner product (the mimetic/self-adjoint
Laplace-Beltrami story).
"""
# Wave 3: BlockMatrix (it-1 subset), Gradient, Divergence, Curl,
#    Laplacian -- Stage C2: MetricGradient, MetricDivergence,
#    MetricCurl, MetricLaplacian, RaiseIndex, LowerIndex
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal, final

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.operators.base import (
    FieldLike,
    Identity,
    Operator,
    OperatorRequirements,
    ScaledOperator,
    Zero,
    resolve_codomain,
)
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.mapped import MetricScaled
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.scalars import Variance

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.spatial.operators.registry import (
        OperatorRegistry,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

_CURL_2D = 2
_CURL_3D = 3


def _entry_chain(outer: Operator, inner: Operator) -> Operator:
    """
    Compose two block entries through the operator algebra.

    Description
    -----------
    Plain ``@`` per the class doc (B1): same-axis bound entries
    (the grad/div/laplacian case) yield the per-axis
    ``SeparableComposite``, whose summed halo the chain consumes
    kernel by kernel without intermediate syncs; anything else
    falls back to the whole-space ``Composite`` the algebra builds.

    Parameters
    ----------
    outer : Operator
        The entry applied last.
    inner : Operator
        The entry applied first.

    Returns
    -------
    Operator
        The chain (``Zero`` when either entry is a structural zero).
    """
    if isinstance(outer, Zero) or isinstance(inner, Zero):
        return Zero()
    return outer @ inner


def _bindable_names(space: SpaceLike) -> tuple[str, ...]:
    """Coordinate names of the non-``ConstantSpace`` factors."""
    return space.active_axis_names


# ================================================================
#  BlockMatrix — the iteration-1 block subset the factories need
# ================================================================
@final
class BlockMatrix(Operator):

    """
    Rectangular block of unary operators with tuple signatures.

    Description
    -----------
    The iteration-1 subset of the designed-for ``base.Block`` (a
    Wave-2 stub, read-only for this cluster): rows map operands to
    outputs, ``Zero`` entries are structural zeros, every non-zero
    entry of a row must resolve the identical codomain on its
    column's domain (the row is an operator sum). Application takes
    a ``ScalarField`` (one column) or a ``VectorField`` (components
    in declaration order, matched to columns positionally); one row
    returns a ``ScalarField``, several rows a ``VectorField`` keyed
    by ``output_names``. ``A @ B`` of two blocks is the block
    matmul; entries compose and sum through the operator algebra.

    Parameters
    ----------
    rows : tuple[tuple[Operator, ...], ...]
        The rectangular entry matrix (outputs x operands).
    output_names : tuple[str, ...] | None, optional
        Component names of a multi-row output; required when the
        block has more than one row (default: None).
    """

    def __init__(
        self,
        rows: tuple[tuple[Operator, ...], ...],
        output_names: tuple[str, ...] | None = None,
    ) -> None:
        """Validate rectangularity and naming; store the entries."""
        rows = tuple(tuple(row) for row in rows)
        if not rows or not rows[0]:
            raise ValueError("a block needs at least one entry")
        n_cols = len(rows[0])
        if any(len(row) != n_cols for row in rows):
            raise ValueError("block rows must have equal lengths")
        for row in rows:
            for entry in row:
                if not isinstance(entry, Operator):
                    raise TypeError(
                        f"block entries are operators, got {entry!r}")
            if all(isinstance(entry, Zero) for entry in row):
                raise ValueError(
                    "a block row of structural zeros has no "
                    "codomain; drop the row instead")
        if len(rows) > 1:
            if (output_names is None
                    or len(output_names) != len(rows)):
                raise ValueError(
                    "a multi-row block needs one output name per "
                    "row (they key the resulting VectorField)")
            output_names = tuple(output_names)
        self._rows: tuple[tuple[Operator, ...], ...] = rows
        self._output_names: tuple[str, ...] | None = output_names

    # ------------------------------------------------------------
    #  Structure
    # ------------------------------------------------------------
    @property
    def rows(self) -> tuple[tuple[Operator, ...], ...]:
        """The entry matrix, outputs x operands."""
        return self._rows

    @property
    def output_names(self) -> tuple[str, ...] | None:
        """Component names keying a multi-row output."""
        return self._output_names

    def scalar(self) -> Operator:
        """
        Collapse a 1x1 block to its single scalar-signature entry.

        Description
        -----------
        A ``Div @ Diag @ Grad`` Laplacian block-composes to a 1x1
        block holding the scalar Laplacian operator; ``scalar()``
        recovers that operator (the elegant spelling of
        ``block.rows[0][0]``). It is deliberately *not* an automatic
        ``@`` collapse — only an explicit request unwraps the block.

        Returns
        -------
        Operator
            The block's sole entry.

        Raises
        ------
        SpaceMismatchError
            If the block is not exactly one row by one column.
        """
        if len(self._rows) == 1 and len(self._rows[0]) == 1:
            return self._rows[0][0]
        raise SpaceMismatchError(
            f"scalar() collapses a 1x1 block, but this block is "
            f"{len(self._rows)}x{len(self._rows[0])}",
            operation="scalar")

    # ------------------------------------------------------------
    #  Signature and requirements
    # ------------------------------------------------------------
    def _row_codomains(
        self, *domains: SpaceLike,
    ) -> list[SpaceLike]:
        """Resolve one shared codomain per block row (validated)."""
        if len(domains) != len(self._rows[0]):
            raise SpaceMismatchError(
                f"block of {len(self._rows[0])} columns applied to "
                f"{len(domains)} operand spaces",
                operation="block")
        outputs = []
        for row in self._rows:
            row_codomains = [
                resolve_codomain(entry, domain)
                for entry, domain in zip(row, domains, strict=True)
                if not isinstance(entry, Zero)]
            first = row_codomains[0]
            if any(space is not first for space in row_codomains):
                raise SpaceMismatchError(
                    "the entries of a block row must share a "
                    f"codomain; resolved {row_codomains!r}",
                    operation="block")
            outputs.append(first)
        return outputs

    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """
        Resolve the per-row codomains (tuple signature).

        Parameters
        ----------
        *domains : SpaceLike
            One bare domain space per block column.

        Returns
        -------
        SpaceLike | tuple[SpaceLike, ...]
            The row codomains (a bare space for one row).
        """
        outputs = self._row_codomains(*domains)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def codomains(self, *domains: SpaceLike) -> tuple[SpaceLike, ...]:
        """
        Resolve the per-row codomains, always as a tuple.

        Description
        -----------
        The always-tuple override of
        :meth:`~fridom.spatial.operators.base.Operator.codomains`:
        returns the true per-row codomain tuple directly (a single row
        yields a length-1 tuple, not the bare space
        :meth:`codomain` returns), so block call sites skip the
        ``isinstance(x, tuple)`` normalization.

        Parameters
        ----------
        *domains : SpaceLike
            One bare domain space per block column.

        Returns
        -------
        tuple[SpaceLike, ...]
            The per-row codomains.
        """
        return tuple(self._row_codomains(*domains))

    def requirements(self, domain: SpaceLike) -> OperatorRequirements:
        """
        Per-factor requirements: max over the entries.

        Description
        -----------
        Iteration-1 simplification of the per-block-row halo of
        rules section 3.6 (sound: parallel branches max).

        Parameters
        ----------
        domain : SpaceLike
            The factor space the block is applied on.

        Returns
        -------
        OperatorRequirements
            The combined per-factor requirements record.
        """
        lo = hi = 0
        collective = False
        for row in self._rows:
            for entry in row:
                if isinstance(entry, Zero):
                    continue
                req = entry.requirements(domain)
                lo = max(lo, req.reach[0])
                hi = max(hi, req.reach[1])
                collective = collective or req.collective
        return OperatorRequirements(
            reach=(lo, hi), collective=collective)

    # ------------------------------------------------------------
    #  Block matmul
    # ------------------------------------------------------------
    def __matmul__(
        self, other: Operator | tuple[Operator, ...],
    ) -> Operator:
        """
        Compose; two blocks perform the block matmul.

        Parameters
        ----------
        other : Operator | tuple[Operator, ...]
            The inner operator (applied first).

        Returns
        -------
        Operator
            The block product, or the base-algebra composite.
        """
        if not isinstance(other, BlockMatrix):
            return super().__matmul__(other)
        n_inner = len(self._rows[0])
        if len(other.rows) != n_inner:
            raise SpaceMismatchError(
                f"block matmul shape mismatch: {n_inner} columns "
                f"vs {len(other.rows)} rows", operation="@")
        other_rows = other.rows
        rows = []
        for row in self._rows:
            new_row = []
            for k in range(len(other_rows[0])):
                acc: Operator | None = None
                for j in range(n_inner):
                    term = _entry_chain(row[j], other_rows[j][k])
                    if isinstance(term, Zero):
                        continue
                    acc = term if acc is None else acc + term
                new_row.append(Zero() if acc is None else acc)
            rows.append(tuple(new_row))
        return BlockMatrix(tuple(rows),
                           output_names=self._output_names)

    # ------------------------------------------------------------
    #  Application
    # ------------------------------------------------------------
    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Apply the block: row sums of entry applications.

        Parameters
        ----------
        f : FieldLike
            A ``ScalarField`` (one column) or a ``VectorField``
            whose components match the columns positionally.

        Returns
        -------
        FieldLike
            A ``ScalarField`` (one row) or a ``VectorField`` keyed
            by ``output_names``.
        """
        n_cols = len(self._rows[0])
        operands = tuple(f) if isinstance(f, VectorField) else (f,)
        if len(operands) != n_cols:
            raise ValueError(
                f"block of {n_cols} columns applied to "
                f"{len(operands)} operand component(s)")
        outputs = []
        for row in self._rows:
            acc = None
            for entry, operand in zip(row, operands, strict=True):
                if isinstance(entry, Zero):
                    continue
                term = entry(operand)
                acc = term if acc is None else acc + term
            outputs.append(acc)
        if len(outputs) == 1:
            return outputs[0]
        return VectorField(dict(zip(self._output_names, outputs,
                                    strict=True)))


# ================================================================
#  Expansion helpers (shared by the builders)
# ================================================================
def _diff_entry(
    domain: SpaceLike,
    axis: str,
    order: int | None,
    registry: OperatorRegistry,
) -> Operator:
    """Resolve the per-axis derivative entry, bound to ``axis``."""
    if order is not None:
        return FiniteDifference(order)[axis]
    return registry.resolve("diff", domain.factor(axis))[axis]


def _expand_grad(
    domain: SpaceLike,
    order: int | None,
    registry: OperatorRegistry,
) -> BlockMatrix:
    """Expand the column block ``[[d_x], [d_y], ...]`` on a scalar."""
    axes = _bindable_names(domain)
    if not axes:
        raise SpaceMismatchError(
            f"no bindable factor on {domain!r}; every factor is a "
            "ConstantSpace", left=domain, operation="grad")
    rows = tuple(
        (_diff_entry(domain, axis, order, registry),)
        for axis in axes)
    names = axes if len(axes) > 1 else None
    return BlockMatrix(rows, output_names=names)


def _component_axes(
    domains: tuple[SpaceLike, ...], operation: str,
) -> tuple[str, ...]:
    """Match vector components to axes positionally, validated."""
    axes = _bindable_names(domains[0])
    if any(_bindable_names(domain) != axes for domain in domains[1:]):
        raise SpaceMismatchError(
            "the components of a vector operand must share one "
            "bindable axis family, got "
            f"{[_bindable_names(d) for d in domains]!r}",
            operation=operation)
    if len(domains) != len(axes):
        raise SpaceMismatchError(
            f"{operation} expects one component per axis "
            f"{axes}, got {len(domains)} component(s)",
            operation=operation)
    return axes


def _expand_div(
    domains: tuple[SpaceLike, ...],
    order: int | None,
    registry: OperatorRegistry,
) -> BlockMatrix:
    """Expand the row block ``[[d_x, d_y, ...]]`` on a vector."""
    axes = _component_axes(domains, "div")
    row = tuple(
        _diff_entry(domain, axis, order, registry)
        for domain, axis in zip(domains, axes, strict=True))
    return BlockMatrix((row,))


def _expand_curl(
    domains: tuple[SpaceLike, ...],
    order: int | None,
    registry: OperatorRegistry,
) -> BlockMatrix:
    """Expand the curl block (2-D scalar curl or 3-D matrix)."""
    axes = _component_axes(domains, "curl")

    def entry(i: int, j: int, sign: float) -> Operator:
        op = _diff_entry(domains[j], axes[i], order, registry)
        return op if sign == 1.0 else sign * op

    if len(axes) == _CURL_2D:
        # scalar curl: dv/dx - du/dy
        return BlockMatrix(((entry(1, 0, -1.0), entry(0, 1, 1.0)),))
    if len(axes) == _CURL_3D:
        zero = Zero()
        rows = (
            (zero, entry(2, 1, -1.0), entry(1, 2, 1.0)),
            (entry(2, 0, 1.0), zero, entry(0, 2, -1.0)),
            (entry(1, 0, -1.0), entry(0, 1, 1.0), zero),
        )
        return BlockMatrix(rows, output_names=axes)
    raise SpaceMismatchError(
        f"curl is defined for 2 or 3 axes, got {axes}",
        operation="curl")


# ================================================================
#  Builders
# ================================================================
class _VectorCalculusBuilder(Operator):

    """
    Assembly-resolved builder behind the factory names (B2).

    Description
    -----------
    The registered default for its kind: at application it expands
    against the operand's axis family and ``f.grid.dispatch`` into a
    ``BlockMatrix`` of per-axis ``("diff", factor)`` resolutions
    (late-bound, so module overrides propagate) and applies it. The
    expansion is computed on demand, never materialized as a
    rewritten operator; ``expand`` is the public seam for tests and
    the halo trace. Like ``Dispatched``, an unexpanded builder has
    no signature.
    """

    _kind: Literal["grad", "div", "curl", "laplacian"]

    def __init__(
        self,
        order: int | None,
        metric: Mapping[str, complex | jax.Array] | None = None,
    ) -> None:
        """Store the optional pinned FD order and metric weights."""
        if order is not None:
            FiniteDifference(order)  # validates even, >= 2
        self._order: int | None = order
        self._metric: Mapping[str, complex | jax.Array] | None = metric

    @property
    def order(self) -> int | None:
        """Pinned FD order, or None for dispatched entries."""
        return self._order

    @property
    def metric(self) -> Mapping[str, complex | jax.Array] | None:
        """Per-axis metric weights of a weighted Laplacian (or None)."""
        return self._metric

    def codomain(
        self,
        *domains: SpaceLike,  # noqa: ARG002 — always raises
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Unexpanded builders have no signature: raise."""
        raise DispatchError(
            f"the {self._kind!r} builder expands against a grid's "
            "registry; call expand(domains, grid) (or a registry) or "
            "apply it to a field")

    def expand(
        self,
        domains: SpaceLike | tuple[SpaceLike, ...],
        registry: OperatorRegistry | object,
    ) -> Operator:
        """
        Expand into the concrete block over the axis family.

        Parameters
        ----------
        domains : SpaceLike | tuple[SpaceLike, ...]
            The bare operand space (scalar kinds) or the tuple of
            component spaces (vector kinds).
        registry : OperatorRegistry | object
            The dispatch registry resolving the per-axis entries, or a
            grid: any object carrying a ``dispatch`` attribute is used
            through ``obj.dispatch``, so ``expand(domains, grid)`` and
            ``expand(domains, grid.dispatch)`` are interchangeable.

        Returns
        -------
        Operator
            The expanded ``BlockMatrix``.
        """
        registry = getattr(registry, "dispatch", registry)
        order = self._order
        if self._kind in {"div", "curl"} and not isinstance(
                domains, tuple):
            raise TypeError(
                f"{self._kind} expands over a tuple of component "
                f"spaces, got {domains!r}")
        if self._kind == "grad":
            return _expand_grad(domains, order, registry)
        if self._kind == "div":
            return _expand_div(domains, order, registry)
        if self._kind == "curl":
            return _expand_curl(domains, order, registry)
        # laplacian: div @ grad as a block matmul (1xn @ nx1), with an
        # optional diagonal metric between them: div @ Diag @ grad
        grad = _expand_grad(domains, order, registry)
        mid = grad.codomains(domains)
        div = _expand_div(mid, order, registry)
        if self._metric is not None:
            axes = _bindable_names(mid[0])
            return div @ Diag(self._metric, axes=axes) @ grad
        return div @ grad

    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Expand against the operand's grid and apply.

        Parameters
        ----------
        f : FieldLike
            A ``ScalarField`` (grad/laplacian) or ``VectorField``
            (div/curl) operand.

        Returns
        -------
        FieldLike
            The expanded block's result.
        """
        if self._kind in {"grad", "laplacian"}:
            if isinstance(f, VectorField):
                raise TypeError(
                    f"{self._kind} applies to a scalar field; got a "
                    "VectorField")
            domains: (SpaceLike | tuple[SpaceLike, ...]) = (
                f.function_space.bare)
            grid = f.grid
        else:
            if not isinstance(f, VectorField):
                raise TypeError(
                    f"{self._kind} applies to a VectorField; got "
                    f"{type(f).__name__}")
            domains = tuple(
                component.function_space.bare for component in f)
            grid = f.grid
        return self.expand(domains, grid.dispatch)(f)


@final
class _Gradient(_VectorCalculusBuilder):

    """The ``"grad"`` builder (column block over the axes)."""

    dispatch_kind = "grad"
    _kind = "grad"


@final
class _Divergence(_VectorCalculusBuilder):

    """The ``"div"`` builder (row block over the axes)."""

    dispatch_kind = "div"
    _kind = "div"


@final
class _Curl(_VectorCalculusBuilder):

    """The ``"curl"`` builder (matrix block with structural zeros)."""

    dispatch_kind = "curl"
    _kind = "curl"


@final
class _Laplacian(_VectorCalculusBuilder):

    """The ``"laplacian"`` builder (``div @ grad`` block matmul)."""

    dispatch_kind = "laplacian"
    _kind = "laplacian"


def Gradient(order: int | None = None) -> Operator:  # noqa: N802
    """
    Build the ``"grad"`` builder (B1/B2).

    Description
    -----------
    At expansion it becomes the column block ``[[d_x], [d_y], ...]``
    — scalar ``S`` -> the staggered vector — each entry the
    registry's ``("diff", factor)`` resolution (or
    ``FiniteDifference(order)``) bound to its axis.

    Parameters
    ----------
    order : int | None, optional
        Pinned FD order; None resolves the registered per-axis
        ``"diff"`` entries (default: None).

    Returns
    -------
    Operator
        The builder.
    """
    return _Gradient(order)


def Divergence(order: int | None = None) -> Operator:  # noqa: N802
    """
    Build the ``"div"`` builder.

    Description
    -----------
    At expansion it becomes the row block ``[[d_x, d_y, ...]]`` —
    the staggered vector -> the common cell space. The FV C-grid
    ``div`` resolves the average-space ``"diff"`` entries per axis.

    Parameters
    ----------
    order : int | None, optional
        Pinned FD order; None resolves the registered per-axis
        ``"diff"`` entries (default: None).

    Returns
    -------
    Operator
        The builder.
    """
    return _Divergence(order)


def Curl(order: int | None = None) -> Operator:  # noqa: N802
    """
    Build the ``"curl"`` builder.

    Description
    -----------
    At expansion it becomes the matrix block with ``Zero``
    structural zeros: the staggered vector -> its dual-staggered
    curl (3 axes) or the scalar curl ``dv/dx - du/dy`` (2 axes).

    Parameters
    ----------
    order : int | None, optional
        Pinned FD order; None resolves the registered per-axis
        ``"diff"`` entries (default: None).

    Returns
    -------
    Operator
        The builder.
    """
    return _Curl(order)


def Laplacian(  # noqa: N802
    order: int | None = None,
    metric: Mapping[str, complex | jax.Array] | None = None,
) -> Operator:
    r"""
    Build the ``"laplacian"`` builder: ``div @ grad``.

    Description
    -----------
    At expansion the block matmul ``(1xn) @ (nx1)`` collapses to a
    1x1 block holding the ``OperatorSum`` of per-axis
    second-derivative ``SeparableComposite``s (B1) — so
    div∘grad = laplacian holds by construction.

    With ``metric`` given, the expansion threads a diagonal
    :func:`Diag` block between the factors — ``div @ Diag @ grad`` —
    scaling each gradient component by its axis weight before the
    divergence sums them. This is the elegant pressure Laplacian
    :math:`\nabla\cdot\mathrm{diag}(w)\nabla` (e.g.
    ``metric={"z": 1/dsqr}`` for the non-hydrostatic vertical scaling);
    weights may be Python numbers or 0-d ``jax.Array`` scalars.
    ``Laplacian()`` (no metric) is unchanged.

    Parameters
    ----------
    order : int | None, optional
        Pinned FD order; None resolves the registered per-axis
        ``"diff"`` entries (default: None).
    metric : Mapping[str, complex | jax.Array] | None, optional
        Per-axis diagonal weights inserted between ``div`` and
        ``grad``; axes absent from the mapping take weight ``1``
        (default: None, the unweighted ``div @ grad``).

    Returns
    -------
    Operator
        The builder.
    """
    return _Laplacian(order, metric)


def _is_unit(coeff: complex | jax.Array) -> bool:
    """Whether ``coeff`` is exactly the scalar ``1`` (the neutral)."""
    return isinstance(coeff, int | float | complex) and coeff == 1.0


def Diag(  # noqa: N802
    coeffs: Mapping[str, complex | jax.Array],
    *,
    axes: tuple[str, ...],
    default: complex | jax.Array = 1.0,
) -> BlockMatrix:
    r"""
    Build a diagonal metric block keyed by axis name.

    Description
    -----------
    The per-component weight of the elegant pressure Laplacian
    ``Div @ Diag(...) @ Grad`` (symbol_stack_design.md decision 2): a
    diagonal ``n x n`` :class:`BlockMatrix` over a component vector,
    one entry per axis in ``axes``. The diagonal entry for axis ``a``
    is :class:`~fridom.spatial.operators.base.Identity` when
    ``coeffs.get(a, default)`` is exactly the scalar ``1`` (the
    structural neutral — its ``codomain`` is the identity, so the
    diagonal threads each component's space through unchanged and
    ``Div @ Diag @ Grad`` block-composes), else the scaled identity
    ``coeff * Identity``; off-diagonal entries are ``Zero``.

    A coefficient may be a Python number or a 0-d ``jax.Array`` (a
    traced-but-constant scalar such as ``1/dsqr``); the scaled-identity
    entry carries a symbol through
    :meth:`~fridom.spatial.operators.base.ScaledOperator.eigenvalues`.

    Parameters
    ----------
    coeffs : Mapping[str, complex | jax.Array]
        Per-axis diagonal weights; axes absent from the mapping take
        ``default``.
    axes : tuple[str, ...]
        The component axis family, in order (the operand's bindable
        names); it fixes the block's row/column layout.
    default : complex | jax.Array, optional
        The weight for axes absent from ``coeffs`` (default: 1.0).

    Returns
    -------
    BlockMatrix
        The diagonal metric block (vector -> vector).
    """
    if not axes:
        raise SpaceMismatchError(
            "Diag needs at least one component axis", operation="diag")
    ident = Identity()
    zero = Zero()

    def diagonal(axis: str) -> Operator:
        coeff = coeffs.get(axis, default)
        return ident if _is_unit(coeff) else ScaledOperator(coeff, ident)

    rows = tuple(
        tuple(diagonal(ai) if i == j else zero
              for j in range(len(axes)))
        for i, ai in enumerate(axes))
    names = axes if len(axes) > 1 else None
    return BlockMatrix(rows, output_names=names)


# ================================================================
#  Metric-aware vector calculus (coordinate-systems plan, C2)
# ================================================================
@final
class VarianceRetag(Operator):

    """
    Pointwise variance retag of a unary target's output.

    Description
    -----------
    The explicit variance transition of the metric-aware entries: a
    variance tag is a pure claim (module docstring), so retagging
    touches neither data nor ghost validity — only the interned
    space identity. Requirements delegate to the target (the retag
    is pointwise/halo-neutral); halo tracers pass the target's
    traced result through unretagged, exactly like
    :class:`~fridom.spatial.operators.mapped.MetricScaled` passes
    its scale (the trace only accounts widths).

    Parameters
    ----------
    target : Operator
        The operator whose output is retagged.
    variance : Variance | None
        The variance the output claims; None strips the tag.
    """

    def __init__(self, target: Operator,
                 variance: Variance | None) -> None:
        """Store the target and the claimed variance."""
        if not isinstance(target, Operator):
            raise TypeError(
                f"target must be an Operator, got {target!r}")
        if variance is not None and not isinstance(variance,
                                                   Variance):
            raise TypeError(
                f"variance must be a Variance member or None, got "
                f"{variance!r}")
        self._target: Operator = target
        self._variance: Variance | None = variance

    @property
    def target(self) -> Operator:
        """The retagged operator (static structure)."""
        return self._target

    @property
    def variance(self) -> Variance | None:
        """The variance the output claims (None strips)."""
        return self._variance

    def codomain(
        self, *domains: SpaceLike,
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Resolve the target's codomain, variance-retagged."""
        if len(domains) != 1:
            raise SpaceMismatchError(
                "VarianceRetag wraps unary targets only",
                operation="variance_retag")
        codomain = resolve_codomain(self._target, domains[0])
        return codomain.with_variance(self._variance)

    def requirements(self, domain: SpaceLike) -> OperatorRequirements:
        """Return the target's requirements (pointwise retag)."""
        return self._target.requirements(domain)

    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Apply the target, then re-claim the variance tag.

        Parameters
        ----------
        f : FieldLike
            The operand field (or halo tracer).

        Returns
        -------
        FieldLike
            The result on the variance-retagged codomain.
        """
        out = self._target(f)
        if getattr(f, "_trace_apply", None) is not None:
            return out
        space = out.function_space.with_variance(self._variance)
        if space is out.function_space:
            return out
        return type(out)(
            out.grid, space,
            out._data,  # noqa: SLF001 — plumbing-constructor seam
            out.metadata, halo_valid=out.halo_valid)


def _require_chart_axes(
    axes: tuple[str, ...],
    coords: tuple[str, ...],
    operation: str,
) -> None:
    """Require the operand axis family to be the chart coordinates."""
    if set(axes) == set(coords) and len(axes) == len(coords):
        return
    raise SpaceMismatchError(
        f"the metric-aware {operation!r} entry is registered for "
        f"the chart coordinates {coords}; the operand resolves "
        f"{axes} — fields entering it vary along exactly the chart "
        "coordinates (constant factors elsewhere)",
        operation=operation)


def _require_variance(
    domains: tuple[SpaceLike, ...],
    expected: Variance | None,
    operation: str,
    hint: str,
) -> None:
    """Require every component space to claim ``expected``."""
    for domain in domains:
        if domain.variance is not expected:
            raise SpaceMismatchError(
                f"{operation} on a chart grid consumes "
                f"{'untagged' if expected is None else expected.name}"
                f" component spaces, got {domain!r}; {hint}",
                left=domain, operation=operation)


def _interp_onto(
    domain: SpaceLike,
    target: SpaceLike,
    registry: OperatorRegistry,
    operation: str,
) -> Operator:
    """
    Chain registered interpolations from ``domain`` onto ``target``.

    Description
    -----------
    The cross-component staggering rule of ``RaiseIndex`` /
    ``LowerIndex`` (module docstring): per differing factor the
    registered ``("interpolate", source factor)`` row moves the
    source component one staggering hop, landing on the target
    component's space (the variance tag rides through unchanged).

    Parameters
    ----------
    domain : SpaceLike
        The (bare) source component space.
    target : SpaceLike
        The (bare) target component space.
    registry : OperatorRegistry
        The dispatch registry resolving the interpolate rows.
    operation : str
        The consuming operation, for error messages.

    Returns
    -------
    Operator
        The interpolation chain (``Identity`` for equal spaces).
    """
    chain: Operator = Identity()
    mid = domain
    for name in target.names:
        src = mid.factor(name)
        dst = target.factor(name)
        if src is dst:
            continue
        interp = registry.resolve("interpolate", src)[name]
        chain = interp @ chain
        mid = resolve_codomain(interp, mid)
    if mid is not target:
        raise SpaceMismatchError(
            f"the {operation} cross-component chain lands on "
            f"{mid!r}, not the target component's {target!r}; "
            "register interpolate rows joining them",
            left=mid, right=target, operation=operation)
    return chain


def _expand_metric_grad(
    domain: SpaceLike,
    coords: tuple[str, ...],
    registry: OperatorRegistry,
) -> BlockMatrix:
    """Expand the covariant gradient column over the chart axes."""
    if domain.variance is not None:
        raise SpaceMismatchError(
            f"grad takes an untagged scalar operand, got {domain!r}",
            left=domain, operation="grad")
    axes = _bindable_names(domain)
    _require_chart_axes(axes, coords, "grad")
    rows = tuple(
        (VarianceRetag(
            registry.resolve("diff", domain.factor(axis))[axis],
            Variance.COVARIANT),)
        for axis in axes)
    return BlockMatrix(rows, output_names=axes)


def _expand_metric_div(
    domains: tuple[SpaceLike, ...],
    coords: tuple[str, ...],
    registry: OperatorRegistry,
) -> BlockMatrix:
    """Expand the flux-form divergence row over the chart axes."""
    axes = _component_axes(domains, "div")
    _require_chart_axes(axes, coords, "div")
    _require_variance(
        domains, Variance.CONTRAVARIANT, "div",
        "tag physical components via with_variance or raise the "
        "index (RaiseIndex) first")
    row = []
    for domain, axis in zip(domains, axes, strict=True):
        fd = registry.resolve("diff", domain.factor(axis))[axis]
        flux = fd @ MetricScaled(Identity(), numerator="sqrt_g")
        row.append(VarianceRetag(
            MetricScaled(flux, denominator="sqrt_g"), None))
    return BlockMatrix((tuple(row),))


def _expand_metric_curl(
    domains: tuple[SpaceLike, ...],
    coords: tuple[str, ...],
    registry: OperatorRegistry,
) -> BlockMatrix:
    """Expand the 2D scalar vorticity from covariant components."""
    axes = _component_axes(domains, "curl")
    if len(axes) != _CURL_2D:
        raise SpaceMismatchError(
            "the metric-aware curl is the 2D scalar vorticity "
            f"(1/sqrt_g)(d_u v - d_v u); got axes {axes}",
            operation="curl")
    _require_chart_axes(axes, coords, "curl")
    _require_variance(
        domains, Variance.COVARIANT, "curl",
        "the vorticity contracts covariant components; lower the "
        "index (LowerIndex) first")

    def entry(i: int, j: int, sign: float) -> Operator:
        op = registry.resolve(
            "diff", domains[j].factor(axes[i]))[axes[i]]
        if sign != 1.0:
            op = sign * op
        return VarianceRetag(
            MetricScaled(op, denominator="sqrt_g"), None)

    return BlockMatrix(((entry(1, 0, -1.0), entry(0, 1, 1.0)),))


def _bounded_cross_term_error(
    operation: str,
    coords: tuple[str, ...],
    axis_i: str,
    axis_j: str,
) -> DispatchError:
    """
    Teach the diagonal fix for a bounded chart's missing interp row.

    Description
    -----------
    A non-diagonal index move interpolates the ``axis_j`` component
    onto the ``axis_i`` component's space to contract the
    off-diagonal metric; across a **bounded** chart axis there is no
    such interpolation row (a wall has no legal interpolation), and
    the raw ``DispatchError`` from the registry never names the fix.
    On an orthogonal chart the off-diagonal metric is identically
    zero, so the term is dropped rather than interpolated — declare
    that once and the index move assembles (chart-ergonomics E2).

    Parameters
    ----------
    operation : str
        The consuming kind (``"raise_index"`` / ``"lower_index"``).
    coords : tuple[str, ...]
        The chart's base coordinates.
    axis_i, axis_j : str
        The target and source chart axes of the failing cross term.

    Returns
    -------
    DispatchError
        The taught error to re-raise from the original.
    """
    cls = operation.title().replace("_", "")
    return DispatchError(
        f"{operation} on a bounded chart axis has no interpolation "
        f"row for the {axis_i}<->{axis_j} cross term (a wall has no "
        "legal interpolation). If the chart is orthogonal (its "
        "off-diagonal metric is identically zero — as for the "
        "lat-lon sphere and the torus), declare it and the cross "
        "terms are dropped: pass orthogonal=True to the "
        "CoordinateMapping, or override the kind directly with "
        f"grid.merge_overrides({{{operation!r}: {cls}({coords!r}, "
        "diagonal=True)}}).")


def _expand_index_move(
    domains: tuple[SpaceLike, ...],
    coords: tuple[str, ...],
    registry: OperatorRegistry,
    *,
    prefix: str,
    source: Variance,
    target: Variance,
    diagonal: bool,
    operation: str,
    hint: str,
) -> BlockMatrix:
    """Expand the metric contraction block of raise/lower."""
    axes = _component_axes(domains, operation)
    _require_chart_axes(axes, coords, operation)
    _require_variance(domains, source, operation, hint)
    zero = Zero()
    rows = []
    for i in range(len(axes)):
        row: list[Operator] = []
        for j in range(len(axes)):
            name = f"{prefix}_{axes[i]}{axes[j]}"
            if i == j:
                op: Operator = MetricScaled(Identity(),
                                            numerator=name)
            elif diagonal:
                row.append(zero)
                continue
            else:
                try:
                    chain = _interp_onto(domains[j], domains[i],
                                         registry, operation)
                except DispatchError as exc:
                    raise _bounded_cross_term_error(
                        operation, coords, axes[i], axes[j]) from exc
                op = MetricScaled(chain, numerator=name)
            row.append(VarianceRetag(op, target))
        rows.append(tuple(row))
    return BlockMatrix(tuple(rows), output_names=axes)


class _MetricCalculus(Operator):

    """
    Chart-coupled metric-aware builder (B2 pattern, stage C2).

    Description
    -----------
    The registered default of its kind on chart grids: seeded by the
    grid when its ``CoordinateMapping`` carries an embedding chart
    (mirroring the C1 ``"physical_diff"`` seeding), it expands at
    application against the operand space(s) and ``f.grid.dispatch``
    — so module overrides of the inner ``"diff"`` /
    ``"interpolate"`` rows propagate — and derives every metric
    coefficient via ``grid.metric`` at application time (nothing
    cached, rules 2.3/3.8). ``expand`` is the public seam for tests
    and the halo-accounting trace.

    Parameters
    ----------
    coords : tuple[str, ...]
        The chart's base coordinates (at least two, unique) — the
        axis family the entry expands over and the vocabulary of the
        induced-metric names (``sqrt_g``, ``inv_g_<u><v>``, ...).
    """

    _kind: ClassVar[str]
    #: whether the operand is a scalar field (grad/laplacian)
    _scalar_operand: ClassVar[bool]

    def __init__(self, coords: tuple[str, ...]) -> None:
        """Validate and store the chart coordinate family."""
        coords = tuple(coords)
        if (len(coords) < 2  # noqa: PLR2004 — a chart couples >= 2
                or not all(isinstance(c, str) for c in coords)
                or len(set(coords)) != len(coords)):
            raise TypeError(
                "coords names the chart's base coordinates: at "
                f"least two unique strings, got {coords!r}")
        self._coords: tuple[str, ...] = coords

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def coords(self) -> tuple[str, ...]:
        """The chart coordinate family the entry expands over."""
        return self._coords

    # ------------------------------------------------------------
    #  Signature (builders expand before they have one)
    # ------------------------------------------------------------
    def codomain(
        self,
        *domains: SpaceLike,  # noqa: ARG002 — always raises
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Unexpanded builders have no signature: raise."""
        raise DispatchError(
            f"the metric-aware {self._kind!r} builder expands "
            "against a grid's registry; call expand(domains, grid) "
            "(or a registry) or apply it to a field")

    # ------------------------------------------------------------
    #  Expansion and application
    # ------------------------------------------------------------
    def expand(
        self,
        domains: SpaceLike | tuple[SpaceLike, ...],
        registry: OperatorRegistry | object,
    ) -> Operator:
        """
        Expand into the concrete metric-aware block.

        Parameters
        ----------
        domains : SpaceLike | tuple[SpaceLike, ...]
            The bare operand space (scalar kinds) or the tuple of
            component spaces (vector kinds).
        registry : OperatorRegistry | object
            The dispatch registry resolving the inner rows, or any
            object carrying a ``dispatch`` attribute (a grid).

        Returns
        -------
        Operator
            The expanded metric-aware ``BlockMatrix``.
        """
        registry = getattr(registry, "dispatch", registry)
        if self._scalar_operand:
            if isinstance(domains, tuple):
                raise TypeError(
                    f"{self._kind} expands over one scalar operand "
                    f"space, got {domains!r}")
        elif not isinstance(domains, tuple):
            raise TypeError(
                f"{self._kind} expands over a tuple of component "
                f"spaces, got {domains!r}")
        return self._expand(domains, registry)

    @abstractmethod
    def _expand(
        self,
        domains: SpaceLike | tuple[SpaceLike, ...],
        registry: OperatorRegistry,
    ) -> Operator:
        """
        Kind hook: build the concrete block (subclasses).

        Parameters
        ----------
        domains : SpaceLike | tuple[SpaceLike, ...]
            The arity-validated bare operand space(s).
        registry : OperatorRegistry
            The dispatch registry resolving the inner rows.

        Returns
        -------
        Operator
            The expanded metric-aware block.
        """
        ...

    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Expand against the operand's grid and apply.

        Parameters
        ----------
        f : FieldLike
            A ``ScalarField`` (grad/laplacian) or ``VectorField``
            (div/curl/raise/lower) operand.

        Returns
        -------
        FieldLike
            The expanded block's result.
        """
        if self._scalar_operand:
            if isinstance(f, VectorField):
                raise TypeError(
                    f"{self._kind} applies to a scalar field; got "
                    "a VectorField")
            domains: (SpaceLike | tuple[SpaceLike, ...]) = (
                f.function_space.bare)
        else:
            if not isinstance(f, VectorField):
                raise TypeError(
                    f"{self._kind} applies to a VectorField; got "
                    f"{type(f).__name__}")
            domains = tuple(
                component.function_space.bare for component in f)
        return self.expand(domains, f.grid.dispatch)(f)


@final
class MetricGradient(_MetricCalculus):

    """The chart-grid ``"grad"`` entry: covariant components."""

    dispatch_kind: ClassVar[str | None] = "grad"
    _kind = "grad"
    _scalar_operand = True

    def _expand(
        self,
        domains: SpaceLike,
        registry: OperatorRegistry,
    ) -> Operator:
        """Expand the variance-tagged gradient column."""
        return _expand_metric_grad(domains, self._coords, registry)


@final
class MetricDivergence(_MetricCalculus):

    """The chart-grid ``"div"`` entry: the flux-form divergence."""

    dispatch_kind: ClassVar[str | None] = "div"
    _kind = "div"
    _scalar_operand = False

    def _expand(
        self,
        domains: tuple[SpaceLike, ...],
        registry: OperatorRegistry,
    ) -> Operator:
        """Expand the flux-form divergence row."""
        return _expand_metric_div(domains, self._coords, registry)


@final
class MetricCurl(_MetricCalculus):

    """The chart-grid ``"curl"`` entry: the 2D scalar vorticity."""

    dispatch_kind: ClassVar[str | None] = "curl"
    _kind = "curl"
    _scalar_operand = False

    def _expand(
        self,
        domains: tuple[SpaceLike, ...],
        registry: OperatorRegistry,
    ) -> Operator:
        """Expand the scalar vorticity row."""
        return _expand_metric_curl(domains, self._coords, registry)


@final
class MetricLaplacian(_MetricCalculus):

    """
    The chart-grid ``"laplacian"`` entry: Laplace-Beltrami.

    Description
    -----------
    Registered as the honest composition ``div ∘ raise ∘ grad``
    resolved *through the kinds*: at expansion the ``"grad"``,
    ``"raise_index"``, and ``"div"`` entries are resolved from the
    registry (so overrides of any leg propagate) and block-matmuled
    into one 1x1 block — the discrete Laplace-Beltrami operator,
    self-adjoint under the sqrt(g)-weighted inner product by
    construction (module docstring).
    """

    dispatch_kind: ClassVar[str | None] = "laplacian"
    _kind = "laplacian"
    _scalar_operand = True

    def _expand(
        self,
        domains: SpaceLike,
        registry: OperatorRegistry,
    ) -> Operator:
        """Expand ``div @ raise @ grad`` through the kinds."""
        grad_block = _leg(registry, "grad", domains)
        mid = grad_block.codomains(domains)
        raise_block = _leg(registry, "raise_index", domains, mid)
        raised = raise_block.codomains(*mid)
        div_block = _leg(registry, "div", domains, raised)
        return div_block @ raise_block @ grad_block


def _leg(
    registry: OperatorRegistry,
    kind: str,
    key_space: SpaceLike,
    domains: SpaceLike | tuple[SpaceLike, ...] | None = None,
) -> Operator:
    """
    Resolve and expand one leg of the Laplace-Beltrami chain.

    Parameters
    ----------
    registry : OperatorRegistry
        The dispatch registry (as merged).
    kind : str
        The leg's dispatch kind.
    key_space : SpaceLike
        The space keying the resolution (kind-only rows accept any).
    domains : SpaceLike | tuple[SpaceLike, ...] | None, optional
        The operand space(s) the leg expands over; None uses
        ``key_space`` itself (default: None).

    Returns
    -------
    Operator
        The expanded leg (a ``BlockMatrix`` for the seeded entries).
    """
    op = registry.resolve(kind, key_space)
    expand = getattr(op, "expand", None)
    if expand is None:
        raise DispatchError(
            f"the registered {kind!r} entry {op!r} exposes no "
            "expand(domains, registry) seam; the Laplace-Beltrami "
            "composition needs expandable grad/raise_index/div "
            "entries")
    return expand(key_space if domains is None else domains,
                  registry)


@final
class RaiseIndex(_MetricCalculus):

    """
    Metric-consuming index raising: covariant -> contravariant.

    Description
    -----------
    Contracts covariant components with the inverse metric,
    :math:`u^i = g^{ij} u_j`, reading ``inv_g_<u><v>`` from
    ``grid.metric`` at application and retagging variance
    (operators_composed.md class sketch; validation 6.3 — the vector
    stays thin, one metric owner). Diagonal terms are pointwise
    scalings on the component's own space; cross terms interpolate
    the source component onto the target component's space through
    the registered ``"interpolate"`` rows (module docstring).

    Parameters
    ----------
    coords : tuple[str, ...]
        The chart's base coordinates (metric-name vocabulary).
    diagonal : bool, optional
        Structurally drop the cross terms — an opt-in for charts
        whose induced metric is known diagonal/orthogonal (torus,
        lat-lon sphere), saving the interpolation chains
        (default: False).
    """

    dispatch_kind: ClassVar[str | None] = "raise_index"
    _kind = "raise_index"
    _scalar_operand = False

    def __init__(self, coords: tuple[str, ...], *,
                 diagonal: bool = False) -> None:
        """Store the coordinate family and the diagonal switch."""
        super().__init__(coords)
        self._diagonal: bool = bool(diagonal)

    @property
    def diagonal(self) -> bool:
        """Whether cross terms are structurally dropped."""
        return self._diagonal

    def _expand(
        self,
        domains: tuple[SpaceLike, ...],
        registry: OperatorRegistry,
    ) -> Operator:
        """Expand the inverse-metric contraction block."""
        return _expand_index_move(
            domains, self._coords, registry,
            prefix="inv_g", source=Variance.COVARIANT,
            target=Variance.CONTRAVARIANT,
            diagonal=self._diagonal, operation="raise_index",
            hint="raising consumes covariant components (grad "
                 "outputs, LowerIndex outputs)")


@final
class LowerIndex(_MetricCalculus):

    """
    Metric-consuming index lowering: contravariant -> covariant.

    Description
    -----------
    Contracts contravariant components with the metric,
    :math:`u_i = g_{ij} u^j`, reading ``g_<u><v>`` from
    ``grid.metric`` at application and retagging variance — the
    exact inverse-role twin of :class:`RaiseIndex` (see there for
    the staggering rule and the ``diagonal`` switch).

    Parameters
    ----------
    coords : tuple[str, ...]
        The chart's base coordinates (metric-name vocabulary).
    diagonal : bool, optional
        Structurally drop the cross terms (default: False).
    """

    dispatch_kind: ClassVar[str | None] = "lower_index"
    _kind = "lower_index"
    _scalar_operand = False

    def __init__(self, coords: tuple[str, ...], *,
                 diagonal: bool = False) -> None:
        """Store the coordinate family and the diagonal switch."""
        super().__init__(coords)
        self._diagonal: bool = bool(diagonal)

    @property
    def diagonal(self) -> bool:
        """Whether cross terms are structurally dropped."""
        return self._diagonal

    def _expand(
        self,
        domains: tuple[SpaceLike, ...],
        registry: OperatorRegistry,
    ) -> Operator:
        """Expand the metric contraction block."""
        return _expand_index_move(
            domains, self._coords, registry,
            prefix="g", source=Variance.CONTRAVARIANT,
            target=Variance.COVARIANT,
            diagonal=self._diagonal, operation="lower_index",
            hint="lowering consumes contravariant components "
                 "(RaiseIndex outputs, physical velocities tagged "
                 "via with_variance)")
