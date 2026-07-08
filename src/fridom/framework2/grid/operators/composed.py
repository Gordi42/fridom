"""
Composed vector calculus: the grad/div/curl/laplacian builders.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_composed.md``.
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
halo-accounting trace. ``RaiseIndex``/``LowerIndex`` remain
designed-for (metric grids).
"""
# Wave 3: BlockMatrix (it-1 subset), Gradient, Divergence, Curl,
#    Laplacian -- designed-for: RaiseIndex, LowerIndex
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, final

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.operators.base import (
    FieldLike,
    Identity,
    Operator,
    OperatorRequirements,
    ScaledOperator,
    Zero,
    resolve_codomain,
)
from fridom.framework2.grid.operators.block_symbol import BlockSymbol
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.registry import DispatchError
from fridom.framework2.grid.spaces.constant import ConstantSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.framework2.grid.operators.registry import (
        OperatorRegistry,
    )
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike

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
    return tuple(
        name
        for factor in space.factors
        if not isinstance(factor, ConstantSpace)
        for name in factor.names)


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

    # ------------------------------------------------------------
    #  Signature and requirements
    # ------------------------------------------------------------
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
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

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
        halo = 0
        collective = False
        for row in self._rows:
            for entry in row:
                if isinstance(entry, Zero):
                    continue
                req = entry.requirements(domain)
                halo = max(halo, req.halo)
                collective = collective or req.collective
        return OperatorRequirements(halo=halo, collective=collective)

    # ------------------------------------------------------------
    #  Eigenvalues (per-mode system matrix)
    # ------------------------------------------------------------
    def eigenvalues(
        self, grid: object, *spaces: SpaceLike,
    ) -> BlockSymbol:
        r"""
        Assemble the per-mode block symbol (the system matrix).

        Description
        -----------
        The block generalization of the scalar
        :meth:`~fridom.framework2.grid.operators.base.Operator.eigenvalues`:
        walks the entry matrix and queries each non-``Zero`` entry's
        scalar ``eigenvalues`` on its column's coefficient space (the
        scalar chain / sum / scale algebra runs *inside* each entry,
        reused unchanged), then scatters the results into a
        :class:`BlockSymbol`. The row (codomain) coefficient tuple is
        ``self.codomain(*spaces)`` — which also enforces the per-row
        codomain consistency (staggered entries retag through their own
        phase symbols). Any entry declining its symbol raises
        ``EigenbasisError``, propagated here.

        Parameters
        ----------
        grid : object
            The grid mediating wavenumbers and metric measures.
        *spaces : SpaceLike
            One column (domain) coefficient space per block column.

        Returns
        -------
        BlockSymbol
            The per-mode ``m_out x m_in`` system matrix.
        """
        n_cols = len(self._rows[0])
        if len(spaces) != n_cols:
            raise SpaceMismatchError(
                f"block of {n_cols} columns queried with "
                f"{len(spaces)} operand space(s)", operation="block")
        out = self.codomain(*spaces)
        out_spaces = out if isinstance(out, tuple) else (out,)
        blocks = tuple(
            tuple(
                None if isinstance(entry, Zero)
                else entry.eigenvalues(grid, spaces[j])
                for j, entry in enumerate(row))
            for row in self._rows)
        return BlockSymbol.from_blocks(blocks, spaces, out_spaces)

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

    def __init__(self, order: int | None) -> None:
        """Store the optional pinned FD order."""
        if order is not None:
            FiniteDifference(order)  # validates even, >= 2
        self._order: int | None = order

    @property
    def order(self) -> int | None:
        """Pinned FD order, or None for dispatched entries."""
        return self._order

    def codomain(
        self,
        *domains: SpaceLike,  # noqa: ARG002 — always raises
    ) -> SpaceLike | tuple[SpaceLike, ...]:
        """Unexpanded builders have no signature: raise."""
        raise DispatchError(
            f"the {self._kind!r} builder expands against a grid's "
            "registry; call expand(domains, registry) or apply it "
            "to a field")

    def expand(
        self,
        domains: SpaceLike | tuple[SpaceLike, ...],
        registry: OperatorRegistry,
    ) -> Operator:
        """
        Expand into the concrete block over the axis family.

        Parameters
        ----------
        domains : SpaceLike | tuple[SpaceLike, ...]
            The bare operand space (scalar kinds) or the tuple of
            component spaces (vector kinds).
        registry : OperatorRegistry
            The dispatch registry resolving the per-axis entries.

        Returns
        -------
        Operator
            The expanded ``BlockMatrix``.
        """
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
        # laplacian: div @ grad as a block matmul (1xn @ nx1)
        grad = _expand_grad(domains, order, registry)
        mid = grad.codomain(domains)
        if not isinstance(mid, tuple):
            mid = (mid,)
        div = _expand_div(mid, order, registry)
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


def Laplacian(order: int | None = None) -> Operator:  # noqa: N802
    """
    Build the ``"laplacian"`` builder: ``div @ grad``.

    Description
    -----------
    At expansion the block matmul ``(1xn) @ (nx1)`` collapses to a
    1x1 block holding the ``OperatorSum`` of per-axis
    second-derivative ``SeparableComposite``s (B1) — so
    div∘grad = laplacian holds by construction.

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
    return _Laplacian(order)


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
    is :class:`~fridom.framework2.grid.operators.base.Identity` when
    ``coeffs.get(a, default)`` is exactly the scalar ``1`` (the
    structural neutral — its ``codomain`` is the identity, so the
    diagonal threads each component's space through unchanged and
    ``Div @ Diag @ Grad`` block-composes), else the scaled identity
    ``coeff * Identity``; off-diagonal entries are ``Zero``.

    A coefficient may be a Python number or a 0-d ``jax.Array`` (a
    traced-but-constant scalar such as ``1/dsqr``); the scaled-identity
    entry carries a symbol through
    :meth:`~fridom.framework2.grid.operators.base.ScaledOperator.eigenvalues`.

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
