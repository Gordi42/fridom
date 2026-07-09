r"""
Linear-term block signatures (the H1 pivot vocabulary).

Description
-----------
Design source: ``notes/framework2/linear_term_blocks_plan.md`` (the
sub-plan) and ``notes/framework2/blocksymbol_l_assembly.md`` §3. A
**linear** tendency term declares its :math:`(out, src) \to`
``Operator`` block structure as a tuple of :class:`LinearBlock`
records; two consumers read the *same* declaration (single source of
truth):

- **numeric** — the running model, via :func:`apply_linear_blocks`:
  ``increment[out] += coeff . op(state[src])`` (a plain
  ``dict[out] -> ScalarField``, folded at the one term call site);
- **symbolic** — :func:`linear_blocks`, which resolves each block's
  grid-free operator spec to the retained ``fr.Operator``
  (``src_space -> out_space``) and resolves each coefficient to a
  constant scalar (the provides-implies-constancy gate). This is the
  bridge phase H1 consumes; the ``BlockSymbol`` eigendecomposition is
  built there, not here.

The operator specs (:class:`Interp`, :class:`Diff`, :class:`Scale`,
composed with ``@``) are grid-free records. Because ``.diff()`` /
``.to()`` resolve to the **interned singleton** operators, the numeric
path applies the identical objects the hand closures built transiently
-- so the derived increment is bit-reproducible (sub-plan §3).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fridom.framework2.grid.fields.scalar_field import _conversion_kind
from fridom.framework2.grid.operators.base import (
    EigenbasisError,
    Identity,
    resolve_codomain,
)
from fridom.framework2.model.assembly import _collect_terms
from fridom.framework2.model.terms import (
    EXPLICIT,
    TERM_ATTRIBUTE,
    TendencyTerm,
    Treatment,
)
from fridom.framework2.model.time_dependent import resolve_at

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.operators.base import Operator
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike
    from fridom.framework2.model.context import StepContext
    from fridom.framework2.model.model import Model
    from fridom.framework2.model.params import ParamName


# ================================================================
#  Operator specs (grid-free, resolved at assembly)
# ================================================================
class OperatorSpec(ABC):

    """
    A grid-free operator spec: a block's ``src_space -> out_space`` map.

    Description
    -----------
    Two resolutions off one declaration (sub-plan §1): the numeric
    :meth:`apply_numeric` (applies the interned ``.diff``/``.to``
    operators, bit-identical to the hand closures) and the symbolic
    :meth:`resolve` (returns the retained ``fr.Operator`` phase H1
    queries for its eigenvalues). Specs compose with ``@``
    (``(A @ B)(f) == A(B(f))``).
    """

    @abstractmethod
    def apply_numeric(
        self, field: ScalarField, out_space: SpaceLike,
        state: object, ctx: StepContext,
    ) -> ScalarField:
        """Apply the operator numerically to ``field`` (sub-plan §3)."""

    @abstractmethod
    def resolve(
        self, grid: Grid, in_space: SpaceLike, out_space: SpaceLike,
        gate: Callable[[ParamName], float],
    ) -> tuple[Operator, SpaceLike]:
        """Resolve to ``(fr.Operator, produced_bare_space)`` (§4)."""

    def __matmul__(self, other: object) -> OperatorSpec:
        """Compose: ``(A @ B)`` applies ``B`` first, then ``A``."""
        if not isinstance(other, OperatorSpec):
            return NotImplemented
        return _Composed(self, other)


@dataclass(frozen=True)
class Interp(OperatorSpec):

    """
    Interpolate ``src`` onto the block's ``out`` space (``.to``).

    Description
    -----------
    Resolves to the composite of the per-factor ``LinearInterp``
    kernels the ``.to`` path applies (Coriolis' ``v -> u`` crosses two
    staggered factors, so its operator is a two-factor composite).
    """

    def apply_numeric(
        self, field: ScalarField, out_space: SpaceLike,
        state: object, ctx: StepContext,  # noqa: ARG002 — grid-free
    ) -> ScalarField:
        """``field.to(out_space)`` (the interned interpolation)."""
        return field.to(out_space)

    def resolve(
        self, grid: Grid, in_space: SpaceLike, out_space: SpaceLike,
        gate: Callable[[ParamName], float],  # noqa: ARG002 — no coeff
    ) -> tuple[Operator, SpaceLike]:
        """Compose the per-factor interpolation operators (like ``.to``)."""
        op = _interp_operator(grid, in_space, out_space)
        return op, out_space.bare


@dataclass(frozen=True)
class Diff(OperatorSpec):

    """
    Default derivative along ``axis`` (``.diff``): a ``FiniteDifference``.

    Parameters
    ----------
    axis : str
        The coordinate name to differentiate along.
    """

    axis: str

    def apply_numeric(
        self, field: ScalarField, out_space: SpaceLike,  # noqa: ARG002
        state: object, ctx: StepContext,  # noqa: ARG002 — grid-free
    ) -> ScalarField:
        """``field.diff(axis)`` (the interned finite difference)."""
        return field.diff(self.axis)

    def resolve(
        self,
        grid: Grid,
        in_space: SpaceLike,
        out_space: SpaceLike,  # noqa: ARG002
        gate: Callable[[ParamName], float],  # noqa: ARG002 — no coeff
    ) -> tuple[Operator, SpaceLike]:
        """Return the bound ``FiniteDifference`` and its codomain."""
        factor = in_space.bare.factor(self.axis)
        op = grid.dispatch.resolve("diff", factor)[self.axis]
        return op, resolve_codomain(op, in_space.bare)


@dataclass(frozen=True)
class Scale(OperatorSpec):

    """
    Flux-form multiply by a constant AUX field (``coeff`` inside ``@``).

    Description
    -----------
    Numerically multiplies by the AUX field interpolated onto the
    operand's space (``c.to(u) * u``) — the flux form that keeps
    ``diff(c^2 u)`` bit-identical (sub-plan §3). Symbolically it is the
    constant the AUX field stands for (``const``, gated by
    provides-implies-constancy), folded as ``const * Identity`` into
    the operator (so ``Diff @ Scale`` carries the ``c^2`` in its
    symbol).

    Parameters
    ----------
    aux : str
        The constant AUX field name to scale by (runtime source).
    const : ParamName
        The provided parameter the AUX field stands for (the symbolic
        constant, gated for constancy).
    """

    aux: str
    const: ParamName

    def apply_numeric(
        self, field: ScalarField, out_space: SpaceLike,  # noqa: ARG002
        state: object, ctx: StepContext,  # noqa: ARG002 — reads state
    ) -> ScalarField:
        """``state[aux].to(field.space) * field`` (the flux form)."""
        return state[self.aux].to(field.function_space) * field

    def resolve(
        self,
        grid: Grid,  # noqa: ARG002 — constant fold
        in_space: SpaceLike,
        out_space: SpaceLike,  # noqa: ARG002
        gate: Callable[[ParamName], float],
    ) -> tuple[Operator, SpaceLike]:
        """``const * Identity`` (the constant the AUX field stands for)."""
        return gate(self.const) * Identity(), in_space.bare


@dataclass(frozen=True)
class _Composed(OperatorSpec):

    """A composed spec ``outer @ inner`` (``inner`` applied first)."""

    outer: OperatorSpec
    inner: OperatorSpec

    def apply_numeric(
        self, field: ScalarField, out_space: SpaceLike,
        state: object, ctx: StepContext,
    ) -> ScalarField:
        """Apply ``inner`` then ``outer`` (right-to-left)."""
        inner = self.inner.apply_numeric(field, out_space, state, ctx)
        return self.outer.apply_numeric(inner, out_space, state, ctx)

    def resolve(
        self, grid: Grid, in_space: SpaceLike, out_space: SpaceLike,
        gate: Callable[[ParamName], float],
    ) -> tuple[Operator, SpaceLike]:
        """Thread the space through ``inner`` then ``outer``; compose."""
        inner_op, mid = self.inner.resolve(
            grid, in_space, in_space, gate)
        outer_op, produced = self.outer.resolve(
            grid, mid, out_space, gate)
        return outer_op @ inner_op, produced


def _interp_operator(
    grid: Grid, src_space: SpaceLike, dst_space: SpaceLike,
) -> Operator:
    """
    Compose the per-factor interpolation operators, like ``.to``.

    Description
    -----------
    Replays the ``ScalarField.to`` per-factor resolution (reads the
    conversion kind off the family matrix, resolves the interned
    operator against ``grid.dispatch``) and composes the differing
    factors with ``@``; an all-equal pair yields ``Identity``.
    """
    src = src_space.bare
    dst = dst_space.bare
    op: Operator = Identity()
    for name in dst.names:
        src_factor = src.factor(name)
        dst_factor = dst.factor(name)
        if src_factor is dst_factor:
            continue
        kind = _conversion_kind(src_factor, dst_factor)
        factor_op = grid.dispatch.resolve(kind, src_factor)[name]
        op = factor_op @ op
    return op


# ================================================================
#  Coefficient spec (runtime source + symbolic constant)
# ================================================================
@dataclass(frozen=True)
class Coeff:

    r"""
    A block's coefficient: a runtime source, a symbolic constant, sign.

    Description
    -----------
    One of three runtime sources (sub-plan §1):

    - ``param`` — a traced scalar read from ``ctx.params[param]``
      (buoyancy ``dsqr``, restoring ``N^2``); the symbolic constant is
      the same name resolved off ``model.parameters``;
    - ``aux`` — a constant AUX field (Coriolis ``f_coriolis``) applied
      as a field product; the symbolic constant is the parameter it
      stands for, given as ``const`` (a ``ParamName``);
    - a literal — ``const`` a number (gravity ``-1``), both the runtime
      and the symbolic coefficient.

    ``sign`` (:math:`\pm 1`) and ``invert`` (divide instead of multiply)
    apply to both consumers. The numeric application matches the hand
    closures bit-for-bit: field products for ``aux`` (``.to`` then
    ``*``), raw-array scaling for ``param`` (``ctx.params`` scalars are
    traced arrays, not Python scalars, so they enter through
    ``.data``), Python-scalar scaling for a literal.

    Parameters
    ----------
    param : ParamName | None, optional
        The traced-scalar runtime source (default: None).
    aux : str | None, optional
        The constant AUX-field runtime source (default: None).
    const : ParamName | float | int | None, optional
        The ``aux`` source's backing parameter, or a literal
        coefficient when ``param``/``aux`` are unset (default: None).
    sign : int, optional
        The sign (+1 or -1), applied to both consumers (default: 1).
    invert : bool, optional
        Divide by the coefficient instead of multiplying (default:
        False).
    """

    param: ParamName | None = None
    aux: str | None = None
    const: object = None
    sign: int = 1
    invert: bool = False

    def __post_init__(self) -> None:
        """Validate exactly one runtime source is well-formed."""
        if self.sign not in (1, -1):
            raise ValueError(
                f"Coeff sign must be +1 or -1, got {self.sign!r}")
        if self.param is not None and self.aux is not None:
            raise ValueError(
                "Coeff takes at most one of param=/aux=")
        if self.aux is not None and self.const is None:
            raise ValueError(
                "Coeff(aux=...) needs const= (the backing ParamName)")
        if (self.param is None and self.aux is None
                and self.const is None):
            raise ValueError(
                "Coeff needs a runtime source: param=, aux=, or a "
                "literal const=")

    # ------------------------------------------------------------
    def apply_numeric(
        self, applied: ScalarField, state: object, ctx: StepContext,
    ) -> ScalarField:
        """Scale ``applied`` by the runtime coefficient (bit-identical)."""
        if self.aux is not None:
            coeff_field = state[self.aux].to(applied.function_space)
            result = coeff_field * applied
            return -result if self.sign < 0 else result
        if self.param is not None:
            scalar = ctx.params[self.param]
            data = applied.data
            data = data / scalar if self.invert else scalar * data
            if self.sign < 0:
                data = -data
            return applied.with_data(data)
        result = self.const * applied
        return -result if self.sign < 0 else result

    def constant(
        self, gate: Callable[[ParamName], float],
    ) -> float:
        """Resolve to the constant scalar (the constancy gate, §4)."""
        if self.param is not None:
            value = gate(self.param)
        elif self.aux is not None:
            value = gate(self.const)
        else:
            value = self.const
        if self.invert:
            value = 1.0 / value
        return self.sign * value


# ================================================================
#  LinearBlock
# ================================================================
@dataclass(frozen=True)
class LinearBlock:

    """
    One ``(out, src, op, coeff)`` block of a linear term (sub-plan §1).

    Description
    -----------
    ``out``/``src`` are prognostic component names (the row/column of
    the linearized operator ``L``); ``op`` is a grid-free
    :class:`OperatorSpec` resolving to the retained operator
    ``coeff-space(src) -> space(out)``; ``coeff`` is a :class:`Coeff`.

    Parameters
    ----------
    out : str
        The advanced PROGNOSTIC component (``L`` row).
    src : str
        The read state component (``L`` column).
    op : OperatorSpec
        The grid-free operator spec.
    coeff : Coeff
        The coefficient spec.
    """

    out: str
    src: str
    op: OperatorSpec
    coeff: Coeff

    def _numeric(
        self, state: object, ctx: StepContext,
    ) -> ScalarField:
        """Compute ``coeff . op(state[src])`` on the ``out`` space."""
        src_field = state[self.src]
        out_space = state[self.out].function_space
        applied = self.op.apply_numeric(
            src_field, out_space, state, ctx)
        return self.coeff.apply_numeric(applied, state, ctx)


# ================================================================
#  Numeric consumer — the derived tendency
# ================================================================
def apply_linear_blocks(
    blocks: tuple[LinearBlock, ...], state: object, ctx: StepContext,
) -> dict[str, ScalarField]:
    """
    Derive a term's numeric increment dict from its blocks (§3).

    Description
    -----------
    ``increment[out] += coeff . op(state[src])`` over the blocks in
    declaration order; blocks sharing an ``out`` accumulate through the
    field ``+`` (matching the hand closures' ordering). The single
    source of truth for the running model — the same ``blocks`` tuple
    feeds :func:`linear_blocks`.

    Parameters
    ----------
    blocks : tuple[LinearBlock, ...]
        The term's declared blocks.
    state : VectorField
        The full assembled state (PROGNOSTIC + AUXILIARY).
    ctx : StepContext
        The per-substage context (parameters).

    Returns
    -------
    dict[str, ScalarField]
        PROGNOSTIC-keyed increments.
    """
    result: dict[str, ScalarField] = {}
    for block in blocks:
        contrib = block._numeric(state, ctx)  # noqa: SLF001 — owning module
        if block.out in result:
            result[block.out] = result[block.out] + contrib
        else:
            result[block.out] = contrib
    return result


# ================================================================
#  Term factory — a linear term is fully derived from its blocks
# ================================================================
def linear_term(
    name: str,
    *,
    blocks: tuple[LinearBlock, ...],
    advances: tuple[str, ...] | None = None,
    transports: tuple[str, ...] = (),
    treatment: Treatment = EXPLICIT,
) -> Callable:
    r"""
    Build a ``@fr.term``-stamped function for a pure LINEAR term.

    Description
    -----------
    A linear-block term is *entirely* derived from its ``blocks``:
    the symbolic path already reads ``term.blocks`` (the single
    source of truth), and the numeric path is the mechanical
    ``apply_linear_blocks(blocks, state, ctx)``. This factory
    synthesizes that body once, so a module declares the term as a
    class attribute — ``gravity = fr.linear_term("gravity", ...)`` —
    instead of a decorated method whose body is boilerplate.

    The returned function carries the same ``TendencyTerm`` stamp
    ``@fr.term`` installs (under ``TERM_ATTRIBUTE``), so
    ``Module.tendency_terms`` collects it in class-definition order,
    reading the raw class dict — it is never called bound. The
    synthesized ``fn`` matches the composer's ``(module, state,
    ctx)`` calling convention.

    Parameters
    ----------
    name : str
        The term part of the ``"Module/term"`` attribution key.
    blocks : tuple[LinearBlock, ...]
        The ``(out, src, op, coeff)`` block signatures — the single
        source of truth for both the numeric and symbolic paths.
    advances : tuple[str, ...] | None, optional
        The advanced PROGNOSTIC subset (default: None, derived by
        the assembly dry run).
    transports : tuple[str, ...], optional
        Declared transport intent for the coverage lint
        (default: ()).
    treatment : Treatment, optional
        The integration treatment (default: ``fr.EXPLICIT``).

    Returns
    -------
    Callable
        The stamped function to assign as a class attribute.
    """
    blocks = tuple(blocks)

    def _fn(
        module: object,  # noqa: ARG001 — composer calling convention
        state: object,
        ctx: StepContext,
    ) -> dict[str, ScalarField]:
        """Derive the numeric increment from the shared blocks."""
        return apply_linear_blocks(blocks, state, ctx)

    _fn.__name__ = name
    declaration = TendencyTerm(
        name=name, fn=_fn, treatment=treatment, advances=advances,
        transports=transports, linear=True, blocks=blocks)
    setattr(_fn, TERM_ATTRIBUTE, declaration)
    return _fn


# ================================================================
#  Symbolic consumer — the resolved blocks + constancy gate
# ================================================================
@dataclass(frozen=True)
class ResolvedBlock:

    """
    A block resolved for symbolic ``L`` assembly (the H1 bridge).

    Parameters
    ----------
    out : str
        The advanced PROGNOSTIC component (``L`` row).
    src : str
        The read state component (``L`` column).
    op : Operator
        The retained ``fr.Operator`` (``src_space -> out_space``).
    coeff : float
        The resolved constant coefficient.
    """

    out: str
    src: str
    op: Operator
    coeff: float


def linear_blocks(
    model: Model, *, at_time: float = 0.0,
) -> tuple[ResolvedBlock, ...]:
    r"""
    Resolve the linear terms' blocks for symbolic ``L`` assembly (§4).

    Description
    -----------
    Selects the ``linear=True`` tendency terms, and for each declared
    block resolves the operator spec to the retained ``fr.Operator``
    (``src_space -> out_space``) and the coefficient to a **constant
    scalar** through the provides-implies-constancy gate: a ``param``
    coefficient reads ``model.parameters`` frozen at ``at_time``; an
    ``aux`` coefficient reads the parameter its constant AUX field
    stands for. A model that does not provide a required constant
    (a beta-plane ``f(y)``, a variable-depth ``c^2(x)``) is **declined**
    with :class:`EigenbasisError` — exactly the ``from_model``
    provides-implies-constancy check, expressed through the coeff spec.
    The ``BlockSymbol`` eigendecomposition is phase H1's; it is not
    built here.

    Parameters
    ----------
    model : Model
        The assembled model (any variant; only ``linear`` terms read).
    at_time : float, optional
        The clock time at which to freeze time-dependent parameters
        (default: 0.0).

    Returns
    -------
    tuple[ResolvedBlock, ...]
        The resolved blocks, term then declaration order.

    Raises
    ------
    EigenbasisError
        If a required coefficient is not a provided constant.
    """
    view = model.parameters
    grid = model.grid
    state = model.state
    modules = model._carry.modules  # noqa: SLF001 — owning-model term scan

    def gate(pname: ParamName) -> float:
        """Resolve a parameter to a constant scalar, or decline."""
        if pname not in view:
            raise EigenbasisError(
                f"symbolic L assembly needs a constant {pname!r}: no "
                "provider on this model (a variable-coefficient term, "
                "e.g. a beta-plane f(y), is not Fourier-diagonalizable)")
        return float(resolve_at(view[pname], at_time))

    resolved: list[ResolvedBlock] = []
    for _slot, term in _collect_terms(modules):
        if not getattr(term, "linear", False):
            continue
        blocks: tuple[LinearBlock, ...] = getattr(term, "blocks", ())
        for block in blocks:
            if block.src not in state:
                raise EigenbasisError(
                    f"linear block src {block.src!r} is not a state "
                    "component")
            src_space = state[block.src].function_space
            out_space = state[block.out].function_space
            op, _produced = block.op.resolve(
                grid, src_space, out_space, gate)
            coeff = block.coeff.constant(gate)
            resolved.append(
                ResolvedBlock(block.out, block.src, op, coeff))
    return tuple(resolved)
