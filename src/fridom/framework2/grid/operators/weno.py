"""
``WenoReconstruction``: nonlinear upwind-biased FV reconstruction.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_stencils.md``
(WenoReconstruction section; the class doc places it in the
``reconstruct`` module — it lives in this sibling module because the
Wave-3 ``reconstruct.py`` is read-only for the Wave-4 cluster).

The classic WENO-JS scheme of Jiang & Shu (1996) on cell averages,
numerics-parity with the old stack's
``framework/grid/cartesian/weno_interpolation.py``: ``r = (order +
1) / 2`` candidate ``r``-cell reconstructions (Shu 1998 uniform-mesh
coefficients, computed exactly over the rationals), Jiang-Shu
smoothness indicators, and the nonlinear weights ``alpha_m = d_m /
(beta_m + eps)**2`` with ``eps = 1e-10``, normalized to sum one.
Iteration 1 grounds the reference tables' orders 3 and 5 on periodic
axes; bounded-axis boundary biasing (reduced one-sided stencils near
the wall) is designed-for and raises.

Upwinding is a *pair* of biased instances (``bias="left"`` /
``"right"``) held by flux-splitting advection modules; the flux-sign
selection itself is the ``("select", ...)`` kind
(``operators.select.Where``), never a velocity-consuming parameter
here. The right-biased tables are the exact mirror image of the
left-biased ones (window index and candidate order reversed), so one
fused kernel body serves both biases.

The kernel follows the slice-window rules of
``notes/framework2/02_rules.md`` section 3.5: static Python-float
coefficients baked into the jaxpr, one fused arithmetic expression
over slice views of the halo-extended storage, no ``roll``/gather,
no data-dependent Python branching (the smoothness weighting is
smooth arithmetic, not a ``where``).
"""
# Wave 4: WenoReconstruction (the upwind pair; the sign selection
#    lives in operators.select)
from __future__ import annotations

from fractions import Fraction
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, final

import jax.numpy as jnp

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
)
from fridom.framework2.grid.operators.reconstruct import factor_codomain
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.average import CellAvg

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )

#: regularization of the nonlinear weights (numerics parity with the
#: old stack's ``weno_interpolation.py`` default)
WENO_EPS = 1e-10

#: formal orders grounded in iteration 1 (the r = 2, 3 stencil
#: families of the reference smoothness-indicator tables)
_SUPPORTED_ORDERS = (3, 5)

#: optimal ("linear") stencil weights d_m per stencil size r, indexed
#: by the candidate number m (0 = leftmost candidate of the
#: left-biased scheme); combining the candidates with these weights
#: reproduces the full (2r - 1)-cell reconstruction exactly
_OPTIMAL_WEIGHTS: dict[int, tuple[Fraction, ...]] = {
    2: (Fraction(1, 3), Fraction(2, 3)),
    3: (Fraction(1, 10), Fraction(3, 5), Fraction(3, 10)),
}

#: Jiang-Shu smoothness-indicator difference rows per stencil size:
#: ``beta_m = sum_d scale[d] * (rows[m][d] . cells_m)**2`` (rows[m][d]
#: is the d-th undivided difference over candidate m's r cells)
_SMOOTHNESS_ROWS: dict[
    int, tuple[tuple[tuple[int, ...], ...], ...]] = {
    2: (
        ((-1, 1),),
        ((-1, 1),),
    ),
    3: (
        ((1, -2, 1), (1, -4, 3)),
        ((1, -2, 1), (1, 0, -1)),
        ((1, -2, 1), (3, -4, 1)),
    ),
}

#: the matching per-difference scale factors of the quadratic form
_SMOOTHNESS_SCALE: dict[int, tuple[Fraction, ...]] = {
    2: (Fraction(1),),
    3: (Fraction(13, 12), Fraction(1, 4)),
}


# ================================================================
#  Coefficient computation (host-side, exact)
# ================================================================
def _shu_row(cells: int, face: int) -> tuple[Fraction, ...]:
    r"""
    Exact uniform-mesh reconstruction coefficients (Shu 1998).

    Description
    -----------
    Coefficients :math:`c_i` such that :math:`p(x_k) = \sum_i c_i
    \bar f_i`, where :math:`p` is the polynomial whose averages over
    the ``cells`` unit cells match :math:`\bar f` and :math:`x_k` is
    interface ``face`` (interfaces are indexed ``0 .. cells``). The
    closed form on a uniform mesh is

    .. math::
        c_{ki} = \sum_{m=i+1}^{n} \sum_{j=0, j \neq m}^{n}
            \frac{1}{m - j}
            \prod_{r=0, r \neq m, j}^{n} \frac{k - r}{m - r}

    (the old stack's ``reconstruction_coefficients.py`` formula),
    evaluated here in ``Fraction`` arithmetic at construction time.

    Parameters
    ----------
    cells : int
        The stencil size n (number of cell averages).
    face : int
        The interface index k to reconstruct at (0 .. cells).

    Returns
    -------
    tuple[Fraction, ...]
        The exact ``cells`` reconstruction coefficients.
    """
    coeffs = []
    for i in range(cells):
        total = Fraction(0)
        for m in range(i + 1, cells + 1):
            for j in range(cells + 1):
                if j == m:
                    continue
                term = Fraction(1, m - j)
                for node in range(cells + 1):
                    if node in (m, j):
                        continue
                    term *= Fraction(face - node, m - node)
                total += term
        coeffs.append(total)
    return tuple(coeffs)


class WenoTables(NamedTuple):

    """
    Static coefficient tables of one biased WENO kernel.

    Description
    -----------
    Plain nested tuples of Python floats — hashable static structure
    baked into the jaxpr as constants (never traced). Candidate ``m``
    reads the ``r`` consecutive slice windows starting at
    ``offsets[m]`` within the ``size``-window family.

    Parameters
    ----------
    size : int
        The full window size (= the formal order, 2r - 1).
    offsets : tuple[int, ...]
        Per-candidate first window index within the full window.
    coeffs : tuple[tuple[float, ...], ...]
        Per-candidate reconstruction coefficients (r each).
    optimal : tuple[float, ...]
        The optimal (linear) weights d_m.
    beta_rows : tuple[tuple[tuple[float, ...], ...], ...]
        Per-candidate smoothness-indicator difference rows.
    beta_scale : tuple[float, ...]
        The per-difference scale factors of the quadratic form.
    """

    size: int
    offsets: tuple[int, ...]
    coeffs: tuple[tuple[float, ...], ...]
    optimal: tuple[float, ...]
    beta_rows: tuple[tuple[tuple[float, ...], ...], ...]
    beta_scale: tuple[float, ...]


def _validate(order: int, bias: str) -> None:
    """
    Validate the (order, bias) kernel parameters.

    Parameters
    ----------
    order : int
        The requested formal order.
    bias : str
        The requested bias side.

    Raises
    ------
    ValueError
        On an even order, an unsupported odd order, or an unknown
        bias side.
    """
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError(
            f"the WENO order must be an integer, got {order!r}")
    if order % 2 == 0:
        raise ValueError(
            f"WENO reconstructions have odd formal order, got "
            f"{order}")
    if order not in _SUPPORTED_ORDERS:
        raise ValueError(
            f"iteration 1 grounds the WENO orders "
            f"{_SUPPORTED_ORDERS} (the reference smoothness-"
            f"indicator tables), got {order}")
    if bias not in ("left", "right"):
        raise ValueError(
            f"bias must be 'left' or 'right', got {bias!r}")


@cache
def weno_tables(
    order: int, bias: Literal["left", "right"],
) -> WenoTables:
    """
    Build (and cache) the static tables of one biased kernel.

    Description
    -----------
    Left bias: candidate ``m`` covers window cells ``m .. m + r - 1``
    with the Shu row reconstructing at candidate-interface
    ``r - m`` — the shared output face. Right bias is the exact
    mirror image: window offsets and per-candidate coefficient/
    difference rows reverse, the optimal weights stay attached to
    their (mirrored) candidates. All rationals are floated once here.

    Parameters
    ----------
    order : int
        The odd formal order (iteration 1: 3 or 5).
    bias : Literal["left", "right"]
        The upwind bias side.

    Returns
    -------
    WenoTables
        The static coefficient tables.
    """
    _validate(order, bias)
    r = (order + 1) // 2
    offsets = tuple(range(r))
    coeffs: tuple[tuple[Fraction, ...], ...] = tuple(
        _shu_row(r, r - m) for m in range(r))
    rows: tuple[tuple[tuple[int, ...], ...], ...] = (
        _SMOOTHNESS_ROWS[r])
    if bias == "right":
        offsets = tuple(r - 1 - m for m in range(r))
        coeffs = tuple(tuple(reversed(row)) for row in coeffs)
        rows = tuple(
            tuple(tuple(reversed(diff)) for diff in cand)
            for cand in rows)
    return WenoTables(
        size=order,
        offsets=offsets,
        coeffs=tuple(
            tuple(float(c) for c in cand) for cand in coeffs),
        optimal=tuple(float(d) for d in _OPTIMAL_WEIGHTS[r]),
        beta_rows=tuple(
            tuple(tuple(float(b) for b in diff) for diff in cand)
            for cand in rows),
        beta_scale=tuple(float(s) for s in _SMOOTHNESS_SCALE[r]),
    )


# ================================================================
#  Fused array kernel (slice windows, static coefficients)
# ================================================================
def _window_views(
    arr: Array, axis: int, size: int,
) -> tuple[Array, ...]:
    """
    Slice the ``size`` stencil windows of ``arr`` along ``axis``.

    Description
    -----------
    Window ``j`` is ``arr[..., j : j + m - size + 1, ...]`` along the
    stencil axis (m the input length) — the slice-based
    shape-changing stencil of rules section 3.5 (the
    ``stencil_kernels`` window slicer, restated here because that
    module is Wave-1/2 ground).

    Parameters
    ----------
    arr : Array
        The input array (halo-extended by the caller as needed).
    axis : int
        The stencil axis; may be negative.
    size : int
        The stencil size (number of windows).

    Returns
    -------
    tuple[Array, ...]
        The ``size`` windows, each of length m - size + 1.

    Raises
    ------
    ValueError
        If the axis is shorter than the stencil.
    """
    out_len = arr.shape[axis] - size + 1
    if out_len < 1:
        raise ValueError(
            f"axis {axis} of length {arr.shape[axis]} is shorter "
            f"than the {size}-point WENO stencil")
    index = [slice(None)] * arr.ndim
    views = []
    for offset in range(size):
        index[axis] = slice(offset, offset + out_len)
        views.append(arr[tuple(index)])
    return tuple(views)


def _weighted_sum(
    views: tuple[Array, ...], weights: tuple[float, ...],
) -> Array:
    """
    Static-weight linear combination of slice views.

    Parameters
    ----------
    views : tuple[Array, ...]
        The stencil windows (equal shapes).
    weights : tuple[float, ...]
        The static weights; exact zeros are skipped.

    Returns
    -------
    Array
        The fused weighted sum.
    """
    total = None
    for weight, view in zip(weights, views, strict=True):
        if weight == 0.0:
            continue
        term = view if weight == 1.0 else weight * view
        total = term if total is None else total + term
    return total


def _alpha_candidates(
    windows: tuple[Array, ...], tables: WenoTables,
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    """
    Unnormalized weights and candidate values of every stencil.

    Description
    -----------
    Per candidate ``m``: the reconstruction ``q_m`` (static Shu
    coefficients), the Jiang-Shu smoothness indicator ``beta_m``
    (scaled squares of undivided differences), and the WENO-JS
    weight ``alpha_m = d_m / (beta_m + eps)**2``.

    Parameters
    ----------
    windows : tuple[Array, ...]
        The full-size window family (`_window_views`).
    tables : WenoTables
        The static tables of the biased kernel.

    Returns
    -------
    tuple[tuple[Array, ...], tuple[Array, ...]]
        ``(alphas, candidates)``, one array each per stencil.
    """
    r = len(tables.optimal)
    alphas = []
    candidates = []
    for m, offset in enumerate(tables.offsets):
        cells = windows[offset:offset + r]
        candidates.append(_weighted_sum(cells, tables.coeffs[m]))
        beta = None
        for scale, diff in zip(
                tables.beta_scale, tables.beta_rows[m], strict=True):
            square = _weighted_sum(cells, diff) ** 2
            term = square if scale == 1.0 else scale * square
            beta = term if beta is None else beta + term
        alphas.append(tables.optimal[m] / (beta + WENO_EPS) ** 2)
    return tuple(alphas), tuple(candidates)


def weno_reconstruct(
    arr: Array,
    axis: int,
    order: int = 5,
    bias: Literal["left", "right"] = "left",
) -> Array:
    """
    Biased WENO reconstruction along an axis (fused kernel).

    Description
    -----------
    Output entry ``t`` reconstructs from the input window
    ``t : t + order``; the axis shrinks by ``order - 1``. Within the
    window (cells ``0 .. order - 1``) the reconstructed face is the
    right face of cell ``order // 2`` for ``bias="left"`` and of
    cell ``order // 2 - 1`` for ``bias="right"`` — the caller's
    window alignment maps this onto the storage frame. Periodicity
    and boundaries live entirely in the caller's halo fill (rules
    section 3.5). One fused arithmetic expression over static-shape
    slice views; the nonlinear weighting is smooth arithmetic with
    static coefficients — no data-dependent branching.

    Parameters
    ----------
    arr : Array
        The input cell averages (halo-extended by the caller).
    axis : int
        The stencil axis; may be negative.
    order : int, optional
        The odd formal order, 3 or 5 in iteration 1 (default: 5).
    bias : Literal["left", "right"], optional
        The upwind bias side (default: "left").

    Returns
    -------
    Array
        The reconstructed face values; ``axis`` shrinks by
        ``order - 1``.
    """
    tables = weno_tables(order, bias)
    windows = _window_views(arr, axis, tables.size)
    alphas, candidates = _alpha_candidates(windows, tables)
    total = alphas[0]
    combined = alphas[0] * candidates[0]
    for alpha, candidate in zip(alphas[1:], candidates[1:],
                                strict=True):
        total = total + alpha
        combined = combined + alpha * candidate
    return combined / total


def weno_weights(
    arr: Array,
    axis: int,
    order: int = 5,
    bias: Literal["left", "right"] = "left",
) -> tuple[Array, ...]:
    r"""
    Compute the normalized nonlinear weights of every stencil.

    Description
    -----------
    Diagnostic seam (tests, smoothness analysis): the weights the
    fused kernel applies, aligned with :func:`weno_reconstruct`'s
    output windows. On smooth data away from critical points they
    approach the optimal weights ``d_m`` (at :math:`O(\Delta x^2)`
    for the ``r = 3`` family, :math:`O(\Delta x)` for ``r = 2``).

    Parameters
    ----------
    arr : Array
        The input cell averages (halo-extended by the caller).
    axis : int
        The stencil axis; may be negative.
    order : int, optional
        The odd formal order, 3 or 5 in iteration 1 (default: 5).
    bias : Literal["left", "right"], optional
        The upwind bias side (default: "left").

    Returns
    -------
    tuple[Array, ...]
        One weight array per candidate, in table order; they sum to
        one elementwise.
    """
    tables = weno_tables(order, bias)
    windows = _window_views(arr, axis, tables.size)
    alphas, _ = _alpha_candidates(windows, tables)
    total = alphas[0]
    for alpha in alphas[1:]:
        total = total + alpha
    return tuple(alpha / total for alpha in alphas)


# ================================================================
#  WenoReconstruction
# ================================================================
@final
class WenoReconstruction(SeparableOperator):

    """
    WENO average-to-point reconstruction (``CellAvg -> face``).

    Description
    -----------
    Nonlinear upwind-biased reconstruction — the module-override
    archetype (sketch 4.2): advection schemes register it under the
    ``"reconstruct"`` kind; it is **not** a default-table row.
    Upwinding is a *pair* of biased instances selected by flux sign
    through the ``("select", ...)`` kind (``operators.select.Where``).
    Iteration 1 is periodic-only (parity with the old stack's
    ``weno_interpolation.py``); bounded-axis boundary biasing is
    designed-for, so a bounded registration is a space error, never a
    silent fallback. Nonlinear, hence no ``eigenvalues`` (the raising
    base is correct and automatic).

    Parameters
    ----------
    order : int, optional
        The odd formal order; iteration 1 grounds 3 and 5
        (default: 5).
    bias : Literal["left", "right"], optional
        The upwind bias side of the reconstruction
        (default: "left").
    """

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __init__(
        self,
        order: int = 5,
        bias: Literal["left", "right"] = "left",
    ) -> None:
        """Create a WENO kernel of the given odd order and bias."""
        _validate(order, bias)
        self._order: int = order
        self._bias: Literal["left", "right"] = bias

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def order(self) -> int:
        """Formal order of the WENO reconstruction."""
        return self._order

    @property
    def bias(self) -> Literal["left", "right"]:
        """Upwind bias side of the reconstruction."""
        return self._bias

    # ------------------------------------------------------------
    #  Signature and requirements
    # ------------------------------------------------------------
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve the reconstruction codomain.

        Description
        -----------
        reconstruct: CellAvg -> Right (periodic; both biases land on
        the same face space — the bias is a stencil property, not a
        signature property). The bounded ``CellAvg -> Outer`` biased
        variant is designed-for and raises.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space.

        Returns
        -------
        FunctionSpace
            The face point-value codomain factor.
        """
        if not isinstance(domain, CellAvg):
            raise SpaceMismatchError(
                "WenoReconstruction reconstructs primal cell "
                f"averages onto faces (CellAvg -> Right), got "
                f"{domain!r}", left=domain, operation="reconstruct")
        if domain.scalars is Scalars.COMPLEX:
            raise SpaceMismatchError(
                "the WENO smoothness indicators are real quadratic "
                "forms; complex operands have no iteration-1 "
                f"signature, got {domain!r}",
                left=domain, operation="reconstruct")
        mesh = domain.mesh
        if not mesh.periodic:
            raise SpaceMismatchError(
                "iteration-1 WENO is periodic-only (the wide biased "
                "stencil is valid on wrap-around halos); bounded-"
                "axis boundary biasing (reduced one-sided stencils) "
                f"is designed-for, got {domain!r}",
                left=domain, operation="reconstruct")
        return mesh.right

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — fixed by the order
    ) -> OperatorRequirements:
        """
        Declare halo = order // 2 + 1, layout "any".

        Description
        -----------
        The uniform declaration covering both biases: relative to
        the output face the right-biased window reaches
        ``order // 2 + 1`` cells rightward (the left-biased one
        ``order // 2`` leftward).

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=self._order // 2 + 1)

    # ------------------------------------------------------------
    #  Kernel application (biased window alignment)
    # ------------------------------------------------------------
    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Reconstruct along ``axis`` (biased window-aligned kernel).

        Description
        -----------
        The FV window calculus of ``operators.reconstruct`` with a
        **biased** alignment: an odd-size stencil between staggered
        offsets cannot land its output on the window midpoint (the
        midpoint shift is half-integral) — the bias breaks the tie.
        Kernel output ``t`` covers storage cells ``t .. t + order -
        1`` and lands on the right face of window cell ``order //
        2`` (left bias) or ``order // 2 - 1`` (right bias), so it
        fills output slot ``k = t + m0`` with ``m0 = order // 2``
        (left) / ``order // 2 - 1`` (right). Edge slots the kernel
        cannot compute are zero-filled and repaired by the
        post-application sync.

        Parameters
        ----------
        f : FieldLike
            The operand field (storage-shaped ``_data``).
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The reconstructed field (metadata kept: same quantity).
        """
        space = f.function_space
        bare = space.bare
        codomain = factor_codomain(self, space, axis)
        size = self._order
        m0 = size // 2 if self._bias == "left" else size // 2 - 1

        axis_index = bare.names.index(axis)
        storage = f._data  # noqa: SLF001 — documented storage seam
        decomposition = f.grid.decomposition
        # the codomain storage lives in the operand's layout (the
        # base re-attaches it)
        out_shape = decomposition.storage_shape(codomain, space.layout)
        s_out = out_shape[axis_index]
        n_out = codomain.factor(axis).shape[0]
        try:
            width = decomposition.halo[axis]
        except KeyError:
            width = 0

        # per-side stencil reach beyond the true region, frame-
        # independent (see reconstruct.apply_fv_staggered): the
        # negotiated halo must cover it on every shard
        reach_right = (n_out - bare.factor(axis).shape[0]
                       + size - 1 - m0)
        if m0 > width or reach_right > width:
            raise ValueError(
                f"the negotiated halo width {width} along {axis!r} "
                f"is too small for the {size}-point stencil of "
                f"{type(self).__name__}; renegotiate with a registry "
                "that declares the wider requirement")
        full = weno_reconstruct(storage, axis_index,
                                order=self._order, bias=self._bias)
        length = full.shape[axis_index]
        lo = max(0, m0)
        hi = min(s_out, m0 + length)
        index: list[slice] = [slice(None)] * full.ndim
        index[axis_index] = slice(lo - m0, hi - m0)
        piece = full[tuple(index)]
        pads = [(0, 0)] * full.ndim
        pads[axis_index] = (lo, s_out - hi)
        data = jnp.pad(piece, pads)
        return type(f)(f.grid, codomain, data, f.metadata)
