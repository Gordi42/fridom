"""
``WenoReconstruction``: nonlinear upwind-biased FV reconstruction.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_stencils.md``
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
``design/specs/grid/02_rules.md`` section 3.5: static Python-float
coefficients baked into the jaxpr, one fused arithmetic expression
over slice views of the halo-extended storage, no ``roll``/gather,
no data-dependent Python branching (the smoothness weighting is
smooth arithmetic, not a ``where``).

Stretched (mapped) factors
--------------------------
On a ``MappedIntervalMesh`` the uniform Shu rows are the wrong FV
weights (the scheme drops to 2nd order), so the biased rows take the
route-(ii) treatment of ``design/research/nonuniform_weno_survey.md``:
the tables are **derived from the cell widths** — Shu 1998 eq. 2.20
candidate rows, exact per-face ideal weights, and Shu's general
smoothness forms — by :func:`nonuniform_tables`, a pure ``jnp``
elementwise generator over the width windows. The widths enter the
kernel as a storage-frame co-operand windowed exactly like the data
(:func:`cell_widths`), so nothing is memoized per face and the
generator traces under a moving geometry; for the static geometry of
today it constant-folds. A uniform factor passes ``widths=None`` and
takes the static table path, **bitwise** unchanged.

The right bias on a stretched factor is still the mirror image, but
the mirror moves from the *table* to the *geometry*: reverse the data
window **and** the width window and run the left-biased generator
(``weno_combine``). The static uniform path keeps the pre-mirrored
tables, so its arithmetic is untouched.
"""
# Wave 4: WenoReconstruction (the upwind pair; the sign selection
#    lives in operators.select)
from __future__ import annotations

import math
from fractions import Fraction
from functools import cache
from itertools import combinations
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, final

import jax
import jax.numpy as jnp

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    _ALGEBRA_TABLE,
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
)
from fridom.spatial.operators.reconstruct import (
    apply_fv_staggered,
)
from fridom.spatial.operators.staggering import (
    mapped_factor,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.spatial.operators.fallback import Fallback
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )

#: regularization of the nonlinear weights (numerics parity with the
#: old stack's ``weno_interpolation.py`` default)
WENO_EPS = 1e-10

#: formal orders grounded in iteration 1 (the r = 2, 3 stencil
#: families of the reference smoothness-indicator tables)
_SUPPORTED_ORDERS = (3, 5)

#: constructor ``boundary`` variants (decision R2 parity): "none" is
#: the periodic-only kernel (today's exact behavior); "graded" mints
#: the bounded-legal graded ``Fallback``. The one-sided-stencil
#: variant ("one_sided", R2) is designed-for and not yet a mode here.
_BOUNDARY_MODES = ("none", "graded")

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
    Coefficient tables of one biased WENO kernel.

    Description
    -----------
    On a **uniform** factor (:func:`weno_tables`) these are plain
    nested tuples of Python floats — hashable static structure baked
    into the jaxpr as constants (never traced). On a **stretched**
    factor (:func:`nonuniform_tables`) the same record holds width-
    derived ``Array`` entries of the window shape instead; ``size``
    and ``offsets`` stay static in both cases, and the kernel body
    (:func:`_weno_combine`) is one expression serving both. Candidate
    ``m`` reads the ``r`` consecutive slice windows starting at
    ``offsets[m]`` within the ``size``-window family.

    Parameters
    ----------
    size : int
        The full window size (= the formal order, 2r - 1).
    offsets : tuple[int, ...]
        Per-candidate first window index within the full window.
    coeffs : tuple[tuple[float | Array, ...], ...]
        Per-candidate reconstruction coefficients (r each).
    optimal : tuple[float | Array, ...]
        The optimal (linear) weights d_m.
    beta_rows : tuple[tuple[tuple[float | Array, ...], ...], ...]
        Per-candidate smoothness-indicator rows: ``beta_m =
        sum_d beta_scale[d] * (beta_rows[m][d] . cells_m)**2``.
    beta_scale : tuple[float, ...]
        The per-row scale factors of the quadratic form (all ``1.0``
        for the non-uniform tables, whose rows carry their own
        scaling).
    """

    size: int
    offsets: tuple[int, ...]
    coeffs: tuple[tuple[float | Array, ...], ...]
    optimal: tuple[float | Array, ...]
    beta_rows: tuple[tuple[tuple[float | Array, ...], ...], ...]
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
    views: tuple[Array, ...], weights: tuple[float | Array, ...],
) -> Array:
    """
    Weighted linear combination of slice views.

    Description
    -----------
    The exact-zero / exact-one shortcuts apply to **Python floats**
    only (the static tables): a width-derived array weight is always
    multiplied in, since ``weight == 0.0`` on an array is an array,
    not a branchable truth value. The static path is therefore
    bitwise what it was.

    Parameters
    ----------
    views : tuple[Array, ...]
        The stencil windows (equal shapes).
    weights : tuple[float | Array, ...]
        The weights; exact float zeros are skipped.

    Returns
    -------
    Array
        The fused weighted sum.
    """
    total = None
    for weight, view in zip(weights, views, strict=True):
        if isinstance(weight, float):
            if weight == 0.0:
                continue
            term = view if weight == 1.0 else weight * view
        else:
            term = weight * view
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
        # a sum of squares: never negative in floating point, so the
        # nonlinear weight needs no ``abs`` guard on either path
        alphas.append(tables.optimal[m] / (beta + WENO_EPS) ** 2)
    return tuple(alphas), tuple(candidates)


def _weno_combine(
    windows: tuple[Array, ...], tables: WenoTables,
) -> Array:
    """
    Nonlinear-weight the candidate stencils of a window family.

    Description
    -----------
    The shared reduction tail of the biased WENO kernel: normalize
    the WENO-JS weights ``alpha_m`` and combine the candidate
    reconstructions ``q_m``. Consumes a ready ``size``-window family
    directly, so both the array entry point (:func:`weno_reconstruct`,
    which slices the windows off a halo-extended storage) and the
    selected-input advection kernel (which builds the windows by a
    per-tap upwind ``where``) run the *same* nonlinear machinery once.

    Parameters
    ----------
    windows : tuple[Array, ...]
        The full-size window family (`_window_views`, or the selected
        taps): one array per stencil cell, equal shapes.
    tables : WenoTables
        The static tables of the biased kernel.

    Returns
    -------
    Array
        The reconstructed face values.
    """
    alphas, candidates = _alpha_candidates(windows, tables)
    total = alphas[0]
    combined = alphas[0] * candidates[0]
    for alpha, candidate in zip(alphas[1:], candidates[1:],
                                strict=True):
        total = total + alpha
        combined = combined + alpha * candidate
    return combined / total


# ================================================================
#  Non-uniform tables (stretched factors, route (ii))
# ================================================================
#: sqrt(13/12): the scale of the top-derivative smoothness row of the
#: ``r = 3`` family, folded into the row so every ``beta_scale`` of a
#: non-uniform table is exactly 1.0 (see :func:`_smoothness_rows`)
_BETA_TOP_SCALE = math.sqrt(13.0 / 12.0)

#: candidate stencil sizes r of the two grounded families
#: (formal order = 2r - 1): r = 2 for WENO-3, r = 3 for WENO-5
_R_WENO3 = 2
_R_WENO5 = 3


def _seal_widths(
    widths: tuple[Array, ...],
) -> tuple[Array, ...]:
    r"""
    Replace non-positive cell widths by unity (the generator seal).

    Description
    -----------
    A bounded axis's storage padding carries **exact-zero** measures
    (never-valid ghost slots), and a window that reaches one would
    divide by zero inside the generator: the forward value is
    discarded (the graded ladder overwrites those faces) but the
    ``inf``/``NaN`` would still reach the reverse pass through the
    ``0 * inf`` VJP and poison ``jax.grad``. Sealing to ``1.0`` makes
    the tables of such a window the *uniform* ones — finite, harmless,
    and bitwise transparent on every valid cell (AGENTS.md
    differentiability policy). The ``jnp.where`` also seals the
    reverse pass w.r.t. the widths themselves.

    Parameters
    ----------
    widths : tuple[Array, ...]
        The per-window cell widths.

    Returns
    -------
    tuple[Array, ...]
        The sealed widths, strictly positive everywhere.
    """
    return tuple(jnp.where(w > 0.0, w, 1.0) for w in widths)


def _local_nodes(
    widths: tuple[Array, ...],
) -> tuple[float | Array, ...]:
    r"""
    Cumulative face positions of a window, ``x_0 = 0``.

    Description
    -----------
    Local (window-relative) coordinates: :math:`x_0 = 0`, :math:`x_k
    = \sum_{j<k} w_j`. Every factor of the generator is then a *sum*
    of widths — no differences of large absolute coordinates, which
    is where the naive spelling lost four digits on a 100 m column
    (``nonuniform_weno_survey.md`` section 4).

    Parameters
    ----------
    widths : tuple[Array, ...]
        The ``k`` cell widths of the window.

    Returns
    -------
    tuple[float | Array, ...]
        The ``k + 1`` local face positions.
    """
    nodes: list[float | Array] = [0.0]
    for width in widths:
        nodes.append(nodes[-1] + width)
    return tuple(nodes)


def _product_derivative(
    nodes: tuple[float | Array, ...], x: float | Array, order: int,
) -> float | Array:
    r"""
    Differentiate :math:`\prod_j (x - x_j)` ``order`` times.

    Description
    -----------
    :math:`\frac{d^s}{dx^s} \prod_{j} (x - x_j) = s! \sum_{|S| = s}
    \prod_{j \notin S} (x - x_j)`, the subsets ``S`` enumerated by a
    **static** Python loop (``len(nodes) <= 5`` here), so the result
    is one fused elementwise expression over the width arrays. Past
    the polynomial's degree the enumeration is empty and the result
    is the exact ``0.0`` it should be.

    Parameters
    ----------
    nodes : tuple[float | Array, ...]
        The product's roots.
    x : float | Array
        The evaluation point.
    order : int
        The derivative order ``s``.

    Returns
    -------
    float | Array
        The derivative value.
    """
    count = len(nodes)
    total: float | Array = 0.0
    for dropped in combinations(range(count), order):
        term: float | Array = 1.0
        for j in range(count):
            if j in dropped:
                continue
            term = term * (x - nodes[j])
        total = total + term
    return math.factorial(order) * total


def _lagrange_derivatives(
    nodes: tuple[float | Array, ...], x: float | Array, order: int,
) -> tuple[float | Array, ...]:
    r"""
    Differentiate the Lagrange basis ``order`` times at ``x``.

    Description
    -----------
    :math:`L_k^{(s)}(x) = \left[\frac{d^s}{dx^s} \prod_{l \neq k}
    (x - x_l)\right] / \prod_{l \neq k} (x_k - x_l)`. Both factors are
    products of node *differences*, i.e. sums of cell widths, so on
    sealed (strictly positive) widths no denominator vanishes.

    Parameters
    ----------
    nodes : tuple[float | Array, ...]
        The interpolation nodes (the window's local faces).
    x : float | Array
        The evaluation point.
    order : int
        The derivative order ``s``.

    Returns
    -------
    tuple[float | Array, ...]
        One value per basis function.
    """
    out: list[float | Array] = []
    for k in range(len(nodes)):
        others = tuple(
            nodes[j] for j in range(len(nodes)) if j != k)
        denom: float | Array = 1.0
        for other in others:
            denom = denom * (nodes[k] - other)
        out.append(_product_derivative(others, x, order) / denom)
    return tuple(out)


def _width_row(
    widths: tuple[Array, ...], deriv: int, x: float | Array,
) -> tuple[float | Array, ...]:
    r"""
    Shu 1998 eq. 2.20 row: :math:`p^{(d)}(x) = \sum_j c_j \bar f_j`.

    Description
    -----------
    The primitive-function Lagrange form. With :math:`P` the degree-``k``
    interpolant of the primitive values :math:`V_m = \sum_{j<m} w_j
    \bar f_j` at the window's faces, the reconstruction is
    :math:`p = P'`, hence

    .. math::
        c_j = w_j \sum_{m=j+1}^{k} L_m^{(d+1)}(x),

    linear in the cell averages with coefficients depending on the
    widths alone. ``deriv = 0`` gives the reconstruction row itself
    (the candidate row at the target face, or the full ``2r-1``-cell
    row); ``deriv >= 1`` gives the derivative rows the smoothness
    indicators are built from.

    Parameters
    ----------
    widths : tuple[Array, ...]
        The window's cell widths (sealed, cell-scaled).
    deriv : int
        The derivative order ``d`` of the reconstructed polynomial.
    x : float | Array
        The evaluation point in the window's local coordinates.

    Returns
    -------
    tuple[float | Array, ...]
        One coefficient per cell of the window.
    """
    nodes = _local_nodes(widths)
    basis = _lagrange_derivatives(nodes, x, deriv + 1)
    row: list[float | Array] = []
    tail: float | Array = 0.0
    for j in range(len(widths) - 1, -1, -1):
        tail = tail + basis[j + 1]
        row.append(widths[j] * tail)
    return tuple(reversed(row))


def _smoothness_rows(
    widths: tuple[Array, ...], lo: float | Array, hi: float | Array,
) -> tuple[tuple[float | Array, ...], ...]:
    r"""
    Square-root factorization of one candidate's smoothness form.

    Description
    -----------
    Shu's general indicator over the **upwind** cell :math:`[lo, hi]`
    of width :math:`w` (which is also the scaling length :math:`D`):

    .. math::
        \beta = \sum_{l=1}^{r-1} D^{2l-1}
                \int_{lo}^{hi} \bigl(p^{(l)}(x)\bigr)^2 dx .

    Expanding each integrand about the cell **midpoint** kills the
    cross terms (:math:`\int (x - x_c)\,dx = 0` over the cell), which
    turns the whole form into a plain sum of squares of linear
    functionals of the data — one row each:

    - ``r = 2`` (:math:`p` linear, :math:`p'` constant):
      :math:`\beta = D w (p')^2 = (D\,p')^2` since :math:`w = D`;
    - ``r = 3`` (:math:`p` quadratic): the :math:`l = 1` term is
      :math:`D\,[w\,p'(x_c)^2 + w^3 (p'')^2 / 12]` and the
      :math:`l = 2` term :math:`D^3 w (p'')^2`, so with :math:`w = D`
      :math:`\beta = (D\,p'(x_c))^2 + \tfrac{13}{12} (D^2 p'')^2` —
      the familiar Jiang-Shu 13/12 and 1/4 coefficients, here as the
      **exact** non-uniform generalization.

    The generator works in cell-scaled coordinates (:math:`D = 1`), so
    the rows are the bare derivative rows with the top one carrying
    ``sqrt(13/12)``. Writing ``beta`` as a sum of squares — rather
    than as a quadratic form ``v^T B v`` — is what keeps it
    non-negative in floating point, so the nonlinear weight needs no
    ``abs`` guard; it is also the *exact-rank* factorization
    (``B`` is rank ``r - 1``: a plain Cholesky would hit a zero pivot
    on the last row).

    Parameters
    ----------
    widths : tuple[Array, ...]
        The candidate's ``r`` cell widths (sealed, cell-scaled).
    lo : float | Array
        The upwind cell's left face, in local coordinates.
    hi : float | Array
        The upwind cell's right face, in local coordinates.

    Returns
    -------
    tuple[tuple[float | Array, ...], ...]
        The ``r - 1`` rows whose squares sum to ``beta``.

    Raises
    ------
    NotImplementedError
        For stencil sizes beyond the grounded ``r = 2, 3`` families.
    """
    mid = 0.5 * (lo + hi)
    first = _width_row(widths, 1, mid)
    if len(widths) == _R_WENO3:
        return (first,)
    if len(widths) == _R_WENO5:
        top = _width_row(widths, 2, mid)
        return (first, tuple(_BETA_TOP_SCALE * c for c in top))
    raise NotImplementedError(  # pragma: no cover — orders are validated
        "the non-uniform smoothness factorization is grounded for "
        f"r = 2, 3 (orders {_SUPPORTED_ORDERS}), got r = "
        f"{len(widths)}")


def nonuniform_tables(
    width_windows: tuple[Array, ...], order: int,
) -> WenoTables:
    r"""
    Derive the **left-biased** WENO tables from the cell widths.

    Description
    -----------
    Route (ii) of ``design/plans/active/high_order_mapped_plan.md``
    (see ``design/research/nonuniform_weno_survey.md``): the
    genuinely non-uniform Shu-1998 tables of one output face,
    computed inside the kernel from the ``2r-1`` cell widths of that
    face's window.

    - **Candidate rows** ``coeffs[m]``: :func:`_width_row` on cells
      ``m .. m + r - 1`` evaluated at the shared output face
      (candidate-local face ``r - m``).
    - **Ideal weights** ``optimal``: from the exact embedding
      :math:`F_i = \sum_m d_m c^{(m)}_{i-m}` of the candidates in the
      full ``2r-1``-cell row ``F``, whose end coefficients each
      involve one candidate only: :math:`d_0 = F_0 / c^{(0)}_0`,
      :math:`d_{r-1} = F_{2r-2} / c^{(r-1)}_{r-1}`, and the middle
      weight by partition of unity (so the weights sum to one to the
      last bit, and a constant is reconstructed exactly). They are
      ratios of products of lengths, hence positive on any monotone
      geometry; the kernel never clips, a test asserts positivity.
    - **Smoothness rows** ``beta_rows[m]``: :func:`_smoothness_rows`
      over the upwind cell (window cell ``r - 1``).

    Everything runs in **cell-scaled local coordinates** (widths
    divided by the upwind cell width, faces measured from the
    window's left edge): the tables are homogeneous of degree zero in
    the widths, so this is exact, and it is what keeps the products
    and quotients well conditioned on a strongly stretched column.

    Pure ``jnp`` elementwise arithmetic over static Python loops — no
    ``jnp.linalg``, no gather, no data-dependent branch — so the
    tables constant-fold for today's static geometry and would trace
    for a moving one.

    Parameters
    ----------
    width_windows : tuple[Array, ...]
        The ``order`` cell-width windows of the output faces, in the
        same window family as the data (`_window_views`).
    order : int
        The odd formal order (3 or 5).

    Returns
    -------
    WenoTables
        The array-valued left-biased tables (``beta_scale`` all 1.0).
    """
    _validate(order, "left")
    if len(width_windows) != order:
        raise ValueError(
            f"the width window family must hold {order} windows (one "
            f"per stencil cell), got {len(width_windows)}")
    r = (order + 1) // 2
    sealed = _seal_widths(width_windows)
    reference = sealed[r - 1]  # the upwind cell: the scaling length
    scaled = tuple(w / reference for w in sealed)

    coeffs = []
    beta_rows = []
    for m in range(r):
        local = scaled[m:m + r]
        nodes = _local_nodes(local)
        coeffs.append(_width_row(local, 0, nodes[r - m]))
        beta_rows.append(
            _smoothness_rows(local, nodes[r - 1 - m], nodes[r - m]))
    full = _width_row(scaled, 0, _local_nodes(scaled)[r])
    d_first = full[0] / coeffs[0][0]
    d_last = full[order - 1] / coeffs[r - 1][r - 1]
    optimal = ((d_first, 1.0 - d_first) if r == _R_WENO3
               else (d_first, 1.0 - d_first - d_last, d_last))
    return WenoTables(
        size=order,
        offsets=tuple(range(r)),
        coeffs=tuple(coeffs),
        optimal=optimal,
        beta_rows=tuple(beta_rows),
        beta_scale=(1.0,) * (r - 1),
    )


def _apply_row(
    windows: tuple[Array, ...], row: tuple[float | Array, ...],
) -> Array:
    """
    Fused weighted sum of a window family under a full row.

    Description
    -----------
    The linear (non-WENO) sibling of :func:`_weighted_sum`: every tap
    is multiplied in, in window order and with the weight on the left,
    so a static row reproduces the advection module's historical
    ``_weighted_windows`` arithmetic **bitwise**.

    Parameters
    ----------
    windows : tuple[Array, ...]
        The window family (equal shapes).
    row : tuple[float | Array, ...]
        One coefficient per window.

    Returns
    -------
    Array
        The fused weighted sum.
    """
    total = None
    for weight, view in zip(row, windows, strict=True):
        term = weight * view
        total = term if total is None else total + term
    return total


@cache
def _static_linear_row(
    order: int, bias: Literal["left", "right"],
) -> tuple[float, ...]:
    """
    Full biased reconstruction row of one linear upwind kernel.

    Description
    -----------
    The optimal-weight combination of the WENO candidate stencils —
    exactly the full ``order``-cell Shu row at the biased face (the
    linear-weight consistency identity), i.e. the old stack's
    ``upwind_interpolation.py`` coefficients.

    Parameters
    ----------
    order : int
        The odd formal order (3 or 5).
    bias : Literal["left", "right"]
        The upwind bias side.

    Returns
    -------
    tuple[float, ...]
        The ``order`` static reconstruction coefficients.
    """
    tables = weno_tables(order, bias)
    row = [0.0] * order
    for m, offset in enumerate(tables.offsets):
        for i, coeff in enumerate(tables.coeffs[m]):
            row[offset + i] += tables.optimal[m] * coeff
    return tuple(row)


@cache
def _static_centered_row(size: int) -> tuple[float, ...]:
    """
    Symmetric even-size interpolation row at the middle interface.

    Parameters
    ----------
    size : int
        The even stencil size (2 or 4 for orders 3 and 5).

    Returns
    -------
    tuple[float, ...]
        The ``size`` static interpolation coefficients.
    """
    return tuple(float(c) for c in _shu_row(size, size // 2))


def weno_combine(
    windows: tuple[Array, ...],
    order: int,
    bias: Literal["left", "right"] = "left",
    width_windows: tuple[Array, ...] | None = None,
) -> Array:
    """
    Nonlinear-weight a window family, uniform or stretched.

    Description
    -----------
    The public form of the shared reduction tail: the array entry
    point (:func:`weno_reconstruct`) and the selected-input advection
    kernel (which builds its taps by a per-tap upwind ``where``) run
    the *same* machinery. ``width_windows=None`` takes the static
    table path — bitwise today's arithmetic; otherwise the tables
    come from :func:`nonuniform_tables`, and the **right** bias is
    served by reversing both families and running the left-biased
    generator (the mirror identity, which on a stretched factor lives
    in the geometry rather than in the table).

    Parameters
    ----------
    windows : tuple[Array, ...]
        The ``order``-window family of the data.
    order : int
        The odd formal order (3 or 5).
    bias : Literal["left", "right"], optional
        The upwind bias side (default: "left").
    width_windows : tuple[Array, ...] | None, optional
        The matching cell-width windows of a stretched factor; None
        selects the static uniform tables (default: None).

    Returns
    -------
    Array
        The reconstructed face values.
    """
    if width_windows is None:
        return _weno_combine(windows, weno_tables(order, bias))
    _validate(order, bias)
    if bias == "right":
        windows = tuple(reversed(windows))
        width_windows = tuple(reversed(width_windows))
    return _weno_combine(
        windows, nonuniform_tables(width_windows, order))


def linear_row_windows(
    windows: tuple[Array, ...],
    order: int,
    bias: Literal["left", "right"] = "left",
    width_windows: tuple[Array, ...] | None = None,
) -> Array:
    """
    Apply the full biased (linear upwind) row to a window family.

    Description
    -----------
    The linear-weighting sibling of :func:`weno_combine`: the full
    ``order``-cell Shu row at the biased face. On a uniform factor
    that is the optimal-weight combination of the WENO candidates
    (:func:`_static_linear_row`); on a stretched one it is the
    ``2r-1``-cell row of :func:`_width_row`, the right bias again
    served by reversing both window families.

    Parameters
    ----------
    windows : tuple[Array, ...]
        The ``order``-window family of the data.
    order : int
        The odd formal order (3 or 5).
    bias : Literal["left", "right"], optional
        The upwind bias side (default: "left").
    width_windows : tuple[Array, ...] | None, optional
        The matching cell-width windows of a stretched factor; None
        selects the static uniform row (default: None).

    Returns
    -------
    Array
        The reconstructed face values.
    """
    if width_windows is None:
        return _apply_row(windows, _static_linear_row(order, bias))
    _validate(order, bias)
    if bias == "right":
        windows = tuple(reversed(windows))
        width_windows = tuple(reversed(width_windows))
    scaled = _rescaled(width_windows, order // 2)
    return _apply_row(
        windows, _width_row(scaled, 0, _local_nodes(scaled)[
            (order + 1) // 2]))


def centered_row_windows(
    windows: tuple[Array, ...],
    size: int,
    width_windows: tuple[Array, ...] | None = None,
) -> Array:
    """
    Apply the symmetric even-size row to a window family.

    Description
    -----------
    The order-coupled velocity interpolation of the biased advection
    modules (sizes 2 and 4): the Shu row on ``size`` cells at the
    middle face ``size // 2``. Uniform: the static ``_shu_row``.
    Stretched: the same row from the cell widths — for ``size = 2``
    the familiar ``(w_1 v_0 + w_0 v_1) / (w_0 + w_1)``.

    Parameters
    ----------
    windows : tuple[Array, ...]
        The ``size``-window family of the data.
    size : int
        The even stencil size.
    width_windows : tuple[Array, ...] | None, optional
        The matching cell-width windows of a stretched factor; None
        selects the static uniform row (default: None).

    Returns
    -------
    Array
        The interpolated face values.
    """
    if width_windows is None:
        return _apply_row(windows, _static_centered_row(size))
    if len(width_windows) != size:
        raise ValueError(
            f"the width window family must hold {size} windows (one "
            f"per stencil cell), got {len(width_windows)}")
    scaled = _rescaled(width_windows, size // 2 - 1)
    return _apply_row(
        windows, _width_row(scaled, 0, _local_nodes(scaled)[size // 2]))


def _rescaled(
    width_windows: tuple[Array, ...], reference: int,
) -> tuple[Array, ...]:
    """
    Seal a width window family and scale it by one of its cells.

    Parameters
    ----------
    width_windows : tuple[Array, ...]
        The cell-width windows.
    reference : int
        The index of the cell whose width becomes the unit length.

    Returns
    -------
    tuple[Array, ...]
        The sealed, cell-scaled widths.
    """
    sealed = _seal_widths(width_windows)
    return tuple(w / sealed[reference] for w in sealed)


# ================================================================
#  The width co-operand (storage frame of the operand)
# ================================================================
#: node sets whose lattice cells are the primal mesh cells (the
#: reconstruction reads cell averages / centered point values)
_PRIMAL_NODE_SETS = (NodeSet.CENTER,)

#: node sets whose lattice cells are the DUAL cells around the mesh
#: faces (the wall-normal direction of the C-grid pair), paired with
#: the axis topology each is the reconstruction frame of: ``Right``
#: holds faces 1..n of a periodic axis, ``Inner`` the interior faces
#: 1..n-1 of a bounded one (its two wall faces are ghost slots)
_DUAL_NODE_SETS = {NodeSet.RIGHT: False, NodeSet.INNER: True}


def cell_widths(f: FieldLike, axis: str) -> Array | None:
    r"""
    Materialize the lattice-cell widths of ``f``'s factor.

    Description
    -----------
    The geometry co-operand of the non-uniform biased rows: a storage
    array in **``f._data``'s frame** (same length along ``axis``,
    size-1 on every other axis, halo-extended and sharded exactly like
    the data), so a caller windows it with the very same offsets it
    windows the data with. ``None`` on a uniform factor — the callers
    then take the static-table path, bitwise unchanged.

    Which measure is the lattice cell depends on the operand's node
    set, exactly as in the reconstruction's own geometry:

    - **primal** factors (``CellAvg``, ``Center``): the primal cell
      widths, i.e. ``grid.measure`` on the field's own space;
    - **dual** factors (``Right`` on a periodic axis, ``Inner`` on a
      bounded one): the dual cells around the faces. ``Right`` holds
      faces ``1 .. n`` and its measure is already the frame; ``Inner``
      holds faces ``1 .. n-1`` as true DOFs and the two **wall** faces
      ``0`` and ``n`` as ghost slots, whose lattice cells are the
      clipped half cells ``x_c[0] - x_min`` and ``x_max - x_c[n-1]``
      (the graded ladder synthesizes the wall *values* as zeros but
      must read their real *widths*), so those two ghost slots are
      filled here through the decomposition's physical-end seam.

    The result is synced (a periodic axis's ghost slots hold the wrap
    fill, which is the measure's exact periodic extension) and sealed
    to be strictly positive, so no window can carry a zero width into
    the generator.

    Parameters
    ----------
    f : FieldLike
        The operand field.
    axis : str
        The resolved coordinate axis.

    Returns
    -------
    Array | None
        The storage-frame cell widths, or None on a uniform factor.

    Raises
    ------
    SpaceMismatchError
        If the operand's node set is not one of the reconstruction
        families (primal cells or the dual face cells).
    ValueError
        If a bounded dual frame has no ghost slot to carry the wall
        half cells (a halo-0 axis).
    """
    space = f.function_space
    factor = space.bare.factor(axis)
    if not mapped_factor(factor):
        return None
    dual_wall = _classify_frame(factor, axis)
    grid = f.grid
    widths = grid.sync(grid.measure(space, name=axis))
    data = widths._data  # noqa: SLF001 — documented storage seam
    if dual_wall:
        data = _fill_wall_cells(f, widths, axis, data)
    return jnp.where(data > 0.0, data, 1.0)


def _classify_frame(factor: FunctionSpace, axis: str) -> bool:
    """
    Validate the operand's node set; flag the bounded dual frame.

    Parameters
    ----------
    factor : FunctionSpace
        The bare 1D factor space of the operand.
    axis : str
        The coordinate axis (error attribution).

    Returns
    -------
    bool
        True iff the lattice cells are the dual face cells of a
        **bounded** axis (the ``Inner`` frame, whose two wall cells
        live in ghost slots).

    Raises
    ------
    SpaceMismatchError
        On a node set outside the reconstruction families.
    """
    bounded = not getattr(factor.mesh, "periodic", False)
    if isinstance(factor, CellAvg):
        return False
    if isinstance(factor, NodalSpace):
        if factor.node_set in _PRIMAL_NODE_SETS:
            return False
        # a dual node set is the reconstruction frame of exactly one
        # topology; the other pairing (bounded ``Right``, periodic
        # ``Inner``) is not a C-grid frame and would put the wall
        # cells in the wrong slots, so it refuses rather than guesses
        if _DUAL_NODE_SETS.get(factor.node_set) is bounded:
            return bounded
    raise SpaceMismatchError(
        f"no stretched-mesh cell widths for {factor!r} along "
        f"{axis!r}: the biased reconstructions read primal cells "
        "(CellAvg, Center) or the dual face cells (Right on a "
        "periodic axis, Inner on a bounded one)",
        left=factor, operation="reconstruct")


def _wall_half_cells(mesh: object) -> tuple[float, float]:
    """
    Evaluate the clipped dual cells of the two wall faces.

    Description
    -----------
    ``x_c[0] - x_min`` and ``x_max - x_c[n-1]``: the halves of the
    first and last primal cell that the wall faces own — the same
    boundary-member weights ``grid.measure`` puts on an ``Outer``
    space, evaluated here directly off the coordinate map (the
    ``reconstruct._wall_face_weights`` precedent: a host evaluation of
    the map at static computational positions).

    Parameters
    ----------
    mesh : object
        The bounded mapped 1D mesh.

    Returns
    -------
    tuple[float, float]
        The (left, right) wall dual-cell widths.
    """
    coord = mesh.coordinate_map
    n = mesh.n_cells
    x_min, x_max = mesh.extent
    first = float(coord(jnp.asarray(0.5 / n)))
    last = float(coord(jnp.asarray((n - 0.5) / n)))
    return (first - x_min, x_max - last)


def _fill_wall_cells(
    f: FieldLike, widths: FieldLike, axis: str, data: Array,
) -> Array:
    """
    Write the wall half cells into the two ``Inner`` ghost slots.

    Description
    -----------
    In the ``Inner`` frame the lattice cell of face ``F`` sits at
    storage slot ``width + F - 1``, so the wall faces ``0`` and ``n``
    land one slot outside the true DOFs on each side. The write goes
    through ``Decomposition.patch_physical_ends``, so it lands on the
    boundary shards' blocks and is correct whether the axis is
    undistributed or sharded.

    Parameters
    ----------
    f : FieldLike
        The operand field (supplies the grid and the layout).
    widths : FieldLike
        The synced measure field on the operand's frame.
    axis : str
        The resolved coordinate axis.
    data : Array
        The measure field's storage array.

    Returns
    -------
    Array
        ``data`` with the two wall ghost slots holding the half cells.

    Raises
    ------
    ValueError
        If the axis carries no ghost slot to write into.
    """
    space = widths.function_space
    axis_index = space.names.index(axis)
    decomposition = f.grid.decomposition
    if decomposition.halo[axis] < 1:
        raise ValueError(
            f"the bounded dual reconstruction along {axis!r} needs a "
            "halo of at least one slot to carry the wall dual-cell "
            "widths (the wall faces are ghost slots of the Inner "
            f"frame), but the negotiated halo is "
            f"{decomposition.halo[axis]}")
    halves = _wall_half_cells(f.function_space.bare.factor(axis).mesh)

    def patch(
        in_block: Array,  # noqa: ARG001 — the write is value-only
        out_block: Array,
        side: int,
        width_in: int,  # noqa: ARG001 — same frame as the output
        t_in: int | Array,  # noqa: ARG001 — same frame as the output
        width_out: int,
        t_out: int | Array,
    ) -> Array:
        """Write one wall's half cell into its ghost slot."""
        template = jax.lax.dynamic_slice_in_dim(
            out_block, 0, 1, axis_index)
        value = jnp.full_like(template, halves[side])
        slot = width_out - 1 if side == 0 else width_out + t_out
        return jax.lax.dynamic_update_slice_in_dim(
            out_block, value, slot, axis_index)

    return decomposition.patch_physical_ends(
        data, data, space, space, axis, patch, layout=space.layout)


def weno_reconstruct(
    arr: Array,
    axis: int,
    order: int = 5,
    bias: Literal["left", "right"] = "left",
    widths: Array | None = None,
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
    widths : Array | None, optional
        The lattice-cell widths of a stretched factor, in ``arr``'s
        storage frame (:func:`cell_widths`): windowed exactly like
        ``arr`` and turned into per-face tables by
        :func:`nonuniform_tables`. None takes the static uniform
        tables (default: None).

    Returns
    -------
    Array
        The reconstructed face values; ``axis`` shrinks by
        ``order - 1``.
    """
    if widths is None:
        tables = weno_tables(order, bias)
        windows = _window_views(arr, axis, tables.size)
        return _weno_combine(windows, tables)
    _validate(order, bias)
    return weno_combine(
        _window_views(arr, axis, order), order, bias,
        _window_views(widths, axis, order))


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
    ``weno_interpolation.py``); the bare kernel's bounded-axis
    boundary biasing is designed-for, so a bounded registration is a
    space error, never a silent fallback. A stretched (mapped)
    factor is accepted: the rows are then derived from the cell
    widths inside the kernel (:func:`nonuniform_tables`, fed by
    :func:`cell_widths`), so the design order survives the
    stretching; a uniform factor keeps the static tables bitwise.
    Nonlinear, hence no ``eigenvalues`` (the raising base is correct
    and automatic).

    The ``boundary`` knob (decision R2 parity) is a constructor
    variant, not per-application state: ``boundary="none"`` (default)
    is the periodic-only kernel above; ``boundary="graded"`` does
    **not** build a ``WenoReconstruction`` at all — it mints and
    returns the bounded-legal graded :class:`~.fallback.Fallback`
    (``CellAvg -> Inner``) via :func:`~.fallback.graded_reconstruction`,
    retiring the periodic-only restriction by construction. The
    "one_sided" variant (R2) is designed-for.

    Registration / override usage (sketch 4.2): an advection module
    installs the wide graded row under the ``"reconstruct"`` kind by
    merging the constructor variant, e.g.

    .. code-block:: python

        grid.dispatch.merge({
            ("reconstruct", mesh.cell_avg):
                WenoReconstruction(5, boundary="graded")})

    so bounded and periodic axes both resolve a legal reconstruction.

    ``order`` is the order of the **reconstruction**, and that is all
    it is: a flux-form advection scheme that multiplies this face
    value by a face velocity and differences the product contributes a
    formally 2nd-order tendency whenever the advecting velocity varies
    along the flux axis (the product-rule / deconvolution mismatch —
    the high-order face quantity is the deconvolved *flux*, not the
    deconvolved state). See ``fridom.model.modules.advection``.

    Parameters
    ----------
    order : int, optional
        The odd formal order; iteration 1 grounds 3 and 5
        (default: 5).
    bias : Literal["left", "right"], optional
        The upwind bias side of the reconstruction
        (default: "left").
    boundary : Literal["none", "graded"], optional
        Construction variant (default: "none"). "none" is the
        periodic-only kernel; "graded" returns the bounded-legal
        graded ``Fallback`` (a different class — ``__init__`` is then
        skipped).
    """

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __new__(
        cls,
        order: int = 5,
        bias: Literal["left", "right"] = "left",
        boundary: Literal["none", "graded"] = "none",
    ) -> WenoReconstruction | Fallback:
        """
        Dispatch the ``boundary`` variant (R2 parity, sketch 4.2).

        Description
        -----------
        ``boundary="none"`` (default) returns the interned plain kernel
        for ``(order, bias)`` (D6 self-intern) — today's exact
        periodic-only kernel, unchanged in behavior. ``boundary="graded"``
        returns the graded :class:`~.fallback.Fallback` built by
        :func:`~.fallback.graded_reconstruction`; because the leaf rungs
        now self-intern, that ``Fallback`` coalesces on its own structure,
        so the graded spelling is already a stable interned handle (no
        separate memo needed). Since it is a different class, Python skips
        ``__init__`` for it. The ``copy.copy`` seam of ``_rebind`` never
        reaches this path: ``__copy__`` takes precedence and returns a
        fresh mutable clone, so the plain path can intern unconditionally.

        Parameters
        ----------
        order : int, optional
            The odd formal order (default: 5).
        bias : Literal["left", "right"], optional
            The upwind bias side (default: "left").
        boundary : Literal["none", "graded"], optional
            The construction variant (default: "none").

        Returns
        -------
        WenoReconstruction | Fallback
            A plain kernel for "none", the graded ``Fallback`` for
            "graded".

        Raises
        ------
        ValueError
            On an unknown ``boundary`` mode, or an invalid
            ``order``/``bias``.
        """
        if boundary not in _BOUNDARY_MODES:
            raise ValueError(
                f"boundary must be one of {_BOUNDARY_MODES}: 'none' "
                "is the periodic-only kernel, 'graded' mints a "
                "bounded-legal Fallback ('one_sided' is designed-for)"
                f", got {boundary!r}")
        _validate(order, bias)
        if boundary == "graded":
            from fridom.spatial.operators.fallback import (  # noqa: PLC0415 — import-cycle seam (fallback imports weno)
                graded_reconstruction,
            )
            return graded_reconstruction(order, bias)

        def build() -> WenoReconstruction:
            obj = super(WenoReconstruction, cls).__new__(cls)
            obj._order = order
            obj._bias = bias
            return obj

        return _ALGEBRA_TABLE.intern((cls, order, bias), build)

    def __init__(
        self,
        order: int = 5,
        bias: Literal["left", "right"] = "left",
        boundary: Literal["none", "graded"] = "none",
    ) -> None:
        """No-op: attributes are set in the interning ``__new__``.

        Only ``boundary="none"`` reaches ``__init__`` on a
        ``WenoReconstruction`` at all (the "graded" variant returns a
        ``Fallback`` from ``__new__``); it must not re-validate or
        clobber the interned singleton's attributes.
        """

    def __copy__(self) -> WenoReconstruction:
        """Fresh shallow clone bypassing interning (the ``_rebind`` seam).

        ``SeparableOperator._rebind`` does ``copy.copy(base)`` then
        mutates the clone's ``bound_axis``; without this the copy would
        reconstruct via the interning ``__new__`` and hand back the
        shared unbound singleton, which ``_rebind`` would then corrupt.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

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
        variant is designed-for and raises. Stretched (mapped)
        factors resolve like uniform ones; the kernel derives its
        rows from the cell widths (:func:`cell_widths`).

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
        (left) / ``order // 2 - 1`` (right) — passed to the shared
        ``apply_fv_staggered`` tail as its explicit ``align``.

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
        size = self._order
        bias = self._bias
        m0 = size // 2 if bias == "left" else size // 2 - 1
        widths = cell_widths(f, axis)

        def kernel(arr: Array, axis_index: int, *co: Array) -> Array:
            return weno_reconstruct(
                arr, axis_index, order=size, bias=bias,
                widths=co[0] if co else None)

        return apply_fv_staggered(
            self, f, axis, size, kernel, metadata=f.metadata, align=m0,
            co_operands=() if widths is None else (widths,))
