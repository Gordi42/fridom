r"""
Generic slaving-expansion core for balance recursions (private).

Description
-----------
Array-agnostic implementation of the balance (nonlinear normal mode
decomposition) recursion of ``notes/framework2/nnmd_design_note.md``
S1.2, for systems

.. math::

    \partial_t \phi = L\phi + B(\phi, \phi)

with a declared slow projector :math:`V` (not necessarily
:math:`\ker L`), fast projector :math:`W = I - V`, and
:math:`L_w = LW` invertible on :math:`\mathrm{ran}\,W`. The operators
are injected as plain callables (:class:`SlavingOps`), so the same
core runs on toy ODEs (numpy vectors) and on framework2 states.

Two partial-sum organizations are implemented:

- ``scheme="direct"`` -- the doubly indexed recursion (series order
  ``n``, slow-derivative order ``k``) of the design note:

  .. math::

      \phi_0^{(k)} &= L\phi_0^{(k-1)} + V\sum_m \binom{k-1}{m}
          B(\phi_0^{(m)}, \phi_0^{(k-1-m)}) \\
      \phi_{n+1}^{(k)} &= L_w^{-1} W\Big[\phi_n^{(k+1)}
          - \sum_m \binom{k}{m} I_n^{(m,k-m)}\Big], \quad
      I_n^{(a,b)} = \sum_{j=0}^n B(\phi_j^{(a)}, \phi_{n-j}^{(b)})

  with balanced state :math:`z_b = \sum_n \phi_n^{(0)}`,
  :math:`\phi_0 = Vz`.

- ``scheme="telescoping"`` -- the resummed organization of the vault
  note "NNMD - Generalized v2":

  .. math::

      \psi_m = -\sum_{n=0}^{m-1} L_w^{-(n+1)} W \sum_{k=0}^n
          \binom{n}{k}\, J_{m-1-n}(k, n-k), \quad
      J_q(a,b) = \sum_{j=0}^q B(\zeta_j^{(a)}, \zeta_{q-j}^{(b)})

  where :math:`\zeta_m^{(n)}` is the order-``m`` part of the ``n``-th
  slow-flow derivative of the balanced state, with the derivative
  recursion

  .. math::

      \zeta_m^{(n)} = L\,V\zeta_m^{(n-1)} + L\,W\zeta_{m+1}^{(n-1)}
          + \sum_{k=0}^{n-1} \binom{n-1}{k} J_m(k, n-1-k).

  Note: the reference implementation applies the full ``L`` to slot
  ``m+1`` plus a ``m == 0`` extra term; the split form above is the
  correct generalization for non-stationary slow spaces (``LV != 0``)
  and coincides with it when ``LV = 0`` -- see the P1 report.

  M2/R2 finding (toy harness, machine precision): with this
  derivative line the telescoping series is an exact term-by-term
  resummation of the direct recursion -- the two truncations
  coincide identically at every order, for both closures, with and
  without slow detuning. The historical "telescoping worse at order
  >= 3" observation cannot be a property of consistent
  implementations of either scheme.

Closure bookkeeping (design note S1.4)
--------------------------------------
- ``closure="leading"`` (v1 bookkeeping): slow derivatives are driven
  by the leading-order slow flow only,
  :math:`\dot v \approx Lv + VB(v, v)`; the slow tendency has no
  series corrections. In the telescoping scheme this corresponds to
  dropping the slow part of the derivative table entries with series
  order :math:`m \ge 1` (``W``-filtering :math:`\zeta_m^{(n)}`).
- ``closure="consistent"``: order-consistent closure. Slow-mode terms
  get their own series index: the slow derivative table
  :math:`S_p^{(k)}` (order-``p`` part of :math:`d^k v/dt^k`, with
  :math:`S_p^{(0)} = \delta_{p0} v`) obeys

  .. math::

      S_p^{(k)} = L S_p^{(k-1)} + V \sum_{m=0}^{k-1} \binom{k-1}{m}
          \sum_{a+b=p} B(\Phi_a^{(m)}, \Phi_b^{(k-1-m)})

  where :math:`\Phi_a^{(m)} = S_a^{(m)} + F_a^{(m)}` and
  :math:`F_p^{(k)}` is the fast table. The wave feedback
  :math:`V\sum_{j+k=n} B(\phi_j, \phi_k)` thereby enters the slow
  tendency at matching series order (it first changes the balanced
  state at order 3). The fast recursion keeps the same shape with
  :math:`\Phi` in every slot; ``closure="leading"`` is the special
  case :math:`S_{p\ge 1}^{(k)} = 0`.

All tables are memoized per call; binomials via :func:`math.comb`.
The module is dependency-free (stdlib only) and private -- it is not
exported from ``fridom.framework2.transforms``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


# ================================================================
#  Injected operators and result container
# ================================================================
@dataclass(frozen=True)
class SlavingOps:

    """
    The injected operator set for :func:`balance_expansion`.

    Description
    -----------
    All callables act on an opaque vector type supporting ``+``, ``-``
    and multiplication by Python scalars (plain numpy arrays for the
    toy harness, framework2 states later).

    Attributes
    ----------
    slow : Callable
        The slow projector ``V``.
    fast : Callable
        The fast projector ``W = I - V``.
    inv_lw : Callable
        ``L_w**-1`` on the fast subspace (single application; integer
        powers are formed by repeated application).
    bilinear : Callable
        The symmetric bilinear form ``B(x, y)`` of the quadratic term.
    lin : Callable
        The full linear operator ``L`` (used for the slow-rotation
        terms ``L phi_0^(k)``; for stationary slow spaces
        ``V . lin . V = 0`` and the term drops out numerically).
    """

    slow: Callable[[Any], Any]
    fast: Callable[[Any], Any]
    inv_lw: Callable[[Any], Any]
    bilinear: Callable[[Any, Any], Any]
    lin: Callable[[Any], Any]


@dataclass(frozen=True)
class BalanceResult:

    """
    The result of :func:`balance_expansion`.

    Attributes
    ----------
    state : Any
        The balanced state ``z_b = sum_n phi_n``.
    terms : tuple
        The per-order terms ``(phi_0, ..., phi_N)`` (diagnostics; the
        last entry of an order ``N + 1`` call is the series residual).
    """

    state: Any
    terms: tuple[Any, ...]


# ================================================================
#  Direct scheme (design note S1.2)
# ================================================================
class _DirectExpansion:

    """Memoized doubly indexed recursion (series x derivative)."""

    def __init__(self, base: Any, ops: SlavingOps,
                 *, consistent: bool) -> None:
        self._base = base
        self._ops = ops
        self._consistent = consistent
        self._zero = 0.0 * base
        self._slow_cache: dict[tuple[int, int], Any] = {}
        self._fast_cache: dict[tuple[int, int], Any] = {}

    def term(self, n: int) -> Any:
        """Return the series term ``phi_n`` (the ``k = 0`` column)."""
        if n == 0:
            return self._base
        return self._fast(n, 0)

    def _full(self, n: int, k: int) -> Any:
        """Return ``Phi_n^(k) = S_n^(k) + F_n^(k)``."""
        if n == 0:
            return self._slow(0, k)
        return self._slow(n, k) + self._fast(n, k)

    def _slow(self, p: int, k: int) -> Any:
        """Return ``S_p^(k)``, the slow-derivative table entry."""
        key = (p, k)
        if key not in self._slow_cache:
            self._slow_cache[key] = self._slow_value(p, k)
        return self._slow_cache[key]

    def _slow_value(self, p: int, k: int) -> Any:
        if k == 0:
            return self._base if p == 0 else self._zero
        if p > 0 and not self._consistent:
            return self._zero
        ops = self._ops
        rotation = ops.lin(self._slow(p, k - 1))
        acc = self._zero
        for m in range(k):
            coef = float(math.comb(k - 1, m))
            for a in range(p + 1):
                acc = acc + coef * ops.bilinear(
                    self._full(a, m), self._full(p - a, k - 1 - m))
        return rotation + ops.slow(acc)

    def _fast(self, p: int, k: int) -> Any:
        """Return ``F_p^(k)``, the fast-correction table entry."""
        key = (p, k)
        if key not in self._fast_cache:
            self._fast_cache[key] = self._fast_value(p, k)
        return self._fast_cache[key]

    def _fast_value(self, p: int, k: int) -> Any:
        ops = self._ops
        derivative = self._full(p - 1, k + 1)
        acc = self._zero
        for m in range(k + 1):
            coef = float(math.comb(k, m))
            for a in range(p):
                acc = acc + coef * ops.bilinear(
                    self._full(a, m), self._full(p - 1 - a, k - m))
        return ops.inv_lw(ops.fast(derivative - acc))


# ================================================================
#  Telescoping scheme (vault note "NNMD - Generalized v2")
# ================================================================
class _TelescopingExpansion:

    """Memoized telescoping recursion (``L_w**-(n+1)`` powers)."""

    def __init__(self, base: Any, ops: SlavingOps,
                 *, consistent: bool) -> None:
        self._base = base
        self._ops = ops
        self._consistent = consistent
        self._zero = 0.0 * base
        self._cache: dict[tuple[int, int], Any] = {}

    def term(self, m: int) -> Any:
        """Return the series term ``psi_m``."""
        return self._zeta(m, 0)

    def _zeta(self, m: int, n: int) -> Any:
        """Return ``zeta_m^(n)`` (order-``m`` part, derivative ``n``)."""
        key = (m, n)
        if key not in self._cache:
            if n == 0:
                self._cache[key] = self._series_value(m)
            else:
                self._cache[key] = self._derivative_value(m, n)
        return self._cache[key]

    def _series_value(self, m: int) -> Any:
        if m == 0:
            return self._base
        ops = self._ops
        total = self._zero
        for n in range(m):
            inner = self._zero
            for k in range(n + 1):
                coef = float(math.comb(n, k))
                inner = inner + coef * self._j(m - 1 - n, k, n - k)
            step = ops.fast(inner)
            for _ in range(n + 1):
                step = ops.inv_lw(step)
            total = total + step
        return -1.0 * total

    def _derivative_value(self, m: int, n: int) -> Any:
        ops = self._ops
        value = (ops.lin(ops.slow(self._zeta(m, n - 1)))
                 + ops.lin(ops.fast(self._zeta(m + 1, n - 1))))
        for k in range(n):
            coef = float(math.comb(n - 1, k))
            value = value + coef * self._j(m, k, n - 1 - k)
        if m >= 1 and not self._consistent:
            # leading closure: no slow-series corrections in the
            # derivative table (v1 bookkeeping)
            value = ops.fast(value)
        return value

    def _j(self, q: int, a: int, b: int) -> Any:
        """Return the order-``q`` bilinear collection ``J_q(a, b)``."""
        total = self._zero
        for j in range(q + 1):
            total = total + self._ops.bilinear(
                self._zeta(j, a), self._zeta(q - j, b))
        return total


# ================================================================
#  Public entry point
# ================================================================
def balance_expansion(
    z: Any,
    *,
    order: int,
    ops: SlavingOps,
    scheme: Literal["direct", "telescoping"] = "direct",
    closure: Literal["leading", "consistent"] = "leading",
) -> BalanceResult:
    """
    Balance ``z`` by the slaving expansion at the given order.

    Description
    -----------
    Computes the balanced state ``z_b = sum_{n=0}^{order} phi_n``
    with base point ``phi_0 = V z`` (the fast content of ``z`` is
    discarded) by the recursion stated in the module docstring
    (``notes/framework2/nnmd_design_note.md`` S1.2). All tables are
    memoized per call.

    Parameters
    ----------
    z : Any
        The state to balance (any vector supporting ``+``, ``-`` and
        scalar multiplication).
    order : int
        The truncation order ``N >= 0``; ``0`` returns ``V z``.
    ops : SlavingOps
        The injected operator set.
    scheme : {"direct", "telescoping"}, optional
        The partial-sum organization (default: "direct").
    closure : {"leading", "consistent"}, optional
        The slow-tendency closure, see the module docstring
        (default: "leading").

    Returns
    -------
    BalanceResult
        The balanced state and the per-order terms.

    Raises
    ------
    ValueError
        On a negative order or an unknown scheme/closure.
    """
    if isinstance(order, bool) or not isinstance(order, int) \
            or order < 0:
        raise ValueError(
            f"order must be a non-negative int, got {order!r}")
    if scheme not in ("direct", "telescoping"):
        raise ValueError(
            f"scheme must be 'direct' or 'telescoping', got {scheme!r}")
    if closure not in ("leading", "consistent"):
        raise ValueError(
            f"closure must be 'leading' or 'consistent', "
            f"got {closure!r}")
    base = ops.slow(z)
    consistent = closure == "consistent"
    expansion: _DirectExpansion | _TelescopingExpansion
    if scheme == "direct":
        expansion = _DirectExpansion(base, ops, consistent=consistent)
    else:
        expansion = _TelescopingExpansion(base, ops,
                                          consistent=consistent)
    terms = tuple(expansion.term(n) for n in range(order + 1))
    state = terms[0]
    for term in terms[1:]:
        state = state + term
    return BalanceResult(state=state, terms=terms)
