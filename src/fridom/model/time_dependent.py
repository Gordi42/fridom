"""
Time-dependent scalar values: ``TimeDependent``, ``Ramp``, ``resolve_at``.

Description
-----------
Owning class doc: ``design/specs/model/classes/declarations.md``
(section "TimeDependent, fr.Ramp, and resolve_at"). A time-dependent
scalar value is a frozen callable pytree, never a module: it
*describes* a curve and never writes (D2.2). Scalars get time
dependence via ``resolve_at`` at the point of use; field-consumed
parameters get it via the owner's ``self_update`` rewriting the
AUXILIARY field. ``Ramp``'s static/dynamic split is the point:
changing the ``curve`` is different math — exactly one recompile;
sweeping endpoints/timing (``v0``/``v1``/``t0``/``period``, dynamic
leaves) never recompiles. Evaluation is branch-free (``jnp.clip`` +
shape), valid at every scan step. Endpoints are SIGNED times
(02_rules V-S2): a backward Ramp spans ``[-T, 0]``; ``reversed()``
reflects the time domain — a naive value-endpoint swap over
``[0, T]`` clips to a constant for t <= 0.
"""
# Wave 2 B: TimeDependent, Ramp, resolve_at
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Final

import jax
import jax.numpy as jnp

from fridom.framework.utils import jaxify

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


def _leaf_repr(value: object) -> str:
    """Format a dynamic leaf for repr (plain scalar if concrete)."""
    item = getattr(value, "item", None)
    if item is None:
        return repr(value)
    try:
        return repr(item())
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return repr(value)


# ================================================================
#  TimeDependent (the abstract curve)
# ================================================================
class TimeDependent(ABC):

    """
    A pure, branch-free scalar time curve; describes, never writes.

    Description
    -----------
    Inside the trace, evaluation happens at stage time through
    ``ctx.params`` (``eval_params`` per substage). Host-side,
    ``model.parameters`` returns the object itself — ``at_time(t)``
    is the sanctioned explicit spelling; assembly-time (``bind``)
    bare reads raise unless spelled ``at_time(0.0)`` (D2.4). Scalar
    composition (needed by OptimalBalance) returns derived
    `TimeDependent` nodes; there is deliberately NO field/array
    arithmetic — a consumer forgetting ``resolve_at`` fails loudly
    in the assembly dry run.
    """

    @abstractmethod
    def __call__(self, t: jax.Array | float) -> jax.Array:
        """Traced evaluation at clock time ``t``; valid for all t."""

    def at_time(self, t: float) -> jax.Array:
        """
        Evaluate explicitly at a host/assembly-side time.

        Parameters
        ----------
        t : float
            The (signed) clock time to evaluate at.

        Returns
        -------
        jax.Array
            The curve value at ``t``.
        """
        return self(t)

    # ================================================================
    #  Scalar composition (derived nodes; no field/array arithmetic)
    # ================================================================
    def __mul__(self, other: complex) -> TimeDependent:
        """Scale the curve by a scalar."""
        if not isinstance(other, Number):
            return NotImplemented
        return _Affine(self, scale=other, offset=0.0)

    def __rmul__(self, other: complex) -> TimeDependent:
        """Scale the curve by a scalar (reflected)."""
        if not isinstance(other, Number):
            return NotImplemented
        return _Affine(self, scale=other, offset=0.0)

    def __add__(self, other: complex) -> TimeDependent:
        """Shift the curve by a scalar."""
        if not isinstance(other, Number):
            return NotImplemented
        return _Affine(self, scale=1.0, offset=other)

    def __radd__(self, other: complex) -> TimeDependent:
        """Shift the curve by a scalar (reflected)."""
        if not isinstance(other, Number):
            return NotImplemented
        return _Affine(self, scale=1.0, offset=other)

    def __sub__(self, other: complex) -> TimeDependent:
        """Shift the curve by the negated scalar."""
        if not isinstance(other, Number):
            return NotImplemented
        return _Affine(self, scale=1.0, offset=-other)

    def __rsub__(self, other: complex) -> TimeDependent:
        """Reflected subtraction: ``other - self``."""
        if not isinstance(other, Number):
            return NotImplemented
        return _Affine(self, scale=-1.0, offset=other)

    def __neg__(self) -> TimeDependent:
        """Negate the curve."""
        return _Affine(self, scale=-1.0, offset=0.0)


@partial(jaxify, dynamic=("_inner", "_scale", "_offset"))
class _Affine(TimeDependent):

    """
    Derived composition node: ``scale * inner(t) + offset``.

    Description
    -----------
    All three slots are dynamic pytree children (``_inner`` is
    itself a pytree, e.g. a `Ramp`), so composed curves keep the
    zero-recompile sweep property of their parts.

    Parameters
    ----------
    inner : TimeDependent
        The wrapped curve.
    scale : complex
        Multiplicative factor (dynamic leaf).
    offset : complex
        Additive shift (dynamic leaf).
    """

    def __init__(
        self,
        inner: TimeDependent,
        scale: complex,
        offset: complex,
    ) -> None:
        """Coerce the scalars to array leaves; see class docstring."""
        self._inner = inner
        self._scale = jnp.asarray(scale)
        self._offset = jnp.asarray(offset)

    def __call__(self, t: jax.Array | float) -> jax.Array:
        """Evaluate ``scale * inner(t) + offset``."""
        return self._scale * self._inner(t) + self._offset

    def __repr__(self) -> str:
        """Composition repr, e.g. ``(2.0 * Ramp(...) + 1.0)``."""
        return (f"({_leaf_repr(self._scale)} * {self._inner!r}"
                f" + {_leaf_repr(self._offset)})")


# ================================================================
#  Named curves (static shape functions; shape(0)=0, shape(1)=1)
# ================================================================
def _curve_linear(s: jax.Array) -> jax.Array:
    """Linear shape: ``f(s) = s``."""
    return s


def _curve_cosine(s: jax.Array) -> jax.Array:
    """Cosine shape: ``f(s) = (1 - cos(pi s)) / 2``."""
    return 0.5 * (1.0 - jnp.cos(jnp.pi * s))


# floor keeping the exponential shape branch-free at s in {0, 1}
# (the old Ramper's max(1e-32, theta), spelled with jnp.maximum)
_EXP_FLOOR: Final[float] = 1e-32


def _curve_exp(s: jax.Array) -> jax.Array:
    """Exponential shape: ``e^(-1/s) / (e^(-1/s) + e^(-1/(1-s)))``."""
    e0 = jnp.exp(-1.0 / jnp.maximum(s, _EXP_FLOOR))
    e1 = jnp.exp(-1.0 / jnp.maximum(1.0 - s, _EXP_FLOOR))
    return e0 / (e0 + e1)


_NAMED_CURVES: Final[dict[str, Callable[[jax.Array], jax.Array]]] = {
    "linear": _curve_linear,
    "cosine": _curve_cosine,
    "exp": _curve_exp,
}


# ================================================================
#  Ramp
# ================================================================
@partial(jaxify, dynamic=("v0", "v1", "t0", "period"))
class Ramp(TimeDependent):

    """
    The ramp curve ``v0 + (v1 - v0) * shape(clip((t-t0)/period))``.

    Description
    -----------
    ``curve`` is STATIC (a curve change is different math — exactly
    one recompile); the endpoints and timing are dynamic leaves
    (sweeps never recompile). A scalar <-> Ramp swap in a slot
    changes the treedef: one recompile plus a restart-fingerprint
    change (the fingerprint treats the Ramp shape as structure, its
    endpoints as leaves — 02_rules). Evaluation is branch-free and
    valid at every scan step. ``t0``/``period`` are SIGNED times
    (V-S2): a backward Ramp spans ``[-T, 0]`` via ``t0=-T``.

    Parameters
    ----------
    v0 : float
        Value before the window (dynamic leaf).
    v1 : float
        Value after the window (dynamic leaf).
    period : float
        Signed window duration (dynamic leaf); keyword-only.
    t0 : float, optional
        Signed window start time (dynamic leaf; default: 0.0).
    curve : str | Callable, optional
        STATIC shape: a named curve ("linear" | "cosine" | "exp")
        or a callable with shape(0)=0, shape(1)=1
        (default: "linear").
    """

    def __init__(
        self,
        v0: float,
        v1: float,
        *,
        period: float,
        t0: float = 0.0,
        curve: str | Callable[[jax.Array], jax.Array] = "linear",
    ) -> None:
        """Resolve the static curve, coerce the dynamic leaves."""
        # NOTE: no value-dependent checks here — reversed() calls
        # this constructor with possibly-traced leaves.
        if callable(curve):
            shape = curve
        else:
            try:
                shape = _NAMED_CURVES[curve]
            except KeyError:
                names = ", ".join(sorted(_NAMED_CURVES))
                raise ValueError(
                    f"unknown named curve {curve!r}; available: "
                    f"{names} (or pass a callable with shape(0)=0, "
                    "shape(1)=1)") from None
        self._curve_spec = curve   # static: reversed() and repr
        self._shape = shape        # static: the resolved callable
        self.v0 = jnp.asarray(v0)
        self.v1 = jnp.asarray(v1)
        self.t0 = jnp.asarray(t0)
        self.period = jnp.asarray(period)

    def __call__(self, t: jax.Array | float) -> jax.Array:
        """Branch-free evaluation (clip + shape); valid for all t."""
        s = jnp.clip((jnp.asarray(t) - self.t0) / self.period,
                     0.0, 1.0)
        return self.v0 + (self.v1 - self.v0) * self._shape(s)

    def reversed(self) -> Ramp:
        """
        Reflect the active window across ``t = 0`` (02_rules V-S2).

        Description
        -----------
        Same values and curve, ``t0 -> -(t0 + period)`` — a backward
        leg (dt < 0 from clock 0) retraces this ramp's values
        exactly: ``r.reversed()(-s) == r(2*t0 + period - s)`` for
        every curve, symmetric or not (backward progress ``s`` sees
        the forward value at remaining time). The naive
        value-endpoint swap over ``[0, T]`` clips to a constant.

        Returns
        -------
        Ramp
            The time-domain-reflected ramp.
        """
        return Ramp(self.v0, self.v1, period=self.period,
                    t0=-(self.t0 + self.period),
                    curve=self._curve_spec)

    def __repr__(self) -> str:
        """Round-tripping repr (concrete leaves as plain scalars)."""
        curve = self._curve_spec
        if not isinstance(curve, str):
            curve = getattr(curve, "__name__", repr(curve))
            return (f"Ramp({_leaf_repr(self.v0)}, "
                    f"{_leaf_repr(self.v1)}, "
                    f"period={_leaf_repr(self.period)}, "
                    f"t0={_leaf_repr(self.t0)}, curve={curve})")
        return (f"Ramp({_leaf_repr(self.v0)}, {_leaf_repr(self.v1)}, "
                f"period={_leaf_repr(self.period)}, "
                f"t0={_leaf_repr(self.t0)}, curve={curve!r})")


# ================================================================
#  resolve_at (the universal consumption idiom)
# ================================================================
def resolve_at(value: object, t: jax.Array | float) -> object:
    """
    Evaluate ``value`` at time ``t`` if it is time-dependent.

    Description
    -----------
    ``value(t)`` if ``value`` is a `TimeDependent`, else ``value``
    unchanged. The identity on plain scalars is load-bearing: every
    scalar slot is Ramp-able with zero consumer changes (the
    universal idiom, D2.2).

    Parameters
    ----------
    value : object
        A scalar or a `TimeDependent` curve.
    t : jax.Array | float
        The (signed, possibly traced) clock time.

    Returns
    -------
    object
        ``value(t)`` for time-dependent values; ``value`` itself
        otherwise.
    """
    if isinstance(value, TimeDependent):
        return value(t)
    return value
