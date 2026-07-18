r"""
Scheduled fields: the shared ``(coords, t)`` recompute + ``ProfileFunction``.

Description
-----------
General time-dependent fields plan, wave 2 (TDF-D1/D2). A *scheduled*
AUXILIARY field evolves in time by re-evaluating a user law at the
substage clock and rewriting its carry-resident data through a
SELF_UPDATE stage — the discipline :class:`MovingGeometry`
(``modules/moving_geometry.py``) shipped for mapping parameters,
extracted here as reusable machinery.

- :func:`sample_law` is the sampling core: a pure ``t -> value``
  recompute at a field's own nodes, optionally paired with its time
  derivative through one ``jax.jvp`` in ``t`` (the ``_dot`` companion
  is opt-in — :class:`MovingGeometry` needs it, the Coriolis ``f`` and
  shallow-water ``csqr`` laws do not). There is **no clipping or
  masking** (TDF-D8): the law is evaluated plainly so the reverse-mode
  VJP has no masked singularity.
- :func:`profile_coords` collects a field's node coordinates from its
  declared space (arity-dispatched by the requested names), the
  positional coordinate tuple a :class:`ProfileFunction` law consumes.
- :class:`ProfileFunction` is the user surface: a static law
  ``fn(*coords, t, *params)`` with dynamic leaf parameters — the field
  analogue of ``TimeFunction`` (``time_dependent.py``). Changing the
  law is one recompile; sweeping ``params`` never recompiles and
  ``jax.grad`` flows through them. Like the scalar ``TimeDependent``
  family it supports **no** field/array arithmetic: a consumer that
  forgets to :meth:`~ProfileFunction.sample` it fails loudly.
"""
# Wave 2: TDF-D1 sampling helper + TDF-D2 ProfileFunction
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework.utils import dtype_real, jaxify

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import SpaceLike


def _leaf_repr(value: object) -> str:
    """Format a dynamic leaf for repr (a plain scalar when concrete)."""
    item = getattr(value, "item", None)
    if item is None:
        return repr(value)
    try:
        return repr(item())
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return repr(value)


# ================================================================
#  The sampling core (TDF-D1)
# ================================================================
def sample_law(
    call: Callable[[object, jax.Array], jax.Array],
    coords: object,
    time: jax.Array | float,
    shape: tuple[int, ...],
    *,
    derivative: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    r"""
    Evaluate a scheduled law at ``time`` (optionally with its ``t`` slope).

    Description
    -----------
    The shared recompute core: ``call(coords, t)`` is the arity-dispatched
    law evaluation (the consumer's own coordinate/parameter convention),
    coerced to an array and **broadcast to** ``shape`` — the field's own
    node layout. With ``derivative=True`` a single ``jax.jvp`` in ``t``
    (unit tangent) returns the value and its time derivative together (the
    ``<name>_dot`` mesh-velocity ingredient); otherwise only the value.
    The evaluation is plain — no clipping, no masking — so the reverse-mode
    VJP through ``params`` is singularity-free (TDF-D8).

    Parameters
    ----------
    call : Callable
        The law adapter ``(coords, t) -> value``; ``coords`` is passed
        through opaquely (a name-keyed mapping or a positional tuple —
        whatever the caller's convention is).
    coords : object
        The coordinate arrays at the field's own nodes, in the caller's
        convention (passed straight to ``call``).
    time : jax.Array | float
        The (possibly traced) stage clock time.
    shape : tuple[int, ...]
        The field's laid-out space shape the value is broadcast to.
    derivative : bool, optional
        Also return the time derivative via one ``jax.jvp`` in ``t``
        (default: False).

    Returns
    -------
    jax.Array | tuple[jax.Array, jax.Array]
        The value, or ``(value, dot)`` when ``derivative`` is set — each
        broadcast to ``shape``.
    """
    t = jnp.asarray(time, dtype=dtype_real())

    def at_time(tt: jax.Array) -> jax.Array:
        return jnp.asarray(call(coords, tt))

    if derivative:
        value, dot = jax.jvp(at_time, (t,), (jnp.ones((), t.dtype),))
        return (jnp.broadcast_to(value, shape),
                jnp.broadcast_to(dot, shape))
    return jnp.broadcast_to(at_time(t), shape)


def profile_coords(
    grid: Grid,
    space: SpaceLike,
    names: Sequence[str],
) -> tuple[jax.Array, ...]:
    """
    Collect a field's node coordinates from its space (positional order).

    Description
    -----------
    One array per requested coordinate, at the field's own evaluation
    nodes (``grid.evaluation_nodes`` — each broadcasts exactly under the
    strict algebra), in ``names`` order. This is the positional coordinate
    tuple a :class:`ProfileFunction` law consumes as ``fn(*coords, t,
    ...)``; the arity is the length of ``names``.

    Parameters
    ----------
    grid : Grid
        The grid supplying the evaluation nodes.
    space : SpaceLike
        The field's own (laid-out) space.
    names : Sequence[str]
        The coordinate names to materialize, in the law's argument order.

    Returns
    -------
    tuple[jax.Array, ...]
        The per-coordinate node arrays, in ``names`` order.
    """
    return tuple(
        grid.evaluation_nodes(space, name).data for name in names)


# ================================================================
#  ProfileFunction (a static spatial law fn(*coords, t, *params))
# ================================================================
@partial(jaxify, dynamic=("params",))
class ProfileFunction:

    r"""
    A static spatial-and-temporal law ``fn(*coords, t, *params)``.

    Description
    -----------
    The field analogue of ``TimeFunction`` (``time_dependent.py``): a
    pure, branch-free law of a field's own node coordinates and the
    clock, sampled at stage time into the AUXILIARY field its consumer
    owns (the Coriolis ``f(y,t)``, the shallow-water ``csqr(y,t)``).
    ``fn`` is a **static descriptor** (a law change is different maths —
    exactly one recompile, the ``Ramp.curve`` / ``TimeFunction.fn``
    split); ``params`` are jaxified dynamic leaves, so sweeping them
    never recompiles and ``jax.grad`` flows through them (TDF-D8).

    ``coords`` are the field's own node coordinates from its declared
    space — a ``Profile("y")`` field passes its 1-D ``y`` array, a
    two-coordinate profile passes both in factor order (the consumer
    collects them with :func:`profile_coords`). The evaluation is
    pointwise (zero stencil). ``fn`` must be valid at every ``t``
    (evaluated at every scan step) and free of Python branches on the
    traced time — the ``TimeDependent`` contract, applied to a field.

    Like the scalar ``TimeDependent`` family, ``ProfileFunction``
    supports **no** field/array arithmetic — a consumer forgetting to
    :meth:`sample` it fails loudly rather than silently reading the law
    object as an operand.

    Parameters
    ----------
    fn : Callable
        The pure law ``fn(*coords, t, *params) -> array``; STATIC (hashed
        by identity — a distinct ``fn`` recompiles once).
    params : tuple, optional
        Extra dynamic-leaf arguments passed after ``t`` (each coerced with
        ``jnp.asarray``); swept without recompiling (default: ()).
    """

    def __init__(
        self,
        fn: Callable[..., jax.Array],
        params: tuple = (),
    ) -> None:
        """Store the static law; coerce the extra arguments to leaves."""
        if not callable(fn):
            raise TypeError(
                f"ProfileFunction fn must be callable "
                f"fn(*coords, t, *params), got {fn!r}")
        self._fn = fn                                   # static law
        self.params = tuple(jnp.asarray(p) for p in params)

    # ================================================================
    #  Sampling
    # ================================================================
    def sample(
        self,
        coords: Sequence[jax.Array],
        time: jax.Array | float,
        shape: tuple[int, ...],
    ) -> jax.Array:
        r"""
        Sample the law at ``time`` and ``coords`` (broadcast to ``shape``).

        Description
        -----------
        Evaluates ``fn(*coords, t, *params)`` at the stage clock time and
        broadcasts to the field's laid-out ``shape`` through the shared
        :func:`sample_law` core (no derivative — the ``_dot`` companion is
        for :class:`MovingGeometry`, not for ``f`` / ``csqr``; no clipping,
        so the VJP is singularity-free).

        Parameters
        ----------
        coords : Sequence[jax.Array]
            The field's node coordinates in the law's argument order (from
            :func:`profile_coords`).
        time : jax.Array | float
            The (possibly traced) stage clock time.
        shape : tuple[int, ...]
            The field's laid-out space shape to broadcast to.

        Returns
        -------
        jax.Array
            The sampled law values on ``shape``.
        """
        fn, params = self._fn, self.params

        def call(cc: Sequence[jax.Array], tt: jax.Array) -> jax.Array:
            return fn(*cc, tt, *params)

        return sample_law(call, tuple(coords), time, shape)

    def __repr__(self) -> str:
        """Repr naming the law and its concrete leaves."""
        name = getattr(self._fn, "__name__", repr(self._fn))
        params = ", ".join(_leaf_repr(p) for p in self.params)
        return f"ProfileFunction({name}, params=({params}))"
