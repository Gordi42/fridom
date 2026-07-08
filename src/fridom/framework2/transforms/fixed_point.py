"""
FixedPoint: iteration over a transform (or transform factory) (2.8).

Description
-----------
Iterate ``T`` until the norm of the update converges (§10.2; owning
class spec ``notes/framework2/model/classes/transforms.md``
§"FixedPoint"). ``T`` is an endo ``StateTransform`` **or** a factory
``State -> StateTransform`` evaluated on the current iterate per
iteration (this absorbs OB's ``update_base_point`` cleanly). Host
object, not a pytree (``traceable=False`` in iteration 1; a traced
``lax.while_loop`` variant is designed-for). The pinned kwargs are
the whole surface — nothing model-flavored (a model-supplied default
norm is rejected: invisible criterion, re-coupling).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from fridom.framework2.transforms.base import StateTransform
from fridom.framework2.transforms.errors import (
    FixedPointDivergenceError,
    SignatureMismatchError,
)
from fridom.framework2.transforms.info import (
    TransformCost,
    TransformInfo,
    TransformProgress,
)
from fridom.framework2.transforms.norms import relative_l2

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.transforms.signature import StateSignature


class FixedPoint(StateTransform):

    """Iterate ``T`` until the norm of the update converges."""

    def __init__(
        self,
        transform: (StateTransform
                    | Callable[[VectorField], StateTransform]),
        *,
        tol: float = 1e-9,
        max_it: int = 3,
        norm: Callable[[VectorField, VectorField], float] = relative_l2,
        on_divergence: Literal[
            "stop_best", "raise", "ignore"] = "stop_best",
        on_iteration: Callable[[TransformProgress], None] | None = None,
    ) -> None:
        """
        Configure the fixed-point iteration.

        Parameters
        ----------
        transform : StateTransform | Callable[[VectorField], StateTransform]
            An endo transform, or a factory evaluated on the current
            iterate per iteration (absorbs OB's update_base_point).
        tol : float, optional
            Convergence tolerance; ``0`` runs exactly ``max_it``
            iterations (default: 1e-9).
        max_it : int, optional
            Maximum iterations; ``0`` returns the input unchanged
            (default: 3).
        norm : Callable, optional
            The update distance (default: :func:`relative_l2`).
        on_divergence : {"stop_best", "raise", "ignore"}, optional
            Policy when ``err > prev`` (default: "stop_best").
        on_iteration : Callable | None, optional
            Host observer receiving a ``TransformProgress`` per
            iteration; never influences results (default: None).
        """
        if isinstance(max_it, bool) or not isinstance(max_it, int) \
                or max_it < 0:
            raise ValueError(
                f"max_it must be a non-negative int, got {max_it!r}")
        if on_divergence not in ("stop_best", "raise", "ignore"):
            raise ValueError(
                f"on_divergence must be 'stop_best', 'raise', or "
                f"'ignore', got {on_divergence!r}")
        self._is_factory = not isinstance(transform, StateTransform)
        if not self._is_factory:
            self._require_endo(transform)
        self._transform = transform
        self._tol = float(tol)
        self._max_it = max_it
        self._norm = norm
        self._on_divergence = on_divergence
        self._on_iteration = on_iteration

    @staticmethod
    def _require_endo(transform: StateTransform) -> None:
        """Reject a non-endo wrapped transform at construction."""
        dom, cod = transform.domain, transform.codomain
        if dom is not None and cod is not None and dom != cod:
            raise SignatureMismatchError(
                f"FixedPoint wraps an endo transform (domain == "
                f"codomain); {type(transform).__name__} maps\n"
                f"  {dom!r}\n  -> {cod!r}")

    # ================================================================
    #  Declared structure
    # ================================================================
    @property
    def traceable(self) -> bool:
        """Tier 2: host-level in iteration 1 (never under a trace)."""
        return False

    @property
    def domain(self) -> StateSignature | None:
        """The wrapped endo signature; ``None`` for the factory form."""
        if self._is_factory:
            return None
        return self._transform.domain

    @property
    def codomain(self) -> StateSignature | None:
        """The wrapped endo signature; ``None`` for the factory form."""
        return self.domain

    # ================================================================
    #  Iteration
    # ================================================================
    def _resolve(self, iterate: VectorField) -> StateTransform:
        """Return the transform for one iteration (factory-aware)."""
        if not self._is_factory:
            return self._transform
        produced = self._transform(iterate)
        self._require_endo(produced)
        return produced

    def _evaluate(
        self, state: VectorField,
    ) -> tuple[VectorField, TransformInfo]:
        """Run the iteration; return the best/converged iterate."""
        current = state
        result = state
        errors: list[float] = []
        children: list[tuple[str, TransformInfo]] = []
        best_state, best_err, best_iter = state, math.inf, 0
        prev_err = math.inf
        stopped_by = "max_it"
        broke = False
        for k in range(self._max_it):
            transform = self._resolve(current)
            nxt, info = transform.call_with_info(current)
            err = float(self._norm(nxt, current))
            errors.append(err)
            children.append((f"iteration[{k}]", info))
            self._report(k)
            if err < best_err:
                best_state, best_err, best_iter = nxt, err, k + 1
            if self._tol > 0.0 and err <= self._tol:
                stopped_by, result = "tol", nxt
                best_iter, broke = k + 1, True
                break
            if err > prev_err:
                if self._on_divergence == "raise":
                    raise FixedPointDivergenceError(
                        "FixedPoint diverged (err > prev) at "
                        f"iteration {k}: error series {tuple(errors)}")
                if self._on_divergence == "stop_best":
                    stopped_by = "divergence"
                    result, broke = best_state, True
                    break
                # "ignore": keep iterating, stopped_by unchanged
            current = nxt
            prev_err = err
        if not broke:
            result, best_iter = current, self._max_it
        return result, self._info(errors, best_iter, stopped_by,
                                  tuple(children))

    def _report(self, k: int) -> None:
        """Invoke the host iteration observer, if any."""
        if self._on_iteration is not None:
            self._on_iteration(TransformProgress(
                path=type(self).__name__, steps_done=k + 1,
                steps_total=self._max_it))

    @staticmethod
    def _info(
        errors: list[float],
        iterations: int,
        stopped_by: str,
        children: tuple[tuple[str, TransformInfo], ...],
    ) -> TransformInfo:
        """Assemble the FixedPoint call record."""
        return TransformInfo(
            iterations=iterations, errors=tuple(errors),
            extra={"stopped_by": stopped_by,
                   "returned_iteration": iterations},
            children=children)

    def cost(self) -> TransformCost:
        """``max_it *`` the per-iteration cost, flagged upper-bound."""
        if self._is_factory:
            return TransformCost(upper_bound=True)
        inner = self._transform.cost()
        return TransformCost(
            model_steps=inner.model_steps * self._max_it,
            upper_bound=True)

    def __repr__(self) -> str:
        """Compact host-side summary."""
        inner = ("factory" if self._is_factory
                 else repr(self._transform))
        return (f"FixedPoint({inner}, max_it={self._max_it}, "
                f"tol={self._tol:g})")
