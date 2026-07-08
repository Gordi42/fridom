"""
The state-transform base: composable ``State -> State`` maps (2.8).

Description
-----------
``StateTransform`` is the decision-D5 base; the algebra mirrors the
operator algebra (``notes/framework2/model/08_state_transforms.md``
§10.1/§10.2, laws §10.3). Owning class spec:
``notes/framework2/model/classes/transforms.md`` §"fr.StateTransform".

Two tiers, one abstraction: Tier 1 (closed-form) subclasses are
``fr.utils.jaxify``-registered frozen pytrees (jit/vmap-able); Tier 2
(dynamical) subclasses are host objects with a **trace guard**
(``TraceError`` on tracer-valued input). ``traceable`` ANDs under
every combinator. Info is **returned** (``call_with_info``), never a
mutating attribute; the law ``T(s) == call_with_info(s)[0]`` holds by
construction (``__call__`` delegates to ``call_with_info``).
"""
from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax

from fridom.framework2.transforms.errors import TraceError
from fridom.framework2.transforms.info import TransformCost, TransformInfo

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.transforms.signature import StateSignature

# plain scalars accepted by scalar scaling (Ramp coefficients raise)
_SCALARS = (numbers.Number,)


def _is_traced(state: object) -> bool:
    """Whether any leaf of the input is a jax tracer."""
    return any(isinstance(leaf, jax.core.Tracer)
               for leaf in jax.tree_util.tree_leaves(state))


class StateTransform(ABC):

    """
    Abstract composable ``State -> State`` map (decision D5).

    Description
    -----------
    Subclasses implement ``domain``, ``codomain`` and the core
    :meth:`_evaluate` (returning ``(State, TransformInfo)`` without
    validation); the base provides validation, the trace guard, the
    ``call_with_info``/``__call__`` law, and the algebra dunders.
    Product-ready (CS-14): the base never touches concrete ``State``
    types — component access is duck-typed, signatures are opaque
    compared values; endo-ness is asserted only where required.
    """

    # ================================================================
    #  Declared structure
    # ================================================================
    @property
    @abstractmethod
    def domain(self) -> StateSignature | None:
        """Input signature (``None`` iff signature-polymorphic)."""

    @property
    @abstractmethod
    def codomain(self) -> StateSignature | None:
        """Output signature (``None`` iff signature-polymorphic)."""

    @property
    def traceable(self) -> bool:
        """Tier 1 iff True; ANDs under every combinator (default)."""
        return True

    @property
    def idempotent(self) -> bool:
        """Declared, never detected (default False)."""
        return False

    # ================================================================
    #  Application (the law: __call__ == call_with_info[0])
    # ================================================================
    @abstractmethod
    def _evaluate(
        self, state: object,
    ) -> tuple[object, TransformInfo]:
        """Compute the transform (no validation, no guard)."""

    def call_with_info(
        self, state: object,
    ) -> tuple[object, TransformInfo]:
        """
        Apply and return structural info (a mirrored tree).

        Description
        -----------
        Validates the domain signature (trace-time under jit) and
        trace-guards ``traceable=False`` transforms, then runs
        :meth:`_evaluate`. Default leaf info is ``TransformInfo.EMPTY``
        (built by the concrete ``_evaluate``).

        Parameters
        ----------
        state : VectorField
            The input state (duck-typed component container).

        Returns
        -------
        tuple[VectorField, TransformInfo]
            The transformed state and its structural info.
        """
        self._check(state)
        return self._evaluate(state)

    def __call__(self, state: object) -> object:
        """Apply the transform (``call_with_info(state)[0]``)."""
        return self.call_with_info(state)[0]

    def _check(self, state: object) -> None:
        """Trace-guard (Tier 2) then recheck the domain signature."""
        if not self.traceable and _is_traced(state):
            raise TraceError(
                f"{type(self).__name__} is a Tier-2 (traceable="
                "False) transform: it runs a model internally and "
                "cannot appear under jit/vmap/grad. Hoist the call "
                "to the host level, or use a Tier-1 transform inside "
                "the trace.")
        if self.domain is not None:
            self.domain.validate_input(state, path=self._path())

    def _path(self) -> str:
        """Return the composition-tree path for attribution (leaf: name)."""
        return type(self).__name__

    # ================================================================
    #  Cost and repr (the cost-opacity answer)
    # ================================================================
    def cost(self) -> TransformCost:
        """Return the model-step cost (Tier 1: zero); sums structurally."""
        return TransformCost()

    def __repr__(self) -> str:
        """Compact node repr (subclasses override with detail)."""
        return f"{type(self).__name__}()"

    # ================================================================
    #  Sugar
    # ================================================================
    @property
    def complement(self) -> StateTransform:
        """
        ``Identity() - self``; idempotent-gated.

        Description
        -----------
        One of idempotency's three declared consumers (with the
        ``P @ P`` lint hint and ``assert_idempotent``). A
        non-idempotent transform raises a taught ``TypeError``: the
        residual ``I - T`` is only a projector when ``T`` is.

        Returns
        -------
        StateTransform
            The complement transform ``Identity() - self``.
        """
        if not self.idempotent:
            raise TypeError(
                f"{type(self).__name__}.complement requires an "
                "idempotent transform (I - T is a projector only "
                "when T is one); declare idempotent=True or build "
                "the residual explicitly")
        from fridom.framework2.transforms.identity import (  # noqa: PLC0415 — avoids the base<->identity import cycle
            Identity,
        )
        return Identity() - self

    # ================================================================
    #  The algebra (dunders build the node classes)
    # ================================================================
    def __matmul__(self, other: object) -> StateTransform:
        """Composition, right-to-left: ``(A @ B)(s) = A(B(s))``."""
        if not isinstance(other, StateTransform):
            return NotImplemented
        from fridom.framework2.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_compose,
        )
        return make_compose(self, other)

    def __add__(self, other: object) -> StateTransform:
        """Pointwise sum on outputs; domains AND codomains equal."""
        if not isinstance(other, StateTransform):
            return NotImplemented
        from fridom.framework2.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_sum,
        )
        return make_sum(self, other)

    def __sub__(self, other: object) -> StateTransform:
        """``self + (-1) * other`` (structural)."""
        if not isinstance(other, StateTransform):
            return NotImplemented
        return self + (-other)

    def __mul__(self, scalar: object) -> StateTransform:
        """Scalar scaling of the output; plain scalars only."""
        return self._scale(scalar)

    def __rmul__(self, scalar: object) -> StateTransform:
        """Scalar scaling of the output (``scalar * self``)."""
        return self._scale(scalar)

    def _scale(self, scalar: object) -> StateTransform:
        """Build the scaled node, rejecting Ramp coefficients."""
        from fridom.framework2.model.time_dependent import (  # noqa: PLC0415 — avoids the model<->transforms import cycle
            TimeDependent,
        )
        if isinstance(scalar, TimeDependent):
            raise TypeError(
                "transforms scale by plain scalars only: a Ramp "
                "coefficient is rejected (transforms are autonomous "
                "maps — no clock in the surface). Evaluate the ramp "
                "explicitly and pass the value.")
        if isinstance(scalar, bool) or not isinstance(scalar,
                                                      _SCALARS):
            return NotImplemented
        from fridom.framework2.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_scaled,
        )
        return make_scaled(scalar, self)

    def __neg__(self) -> StateTransform:
        """``(-1) * self``."""
        from fridom.framework2.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_scaled,
        )
        return make_scaled(-1, self)

    def __pow__(self, n: object) -> StateTransform:
        """Iterate the transform; ``n == 0`` -> Identity, ``n < 0`` raises."""
        if isinstance(n, bool) or not isinstance(n, int):
            return NotImplemented
        from fridom.framework2.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_power,
        )
        return make_power(self, n)
