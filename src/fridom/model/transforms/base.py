"""
The state-transform base: composable ``State -> State`` maps (2.8).

Description
-----------
``StateTransform`` is the decision-D5 base; the algebra mirrors the
operator algebra (``design/specs/model/08_state_transforms.md``
§10.1/§10.2, laws §10.3). Owning class spec:
``design/specs/model/classes/transforms.md`` §"fr.StateTransform".

Two tiers, one abstraction: Tier 1 (closed-form) subclasses are
``fr.utils.jaxify``-registered frozen pytrees (jit/vmap-able); Tier 2
(dynamical) subclasses are host objects with a **trace guard**
(``TraceError`` on tracer-valued input). ``traceable`` ANDs under
every combinator. Info is **returned** (``call_with_info``), never a
mutating attribute; the law ``T(s) == call_with_info(s)[0]`` holds by
construction (``__call__`` delegates to ``call_with_info``). The
domain signature's ``rest`` policy is applied centrally here
(``_apply_rest`` — output completion after ``_evaluate``, §10.7.2).
"""
from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.model.transforms.errors import TraceError
from fridom.model.transforms.info import TransformCost, TransformInfo

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.transforms.signature import StateSignature

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
        :meth:`_evaluate` and completes its output per the domain's
        ``rest`` policy (:meth:`_apply_rest`, §10.7.2). Default leaf
        info is ``TransformInfo.EMPTY`` (built by the concrete
        ``_evaluate``).

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
        out, info = self._evaluate(state)
        return self._apply_rest(state, out), info

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

    def _apply_rest(self, state: object, out: object) -> object:
        """
        Complete the output per the domain's ``rest`` policy.

        Description
        -----------
        The central wiring of ``StateSignature.rest`` (§10.3 law 2,
        §10.7.2): every input component outside the domain's mapped
        subset that :meth:`_evaluate` did not emit is attached to the
        output — ``rest="zero"`` as a zero field on the component's
        own space (completeness holds restricted to the mapped
        family, so ``state - P(state)`` carries the full extra),
        ``rest="pass"`` as the input component object itself. The
        completed output keeps the **input's** component order for
        shared names (payload-emitted components win; extras sit at
        their input positions), so summands stay componentwise
        compatible under the ``Sum`` node; payload-only components
        (a non-endo codomain) append after, and a *mapped* component
        the payload dropped stays dropped. Extras are handled
        uniformly — a state exposes no lifecycle, and the Tier-2
        read-back law (§10.3 law 1) makes transform inputs
        PROGNOSTIC-only — mirroring ``validate_input``, which
        permits any extra. Signature-polymorphic transforms
        (``domain is None``) have no mapped subset and skip the
        completion. Algebra nodes need no extra rule: each concrete
        leaf completes its own output, and the node-level pass (its
        derived signature) is a no-op on already-complete children.

        Parameters
        ----------
        state : VectorField
            The validated input state.
        out : VectorField
            The payload output from :meth:`_evaluate`.

        Returns
        -------
        VectorField
            The completed output (``out`` itself when nothing is
            missing — the fast path).
        """
        domain = self.domain
        if domain is None:
            return out
        mapped = set(domain.names)
        emitted = set(out.component_names)
        missing = tuple(
            name for name in state.component_names
            if name not in mapped and name not in emitted)
        if not missing:
            return out
        completed = {}
        for name in state.component_names:
            if name in emitted:
                completed[name] = out[name]
            elif name not in mapped:
                field = state[name]
                if domain.rest == "zero":
                    field = field.with_data(
                        jnp.zeros_like(field.data))
                completed[name] = field
            # a mapped component the payload dropped stays dropped
        for name in out.component_names:
            if name not in completed:
                completed[name] = out[name]
        return type(out)(completed)

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
        from fridom.model.transforms.identity import (  # noqa: PLC0415 — avoids the base<->identity import cycle
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
        from fridom.model.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_compose,
        )
        return make_compose(self, other)

    def __add__(self, other: object) -> StateTransform:
        """Pointwise sum on outputs; domains AND codomains equal."""
        if not isinstance(other, StateTransform):
            return NotImplemented
        from fridom.model.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
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
        from fridom.model.time_dependent import (  # noqa: PLC0415 — avoids the model<->transforms import cycle
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
        from fridom.model.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_scaled,
        )
        return make_scaled(scalar, self)

    def __neg__(self) -> StateTransform:
        """``(-1) * self``."""
        from fridom.model.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_scaled,
        )
        return make_scaled(-1, self)

    def __pow__(self, n: object) -> StateTransform:
        """Iterate the transform; ``n == 0`` -> Identity, ``n < 0`` raises."""
        if isinstance(n, bool) or not isinstance(n, int):
            return NotImplemented
        from fridom.model.transforms.algebra import (  # noqa: PLC0415 — avoids the base<->algebra import cycle
            make_power,
        )
        return make_power(self, n)
