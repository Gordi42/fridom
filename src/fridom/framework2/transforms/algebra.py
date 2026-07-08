"""
Algebra nodes: structural composition, sum, scaling, power (2.8).

Description
-----------
The internal node classes built by ``StateTransform``'s dunders —
never user-constructed; named so reprs and info paths stay
addressable (``notes/framework2/model/08_state_transforms.md`` §10.2;
owning class spec ``notes/framework2/model/classes/transforms.md``
§"The algebra nodes"). Normalization is **structural only**: flatten
nested ``Compose``/``Sum`` and elide ``Identity`` — no rewriting, no
idempotent folding (what you wrote is what runs). All signatures are
concrete at construction, so every algebra check is **eager at
compose time** (plus the call-time recheck in ``call_with_info``).

Derived structure: ``traceable`` = AND of children; ``cost()`` = sum
of children (``Power``: ``n *`` inner); signatures derived at node
construction with the eager checks.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from fridom.framework.utils import jaxify
from fridom.framework2.transforms.base import StateTransform
from fridom.framework2.transforms.errors import SignatureMismatchError
from fridom.framework2.transforms.info import TransformCost, TransformInfo

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.transforms.signature import StateSignature


# ================================================================
#  Signature resolution helpers (None == signature-polymorphic)
# ================================================================
def _resolve(
    signatures: tuple, what: str,
) -> StateSignature | None:
    """Resolve equal signatures, tolerating polymorphic (None) ones."""
    concrete = [sig for sig in signatures if sig is not None]
    if not concrete:
        return None
    first = concrete[0]
    for sig in concrete[1:]:
        if sig != first:
            raise SignatureMismatchError(
                f"{what}: operands carry incompatible signatures\n"
                f"  {first!r}\n  vs {sig!r}")
    return first


def _check_adjacent(
    outer: StateTransform, inner: StateTransform,
) -> None:
    """Eager compose check: ``inner.codomain == outer.domain``."""
    dom, cod = outer.domain, inner.codomain
    if dom is not None and cod is not None and dom != cod:
        raise SignatureMismatchError(
            f"cannot compose {type(outer).__name__} @ "
            f"{type(inner).__name__}: the inner codomain does not "
            f"match the outer domain\n  inner.codomain = {cod!r}\n"
            f"  outer.domain   = {dom!r}")


def _aggregate(
    children: tuple[tuple[str, TransformInfo], ...],
) -> TransformInfo:
    """Fold child infos into the mirrored composite record."""
    model_steps = sum(child.model_steps for _, child in children)
    elapsed = sum(child.elapsed_model_time for _, child in children)
    return TransformInfo(
        model_steps=model_steps, elapsed_model_time=elapsed,
        children=children)


# ================================================================
#  Compose — right-to-left composition chain
# ================================================================
@partial(jaxify, dynamic=("_parts",))
class Compose(StateTransform):

    """Right-to-left composition chain: ``parts[0]`` applied last."""

    def __init__(self, parts: tuple[StateTransform, ...]) -> None:
        """Freeze the flattened chain and derive the signatures."""
        self._parts = tuple(parts)
        for outer, inner in zip(self._parts, self._parts[1:],
                                strict=False):
            _check_adjacent(outer, inner)
        self._domain = self._parts[-1].domain
        self._codomain = self._parts[0].codomain
        self._traceable = all(p.traceable for p in self._parts)

    @property
    def parts(self) -> tuple[StateTransform, ...]:
        """The flattened chain (nested Compose flattened)."""
        return self._parts

    @property
    def domain(self) -> StateSignature | None:
        """Input signature: the innermost part's domain."""
        return self._domain

    @property
    def codomain(self) -> StateSignature | None:
        """Output signature: the outermost part's codomain."""
        return self._codomain

    @property
    def traceable(self) -> bool:
        """AND of the children's ``traceable``."""
        return self._traceable

    def _evaluate(self, state: object) -> tuple[object, TransformInfo]:
        """Apply innermost-first, thread the state, mirror the info."""
        current = state
        infos: dict[int, TransformInfo] = {}
        for i in range(len(self._parts) - 1, -1, -1):
            current, infos[i] = self._parts[i].call_with_info(current)
        children = tuple(
            (f"{i}:{type(self._parts[i]).__name__}", infos[i])
            for i in range(len(self._parts)))
        return current, _aggregate(children)

    def cost(self) -> TransformCost:
        """Sum of the children's costs."""
        total = TransformCost()
        for part in self._parts:
            total = total + part.cost()
        return total

    def __repr__(self) -> str:
        """Render the composition tree ``A @ B @ C``."""
        return " @ ".join(repr(part) for part in self._parts)


# ================================================================
#  Sum — pointwise sum on outputs
# ================================================================
@partial(jaxify, dynamic=("_parts",))
class Sum(StateTransform):

    """Pointwise sum on outputs; equal domains AND codomains."""

    def __init__(self, parts: tuple[StateTransform, ...]) -> None:
        """Freeze the flattened summands and check the signatures."""
        self._parts = tuple(parts)
        self._domain = _resolve(
            tuple(p.domain for p in self._parts), "sum domains")
        self._codomain = _resolve(
            tuple(p.codomain for p in self._parts), "sum codomains")
        self._traceable = all(p.traceable for p in self._parts)

    @property
    def parts(self) -> tuple[StateTransform, ...]:
        """The flattened summands (nested Sum flattened)."""
        return self._parts

    @property
    def domain(self) -> StateSignature | None:
        """The shared input signature."""
        return self._domain

    @property
    def codomain(self) -> StateSignature | None:
        """The shared output signature."""
        return self._codomain

    @property
    def traceable(self) -> bool:
        """AND of the children's ``traceable``."""
        return self._traceable

    def _evaluate(self, state: object) -> tuple[object, TransformInfo]:
        """Evaluate each summand on the input; sum outputs pointwise."""
        outputs = []
        children = []
        for i, part in enumerate(self._parts):
            out, info = part.call_with_info(state)
            outputs.append(out)
            children.append((f"{i}:{type(part).__name__}", info))
        total = outputs[0]
        for out in outputs[1:]:
            total = total + out
        return total, _aggregate(tuple(children))

    def cost(self) -> TransformCost:
        """Sum of the children's costs."""
        total = TransformCost()
        for part in self._parts:
            total = total + part.cost()
        return total

    def __repr__(self) -> str:
        """Render the sum tree ``A + B``."""
        return "(" + " + ".join(repr(p) for p in self._parts) + ")"


# ================================================================
#  Scaled — plain-scalar scaling of a transform's output
# ================================================================
@partial(jaxify, dynamic=("_inner", "_coefficient"))
class Scaled(StateTransform):

    """Plain-scalar scaling of a transform's output."""

    def __init__(
        self, coefficient: complex, inner: StateTransform,
    ) -> None:
        """Store the scalar coefficient and the inner transform."""
        self._coefficient = coefficient
        self._inner = inner

    @property
    def coefficient(self) -> complex:
        """The scalar coefficient."""
        return self._coefficient

    @property
    def inner(self) -> StateTransform:
        """The scaled transform."""
        return self._inner

    @property
    def domain(self) -> StateSignature | None:
        """The inner transform's domain."""
        return self._inner.domain

    @property
    def codomain(self) -> StateSignature | None:
        """The inner transform's codomain."""
        return self._inner.codomain

    @property
    def traceable(self) -> bool:
        """The inner transform's ``traceable``."""
        return self._inner.traceable

    def _evaluate(self, state: object) -> tuple[object, TransformInfo]:
        """Apply the inner transform and scale its output."""
        out, info = self._inner.call_with_info(state)
        scaled = self._coefficient * out
        label = f"0:{type(self._inner).__name__}"
        return scaled, _aggregate(((label, info),))

    def cost(self) -> TransformCost:
        """Return the inner transform's cost."""
        return self._inner.cost()

    def __repr__(self) -> str:
        """Render the scaled tree ``c * A``."""
        return f"{self._coefficient!r} * {self._inner!r}"


# ================================================================
#  Power — fixed n-fold iteration of an endo transform
# ================================================================
@partial(jaxify, dynamic=("_inner",))
class Power(StateTransform):

    """Fixed n-fold iteration of an endo transform (n >= 2)."""

    def __init__(self, inner: StateTransform, n: int) -> None:
        """Store the endo transform and the iteration count."""
        self._inner = inner
        self._n = n

    @property
    def n(self) -> int:
        """The iteration count."""
        return self._n

    @property
    def inner(self) -> StateTransform:
        """The iterated transform."""
        return self._inner

    @property
    def domain(self) -> StateSignature | None:
        """The endo signature (domain == codomain)."""
        return self._inner.domain

    @property
    def codomain(self) -> StateSignature | None:
        """The endo signature (domain == codomain)."""
        return self._inner.codomain

    @property
    def traceable(self) -> bool:
        """The inner transform's ``traceable``."""
        return self._inner.traceable

    def _evaluate(self, state: object) -> tuple[object, TransformInfo]:
        """Apply the inner transform ``n`` times."""
        current = state
        children = []
        for k in range(self._n):
            current, info = self._inner.call_with_info(current)
            children.append((f"{k}", info))
        return current, _aggregate(tuple(children))

    def cost(self) -> TransformCost:
        """``n *`` the inner transform's cost."""
        return self._inner.cost() * self._n

    def __repr__(self) -> str:
        """Render the power tree ``A ** n``."""
        return f"{self._inner!r} ** {self._n}"


# ================================================================
#  Factories (the dunders' entry points; normalization here)
# ================================================================
def _is_identity(transform: StateTransform) -> bool:
    """Whether a transform is an ``Identity`` (elided in chains)."""
    from fridom.framework2.transforms.identity import (  # noqa: PLC0415 — avoids the algebra<->identity import cycle
        Identity,
    )
    return isinstance(transform, Identity)


def _compose_parts(transform: StateTransform) -> tuple:
    """Flatten a Compose (else a singleton), for chain building."""
    if isinstance(transform, Compose):
        return transform.parts
    return (transform,)


def make_compose(
    outer: StateTransform, inner: StateTransform,
) -> StateTransform:
    """Build ``outer @ inner`` (flatten Compose, elide Identity)."""
    parts = tuple(
        part for part in (*_compose_parts(outer),
                          *_compose_parts(inner))
        if not _is_identity(part))
    if not parts:
        from fridom.framework2.transforms.identity import (  # noqa: PLC0415 — avoids the algebra<->identity import cycle
            Identity,
        )
        return Identity()
    if len(parts) == 1:
        return parts[0]
    return Compose(parts)


def _sum_parts(transform: StateTransform) -> tuple:
    """Flatten a Sum (else a singleton), for sum building."""
    if isinstance(transform, Sum):
        return transform.parts
    return (transform,)


def make_sum(
    left: StateTransform, right: StateTransform,
) -> StateTransform:
    """Build ``left + right`` (flatten nested Sum; no Identity elide)."""
    parts = (*_sum_parts(left), *_sum_parts(right))
    if len(parts) == 1:  # pragma: no cover — binary + never singleton
        return parts[0]
    return Sum(parts)


def make_scaled(
    coefficient: complex, inner: StateTransform,
) -> StateTransform:
    """Build ``coefficient * inner`` (no folding — what you wrote runs)."""
    return Scaled(coefficient, inner)


def make_power(inner: StateTransform, n: int) -> StateTransform:
    """Build ``inner ** n`` (0 -> Identity, 1 -> inner, n>=2 endo)."""
    if n < 0:
        raise ValueError(
            f"a transform has no algebraic inverse: {type(inner).__name__}"
            f" ** {n} is undefined (backward is a physical, not "
            "algebraic, inverse of forward)")
    if n == 0:
        from fridom.framework2.transforms.identity import (  # noqa: PLC0415 — avoids the algebra<->identity import cycle
            Identity,
        )
        return Identity(inner.domain)
    if n == 1:
        return inner
    dom, cod = inner.domain, inner.codomain
    if dom is not None and cod is not None and dom != cod:
        raise SignatureMismatchError(
            f"{type(inner).__name__} ** {n} requires an endo "
            "transform (domain == codomain); this transform maps\n"
            f"  {dom!r}\n  -> {cod!r}")
    return Power(inner, n)
