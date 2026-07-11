"""
Shift: the affine piece ``s -> s + state0`` (2.8).

Description
-----------
The affine building block (§10.2; owning class spec
``design/specs/model/classes/transforms.md`` §"Shift") — OB's
base-point exchange. Tier 1: a ``fr.utils.jaxify`` frozen pytree with
the captured state as the dynamic leaf (the ``fr.Ramp`` pattern);
``traceable=True``, ``idempotent=False``; cost zero. The endo
signature derives from ``state0``'s components (grid + mapped subset);
the sum runs through the State's componentwise arithmetic.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from fridom.framework.utils import jaxify
from fridom.framework2.transforms.base import StateTransform
from fridom.framework2.transforms.info import TransformInfo
from fridom.framework2.transforms.signature import StateSignature

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.fields.vector_field import VectorField


@partial(jaxify, dynamic=("_state0",))
class Shift(StateTransform):

    """``s -> s + state0`` (State arithmetic; componentwise)."""

    def __init__(self, state0: VectorField) -> None:
        """
        Capture the shift state and derive the endo signature.

        Parameters
        ----------
        state0 : VectorField
            The state added to every input (the affine offset).
        """
        self._state0 = state0
        self._signature = StateSignature.of_prognostic(state0)

    @property
    def state0(self) -> VectorField:
        """The captured shift state (the dynamic leaf)."""
        return self._state0

    @property
    def domain(self) -> StateSignature:
        """The endo signature derived from ``state0``."""
        return self._signature

    @property
    def codomain(self) -> StateSignature:
        """The endo signature derived from ``state0``."""
        return self._signature

    def _evaluate(self, state: object) -> tuple[object, TransformInfo]:
        """Add ``state0`` onto the matching components of the input."""
        contributions = {
            name: self._state0[name]
            for name in self._state0.component_names}
        return state.add(**contributions), TransformInfo.EMPTY

    def __repr__(self) -> str:
        """``Shift`` over its component names."""
        return f"Shift({list(self._state0.component_names)})"
