"""
Identity: the signature-polymorphic unit of the algebra (2.8).

Description
-----------
``s -> s``; signature-polymorphic until composed or called (§10.2;
owning class spec ``design/specs/model/classes/transforms.md``
§"Identity"). ``traceable=True``, ``idempotent=True``; **elided in
Compose chains** during structural normalization (so ``Identity() -
P`` is a real ``Sum`` but ``A @ Identity() @ B`` is ``A @ B``). While
polymorphic it matches any signature at compose time and adopts the
partner's; a pinned Identity (``A ** 0`` pins to ``A.domain``) checks
like any transform.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework.utils import jaxify
from fridom.model.transforms.base import StateTransform
from fridom.model.transforms.info import TransformInfo

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.transforms.signature import StateSignature


@jaxify
class Identity(StateTransform):

    """``s -> s``; signature-polymorphic until composed or called."""

    def __init__(self, domain: StateSignature | None = None) -> None:
        """
        Optionally pin the identity to a signature.

        Parameters
        ----------
        domain : StateSignature | None, optional
            The pinned endo signature (``A ** 0`` pins to
            ``A.domain``); ``None`` is polymorphic (default: None).
        """
        self._domain = domain

    @property
    def domain(self) -> StateSignature | None:
        """The pinned signature, or ``None`` (polymorphic)."""
        return self._domain

    @property
    def codomain(self) -> StateSignature | None:
        """The pinned signature, or ``None`` (polymorphic)."""
        return self._domain

    @property
    def idempotent(self) -> bool:
        """The identity is idempotent."""
        return True

    def _evaluate(self, state: object) -> tuple[object, TransformInfo]:
        """Return the input unchanged with empty info."""
        return state, TransformInfo.EMPTY

    def __repr__(self) -> str:
        """``Identity()`` (polymorphic) or ``Identity(sig)``."""
        if self._domain is None:
            return "Identity()"
        return f"Identity({self._domain!r})"
