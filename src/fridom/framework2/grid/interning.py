"""
Weak intern table for identity-hashed static objects.

Description
-----------
Shared interning helper for the meshes and spaces clusters (cluster
rules in ``notes/framework2/classes/meshes.md`` and the intern-table
notes in ``notes/framework2/classes/product_spaces.md``): interning
turns value equality into identity — value-equal requests return the
identical object, so the strict-algebra equality check is ``a is b``.
The table holds its entries through weak references, so unreferenced
entries are collectable (nothing is pinned, no state leaks across
tests).
"""
# Wave 0: InternTable
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Hashable

T = TypeVar("T")


class InternTable:

    """
    Weak intern table mapping value-hashable keys to objects.

    Description
    -----------
    Each owner (a mesh's space registry, the ``TensorProductSpace``
    class table, ...) holds its own instance. Keys are value-hashable
    tuples of static descriptors; values must support weak references
    (all interned classes here do). Entries whose object is no longer
    referenced elsewhere are collected automatically.
    """

    def __init__(self) -> None:
        self._entries: weakref.WeakValueDictionary[Hashable, object] = (
            weakref.WeakValueDictionary())

    def intern(self, key: Hashable, factory: Callable[[], T]) -> T:
        """
        Return the entry for `key`, building it on first request.

        Parameters
        ----------
        key : Hashable
            The value-hashable interning key.
        factory : Callable[[], T]
            Zero-argument factory invoked only when `key` has no live
            entry; its result is stored and returned.

        Returns
        -------
        T
            The interned object; the identical object is returned for
            every value-equal key while it stays alive.
        """
        obj = self._entries.get(key)
        if obj is None:
            obj = factory()
            self._entries[key] = obj
        return obj

    def __contains__(self, key: Hashable) -> bool:
        """Check whether `key` currently has a live entry."""
        return key in self._entries

    def __len__(self) -> int:
        """Return the number of live entries."""
        return len(self._entries)
