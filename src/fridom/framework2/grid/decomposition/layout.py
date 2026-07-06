"""
The device layout descriptor.

Description
-----------
Owning class doc: ``notes/framework2/classes/decomposition.md``
(section 5.1 of ``04_decomposition.md``). A ``Layout`` is the value
that enters the function-space interning key when set; the space
clusters import it from here as an opaque hashable value, so this
module stays dependency-free.
"""
# Wave 1: Layout
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


@dataclass(frozen=True, init=False)
class Layout:

    """
    One assignment of coordinate names to device-mesh axes.

    Description
    -----------
    Purely combinatorial and semantic-only: unmapped coordinate names
    are device-local, and halo widths / stagger padding are
    deliberately *not* part of a layout — they are storage the owning
    decomposition pairs with it internally. Everything array-shaped
    (partition specs, local slices, storage shapes) is derived by the
    owning ``Decomposition``. Storage is a sorted
    ``tuple[tuple[str, str], ...]`` so layouts are hashable values
    (part of jit cache keys via the spaces); the constructor accepts
    any ``Mapping[str, str]`` and normalizes.

    Parameters
    ----------
    device_axes : Mapping[str, str]
        Coordinate name -> device-mesh axis name; each device axis
        may carry at most one coordinate name.
    """

    device_axes: tuple[tuple[str, str], ...]

    def __init__(self, device_axes: Mapping[str, str]) -> None:
        """Normalize the mapping to sorted tuple storage (hashable)."""
        items = tuple(sorted(device_axes.items()))
        axes = [axis for _, axis in items]
        if len(set(axes)) != len(axes):
            raise ValueError(
                "each device-mesh axis may shard at most one "
                f"coordinate name, got {items}")
        object.__setattr__(self, "device_axes", items)

    def is_local(self, name: str) -> bool:
        """
        Return whether the factor carrying `name` is device-local.

        Parameters
        ----------
        name : str
            A coordinate name.

        Returns
        -------
        bool
            True if `name` is not sharded across a device axis here.
        """
        return all(key != name for key, _ in self.device_axes)
