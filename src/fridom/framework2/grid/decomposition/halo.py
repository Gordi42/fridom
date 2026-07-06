"""
Halo machinery: negotiated ghost-layer widths.

Description
-----------
Owning class doc: ``notes/framework2/classes/decomposition.md``.
``HaloSpec`` is the negotiated replacement of the global halo
integer: per-coordinate-name ghost widths, keyed by name because
names are the stable addressing scheme of the flat product. It is a
static, hashable value that enters jit cache keys through the
decomposition.
"""
# Wave 1: HaloSpec -- Wave 3: HaloTracer, trace_halo
#    (GhostFill is designed-for)
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


@dataclass(frozen=True, init=False)
class HaloSpec:

    """
    Negotiated ghost-layer widths, one per coordinate name.

    Description
    -----------
    Storage is a sorted ``tuple[tuple[str, int], ...]``, not a
    mapping: frozen dataclasses used as jit-cache-key components must
    be hashable. The constructor accepts any ``Mapping[str, int]``
    and normalizes, so value-equal mappings produce equal (and
    equally hashing) specs. ``grow`` and ``merge_max`` are the two
    accumulation rules of the halo-accounting trace: sequential
    un-synced applications *add*, parallel expression branches *max*.

    Parameters
    ----------
    widths : Mapping[str, int]
        Per-coordinate-name ghost widths; all widths must be >= 0.
    """

    widths: tuple[tuple[str, int], ...]

    def __init__(self, widths: Mapping[str, int]) -> None:
        """Normalize the mapping to sorted tuple storage (hashable)."""
        items = []
        for name in sorted(widths):
            width = widths[name]
            if width < 0:
                raise ValueError(
                    f"halo width along {name!r} must be >= 0, "
                    f"got {width}")
            items.append((name, int(width)))
        object.__setattr__(self, "widths", tuple(items))

    @classmethod
    def zero(cls, names: tuple[str, ...]) -> HaloSpec:
        """
        Build a spec with width 0 on every name.

        Parameters
        ----------
        names : tuple[str, ...]
            The coordinate names the spec covers.

        Returns
        -------
        HaloSpec
            The all-zero spec over `names`.
        """
        return cls(dict.fromkeys(names, 0))

    def __getitem__(self, name: str) -> int:
        """
        Return the width along `name`.

        Parameters
        ----------
        name : str
            A coordinate name covered by this spec.

        Returns
        -------
        int
            The ghost width along `name`.

        Raises
        ------
        KeyError
            If `name` is not covered by this spec.
        """
        for key, width in self.widths:
            if key == name:
                return width
        raise KeyError(name)

    def grow(self, name: str, by: int) -> HaloSpec:
        """
        Return a new spec with `name` widened by `by`.

        Description
        -----------
        The *sequential* accumulation rule: un-synced composition
        chains add their per-operator widths.

        Parameters
        ----------
        name : str
            A coordinate name covered by this spec.
        by : int
            The additional width; must be >= 0.

        Returns
        -------
        HaloSpec
            The widened spec; `self` is unchanged.
        """
        if by < 0:
            raise ValueError(f"grow amount must be >= 0, got {by}")
        merged = dict(self.widths)
        merged[name] = self[name] + by
        return HaloSpec(merged)

    def merge_max(self, other: HaloSpec) -> HaloSpec:
        """
        Return the pointwise maximum of two specs.

        Description
        -----------
        The *parallel* accumulation rule: independent tendency terms
        contribute their maximum, not their sum. The result covers
        the union of the two name sets; a name missing from one spec
        counts as width 0.

        Parameters
        ----------
        other : HaloSpec
            The spec to merge with.

        Returns
        -------
        HaloSpec
            The pointwise-maximum spec over the union of names.
        """
        merged = dict(self.widths)
        for name, width in other.widths:
            merged[name] = max(merged.get(name, 0), width)
        return HaloSpec(merged)
