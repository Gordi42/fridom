"""
Info, cost, and progress records for state transforms (task 2.8).

Description
-----------
The info/cost/progress vocabulary of
``notes/framework2/model/08_state_transforms.md`` §10.1: everything
is **returned** or **observed**, nothing mutates the transform. The
one law: ``T(s) == call_with_info(s)[0]`` bitwise; info composes
structurally (a tree mirroring the composition tree). No mutating
``last_info`` attribute, no ``return_details=True`` (both rejected).
Owning class spec: ``notes/framework2/model/classes/transforms.md``
§"TransformInfo, TransformCost, TransformProgress".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


@dataclass(frozen=True)
class TransformInfo:

    """
    Structural call record; a tree mirroring the composition tree.

    Description
    -----------
    The one info spelling (§10.7.1): produced by ``call_with_info``,
    never mutated onto the transform. Composite nodes label their
    children for path addressing (the labels are the repr's node
    names). Unknown attribute reads resolve into ``extra`` (spec
    completion 1: keeps sketch 7.9's ``info.stopped_by`` spelling
    without widening the base field list).

    Parameters
    ----------
    iterations : int | None, optional
        Iteration count (FixedPoint); ``None`` for leaves/composites
        (default: None).
    errors : tuple[float, ...], optional
        The convergence error series (FixedPoint) (default: ()).
    model_steps : int, optional
        Internal model steps executed (Tier 2) (default: 0).
    elapsed_model_time : float, optional
        The final internal-model clock of a Tier-2 call; never in the
        returned State (default: 0.0).
    extra : Mapping[str, object], optional
        Preset-specific detail (``stopped_by``, ...) (default: {}).
    children : tuple[tuple[str, TransformInfo], ...], optional
        Labeled child infos, composition-tree order (default: ()).
    """

    iterations: int | None = None
    errors: tuple[float, ...] = ()
    model_steps: int = 0
    elapsed_model_time: float = 0.0
    extra: Mapping[str, object] = field(default_factory=dict)
    children: tuple[tuple[str, TransformInfo], ...] = ()

    EMPTY: ClassVar[TransformInfo]

    def __getitem__(self, path: str | int) -> TransformInfo:
        """
        Return a child info by composition-tree label or index.

        Parameters
        ----------
        path : str | int
            The child's label, or its positional index.

        Returns
        -------
        TransformInfo
            The addressed child info.

        Raises
        ------
        KeyError
            On an unknown label.
        IndexError
            On an out-of-range index.
        """
        if isinstance(path, int) and not isinstance(path, bool):
            return self.children[path][1]
        for label, child in self.children:
            if label == path:
                return child
        raise KeyError(
            f"no child info labeled {path!r}; children: "
            f"{tuple(label for label, _ in self.children)}")

    def __getattr__(self, name: str) -> object:
        """Read-only fallback into ``extra`` (spec completion 1)."""
        if name.startswith("_"):
            raise AttributeError(name)
        extra = object.__getattribute__(self, "__dict__").get("extra")
        if extra is not None and name in extra:
            return extra[name]
        raise AttributeError(
            f"TransformInfo has no attribute or extra key {name!r}")


TransformInfo.EMPTY = TransformInfo()


@dataclass(frozen=True)
class TransformCost:

    """
    Static cost estimate: internal model steps.

    Description
    -----------
    Answers "what will this cost *before* calling"; sums structurally
    under composition (``FixedPoint`` reports ``max_it *`` the
    per-iteration steps, flagged ``upper_bound``).

    Parameters
    ----------
    model_steps : int, optional
        Internal model steps (Tier 1: zero) (default: 0).
    upper_bound : bool, optional
        Whether the estimate is an upper bound (default: False).
    """

    model_steps: int = 0
    upper_bound: bool = False

    def __add__(self, other: TransformCost) -> TransformCost:
        """Add two costs: steps sum, the upper-bound flag ORs."""
        if not isinstance(other, TransformCost):
            return NotImplemented
        return TransformCost(
            model_steps=self.model_steps + other.model_steps,
            upper_bound=self.upper_bound or other.upper_bound)

    def __mul__(self, factor: int) -> TransformCost:
        """Scale the step count (``Power``/``FixedPoint``)."""
        if isinstance(factor, bool) or not isinstance(factor, int):
            return NotImplemented
        return TransformCost(
            model_steps=self.model_steps * factor,
            upper_bound=self.upper_bound)

    __rmul__ = __mul__


@dataclass(frozen=True)
class TransformProgress:

    """
    Host-side observer payload for progress hooks (spec completion 2).

    Description
    -----------
    Delivered to ``on_progress`` (Tier-2 constructors) and
    ``on_iteration`` (FixedPoint), path-prefixed through composites.
    Observers may never influence results (normative).

    Parameters
    ----------
    path : str
        The composition-tree path of the reporting node.
    steps_done : int, optional
        Steps/iterations completed (default: 0).
    steps_total : int | None, optional
        Planned total, when known (default: None).
    elapsed_seconds : float, optional
        Wall-clock seconds elapsed (default: 0.0).
    """

    path: str
    steps_done: int = 0
    steps_total: int | None = None
    elapsed_seconds: float = 0.0
