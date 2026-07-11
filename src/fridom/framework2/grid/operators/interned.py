"""
The ``@interned`` class decorator for leaf operators (D6).

Description
-----------
Owning discipline: ``notes/framework2/classes/operators_base.md`` (the
D6 interning rule). Operators are identity-hashed (``base.py``:
``__eq__`` is ``self is other``, ``__hash__`` is ``id``), which is only
sound if structurally-equal operators are the *same object*. The
algebra objects built by the dunders (``Composite``,
``SeparableComposite``, ``ScaledOperator`` on scalar coefficients,
``Dispatched``, ``Reshard``, ``Sync``) already intern on their static
structure through ``_ALGEBRA_TABLE``; this decorator extends the same
guarantee to the concrete *leaf* stencil operators without every leaf
hand-rolling an interning ``__new__`` / ``__copy__`` pair.

Applied to a leaf class it:

- interns each instance on ``(cls, *self._intern_key())`` in the shared
  ``_ALGEBRA_TABLE`` (a ``WeakValueDictionary``), so structurally-equal
  requests return the identical object;
- runs the real constructor (validation intact) inside ``__new__``
  before interning, then makes ``__init__`` a no-op (Python still calls
  it on the returned instance);
- installs a ``__copy__`` that bypasses interning, which is the seam
  ``SeparableOperator._rebind`` (``base.py``) relies on:
  ``copy.copy(base_op)`` must hand back a *fresh mutable* clone so
  ``_rebind`` can stamp ``bound_axis`` on it without corrupting the
  shared unbound singleton.

Requirements on the decorated class
------------------------------------
- Must define ``_intern_key(self) -> tuple`` returning a hashable tuple
  over its *structural* constructor state (explicit — never a
  ``__dict__`` default — so derived/mutable state can't leak into the
  key). The class object is prepended automatically, so distinct
  classes can never collide.
- ``@interned`` must be the **innermost** decorator (closest to
  ``class``, below ``@final`` / ``@fr.utils.jaxify``) so it captures the
  real ``__init__`` before any wrapper. ``fr.utils.jaxify`` leaves
  ``__eq__`` / ``__hash__`` untouched (they are the identity methods on
  ``Operator``) and its ``_tree_unflatten`` rebuilds via
  ``object.__new__`` (bypassing this ``__new__``), so a jaxify pytree
  round-trip is safe.
"""
# Wave 4.x: interned (family-wide leaf operator interning, D6)
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from fridom.framework2.grid.operators.base import _ALGEBRA_TABLE

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

T = TypeVar("T", bound=type)


def interned(cls: T) -> T:
    """
    Intern a leaf operator on its structural constructor state (D6).

    Description
    -----------
    See the module docstring. The decorated class must define
    ``_intern_key(self) -> tuple``; the class is prepended to that key
    automatically.

    Parameters
    ----------
    cls : type
        The leaf operator class (innermost decorator position).

    Returns
    -------
    type
        The same class, with interning ``__new__`` / no-op ``__init__``
        / bypassing ``__copy__`` installed.
    """
    orig_init = cls.__init__

    def __new__(kls: type, *args: Any, **kwargs: Any) -> Any:  # noqa: N807
        """Build + validate, then return the interned singleton."""
        obj = object.__new__(kls)
        orig_init(obj, *args, **kwargs)  # real ctor/validation
        key = (kls, *obj._intern_key())  # noqa: SLF001 — intern seam
        return _ALGEBRA_TABLE.intern(key, lambda: obj)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: N807
        """No-op: ``__new__`` already built and initialized ``self``."""

    def __copy__(self: Any) -> Any:  # noqa: N807
        """Fresh clone bypassing interning (the ``_rebind`` seam).

        ``SeparableOperator._rebind`` does ``copy.copy(base)`` then
        mutates the clone's ``bound_axis``; without this the copy would
        route back through the interning ``__new__`` and hand out the
        shared unbound singleton, which ``_rebind`` would then corrupt.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    cls.__new__ = __new__
    cls.__init__ = __init__
    cls.__copy__ = __copy__
    return cls
