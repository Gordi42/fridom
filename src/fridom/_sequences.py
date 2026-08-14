"""Scalar-or-sequence normalization for the public keyword surface.

Description
-----------
Public entry points that take a collection accept a bare element
wherever a one-element collection would do, because passing exactly
one is the overwhelmingly common case: ``run(outputs=writer)`` rather
than ``run(outputs=(writer,))``, ``modules_extra=wave_maker`` rather
than ``modules_extra=(wave_maker,)``.

One rule covers every such parameter, so none of them has to be
learned separately:

    **a list or a tuple is many; anything else is one.**

No iterability test is involved, and that is the point. Duck-typing
"is this a sequence?" splices a ``str`` into its characters
(``fields="temp"`` is one field name, never four), and would splice
any element type that later grows an ``__iter__`` — fridom already
carries classes that iterate over *names*
(:class:`fridom.model.module.BindParameterView`,
:class:`fridom.io.Series`), so that is not a hypothetical failure
mode. Whitelisting the two container types cannot fail that way. The
cost is that an exotic container (a ``set``, a generator, a
``dict.keys()`` view) counts as a single element and fails loudly
downstream; wrap those in ``list()``.

The rule is deliberately **not** applied where the element type is
itself a tuple — ``Sequence[tuple[OutputStream, Path]]`` in
:mod:`fridom.io.streams`, the composer's ``terms`` / ``stages``,
``sel_specs`` in the graded operators. There ``(a, b)`` is genuinely
ambiguous between one pair and two elements. Those parameters are
internal and stay strict.
"""
from __future__ import annotations


# ================================================================
#  Normalization
# ================================================================
def as_tuple(value: object) -> tuple:
    """
    Normalize a scalar-or-sequence argument to a tuple.

    Description
    -----------
    The module rule in one function: a ``list`` or a ``tuple`` is a
    collection of elements and passes through as a tuple; anything
    else — including a ``str``, which is why no iterability test is
    used — is a single element and is wrapped. Call it once, at the
    public boundary; everything below keeps its strict tuple
    contract.

    ``None`` is *not* special-cased, because a ``None`` default
    usually carries its own meaning (``Writer(fields=None)`` selects
    the lifecycle default, which is not the same as selecting
    nothing). Guard it at the call site.

    Parameters
    ----------
    value : object
        A list or tuple of elements, or a single element.

    Returns
    -------
    tuple
        ``tuple(value)`` for a list or a tuple, ``(value,)``
        otherwise.

    Examples
    --------
    .. code-block:: python

        as_tuple(writer)            # -> (writer,)
        as_tuple((writer,))         # -> (writer,)
        as_tuple([w1, w2])          # -> (w1, w2)
        as_tuple("temp")            # -> ("temp",)   not the letters
        as_tuple(())                # -> ()
    """
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)
