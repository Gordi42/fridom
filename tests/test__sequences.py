"""The scalar-or-sequence rule itself: list/tuple is many, else one.

The rule is structural on purpose — no iterability test — so the
cases that pin it down are the iterable non-containers: a ``str``
(never its letters) and a mapping-like object that iterates over
names (fridom carries several, and any of them arriving as a single
element must stay one element).
"""
import numpy as np
import pytest

from fridom._sequences import as_tuple


class Named:

    """An element that iterates over names, like a Mapping does."""

    def __init__(self, *names):
        self.names = names

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


# ================================================================
#  list / tuple are many
# ================================================================
@pytest.mark.parametrize("value", [
    pytest.param([], id="empty-list"),
    pytest.param((), id="empty-tuple"),
])
def test_an_empty_container_stays_empty(value):
    assert as_tuple(value) == ()


def test_a_list_becomes_a_tuple_of_the_same_elements():
    a, b = object(), object()
    assert as_tuple([a, b]) == (a, b)


def test_a_tuple_passes_through():
    a, b = object(), object()
    assert as_tuple((a, b)) == (a, b)


# ================================================================
#  everything else is one
# ================================================================
def test_a_bare_object_is_wrapped():
    a = object()
    assert as_tuple(a) == (a,)


def test_a_string_is_one_name_never_its_letters():
    assert as_tuple("temp") == ("temp",)


def test_a_one_letter_string_is_one_name_too():
    # the case that accidentally worked under tuple(), which is what
    # kept the bug hidden until someone wrote a longer field name
    assert as_tuple("u") == ("u",)


def test_an_iterable_non_container_is_one_element():
    named = Named("a", "b")
    assert as_tuple(named) == (named,)


def test_an_array_is_one_element_not_its_entries():
    arr = np.arange(3)
    result = as_tuple(arr)
    assert len(result) == 1
    assert result[0] is arr


def test_none_is_one_element_and_not_special_cased():
    # call sites that give None its own meaning guard it themselves
    assert as_tuple(None) == (None,)
