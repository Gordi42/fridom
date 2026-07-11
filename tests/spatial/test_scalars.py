"""Tests for the static scalar markers (spatial/scalars.py)."""
import pytest

from fridom.spatial.scalars import Complex, Real, Scalars, Variance


# ================================================================
#  Scalars
# ================================================================
def test_scalars_members():
    assert list(Scalars) == [Scalars.REAL, Scalars.COMPLEX]


def test_module_level_aliases():
    assert Real is Scalars.REAL
    assert Complex is Scalars.COMPLEX


def test_scalars_are_hashable_singletons():
    key = {Scalars.REAL: "rfft", Scalars.COMPLEX: "fft"}
    assert key[Real] == "rfft"
    assert key[Complex] == "fft"
    assert Scalars["REAL"] is Real


@pytest.mark.parametrize("member", list(Scalars))
def test_scalars_identity_equality(member):
    assert member is Scalars[member.name]
    assert hash(member) == hash(Scalars[member.name])


# ================================================================
#  Variance
# ================================================================
def test_variance_members():
    assert list(Variance) == [Variance.COVARIANT, Variance.CONTRAVARIANT]


def test_variance_is_hashable():
    assert len({Variance.COVARIANT, Variance.CONTRAVARIANT}) == 2
    assert Variance.COVARIANT is not Variance.CONTRAVARIANT
