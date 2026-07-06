"""Tests for the per-mesh decomposition traits."""
import dataclasses

import pytest

from fridom.framework2.grid.decomposition.traits import (
    HaloStrategy,
    MeshDecompositionTraits,
)


def test_halo_strategy_members():
    members = {s.name for s in HaloStrategy}
    assert members == {"GHOST", "TRANSPOSE", "LOCAL", "GRAPH"}


def test_default_min_local_size():
    traits = MeshDecompositionTraits(strategies=(HaloStrategy.GHOST,))
    assert traits.min_local_size == 1


def test_strategies_keep_preference_order():
    traits = MeshDecompositionTraits(
        strategies=(HaloStrategy.TRANSPOSE, HaloStrategy.LOCAL))
    assert traits.strategies == (
        HaloStrategy.TRANSPOSE, HaloStrategy.LOCAL)


def test_strategies_normalized_to_tuple():
    traits = MeshDecompositionTraits(
        strategies=[HaloStrategy.GHOST, HaloStrategy.TRANSPOSE])
    assert traits.strategies == (
        HaloStrategy.GHOST, HaloStrategy.TRANSPOSE)
    # normalization keeps the dataclass hashable
    assert isinstance(hash(traits), int)


def test_equality_and_hash_across_reconstruction():
    a = MeshDecompositionTraits(
        strategies=(HaloStrategy.GHOST,), min_local_size=2)
    b = MeshDecompositionTraits(
        strategies=(HaloStrategy.GHOST,), min_local_size=2)
    assert a == b
    assert hash(a) == hash(b)
    assert {a: "entry"}[b] == "entry"


def test_distinct_traits_differ():
    a = MeshDecompositionTraits(strategies=(HaloStrategy.GHOST,))
    b = MeshDecompositionTraits(strategies=(HaloStrategy.LOCAL,))
    assert a != b


def test_frozen():
    traits = MeshDecompositionTraits(strategies=(HaloStrategy.LOCAL,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        traits.min_local_size = 3


def test_empty_strategies_rejected():
    with pytest.raises(ValueError, match="at least one"):
        MeshDecompositionTraits(strategies=())


def test_min_local_size_below_one_rejected():
    with pytest.raises(ValueError, match="min_local_size"):
        MeshDecompositionTraits(
            strategies=(HaloStrategy.GHOST,), min_local_size=0)
