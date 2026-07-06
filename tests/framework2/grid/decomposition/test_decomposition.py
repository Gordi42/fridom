"""Tests for the Decomposition ABC, ReshardingReport, and SpaceLike."""
import dataclasses
from dataclasses import dataclass

import pytest

from fridom.framework2.grid.decomposition.decomposition import (
    Decomposition,
    ReshardingReport,
    SpaceLike,
)
from fridom.framework2.grid.decomposition.layout import Layout


@dataclass(frozen=True)
class StandInSpace:

    """Minimal frozen stand-in implementing the SpaceLike protocol."""

    shape: tuple
    names: tuple
    layout: object = None

    @property
    def factors(self):
        return (self,)

    def factor(self, name):
        if name in self.names:
            return self
        raise KeyError(name)


def test_decomposition_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Decomposition()


def test_abstract_surface_matches_class_doc():
    expected = {
        "halo", "default_layout", "layouts", "sharding",
        "local_slice", "storage_shape", "zeros", "pad", "unpad",
        "sync", "layout_for", "redistribute", "gather",
    }
    assert set(Decomposition.__abstractmethods__) == expected


def test_stand_in_satisfies_space_protocol():
    space = StandInSpace(shape=(4,), names=("x",))
    assert isinstance(space, SpaceLike)
    assert space.factor("x") is space
    with pytest.raises(KeyError):
        space.factor("y")


def test_resharding_report_fields():
    old = Layout({})
    new = Layout({"x": "px"})
    report = ReshardingReport(old=old, new=new, changed=True)
    assert report.old == old
    assert report.new == new
    assert report.changed


def test_resharding_report_hashable_and_frozen():
    report = ReshardingReport(
        old=Layout({}), new=Layout({}), changed=False)
    same = ReshardingReport(
        old=Layout({}), new=Layout({}), changed=False)
    assert report == same
    assert hash(report) == hash(same)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.changed = True
