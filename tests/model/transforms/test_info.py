"""Tests for TransformInfo / TransformCost / TransformProgress."""
import pytest

from fridom.model.transforms.info import (
    TransformCost,
    TransformInfo,
    TransformProgress,
)


def test_empty_singleton():
    assert isinstance(TransformInfo.EMPTY, TransformInfo)
    assert TransformInfo.EMPTY.model_steps == 0
    assert TransformInfo.EMPTY.children == ()


def test_getitem_by_index_and_label():
    child = TransformInfo(model_steps=3)
    parent = TransformInfo(children=(("0:A", child),))
    assert parent[0] is child
    assert parent["0:A"] is child


def test_getitem_unknown_label():
    parent = TransformInfo(children=(("0:A", TransformInfo()),))
    with pytest.raises(KeyError, match="no child"):
        _ = parent["missing"]


def test_getattr_falls_back_into_extra():
    info = TransformInfo(extra={"stopped_by": "tol"})
    assert info.stopped_by == "tol"


def test_getattr_unknown_raises():
    info = TransformInfo(extra={"a": 1})
    with pytest.raises(AttributeError):
        _ = info.nonexistent


def test_getattr_private_raises():
    info = TransformInfo()
    with pytest.raises(AttributeError):
        _ = info._secret


def test_cost_addition():
    a = TransformCost(model_steps=3)
    b = TransformCost(model_steps=5, upper_bound=True)
    total = a + b
    assert total.model_steps == 8
    assert total.upper_bound is True


def test_cost_multiplication():
    a = TransformCost(model_steps=4, upper_bound=True)
    assert (a * 3).model_steps == 12
    assert (3 * a).model_steps == 12
    assert (a * 3).upper_bound is True


def test_cost_add_notimplemented():
    assert TransformCost().__add__(5) is NotImplemented
    assert TransformCost().__mul__(1.5) is NotImplemented


def test_progress_payload():
    p = TransformProgress(path="P", steps_done=2, steps_total=3)
    assert p.path == "P"
    assert p.steps_done == 2
    assert p.steps_total == 3
    assert p.elapsed_seconds == 0.0
