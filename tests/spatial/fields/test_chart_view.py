"""The generic read-only ``ChartView`` (state.chart mechanics)."""
import pytest

from fridom.spatial.fields.chart_view import ChartView


class _FakeState:

    """A minimal name -> value mapping standing in for a State."""

    def __init__(self, components):
        self._components = components

    def __getitem__(self, name):
        return self._components[name]


def _identity(state, name):
    """Return the state's component unchanged (an identity hook)."""
    return state[name]


def test_getitem_and_attr_route_through_the_hook():
    state = _FakeState({"u": 1, "v": 2, "w": 3})
    chart = ChartView(state, _identity, ("u", "v", "w"))
    assert chart["w"] == 3
    assert chart.w == 3
    assert chart["u"] == 1
    assert chart.v == 2


def test_velocities_destructure_in_declared_order():
    state = _FakeState({"u": 1, "v": 2, "w": 3})
    chart = ChartView(state, _identity, ("u", "v", "w"))
    u, v, w = chart.velocities
    assert (u, v, w) == (1, 2, 3)


def test_getitem_write_is_refused():
    chart = ChartView(_FakeState({"w": 3}), _identity, ("w",))
    with pytest.raises(TypeError, match="read-only"):
        chart["w"] = 0


def test_attr_write_is_refused():
    chart = ChartView(_FakeState({"w": 3}), _identity, ("w",))
    with pytest.raises(AttributeError, match="read-only"):
        chart.w = 0


def test_getitem_requires_a_string_name():
    chart = ChartView(_FakeState({}), _identity, ())
    with pytest.raises(TypeError, match="indexed by name"):
        _ = chart[0]


def test_underscore_attribute_raises_attributeerror():
    chart = ChartView(_FakeState({}), _identity, ())
    with pytest.raises(AttributeError):
        _ = chart._not_a_component


def test_repr_carries_the_velocity_names():
    chart = ChartView(_FakeState({}), _identity, ("u", "v", "w"))
    text = repr(chart)
    assert "ChartView" in text
    assert "velocities" in text
