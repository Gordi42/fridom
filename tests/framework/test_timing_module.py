"""Tests for the timing module."""
import fridom.framework as fr
from fridom.framework.timing_module import TimingComponent, TimingModule


# ================================================================
#  TimingComponent
# ================================================================
def test_component_accumulates_time():
    component = TimingComponent("test")
    component.start()
    assert component.is_active
    component.stop()
    assert not component.is_active
    assert component.time >= 0


def test_component_double_start_and_stop():
    component = TimingComponent("test")
    component.start()
    start_time = component.start_time
    # a second start is ignored
    component.start()
    assert component.start_time == start_time

    component.stop()
    elapsed = component.time
    # a second stop is ignored
    component.stop()
    assert component.time == elapsed


def test_component_reset():
    component = TimingComponent("test")
    component.start()
    component.stop()
    component.reset()
    assert component.time == 0.0
    assert not component.is_active


def test_component_string():
    component = TimingComponent("test")
    component.time = 3723.5  # 1 h, 2 min, 3.5 s
    assert str(component).endswith("01:02:03s")


# ================================================================
#  TimingModule
# ================================================================
def test_get_adds_missing_component():
    timer = TimingModule()
    assert len(timer.components) == 1

    component = timer.get("my component")
    assert component.name == "my component"
    assert len(timer.components) == 2

    # a second get returns the same component
    assert timer.get("my component") is component


def test_context_manager():
    timer = TimingModule()
    with timer["my component"] as component:
        assert component.is_active
    assert not component.is_active
    assert timer.get("my component").time >= 0


def test_reset():
    timer = TimingModule()
    with timer["my component"]:
        pass
    timer.reset()
    assert timer.get("my component").time == 0.0
    assert timer.total.time == 0.0


def test_equality_and_hash():
    # all timing modules are equal so that jit cache keys stay stable
    timer_a = TimingModule()
    timer_b = TimingModule()
    with timer_a["my component"]:
        pass

    assert timer_a == timer_b
    assert hash(timer_a) == hash(timer_b)
    assert timer_a.__eq__(42) is NotImplemented


def test_string_representation():
    timer = TimingModule()
    assert "No time recorded" in str(timer)

    timer.total.time = 10.0
    timer.get("my component").time = 5.0
    summary = str(timer)
    assert "Timing Summary" in summary
    assert "my component" in summary
    assert "(50.0%)" in summary
    assert repr(timer) == str(timer)


def test_model_records_timings():
    grid = fr.grid.cartesian.Grid(shape=(4,), domain_size=(1.0,))
    mset = fr.ModelSettingsBase(grid=grid)
    mset.setup()
    assert isinstance(mset.timer, TimingModule)
