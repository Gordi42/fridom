"""Tests for fridom.framework2.io.triggers (algebra + lowering)."""
import numpy as np
import pytest

from fridom.framework2.io.triggers import (
    At,
    Every,
    Trigger,
    Union,
    Window,
    at,
    every,
    lower_trigger,
)


# ================================================================
#  every() — the cadence factory
# ================================================================
def test_every_requires_exactly_one_cadence():
    with pytest.raises(ValueError, match="exactly one cadence"):
        every()
    with pytest.raises(ValueError, match="exactly one cadence"):
        every(steps=2, seconds=1.0)


def test_every_steps_node():
    trig = every(steps=4)
    assert isinstance(trig, Every)
    assert trig.steps == 4
    assert trig.seconds is None
    assert trig.has_walltime is False


def test_every_steps_validation():
    with pytest.raises(ValueError, match=">= 1"):
        every(steps=0)
    with pytest.raises(TypeError, match="int"):
        every(steps=2.5)


def test_every_time_cadences_fold_into_seconds():
    assert every(seconds=30.0).seconds == 30.0
    assert every(minutes=2).seconds == 120.0
    assert every(hours=1.5).seconds == 5400.0
    assert every(days=1).seconds == 86400.0


def test_every_accepts_timedelta64():
    assert every(seconds=np.timedelta64(90, "s")).seconds == 90.0
    assert every(hours=np.timedelta64(2, "h")).seconds == 2 * 3600.0


def test_every_rejects_nonpositive_period():
    with pytest.raises(ValueError, match="positive"):
        every(seconds=0.0)
    with pytest.raises(ValueError, match="positive"):
        every(hours=-1.0)


def test_every_walltime_spellings():
    assert every(walltime="7.5h").walltime == 7.5 * 3600.0
    assert every(walltime="90m").walltime == 5400.0
    assert every(walltime=3600.0).walltime == 3600.0
    assert every(walltime=np.timedelta64(2, "h")).walltime == 7200.0


def test_every_walltime_sets_has_walltime():
    assert every(walltime="1h").has_walltime is True


def test_every_walltime_bad_string():
    with pytest.raises(ValueError, match="walltime"):
        every(walltime="7.5x")
    with pytest.raises(ValueError, match="walltime"):
        every(walltime="h")


# ================================================================
#  at() — explicit model times
# ================================================================
def test_at_builds_node():
    trig = at([1.0, 2.5])
    assert isinstance(trig, At)
    assert trig.times == (1.0, 2.5)
    assert trig.has_walltime is False


def test_at_accepts_timedelta64():
    trig = at([np.timedelta64(90, "s"), 3.0])
    assert trig.times == (90.0, 3.0)


def test_at_rejects_empty_and_scalar():
    with pytest.raises(ValueError, match="at least one"):
        at([])
    with pytest.raises(TypeError, match="sequence"):
        at(1.0)


# ================================================================
#  Union and window algebra
# ================================================================
def test_or_builds_flat_union():
    a, b, c = every(steps=2), at([1.0]), every(seconds=5.0)
    union = a | b | c
    assert isinstance(union, Union)
    assert union.operands == (a, b, c)


def test_union_has_walltime_propagates():
    mixed = every(steps=1) | every(walltime="1h")
    assert mixed.has_walltime is True
    plain = every(steps=1) | at([1.0])
    assert plain.has_walltime is False


def test_or_with_non_trigger_is_a_type_error():
    with pytest.raises(TypeError):
        _ = every(steps=1) | 3


def test_window_wrapping_and_has_walltime():
    trig = every(steps=2, after=1.0, until=2.0)
    assert isinstance(trig, Window)
    assert isinstance(trig.inner, Every)
    assert trig.after == 1.0
    assert trig.until == 2.0
    assert trig.has_walltime is False
    walled = every(walltime="1h", until=5.0)
    assert walled.has_walltime is True


def test_triggers_are_frozen_data():
    trig = every(steps=2)
    assert isinstance(trig, Trigger)
    with pytest.raises(AttributeError):
        trig.steps = 3


# ================================================================
#  lower_trigger — step cadences and step-0 inclusion
# ================================================================
def test_lower_steps_includes_step_zero():
    ks = lower_trigger(every(steps=4), t0=0.0, dt=0.1, n_steps=10)
    assert ks == (0, 4, 8)


def test_lower_steps_every_step():
    ks = lower_trigger(every(steps=1), t0=0.0, dt=0.1, n_steps=3)
    assert ks == (0, 1, 2, 3)


def test_lower_steps_ignores_t0_and_dt_sign():
    trig = every(steps=3)
    forward = lower_trigger(trig, t0=5.0, dt=0.1, n_steps=10)
    backward = lower_trigger(trig, t0=5.0, dt=-0.1, n_steps=10)
    assert forward == backward == (0, 3, 6, 9)


# ================================================================
#  lower_trigger — model-time cadences (sign-agnostic)
# ================================================================
def test_lower_seconds_exact_multiples():
    ks = lower_trigger(every(seconds=0.5), t0=0.0, dt=0.1, n_steps=10)
    assert ks == (0, 5, 10)


def test_lower_seconds_snaps_up_with_ceil():
    # firing times 0, 0.25, 0.5, 0.75, 1.0 -> k = 0, 2.5, 5, 7.5, 10
    ks = lower_trigger(every(seconds=0.25), t0=0.0, dt=0.1, n_steps=10)
    assert ks == (0, 3, 5, 8, 10)


def test_lower_seconds_backward_dt_is_sign_agnostic():
    trig = every(seconds=0.5)
    forward = lower_trigger(trig, t0=0.0, dt=0.1, n_steps=10)
    backward = lower_trigger(trig, t0=0.0, dt=-0.1, n_steps=10)
    assert backward == forward == (0, 5, 10)


def test_lower_seconds_subs_step_cadence_fires_every_step():
    ks = lower_trigger(every(seconds=0.01), t0=0.0, dt=0.1, n_steps=4)
    assert ks == (0, 1, 2, 3, 4)


def test_lower_seconds_float_noise_snaps_to_exact_step():
    # 0.3 / 0.1 is 2.9999999999999996 in floats; the snap keeps the
    # exact-multiple firing on step 3, not step 3-then-ceil artifacts
    ks = lower_trigger(every(seconds=0.3), t0=0.0, dt=0.1, n_steps=6)
    assert ks == (0, 3, 6)


# ================================================================
#  lower_trigger — fr.at times
# ================================================================
def test_lower_at_ceil_snap_and_realized_time():
    ks = lower_trigger(at([0.35]), t0=0.0, dt=0.1, n_steps=10)
    assert ks == (4,)
    realized = 0.0 + ks[0] * 0.1
    assert realized == pytest.approx(0.4)


def test_lower_at_backward_maps_decreasing_times_up():
    # k = (t - t0) / dt with dt < 0: decreasing times, increasing k
    ks = lower_trigger(at([-0.35]), t0=0.0, dt=-0.1, n_steps=10)
    assert ks == (4,)
    realized = 0.0 + ks[0] * -0.1
    assert realized == pytest.approx(-0.4)


def test_lower_at_exact_time_with_float_noise():
    ks = lower_trigger(at([0.3]), t0=0.0, dt=0.1, n_steps=10)
    assert ks == (3,)


def test_lower_at_with_nonzero_t0():
    ks = lower_trigger(at([1.4]), t0=1.0, dt=0.1, n_steps=10)
    assert ks == (4,)


def test_lower_at_out_of_interval_errors_at_planning():
    with pytest.raises(ValueError, match="outside the run"):
        lower_trigger(at([2.0]), t0=0.0, dt=0.1, n_steps=10)
    with pytest.raises(ValueError, match="outside the run"):
        lower_trigger(at([-0.1]), t0=0.0, dt=0.1, n_steps=10)
    # sign-agnostic: a positive time is outside a backward run
    with pytest.raises(ValueError, match="outside the run"):
        lower_trigger(at([0.1]), t0=0.0, dt=-0.1, n_steps=10)


def test_lower_at_endpoints_are_inside():
    ks = lower_trigger(at([0.0, 1.0]), t0=0.0, dt=0.1, n_steps=10)
    assert ks == (0, 10)


# ================================================================
#  lower_trigger — unions, windows, walltime
# ================================================================
def test_lower_union_is_sorted_dedup():
    trig = every(steps=4) | at([0.35, 0.8])
    ks = lower_trigger(trig, t0=0.0, dt=0.1, n_steps=10)
    assert ks == (0, 4, 8)


def test_lower_window_filters_run_direction():
    trig = every(steps=1, after=0.35, until=0.75)
    ks = lower_trigger(trig, t0=0.0, dt=0.1, n_steps=10)
    assert ks == (4, 5, 6, 7)


def test_lower_window_backward_is_sign_agnostic():
    trig = every(steps=1, after=-0.35, until=-0.75)
    ks = lower_trigger(trig, t0=0.0, dt=-0.1, n_steps=10)
    assert ks == (4, 5, 6, 7)


def test_lower_window_exact_endpoints_included():
    trig = every(steps=1, after=0.3, until=0.7)
    ks = lower_trigger(trig, t0=0.0, dt=0.1, n_steps=10)
    assert ks == (3, 4, 5, 6, 7)


def test_lower_window_one_sided():
    after_only = every(steps=1, after=0.75)
    assert lower_trigger(
        after_only, t0=0.0, dt=0.1, n_steps=10) == (8, 9, 10)
    until_only = every(steps=1, until=0.25)
    assert lower_trigger(
        until_only, t0=0.0, dt=0.1, n_steps=10) == (0, 1, 2)


def test_lower_walltime_is_the_empty_set():
    ks = lower_trigger(every(walltime="1h"), t0=0.0, dt=0.1,
                       n_steps=10)
    assert ks == ()


def test_lower_mixed_union_drops_only_the_walltime_part():
    trig = every(steps=5) | every(walltime="1h")
    ks = lower_trigger(trig, t0=0.0, dt=0.1, n_steps=10)
    assert ks == (0, 5, 10)


# ================================================================
#  lower_trigger — argument validation
# ================================================================
def test_lower_rejects_zero_dt():
    with pytest.raises(ValueError, match="nonzero"):
        lower_trigger(every(steps=1), t0=0.0, dt=0.0, n_steps=10)


def test_lower_rejects_bad_n_steps():
    with pytest.raises(ValueError, match="n_steps"):
        lower_trigger(every(steps=1), t0=0.0, dt=0.1, n_steps=-1)
    with pytest.raises(TypeError, match="n_steps"):
        lower_trigger(every(steps=1), t0=0.0, dt=0.1, n_steps=2.5)


def test_lower_rejects_non_trigger():
    with pytest.raises(TypeError, match="Trigger"):
        lower_trigger("every 5", t0=0.0, dt=0.1, n_steps=10)


def test_lower_zero_step_run_keeps_step_zero():
    assert lower_trigger(every(steps=3), t0=0.0, dt=0.1,
                         n_steps=0) == (0,)
