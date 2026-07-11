"""Tests for the traced clock (model/clock.py).

Covers the traced leaf widths (float64/int64 under the default
x64-on suite), signed tick semantics (it increments on backward
steps too), it exactness, shifted stage clocks, the re-anchor
primitive, reset, the host-side calendar read, the frozen
discipline, and the jaxify round trip.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.model.clock import Clock


# ================================================================
#  Construction and leaf widths
# ================================================================
def test_fresh_clock_starts_at_zero():
    clock = Clock()
    assert float(clock.start) == 0.0
    assert float(clock.elapsed) == 0.0
    assert int(clock.it) == 0
    assert clock.start_date is None


def test_leaf_widths_follow_the_global_width():
    # the default suite runs x64-on: float64 time, int64 counter
    clock = Clock(3.0)
    assert clock.start.dtype == jnp.float64
    assert clock.elapsed.dtype == jnp.float64
    assert clock.it.dtype == jnp.int64


def test_start_offset_and_time_axis():
    clock = Clock(100.0)
    assert float(clock.start) == 100.0
    assert float(clock.time) == 100.0
    clock = clock.tick(2.5)
    assert float(clock.time) == 102.5


def test_start_must_be_a_real_number():
    with pytest.raises(TypeError, match="seconds"):
        Clock("tomorrow")


def test_start_date_must_be_datetime64():
    with pytest.raises(TypeError, match="datetime64"):
        Clock(start_date="2000-01-01")


# ================================================================
#  tick — signed advancement
# ================================================================
def test_tick_accumulates_elapsed_and_increments_it():
    clock = Clock()
    for _ in range(3):
        clock = clock.tick(0.5)
    assert float(clock.elapsed) == pytest.approx(1.5)
    assert int(clock.it) == 3


def test_tick_backward_still_increments_it():
    # it counts steps, not direction (the backward-run primitive)
    clock = Clock().tick(-2.0).tick(-2.0)
    assert float(clock.elapsed) == pytest.approx(-4.0)
    assert int(clock.it) == 2


def test_it_is_exact_over_many_steps():
    clock = Clock()
    for _ in range(1000):
        clock = clock.tick(1e-3)
    assert int(clock.it) == 1000


def test_tick_is_functional():
    clock = Clock()
    ticked = clock.tick(1.0)
    assert int(clock.it) == 0
    assert int(ticked.it) == 1
    assert ticked is not clock


# ================================================================
#  shifted — stage clocks
# ================================================================
def test_shifted_moves_time_but_not_it():
    clock = Clock(10.0).tick(1.0)
    stage = clock.shifted(0.25)
    assert float(stage.time) == pytest.approx(11.25)
    assert int(stage.it) == int(clock.it)


def test_shifted_carries_the_sign():
    stage = Clock().shifted(-0.5)
    assert float(stage.elapsed) == pytest.approx(-0.5)
    assert int(stage.it) == 0


# ================================================================
#  reset and reanchored
# ================================================================
def test_reset_preserves_start_and_zeroes_progress():
    anchor = np.datetime64("2000-01-01T00:00:00")
    clock = Clock(7.0, start_date=anchor).tick(1.0).tick(1.0)
    fresh = clock.reset()
    assert float(fresh.start) == 7.0
    assert float(fresh.elapsed) == 0.0
    assert int(fresh.it) == 0
    assert fresh.start_date == anchor


def test_reanchored_overwrites_elapsed_only():
    clock = Clock(5.0).tick(1.0).tick(1.0)
    anchored = clock.reanchored(2.0000001)
    assert float(anchored.elapsed) == pytest.approx(2.0000001)
    assert float(anchored.start) == 5.0
    assert int(anchored.it) == 2
    assert anchored.elapsed.dtype == clock.elapsed.dtype


# ================================================================
#  The host-side calendar read
# ================================================================
def test_date_reads_the_calendar_anchor():
    anchor = np.datetime64("2000-01-01T00:00:00")
    clock = Clock(start_date=anchor).tick(90.0)
    assert clock.date == np.datetime64("2000-01-01T00:01:30")


def test_date_without_anchor_raises():
    with pytest.raises(ValueError, match="calendar anchor"):
        _ = Clock().date


# ================================================================
#  Frozen discipline
# ================================================================
def test_clock_is_frozen():
    clock = Clock()
    with pytest.raises(AttributeError, match="frozen"):
        clock.elapsed = jnp.asarray(1.0)
    with pytest.raises(AttributeError, match="frozen"):
        del clock.it


# ================================================================
#  Pytree behavior
# ================================================================
def test_jaxify_round_trip():
    anchor = np.datetime64("2000-01-01")
    clock = Clock(3.0, start_date=anchor).tick(0.5)
    leaves, treedef = jax.tree_util.tree_flatten(clock)
    assert len(leaves) == 3  # start, elapsed, it
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is Clock
    assert float(rebuilt.start) == 3.0
    assert float(rebuilt.elapsed) == pytest.approx(0.5)
    assert int(rebuilt.it) == 1
    assert rebuilt.start_date == anchor


def test_start_date_is_static_aux_not_a_leaf():
    clock = Clock(start_date=np.datetime64("2000-01-01"))
    leaves = jax.tree_util.tree_leaves(clock)
    assert not any(isinstance(leaf, np.datetime64) for leaf in leaves)


def test_tick_works_under_jit():
    @jax.jit
    def advance(clock, dt):
        return clock.tick(dt)

    clock = advance(Clock(), jnp.asarray(2.0))
    clock = advance(clock, jnp.asarray(-0.5))
    assert float(clock.elapsed) == pytest.approx(1.5)
    assert int(clock.it) == 2


def test_repr_smoke():
    text = repr(Clock(start_date=np.datetime64("2000-01-01")))
    assert "Clock(" in text
    assert "start_date" in text
