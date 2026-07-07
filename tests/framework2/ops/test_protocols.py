"""Tests for fridom.framework2.ops.protocols."""
import dataclasses

import pytest

from fridom.framework2.ops.protocols import (
    ChunkStats,
    ProgressReporter,
    WalltimeGuard,
)


class FakeClock:

    """A hand-cranked monotonic clock."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# ================================================================
#  ChunkStats
# ================================================================
def test_chunkstats_is_frozen_plain_data():
    stats = ChunkStats(name="ocn", iteration=256, time=25.6,
                       steps_done=256, wall_seconds=1.28,
                       steps_per_second=200.0)
    assert stats.name == "ocn"
    assert stats.iteration == 256
    assert stats.steps_per_second == 200.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.iteration = 0


def test_chunkstats_allows_unnamed_models():
    stats = ChunkStats(name=None, iteration=1, time=0.1,
                       steps_done=1, wall_seconds=0.01,
                       steps_per_second=100.0)
    assert stats.name is None


# ================================================================
#  ProgressReporter protocol (the three normative hook names)
# ================================================================
class FullReporter:

    """Implements all three normative hooks."""

    def on_run_start(self, *, models, n_steps):
        pass

    def on_chunk(self, stats):
        pass

    def on_run_end(self, results):
        pass


class MissingChunkHook:

    """No on_chunk — must fail the protocol check."""

    def on_run_start(self, *, models, n_steps):
        pass

    def on_run_end(self, results):
        pass


def test_reporter_protocol_isinstance():
    assert isinstance(FullReporter(), ProgressReporter)
    assert not isinstance(MissingChunkHook(), ProgressReporter)
    assert not isinstance(object(), ProgressReporter)


# ================================================================
#  WalltimeGuard — construction
# ================================================================
def test_guard_validates_budget_and_margin():
    with pytest.raises(ValueError, match="budget"):
        WalltimeGuard(0.0)
    with pytest.raises(ValueError, match="budget"):
        WalltimeGuard(-10.0)
    with pytest.raises(ValueError, match="snapshot_margin"):
        WalltimeGuard(100.0, snapshot_margin=-1.0)


def test_guard_rejects_negative_observations():
    guard = WalltimeGuard(100.0, clock=FakeClock())
    with pytest.raises(ValueError, match="wall_seconds"):
        guard.on_chunk(-1.0)
    with pytest.raises(ValueError, match="wall_seconds"):
        guard.on_snapshot(-1.0)


# ================================================================
#  WalltimeGuard — the predictive stop
# ================================================================
def test_fresh_guard_does_not_stop():
    guard = WalltimeGuard(100.0, snapshot_margin=10.0,
                          clock=FakeClock())
    assert guard.should_stop() is False


def test_guard_stops_before_the_chunk_that_would_blow_it():
    clock = FakeClock()
    guard = WalltimeGuard(100.0, snapshot_margin=10.0, clock=clock)
    stopped_at = None
    for _ in range(10):
        if guard.should_stop():
            stopped_at = clock.now
            break
        clock.now += 20.0  # the chunk runs...
        guard.on_chunk(20.0)  # ...and is observed at the boundary
    # 60 + 20 + 10 = 90 <= 100: continue; 80 + 20 + 10 = 110 > 100:
    # stop — with wall time (80) still short of the budget (100),
    # i.e. BEFORE the chunk that would blow it.
    assert stopped_at == 80.0
    assert stopped_at < guard.budget


def test_guard_boundary_equality_continues():
    clock = FakeClock()
    guard = WalltimeGuard(100.0, snapshot_margin=10.0, clock=clock)
    clock.now = 70.0
    guard.on_chunk(20.0)
    # 70 + 20 + 10 == 100: not strictly over the budget
    assert guard.should_stop() is False
    clock.now = 70.1
    assert guard.should_stop() is True


def test_guard_prediction_is_a_smoothed_ema():
    guard = WalltimeGuard(1000.0, clock=FakeClock())
    guard.on_chunk(10.0)
    assert guard.predicted_chunk_seconds == 10.0
    guard.on_chunk(20.0)
    # alpha = 0.3: 0.3 * 20 + 0.7 * 10
    assert guard.predicted_chunk_seconds == pytest.approx(13.0)


def test_guard_measured_snapshot_margin():
    clock = FakeClock()
    guard = WalltimeGuard(100.0, clock=clock)
    assert guard.snapshot_margin == 0.0
    guard.on_snapshot(7.0)
    assert guard.snapshot_margin == 7.0
    guard.on_snapshot(3.0)  # max seen, not last seen
    assert guard.snapshot_margin == 7.0


def test_guard_fixed_margin_wins_over_measurements():
    guard = WalltimeGuard(100.0, snapshot_margin=5.0,
                          clock=FakeClock())
    guard.on_snapshot(50.0)
    assert guard.snapshot_margin == 5.0


def test_single_oversized_chunk_caveat_is_undetectable():
    # the documented d4_3 risk-3 caveat: the guard predicts from
    # the smoothed history, so a single chunk larger than the whole
    # remaining allocation cannot be foreseen — bounding max_chunk
    # is the mitigation, not this guard.
    clock = FakeClock()
    guard = WalltimeGuard(50.0, clock=clock)
    clock.now = 1.0
    guard.on_chunk(1.0)
    assert guard.should_stop() is False  # predicted 1 s, not 100 s


def test_guard_elapsed_tracks_the_injected_clock():
    clock = FakeClock()
    clock.now = 5.0
    guard = WalltimeGuard(100.0, clock=clock)
    clock.now = 12.5
    assert guard.elapsed == 7.5
