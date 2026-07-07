"""
The normative ops protocols.

Description
-----------
Free-standing protocol-level utilities, importable by any driver
(CS-3: ``run()`` must be implementable purely on ``advance`` plus
these): ``WalltimeGuard`` — the predictive walltime budget check —
and the ``ProgressReporter`` protocol with its ``ChunkStats``
payload. The three reporter hook NAMES are the normative
commitment; the payload dataclass is provisional and grows
compatibly. Owning class spec:
``notes/framework2/model/classes/io_ops.md``.
"""
# Wave 4 C: WalltimeGuard, ProgressReporter, ChunkStats
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# smoothing factor of the chunk-rate EMA (weight of the newest
# observation)
_EMA_ALPHA: Final[float] = 0.3


# ================================================================
#  WalltimeGuard — predictive walltime budget check
# ================================================================
class WalltimeGuard:

    """
    Predictive walltime budget check.

    Description
    -----------
    Stops the run BEFORE the chunk that would blow the budget:
    ``should_stop()`` is true once ``elapsed + predicted_next_chunk
    + snapshot_margin > budget``. The prediction is a smoothed
    (EMA) chunk rate fed through ``on_chunk`` — the same
    observation stream the progress reporter renders. Documented
    caveat (d4_3 risk 3): a single chunk exceeding the remaining
    allocation still dies — bounding ``max_chunk`` is the
    mitigation. Multi-process consensus (rank-0 broadcast at
    boundaries) is the parked 3.2/3.3 residual.

    Parameters
    ----------
    budget : float
        The walltime budget in wall seconds (from the Snapshots
        walltime trigger).
    snapshot_margin : float, optional
        Wall seconds reserved for the final snapshot store. When
        ``None`` the margin is measured from the snapshots fed
        through ``on_snapshot`` (0.0 until the first one).
    clock : callable, optional
        Monotonic wall-clock source in seconds; a mockable seam,
        ``time.monotonic`` by default.
    """

    def __init__(
        self,
        budget: float,
        *,
        snapshot_margin: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        budget = float(budget)
        if budget <= 0.0:
            raise ValueError(
                f"budget must be positive wall seconds; got "
                f"{budget}")
        if snapshot_margin is not None:
            snapshot_margin = float(snapshot_margin)
            if snapshot_margin < 0.0:
                raise ValueError(
                    f"snapshot_margin must be >= 0; got "
                    f"{snapshot_margin}")
        self._budget = budget
        self._fixed_margin = snapshot_margin
        self._measured_margin = 0.0
        self._clock = clock
        self._start = clock()
        self._ema: float | None = None

    # ------------------------------------------------------------
    #  Read-only accounting
    # ------------------------------------------------------------
    @property
    def budget(self) -> float:
        """The walltime budget in wall seconds."""
        return self._budget

    @property
    def elapsed(self) -> float:
        """Wall seconds since the guard was created."""
        return self._clock() - self._start

    @property
    def snapshot_margin(self) -> float:
        """The reserved store time: fixed, or measured (max seen)."""
        if self._fixed_margin is not None:
            return self._fixed_margin
        return self._measured_margin

    @property
    def predicted_chunk_seconds(self) -> float:
        """The EMA prediction of the next chunk's wall time."""
        return self._ema if self._ema is not None else 0.0

    # ------------------------------------------------------------
    #  Observations
    # ------------------------------------------------------------
    def on_chunk(self, wall_seconds: float) -> None:
        """Feed one chunk's wall time into the smoothed rate.

        Parameters
        ----------
        wall_seconds : float
            The chunk's wall time; must be >= 0.
        """
        wall_seconds = float(wall_seconds)
        if wall_seconds < 0.0:
            raise ValueError(
                f"wall_seconds must be >= 0; got {wall_seconds}")
        if self._ema is None:
            self._ema = wall_seconds
        else:
            self._ema = (_EMA_ALPHA * wall_seconds
                         + (1.0 - _EMA_ALPHA) * self._ema)

    def on_snapshot(self, wall_seconds: float) -> None:
        """Feed one snapshot store's wall time (measured margin).

        Ignored when a fixed ``snapshot_margin`` was given; the
        measured margin is the largest store time seen.

        Parameters
        ----------
        wall_seconds : float
            The snapshot store's wall time; must be >= 0.
        """
        wall_seconds = float(wall_seconds)
        if wall_seconds < 0.0:
            raise ValueError(
                f"wall_seconds must be >= 0; got {wall_seconds}")
        self._measured_margin = max(
            self._measured_margin, wall_seconds)

    # ------------------------------------------------------------
    #  The predictive check
    # ------------------------------------------------------------
    def should_stop(self) -> bool:
        """Whether to stop BEFORE the chunk that would blow it.

        Returns
        -------
        bool
            True once ``elapsed + predicted_next_chunk +
            snapshot_margin > budget``.
        """
        return (self.elapsed
                + self.predicted_chunk_seconds
                + self.snapshot_margin) > self._budget


# ================================================================
#  ProgressReporter — the normative progress protocol
# ================================================================
@dataclass(frozen=True)
class ChunkStats:

    """
    Per-chunk observation.

    Description
    -----------
    The payload is provisional and grows compatibly; the three
    reporter hook NAMES are normative. Fed at each chunk boundary
    with one scalar D2H piggybacking on the panic sync.

    Parameters
    ----------
    name : str or None
        Model attribution (``model.name``).
    iteration : int
        The iteration after the chunk.
    time : float
        The model time after the chunk.
    steps_done : int
        Steps advanced in this chunk.
    wall_seconds : float
        The chunk's wall time.
    steps_per_second : float
        The chunk's step rate.
    """

    name: str | None
    iteration: int
    time: float
    steps_done: int
    wall_seconds: float
    steps_per_second: float


@runtime_checkable
class ProgressReporter(Protocol):

    """
    The progress protocol; ``progress=False | Reporter()``.

    Description
    -----------
    The old ProgressBar module's host-side successor: per-chunk
    cadence, zero extra host syncs, rank-0-only rendering in the
    default implementation (2.6). In-trace ``io_callback`` progress
    ticks are rejected (archive, d4_2 section 4).
    """

    def on_run_start(
        self, *, models: Any, n_steps: int | None,
    ) -> None:
        """Receive the run-start event (once, at ``__enter__``)."""
        ...

    def on_chunk(self, stats: ChunkStats) -> None:
        """Receive one chunk-boundary observation (post panic sync)."""
        ...

    def on_run_end(self, results: Any) -> None:
        """Receive the run-end event (every exit path, teardown)."""
        ...
