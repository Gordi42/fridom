"""In-memory scalar time series (``fr.io.Series``).

Description
-----------
The in-memory twin of `fr.io.TimeSeries`: the same
``(model_state) -> Field | scalar`` columns, the same trigger, the
same chunk-boundary evaluation — but the rows are kept as Python
lists and read back as arrays after the run instead of being appended
to a CSV. Owning class spec:
``design/specs/model/classes/io_ops.md`` (the TimeSeries section; the
sink is the only difference).

The point is the interactive / notebook / docs case, where the series
is wanted *as data* and a file on disk is pure friction::

    series = fr.io.Series(
        {"energy": lambda ms: metric.inner(ms.state, ms.state),
         "buoyancy": lambda ms: ms.state["b"].integrate()},
        trigger=fr.io.every(steps=60))
    model.run(runlen=3600.0, outputs=(series,))
    plt.plot(series["time"], series["energy"])

Because the run loop plans its chunk boundaries from the bound
streams' triggers (``fr.ops.Session._boundaries``), one ``run()`` with
a ``Series`` attached is the same chunk plan — and the same arithmetic
— as a hand-written loop of short ``run()`` calls with a host-side
accumulator between them. The columns are evaluated on the
boundary-synced carry, on the host, outside every trace: a diagnostic
never enters the compiled step and never triggers a recompile.

**Memory.** A ``Series`` retains every firing for the lifetime of the
object, so it is a *bounded-run* sink: 61 rows of five columns is
nothing, 10^7 rows of fifty is not. Long production runs want
`fr.io.TimeSeries` (flushed per firing, tail-able, survives an abort).
This is not the time-integral state CS-11 sends to the carry — a
``Series`` never feeds the model, never restarts one, and holds no
running sums; it is a sink holding its own output, exactly as the CSV
holds its rows.
"""
# Wave 5 C: Series (in-memory scalar sink)
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fridom.io.streams import (
    LEADING_COLUMNS,
    check_columns,
    coerce_scalar,
    reject_walltime_trigger,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator, Mapping

    from fridom.io.triggers import Trigger


# ================================================================
#  Series
# ================================================================
class Series:

    """
    Trigger-driven scalar time series kept in memory.

    Description
    -----------
    Reusable configuration until :meth:`bind`, a growing table of
    rows after it (host infrastructure — never a pytree, never
    traced). Implements the ``OutputStream`` protocol; it owns no
    store, so it exposes no ``path`` and dedupes by identity.

    Rows accumulate across ``bind``/``close`` cycles, so a series
    survives a sequence of ``run()`` calls as one axis. A firing whose
    iteration equals the last recorded one **replaces** that row
    rather than appending, which is what keeps the axis fork-free when
    a second ``run()`` re-fires its start-of-run boundary on the state
    the previous run ended at.

    Parameters
    ----------
    columns : Mapping[str, Callable]
        Named scalar expressions, each a pure
        ``(model_state) -> Field | scalar`` callable returning a 0-d
        value; the mapping order is the column order. The names
        ``"iteration"`` and ``"time"`` are reserved for the
        coordinates.
    trigger : Trigger
        The firing trigger (walltime-bearing triggers are rejected at
        bind).
    """

    def __init__(
        self,
        columns: Mapping[str, Callable],
        *,
        trigger: Trigger,
    ) -> None:
        """Configure the columns; nothing is evaluated here."""
        columns = check_columns(columns, owner="Series")
        reserved = [name for name in columns if name in LEADING_COLUMNS]
        if reserved:
            raise ValueError(
                f"Series column names {reserved!r} are reserved for "
                f"the coordinates {list(LEADING_COLUMNS)!r}; rename "
                "them")
        self._columns = columns
        self._trigger = trigger
        self._names = tuple(columns)
        self._rows: list[tuple[float, ...]] = []
        self._iterations: list[int] = []
        self._bound = False

    # ================================================================
    #  Protocol slots
    # ================================================================
    @property
    def trigger(self) -> Trigger:
        """The stream's firing trigger (protocol slot)."""
        return self._trigger

    def bind(self, model: Any) -> None:
        """
        Reject walltime triggers and dry-evaluate every column.

        Description
        -----------
        Called at RUN START. Dry-evaluates every column on the model's
        carry (checking scalar-ness with a hinted error) so a broken
        column fails before the run rather than at the first firing.
        Recorded rows are **kept**: rebinding continues the series.

        Parameters
        ----------
        model : fr.model.Model
            The bound model (duck-typed: ``carry`` — the model_state
            the columns evaluate on).
        """
        if self._bound:
            raise RuntimeError(
                "this Series is already bound; close() it before "
                "binding it to another run")
        reject_walltime_trigger(self._trigger, stream="Series")
        carry = getattr(model, "carry", None)
        if carry is None:
            raise TypeError(
                "Series.bind expects a model exposing a carry "
                f"(model_state); got {type(model).__name__}")
        for name, function in self._columns.items():
            coerce_scalar(function(carry), column=name, owner="Series")
        self._bound = True

    def write(self, model_state: Any) -> None:
        """
        Record one row from the boundary-synced carry.

        Parameters
        ----------
        model_state : object
            The synced carry: ``clock`` (the iteration/time
            coordinates) plus whatever the column callables read.
        """
        self._require_bound()
        clock = model_state.clock
        iteration = int(np.asarray(clock.it))
        row = (float(np.asarray(clock.time)), *(
            coerce_scalar(function(model_state), column=name,
                          owner="Series")
            for name, function in self._columns.items()))
        if self._iterations and self._iterations[-1] == iteration:
            self._rows[-1] = row
            return
        self._iterations.append(iteration)
        self._rows.append(row)

    def truncate_after(self, iteration: int) -> None:
        """
        Drop rows past the given iteration coordinate.

        Description
        -----------
        Iteration-keyed resume alignment (V-S1), the in-memory twin of
        the CSV tail rewrite: every row whose iteration is
        ``<= iteration`` is kept, the rest dropped.

        Parameters
        ----------
        iteration : int
            The snapshot iteration; rows past it are dropped.
        """
        self._require_bound()
        keep = sum(1 for it in self._iterations if it <= iteration)
        del self._iterations[keep:]
        del self._rows[keep:]

    def close(self) -> None:
        """
        Release the stream; the recorded rows survive.

        Description
        -----------
        Idempotent. Nothing is buffered (each firing appends a
        completed row), so ``close`` only drops the bound state — the
        series is read *after* the run, so the rows must outlive it.
        """
        self._bound = False

    # ================================================================
    #  Reading the series
    # ================================================================
    @property
    def columns(self) -> tuple[str, ...]:
        """The user column names, in mapping order."""
        return self._names

    @property
    def iteration(self) -> np.ndarray:
        """The recorded iteration coordinates."""
        return np.array(self._iterations, dtype=int)

    @property
    def time(self) -> np.ndarray:
        """The recorded model-time coordinates."""
        return self._column_array(0)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return every coordinate and column as a name -> array map.

        Returns
        -------
        dict[str, np.ndarray]
            ``{"iteration": ..., "time": ..., <column>: ...}``.
        """
        return {name: self[name]
                for name in (*LEADING_COLUMNS, *self._names)}

    def __getitem__(self, name: str) -> np.ndarray:
        """Return one coordinate or column as an array."""
        if name == "iteration":
            return self.iteration
        if name in LEADING_COLUMNS:
            return self.time
        try:
            index = self._names.index(name)
        except ValueError:
            known = [*LEADING_COLUMNS, *self._names]
            raise KeyError(
                f"Series has no column {name!r}; it has {known}",
            ) from None
        return self._column_array(index + 1)

    def __contains__(self, name: object) -> bool:
        """Whether a coordinate or column of this name exists."""
        return name in LEADING_COLUMNS or name in self._names

    def __iter__(self) -> Iterator[str]:
        """Iterate the coordinate and column names."""
        return iter((*LEADING_COLUMNS, *self._names))

    def __len__(self) -> int:
        """Return the number of recorded rows."""
        return len(self._rows)

    def __repr__(self) -> str:
        """Show the column names and the row count."""
        return (f"Series(columns={list(self._names)}, "
                f"rows={len(self._rows)})")

    # ================================================================
    #  Helpers
    # ================================================================
    def _column_array(self, index: int) -> np.ndarray:
        """Return one stored row slot as a float array."""
        return np.array([row[index] for row in self._rows],
                        dtype=float)

    def _require_bound(self) -> None:
        """Raise if used before bind."""
        if not self._bound:
            raise RuntimeError(
                "this Series is not bound; call bind(model) at run "
                "start before write/truncate_after")
