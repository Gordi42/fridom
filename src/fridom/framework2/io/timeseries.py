"""
Scalar time-series output (``fr.io.TimeSeries``).

Description
-----------
Wave 5 C: ``TimeSeries`` (CSV sink, ``truncate_after`` resume). Owning
class spec: ``design/specs/model/classes/io_ops.md`` (the
TimeSeries section).

A ``TimeSeries`` writes named scalar expressions to a tail-able CSV:
one header row from the bound columns, then one row per firing
(``iteration``, ``time``, then the column values). Each column is a
pure ``(model_state) -> Field | scalar`` output (D2.3) returning a
0-d value; jax/numpy scalars and 0-d Fields are coerced to ``float``.
On multi-device the 0-d result is replicated, so the fetch is a
single-scalar read — no gather. Rows are flushed at every firing, so
the file is tail-able during the run and partial output survives an
abort.

``truncate_after(iteration)`` is the CSV resume-alignment mechanic
(the 2.6 residual, decided here): a tail rewrite keyed on the
``iteration`` column — every row whose iteration is ``<= iteration``
is kept, the rest dropped. Implements the ``OutputStream`` protocol.
"""
# Wave 5 C: TimeSeries (CSV sink)
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fridom.framework2.io.streams import reject_walltime_trigger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.framework2.io.triggers import Trigger

# the two leading columns before the user's scalar expressions
_LEADING = ("iteration", "time")


# ================================================================
#  TimeSeries
# ================================================================
class TimeSeries:

    """
    Trigger-driven scalar time series (CSV, tail-able).

    Description
    -----------
    Reusable configuration until :meth:`bind`, a single CSV after it
    (host infrastructure — never a pytree, never traced). Implements
    the ``OutputStream`` protocol with the same binding-split rules as
    ``Writer``.

    Parameters
    ----------
    path : str | Path
        The CSV path (the stream owns it exclusively).
    columns : Mapping[str, Callable]
        Named scalar expressions, each a pure
        ``(model_state) -> Field | scalar`` callable returning a 0-d
        value; the mapping order is the column order.
    trigger : Trigger
        The firing trigger (walltime-bearing triggers are rejected at
        bind).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        columns: Mapping[str, Callable],
        trigger: Trigger,
    ) -> None:
        """Configure the columns; no file IO happens here."""
        columns = dict(columns)
        if not columns:
            raise ValueError(
                "TimeSeries needs at least one column (a named "
                "(model_state) -> scalar callable)")
        for name, function in columns.items():
            if not callable(function):
                raise TypeError(
                    f"TimeSeries column {name!r} must be callable, "
                    f"got {function!r}")
        self._path = Path(path)
        self._columns = columns
        self._trigger = trigger
        self._names = tuple(columns)
        self._header = (*_LEADING, *self._names)
        self._bound = False

    # ================================================================
    #  Protocol slots
    # ================================================================
    @property
    def trigger(self) -> Trigger:
        """The stream's firing trigger (protocol slot)."""
        return self._trigger

    @property
    def path(self) -> Path:
        """The resolved CSV path."""
        return self._path

    def bind(self, model: Any) -> None:
        """
        Reject walltime triggers, dry-evaluate columns, open the CSV.

        Description
        -----------
        Called at RUN START. Dry-evaluates every column on the
        model's carry (checking scalar-ness with a hinted error).
        Opens the store: an existing CSV with a matching header is
        continued (append — the resume/tail-able path); otherwise a
        fresh file with the header row is created.

        Parameters
        ----------
        model : fr.Model
            The bound model (duck-typed: ``carry`` — the model_state
            the columns evaluate on).
        """
        if self._bound:
            raise RuntimeError(
                f"this TimeSeries is already bound to {self._path}")
        reject_walltime_trigger(
            self._trigger, stream=f"TimeSeries({self._path})")
        model_state = self._model_state(model)
        for name, function in self._columns.items():
            self._scalar(function(model_state), name)
        if self._path.exists():
            self._verify_header()
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", newline="") as handle:
                csv.writer(handle).writerow(self._header)
        self._bound = True

    def write(self, model_state: Any) -> None:
        """
        Append one row from the boundary-synced carry.

        Parameters
        ----------
        model_state : object
            The synced carry: ``clock`` (the iteration/time columns)
            plus whatever the column callables read.
        """
        self._require_bound()
        clock = model_state.clock
        row = [int(np.asarray(clock.it)), float(np.asarray(clock.time))]
        row.extend(
            self._scalar(function(model_state), name)
            for name, function in self._columns.items())
        with self._path.open("a", newline="") as handle:
            csv.writer(handle).writerow(row)

    def truncate_after(self, iteration: int) -> None:
        """
        Drop rows past the given iteration coordinate (tail rewrite).

        Description
        -----------
        Iteration-keyed resume alignment (V-S1): reads the CSV, keeps
        the header plus every row whose ``iteration`` column is
        ``<= iteration``, and rewrites the file. Skipping it on resume
        would fork the time axis (a crashed segment may have written
        rows past the snapshot).

        Parameters
        ----------
        iteration : int
            The snapshot iteration; rows past it are dropped.
        """
        self._require_bound()
        with self._path.open("r", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return
        header, body = rows[0], rows[1:]
        kept = [row for row in body
                if row and int(row[0]) <= iteration]
        with self._path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(kept)

    def close(self) -> None:
        """
        Release the stream; partial output survives.

        Description
        -----------
        Idempotent. Rows are flushed per firing (each ``write`` opens,
        appends, and closes), so nothing is buffered — ``close`` only
        drops the bound state.
        """
        self._bound = False

    # ================================================================
    #  Helpers
    # ================================================================
    @staticmethod
    def _model_state(model: Any) -> Any:
        """Return the model's carry (the dry-evaluation model_state)."""
        carry = getattr(model, "carry", None)
        if carry is None:
            raise TypeError(
                "TimeSeries.bind expects a model exposing a carry "
                "(model_state); got "
                f"{type(model).__name__}")
        return carry

    def _verify_header(self) -> None:
        """Check that an existing CSV's header matches the columns."""
        with self._path.open("r", newline="") as handle:
            existing = next(csv.reader(handle), None)
        if existing != list(self._header):
            raise ValueError(
                f"cannot append to {self._path}: its header "
                f"{existing} does not match the bound columns "
                f"{list(self._header)} (schema mismatch on resume)")

    def _scalar(self, value: Any, column: str) -> float:
        """Coerce a column output to a Python float (scalar check).

        A single-element value is a scalar: a plain number, a 0-d
        jax/numpy scalar, or a reduced Field (framework reductions
        leave a length-1 ConstantSpace axis, so shape ``(1,)`` counts).
        """
        candidate = value
        if hasattr(value, "data") and hasattr(value, "shape"):
            if int(np.prod(value.shape)) != 1:
                raise ValueError(
                    f"TimeSeries column {column!r} returned a field "
                    f"of shape {tuple(value.shape)}, not a scalar; "
                    "reduce it first (e.g. f.integrate() / f.mean())")
            candidate = value.data
        array = np.asarray(candidate)
        if array.size != 1:
            raise ValueError(
                f"TimeSeries column {column!r} returned a value of "
                f"shape {array.shape}, not a scalar")
        return float(array.reshape(()))

    def _require_bound(self) -> None:
        """Raise if used before bind."""
        if not self._bound:
            raise RuntimeError(
                f"TimeSeries({self._path}) is not bound; call "
                "bind(model) at run start before write/truncate_after")
