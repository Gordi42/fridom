"""
The multi-model driver session (``fr.ops``).

Description
-----------
Wave 5 A: ``Session`` (single-model form; the ``run()`` substrate).
A context manager that plans triggers, binds outputs, resume-checks
snapshots, and walks each model boundary-to-boundary via
``model.advance`` while running the normative boundary sequence
(device sync -> panic check -> writer flush in binding order ->
progress ``on_chunk`` -> walltime check). ``run()`` is reimplemented
as a single-model Session loop (the facade law:
``run() == a hand-written Session loop``, chunk for chunk, bitwise,
same jit-cache entries). Owning class spec:
``notes/framework2/model/classes/io_ops.md`` (the ``fr.ops.Session``
section) and ``notes/framework2/model/04_run_loop_io.md`` §6.3.

The constructor signature is PROVISIONAL (normative note, §6.3): the
protocols and the ``__enter__``/``advance``/``__exit__`` duties are
the commitment; the exact signature is confirm-at-first-use.

Iteration 1 is solid for ONE model; the multi-model maturation is
3.2. The ``models`` tuple and per-model bookkeeping shape are kept so
3.2 does not require a rewrite, but multi-model output routing and
the cross-model window/clock assertions are not implemented here.
"""
# Wave 5 A: Session (single-model form; run() substrate)
from __future__ import annotations

import contextlib
import logging
import signal
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from fridom.framework2.io.snapshots import (
    Snapshots,
    find_latest,
    rotate,
)
from fridom.framework2.io.streams import (
    dedupe_streams,
    reject_snapshots_config,
    reject_walltime_trigger,
)
from fridom.framework2.io.triggers import (
    Every,
    Trigger,
    Union,
    Window,
    lower_trigger,
)
from fridom.framework2.model.model import _CHUNK_COMPILE_LOG
from fridom.framework2.model.results import (
    AdvanceResult,
    PanicError,
    RunResult,
    RunStatus,
)
from fridom.framework2.ops.protocols import (
    ChunkStats,
    WalltimeGuard,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence
    from types import TracebackType
    from typing import Self

    from fridom.framework2.io.streams import OutputStream
    from fridom.framework2.model.model import Model

_log = logging.getLogger(__name__)

# the default host-sync granularity (04 §6.3: max_chunk auto ~256).
# When None the model's own chunk_size is used (its {C, 1} granularity
# — so a Session subdivision never forces sub-chunk length-1 tails).
_DEFAULT_MAX_CHUNK: int | None = None

# the sign-agnostic run-target snap tolerance (04 §6.3: ceil(k - eps)).
_TARGET_EPS: float = 1e-9

# process-global guard: a model bound to one active Session may not be
# entered into another (store-collision protection; §11.2 guard).
_BOUND_MODELS: set[int] = set()
_BOUND_LOCK = threading.Lock()


# ================================================================
#  Small host helpers
# ================================================================
def _compile_seconds_total() -> float:
    """Sum the recorded chunk-compile seconds across the process."""
    return float(sum(sec for sec, _ in _CHUNK_COMPILE_LOG.values()))


def _to_seconds(value: object, *, name: str) -> float:
    """Convert a model-time spelling to float seconds."""
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "s"))
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name}= takes float seconds or np.timedelta64; got "
            f"{value!r}") from exc


def _walltime_budget(trigger: Trigger) -> float | None:
    """Extract the walltime budget (wall seconds) from a trigger.

    Returns the smallest walltime cadence found across the trigger's
    nodes, or ``None`` when the trigger carries no walltime component.
    """
    budgets: list[float] = []

    def _walk(node: Trigger) -> None:
        if isinstance(node, Union):
            for operand in node.operands:
                _walk(operand)
        elif isinstance(node, Window):
            _walk(node.inner)
        elif isinstance(node, Every) and node.walltime is not None:
            budgets.append(float(node.walltime))

    _walk(trigger)
    return min(budgets) if budgets else None


class _Book:

    """Per-model run bookkeeping (transient; discarded at exit)."""

    def __init__(self, name: str | None) -> None:
        """Start the counters cleared."""
        self.name = name
        self.status = RunStatus.COMPLETED
        self.steps_done = 0
        self.wall_seconds = 0.0


class _LoggingProgress:

    """The minimal default progress reporter (rank-0, logs).

    Description
    -----------
    The full tqdm/StringIO renderer is a 2.6 item; iteration 1 ships
    a rank-0-guarded logging reporter so ``progress=True`` has a
    concrete default. Custom reporters implement the same three
    normative hook names.
    """

    def __init__(self) -> None:
        """Guard rendering to process rank 0."""
        self._rank0 = jax.process_index() == 0

    def on_run_start(self, *, models: Any, n_steps: int | None) -> None:
        """Log the run start (rank 0 only)."""
        if self._rank0:
            _log.info("run start: %d model(s), n_steps=%s",
                      len(models), n_steps)

    def on_chunk(self, stats: ChunkStats) -> None:
        """Log one chunk-boundary observation (rank 0 only)."""
        if self._rank0:
            _log.debug(
                "chunk: model=%s it=%d t=%.6g steps=%d rate=%.1f/s",
                stats.name, stats.iteration, stats.time,
                stats.steps_done, stats.steps_per_second)

    def on_run_end(self, results: Any) -> None:
        """Log the run end (rank 0 only)."""
        if self._rank0:
            _log.info("run end: %s",
                      {k: v.status.value for k, v in results.items()})


# ================================================================
#  Session
# ================================================================
class Session:

    """
    Context manager over N models + outputs/snapshots/progress.

    Description
    -----------
    Composes the walltime, progress, and stream-binding protocols
    (io_ops cluster). Iteration 1 drives ONE model solidly; the
    models tuple and per-model bookkeeping are kept for the 3.2
    multi-model maturation. Use inside a ``with`` block:
    ``advance``/``active``/``result`` outside it raise a hinted
    ``RuntimeError``; a Session is single-use.

    Parameters
    ----------
    models : Model or sequence of Model
        The model(s) to drive; keyed by ``model.name`` (a lone
        unnamed model gets the key ``"model"``). Duplicate or
        missing names in a multi-model session raise.
    outputs : sequence of OutputStream, optional
        Standing + per-run output streams; bound at ``__enter__``.
        Rejects a ``Snapshots`` instance (run-config only)
        (default: ()).
    snapshots : Snapshots or None, optional
        The restart-snapshot run config (the only home for a
        ``Snapshots``); its walltime component feeds the
        ``WalltimeGuard`` (default: None).
    progress : bool or ProgressReporter, optional
        ``True`` installs the default logging reporter; ``False``
        disables progress; a reporter object is used as-is
        (default: True).
    max_chunk : int or None, optional
        Host-sync granularity: the largest sub-advance between
        boundaries. ``None`` uses each model's ``chunk_size``
        (default: None).
    jit : bool, optional
        Reserved chunk-dispatch policy; ``jit=False`` (eager) is not
        wired in iteration 1 (default: True).
    debug_nan : bool, optional
        Forwarded to ``model.advance`` — keep a chunk-start carry
        copy for ``replay_nan`` (default: False).
    """

    def __init__(
        self,
        models: Model | Sequence[Model],
        *,
        outputs: Sequence[OutputStream] = (),
        snapshots: Snapshots | None = None,
        progress: bool | Any = True,
        max_chunk: int | None = _DEFAULT_MAX_CHUNK,
        jit: bool = True,
        debug_nan: bool = False,
    ) -> None:
        """Validate and store config; no binding happens here."""
        self._models = self._keyed_models(models)
        if snapshots is not None and not isinstance(snapshots, Snapshots):
            raise TypeError(
                "snapshots= must be an fr.io.Snapshots config; got "
                f"{snapshots!r}")
        for stream in outputs:
            reject_snapshots_config(stream, slot="outputs")
        self._outputs = tuple(outputs)
        self._snapshots = snapshots
        if max_chunk is not None and (
                isinstance(max_chunk, bool)
                or not isinstance(max_chunk, int)
                or max_chunk < 1):
            raise ValueError(
                f"max_chunk must be an int >= 1 or None; got "
                f"{max_chunk!r}")
        self._max_chunk = max_chunk
        if not jit:
            raise NotImplementedError(
                "Session(jit=False) — eager mode is not wired in "
                "iteration 1 (advance() always runs the jitted "
                "step_chunk); it lands with the run(jit=False) "
                "per-term timing table")
        self._jit = jit
        self._debug_nan = bool(debug_nan)
        if progress is True:
            self._reporter: Any = _LoggingProgress()
        elif progress is False:
            self._reporter = None
        else:
            self._reporter = progress
        # lifecycle flags
        self._entered = False
        self._exited = False
        # transient run state (built at __enter__)
        self._book: dict[str, _Book] = {}
        self._bound: tuple[tuple[OutputStream, Any], ...] = ()
        self._t0: dict[str, float] = {}
        self._it0: dict[str, int] = {}
        self._interrupted = False
        self._walltime_stop = False
        self._guard: WalltimeGuard | None = None
        self._compile_before = 0.0
        self._prev_sigint: Any = None
        self._snapshot_written = False

    # ------------------------------------------------------------
    #  Construction helpers
    # ------------------------------------------------------------
    @staticmethod
    def _keyed_models(
        models: Model | Sequence[Model],
    ) -> dict[str, Model]:
        """Key the model(s) by ``model.name`` (hinted collisions)."""
        if isinstance(models, (list, tuple)):
            items = tuple(models)
        else:
            items = (models,)
        if not items:
            raise ValueError("a Session needs at least one model")
        keyed: dict[str, Model] = {}
        for model in items:
            name = model.name
            if name is None:
                if len(items) > 1:
                    raise ValueError(
                        "a multi-model Session needs a name= on every "
                        "model (keys are model.name); an unnamed model "
                        "is only allowed as the lone model")
                name = "model"
            if name in keyed:
                raise ValueError(
                    f"two models share the name {name!r}; Session "
                    "keys models by model.name — give each a unique "
                    "name=")
            keyed[name] = model
        return keyed

    @property
    def models(self) -> Mapping[str, Model]:
        """The bound models, keyed by ``model.name``."""
        return dict(self._models)

    # ------------------------------------------------------------
    #  Enter: plan, bind, resume-check, install handlers
    # ------------------------------------------------------------
    def __enter__(self) -> Self:
        """Plan triggers, bind outputs, resume, start progress.

        Returns
        -------
        Session
            ``self`` (the entered session).
        """
        if self._entered:
            raise RuntimeError(
                "this Session was already entered; a Session is "
                "single-use — build a fresh one")
        self._entered = True
        models = tuple(self._models.values())
        with _BOUND_LOCK:
            already = [m for m in models if id(m) in _BOUND_MODELS]
            if already:
                names = ", ".join(
                    repr(m.name) for m in already)
                raise RuntimeError(
                    f"model(s) {names} are already bound to an active "
                    "Session — one active Session per model (store-"
                    "collision protection)")
            for model in models:
                _BOUND_MODELS.add(id(model))
        try:
            self._enter_body()
        except BaseException:
            with _BOUND_LOCK:
                for model in models:
                    _BOUND_MODELS.discard(id(model))
            self._exited = True
            raise
        return self

    def _enter_body(self) -> None:
        """Run the enter duties after the models are reserved."""
        # 1. resume-check snapshots (before capturing t0/it0, so the
        #    restored clock anchors the plan) --------------------------
        if self._snapshots is not None and self._snapshots.resume:
            self._resume()
        # 2. per-model bookkeeping + clock anchors ---------------------
        for name, model in self._models.items():
            self._book[name] = _Book(name)
            clock = model.clock
            self._t0[name] = float(clock.time)
            self._it0[name] = int(clock.it)
        # 3. bind outputs (dedupe by path, reject walltime) ------------
        self._bind_outputs()
        # 4. walltime guard --------------------------------------------
        if self._snapshots is not None:
            budget = _walltime_budget(self._snapshots.trigger)
            if budget is not None:
                self._guard = WalltimeGuard(budget)
        # 5. interrupt handler + progress start ------------------------
        self._install_interrupt()
        self._compile_before = _compile_seconds_total()
        if self._reporter is not None:
            self._reporter.on_run_start(
                models=self._models, n_steps=None)

    def _bind_outputs(self) -> None:
        """Dedupe by resolved path, reject walltime, call bind()."""
        pairs = []
        for stream in self._outputs:
            reject_walltime_trigger(
                stream.trigger,
                stream=type(stream).__name__)
            pairs.append((stream, self._stream_path(stream)))
        self._bound = dedupe_streams(pairs)
        # single-model iteration 1: every stream binds to the one model
        model = next(iter(self._models.values()))
        for stream, _path in self._bound:
            stream.bind(model)

    @staticmethod
    def _stream_path(stream: OutputStream) -> Any:
        """Return the stream's store path (the dedupe key)."""
        path = getattr(stream, "path", None)
        if path is None:
            # a store-less stream (test doubles): dedupe by identity
            return f"<stream {id(stream)}>"
        return path

    def _resume(self) -> None:
        """Load the newest complete snapshot into the model(s).

        Iteration 1 resume is single-model: find the newest complete
        snapshot, load it into the model (fingerprint + dt checks are
        the model's), and flip bound writers to append + truncate.
        """
        snapshots = self._snapshots
        if snapshots is None:  # pragma: no cover — guarded by caller
            return
        latest = find_latest(snapshots.path)
        if latest is None:
            return
        model = next(iter(self._models.values()))
        model.load_snapshot(latest)
        it_snap = int(model.clock.it)
        for stream in self._outputs:
            truncate = getattr(stream, "truncate_after", None)
            if callable(truncate):
                truncate(it_snap)

    def _install_interrupt(self) -> None:
        """Install the graceful first-Ctrl-C handler (main thread)."""
        self._prev_sigint = None
        if threading.current_thread() is not threading.main_thread():
            return  # signals only install on the main thread
        try:
            self._prev_sigint = signal.signal(
                signal.SIGINT, self._on_sigint)
        except (ValueError, OSError):  # pragma: no cover
            self._prev_sigint = None

    def _on_sigint(
        self, signum: int, frame: Any,  # noqa: ARG002 — signal API
    ) -> None:
        """First Ctrl-C flags a graceful stop; second re-raises."""
        if self._interrupted:
            # second interrupt: restore and let it raise
            if self._prev_sigint is not None:
                signal.signal(signal.SIGINT, self._prev_sigint)
            raise KeyboardInterrupt
        self._interrupted = True

    # ------------------------------------------------------------
    #  Active / result
    # ------------------------------------------------------------
    @property
    def active(self) -> bool:
        """Keep-going predicate: no panic, interrupt, or walltime.

        Returns
        -------
        bool
            ``True`` while no model has panicked, no interrupt has
            been received, and the predictive walltime check has
            headroom. Does NOT track target exhaustion (the caller's
            condition).
        """
        self._require_active("active")
        if self._interrupted or self._walltime_stop:
            return False
        return not any(m.panicked for m in self._models.values())

    @property
    def result(self) -> Mapping[str, RunResult]:
        """Aggregated per-model results (shared status enum)."""
        self._require_active("result")
        return self._aggregate()

    def _aggregate(self) -> dict[str, RunResult]:
        """Build the per-model ``RunResult`` mapping from bookkeeping."""
        compile_seconds = max(
            0.0, _compile_seconds_total() - self._compile_before)
        results: dict[str, RunResult] = {}
        for name, model in self._models.items():
            book = self._book.get(name) or _Book(name)
            clock = model.clock
            run_seconds = max(0.0, book.wall_seconds - compile_seconds)
            rate = (book.steps_done / run_seconds
                    if run_seconds > 0.0 else 0.0)
            results[name] = RunResult(
                status=book.status,
                steps_done=book.steps_done,
                final_it=int(clock.it),
                final_time=float(clock.time),
                compile_seconds=compile_seconds,
                run_seconds=run_seconds,
                steps_per_second=rate)
        return results

    # ------------------------------------------------------------
    #  Advance
    # ------------------------------------------------------------
    def advance(
        self,
        plan: Mapping[Any, int] | None = None,
        /,
        **steps_by_name: int,
    ) -> Mapping[str, AdvanceResult]:
        """Advance each named model by its step count.

        Description
        -----------
        Dispatch-then-sync is built in (each model's chunks dispatch
        before its panic read). Each model's quantum is subdivided at
        its planned trigger boundaries; at each boundary the normative
        sequence runs: sync -> panic check -> writer flush (binding
        order) -> snapshot write+rotate -> progress ``on_chunk`` ->
        walltime check. On interrupt or walltime the call returns
        early (``AdvanceResult.steps_done`` < requested; ``active``
        flips); on panic ``advance`` raises ``PanicError`` at the
        abort boundary.

        Parameters
        ----------
        plan : mapping of (Model or str) to int, optional
            Per-model step counts, keyed by the model object or its
            name (default: None).
        **steps_by_name : int
            Per-model step counts keyed by name (the
            ``s.advance(atm=10, ocn=1)`` spelling).

        Returns
        -------
        mapping of str to AdvanceResult
            The per-model advance results (name-keyed).
        """
        self._require_active("advance")
        requests = self._resolve_plan(plan, steps_by_name)
        results: dict[str, AdvanceResult] = {}
        for name, steps in requests.items():
            results[name] = self._advance_one(name, steps)
        return results

    def _resolve_plan(
        self,
        plan: Mapping[Any, int] | None,
        steps_by_name: Mapping[str, int],
    ) -> dict[str, int]:
        """Normalize the plan/kwargs into a name -> steps mapping."""
        requests: dict[str, int] = {}
        by_model = {id(m): name for name, m in self._models.items()}
        if plan is not None:
            for key, steps in plan.items():
                if isinstance(key, str):
                    name = key
                elif id(key) in by_model:
                    name = by_model[id(key)]
                else:
                    raise KeyError(
                        f"advance plan key {key!r} is not a model of "
                        "this Session")
                requests[name] = int(steps)
        for name, steps in steps_by_name.items():
            requests[name] = int(steps)
        unknown = [n for n in requests if n not in self._models]
        if unknown:
            raise KeyError(
                f"advance targets {unknown} are not models of this "
                f"Session (models: {sorted(self._models)})")
        for name, steps in requests.items():
            if isinstance(steps, bool) or steps < 0:
                raise ValueError(
                    f"advance({name}=...) needs a non-negative int; "
                    f"got {steps!r}")
        return requests

    def _advance_one(self, name: str, steps: int) -> AdvanceResult:
        """Advance one model boundary-to-boundary; run the sequence."""
        model = self._models[name]
        book = self._book[name]
        cursor = int(model.clock.it) - self._it0[name]  # local start
        end_local = cursor + steps
        firings = self._firing_sets(name, end_local)
        snap_firings = self._snapshot_firings(name, end_local)
        boundaries = self._boundaries(
            name, cursor, end_local, firings, snap_firings)
        total_steps = 0
        total_wall = 0.0
        panicked = False
        panic_it: int | None = None
        # step-0 initial outputs (every()/at include step 0) ----------
        if cursor == 0:
            self._fire_boundary(
                name, model, 0, firings, snap_firings,
                chunk_steps=0, wall_seconds=0.0, is_chunk=False)
        try:
            for boundary in boundaries:
                if self._interrupted:
                    book.status = RunStatus.INTERRUPTED
                    break
                sub = boundary - cursor
                if sub <= 0:
                    continue
                result = model.advance(sub, debug_nan=self._debug_nan)
                cursor = boundary
                total_steps += result.steps_done
                total_wall += result.wall_seconds
                book.steps_done += result.steps_done
                book.wall_seconds += result.wall_seconds
                stop = self._fire_boundary(
                    name, model, boundary, firings, snap_firings,
                    chunk_steps=result.steps_done,
                    wall_seconds=result.wall_seconds, is_chunk=True)
                if stop or self._interrupted:
                    if self._interrupted:
                        book.status = RunStatus.INTERRUPTED
                    break
        except PanicError as err:
            book.status = RunStatus.NAN_ABORT
            if err.partial is not None:
                book.steps_done += err.partial.steps_done
                total_steps += err.partial.steps_done
            panicked = True
            panic_it = err.first_bad_it
            raise
        except KeyboardInterrupt:
            # a reporter/stream hook raised mid-loop: graceful stop,
            # the in-flight chunk already committed (zero steps lost)
            self._interrupted = True
            book.status = RunStatus.INTERRUPTED
        return AdvanceResult(
            steps_done=total_steps, panicked=panicked,
            panic_it=panic_it, wall_seconds=total_wall)

    # ------------------------------------------------------------
    #  Boundary planning + firing
    # ------------------------------------------------------------
    def _firing_sets(
        self, name: str, n_local: int,
    ) -> list[tuple[OutputStream, frozenset[int]]]:
        """Lower each bound stream's trigger over ``[0, n_local]``."""
        dt = float(self._models[name]._stepper.dt)  # noqa: SLF001
        t0 = self._t0[name]
        sets: list[tuple[OutputStream, frozenset[int]]] = []
        for stream, _path in self._bound:
            fired = lower_trigger(
                stream.trigger, t0=t0, dt=dt, n_steps=n_local)
            sets.append((stream, frozenset(fired)))
        return sets

    def _snapshot_firings(
        self, name: str, n_local: int,
    ) -> frozenset[int]:
        """Lower the snapshot trigger's model-time firings to steps."""
        if self._snapshots is None:
            return frozenset()
        dt = float(self._models[name]._stepper.dt)  # noqa: SLF001
        return frozenset(lower_trigger(
            self._snapshots.trigger, t0=self._t0[name], dt=dt,
            n_steps=n_local))

    def _boundaries(
        self,
        name: str,
        cursor: int,
        end_local: int,
        firings: list[tuple[OutputStream, frozenset[int]]],
        snap_firings: frozenset[int],
    ) -> list[int]:
        """Boundaries = firings in the window + endpoint, subdivided."""
        union: set[int] = set(snap_firings)
        for _stream, fired in firings:
            union |= fired
        window = sorted(b for b in union if cursor < b <= end_local)
        if end_local > cursor:
            window.append(end_local)
        window = sorted(set(window))
        max_chunk = self._max_chunk_for(name)
        out: list[int] = []
        prev = cursor
        for boundary in window:
            while boundary - prev > max_chunk:
                prev += max_chunk
                out.append(prev)
            out.append(boundary)
            prev = boundary
        return out

    def _max_chunk_for(self, name: str) -> int:
        """Return the host-sync granularity for one model (>= 1)."""
        if self._max_chunk is not None:
            return self._max_chunk
        return self._models[name].chunk_size

    def _fire_boundary(
        self,
        name: str,
        model: Model,
        boundary: int,
        firings: list[tuple[OutputStream, frozenset[int]]],
        snap_firings: frozenset[int],
        *,
        chunk_steps: int,
        wall_seconds: float,
        is_chunk: bool,
    ) -> bool:
        """Run the boundary sequence; return whether to stop.

        The order (normative, §6.3): writer flush (binding order) ->
        snapshot write + rotate -> progress ``on_chunk`` -> walltime
        check. ``model.advance`` already performed the device sync and
        the panic check (it raises on panic before we get here).
        """
        carry = model.carry  # the boundary-synced, copy-on-read carry
        # -- writers (binding order) ----------------------------------
        for stream, fired in firings:
            if boundary in fired:
                stream.write(carry)
        # -- snapshot write + rotate ----------------------------------
        if boundary in snap_firings:
            self._write_snapshot(model)
        # -- progress + walltime (only after a real chunk) ------------
        if is_chunk:
            if self._reporter is not None:
                clock = model.clock
                rate = (chunk_steps / wall_seconds
                        if wall_seconds > 0.0 else 0.0)
                self._reporter.on_chunk(ChunkStats(
                    name=name,
                    iteration=int(clock.it),
                    time=float(clock.time),
                    steps_done=int(chunk_steps),
                    wall_seconds=float(wall_seconds),
                    steps_per_second=float(rate)))
            if self._guard is not None:
                self._guard.on_chunk(wall_seconds)
                if self._guard.should_stop():
                    self._walltime_stop = True
                    self._book[name].status = RunStatus.WALLTIME
                    return True
        return False

    def _write_snapshot(self, model: Model) -> None:
        """Write one committed snapshot subdir + rotate."""
        snapshots = self._snapshots
        if snapshots is None:  # pragma: no cover — guarded by caller
            return
        base = Path(snapshots.path)
        it = int(model.clock.it)
        target = base / f"it{it:012d}"
        if target.exists():  # already snapshotted this iteration
            return
        started = time.perf_counter()
        model.snapshot(target)
        if self._guard is not None:
            self._guard.on_snapshot(time.perf_counter() - started)
        if snapshots.keep is not None:
            rotate(base, snapshots.keep)
        self._snapshot_written = True

    # ------------------------------------------------------------
    #  Exit
    # ------------------------------------------------------------
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Cleanup guarantees on every exit path; never suppresses.

        Order: wait for in-flight device work -> writer flush + close
        (partial output survives) -> triggered snapshot -> on_walltime.
        On ``PanicError``: writers are flushed but there is NO
        automatic crash snapshot (carries left in memory for autopsy).
        Always returns ``False``.
        """
        panicking = exc_type is not None and issubclass(
            exc_type, PanicError)
        cleanup_error: BaseException | None = None
        try:
            self._exit_body(panicking=panicking)
        except Exception as err:  # noqa: BLE001 — chained below
            cleanup_error = err
        # teardown (always) -------------------------------------------
        try:
            if self._reporter is not None:
                self._reporter.on_run_end(self._aggregate())
        except Exception:  # noqa: BLE001 — teardown must not mask
            _log.exception("progress on_run_end failed")
        self._restore_interrupt()
        with _BOUND_LOCK:
            for model in self._models.values():
                _BOUND_MODELS.discard(id(model))
        self._exited = True
        if cleanup_error is not None:
            if exc is None:
                raise cleanup_error
            # an exception is already in flight: it wins; the cleanup
            # failure is chained via __context__ and logged
            _log.error(
                "Session cleanup failed while an exception was in "
                "flight; the in-flight exception propagates",
                exc_info=cleanup_error)
        return False

    def _exit_body(self, *, panicking: bool) -> None:
        """Run the ordered exit guarantees (before teardown)."""
        # 1. wait for in-flight device work ---------------------------
        for model in self._models.values():
            jax.block_until_ready(
                jax.tree_util.tree_leaves(model.carry))
        # 2. writer flush + close (partial output survives) -----------
        for stream, _path in self._bound:
            try:
                stream.close()
            except Exception:  # noqa: BLE001 — one bad sink must not
                _log.exception(  # block the others / the snapshot
                    "closing output stream %r failed", stream)
        if panicking:
            return  # no automatic crash snapshot (autopsy)
        # 3. triggered snapshot (walltime exit path) ------------------
        if self._walltime_stop and self._snapshots is not None:
            for model in self._models.values():
                self._write_snapshot(model)
            # 4. on_walltime action -----------------------------------
            action = self._snapshots.on_walltime
            if action is not None:
                action()

    def _restore_interrupt(self) -> None:
        """Restore the previous SIGINT handler, if we installed one."""
        if self._prev_sigint is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGINT, self._prev_sigint)
            self._prev_sigint = None

    # ------------------------------------------------------------
    #  Guards
    # ------------------------------------------------------------
    def _require_active(self, what: str) -> None:
        """Raise unless inside the ``with`` block."""
        if not self._entered or self._exited:
            raise RuntimeError(
                f"Session.{what} is only valid inside the `with "
                "fr.ops.Session(...) as s:` block (a Session is "
                "single-use; enter it first)")

    def __repr__(self) -> str:
        """Compact host-side summary."""
        state = ("exited" if self._exited
                 else "active" if self._entered else "unentered")
        return (f"<Session {state}: models="
                f"{tuple(self._models)}, outputs={len(self._outputs)}>")
