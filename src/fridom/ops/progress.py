"""
The opt-in rendered progress bar (``fr.ops.ProgressBar``).

Description
-----------
A host-side ``ProgressReporter`` that renders the old
``fr.modules.ProgressBar`` format from the new stack's chunk-boundary
observation stream — zero extra device syncs, rank-0 only. Opt in
with ``model.run(..., progress=fr.ops.ProgressBar())``; ``progress=
True`` keeps the logging placeholder until that default is flipped
separately. Owning class spec:
``design/specs/model/classes/io_ops.md`` (the ``ProgressReporter``
section).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Final, Literal

import jax
from tqdm import tqdm

from fridom.framework.utils import humanize_number

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.ops.protocols import ChunkStats

_log = logging.getLogger(__name__)

# the NOTICE level (fridom.framework.logger adds it); log mode renders
# there so a batch job's progress survives the default fridom level
_NOTICE: Final[int] = 25

# the old bar's format, ported verbatim (framework/modules/
# progress_bar.py: bar_format + postfix)
_BAR_FORMAT: Final[str] = (
    "{percentage:3.2f}%|{bar}| [{elapsed}<{remaining}]{postfix}")

# the first chunk of a leg pays the jit compilation; its rate is not
# the model's rate and is marked as such
_COMPILE_MARK: Final[str] = "  (first chunk: incl. compile)"

_MODES: Final[tuple[str, ...]] = (
    "auto", "notebook", "tty", "log", "off")


# ================================================================
#  ProgressBar
# ================================================================
class ProgressBar:

    """
    The opt-in rendered progress reporter.

    Description
    -----------
    Implements the three normative hooks (``on_run_start``,
    ``on_chunk``, ``on_run_end``) plus the optional additive
    ``on_leg_start``, which is where the leg's step count — and
    hence the bar's ``total`` — becomes known (``on_run_start``
    always fires with ``n_steps=None``, because the run target is
    planned after ``__enter__`` so a snapshot resume can re-plan).

    **One bar per ``advance()`` leg.** A second ``on_leg_start``
    closes the open bar and opens a fresh one. When ``on_chunk``
    arrives with no bar open (a reporter driven without the leg
    hook) the bar is created lazily from
    ``ChunkStats.leg_steps_total``, which may be ``None`` — then the
    bar is countless.

    **Compile-time skew.** The first chunk of a leg includes the jit
    compilation, so its rate is wildly off (measured: 83 steps/s
    against 27000 steady). Its postfix is marked accordingly, and
    the bar's timing basis is reset right after it so
    ``[elapsed<remaining]`` measures the steady state only.

    Parameters
    ----------
    mode : {"auto", "notebook", "tty", "log", "off"}, optional
        The rendering mode, resolved at ``on_run_start`` (the
        environment must be read at run time, not at construction):

        - ``"auto"`` — ``"notebook"`` inside an ipykernel,
          ``"tty"`` when stderr is a terminal, else ``"log"``.
        - ``"notebook"`` — the ``tqdm.notebook`` widget bar; without
          ipywidgets it falls back to a text bar on **stdout**
          (tqdm's default stderr renders as a red error block in
          Jupyter).
        - ``"tty"`` — the standard ``tqdm`` bar on stderr.
        - ``"log"`` — no tqdm: one plain line per chunk at level 25
          (NOTICE), for batch/SLURM logs.
        - ``"off"`` — render nothing.

        Any mode resolves to ``"off"`` on process ranks other than 0
        (default: "auto").
    """

    def __init__(
        self,
        *,
        mode: Literal["auto", "notebook", "tty", "log", "off"] = "auto",
    ) -> None:
        """Validate the mode; the environment is read at run start."""
        if mode not in _MODES:
            raise ValueError(
                f"mode= must be one of {_MODES}; got {mode!r}")
        self._mode: str = mode
        self._resolved: str | None = None
        self._bar: Any = None
        self._leg_open: bool = False
        self._first_chunk: bool = True
        self._steps: int = 0
        self._total: int | None = None
        self._fallback_total: int | None = None
        self._log_prints: bool = False

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def mode(self) -> str:
        """The requested mode (``"auto"`` until resolved)."""
        return self._mode

    @property
    def resolved_mode(self) -> str | None:
        """The mode resolved at ``on_run_start`` (None before it)."""
        return self._resolved

    # ------------------------------------------------------------
    #  The normative hooks
    # ------------------------------------------------------------
    def on_run_start(
        self, *, models: Any, n_steps: int | None,
    ) -> None:
        """Resolve the mode against the live environment.

        Parameters
        ----------
        models : Any
            The session's models (used only for a debug line).
        n_steps : int or None
            Always ``None`` from ``Session``; kept as the fallback
            bar total for drivers that do know it.
        """
        self._close_bar()
        self._resolved = self._resolve_mode()
        self._fallback_total = n_steps
        self._steps = 0
        self._total = None
        self._first_chunk = True
        # level 25 renders only because fridom.framework.logger
        # configured the `fridom` logger — a dependency that
        # disappears at cutover. If the record would be dropped,
        # print instead so a batch log never goes silent.
        self._log_prints = (self._resolved == "log"
                            and not _log.isEnabledFor(_NOTICE))
        _log.debug("progress bar: mode=%s, %d model(s), n_steps=%s",
                   self._resolved, len(models), n_steps)

    def on_leg_start(self, *, plan: Mapping[str, int]) -> None:
        """Open a fresh bar for one ``advance()`` leg (0% frame).

        Parameters
        ----------
        plan : mapping of str to int
            The leg's per-model step counts; their sum is the bar's
            ``total``.
        """
        if self._resolved in (None, "off"):
            return
        total = sum(int(steps) for steps in plan.values()) if plan \
            else None
        self._open_leg(total)

    def on_chunk(self, stats: ChunkStats) -> None:
        """Advance the bar by one chunk-boundary observation.

        Parameters
        ----------
        stats : ChunkStats
            The chunk payload; ``leg_steps_done`` /
            ``leg_steps_total`` drive the position and the total when
            present.
        """
        if self._resolved in (None, "off"):
            return
        if not self._leg_open:
            # driven without the leg hook: create the bar lazily
            total = stats.leg_steps_total
            if total is None:
                total = self._fallback_total
            self._open_leg(total)
        if stats.leg_steps_done is not None:
            self._steps = int(stats.leg_steps_done)
        else:
            self._steps += int(stats.steps_done)
        postfix = self._postfix(stats)
        if self._resolved == "log":
            self._emit_log(stats, postfix)
        else:
            self._emit_bar(postfix)
        if self._first_chunk:
            self._reset_timing()
            self._first_chunk = False

    def on_run_end(self, results: Any) -> None:
        """Close the bar on every exit path (panic included).

        Parameters
        ----------
        results : Any
            The aggregated per-model results (unused; the bar is a
            live renderer, not a summary).
        """
        _log.debug("progress bar: run end (%s)", type(results).__name__)
        try:
            self._close_bar()
        except Exception:  # noqa: BLE001 — teardown must not mask
            _log.exception("closing the progress bar failed")

    # ------------------------------------------------------------
    #  Mode resolution
    # ------------------------------------------------------------
    def _resolve_mode(self) -> str:
        """Resolve ``mode=`` against the live environment."""
        # rank guard first: only process 0 renders (the old bar never
        # finished this — progress_bar.py's open TODO)
        if jax.process_index() != 0:
            return "off"
        if self._mode != "auto":
            return self._mode
        if "ipykernel.zmqshell" in sys.modules:
            return "notebook"
        # NOT fr.utils.stdout_is_file(): it calls sys.stdout.fileno()
        # before its IPython check and misreports on detached streams
        try:
            is_tty = os.isatty(sys.stderr.fileno())
        except (OSError, ValueError):
            return "log"
        return "tty" if is_tty else "log"

    # ------------------------------------------------------------
    #  Bar lifecycle
    # ------------------------------------------------------------
    def _open_leg(self, total: int | None) -> None:
        """Close any open bar and start a fresh leg at 0%."""
        self._close_bar()
        self._leg_open = True
        self._first_chunk = True
        self._steps = 0
        self._total = total
        if self._resolved != "log":
            self._bar = self._make_bar(total)

    def _make_bar(self, total: int | None) -> Any:
        """Build the tqdm bar for the resolved rendering mode."""
        if self._resolved == "notebook":
            return self._make_notebook_bar(total)
        return tqdm(total=total, bar_format=_BAR_FORMAT, unit="it",
                    file=sys.stderr)

    def _make_notebook_bar(self, total: int | None) -> Any:
        """Build the widget bar, or a DELIBERATE text bar on stdout."""
        import tqdm.notebook as tqdm_nb  # noqa: PLC0415 — notebook-only cost

        if tqdm_nb.IProgress is not None:
            try:
                return tqdm_nb.tqdm(
                    total=total, bar_format=_BAR_FORMAT, unit="it")
            except ImportError:
                _log.debug("ipywidgets unusable; text bar on stdout")
        # the widget is unavailable: a text bar on STDOUT, not tqdm's
        # default stderr — Jupyter renders stderr as a red error block
        return tqdm(total=total, bar_format=_BAR_FORMAT, unit="it",
                    file=sys.stdout)

    def _close_bar(self) -> None:
        """Close the open bar, if any; idempotent."""
        bar = self._bar
        self._bar = None
        self._leg_open = False
        if bar is not None:
            bar.close()

    # ------------------------------------------------------------
    #  Rendering
    # ------------------------------------------------------------
    @staticmethod
    def _milliseconds_per_step(stats: ChunkStats) -> float:
        """Return the chunk's per-step cost in milliseconds."""
        if stats.steps_done > 0 and stats.wall_seconds > 0.0:
            return 1000.0 * stats.wall_seconds / stats.steps_done
        if stats.steps_per_second > 0.0:
            return 1000.0 / stats.steps_per_second
        return 0.0

    def _postfix(self, stats: ChunkStats) -> str:
        """Build the old bar's postfix (+ the chunk-1 compile mark)."""
        milliseconds = self._milliseconds_per_step(stats)
        time_str = humanize_number(float(stats.time), unit="seconds")
        postfix = (f"{int(milliseconds)} ms/it  at It: "
                   f"{stats.iteration} - Time: {time_str}")
        if self._first_chunk:
            postfix += _COMPILE_MARK
        return postfix

    def _emit_bar(self, postfix: str) -> None:
        """Move the tqdm bar and refresh it with the new postfix."""
        bar = self._bar
        if bar is None:  # pragma: no cover — _open_leg always builds one
            return
        total = bar.total
        bar.n = min(self._steps, total) if total else self._steps
        bar.set_postfix_str(postfix)

    def _emit_log(self, stats: ChunkStats, postfix: str) -> None:
        """One plain line per chunk (no glyphs, no carriage returns)."""
        time_str = humanize_number(float(stats.time), unit="seconds")
        head = ""
        if self._total:
            head = f"{100.0 * self._steps / self._total:.1f}%|  "
        line = (f"{head}it {stats.iteration}  t={time_str}  "
                f"{stats.steps_per_second:.0f} steps/s")
        if postfix.endswith(_COMPILE_MARK):
            line += _COMPILE_MARK
        if self._log_prints:
            print(line, flush=True)  # noqa: T201 — the log-mode fallback
        else:
            _log.log(_NOTICE, "%s", line)

    def _reset_timing(self) -> None:
        """Rebase the bar's clock so the ETA excludes compile time.

        tqdm computes ``rate = (n - initial) / elapsed`` with
        ``elapsed = now - start_t``; rebasing both after the first
        chunk makes ``[elapsed<remaining]`` a steady-state estimate.
        """
        bar = self._bar
        if bar is None:
            return
        now = time.time()
        try:
            bar.start_t = now
            bar.initial = bar.n
            bar.last_print_t = now
            bar.last_print_n = bar.n
        except AttributeError:  # pragma: no cover — tqdm always has them
            _log.debug("could not rebase the progress bar's clock")

    def __repr__(self) -> str:
        """Compact host-side summary."""
        return (f"<ProgressBar mode={self._mode!r} "
                f"resolved={self._resolved!r}>")
