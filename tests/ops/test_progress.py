"""Tests for fridom.ops.progress (fr.ops.ProgressBar).

Covers the opt-in rendered reporter: mode resolution against the live
environment (rank guard, ipykernel, tty, detached streams), the three
rendering modes (tty/notebook/log), the log-mode NOTICE line and its
print fallback, the leg lifecycle (0% frame, one bar per advance()
leg, lazy creation without the leg hook), the compile-skew timing
rebase, and the format parity with the old ProgressBar module.
"""
import io
import logging
import sys
from functools import partial

import jax.numpy as jnp
import pytest
import tqdm.notebook as tqdm_nb

from fridom.framework.utils import dtype_real, humanize_number, jaxify
from fridom.model.declarations import FieldDeclaration
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.terms import term
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.ops import progress as progress_mod
from fridom.ops.progress import ProgressBar
from fridom.ops.protocols import ChunkStats
from fridom.ops.session import Session
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

DT = 0.5
N = 8
LOGGER_NAME = "fridom.ops.progress"


# ================================================================
#  Toy model: du/dt = gain (constant forcing)
# ================================================================
@partial(jaxify, dynamic=("gain",))
class GainForcing(Module):

    """One PROGNOSTIC field forced by a provided parameter."""

    def __init__(self, gain=1.0):
        self.gain = jnp.asarray(gain, dtype=dtype_real())

    extra_halo = HaloSpec({"x": 0})

    field_declarations = (
        FieldDeclaration("u", space=Collocated(), long_name="Forced"),)
    parameter_declarations = (
        ParameterDeclaration("toy.gain", attr="gain", units="1"),)

    @term(advances=("u",))
    def force(self, state, ctx):
        gain = ctx.params.get("toy.gain", 0.0)
        u = state["u"]
        return {"u": u.with_data(
            jnp.broadcast_to(gain, u.data.shape).astype(u.data.dtype))}


def make_model(gain=1.0, dt=DT, **kwargs):
    grid = Grid((IntervalMesh(N, (0.0, 1.0), periodic=True, name="x"),))
    return Model(grid=grid, modules=(GainForcing(gain),),
                 time_stepper=AdamBashforth(dt, order=1), **kwargs)


# ================================================================
#  Small helpers
# ================================================================
def make_stats(*, it=1, t=0.5, steps=1, wall=0.01, done=None,
               total=None):
    """Build one ChunkStats payload."""
    return ChunkStats(
        name="model", iteration=it, time=t, steps_done=steps,
        wall_seconds=wall,
        steps_per_second=(steps / wall if wall > 0.0 else 0.0),
        leg_steps_done=done, leg_steps_total=total)


@pytest.fixture
def streams(monkeypatch):
    """Return a redirector: call it INSIDE the test body.

    pytest's global capture reinstalls ``sys.stdout``/``sys.stderr``
    between the setup and call phases, so a patch applied during
    fixture setup would be undone before the test runs.
    """
    def redirect():
        out, err = io.StringIO(), io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        return out, err

    return redirect


def started(bar, n_steps=None):
    """Fire on_run_start with a one-model session."""
    bar.on_run_start(models={"model": object()}, n_steps=n_steps)
    return bar


# ================================================================
#  Construction
# ================================================================
def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode="):
        ProgressBar(mode="fancy")


def test_mode_is_not_resolved_before_the_run_starts():
    bar = ProgressBar()
    assert bar.mode == "auto"
    assert bar.resolved_mode is None
    assert "auto" in repr(bar)


# ================================================================
#  Mode resolution (at on_run_start, against the live environment)
# ================================================================
def test_rank_guard_forces_off(monkeypatch, streams):
    out, err = streams()
    monkeypatch.setattr(progress_mod.jax, "process_index", lambda: 1)
    bar = started(ProgressBar(mode="tty"))
    assert bar.resolved_mode == "off"
    bar.on_leg_start(plan={"model": 4})
    bar.on_chunk(make_stats(done=2, total=4))
    bar.on_run_end({})
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_explicit_mode_is_taken_as_given(monkeypatch):
    monkeypatch.setattr(progress_mod.jax, "process_index", lambda: 0)
    assert started(ProgressBar(mode="log")).resolved_mode == "log"
    assert started(ProgressBar(mode="off")).resolved_mode == "off"


def test_auto_picks_notebook_inside_an_ipykernel(monkeypatch):
    monkeypatch.setitem(sys.modules, "ipykernel.zmqshell", object())
    assert started(ProgressBar()).resolved_mode == "notebook"


def test_auto_picks_tty_when_stderr_is_a_terminal(monkeypatch):
    monkeypatch.delitem(sys.modules, "ipykernel.zmqshell", raising=False)
    monkeypatch.setattr(progress_mod.os, "isatty", lambda fd: True)  # noqa: ARG005
    assert started(ProgressBar()).resolved_mode == "tty"


def test_auto_picks_log_when_stderr_is_redirected(monkeypatch):
    monkeypatch.delitem(sys.modules, "ipykernel.zmqshell", raising=False)
    monkeypatch.setattr(progress_mod.os, "isatty", lambda fd: False)  # noqa: ARG005
    assert started(ProgressBar()).resolved_mode == "log"


def test_auto_picks_log_for_a_detached_stream(monkeypatch):
    monkeypatch.delitem(sys.modules, "ipykernel.zmqshell", raising=False)

    class Detached(io.StringIO):
        def fileno(self):
            raise ValueError("I/O operation on detached stream")

    monkeypatch.setattr(sys, "stderr", Detached())
    assert started(ProgressBar()).resolved_mode == "log"


# ================================================================
#  off mode / hooks before on_run_start
# ================================================================
def test_off_mode_renders_nothing(streams):
    out, err = streams()
    bar = started(ProgressBar(mode="off"))
    bar.on_leg_start(plan={"model": 4})
    bar.on_chunk(make_stats(done=4, total=4))
    bar.on_run_end({})
    assert bar._bar is None
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_hooks_before_run_start_are_inert(streams):
    _out, err = streams()
    bar = ProgressBar(mode="tty")
    bar.on_leg_start(plan={"model": 4})
    bar.on_chunk(make_stats(done=4, total=4))
    assert bar._bar is None
    assert err.getvalue() == ""


# ================================================================
#  tty mode: the leg lifecycle and format parity
# ================================================================
def test_leg_start_draws_the_zero_percent_frame(streams):
    _out, err = streams()
    bar = started(ProgressBar(mode="tty"))
    assert err.getvalue() == ""
    bar.on_leg_start(plan={"model": 10})
    assert "0.00%" in err.getvalue()
    assert bar._bar.total == 10
    bar.on_run_end({})


def test_empty_plan_gives_a_countless_bar(streams):
    streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={})
    assert bar._bar.total is None
    bar.on_run_end({})


def test_postfix_matches_the_old_bar_format(streams):
    _out, err = streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={"model": 40})
    bar.on_chunk(make_stats(it=10, t=5.0, steps=10, wall=0.25,
                            done=10, total=40))
    bar.on_chunk(make_stats(it=20, t=10.0, steps=10, wall=0.25,
                            done=20, total=40))
    text = err.getvalue()
    time_str = humanize_number(10.0, unit="seconds")
    assert f"25 ms/it  at It: 20 - Time: {time_str}" in text
    # the first chunk pays the compile and says so; the second does not
    assert "at It: 10" in text
    assert text.count("(first chunk: incl. compile)") == 1
    assert "50.00%" in text
    bar.on_run_end({})


def test_rate_falls_back_to_steps_per_second(streams):
    _out, err = streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={"model": 10})
    bar.on_chunk(ChunkStats(
        name="model", iteration=5, time=2.5, steps_done=5,
        wall_seconds=0.0, steps_per_second=250.0,
        leg_steps_done=5, leg_steps_total=10))
    assert "4 ms/it" in err.getvalue()
    bar.on_run_end({})


def test_rate_is_zero_without_any_timing(streams):
    _out, err = streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={"model": 10})
    bar.on_chunk(ChunkStats(
        name="model", iteration=5, time=2.5, steps_done=0,
        wall_seconds=0.0, steps_per_second=0.0,
        leg_steps_done=5, leg_steps_total=10))
    assert "0 ms/it" in err.getvalue()
    bar.on_run_end({})


def test_the_bar_never_overruns_its_total(streams):
    streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={"model": 4})
    bar.on_chunk(make_stats(it=9, t=4.5, steps=9, wall=0.1,
                            done=9, total=4))
    assert bar._bar.n == 4
    bar.on_run_end({})


# ================================================================
#  Compile-skew: the timing rebase after the first chunk
# ================================================================
def test_first_chunk_rebases_the_bar_clock(streams):
    streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={"model": 20})
    inner = bar._bar
    inner.start_t = 0.0  # a clearly stale (epoch) basis
    bar.on_chunk(make_stats(it=10, t=5.0, steps=10, wall=3.0,
                            done=10, total=20))
    assert inner.initial == 10  # rate measures from here on
    assert inner.start_t > 1.0e9  # rebased to time.time()
    rebased = inner.start_t
    bar.on_chunk(make_stats(it=20, t=10.0, steps=10, wall=0.1,
                            done=20, total=20))
    assert inner.start_t == rebased  # only the FIRST chunk rebases
    assert inner.initial == 10
    bar.on_run_end({})


def test_the_rebase_is_a_no_op_in_log_mode(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    bar = started(ProgressBar(mode="log"))
    bar.on_leg_start(plan={"model": 4})
    bar.on_chunk(make_stats(done=4, total=4))  # no bar to rebase
    bar.on_run_end({})
    assert bar._bar is None


# ================================================================
#  Lazy bar creation (a reporter driven without the leg hook)
# ================================================================
def test_chunk_without_a_leg_hook_creates_the_bar_lazily(streams):
    _out, err = streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_chunk(make_stats(it=5, t=2.5, steps=5, wall=0.1,
                            done=5, total=10))
    assert bar._bar.total == 10
    assert "50.00%" in err.getvalue()
    bar.on_run_end({})


def test_lazy_bar_falls_back_to_the_run_start_step_count(streams):
    streams()
    bar = started(ProgressBar(mode="tty"), n_steps=12)
    bar.on_chunk(make_stats(it=6, t=3.0, steps=6, wall=0.1))
    assert bar._bar.total == 12
    assert bar._bar.n == 6  # no leg_steps_done: accumulated locally
    bar.on_chunk(make_stats(it=12, t=6.0, steps=6, wall=0.1))
    assert bar._bar.n == 12
    bar.on_run_end({})


def test_lazy_bar_is_countless_when_nothing_is_known(streams):
    streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_chunk(make_stats(it=5, t=2.5, steps=5, wall=0.1))
    assert bar._bar.total is None
    assert bar._bar.n == 5
    bar.on_run_end({})


# ================================================================
#  notebook mode
# ================================================================
def test_notebook_mode_uses_the_widget_bar(monkeypatch, streams):
    made = []

    class FakeWidgetBar:
        def __init__(self, **kwargs):
            self.total = kwargs.get("total")
            self.n = 0
            self.initial = 0
            self.start_t = 0.0
            self.last_print_t = 0.0
            self.last_print_n = 0
            self.closed = False
            made.append(self)

        def set_postfix_str(self, text):
            self.postfix = text

        def close(self):
            self.closed = True

    out, err = streams()
    monkeypatch.setattr(tqdm_nb, "IProgress", object())
    monkeypatch.setattr(tqdm_nb, "tqdm", FakeWidgetBar)
    bar = started(ProgressBar(mode="notebook"))
    bar.on_leg_start(plan={"model": 4})
    bar.on_chunk(make_stats(it=4, t=2.0, steps=4, wall=0.1,
                            done=4, total=4))
    bar.on_run_end({})
    assert len(made) == 1
    assert made[0].total == 4
    assert made[0].n == 4
    assert made[0].closed
    assert out.getvalue() == ""  # the widget writes nothing to stdout
    assert err.getvalue() == ""


def test_notebook_falls_back_to_text_on_stdout(monkeypatch, streams):
    monkeypatch.setattr(tqdm_nb, "IProgress", None)  # no ipywidgets
    out, err = streams()
    bar = started(ProgressBar(mode="notebook"))
    bar.on_leg_start(plan={"model": 4})
    bar.on_chunk(make_stats(it=4, t=2.0, steps=4, wall=0.1,
                            done=4, total=4))
    bar.on_run_end({})
    # DELIBERATE: stdout, not tqdm's default stderr (which Jupyter
    # renders as a red error block)
    assert "100.00%" in out.getvalue()
    assert err.getvalue() == ""


def test_notebook_falls_back_when_the_widget_import_fails(
        monkeypatch, streams):
    def boom(**_kwargs):
        raise ImportError("IProgress not found")

    monkeypatch.setattr(tqdm_nb, "IProgress", object())
    monkeypatch.setattr(tqdm_nb, "tqdm", boom)
    out, err = streams()
    bar = started(ProgressBar(mode="notebook"))
    bar.on_leg_start(plan={"model": 4})
    bar.on_run_end({})
    assert "0.00%" in out.getvalue()
    assert err.getvalue() == ""


# ================================================================
#  log mode
# ================================================================
def test_log_mode_logs_one_notice_line_per_chunk(caplog):
    caplog.set_level(25, logger=LOGGER_NAME)
    model = make_model(chunk_size=2)
    model.run(steps=6, progress=ProgressBar(mode="log"), max_chunk=2)
    lines = [r.getMessage() for r in caplog.records
             if r.name == LOGGER_NAME and r.levelno == 25]
    assert len(lines) == 3
    assert lines[0].startswith("33.3%|  it 2  ")
    assert lines[0].endswith("(first chunk: incl. compile)")
    assert "steps/s" in lines[0]
    assert lines[-1].startswith("100.0%|  it 6  ")
    # no tqdm glyphs, no carriage returns
    assert not any("\r" in ln or "█" in ln for ln in lines)


def test_log_mode_prints_when_the_notice_level_is_disabled(
        caplog, capsys):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    model = make_model(chunk_size=2)
    model.run(steps=4, progress=ProgressBar(mode="log"), max_chunk=2)
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "steps/s" in ln]
    assert len(lines) == 2
    assert lines[-1].startswith("100.0%|  it 4  ")
    assert not any(r.name == LOGGER_NAME and r.levelno == 25
                   for r in caplog.records)


def test_log_mode_omits_the_percentage_without_a_total(caplog):
    caplog.set_level(25, logger=LOGGER_NAME)
    bar = started(ProgressBar(mode="log"))
    bar.on_chunk(make_stats(it=3, t=1.5, steps=3, wall=0.1))
    bar.on_run_end({})
    lines = [r.getMessage() for r in caplog.records
             if r.name == LOGGER_NAME and r.levelno == 25]
    assert len(lines) == 1
    assert lines[0].startswith("it 3  ")


# ================================================================
#  Percentage and total derived from the run target (no user total)
# ================================================================
@pytest.mark.parametrize(
    "target",
    [pytest.param({"steps": 6}, id="steps"),
     pytest.param({"runlen": 3.0}, id="runlen"),
     pytest.param({"end_time": 3.0}, id="end_time")])
def test_run_targets_fill_the_bar_to_a_hundred_percent(streams, target):
    _out, err = streams()
    bar = ProgressBar(mode="tty")
    model = make_model(chunk_size=2)
    result = model.run(progress=bar, max_chunk=2, **target)
    assert result.steps_done == 6
    text = err.getvalue()
    assert "0.00%" in text  # the leg-start frame
    assert "100.00%" in text
    assert bar._total == 6


# ================================================================
#  Multi-leg: one bar per advance() leg
# ================================================================
def test_two_legs_produce_two_bars(monkeypatch, streams):
    streams()
    made = []
    real_tqdm = progress_mod.tqdm

    def spy(**kwargs):
        made.append(real_tqdm(**kwargs))
        return made[-1]

    monkeypatch.setattr(progress_mod, "tqdm", spy)
    bar = ProgressBar(mode="tty")
    with Session(make_model(chunk_size=2), progress=bar) as s:
        s.advance(model=4)
        assert len(made) == 1
        s.advance(model=6)
    assert len(made) == 2
    assert [b.total for b in made] == [4, 6]
    assert made[0].n == 4
    assert made[1].n == 6
    assert all(b.disable for b in made)  # tqdm.close() disables


def test_a_second_leg_closes_the_open_bar(streams):
    streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={"model": 4})
    first = bar._bar
    bar.on_leg_start(plan={"model": 8})
    assert first.disable  # closed
    assert bar._bar is not first
    assert bar._bar.total == 8
    bar.on_run_end({})


def test_run_start_closes_a_leftover_bar(streams):
    streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={"model": 4})
    leftover = bar._bar
    started(bar)  # a fresh run on the same reporter object
    assert leftover.disable
    assert bar._bar is None


# ================================================================
#  Teardown robustness
# ================================================================
def test_on_run_end_swallows_a_failing_close(caplog):
    class Exploding:
        def close(self):
            raise RuntimeError("no")

    bar = started(ProgressBar(mode="tty"))
    bar._bar = Exploding()
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        bar.on_run_end({})
    assert "closing the progress bar failed" in caplog.text


def test_on_run_end_is_idempotent(streams):
    streams()
    bar = started(ProgressBar(mode="tty"))
    bar.on_leg_start(plan={"model": 4})
    bar.on_run_end({})
    bar.on_run_end({})
    assert bar._bar is None
