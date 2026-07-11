"""Tests for fridom.model.ops.session (fr.ops.Session).

Covers the single-model iteration-1 surface: constructor keying and
rejections, the reentrancy/single-use guards, the __enter__ duties
(bind + dedupe + walltime rejection + resume), the normative boundary
sequence (writers fire in binding order at firing steps; step-0
initial output; snapshot firings write + rotate), the ``active``
predicate, the __exit__ guarantees on every exit path (flush+close,
no crash snapshot on panic, on_walltime only on the walltime path,
never suppresses), graceful first-Ctrl-C, and predictive walltime.
"""
from functools import partial

import jax.numpy as jnp
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.model.io import snapshots as snap
from fridom.model.io import triggers
from fridom.model.io.streams import IOCollisionError
from fridom.model.declarations import FieldDeclaration
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.results import PanicError, RunStatus
from fridom.model.space_patterns import Collocated
from fridom.model.terms import term
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.model.ops import session as session_mod
from fridom.model.ops.protocols import WalltimeGuard
from fridom.model.ops.session import Session

DT = 0.5
N = 8


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


def make_grid():
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))


def make_model(grid=None, gain=1.0, dt=DT, **kwargs):
    if grid is None:
        grid = make_grid()
    return Model(grid=grid, modules=(GainForcing(gain),),
                 time_stepper=AdamBashforth(dt, order=1), **kwargs)


def poison(model):
    """Overwrite the state with NaN (a panic on the next step)."""
    state = model._carry.state
    model._carry = model._carry.replace(
        state=state.replace(u=state["u"].with_data(jnp.full(N, jnp.nan))))


# ================================================================
#  Fake output streams (list-collecting; no real Writer imported)
# ================================================================
class FakeStream:

    """A minimal OutputStream double: records writes by iteration."""

    def __init__(self, trigger, path=None, log=None, label="s"):
        self.trigger = trigger
        self.path = path
        self.writes = []
        self.truncated = None
        self.bound = False
        self.closed = False
        self._log = log
        self._label = label

    def bind(self, model):  # noqa: ARG002
        self.bound = True

    def write(self, model_state):
        self.writes.append(int(model_state.clock.it))
        if self._log is not None:
            self._log.append(self._label)

    def truncate_after(self, iteration):
        self.truncated = iteration

    def close(self):
        self.closed = True


class RecordingReporter:

    """Captures the three normative reporter hooks."""

    def __init__(self):
        self.started = None
        self.chunks = []
        self.ended = None

    def on_run_start(self, *, models, n_steps):
        self.started = (tuple(models), n_steps)

    def on_chunk(self, stats):
        self.chunks.append(stats)

    def on_run_end(self, results):
        self.ended = results


# ================================================================
#  Constructor keying and rejections
# ================================================================
def test_lone_unnamed_model_gets_the_default_key():
    s = Session(make_model(), progress=False)
    assert set(s.models) == {"model"}


def test_named_models_are_keyed_by_name():
    s = Session(make_model(name="ocn"), progress=False)
    assert set(s.models) == {"ocn"}


def test_multi_model_needs_names():
    with pytest.raises(ValueError, match="name="):
        Session([make_model(), make_model()], progress=False)


def test_duplicate_model_names_raise():
    with pytest.raises(ValueError, match="share the name"):
        Session([make_model(name="m"), make_model(name="m")],
                progress=False)


def test_outputs_rejects_a_snapshots_config():
    cfg = snap.Snapshots("snaps", trigger=triggers.every(steps=2))
    with pytest.raises(IOCollisionError, match="run-config only"):
        Session(make_model(), outputs=(cfg,), progress=False)


def test_max_chunk_must_be_positive_int_or_none():
    with pytest.raises(ValueError, match="max_chunk"):
        Session(make_model(), max_chunk=0, progress=False)


def test_jit_false_is_not_wired():
    with pytest.raises(NotImplementedError, match="eager"):
        Session(make_model(), jit=False, progress=False)


# ================================================================
#  Reentrancy / single-use guards
# ================================================================
def test_advance_outside_with_raises():
    s = Session(make_model(), progress=False)
    with pytest.raises(RuntimeError, match="only valid inside"):
        s.advance(model=1)


def test_active_outside_with_raises():
    s = Session(make_model(), progress=False)
    with pytest.raises(RuntimeError, match="only valid inside"):
        _ = s.active


def test_result_outside_with_raises():
    s = Session(make_model(), progress=False)
    with pytest.raises(RuntimeError, match="only valid inside"):
        _ = s.result


def test_session_is_single_use():
    s = Session(make_model(), progress=False)
    with s:
        pass
    with pytest.raises(RuntimeError, match="single-use"):
        s.__enter__()


def test_entering_over_a_bound_model_raises():
    model = make_model()
    with Session(model, progress=False), \
            pytest.raises(RuntimeError, match="already bound"):
        Session(model, progress=False).__enter__()


def test_exit_releases_the_model_binding():
    model = make_model()
    with Session(model, progress=False):
        pass
    # a second session over the same model now enters fine
    with Session(model, progress=False) as s:
        assert s.active


# ================================================================
#  __enter__ duties: bind, dedupe, walltime rejection
# ================================================================
def test_enter_binds_every_output():
    stream = FakeStream(triggers.every(steps=2))
    with Session(make_model(), outputs=(stream,), progress=False):
        assert stream.bound


def test_distinct_streams_on_one_path_collide():
    a = FakeStream(triggers.every(steps=2), path="out.zarr")
    b = FakeStream(triggers.every(steps=2), path="out.zarr")
    with pytest.raises(IOCollisionError, match="distinct"):
        Session(make_model(), outputs=(a, b), progress=False).__enter__()


def test_same_stream_listed_twice_dedupes():
    a = FakeStream(triggers.every(steps=2), path="out.zarr")
    with Session(make_model(), outputs=(a, a), progress=False) as s:
        assert s.active


def test_walltime_trigger_rejected_on_a_data_stream():
    stream = FakeStream(triggers.every(walltime="1h"))
    with pytest.raises(ValueError, match="walltime"):
        Session(make_model(), outputs=(stream,),
                progress=False).__enter__()


# ================================================================
#  The boundary sequence
# ================================================================
def test_streams_fire_at_their_lowered_steps():
    stream = FakeStream(triggers.every(steps=2))
    with Session(make_model(), outputs=(stream,), progress=False) as s:
        s.advance(model=6)
    assert stream.writes == [0, 2, 4, 6]  # every(steps=2) incl. step 0


def test_streams_fire_in_binding_order():
    log = []
    a = FakeStream(triggers.every(steps=2), log=log, label="a")
    b = FakeStream(triggers.every(steps=2), log=log, label="b")
    with Session(make_model(), outputs=(a, b), progress=False) as s:
        s.advance(model=2)
    # step 0 then step 2, each time a before b
    assert log == ["a", "b", "a", "b"]


def test_at_trigger_fires_on_realized_steps():
    stream = FakeStream(triggers.at([1.0, 2.0]))  # times 1.0, 2.0 s
    with Session(make_model(), outputs=(stream,), progress=False) as s:
        s.advance(model=6)
    # dt=0.5: t=1.0 -> it 2, t=2.0 -> it 4
    assert stream.writes == [2, 4]


def test_progress_receives_one_chunkstats_per_chunk():
    reporter = RecordingReporter()
    with Session(make_model(chunk_size=2), progress=reporter) as s:
        s.advance(model=6)
    assert reporter.started[1] is None  # n_steps unknown at enter
    assert [c.steps_done for c in reporter.chunks] == [2, 2, 2]
    assert [c.iteration for c in reporter.chunks] == [2, 4, 6]
    assert reporter.ended is not None  # on_run_end fired at exit


def test_advance_returns_per_model_advance_results():
    with Session(make_model(), progress=False) as s:
        out = s.advance(model=4)
    assert set(out) == {"model"}
    assert out["model"].steps_done == 4


def test_advance_by_model_object_key():
    model = make_model()
    with Session(model, progress=False) as s:
        out = s.advance({model: 3})
    assert out["model"].steps_done == 3


# ================================================================
#  active predicate
# ================================================================
def test_active_true_until_target_is_the_callers_business():
    with Session(make_model(), progress=False) as s:
        assert s.active
        s.advance(model=4)
        assert s.active  # active does not track target exhaustion


def test_active_flips_after_a_panic():
    model = make_model(chunk_size=4)
    poison(model)
    with pytest.raises(PanicError), \
            Session(model, progress=False) as s:
        s.advance(model=4)
    assert model.panicked


# ================================================================
#  __exit__ guarantees
# ================================================================
def test_exit_closes_every_stream_on_normal_completion():
    stream = FakeStream(triggers.every(steps=2))
    with Session(make_model(), outputs=(stream,), progress=False) as s:
        s.advance(model=4)
    assert stream.closed


def test_exit_never_suppresses_an_in_flight_exception():
    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError), Session(make_model(), progress=False):
        raise BoomError


def test_panic_flushes_streams_but_writes_no_snapshot(tmp_path):
    stream = FakeStream(triggers.every(steps=2))
    # a snapshot trigger that does NOT fire at step 0 (t=2.0 -> it 4),
    # so the only way a snapshot could appear is a crash snapshot
    cfg = snap.Snapshots(tmp_path / "snaps",
                         trigger=triggers.at([2.0]))
    model = make_model(chunk_size=4)
    poison(model)
    with pytest.raises(PanicError), \
            Session(model, outputs=(stream,), snapshots=cfg,
                    progress=False) as s:
        s.advance(model=8)
    assert stream.closed                       # flushed on the panic path
    # NO automatic crash snapshot: the panic-boundary firing never runs
    # (advance raises first) and __exit__ writes nothing on a panic
    snaps = sorted((tmp_path / "snaps").glob("it*")) \
        if (tmp_path / "snaps").is_dir() else []
    assert snaps == []


# ================================================================
#  Snapshot firings write + rotate
# ================================================================
def test_snapshot_fires_and_rotates(tmp_path):
    cfg = snap.Snapshots(tmp_path / "snaps",
                         trigger=triggers.every(steps=2), keep=2)
    with Session(make_model(chunk_size=2), snapshots=cfg,
                 progress=False) as s:
        s.advance(model=6)
    committed = sorted(p.name for p in (tmp_path / "snaps").glob("it*"))
    assert len(committed) == 2                 # keep=2 rotated the rest
    assert committed == ["it000000000004", "it000000000006"]


# ================================================================
#  Graceful first Ctrl-C (hook-raised mid-loop)
# ================================================================
def test_first_ctrl_c_returns_gracefully_zero_loss():
    class Interrupting:
        def __init__(self):
            self.n = 0

        def on_run_start(self, *, models, n_steps):
            pass

        def on_chunk(self, stats):  # noqa: ARG002 — protocol hook
            self.n += 1
            if self.n == 2:
                raise KeyboardInterrupt

        def on_run_end(self, results):
            pass

    model = make_model(chunk_size=1)
    with Session(model, progress=Interrupting()) as s:
        out = s.advance(model=10)
        assert not s.active                    # interrupt flag flips it
        status = s.result["model"].status
    assert status is RunStatus.INTERRUPTED
    # zero steps lost: the in-flight chunk committed (2 chunks of 1)
    assert out["model"].steps_done == 2
    assert int(model.clock.it) == 2


# ================================================================
#  Predictive walltime (fake clock)
# ================================================================
class FakeMonotonic:

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_walltime_stops_predictively_and_fires_on_walltime(
        tmp_path, monkeypatch):
    clock = FakeMonotonic()
    monkeypatch.setattr(
        session_mod, "WalltimeGuard",
        lambda budget: WalltimeGuard(budget, clock=clock))
    fired = []
    cfg = snap.Snapshots(
        tmp_path / "snaps",
        trigger=triggers.every(walltime=10.0) | triggers.every(steps=2),
        on_walltime=lambda: fired.append(True))

    class Pusher:  # push the fake clock past the budget after a chunk
        def on_run_start(self, *, models, n_steps):
            pass

        def on_chunk(self, stats):  # noqa: ARG002 — protocol hook
            clock.now = 999.0

        def on_run_end(self, results):
            pass

    model = make_model(chunk_size=1)
    with Session(model, snapshots=cfg, progress=Pusher()) as s:
        s.advance(model=100)
        status = s.result["model"].status
    assert status is RunStatus.WALLTIME
    assert fired == [True]                      # on_walltime ran at exit
    assert model.panicked is False


def test_on_walltime_not_fired_on_normal_completion(tmp_path):
    fired = []
    cfg = snap.Snapshots(tmp_path / "snaps",
                         trigger=triggers.every(steps=2),
                         on_walltime=lambda: fired.append(True))
    with Session(make_model(chunk_size=2), snapshots=cfg,
                 progress=False) as s:
        s.advance(model=4)
    assert fired == []


# ================================================================
#  Snapshot resume
# ================================================================
def test_resume_loads_the_newest_snapshot_and_truncates(tmp_path):
    cfg = snap.Snapshots(tmp_path / "snaps",
                         trigger=triggers.every(steps=2))
    first = make_model(chunk_size=2)
    with Session(first, snapshots=cfg, progress=False) as s:
        s.advance(model=6)
    assert int(first.clock.it) == 6
    # a fresh model resumes to it=6
    stream = FakeStream(triggers.every(steps=2))
    second = make_model(chunk_size=2)
    resume_cfg = snap.Snapshots(tmp_path / "snaps",
                                trigger=triggers.every(steps=2),
                                resume=True)
    with Session(second, outputs=(stream,), snapshots=resume_cfg,
                 progress=False) as s:
        assert int(second.clock.it) == 6       # resumed before advancing
    assert stream.truncated == 6               # writers aligned on resume
