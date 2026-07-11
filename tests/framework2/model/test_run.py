"""Tests for Model.run (the single-model Session-loop sugar).

Covers the run-target planning (exactly one of steps/runlen/end_time;
the sign-agnostic reduction and the (end-t0)*dt>0 precondition), the
RunResult aggregation, the NaN handling (raise_on_nan converts vs
raises), the graceful Ctrl-C path, walltime, and THE FACADE LAW:
run(n) is bitwise-identical to a hand-written Session loop with the
same chunk plan and adds ZERO jit-cache entries; and adding an output
stream never recompiles the physics.
"""
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.io import snapshots as snap
from fridom.framework2.io import triggers
from fridom.framework2.model.declarations import FieldDeclaration
from fridom.framework2.model.model import Model, chunk_cache_size
from fridom.framework2.model.module import Module
from fridom.framework2.model.parameters import ParameterDeclaration
from fridom.framework2.model.results import (
    PanicError,
    RunResult,
    RunStatus,
    RunTargetError,
)
from fridom.framework2.model.space_patterns import Collocated
from fridom.framework2.model.terms import term
from fridom.framework2.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.framework2.ops import session as session_mod
from fridom.framework2.ops.protocols import WalltimeGuard
from fridom.framework2.ops.session import Session

DT = 0.5
N = 8


# ================================================================
#  Toy model: du/dt = gain
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
    state = model._carry.state
    model._carry = model._carry.replace(
        state=state.replace(u=state["u"].with_data(jnp.full(N, jnp.nan))))


class FakeStream:

    """A minimal OutputStream double."""

    def __init__(self, trigger):
        self.trigger = trigger
        self.path = None
        self.writes = []
        self.closed = False

    def bind(self, model):
        pass

    def write(self, model_state):
        self.writes.append(int(model_state.clock.it))

    def truncate_after(self, iteration):
        pass

    def close(self):
        self.closed = True


# ================================================================
#  Basic run(steps)
# ================================================================
def test_run_steps_returns_a_completed_runresult():
    model = make_model(gain=2.0)
    result = model.run(steps=4, progress=False)
    assert isinstance(result, RunResult)
    assert result.status is RunStatus.COMPLETED
    assert result.steps_done == 4
    assert result.final_it == 4
    assert result.final_time == pytest.approx(4 * DT)
    # AB1 constant tendency: u = n * dt * gain, exactly
    assert float(model.state["u"].data[0]) == pytest.approx(4 * DT * 2.0)


def test_run_result_reports_positive_throughput():
    result = make_model().run(steps=4, progress=False)
    assert result.run_seconds >= 0.0
    assert result.compile_seconds >= 0.0
    assert result.steps_per_second >= 0.0


def test_repeated_run_continues_from_the_carry():
    model = make_model()
    model.run(steps=2, progress=False)
    result = model.run(steps=2, progress=False)
    assert result.final_it == 4
    assert float(model.state["u"].data[0]) == pytest.approx(4 * DT)


# ================================================================
#  Run-target planning
# ================================================================
def test_run_needs_exactly_one_target():
    model = make_model()
    with pytest.raises(RunTargetError, match="exactly one"):
        model.run(progress=False)
    with pytest.raises(RunTargetError, match="exactly one"):
        model.run(steps=4, runlen=2.0, progress=False)


def test_runlen_reduces_to_steps_by_overshoot():
    model = make_model()
    # runlen 2.1 s at dt 0.5 -> ceil(4.2) = 5 steps (overshoot)
    result = model.run(runlen=2.1, progress=False)
    assert result.steps_done == 5


def test_end_time_reduces_against_the_absolute_target():
    model = make_model()
    result = model.run(end_time=2.0, progress=False)  # 2.0 / 0.5 = 4
    assert result.steps_done == 4
    assert result.final_time == pytest.approx(2.0)


def test_end_time_wrong_direction_raises():
    model = make_model()  # dt > 0
    with pytest.raises(RunTargetError, match="precondition"):
        model.run(end_time=-1.0, progress=False)


def test_backward_leg_runs_when_dt_is_flipped():
    model = make_model()
    model.update_parameters({"stepper.dt": -DT})
    result = model.run(end_time=-2.0, progress=False)
    assert result.steps_done == 4
    assert result.final_time == pytest.approx(-2.0)


def test_run_steps_rejects_a_negative_count():
    with pytest.raises(RunTargetError, match="non-negative"):
        make_model().run(steps=-1, progress=False)


# ================================================================
#  NaN handling
# ================================================================
def test_raise_on_nan_false_returns_nan_abort():
    model = make_model(chunk_size=4)
    poison(model)
    result = model.run(steps=8, progress=False)  # default raise_on_nan
    assert result.status is RunStatus.NAN_ABORT
    assert result.steps_done == 4                # the aborted chunk


def test_raise_on_nan_true_reraises():
    model = make_model(chunk_size=4)
    poison(model)
    with pytest.raises(PanicError) as err:
        model.run(steps=8, progress=False, raise_on_nan=True)
    assert err.value.first_bad_it == 1


# ================================================================
#  Outputs bind through run()
# ================================================================
def test_run_binds_and_fires_outputs():
    stream = FakeStream(triggers.every(steps=2))
    make_model().run(steps=6, outputs=(stream,), progress=False)
    assert stream.writes == [0, 2, 4, 6]
    assert stream.closed


# ================================================================
#  THE FACADE LAW
# ================================================================
def test_run_equals_a_handwritten_session_loop_bitwise():
    grid = make_grid()
    via_run = make_model(grid=grid, chunk_size=4)
    via_loop = make_model(grid=grid, chunk_size=4)
    via_run.run(steps=11, progress=False)
    with Session(via_loop, progress=False) as s:
        s.advance({via_loop: 11})
    assert np.array_equal(
        np.asarray(via_run.state["u"].data),
        np.asarray(via_loop.state["u"].data))
    assert float(via_run.clock.elapsed) == float(via_loop.clock.elapsed)


def test_second_run_path_adds_zero_compiles(compile_counter):
    grid = make_grid()
    warm = make_model(grid=grid, chunk_size=4)
    warm.run(steps=11, progress=False)           # warm every length
    compile_counter.reset()
    other = make_model(grid=grid, chunk_size=4)  # identical re-assembly
    other.run(steps=11, progress=False)
    assert compile_counter.count == 0


def test_adding_a_stream_never_recompiles_physics(compile_counter):
    grid = make_grid()
    warm = make_model(grid=grid, chunk_size=4)
    warm.run(steps=8, progress=False)            # boundaries [4, 8]
    reference = chunk_cache_size()
    compile_counter.reset()
    # a stream firing only at step 0 (every(steps=100) > n_steps) keeps
    # the chunk boundaries [4, 8] identical -> no new chunk program
    stream = FakeStream(triggers.every(steps=100))
    other = make_model(grid=grid, chunk_size=4)
    other.run(steps=8, outputs=(stream,), progress=False)
    assert stream.writes == [0]                  # step-0 initial output
    assert chunk_cache_size() == reference
    assert compile_counter.count == 0


# ================================================================
#  Graceful Ctrl-C through run()
# ================================================================
def test_ctrl_c_returns_interrupted_no_exception_escapes():
    class Interrupting:
        def __init__(self):
            self.n = 0

        def on_run_start(self, *, models, n_steps):
            pass

        def on_chunk(self, stats):  # noqa: ARG002 — protocol hook
            self.n += 1
            if self.n == 3:
                raise KeyboardInterrupt

        def on_run_end(self, results):
            pass

    model = make_model(chunk_size=1)
    result = model.run(steps=100, progress=Interrupting())
    assert result.status is RunStatus.INTERRUPTED
    assert result.steps_done == 3                # zero steps lost
    assert int(model.clock.it) == 3


# ================================================================
#  Walltime through run()
# ================================================================
class FakeMonotonic:

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_run_walltime_stops_and_reports_walltime(tmp_path, monkeypatch):
    clock = FakeMonotonic()
    monkeypatch.setattr(
        session_mod, "WalltimeGuard",
        lambda budget: WalltimeGuard(budget, clock=clock))

    class Pusher:
        def on_run_start(self, *, models, n_steps):
            pass

        def on_chunk(self, stats):  # noqa: ARG002 — protocol hook
            clock.now = 999.0

        def on_run_end(self, results):
            pass

    cfg = snap.Snapshots(tmp_path / "snaps",
                         trigger=triggers.every(walltime=10.0))
    model = make_model(chunk_size=1)
    result = model.run(steps=100, snapshots=cfg, progress=Pusher())
    assert result.status is RunStatus.WALLTIME
    assert result.steps_done < 100
