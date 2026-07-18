"""Tests for the async two-tier chunk compile (model/model.py).

Covers the opt-in ``Model(async_chunk_compile=True)`` path: on a chunk
cache miss whose natural unroll > 1, ``step_chunk`` serves a cheap
``force_unroll=1`` length-C executable through a ``_TwoTier`` holder
while the full-unroll executable compiles on a daemon thread, then
swaps the cache entry to the full executable at a later chunk boundary
(steady state is the single-tier baseline). Covers equivalence + swap,
the ``_TwoTier`` state machine with stubs (deterministic branch
coverage), background-error propagation through ``step_chunk``, the
sync fallbacks (n == 1, scan_unroll == 1), the knob surface, and
backward-compatible sync-path accounting.

Trap: two identically-configured models share the SAME chunk-cache
key, so an async model and its sync twin would hit each other's cache
entries. Every test runs under an autouse fixture that clears the two
shared cache dicts before and after it, and the equivalence test also
clears between the sync twin and the async run.
"""
import threading
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.model import model as mm
from fridom.model.declarations import FieldDeclaration
from fridom.model.model import (
    _CHUNK_COMPILE_LOG,
    _CHUNK_EXECUTABLES,
    Model,
    _chunk_key,
    _TwoTier,
    chunk_cache_size,
    step_chunk,
)
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.terms import term
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

DT = 0.5
N = 8


# ================================================================
#  Toy module: constant forcing du/dt = gain (duplicated builder)
# ================================================================
@partial(jaxify, dynamic=("gain",))
class GainForcing(Module):

    """One PROGNOSTIC field forced by a provided parameter.

    Duplicated from ``test_step_chunk`` (self-contained shards):
    the term reads the parameter through ``ctx.params`` and scales
    the raw data, declaring its (zero) halo demand.
    """

    def __init__(self, gain=1.0):
        self.gain = jnp.asarray(gain, dtype=dtype_real())

    extra_halo = HaloSpec({"x": 0})

    field_declarations = (
        FieldDeclaration("u", space=Collocated(),
                         long_name="Forced"),)
    parameter_declarations = (
        ParameterDeclaration("toy.gain", attr="gain", units="1"),)

    @term(advances=("u",))
    def force(self, state, ctx):
        gain = ctx.params.get("toy.gain", 0.0)
        u = state["u"]
        return {"u": u.with_data(
            jnp.broadcast_to(gain, u.data.shape)
            .astype(u.data.dtype))}


# ================================================================
#  Fixtures and helpers
# ================================================================
def make_grid():
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))


def make_model(grid=None, gain=1.0, dt=DT, order=2, **kwargs):
    """AB(order) model; order >= 2 gives scan_unroll > 1."""
    if grid is None:
        grid = make_grid()
    return Model(grid=grid, modules=(GainForcing(gain),),
                 time_stepper=AdamBashforth(dt, order=order), **kwargs)


def model_key(model, n):
    """Return the chunk-cache key for a fresh carry at length n."""
    return _chunk_key(model._artifacts.record, model._carry,
                     model._stepper, n)


def run_chunk(model, n, carry=None, **kwargs):
    """Run one chunk over the model's (or a given) carry."""
    if carry is None:
        carry = model._carry
    return step_chunk(model._artifacts.record, carry,
                     model._stepper, n, **kwargs)


@pytest.fixture(autouse=True)
def _clear_chunk_cache():
    """Isolate the shared chunk-cache dicts around every test."""
    _CHUNK_EXECUTABLES.clear()
    _CHUNK_COMPILE_LOG.clear()
    yield
    _CHUNK_EXECUTABLES.clear()
    _CHUNK_COMPILE_LOG.clear()


# ================================================================
#  Stubs for deterministic _TwoTier branch coverage
# ================================================================
class _BlockingLowered:

    """A fake ``Lowered`` whose ``compile()`` blocks on an event."""

    def __init__(self, event, result):
        self.event = event
        self.result = result

    def compile(self):
        self.event.wait()
        return self.result


class _BoomLowered:

    """A fake ``Lowered`` whose ``compile()`` raises."""

    def compile(self):
        raise RuntimeError("boom background compile")


class _FullStub:

    """A fake compiled executable with a memory-analysis report."""

    def memory_analysis(self):
        return "stub-memory"


def _fake_cheap(carry, stepper):
    """Echo the inputs, standing in for a cheap executable."""
    return ("cheap", carry, stepper)


# ================================================================
#  Equivalence + swap (integration through advance)
# ================================================================
def test_async_matches_sync_twin_and_swaps():
    grid = make_grid()
    # sync twin: two advances, snapshot the state after each
    sync = make_model(grid=grid, gain=0.7, order=2, chunk_size=4)
    sync.advance(4)
    after_first = np.asarray(sync.state["u"].data).copy()
    sync.advance(4)
    after_second = np.asarray(sync.state["u"].data).copy()
    # clear so the async run does not hit the twin's cache entry
    _CHUNK_EXECUTABLES.clear()
    _CHUNK_COMPILE_LOG.clear()

    model = make_model(grid=grid, gain=0.7, order=2, chunk_size=4,
                       async_chunk_compile=True)
    key = model_key(model, 4)
    model.advance(4)
    holder = _CHUNK_EXECUTABLES[key]
    assert type(holder) is _TwoTier
    # the miss logged the cheap tier's compile seconds
    assert isinstance(_CHUNK_COMPILE_LOG[key][0], float)
    np.testing.assert_array_equal(
        np.asarray(model.state["u"].data), after_first)

    holder.wait()
    full_log = holder.compile_log()
    assert full_log[0] is not None
    model.advance(4)
    # the swap replaced the holder with a plain executable and
    # overwrote the log with the full tier's (seconds, memory)
    assert type(_CHUNK_EXECUTABLES[key]) is not _TwoTier
    assert _CHUNK_COMPILE_LOG[key] == full_log
    np.testing.assert_array_equal(
        np.asarray(model.state["u"].data), after_second)


# ================================================================
#  _TwoTier state machine (stubs)
# ================================================================
def test_two_tier_serves_cheap_while_pending_then_settles():
    event = threading.Event()
    full = _FullStub()
    holder = _TwoTier(_fake_cheap, _BlockingLowered(event, full))
    # pending: settled() is None, serve_cheap runs the cheap callable
    assert holder.settled() is None
    assert holder.serve_cheap("carry", "stepper") == (
        "cheap", "carry", "stepper")
    # release the daemon and join
    event.set()
    holder.wait()
    assert holder.settled() is full
    assert holder.compile_log()[1] == "stub-memory"


def test_two_tier_reraises_background_compile_error():
    holder = _TwoTier(_fake_cheap, _BoomLowered())
    holder.wait()
    with pytest.raises(RuntimeError, match="boom background compile"):
        holder.settled()


# ================================================================
#  Error propagation through step_chunk
# ================================================================
def test_step_chunk_reraises_background_error_on_next_call(
        monkeypatch):
    model = make_model(order=2, chunk_size=4,
                       async_chunk_compile=True)
    record = model._artifacts.record
    stepper = model._stepper
    real_lower = mm._lower_chunk

    def fake_lower(record, carry, stepper, n, force_unroll=None):
        # only the full tier (force_unroll is None) is doomed; the
        # cheap tier (force_unroll=1) still lowers/compiles for real
        if force_unroll is None:
            return _BoomLowered()
        return real_lower(record, carry, stepper, n, force_unroll)

    monkeypatch.setattr(mm, "_lower_chunk", fake_lower)
    key = model_key(model, 4)
    # the miss serves the cheap tier and returns without raising
    out = step_chunk(record, model._carry, stepper, 4,
                    async_compile=True)
    holder = _CHUNK_EXECUTABLES[key]
    holder.wait()
    # the NEXT call surfaces the captured background compile error
    with pytest.raises(RuntimeError, match="boom background compile"):
        step_chunk(record, out, stepper, 4, async_compile=True)


# ================================================================
#  Sync fallbacks (no _TwoTier)
# ================================================================
def test_async_falls_back_to_sync_when_length_is_one():
    model = make_model(order=2, async_chunk_compile=True)
    key = model_key(model, 1)
    run_chunk(model, 1, async_compile=True)
    assert type(_CHUNK_EXECUTABLES[key]) is not _TwoTier


def test_async_falls_back_to_sync_when_unroll_is_one():
    # AB order=1 has scan_unroll == 1: no cheap tier to serve
    model = make_model(order=1, async_chunk_compile=True)
    key = model_key(model, 4)
    run_chunk(model, 4, async_compile=True)
    assert type(_CHUNK_EXECUTABLES[key]) is not _TwoTier


# ================================================================
#  Knob surface (constructor validation + property)
# ================================================================
def test_async_chunk_compile_rejects_non_bool():
    with pytest.raises(ValueError,
                       match="async_chunk_compile must be a bool"):
        make_model(async_chunk_compile="yes")
    with pytest.raises(ValueError,
                       match="async_chunk_compile must be a bool"):
        make_model(async_chunk_compile=1)
    with pytest.raises(ValueError,
                       match="async_chunk_compile must be a bool"):
        make_model(async_chunk_compile=0)


def test_async_chunk_compile_property_roundtrips():
    assert make_model().async_chunk_compile is False
    assert make_model(async_chunk_compile=True).async_chunk_compile \
        is True
    assert make_model(async_chunk_compile=False).async_chunk_compile \
        is False


# ================================================================
#  Backward compatibility (plain positional sync path)
# ================================================================
def test_positional_sync_path_accounts_and_caches_one_entry():
    model = make_model(order=2)
    record = model._artifacts.record
    stepper = model._stepper
    key = model_key(model, 4)
    # legacy call: no async_compile keyword, natural unroll > 1
    step_chunk(record, model._carry, stepper, 4)
    assert key in _CHUNK_COMPILE_LOG
    assert chunk_cache_size() == 1
    assert type(_CHUNK_EXECUTABLES[key]) is not _TwoTier
