"""Tests for the jitted run loop (model/model.py: step_chunk).

Covers the shared jitted entry: treedef stability through the
scanned chunk, clock advance, the S5 isfinite reduction (sticky
flag, exact first-failure iteration), the compiled-executable cache
(one entry per (record, length, structure); identical re-assemblies
on one grid SHARE it; the carry-canonicalization discipline keeps
host writes on the same entry), donation ergonomics, live
``ctx.params`` from the carry's module leaves and the loop-invariant
stepper input (zero-recompile parameter and dt sweeps), and the
chunk-plan contract (lengths {C, 1} only; greedy split equivalence).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.model.declarations import FieldDeclaration
from fridom.model.model import (
    _CHUNK_EXECUTABLES,
    Model,
    PanicState,
    _seal_carry_ghosts,
    chunk_cache_size,
    step_chunk,
)
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.results import PanicError
from fridom.model.terms import term
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

DT = 0.5
N = 8


# ================================================================
#  Toy module: constant forcing du/dt = gain (ctx.params-read)
# ================================================================
@partial(jaxify, dynamic=("gain",))
class GainForcing(Module):

    """One PROGNOSTIC field forced by a provided parameter.

    The term reads the parameter through ``ctx.params`` (the
    ``.get`` spelling tolerates the landed dry run's empty params);
    the scaling itself is spelled on the raw data, so the module
    declares its (zero) halo demand and is halo-trace exempt (V-N2).
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


def make_model(grid=None, gain=1.0, dt=DT, **kwargs):
    if grid is None:
        grid = make_grid()
    return Model(grid=grid, modules=(GainForcing(gain),),
                 time_stepper=AdamBashforth(dt, order=1), **kwargs)


def chunk(model, n, carry=None):
    """Run one chunk over the model's (or a given) carry."""
    if carry is None:
        carry = model._carry
    return step_chunk(model._artifacts.record, carry,
                      model._stepper, n)


# ================================================================
#  Scan mechanics: treedef, clock, forcing arithmetic
# ================================================================
def test_chunk_preserves_the_carry_treedef():
    model = make_model()
    before = jax.tree_util.tree_structure(model._carry)
    out = chunk(model, 3)
    assert jax.tree_util.tree_structure(out) == before


def test_chunk_advances_clock_and_state():
    model = make_model(gain=2.0)
    out = chunk(model, 4)
    assert int(out.clock.it) == 4
    assert float(out.clock.elapsed) == pytest.approx(4 * DT)
    # AB1 with constant tendency: u = n * dt * gain, exactly
    assert np.allclose(np.asarray(out.state["u"].data),
                       4 * DT * 2.0)


def test_scan_chunk_matches_repeated_chunk1():
    model = make_model(gain=0.7)
    scanned = chunk(model, 4)
    other = make_model(grid=make_grid(), gain=0.7)
    stepped = other._carry
    for _ in range(4):
        stepped = chunk(other, 1, carry=stepped)
    # different programs (scan-4 vs 4 x chunk-1): tolerance-based
    # (phase-1 finding 1); the op sequence is identical
    assert np.allclose(np.asarray(scanned.state["u"].data),
                       np.asarray(stepped.state["u"].data))
    assert int(scanned.clock.it) == int(stepped.clock.it)


# ================================================================
#  S5 — the once-per-chunk isfinite reduction
# ================================================================
def poisoned_carry(model):
    state = model._carry.state
    bad = state.replace(
        u=state["u"].with_data(jnp.full(N, jnp.nan)))
    return model._carry.replace(state=bad)


def test_panic_records_the_detecting_chunk_boundary():
    model = make_model()
    out = chunk(model, 3, carry=poisoned_carry(model))
    assert bool(out.panic.flag)
    assert int(out.panic.it) == 3        # the one S5 check per chunk


def test_panic_flag_is_sticky_and_it_stays_at_detection():
    model = make_model()
    out = chunk(model, 5, carry=poisoned_carry(model))
    assert bool(out.panic.flag)
    assert int(out.panic.it) == 5        # the detecting boundary
    again = chunk(model, 2, carry=out)
    assert bool(again.panic.flag)
    assert int(again.panic.it) == 5      # later chunks never move it


def test_inf_counts_as_non_finite():
    model = make_model()
    state = model._carry.state
    carry = model._carry.replace(state=state.replace(
        u=state["u"].with_data(jnp.full(N, jnp.inf))))
    out = chunk(model, 2, carry=carry)
    assert bool(out.panic.flag)


def test_fresh_panic_state_is_cleared():
    fresh = PanicState(flag=jnp.zeros((), dtype=bool),
                       it=jnp.asarray(0, jnp.result_type(int)))
    assert not bool(fresh.flag)
    assert int(fresh.it) == 0


# ================================================================
#  The compiled-executable cache (the shared-runner discipline)
# ================================================================
def test_one_executable_per_record_and_length():
    model = make_model()
    base = chunk_cache_size()
    out = chunk(model, 3)
    assert chunk_cache_size() - base == 1
    reference = chunk_cache_size()
    out = chunk(model, 3, carry=out)     # same key: cache hit
    assert chunk_cache_size() == reference
    chunk(model, 2, carry=out)           # new length: new entry
    assert chunk_cache_size() == reference + 1


def test_identical_reassembly_on_one_grid_shares_the_cache():
    grid = make_grid()
    first = make_model(grid=grid)
    first.advance(2)
    reference = chunk_cache_size()
    second = make_model(grid=grid)       # identical re-assembly
    second.advance(2)                    # SAME record, same entry
    assert chunk_cache_size() == reference


def test_canonicalization_keeps_host_writes_on_one_entry():
    # the wave-4.1 finding: host-born fields (create_field) and jit
    # outputs must key ONE entry — set_fields + advance must not
    # compile a second chunk executable
    model = make_model()
    model.advance(1)
    reference = chunk_cache_size()
    model.set_fields(u=np.zeros(N))      # host-born carry leaves
    model.advance(1)
    assert chunk_cache_size() == reference
    model.update_parameters({"toy.gain": 3.0})
    model.advance(1)
    assert chunk_cache_size() == reference


def test_compile_counter_zero_across_repeated_advance(
        compile_counter):
    model = make_model()
    model.advance(3)                     # warm every code path
    compile_counter.reset()
    model.advance(3)
    assert compile_counter.count == 0


# ================================================================
#  Live parameters (ctx.params from carry modules + live stepper)
# ================================================================
def test_ctx_params_read_the_live_module_leaf():
    model = make_model(gain=1.0)
    model.advance(2)
    first = float(model.state["u"].data[0])
    assert first == pytest.approx(2 * DT * 1.0)
    model.update_parameters({"toy.gain": 5.0})
    model.reset()
    model.advance(2)
    second = float(model.state["u"].data[0])
    assert second == pytest.approx(2 * DT * 5.0)


def test_ctx_params_carry_the_live_stepper_dt():
    # the dt leaf is a traced input, never a baked constant: a dt
    # sweep changes the numbers with ZERO new executables
    model = make_model(gain=1.0)
    model.advance(1)
    reference = chunk_cache_size()
    model.update_parameters({"stepper.dt": 2 * DT})
    model.reset()
    model.advance(1)
    assert chunk_cache_size() == reference
    assert float(model.clock.elapsed) == pytest.approx(2 * DT)
    assert float(model.state["u"].data[0]) == pytest.approx(
        2 * DT * 1.0)


# ================================================================
#  The chunk plan ({C, 1} only; greedy split equivalence)
# ================================================================
def test_chunk_plan_lengths_are_c_and_one():
    model = make_model(chunk_size=4)
    assert list(model._chunk_plan(11)) == [4, 4, 1, 1, 1]
    assert list(model._chunk_plan(8)) == [4, 4]
    assert list(model._chunk_plan(3)) == [1, 1, 1]
    assert list(model._chunk_plan(0)) == []


def test_greedy_plan_makes_split_advance_bitwise():
    grid = make_grid()
    whole = make_model(grid=grid, chunk_size=4)
    split = make_model(grid=grid, chunk_size=4)
    whole.advance(11)
    split.advance(8)
    split.advance(3)
    # same greedy plan [4,4,1,1,1] == [4,4] + [1,1,1]: the same
    # executables in the same order -> bitwise
    assert np.array_equal(
        np.asarray(whole.state["u"].data),
        np.asarray(split.state["u"].data))
    assert float(whole.clock.elapsed) == float(
        split.clock.elapsed)


def test_panic_aborts_at_the_chunk_boundary_not_midchunk():
    model = make_model(chunk_size=4)
    model._carry = poisoned_carry(model)
    with pytest.raises(PanicError) as err:
        model.advance(8)
    # the abort fires AT the first boundary: 4 steps committed
    assert err.value.partial.steps_done == 4
    assert err.value.first_bad_it == 4


# ================================================================
#  Donation ergonomics
# ================================================================
def test_stepper_is_not_donated():
    model = make_model()
    chunk(model, 2)
    # the stepper leaf survives (loop-invariant, non-donated)
    assert float(model._stepper.dt) == pytest.approx(DT)


def test_chunk_output_feeds_the_next_chunk():
    model = make_model()
    out = chunk(model, 2)
    out = chunk(model, 2, carry=out)     # donated hand-off
    assert int(out.clock.it) == 4


# ================================================================
#  The carry seal — production-side ghost fill
# ================================================================
class HaloGainForcing(GainForcing):

    """The toy forcing on a width-1 negotiated halo (seal probe)."""

    extra_halo = HaloSpec({"x": 1})


def make_halo_model(**kwargs):
    return Model(grid=make_grid(), modules=(HaloGainForcing(),),
                 time_stepper=AdamBashforth(DT, order=1), **kwargs)


def test_seal_syncs_zero_claim_fields_and_skips_sealed_ones():
    # _seal_carry_ghosts is the production-side twin of the claim
    # reset: a field with partial claims is synced to the full
    # negotiated widths (same values as grid.sync); a field already
    # claiming full validity passes through untouched (the identity
    # keeps untouched AUXILIARY carry fields free)
    model = make_halo_model()
    field = model._carry.state["u"]
    grid = field.grid
    bare = type(field)(grid, field.function_space,
                       field._data, field.metadata)   # zero claims
    sealed = _seal_carry_ghosts((bare,))[0]
    full = grid.decomposition.halo.over(
        tuple(field.function_space.names))
    assert sealed.halo_valid == full
    assert jnp.array_equal(sealed._data, grid.sync(bare)._data)
    assert _seal_carry_ghosts((sealed,))[0] is sealed


def test_chunk_seals_the_carry_at_the_boundary():
    # The compiled chunk carries the seal: the state's ghost fill
    # happens at the carry boundary (where the buffers materialize
    # anyway) as in-place DUS writes behind an optimization_barrier,
    # NOT at consumption inside the next step's kernels. This is the
    # perf contract that removed the advective fill cost (2026-07-15).
    # On one device the lazy consumption fill is a pure gather, so a
    # dynamic-update-slice in the compiled chunk exists if and only
    # if the seal's write spelling is in place (the barrier itself is
    # consumed during optimization and leaves no opcode behind).
    model = make_halo_model()
    before = chunk_cache_size()
    chunk(model, 2)
    assert chunk_cache_size() == before + 1
    text = list(_CHUNK_EXECUTABLES.values())[-1].as_text()
    assert "dynamic-update-slice" in text
