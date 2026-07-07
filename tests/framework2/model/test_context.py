"""Tests for the step context (framework2/model/context.py)."""
import jax
import jax.numpy as jnp
import pytest

from fridom.framework2.model.context import StepContext
from fridom.framework2.model.terms import Treatment


def make_context(f0=1e-4, time=0.0, dt=60.0, stage_dt=30.0):
    return StepContext(
        params={"coriolis.f0": jnp.asarray(f0)},
        clock=jnp.asarray(time),
        dt=jnp.asarray(dt),
        stage_dt=jnp.asarray(stage_dt),
    )


# ================================================================
#  Construction and accessors
# ================================================================
def test_records_attributes():
    ctx = make_context(f0=1e-4, time=120.0, dt=60.0, stage_dt=30.0)
    assert ctx.params["coriolis.f0"] == pytest.approx(1e-4)
    assert ctx.clock == pytest.approx(120.0)
    assert ctx.dt == pytest.approx(60.0)
    assert ctx.stage_dt == pytest.approx(30.0)
    assert ctx.tendency_sums is None


def test_params_are_copied_at_construction():
    params = {"a": jnp.asarray(1.0)}
    ctx = StepContext(params=params, clock=jnp.asarray(0.0),
                      dt=jnp.asarray(1.0), stage_dt=jnp.asarray(1.0))
    params["b"] = jnp.asarray(2.0)
    assert "b" not in ctx.params


def test_scalars_are_coerced_to_arrays():
    ctx = StepContext(params={}, clock=jnp.asarray(0.0),
                      dt=60.0, stage_dt=30.0)
    assert isinstance(ctx.dt, jax.Array)
    assert isinstance(ctx.stage_dt, jax.Array)


def test_tendency_sums_seam_is_stored():
    # the post-TENDENCY extension slot (populated by the composer,
    # wave 3+); the vocabulary only carries it
    sums = {Treatment.EXPLICIT: "state-shaped"}
    ctx = StepContext(params={}, clock=jnp.asarray(0.0),
                      dt=jnp.asarray(1.0), stage_dt=jnp.asarray(1.0),
                      tendency_sums=sums)
    assert ctx.tendency_sums is sums


def test_repr_smoke():
    assert "StepContext(" in repr(make_context())


# ================================================================
#  Frozenness
# ================================================================
def test_setattr_is_rejected():
    ctx = make_context()
    with pytest.raises(AttributeError, match="frozen"):
        ctx.dt = jnp.asarray(1.0)


def test_new_attributes_are_rejected():
    ctx = make_context()
    with pytest.raises(AttributeError, match="frozen"):
        ctx.extra = 1


def test_delattr_is_rejected():
    ctx = make_context()
    with pytest.raises(AttributeError, match="frozen"):
        del ctx.dt


# ================================================================
#  Pytree round trip
# ================================================================
def test_flatten_unflatten_round_trip():
    ctx = make_context(f0=2e-4, time=60.0, dt=10.0, stage_dt=5.0)
    leaves, treedef = jax.tree_util.tree_flatten(ctx)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is StepContext
    assert rebuilt.params["coriolis.f0"] == pytest.approx(2e-4)
    assert rebuilt.clock == pytest.approx(60.0)
    assert rebuilt.dt == pytest.approx(10.0)
    assert rebuilt.stage_dt == pytest.approx(5.0)
    assert rebuilt.tendency_sums is None


def test_unflattened_context_is_still_frozen():
    ctx = make_context()
    leaves, treedef = jax.tree_util.tree_flatten(ctx)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    with pytest.raises(AttributeError, match="frozen"):
        rebuilt.dt = jnp.asarray(1.0)


def test_all_leaves_are_scalars_in_base_form():
    # load-bearing for halo-tracer indifference: the tracer wraps
    # state; the base ctx is scalars — zero mimicry machinery
    leaves = jax.tree_util.tree_leaves(make_context())
    assert leaves
    assert all(jnp.shape(leaf) == () for leaf in leaves)


def test_value_different_contexts_share_a_treedef():
    ctx1 = make_context(f0=1e-4, dt=60.0)
    ctx2 = make_context(f0=5e-4, dt=1.0)
    treedef1 = jax.tree_util.tree_structure(ctx1)
    treedef2 = jax.tree_util.tree_structure(ctx2)
    assert treedef1 == treedef2


# ================================================================
#  Trace behavior
# ================================================================
def test_usable_as_scan_carry():
    def step(carry, _):
        advanced = StepContext(
            params={"coriolis.f0": carry.params["coriolis.f0"]},
            clock=carry.clock + carry.dt,
            dt=carry.dt,
            stage_dt=carry.stage_dt,
        )
        return advanced, carry.clock

    ctx0 = make_context(time=0.0, dt=2.0)
    final, times = jax.lax.scan(step, ctx0, None, length=5)
    assert final.clock == pytest.approx(10.0)
    assert times.shape == (5,)


def test_dynamic_scalar_changes_share_one_trace(compile_counter):
    @jax.jit
    def evaluate(ctx):
        return ctx.dt * 2.0 + ctx.params["coriolis.f0"]

    ctx1 = make_context(f0=1e-4, time=0.0, dt=60.0, stage_dt=30.0)
    ctx2 = make_context(f0=9e-4, time=99.0, dt=1.0, stage_dt=0.5)
    evaluate(ctx1)  # compile once
    compile_counter.reset()
    evaluate(ctx2)  # differs only in dynamic scalar values
    assert compile_counter.count == 0
