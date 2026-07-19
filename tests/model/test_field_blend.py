"""The generic declaration-level FieldBlend mechanism (AR-D2, R2).

A field-valued parameter may be an affine combination of
assembly-materialized ingredient profiles with stage-time scalar
weights, ``p(t) = sum_i w_i(t) * P_i``. These tests exercise the
mechanism on a TOY module independent of Coriolis: the blend value at
stage time equals the affine combination, the ingredient profiles are
static AUXILIARY (scan-stable treedef, halo-neutral pointwise blend),
and sweeping the weight-ramp endpoints never recompiles (the weights
are dynamic leaves). The two-endpoint blend {p_ref, p_target - p_ref}
with weights {1, lambda(t)} is the special case of a constant and a
leaf weight.
"""
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import jaxify
from fridom.model.declarations import FieldDeclaration, Lifecycle
from fridom.model.field_blend import BlendIngredient, FieldBlend
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration, leaf
from fridom.model.stages import StageKind
from fridom.model.terms import term
from fridom.model.time_dependent import Ramp
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated, Profile

N = 8


# ================================================================
#  A toy module: g(x,t) = a0(t)*1 + a1(t)*x drives du/dt = g
# ================================================================
def _toy_const(self, grid, space):  # noqa: ARG001
    """Owner-method default: the constant unit profile P_0 = 1."""
    return grid.create_field(space, data=jnp.ones(space.shape),
                             name="g_const")


def _toy_coord(self, grid, space):  # noqa: ARG001
    """Owner-method default: the coordinate profile P_1 = x."""
    return grid.create_field(space, init=lambda x: x, name="g_coord")


_TOY_BLEND = FieldBlend((
    BlendIngredient("g_const", weight="a0", build=_toy_const),
    BlendIngredient("g_coord", weight="a1", build=_toy_coord),
))


@partial(jaxify, dynamic=("a0", "a1"))
class BlendedForcing(Module):

    """One PROGNOSTIC ``u`` forced by a FieldBlend AUXILIARY ``g(x,t)``.

    ``g(x,t) = a0(t)*1 + a1(t)*x`` is the generic two-ingredient blend;
    ``du/dt = g`` (interpolated onto the ``u`` nodes). The weights
    ``a0``/``a1`` are ordinary scalar leaves (plain floats or
    ``fr.Ramp``) — the module wires its own ingredients (author-level
    machinery only).
    """

    def __init__(self, a0=1.0, a1=0.0):
        """Store the two blend weights as scalar leaves."""
        self.a0 = leaf(a0)
        self.a1 = leaf(a1)

    parameter_declarations = (
        ParameterDeclaration("toy.a0", attr="a0", units="1"),
        ParameterDeclaration("toy.a1", attr="a1", units="1"),
    )

    @property
    def blend_active(self):
        """Whether either weight is time-dependent (the FieldBlend)."""
        return _TOY_BLEND.is_active(self)

    @property
    def field_declarations(self):
        """The PROGNOSTIC ``u`` plus the two blend ingredients."""
        return (
            FieldDeclaration("u", space=Collocated(),
                             long_name="Forced"),
            *_TOY_BLEND.field_declarations(
                space=Profile("x"), long_name="blend ingredient"),
        )

    @term(advances=("u",), linear=True)
    def force(self, state, ctx):
        """``du/dt = g(x,t)`` from the stage-time blend."""
        time = getattr(ctx.clock, "time", ctx.clock)
        g = _TOY_BLEND.evaluate(self, state, time)
        return {"u": g.to(state["u"])}


def make_grid(device_ids=None):
    """Return a small periodic-x grid (one blocked factor to shard)."""
    return Grid((IntervalMesh(N, (0.0, 1.0), periodic=True, name="x"),),
                device_ids=device_ids)


def make_model(a0=1.0, a1=0.0, *, dt=1e-2, grid=None):
    """Return a one-module model forced by the FieldBlend."""
    return Model(
        grid=make_grid() if grid is None else grid,
        modules=(BlendedForcing(a0, a1),),
        time_stepper=AdamBashforth(dt, order=1))


def _x_nodes(model):
    """Return the x-coordinate at the ``g`` profile's own nodes."""
    g = model.state["g_coord"]
    return np.asarray(g.data)


# ================================================================
#  is_active: the static/ramp predicate (host-side)
# ================================================================
def test_is_active_reflects_time_dependent_weights():
    assert not _TOY_BLEND.is_active(BlendedForcing(1.0, 2.0))
    ramp = Ramp(0.0, 1.0, period=1.0)
    assert _TOY_BLEND.is_active(BlendedForcing(a0=ramp))
    assert _TOY_BLEND.is_active(BlendedForcing(a1=ramp))


def test_field_declarations_carry_the_ingredient_profiles():
    assert tuple(i.field for i in _TOY_BLEND.ingredients) == (
        "g_const", "g_coord")
    decls = {d.name: d for d in BlendedForcing().field_declarations}
    assert set(decls) == {"u", "g_const", "g_coord"}
    for name in ("g_const", "g_coord"):
        assert decls[name].lifecycle is Lifecycle.AUXILIARY
        assert decls[name].default_form == "owner_method"


def test_a_constant_weight_ingredient_needs_no_leaf():
    # the two-endpoint form {p_ref, p_target - p_ref} x {1, lambda}:
    # a constant (float) weight rides with no module attribute
    blend = FieldBlend((
        BlendIngredient("g_const", weight=1.0, build=_toy_const),
        BlendIngredient("g_coord", weight="a1", build=_toy_coord)))
    # a constant-only-weight blend is never "active" on its own
    assert not blend.is_active(BlendedForcing(1.0, 2.0))
    assert blend.is_active(BlendedForcing(a1=Ramp(0.0, 1.0, period=1.0)))


# ================================================================
#  evaluate: the stage-time affine combination
# ================================================================
@pytest.mark.parametrize("t", [0.0, 0.3, 0.75, 1.5])
def test_evaluate_equals_the_affine_combination(t):
    ramp0 = Ramp(0.4, 1.1, period=1.0, curve="cosine")
    ramp1 = Ramp(0.0, 2.0, period=1.0, curve="exp")
    model = make_model(a0=ramp0, a1=ramp1)
    module = model.module(BlendedForcing)
    x = _x_nodes(model)
    g = _TOY_BLEND.evaluate(module, model.state, t)
    want = float(ramp0.at_time(t)) + float(ramp1.at_time(t)) * x
    np.testing.assert_allclose(np.asarray(g.data), want, rtol=0.0,
                               atol=1e-13)


# ================================================================
#  The SELF_UPDATE rewrite path (TDF-D11): stage + rewrite helpers
# ================================================================
def test_stage_declares_a_self_update_rewrite():
    """FieldBlend.stage wires the reusable SELF_UPDATE stage (TDF-D11).

    A future consumer wires two thin lines: this stage() call (reading
    the target plus every ingredient, writing the target, no extra_halo)
    and a one-line rewrite method pointing fn at rewrite().
    """
    stage = _TOY_BLEND.stage("_rewrite_g", target="g", name="g_blend")
    assert stage.kind is StageKind.SELF_UPDATE
    assert stage.fn == "_rewrite_g"
    assert stage.name == "g_blend"
    assert stage.reads == ("g", "g_const", "g_coord")
    assert stage.writes == ("g",)


@pytest.mark.parametrize("t", [0.0, 0.4, 1.0])
def test_rewrite_wraps_evaluate_at_the_substage_clock(t):
    """FieldBlend.rewrite returns {target: evaluate at ctx clock time}."""
    ramp1 = Ramp(0.0, 2.0, period=1.0, curve="exp")
    model = make_model(a0=0.5, a1=ramp1)
    module = model.module(BlendedForcing)
    ctx = SimpleNamespace(clock=t)
    out = _TOY_BLEND.rewrite(module, model.state, ctx, target="g")
    assert set(out) == {"g"}
    want = 0.5 + float(ramp1.at_time(t)) * _x_nodes(model)
    np.testing.assert_allclose(np.asarray(out["g"].data), want, atol=1e-12)


def test_evaluate_two_endpoint_form_is_p_ref_plus_lambda_delta():
    # p(t) = 1 * p_ref + lambda(t) * delta, p_ref = P0, delta = P1
    lam = Ramp(0.0, 1.0, period=1.0, curve="exp")
    blend = FieldBlend((
        BlendIngredient("g_const", weight=1.0, build=_toy_const),
        BlendIngredient("g_coord", weight="a1", build=_toy_coord)))
    model = make_model(a1=lam)
    module = model.module(BlendedForcing)
    x = _x_nodes(model)
    for t in (0.0, 0.5, 1.0):
        g = blend.evaluate(module, model.state, t)
        want = 1.0 + float(lam.at_time(t)) * x
        np.testing.assert_allclose(np.asarray(g.data), want, atol=1e-13)


def test_tendency_reads_the_stage_time_blend():
    # du/dt = g(x,t): the model tendency at t equals a0(t) + a1(t)*x
    ramp1 = Ramp(0.0, 3.0, period=1.0, curve="exp")
    model = make_model(a0=0.5, a1=ramp1)
    for t in (0.0, 0.4, 1.0):
        dz = model.tendency(model.state, t=t)
        x = _x_nodes(model)
        want = 0.5 + float(ramp1.at_time(t)) * x
        np.testing.assert_allclose(
            np.asarray(dz["u"].data), want, rtol=0.0, atol=1e-12)


# ================================================================
#  Treedef stability: the ingredients are static AUXILIARY
# ================================================================
def test_carry_treedef_is_stable_across_steps():
    model = make_model(a0=0.7, a1=Ramp(0.0, 1.0, period=1.0))
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(4)
    after = jax.tree_util.tree_structure(model._carry)
    assert after == before
    # the ingredient fields are untouched by the advance (static AUX)
    for name in ("g_const", "g_coord"):
        np.testing.assert_array_equal(
            np.asarray(model.state[name].data),
            np.asarray(make_model(a0=0.7).state[name].data))


def test_ramped_run_differs_from_the_frozen_endpoint():
    # sanity: the ramp actually drives the solution
    ramp = Ramp(0.0, 4.0, period=6e-2, curve="cosine")
    model = make_model(a1=ramp)
    model.advance(6)
    ramped = np.asarray(model.state["u"].data)
    frozen = make_model(a1=float(ramp.at_time(0.0)))
    frozen.advance(6)
    assert not np.allclose(ramped, np.asarray(frozen.state["u"].data))


# ================================================================
#  Zero-recompile weight sweeps (the weights are dynamic leaves)
# ================================================================
def test_weight_endpoint_sweeps_do_not_recompile(compile_counter):
    model = make_model()
    state = model.state
    t = jnp.asarray(0.5)

    @jax.jit
    def blend_value(module, state, t):
        return _TOY_BLEND.evaluate(module, state, t).data

    # build the swept modules BEFORE reset(): eager Ramp construction
    # (jnp.asarray on the leaves) itself traces and would be counted.
    # Warm and swept modules are all freshly built, so their leaves
    # share one (weak) dtype and the jaxpr is reused across endpoints.
    warm = BlendedForcing(a0=Ramp(0.2, 0.9, period=1.0),
                          a1=Ramp(0.0, 2.0, period=1.0))
    swept = [
        BlendedForcing(a0=Ramp(0.5, -1.0, period=3.0, t0=-2.0),
                       a1=Ramp(1.0, 4.0, period=0.5, t0=1.0)),
        BlendedForcing(a0=Ramp(-2.0, 2.0, period=7.0),
                       a1=Ramp(0.3, 0.3, period=2.0)),
    ]
    blend_value(warm, state, t).block_until_ready()

    compile_counter.reset()
    for module in swept:
        blend_value(module, state, t).block_until_ready()
    assert compile_counter.count == 0


def test_advancing_a_swept_ramp_does_not_recompile(compile_counter):
    # the two swept ramps are built up front (Ramp construction traces)
    warm_ramp = Ramp(0.7, 0.3, period=1.5)
    swept_ramp = Ramp(1.0, -3.0, period=0.5, t0=1.0)
    model = make_model(a0=0.5, a1=Ramp(0.0, 2.0, period=1.0))
    model.advance(2)                     # warm the chunk executable
    model.update_parameters({"toy.a1": warm_ramp})  # warm the update path
    model.advance(2)

    compile_counter.reset()
    model.update_parameters({"toy.a1": swept_ramp})
    model.advance(2)
    assert compile_counter.count == 0


# ================================================================
#  Multi-device: the pointwise blend is halo-neutral (gate d)
# ================================================================
@pytest.mark.multi_device
def test_blend_is_device_count_invariant(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    ramp = Ramp(0.0, 2.0, period=5e-2, curve="exp")

    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        model = make_model(a0=0.7, a1=ramp,
                           grid=make_grid(device_ids=device_ids))
        model.advance(5)
        results[tag] = np.asarray(model.state["u"].data)
        if tag == "many":
            assert model.state["u"]._data.sharding.spec[0] == "devices"
    np.testing.assert_allclose(results["many"], results["one"],
                               rtol=1e-11, atol=1e-13)
