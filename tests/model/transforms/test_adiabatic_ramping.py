"""Tests for the Tier-2 ``AdiabaticRamping`` base surface (R3).

The generalized ramping leg (08 §10.9, AR-D1..D9): tuple-sugar and
verbatim-window ramps feed a Ramp-valued internal :class:`Propagator`;
``.down`` / ``.backward`` derive the four legs; the AR-D6 guard refuses
irreversible backward legs. The dynamical laws (endpoint exactness,
the four-leg matrix, near-inverse pairs, window-vs-composition
equivalence) are exercised on a small RK4 inertial-oscillation model
with a ``scaling.nonlinearity``-scaled nonlinear term and a
``coriolis.f0``-scaled rotation term (both read from ``ctx.params``, so
the ramps genuinely deform the operator).

Measured tolerances (RK4, dt=2e-3, 20 ramp steps, this grid):
  * near-inverse residual ``backward @ up`` ~ 1.7e-11 (asserted < 1e-7);
  * negative law ``down @ up`` ~ 0.63 relative (>> the near-inverse
    residual — the AR-D8 phase argument);
  * window-vs-composition ~ 1.8e-17 (RK4 is a single-step method, so a
    leg-boundary restart is a no-op and the two protocol forms are
    bitwise identical; asserted < 1e-12). A multistep stepper would
    show a genuine restart tolerance here.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.model import params
from fridom.model import term_predicates as terms
from fridom.model.closures.base import ClosureBase
from fridom.model.declarations import FieldDeclaration
from fridom.model.errors import IrreversibleTermError
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.terms import term
from fridom.model.time_dependent import Ramp
from fridom.model.time_steppers.runge_kutta import (
    ExplicitRungeKutta,
    tableaus,
)
from fridom.model.transforms.adiabatic_ramping import AdiabaticRamping
from fridom.model.transforms.errors import TraceError
from fridom.model.transforms.norms import relative_l2
from fridom.model.transforms.propagator import Propagator
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

N = 8
DT = 2e-3
RAMP = 0.04  # -> round(0.04 / 2e-3) = 20 ramp steps per leg
RO = 0.8
F0 = 8.0


# ================================================================
#  Toy dynamical modules (terms read the ramped params from ctx)
# ================================================================
@jaxify
class RotationScaled(Module):

    """Linear rotation scaled by ``coriolis.f0`` read from ctx."""

    field_declarations = (
        FieldDeclaration("u", space=Collocated()),
        FieldDeclaration("v", space=Collocated()),
    )

    @term(name="cor", advances=("u", "v"), linear=True)
    def cor(self, state, ctx):
        f0 = ctx.params[params.CORIOLIS_F0]
        return {"u": state["v"] * f0, "v": state["u"] * (-f0)}


@partial(jaxify, dynamic=("f0",))
class F0Provider(Module):

    """A pure provider binding ``coriolis.f0``."""

    field_declarations = ()
    parameter_declarations = (
        ParameterDeclaration("coriolis.f0", attr="f0", units="1/s"),)

    def __init__(self, f0=F0):
        self.f0 = jnp.asarray(f0, dtype=dtype_real())


@jaxify
class NonlinearScaled(Module):

    """A nonlinear self-advection scaled by ``scaling.nonlinearity``."""

    field_declarations = ()

    @term(name="adv", advances=("u", "v"))
    def adv(self, state, ctx):
        ro = ctx.params[params.SCALING_NONLINEARITY]
        return {"u": state["u"] * state["v"] * ro,
                "v": -state["v"] * state["u"] * ro}


@partial(jaxify, dynamic=("rossby",))
class RossbyProvider(Module):

    """A pure provider binding ``scaling.nonlinearity``."""

    field_declarations = ()
    parameter_declarations = (
        ParameterDeclaration("scaling.nonlinearity", attr="rossby",
                             units="1"),)

    def __init__(self, rossby=RO):
        self.rossby = jnp.asarray(rossby, dtype=dtype_real())


@jaxify
class ToyDrag(ClosureBase):

    """A dissipative closure (``ClosureBase``) — a linear u drag."""

    default_targets = None
    field_declarations = ()

    @term(name="drag", advances=("u",), linear=True)
    def drag(self, state, _ctx):
        return {"u": state["u"] * (-0.1)}


# ================================================================
#  Builders
# ================================================================
def make_model(modules, dt=DT):
    grid = Grid((IntervalMesh(N, (0.0, 1.0), periodic=True, name="x"),))
    return Model(grid=grid, modules=modules,
                 time_stepper=ExplicitRungeKutta(dt, tableau=tableaus.RK4),
                 name="toy")


def rossby_model():
    """Build a model whose nonlinear term is scaled by rossby."""
    return make_model((RotationScaled(), F0Provider(),
                       NonlinearScaled(), RossbyProvider()))


def wave_state(model, u_shift=0.5):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    model.set_fields(u=np.sin(2 * np.pi * x) + u_shift,
                     v=np.cos(2 * np.pi * x))
    return model.state


@pytest.fixture
def model():
    return rossby_model()


@pytest.fixture
def z0(model):
    return wave_state(model)


@pytest.fixture
def up(model):
    return AdiabaticRamping(
        model, ramps={params.SCALING_NONLINEARITY: (0.0, RO)},
        ramp_period=RAMP, curve="exp")


# ================================================================
#  Law 1: endpoint exactness
# ================================================================
def test_up_endpoint_exactness(up):
    ramp = up.model.parameters[params.SCALING_NONLINEARITY]
    assert float(ramp.at_time(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(ramp.at_time(RAMP)) == pytest.approx(RO, abs=1e-12)


def test_resolved_ramp_carries_the_endpoints(up):
    ramp = up.ramps[params.SCALING_NONLINEARITY]
    assert isinstance(ramp, Ramp)
    assert float(ramp.v0) == pytest.approx(0.0)
    assert float(ramp.v1) == pytest.approx(RO)


def test_one_step_probe_runs_with_the_ramp_live(model, z0):
    # a 1-step leg advances the internal model with the ramp active
    leg = AdiabaticRamping(
        model, ramps={params.SCALING_NONLINEARITY: (0.0, RO)},
        ramp_period=RAMP, steps=1)
    out = leg(z0)
    assert out.component_names == ("u", "v")
    # the leg advanced exactly one dt from the reset clock
    assert float(leg.model.clock.time) == pytest.approx(DT)
    assert float(relative_l2(out, z0)) > 0.0  # the state moved


def test_window_form_per_parameter_endpoint_exactness(model):
    half = RAMP / 2
    leg = AdiabaticRamping(
        model,
        ramps={
            params.CORIOLIS_F0: Ramp(
                4.0, F0, period=half, t0=0.0, curve="exp"),
            params.SCALING_NONLINEARITY: Ramp(
                0.0, RO, period=half, t0=half, curve="exp")},
        ramp_period=RAMP)
    f0 = leg.model.parameters[params.CORIOLIS_F0]
    ro = leg.model.parameters[params.SCALING_NONLINEARITY]
    # f0 ramps in [0, half]; rossby in [half, RAMP]
    assert float(f0.at_time(0.0)) == pytest.approx(4.0)
    assert float(f0.at_time(RAMP)) == pytest.approx(F0)
    assert float(ro.at_time(0.0)) == pytest.approx(0.0)
    assert float(ro.at_time(RAMP)) == pytest.approx(RO)


# ================================================================
#  Law: the four-leg matrix (dt sign x lambda path)
# ================================================================
def _leg(up, which):
    return {"up": up, "down": up.down,
            "up.backward": up.backward,
            "down.backward": up.down.backward}[which]


@pytest.mark.parametrize(
    ("which", "backward", "p_start", "p_end"),
    [
        pytest.param("up", False, 0.0, RO, id="up"),
        pytest.param("down", False, RO, 0.0, id="down"),
        pytest.param("up.backward", True, RO, 0.0, id="up.backward"),
        pytest.param("down.backward", True, 0.0, RO, id="down.backward"),
    ],
)
def test_four_leg_matrix(up, which, backward, p_start, p_end):
    leg = _leg(up, which)
    dt = float(leg.model.parameters[params.TIME_STEP])
    assert leg.is_backward is backward
    assert (dt < 0.0) is backward
    ramp = leg.model.parameters[params.SCALING_NONLINEARITY]
    t_end = -RAMP if backward else RAMP
    assert float(ramp.at_time(0.0)) == pytest.approx(p_start, abs=1e-9)
    assert float(ramp.at_time(t_end)) == pytest.approx(p_end, abs=1e-9)


def test_down_and_backward_are_involutions(up):
    back_to_up = up.down.down.ramps[params.SCALING_NONLINEARITY]
    assert float(back_to_up.at_time(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert float(back_to_up.at_time(RAMP)) == pytest.approx(RO, abs=1e-9)
    rev2 = up.backward.backward
    assert not rev2.is_backward
    r = rev2.ramps[params.SCALING_NONLINEARITY]
    assert float(r.at_time(0.0)) == pytest.approx(0.0, abs=1e-9)


def test_down_reflects_a_verbatim_window(model):
    half = RAMP / 2
    leg = AdiabaticRamping(
        model,
        ramps={params.SCALING_NONLINEARITY: Ramp(
            0.0, RO, period=half, t0=half, curve="exp")},
        ramp_period=RAMP)
    d = leg.down.ramps[params.SCALING_NONLINEARITY]
    # window [half, RAMP] reflects to [0, half]; endpoints swap
    assert float(d.t0) == pytest.approx(0.0, abs=1e-12)
    assert float(d.period) == pytest.approx(half)
    assert float(d.v0) == pytest.approx(RO)
    assert float(d.v1) == pytest.approx(0.0)


# ================================================================
#  Law 2: near-inverse pairs and the negative law
# ================================================================
def test_backward_is_a_near_inverse_of_up(up, z0):
    residual = float(relative_l2(up.backward(up(z0)), z0))
    assert residual < 1e-7  # measured ~1.7e-11


def test_down_of_up_is_not_a_round_trip(up, z0):
    z_up = up(z0)
    near_inv = float(relative_l2(up.backward(z_up), z0))
    negative = float(relative_l2(up.down(z_up), z0))
    # phase evolution: down @ up advances ~2 tau, far from identity
    assert negative > 0.1
    assert negative > 1e6 * near_inv


def test_down_backward_near_inverse_of_down(up, z0):
    z_down = up.down(z0)
    residual = float(relative_l2(up.down.backward(z_down), z0))
    assert residual < 1e-7


# ================================================================
#  Law (AR-D5): window form vs composed legs
# ================================================================
def test_window_and_composition_agree(model, z0):
    half = RAMP / 2
    single = AdiabaticRamping(
        model,
        ramps={
            params.CORIOLIS_F0: Ramp(
                4.0, F0, period=half, t0=0.0, curve="exp"),
            params.SCALING_NONLINEARITY: Ramp(
                0.0, RO, period=half, t0=half, curve="exp")},
        ramp_period=RAMP)
    leg_a = AdiabaticRamping(
        model, ramps={params.CORIOLIS_F0: (4.0, F0)},
        ramp_period=half, updates={params.SCALING_NONLINEARITY: 0.0})
    leg_b = AdiabaticRamping(
        model, ramps={params.SCALING_NONLINEARITY: (0.0, RO)},
        ramp_period=half, updates={params.CORIOLIS_F0: F0})
    residual = float(relative_l2(single(z0), leg_b(leg_a(z0))))
    assert residual < 1e-12  # RK4 single-step: machine-exact (~1.8e-17)
    assert single.steps == leg_a.steps + leg_b.steps


# ================================================================
#  AR-D6: the backward-irreversibility guard
# ================================================================
def test_backward_refuses_a_closure_leg():
    m = make_model((RotationScaled(), F0Provider(),
                    ToyDrag(fields=("u",))))
    up = AdiabaticRamping(m, ramps={}, ramp_period=RAMP)
    with pytest.raises(IrreversibleTermError, match="ToyDrag/drag"):
        up.backward  # noqa: B018 — property access triggers the guard


def test_term_filter_drops_the_closure_and_builds():
    m = make_model((RotationScaled(), F0Provider(),
                    ToyDrag(fields=("u",))))
    up = AdiabaticRamping(
        m, ramps={}, ramp_period=RAMP,
        term_filter=~terms.owned_by(ClosureBase))
    back = up.backward
    assert back.is_backward


def test_replace_then_backward_filters():
    # OB's spelling: forward.replace(term_filter=...).backward
    m = make_model((RotationScaled(), F0Provider(),
                    ToyDrag(fields=("u",))))
    forward = AdiabaticRamping(m, ramps={}, ramp_period=RAMP)
    with pytest.raises(IrreversibleTermError):
        forward.replace().backward  # noqa: B018 — still carries the drag
    filtered = forward.replace(
        term_filter=~terms.owned_by(ClosureBase)).backward
    assert filtered.is_backward


# ================================================================
#  Tuple-vs-verbatim ramps and non-Ramp reflection guard
# ================================================================
def test_verbatim_time_dependent_passes_through(model):
    ramp = Ramp(0.0, RO, period=RAMP, t0=0.0, curve="cosine")
    leg = AdiabaticRamping(
        model, ramps={params.SCALING_NONLINEARITY: ramp}, ramp_period=RAMP)
    assert leg.ramps[params.SCALING_NONLINEARITY] is ramp


def test_tuple_sugar_builds_a_full_window_ramp(up):
    ramp = up.ramps[params.SCALING_NONLINEARITY]
    assert float(ramp.t0) == pytest.approx(0.0)
    assert float(ramp.period) == pytest.approx(RAMP)


def test_bad_ramp_spec_raises(model):
    with pytest.raises(TypeError, match="v_ref, v_target"):
        AdiabaticRamping(
            model, ramps={params.SCALING_NONLINEARITY: 3.0}, ramp_period=RAMP)


def test_down_of_a_non_ramp_curve_raises(model):
    affine = Ramp(0.0, RO, period=RAMP) * 2.0  # an _Affine, not a Ramp
    leg = AdiabaticRamping(
        model, ramps={params.SCALING_NONLINEARITY: affine}, ramp_period=RAMP)
    with pytest.raises(TypeError, match=r"\.down reflects"):
        leg.down  # noqa: B018 — property access triggers the reflection


def test_backward_of_a_non_ramp_curve_raises(model):
    affine = Ramp(0.0, RO, period=RAMP) * 2.0
    leg = AdiabaticRamping(
        model, ramps={params.SCALING_NONLINEARITY: affine}, ramp_period=RAMP)
    with pytest.raises(TypeError, match=r"\.backward reflects"):
        leg.backward  # noqa: B018 — property access triggers the reflection


# ================================================================
#  steps snapping
# ================================================================
def test_steps_default_snaps_to_ramp_period(up):
    assert up.steps == 20  # round(0.04 / 2e-3)


def test_steps_override(model):
    leg = AdiabaticRamping(
        model, ramps={params.SCALING_NONLINEARITY: (0.0, RO)},
        ramp_period=RAMP, steps=5)
    assert leg.steps == 5


def test_steps_floor_is_one(model):
    leg = AdiabaticRamping(
        model, ramps={params.SCALING_NONLINEARITY: (0.0, RO)},
        ramp_period=DT / 4)  # round(0.25) == 0 -> max(1, 0)
    assert leg.steps == 1


# ================================================================
#  replace() frozen-config copy-with
# ================================================================
def test_replace_preserves_ramps_by_default(up):
    replaced = up.replace(name="renamed")
    assert replaced.ramps.keys() == up.ramps.keys()
    r = replaced.ramps[params.SCALING_NONLINEARITY]
    assert float(r.at_time(RAMP)) == pytest.approx(RO)


def test_replace_reresolves_when_ramps_given(up):
    replaced = up.replace(ramps={params.SCALING_NONLINEARITY: (0.0, 0.3)})
    r = replaced.ramps[params.SCALING_NONLINEARITY]
    assert float(r.at_time(RAMP)) == pytest.approx(0.3)


def test_replace_overrides_steps(up):
    assert up.replace(steps=7).steps == 7


# ================================================================
#  Structure, cost, repr, trace guard
# ================================================================
def test_owns_an_internal_propagator(up):
    assert isinstance(up.propagator, Propagator)
    assert up.model is up.propagator.model


def test_is_endo(up):
    assert up.domain == up.codomain


def test_is_down_flag(up):
    assert up.is_down is False
    assert up.down.is_down is True
    assert up.down.down.is_down is False


def test_cost_reports_internal_steps(up):
    assert up.cost().model_steps == 20


def test_repr_reports_config(up):
    text = repr(up)
    assert "AdiabaticRamping(" in text
    assert "ramp_steps=20" in text
    assert "backward=False" in text


def test_trace_guard_raises_on_a_tracer(up, z0):
    assert up.traceable is False
    with pytest.raises(TraceError, match="Tier-2"):
        jax.jit(up)(z0)
