"""Tests for the Tier-2 ``TimeAverage`` preset (wave 7 B).

A linear inertial oscillation is a pure wave mode: averaging over an
inertial period removes it. The tests exercise the wave removal, the
nested descending-period plan, the ``fr.terms.linear`` parity delta
(Smagorinsky/advection dropped), the period=None inertial default, and
the Tier-2 trace guard.

The block at the bottom widens the coverage off the 1-D toy onto
assembled ``sw2``/``nh2`` models (periodic and channel): the wave
signature collapses, the geostrophic invariant survives, and the
projected state is a fixed point.
"""
import jax
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.model import term_predicates as terms
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.model.transforms.errors import TraceError
from fridom.model.transforms.norms import _l2_norm
from fridom.model.transforms.time_average import TimeAverage

from .conftest import (
    F0,
    Coriolis,
    F0Provider,
    NonlinearU,
    make_model,
)

PERIOD = 2 * 3.141592653589793 / F0


# ================================================================
#  Killing a pure wave mode
# ================================================================
def test_time_average_kills_a_pure_wave_mode(toy_model, toy_state):
    before = float(_l2_norm(toy_state))
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=2)
    after = float(_l2_norm(ta(toy_state)))
    # the rotating signal is averaged away over the inertial period
    assert after < before / 100


def test_second_pass_does_not_regrow_the_wave(toy_model, toy_state):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1)
    once = ta(toy_state)
    twice = ta(once)
    # idempotent-ish: the near-balanced state does not grow back
    assert float(_l2_norm(twice)) <= float(_l2_norm(once)) * 2


# ================================================================
#  The period plan (descending vs flat)
# ================================================================
def test_equidistant_gives_descending_periods(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=2)
    # linspace(P/2, P, 3)[1:][::-1] -> descending, longest first
    assert ta.n_steps[0] > ta.n_steps[1]


def test_non_equidistant_gives_equal_periods(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=3,
                     equidistant=False)
    assert len(set(ta.n_steps)) == 1
    assert len(ta.n_steps) == 3


def test_backward_forward_runs_both_legs(toy_model, toy_state):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1,
                     backward_forward=True)
    out = ta(toy_state)
    # both legs executed -> twice the per-pass steps in the cost
    assert ta.cost().model_steps == 2 * sum(ta.n_steps)
    assert float(_l2_norm(out)) < float(_l2_norm(toy_state))


# ================================================================
#  The fr.terms.linear parity delta (drops the nonlinear term)
# ================================================================
def test_default_filter_drops_the_nonlinear_term():
    model = make_model(modules=(Coriolis(), NonlinearU()))
    ta = TimeAverage(model, period=PERIOD, n_ave=1)
    keys = {e.key for e in
            ta._forward._artifacts.schedule.kind_entries(None)}
    assert keys == {"Coriolis/cor"}  # NonlinearU/adv dropped


def test_explicit_filter_is_honored():
    model = make_model(modules=(Coriolis(), NonlinearU()))
    # an explicit filter that KEEPS the nonlinear term (unlike the
    # linear default) — both terms survive
    ta = TimeAverage(
        model, period=PERIOD, n_ave=1,
        filter=terms.named("Coriolis/cor", "NonlinearU/adv"))
    keys = {e.key for e in
            ta._forward._artifacts.schedule.kind_entries(None)}
    assert keys == {"Coriolis/cor", "NonlinearU/adv"}


# ================================================================
#  period=None -> the inertial period
# ================================================================
def test_period_none_reads_the_inertial_period(toy_model):
    ta_default = TimeAverage(toy_model, n_ave=1)
    model = make_model()
    ta_explicit = TimeAverage(model, period=PERIOD, n_ave=1)
    assert ta_default.n_steps == ta_explicit.n_steps


def test_period_none_without_coriolis_errors():
    model = make_model(modules=(Coriolis(),))  # no F0Provider
    with pytest.raises(ValueError, match=r"no 'coriolis\.f0'"):
        TimeAverage(model, n_ave=1)


def test_period_none_with_zero_f0_errors():
    model = make_model(modules=(Coriolis(), F0Provider(f0=0.0)))
    with pytest.raises(ValueError, match=r"coriolis\.f0 is zero"):
        TimeAverage(model, n_ave=1)


class _RampedF0Provider(F0Provider):

    """F0Provider that keeps a time-dependent (Ramp) f0 leaf."""

    def __init__(self, f0):
        # F0Provider coerces to an array; keep the Ramp pytree as-is
        self.f0 = f0


def test_period_none_with_time_dependent_f0_errors():
    # the inertial period 2*pi/f0 is not a single constant for a ramped
    # f0 -- a taught error, not a bare float(Ramp) TypeError
    ramp = fr.model.Ramp(1.0, 2.0, period=1.0)
    model = make_model(modules=(Coriolis(), _RampedF0Provider(ramp)))
    with pytest.raises(ValueError, match="time-dependent"):
        TimeAverage(model, n_ave=1)


# ================================================================
#  Construction guards
# ================================================================
def test_non_positive_period_errors(toy_model):
    with pytest.raises(ValueError, match="period must be positive"):
        TimeAverage(toy_model, period=0.0)


def test_invalid_n_ave_errors(toy_model):
    with pytest.raises(ValueError, match="n_ave must be a positive"):
        TimeAverage(toy_model, period=PERIOD, n_ave=0)


# ================================================================
#  Structure, cost, repr, trace guard
# ================================================================
def test_time_average_is_endo(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1)
    assert ta.domain == ta.codomain


def test_cost_is_the_total_step_count(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=2)
    assert ta.cost().model_steps == sum(ta.n_steps)


def test_repr_reports_the_plan(toy_model):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1)
    assert "TimeAverage(n_steps=" in repr(ta)


def test_trace_guard_raises_on_a_tracer(toy_model, toy_state):
    ta = TimeAverage(toy_model, period=PERIOD, n_ave=1)
    assert ta.traceable is False
    with pytest.raises(TraceError, match="Tier-2"):
        jax.jit(ta)(toy_state)


# ================================================================
#  Real models: shallow water (periodic and channel), nonhydro
#
#  Everything above runs on the 1-D 8-point toy of conftest.py. The
#  block below is the first evidence that the preset does its job on
#  an assembled model: the wave signature (divergence in sw, the
#  vertical velocity in nh) collapses, the geostrophic invariant
#  survives, and the projected state is a fixed point.
# ================================================================
SW_N = 16
SW_F0 = 4.0
SW_DT = 5e-3
NH_N = 8
NH_F0 = 4.0
NH_N2 = 16.0
NH_DT = 1e-2


def make_sw_model(*, periodic_y=True, f0=SW_F0):
    """Assemble a small dimensional rotating shallow-water model."""
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            SW_N, (0.0, 1.0), periodic=(name == "x" or periodic_y),
            name=name)
        for name in ("x", "y"))
    return sw.Model(
        # one device: the internal variant runs a plain host loop, but
        # the transforms suite pins single-device throughout
        grid=fr.spatial.Grid(meshes, device_ids=(0,)),
        core=sw.Core(gravity=1.0, depth=1.0),
        coriolis=sw.modules.FPlaneCoriolis(f0=f0),
        advection=True,
        time_stepper=AdamBashforth(SW_DT, order=3))


def sw_state(model, *, walled=False):
    """Seed a smooth unbalanced ``(u, v, p)`` state."""
    axis = (np.arange(SW_N) + 0.5) / SW_N
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    if walled:
        model.set_fields(
            u=np.sin(2 * np.pi * gx) * np.sin(np.pi * gy),
            p=0.1 * np.sin(2 * np.pi * gx) * np.cos(np.pi * gy))
    else:
        model.set_fields(
            u=np.sin(2 * np.pi * gx) * np.cos(2 * np.pi * gy),
            v=0.3 * np.cos(2 * np.pi * gx),
            p=0.1 * np.sin(2 * np.pi * (gx + gy)))
    return sw.State({c: model.state[c] for c in ("u", "v", "p")})


def sw_divergence(state):
    """Max |div u| — the shallow-water wave signature."""
    div = state["u"].diff("x") + state["v"].diff("y")
    return float(np.abs(np.asarray(div.data)).max())


def sw_linear_pv(state):
    """Linear shallow-water PV ``zeta - f0 p / (g H)`` (g = H = 1)."""
    zeta = state["v"].diff("x") - state["u"].diff("y")
    p = state["p"].to(zeta.function_space)
    return np.asarray((zeta - SW_F0 * p).data)


@pytest.fixture(scope="module")
def sw_projected():
    """``(before, once, twice)`` on a periodic shallow-water model."""
    model = make_sw_model()
    ta = TimeAverage(model, n_ave=2)
    before = sw_state(model)
    once = ta(before)
    return before, once, ta(once)


def test_sw_time_average_kills_the_divergent_wave_content(sw_projected):
    before, once, twice = sw_projected
    # the divergent (inertia-gravity) part is what the inertial-period
    # mean cancels; the balanced part is divergence-free
    assert sw_divergence(once) < sw_divergence(before) / 50
    assert sw_divergence(twice) < sw_divergence(once)


def test_sw_time_average_preserves_the_linear_pv(sw_projected):
    before, once, _ = sw_projected
    # the geostrophic invariant must survive untouched: this is the
    # claim that distinguishes a projection from mere damping
    q0, q1 = sw_linear_pv(before), sw_linear_pv(once)
    assert np.abs(q1 - q0).max() / np.abs(q0).max() < 1e-10


def test_sw_projected_state_is_a_fixed_point(sw_projected):
    _, once, twice = sw_projected
    n1, n2 = float(_l2_norm(once)), float(_l2_norm(twice))
    assert abs(n2 - n1) / n1 < 1e-3


def test_sw_time_average_runs_on_a_channel():
    # a walled meridional axis: the internal variant's ghost fills and
    # the accumulation both have to survive the boundary
    model = make_sw_model(periodic_y=False)
    ta = TimeAverage(model, n_ave=1)
    before = sw_state(model, walled=True)
    after = ta(before)
    assert np.isfinite(np.asarray(after["u"].data)).all()
    assert sw_divergence(after) < sw_divergence(before) / 5
    assert float(_l2_norm(after)) < float(_l2_norm(before))


@pytest.fixture(scope="module")
def nh_projected():
    """``(before, once, twice)`` on a small nonhydrostatic model."""
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            NH_N, (0.0, 2 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z"))
    model = nh.Model(
        core=nh.Core(),
        grid=fr.spatial.Grid(meshes, device_ids=(0,)),
        coriolis=nh.FPlaneCoriolis(f0=NH_F0),
        buoyancy=nh.ConstantStratification(n2=NH_N2),
        time_stepper=AdamBashforth(NH_DT, order=3))
    ta = TimeAverage(model, n_ave=2)
    axis = (np.arange(NH_N) + 0.5) * 2 * np.pi / NH_N
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    model.set_fields(
        u=0.2 * np.sin(gx) * np.cos(gy) * np.cos(gz),
        v=0.2 * np.cos(gx) * np.sin(gy) * np.cos(gz),
        w=0.1 * np.cos(gx) * np.cos(gy) * np.sin(gz),
        b=0.3 * np.cos(gx) * np.cos(gy) * np.cos(gz))
    before = nh.State({c: model.state[c] for c in ("u", "v", "w", "b")})
    once = ta(before)
    return before, once, ta(once)


def _peak(state, name):
    return float(np.abs(np.asarray(state[name].data)).max())


def test_nh_time_average_kills_the_vertical_velocity(nh_projected):
    before, once, _ = nh_projected
    # w is purely wave content in the geostrophic mode: the surviving
    # state must be (near) horizontally non-divergent. The factor is
    # resolution-limited (~40x at 8^3, ~350x at 16^3): the discrete
    # dispersion at the grid scale drifts off the continuous inertial
    # frequency, so the sinc zero does not land exactly
    assert _peak(once, "w") < _peak(before, "w") / 20


def test_nh_time_average_leaves_a_fixed_point(nh_projected):
    _, once, twice = nh_projected
    n1, n2 = float(_l2_norm(once)), float(_l2_norm(twice))
    assert abs(n2 - n1) / n1 < 1e-2
    assert _peak(twice, "w") <= _peak(once, "w")


def test_nh_time_average_keeps_the_balanced_buoyancy(nh_projected):
    before, once, _ = nh_projected
    # thermal wind: b belongs to the geostrophic mode, so unlike w it
    # must NOT be averaged away
    assert _peak(once, "b") > _peak(before, "b") / 4


# ================================================================
#  period=None on a nondimensional model
# ================================================================
def test_period_none_rejects_a_nondimensional_rotation():
    # the nondimensional spelling publishes 'coriolis.rossby', not
    # 'coriolis.f0' -- the documented period=None default only works
    # on a model whose rotation is spelled dimensionally
    meshes = tuple(
        fr.spatial.meshes.IntervalMesh(
            8, (0.0, 1.0), periodic=True, name=name)
        for name in ("x", "y"))
    model = sw.Model(
        grid=fr.spatial.Grid(meshes, device_ids=(0,)),
        core=sw.Core(froude_number=0.1, depth=1.0),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=0.1),
        advection=True,
        time_stepper=AdamBashforth(SW_DT, order=3))
    assert "coriolis.rossby" in dict(model.parameters)
    with pytest.raises(ValueError, match=r"no 'coriolis\.f0'"):
        TimeAverage(model, n_ave=1)
