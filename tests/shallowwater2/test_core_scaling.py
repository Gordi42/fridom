"""The sw.Core scaling surface: variants, thickness, ramps, autodiff.

Prefix-mirrored shard of ``sw/modules/core.py`` (with commit-7 gates
of the nondimensionalization plan): today-parity between the
dimensional and GravityWave-scaled spellings, the DIAGNOSE-stage
thickness (schedule placement + restart), Ramp g/D against
re-assembled constants, the eigen/energy effective-number branches,
and two propagator-based autodiff regressions (thickness surface and
the nondimensional live-ratio path).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.stages import StageKind

from .conftest import make_grid, make_model

N = 16
DT = 5e-3
STEPS = 6
NAMES = ("u", "v", "p")


def dim_model(grid=None, *, gravity=1.0, depth=1.0, f0=0.5,
              advection=True, dt=DT):
    """Return a DIMENSIONAL model (no scaling argument at all)."""
    if grid is None:
        grid = make_grid()
    return sw.Model(
        grid=grid,
        core=sw.Core(gravity=gravity, depth=depth),
        coriolis=sw.modules.FPlaneCoriolis(f0=f0),
        advection=advection,
        time_stepper=fr.model.time_steppers.AdamBashforth(
            dt, order=3))


def random_fields(model, seed=42, amp=0.1):
    rng = np.random.default_rng(seed)
    return {
        "u": amp * rng.standard_normal(model.state["u"].shape),
        "v": amp * rng.standard_normal(model.state["v"].shape),
        "p": 0.3 * amp * rng.standard_normal(model.state["p"].shape)}


# ================================================================
#  Variant surface
# ================================================================
def test_dimensional_needs_no_scaling_argument():
    model = dim_model()
    assert isinstance(model.scaling, fr.scaling.Dimensional)
    assert fr.model.params.SCALING_NONLINEARITY not in model.parameters
    assert float(model.parameters["shallowwater.gravity"]) == 1.0
    assert float(model.parameters["shallowwater.depth"]) == 1.0


def test_nondim_core_under_dimensional_default_is_taught():
    with pytest.raises(fr.model.errors.AssemblyError,
                       match="nondimensional scaling policy"):
        sw.Model(
            grid=make_grid(),
            core=sw.Core(froude_number=0.2),
            time_stepper=fr.model.time_steppers.AdamBashforth(DT))


def test_dim_equals_nondim_parity_run_bitwise():
    # the heart of the refactor: the GravityWave-scaled spelling's
    # live ratios self-normalize, so a matched dimensional run
    # (epsilon = 1 <-> the dim trace with no scaling ops) and the
    # nondim run at froude=1 agree bitwise over a nonlinear run
    grid = make_grid()
    dim = dim_model(grid, gravity=1.0, depth=0.5, f0=0.5)
    nondim = make_model(grid, csqr=0.5, rossby_number=1.0, f0=0.5)
    fields = random_fields(dim)
    dim.set_fields(**fields)
    nondim.set_fields(**fields)
    dim.advance(STEPS)
    nondim.advance(STEPS)
    for name in NAMES:
        assert np.array_equal(np.asarray(dim.state[name].data),
                              np.asarray(nondim.state[name].data))


# ================================================================
#  The thickness DIAGNOSE stage
# ================================================================
def test_thickness_is_a_diagnose_stage_before_the_terms():
    core = sw.Core(froude_number=0.25)
    kinds = [stage.kind for stage in core.stages]
    assert StageKind.DIAGNOSE in kinds
    # the composed schedule places DIAGNOSE (S1') before every term
    model = make_model()
    described = model._artifacts.schedule.describe()
    assert "thickness" in described


def test_thickness_field_is_diagnostic_on_the_centre_space():
    model = make_model()
    record = model.field_table["thickness"]
    assert record.lifecycle is fr.model.Lifecycle.DIAGNOSTIC
    assert (record.space
            is model.field_table["p"].space)


def test_restart_reproduces_the_continuous_run(tmp_path):
    # the DIAGNOSTIC thickness never enters the restart contract as
    # a stale value: the first substage after load recomputes it, so
    # snapshot -> load -> advance is bitwise the continuous run
    grid = make_grid()
    whole = make_model(grid, csqr=1.0, rossby_number=0.25, f0=0.5)
    split = make_model(grid, csqr=1.0, rossby_number=0.25, f0=0.5)
    fields = random_fields(whole)
    whole.set_fields(**fields)
    split.set_fields(**fields)
    whole.advance(6)
    split.advance(2)
    split.snapshot(tmp_path / "snap")
    resumed = make_model(grid, csqr=1.0, rossby_number=0.25, f0=0.5)
    resumed.load_snapshot(tmp_path / "snap")
    resumed.advance(4)
    for name in NAMES:
        assert np.array_equal(np.asarray(whole.state[name].data),
                              np.asarray(resumed.state[name].data))


# ================================================================
#  Ramp g / D against re-assembled constants
# ================================================================
@pytest.mark.parametrize("which", ["gravity", "depth"])
def test_ramped_g_and_d_match_the_reassembled_constant(which):
    ramp = fr.model.Ramp(1.0, 2.0, period=0.05, curve="exp")
    values = {"gravity": 1.0, "depth": 0.5}
    ramped_kwargs = dict(values)
    ramped_kwargs[which] = ramp
    grid = make_grid()
    ramped = sw.Model(
        grid=grid, core=sw.Core(**ramped_kwargs),
        coriolis=sw.modules.FPlaneCoriolis(f0=0.5),
        time_stepper=fr.model.time_steppers.AdamBashforth(DT))
    fields = random_fields(ramped)
    ramped.set_fields(**fields)
    z = sw.State({c: ramped.state[c] for c in NAMES})
    for t in (0.0, 0.02, 0.05):
        const_kwargs = dict(values)
        const_kwargs[which] = float(ramp.at_time(t))
        const = sw.Model(
            grid=grid, core=sw.Core(**const_kwargs),
            coriolis=sw.modules.FPlaneCoriolis(f0=0.5),
            time_stepper=fr.model.time_steppers.AdamBashforth(DT))
        const.set_fields(**{c: np.asarray(z[c].data) for c in NAMES})
        zc = sw.State({c: const.state[c] for c in NAMES})
        got = ramped.tendency(z, t=t)
        want = const.tendency(zc)
        for c in NAMES:
            np.testing.assert_allclose(
                np.asarray(got[c].data), np.asarray(want[c].data),
                rtol=1e-12, atol=1e-13)


def test_ramped_csqr_marks_the_field_time_dependent():
    core = sw.Core(gravity=fr.model.Ramp(1.0, 2.0, period=1.0),
                   depth=0.5)
    decls = {d.name: d for d in core.field_declarations}
    assert decls["csqr"].time_dependent
    kinds = [stage.kind for stage in core.stages]
    assert StageKind.SELF_UPDATE in kinds


# ================================================================
#  Eigen / energy effective-number branches
# ================================================================
def test_eigenmodes_agree_across_the_variants():
    # dim (g*D) and nondim ((eps/Fr)^2 D-tilde) spell the same
    # effective numbers, so the analytic spectra agree bitwise
    grid = make_grid()
    dim = dim_model(grid, gravity=2.0, depth=0.5, f0=1.5,
                    advection=False)
    nondim = sw.Model(
        grid=grid, core=sw.Core(froude_number=0.4),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(
            rossby_number=0.4 / 1.5),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT))
    em_dim = sw.eigenmodes.from_model(dim)
    em_nondim = sw.eigenmodes.from_model(nondim)
    for s in (-1, 0, 1):
        np.testing.assert_allclose(
            np.asarray(em_dim.omega(s).data),
            np.asarray(em_nondim.omega(s).data),
            rtol=1e-15)


def test_energy_metric_effective_weights_across_the_variants():
    grid = make_grid()
    dim = dim_model(grid, gravity=2.0, depth=2.0, f0=1.0,
                    advection=False)
    nondim = sw.Model(
        grid=grid, core=sw.Core(froude_number=0.5, depth=4.0),
        scaling=fr.scaling.GravityWave(),
        coriolis=sw.modules.FPlaneCoriolis(rossby_number=0.5),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT))
    m_dim = fr.model.EnergyMetric.from_model(dim)
    m_nondim = fr.model.EnergyMetric.from_model(nondim)
    assert m_dim.weights["p"] == 0.25
    assert m_nondim.weights["p"] == 0.25


def test_variable_depth_still_refuses_the_analytic_eigenmodes():
    model = sw.Model(
        grid=make_grid(),
        core=sw.Core(gravity=1.0,
                     depth=lambda y: 1.0 + 0.5 * y),
        coriolis=sw.modules.FPlaneCoriolis(
            f0=1.0, metric_weight="csqr"),
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(DT))
    with pytest.raises(ValueError,
                       match="constant squared phase speed"):
        sw.eigenmodes.from_model(model)


# ================================================================
#  Autodiff (policy): propagator-based, vs central FD
# ================================================================
def _directional_check(model, steps=STEPS):
    run = model.propagator(wrt=("p",), steps=steps)
    p0 = model._carry.state["p"].storage

    def loss(field):
        final = run((field,))
        return sum(jnp.sum(final.state[c].data ** 2) for c in NAMES)

    grad = np.asarray(jax.grad(loss)(p0))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(p0.shape),
                            dtype=p0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(p0 + eps * direction))
          - float(loss(p0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


def test_ic_grad_through_the_thickness_surface_matches_fd():
    # the nonlinear Sadourny path consumes the DIAGNOSE-stage
    # thickness; the gradient through it stays finite and FD-exact
    model = make_model(csqr=1.0, rossby_number=0.25, f0=0.5)
    model.set_fields(**random_fields(model, amp=0.05))
    _directional_check(model)


def test_ic_grad_through_the_nondim_ratio_path_matches_fd():
    # route A on the nondim surface: the correction multiplies the
    # (full - linear) difference by the live epsilon/Ro ratio — the
    # gradient through the ratio ops stays finite and FD-exact
    model = make_model(
        csqr=1.0, rossby_number=0.25, f0=0.5,
        modules_extra=(sw.modules.CoriolisEnergyCorrection(),))
    model.set_fields(**random_fields(model, amp=0.05))
    _directional_check(model)
