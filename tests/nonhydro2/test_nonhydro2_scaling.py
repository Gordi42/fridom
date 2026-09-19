"""The nonhydro2 scaling surface (Branch 2): variants, errors, parity.

The prefix-mirrored scaling shard of ``test_nonhydro2.py``: the
dual-variant stratification, the scaling-neutral core, the retired
preset kwargs, the effective-number re-keys (diagnostics, eigenmodes,
energy) and the propagator autodiff gate through the nondim-ratio
paths.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.energy import EnergyMetric
from fridom.model.model import _chunk_body
from fridom.model.time_steppers.adam_bashforth import AdamBashforth

N = 8
DT = 2.0 ** -6
STEPS = 6
NAMES = ("u", "v", "w", "b")


def make_grid():
    im = fr.spatial.meshes.IntervalMesh
    return fr.spatial.Grid(tuple(
        im(N, (0.0, 2.0 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z")), device_ids=(0,))


def dim_model(**kwargs):
    """Build the dimensional twin of :func:`rot_model` (ro=1)."""
    return nh.Model(
        grid=make_grid(), core=nh.Core(aspect_ratio=0.5),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=4.0),
        advection=nh.CenteredAdvection(),
        time_stepper=AdamBashforth(DT, order=3), **kwargs)


def rot_model(*, ro=0.25, **kwargs):
    """Rotational-frame nondim model (parity: f0_eff=1, n2_eff=4)."""
    return nh.Model(
        grid=make_grid(), core=nh.Core(aspect_ratio=0.5),
        scaling=fr.scaling.Rotational(),
        coriolis=nh.FPlaneCoriolis(rossby_number=ro),
        buoyancy=nh.ConstantStratification(froude_number=ro / 2),
        advection=nh.CenteredAdvection(),
        time_stepper=AdamBashforth(DT, order=3), **kwargs)


def random_fields(model, amp=0.05, seed=7):
    rng = np.random.default_rng(seed)
    return {name: amp * rng.standard_normal(model.state[name].shape)
            for name in NAMES}


# ================================================================
#  Retired preset kwargs and params (taught errors)
# ================================================================
@pytest.mark.parametrize(
    ("kwarg", "match"),
    [pytest.param({"dsqr": 0.25}, "dsqr= is retired", id="dsqr"),
     pytest.param({"rossby_number": 0.5},
                  "rossby_number= is retired", id="rossby"),
     pytest.param({"dt": 0.1}, "dt= is retired", id="dt"),
     pytest.param({"stratification": None},
                  "stratification= was renamed to buoyancy=",
                  id="stratification"),
     pytest.param({"pressure_iterations": 5},
                  "moved onto the core", id="solver-kwarg"),
     pytest.param({"family": "fv"}, "moved onto the core",
                  id="family")])
def test_preset_teaches_the_retired_kwargs(kwarg, match):
    with pytest.raises(TypeError, match=match):
        nh.Model(advection=nh.CenteredAdvection(), grid=make_grid(),
                 core=nh.Core(),
                 time_stepper=AdamBashforth(DT, order=3), **kwarg)


def test_params_teach_the_retired_dsqr_name():
    with pytest.raises(AttributeError, match="ASPECT_RATIO"):
        _ = nh.params.DSQR


# ================================================================
#  Stratification dual kwargs (fr.scaling variants)
# ================================================================
def test_stratification_takes_exactly_one_kwarg_set():
    with pytest.raises(TypeError, match="exactly one kwarg set"):
        nh.ConstantStratification()
    with pytest.raises(TypeError, match="exactly one kwarg set"):
        nh.ConstantStratification(n2=1.0, froude_number=0.5)
    with pytest.raises(TypeError, match="froude_number=0"):
        nh.ConstantStratification(froude_number=0.0)


def test_stratification_variant_provides():
    dim = nh.ConstantStratification(n2=2.0)
    assert dim.scaling_variant == "dimensional"
    assert [d.name for d in dim.parameter_declarations] == [
        "stratification.n2"]
    nondim = nh.ConstantStratification(froude_number=0.5)
    assert nondim.scaling_variant == "nondimensional"
    assert [d.name for d in nondim.parameter_declarations] == [
        "stratification.froude"]
    assert nondim.scaling_mechanism == "internal_wave"


def test_meridional_stratification_is_pinned_dimensional():
    # a nondim assembly with the varying profile is a taught refusal
    # (mixed variants), never a silent misread of the n2(y) profile
    module = nh.MeridionalStratification(lambda y: 1.0 + 0.0 * y)
    assert module.scaling_variant == "dimensional"
    with pytest.raises(fr.model.errors.AssemblyError,
                       match="MIXED scaling variants"):
        nh.Model(
            advection=nh.CenteredAdvection(),
            grid=make_grid(), core=nh.Core(),
            scaling=fr.scaling.Rotational(),
            coriolis=nh.FPlaneCoriolis(rossby_number=0.25),
            buoyancy=module,
            time_stepper=AdamBashforth(DT, order=3))


def test_buoyancy_is_opt_in_on_the_preset():
    # buoyancy=None installs NO buoyancy module at all
    model = nh.Model(
        advection=nh.CenteredAdvection(),
        grid=make_grid(), core=nh.Core(),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        time_stepper=AdamBashforth(DT, order=3))
    assert "b" not in model.state.component_names
    assert "stratification.n2" not in model.parameters


def test_core_defaults_to_a_plain_core():
    # core=None (the default) is a plain nh.Core(): the assembly
    # carries the aspect-ratio provide at its default value 1
    model = nh.Model(
        advection=nh.CenteredAdvection(),
        grid=make_grid(),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        time_stepper=AdamBashforth(DT, order=3))
    assert model.module(nh.Core).aspect_ratio == 1.0
    assert "w" in model.state.component_names


def test_internal_wave_frame_self_normalizes():
    # under InternalWave() the restoring ratio (eps/Fr)^2 is an exact
    # 1.0 (alias row: one leaf) — the nondim model runs finite
    model = nh.Model(
        grid=make_grid(), core=nh.Core(aspect_ratio=0.5),
        scaling=fr.scaling.InternalWave(),
        buoyancy=nh.ConstantStratification(froude_number=0.25),
        advection=nh.CenteredAdvection(),
        time_stepper=AdamBashforth(DT, order=3))
    eps = model.parameters[fr.model.params.SCALING_NONLINEARITY]
    froude = model.parameters[
        fr.model.params.STRATIFICATION_FROUDE]
    assert float(eps) == float(froude) == 0.25
    model.set_fields(**random_fields(model))
    model.advance(2)
    for name in NAMES:
        assert np.all(np.isfinite(np.asarray(model.state[name].data)))


# ================================================================
#  Today-parity: the Rotational spelling equals the dim twin
# ================================================================
def _advanced_op_by_op(model, steps):
    """Advance ``steps`` with the pure step kernel evaluated op by op.

    Bitwise claims hold only between identically compiled paths (model
    spec, ``run()``): two DIFFERENT assemblies are two different XLA
    programs, whose fusions contract different mul/add pairs into FMAs,
    so their compiled runs agree to the last bit on some CPUs and
    differ by one ulp on others (CI flake 2026-09-19; reproduced with
    ``XLA_FLAGS=--xla_cpu_use_fusion_emitters=false``). A parity claim
    between two assemblies is a claim about ARITHMETIC, so it is
    checked where there is no fusion to differ.
    """
    with jax.disable_jit():
        return _chunk_body(model._artifacts.record, steps,
                           model._carry, model._stepper).state


def test_rotational_parity_step_is_bitwise():
    # ro=1: eps = Ro = 1 aliased -> every live ratio is exactly 1.0,
    # so the nondim trace reproduces the dimensional one bitwise
    dim = dim_model()
    rot = rot_model(ro=1.0)
    fields = random_fields(dim)
    dim.set_fields(**fields)
    rot.set_fields(**fields)
    dim_state = _advanced_op_by_op(dim, 3)
    rot_state = _advanced_op_by_op(rot, 3)
    for name in NAMES:
        assert np.array_equal(np.asarray(dim_state[name].data),
                              np.asarray(rot_state[name].data)), name
    assert np.abs(np.asarray(dim_state["u"].data)).max() > 0.0


def test_diagnostics_re_key_on_the_effective_numbers():
    # ekin uses delta squared, epot the live internal-wave ratio and
    # linear_pot_vort the effective rotation -- all evaluate
    # identically on the dim twin and the ro=1 Rotational spelling
    dim = dim_model()
    rot = rot_model(ro=1.0)
    fields = random_fields(dim)
    dim.set_fields(**fields)
    rot.set_fields(**fields)
    for name in ("ekin", "epot", "linear_pot_vort", "b_total"):
        a = np.asarray(getattr(dim.diagnostics, name)().data)
        b = np.asarray(getattr(rot.diagnostics, name)().data)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_eigenmodes_re_key_on_the_effective_numbers():
    # the analytic eigenmodes assemble f0_eff = eps/Ro, n2_eff =
    # (eps/Fr)^2 and dsqr = delta^2 from the nondim primitives
    dim = nh.eigenbasis(dim_model())
    rot = nh.eigenbasis(rot_model(ro=1.0))
    assert rot.f0 == pytest.approx(dim.f0)
    assert rot.n2 == pytest.approx(dim.n2)
    assert rot.dsqr == pytest.approx(dim.dsqr)


def test_energy_metric_re_keys_on_the_effective_numbers():
    dim = EnergyMetric.from_model(dim_model())
    rot = EnergyMetric.from_model(rot_model(ro=1.0))
    assert dict(rot.weights) == dict(dim.weights)


# ================================================================
#  Autodiff through the nondim-ratio step path (propagator gate)
# ================================================================
def test_ic_grad_through_the_nondim_ratio_path_matches_fd():
    model = rot_model(ro=0.25)
    model.set_fields(**random_fields(model))
    run = model.propagator(wrt=("u",), steps=STEPS)
    u0 = model._carry.state["u"].storage

    def loss(field):
        final = run((field,))
        return sum(jnp.sum(final.state[c].data ** 2) for c in NAMES)

    grad = np.asarray(jax.grad(loss)(u0))
    assert bool(np.all(np.isfinite(grad)))
    rng = np.random.default_rng(0)
    direction = jnp.asarray(rng.standard_normal(u0.shape),
                            dtype=u0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u0 + eps * direction))
          - float(loss(u0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)
