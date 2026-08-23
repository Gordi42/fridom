r"""``Model(phases=...)``: the staggered step end to end.

Prefix-mirrored shard of ``test_model.py`` for the phase axis. The
load-bearing claims, each with a test below:

- **Parity, phases off.** A 10-step hydrostatic run, a nonhydro2 run
  and a moving-geometry run are BITWISE identical with ``phases=None``
  and with ``phases=fr.model.Phases.total()`` — the one-group
  partition IS the unphased path (same schedule token, same trace).
- **Exactness when nothing couples the groups.** With ``n2 = 0`` and
  no advection the tracer group's tendency is identically zero, so the
  staggered run reproduces the unphased run bitwise on every
  PROGNOSTIC field. (The DIAGNOSTIC ``w`` / ``p_hyd`` legitimately
  differ: S1' re-runs per phase, so the tracer phase re-diagnoses them
  from the momentum phase's advanced velocities. That re-diagnosis is
  the whole point of the axis.)
- **The staggered toy runs and compiles once** across an amplitude
  sweep (the geometry of the trace is static; only values sweep).
- **Differentiability.** ``Model.propagator`` grad through a phased
  run matches a central FD to rtol 1e-4, and ``remat=True`` agrees.
- **Device-count invariance** of a phased run (forced-4).
- The Model surface: the report line, the fingerprint rows,
  ``variant()`` pass-through, and the taught refusal under a stepper
  that cannot loop.

Self-contained per the AGENTS oversized-module rule.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
import fridom.nonhydro2 as nh
from fridom.model.errors import AssemblyError
from fridom.model.modules.moving_geometry import (
    MeshVelocityCorrection,
    MovingGeometry,
)
from fridom.model.params import TIME_STEP
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.model.time_steppers.runge_kutta import LowStorageRK3
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
NZ = 4
DT = 1e-3
STEPS = 10
STAGGERED = fr.model.Phases.staggered()


# ================================================================
#  Builders
# ================================================================
def hy_grid(device_ids=None):
    """Return a doubly-periodic horizontal / bounded-vertical grid."""
    return Grid((
        IntervalMesh(N, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(N, (0.0, 1.0), periodic=True, name="y"),
        IntervalMesh(NZ, (0.0, 1.0), periodic=False, name="z")),
        device_ids=device_ids)


def hy_model(phases=None, *, n2=1.0, advection=True, dt=DT,
             device_ids=None, stepper=None):
    """Assemble a small hydrostatic model with an implicit surface."""
    return hy.Model(
        grid=hy_grid(device_ids),
        core=hy.Core(gravity=4.0),
        time_stepper=stepper or AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        buoyancy=hy.ConstantStratification(n2=n2),
        free_surface=hy.ImplicitFreeSurface(),
        advection=(fr.model.modules.CenteredAdvection()
                   if advection else None),
        phases=phases)


def seed(model, amplitude=1.0, *, tracer=True):
    """Seed a smooth velocity (and optionally buoyancy) pattern."""
    def wave(x, y, z):
        return amplitude * np.sin(2 * np.pi * x) * np.cos(
            2 * np.pi * y) * (1.0 + 0.1 * z)

    fields = {"u": wave, "b": wave} if tracer else {"u": wave}
    model.set_fields(**fields)
    return model


def prognostic(model):
    """Host copy of every PROGNOSTIC component."""
    names = model.field_table.prognostic
    return {name: np.asarray(model.state[name].data)
            for name in names}


def run(model, steps=STEPS, **seed_kwargs):
    """Seed and advance; return the PROGNOSTIC host arrays."""
    seed(model, **seed_kwargs)
    model.advance(steps)
    return prognostic(model)


def nh_model(phases=None):
    """Assemble a tiny nonhydro2 model."""
    grid = Grid(tuple(
        IntervalMesh(6, (0.0, 1.0), periodic=(name != "z"), name=name)
        for name in ("x", "y", "z")))
    return nh.Model(
        grid=grid, core=nh.Core(),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=1.0),
        advection=fr.model.modules.CenteredAdvection(),
        phases=phases)


def moving_model(phases=None):
    """Assemble the zero-physics moving-geometry nonhydro2 model."""
    def schedule(x, t):
        return 0.8 * (1.0 + 0.1 * jnp.sin(x)) + 0.05 * t

    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: schedule(x, 0.0)})
    grid = Grid((
        IntervalMesh(6, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(6, (0.0, 2 * np.pi), periodic=True, name="y"),
        IntervalMesh(6, (0.0, 1.0), periodic=False, name="z")),
        mapping=mapping)
    return nh.Model(
        grid=grid, core=nh.Core(family="nodal"),
        time_stepper=AdamBashforth(1e-2, order=3),
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        buoyancy=nh.ConstantStratification(n2=0.0),
        advection=None,
        modules_extra=(MovingGeometry({"H": schedule}),
                       MeshVelocityCorrection()),
        phases=phases)


def bitwise(left, right):
    """Assert every named array is bit-for-bit equal."""
    for name, want in left.items():
        assert np.array_equal(right[name], want), name


# ================================================================
#  Gate 1 — bitwise parity with the axis off
# ================================================================
def test_total_is_bitwise_the_unphased_hydrostatic_run():
    bitwise(run(hy_model(None)),
            run(hy_model(fr.model.Phases.total())))


def test_total_is_bitwise_the_unphased_nonhydro_run():
    bitwise(run(nh_model(None)), run(nh_model(fr.model.Phases.total())))


def test_total_is_bitwise_the_unphased_moving_geometry_run():
    bitwise(run(moving_model(None), steps=4, tracer=False),
            run(moving_model(fr.model.Phases.total()), steps=4,
                tracer=False))


def test_total_shares_the_unphased_assembly_record():
    plain = hy_model(None)
    total = hy_model(fr.model.Phases.total())
    assert plain.fingerprint.digest == total.fingerprint.digest
    assert len(total.phases) == 1


# ================================================================
#  Gate 4 — the staggered toy
# ================================================================
def test_the_staggered_partition_is_momentum_then_tracers():
    model = hy_model(STAGGERED)
    assert model.phases == (
        frozenset({"u", "v", "ps"}), frozenset({"b"}))


def test_a_staggered_run_stays_finite():
    state = run(hy_model(STAGGERED))
    assert all(np.isfinite(value).all() for value in state.values())


def test_the_staggered_run_differs_from_the_unphased_one():
    # the point of the axis: the tracer phase transports with the
    # POST-solve velocities, so the two runs are not the same scheme
    phased = run(hy_model(STAGGERED))
    plain = run(hy_model(None))
    assert not np.array_equal(phased["b"], plain["b"])
    assert np.abs(phased["b"] - plain["b"]).max() < 1e-2


def test_a_decoupled_staggered_run_is_bitwise_the_unphased_one():
    # THE EXACTNESS CONDITION, stated: the tracer group's tendency
    # (n2 * w, with n2 = 0) is identically zero and no advection
    # couples b to u, v; b starts at zero, so p_hyd -- and hence the
    # momentum tendency -- is zero in both runs. Under those
    # conditions phase 1 changes nothing and the split is exact on
    # every PROGNOSTIC field. (w and p_hyd are DIAGNOSTIC and are
    # re-diagnosed in phase 1 from the advanced u, v -- that
    # difference is the axis working as designed.)
    plain = run(hy_model(None, n2=0.0, advection=False),
                tracer=False)
    phased = run(hy_model(STAGGERED, n2=0.0, advection=False),
                 tracer=False)
    bitwise(plain, phased)
    assert np.abs(plain["b"]).max() == 0.0


def test_a_staggered_run_compiles_once_across_an_amplitude_sweep(
        compile_counter):
    model = hy_model(STAGGERED)
    seed(model, 1.0)
    model.advance(4)
    seed(model, 2.0)
    model.advance(4)
    compile_counter.reset()
    seed(model, 3.0)
    model.advance(4)
    assert compile_counter.count == 0
    assert not model.panicked


# ================================================================
#  Gate 6 — differentiability through a phased run
# ================================================================
def _loss_of(final):
    return sum(jnp.sum(field.data ** 2) for field in final.state)


def _fd(loss, x0, eps=1e-4):
    h = eps * abs(float(x0))
    return (float(loss(x0 + h)) - float(loss(x0 - h))) / (2.0 * h)


@pytest.mark.parametrize("remat", [False, True],
                         ids=["plain", "remat"])
def test_propagator_grad_through_a_phased_run_matches_fd(remat):
    model = seed(hy_model(STAGGERED), 0.5)
    # the stepper leaf: dt threads the per-phase AB weights AND the
    # barotropic solve's stage_dt, so it exercises the whole loop
    step = model.propagator(wrt=(TIME_STEP,), steps=4, remat=remat)

    def loss(dt):
        return _loss_of(step((dt,)))

    dt0 = jnp.asarray(DT)
    grad = float(jax.grad(loss)(dt0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0
    assert grad == pytest.approx(_fd(loss, dt0), rel=1e-4)


# ================================================================
#  Gate 7 — device-count invariance
# ================================================================
@pytest.mark.multi_device
def test_a_phased_run_is_device_count_invariant():
    # to tight rounding, not bitwise (the C2/C3 precedent): the
    # sharded chains and the barotropic solve fuse differently
    many = run(hy_model(STAGGERED))
    one = run(hy_model(STAGGERED, device_ids=(0,)))
    for name, want in one.items():
        np.testing.assert_allclose(many[name], want, rtol=0.0,
                                   atol=1e-11)


# ================================================================
#  The Model surface
# ================================================================
def test_the_report_names_the_groups():
    report = hy_model(STAGGERED).report
    assert "phases: 2 groups" in report.header
    assert "phases (2 groups)" in report.section("schedule")


def test_the_unphased_report_gains_no_phase_line():
    assert "phases:" not in hy_model(None).report.header


def test_the_fingerprint_records_the_groups():
    phased = hy_model(STAGGERED)
    plain = hy_model(None)
    rows = dict(phased.fingerprint.source)
    assert rows["phase 0"] == "ps, u, v"
    assert rows["phase 1"] == "b"
    assert phased.fingerprint.digest != plain.fingerprint.digest


def test_variant_inherits_the_phase_axis():
    variant = hy_model(STAGGERED).variant(
        term_filter=fr.model.term_predicates.linear)
    assert variant.phases == (
        frozenset({"u", "v", "ps"}), frozenset({"b"}))


def test_a_non_multistep_stepper_refuses_the_axis():
    with pytest.raises(AssemblyError, match="supports_phases=False"):
        hy_model(STAGGERED, stepper=LowStorageRK3(DT))


def test_an_unphased_model_accepts_every_stepper():
    model = hy_model(None, stepper=LowStorageRK3(DT))
    assert len(model.phases) == 1
