"""
Dynamic metrics + optional ALE (stage C4, decision CS-D4).

The moving-geometry gates: frozen motion reproduces the static C3
run BITWISE (with and without the ALE module), the manufactured ALE
sign test (a physically constant field under pure geometry motion
stays put with the correction and stays frozen at the computational
nodes without it), the sloped-to-flat northern-boundary morph (the
owner's target experiment: stable with ALE, mapped divergence at
solver tolerance every step, the exactly-conserved discrete volume,
a documented tracer-content bound; runs to completion without ALE),
and the oscillating terrain-following column (stable, divergence at
tolerance, compile-once across the whole run while the geometry
values sweep, forced-4 device-count invariance).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.modules.moving_geometry import (
    MeshVelocityCorrection,
    MovingGeometry,
    mapping_params,
)
from fridom.nonhydro2.modules.mapped_pressure import (
    MappedPressureSolver,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.integrate import Integral

N = 8
DT = 0.01
#: PCG budget of the mapped projection in these gates. The model
#: default (30) is well past convergence here: measured worst mapped
#: divergence over the whole morph is 1.87e-14 at BOTH 16 and 30
#: iterations (gate 5e-11), and 1.5e-17 on the oscillating column at
#: both — i.e. the solve has converged to machine precision by 16, so
#: the budget only bought trace/compile time (the CG loop is
#: unrolled). Every assertion below keeps its original tolerance.
ITERATIONS = 16
DSQR = 0.5
TWO_PI = 2.0 * np.pi


def depth(x):
    """Smooth periodic water depth H(x) (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(x)


# ================================================================
#  Gate 1: frozen motion == static C3, bitwise, with/without ALE
# ================================================================
def make_terrain_model(*modules, n=N, init=depth, advection=True,
                       device_ids=None, family="nodal"):
    """Terrain-following model ``zp = z * H(x)`` (z in [0, 1]).

    ``family`` defaults to ``"nodal"``: the nodal gates below pin it
    explicitly to keep the static reference and the moving models
    like-for-like (auto would flip a moving-geometry model to FV since
    the 2026-07-17 ALE-on-FV closure). The FV counterparts pass
    ``family="fv"`` — the ALE mesh-velocity correction is family-aware,
    so moving geometry runs on FV wherever a static mapped one does.
    """
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": init})
    grid = Grid((
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"),
    ), mapping=mapping, device_ids=device_ids)
    return nh.Model(grid=grid, dt=DT, dsqr=DSQR, family=family,
                    coriolis=nh.FPlaneCoriolis(f0=1.0),
                    advection=advection, modules_extra=modules,
                    pressure_iterations=ITERATIONS)


def terrain_fields(n=N):
    hor = (np.arange(n) + 0.5) * (TWO_PI / n)
    ver = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    return {
        "u": 0.01 * np.sin(y) * np.cos(np.pi * z),
        "v": 0.01 * np.sin(x),
        "b": 0.01 * np.cos(x) * np.cos(np.pi * z),
    }, (x, y, z)


def test_frozen_motion_reproduces_the_static_run_bitwise():
    # the schedule freezes the static default (H_dot == 0 exactly
    # through the jvp), so the dynamic pipeline — MovingGeometry
    # state fields, params= threading through every metric
    # derivation, the ALE correction term — must reproduce the
    # static C3 assembly BITWISE: same parameter values, same
    # derivation arithmetic, and an exact-zero ALE tendency
    static = make_terrain_model()
    frozen = MovingGeometry(
        {"H": lambda x, t: depth(x) + 0.0 * t})
    without_ale = make_terrain_model(frozen)
    with_ale = make_terrain_model(
        MovingGeometry({"H": lambda x, t: depth(x) + 0.0 * t}),
        MeshVelocityCorrection())
    fields, _ = terrain_fields()
    for model in (static, without_ale, with_ale):
        model.set_fields(**fields)
        model.advance(20)
    for c in ("u", "v", "w", "b", "p"):
        want = np.asarray(static.state[c].data)
        assert np.array_equal(
            np.asarray(without_ale.state[c].data), want), c
        assert np.array_equal(
            np.asarray(with_ale.state[c].data), want), c


# ================================================================
#  Gate 2: the manufactured ALE sign test
# ================================================================
H0 = 0.8
RATE = -0.4


def ale_profile(zp):
    """Smooth physical buoyancy profile b(z_phys)."""
    return np.cos(np.pi * zp)


def make_ale_model(n, *modules, family="nodal"):
    """Zero-physics column with a time-only depth H(t).

    8 points per horizontal axis: a 4-wide sharded axis trips a
    PRE-EXISTING XLA spmd-partitioner fault in the (static, C3)
    mapped projection under the forced-4 device suite (a c64/c128
    scalar-constant mix after partitioning) — unrelated to the
    dynamics under test here. ``family`` defaults nodal; the FV
    convergence gate passes ``family="fv"``.
    """
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: H0 + 0.0 * x})
    grid = Grid((
        IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"),
    ), mapping=mapping)
    return nh.Model(
        grid=grid, dt=DT, advection=False, family=family,
        pressure_iterations=ITERATIONS,
        coriolis=nh.FPlaneCoriolis(f0=0.0),
        stratification=nh.ConstantStratification(n2=0.0),
        modules_extra=(
            MovingGeometry({"H": lambda t: H0 + RATE * t}),
            *modules))


def ale_initial(n):
    ver = (np.arange(n) + 0.5) / n
    _, _, z = np.meshgrid(np.zeros(8), np.zeros(8), ver,
                          indexing="ij")
    return ale_profile(z * H0), z


@pytest.mark.single_device
def test_ale_keeps_the_physical_interpretation_in_place():
    # single_device: the tall columns (8 x 8 x 16/32) trip a
    # PRE-EXISTING XLA spmd fault in the (static, C3) mapped
    # projection whenever the column outsizes the sharded axis
    # (fft_thunk layout RET_CHECK; reproduced on dev-equivalent
    # static solves) — the C4 forced-4 gate rides the isotropic
    # oscillating-terrain tests below
    # zero physics, pure geometry motion H(t) = H0 + RATE t: with
    # the correction the field stays physically put — the value at
    # computational node z is the initial physical profile read at
    # the node's CURRENT physical position, b(z, T) = f(z * H(T)) —
    # the sign witness: a flipped correction would run away from
    # the remap at twice the speed. The depth SHRINKS (RATE < 0)
    # so every characteristic of the remap originates inside the
    # initial data — a growing domain has an inflow at the moving
    # top boundary whose missing data contaminates z > H0/H(T)
    # (measured, boundary-layer error floor ~2e-2 there). 2nd
    # order in the column resolution (dt fixed and small; measured
    # errors 1.75e-3 at n = 16, 4.34e-4 at n = 32, 1.14e-4 at
    # n = 64 — orders 2.01, 1.93).
    steps = 50
    t_final = steps * DT
    errors = []
    for n in (16, 32):
        model = make_ale_model(n, MeshVelocityCorrection(("b",)))
        b0, z = ale_initial(n)
        model.set_fields(b=b0)
        model.advance(steps)
        exact = ale_profile(z * (H0 + RATE * t_final))
        errors.append(np.abs(
            np.asarray(model.state["b"].data) - exact).max())
        # the geometry moved far enough that staying frozen would
        # be a gross error (the off-switch magnitude, gate below)
        assert np.abs(b0 - exact).max() > 50 * errors[-1]
    assert errors[0] / errors[1] > 3.5  # 2nd order (4x per level)


@pytest.mark.single_device
def test_without_ale_the_field_stays_frozen_at_the_nodes():
    # single_device: see the sign test above
    # the CS-D4 off switch: omitting the module leaves b with NO
    # tendency at all under pure geometry motion — bitwise frozen
    # at the computational nodes (the documented, deliberate,
    # physically-wrong-during-motion behavior)
    n = 16
    model = make_ale_model(n)
    b0, _ = ale_initial(n)
    model.set_fields(b=b0)
    model.advance(50)
    assert np.array_equal(np.asarray(model.state["b"].data), b0)


# ================================================================
#  Gate 3: the sloped-to-flat northern-boundary morph (target c)
# ================================================================
MORPH_STEPS = 60
MORPH_T = MORPH_STEPS * DT


def channel_width(x, t):
    """Northern boundary Y_N(x, t): sloped -> flat, smoothly."""
    ramp = 0.5 * (1.0 + jnp.cos(
        jnp.pi * jnp.clip(t / MORPH_T, 0.0, 1.0)))
    return 1.0 - 0.2 * jnp.cos(x) * ramp


def make_channel_model(*modules, n=N, family="nodal", dt=DT,
                       pressure_tolerance=1e-8):
    """Boundary-fitted channel ``yp = y * Y_N(x, t)``."""
    mapping = CoordinateMapping(
        maps={"yp": lambda y, YN: y * YN},
        params={"YN": lambda x: channel_width(x, 0.0)})
    grid = Grid((
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"),
    ), mapping=mapping)
    return nh.Model(
        grid=grid, dt=dt, dsqr=DSQR, family=family,
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        pressure_iterations=ITERATIONS,
        pressure_tolerance=pressure_tolerance,
        modules_extra=(
            MovingGeometry({"YN": channel_width}), *modules))


def channel_fields(n=N):
    hor = (np.arange(n) + 0.5) * (TWO_PI / n)
    ver = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(hor, ver, ver, indexing="ij")
    # u = 0.5: undissipated centered advection in the sloped
    # channel is nonlinearly unstable on long horizons (measured
    # blow-up at ~80 steps for u = 1 on the STATIC C3 channel —
    # not a C4 regression); the halved inflow keeps the whole
    # morph window comfortably inside the stable regime
    return {"u": 0.5 + 0.0 * x,
            "b": 0.01 * (2.0 + np.cos(np.pi * z))}, (x, y, z)


def _sigma_frame_integral(field):
    # Raw computational (sigma-frame) full reduction — the escape
    # hatch. The seeded ``.integrate()`` verb is now physical
    # (Jacobian-weighted) AND reads the grid's REFERENCE geometry, not
    # the live morph state; these diagnostics weight by the
    # CURRENT-geometry Jacobian themselves (params-aware ``jac``
    # below), so they reduce through the unweighted raw ``Integral()``
    # to measure exactly int(J q dV) at the morphed geometry.
    result = field
    for name in field.function_space.bare.names:
        result = Integral()[name](result)
    return result


def channel_diagnostics(model):
    """Return (|mapped div|, volume, tracer content) of the state."""
    state = model.state
    grid = state["u"].grid
    params = mapping_params(state, grid)
    solver = MappedPressureSolver(
        grid, state["p"].function_space, iterations=1,
        weights={"z": 1.0 / DSQR}, params=params)
    div = solver.divergence({
        "x": state["u"], "y": state["v"], "z": state["w"]})
    jac = grid.metric(state["b"].function_space, "dyp_dy",
                      params=params)
    volume = float(jnp.sum(_sigma_frame_integral(jac).data))
    tracer = float(jnp.sum(_sigma_frame_integral(state["b"] * jac).data))
    return float(jnp.abs(div.data).max()), volume, tracer


def test_morph_with_ale_is_stable_and_consistent():
    # the owner's target experiment: the northern boundary morphs
    # sloped -> flat during the run, ALE corrections on. Gates:
    # (a) the run is stable; (b) the mapped divergence — measured
    # by the SAME J-weighted operator the projection solves, at the
    # CURRENT geometry — stays at solver tolerance every step;
    # (c) the discrete volume integral(J) is EXACTLY conserved: the
    # schedule preserves the mean width and the midpoint sum of the
    # cos slope over the periodic x nodes is an exact zero, so the
    # morph redistributes volume without creating it (this is the
    # exactly-conserved discrete quantity; the fluid's net mapped
    # volume flux vanishes identically by the telescoping flux
    # form); (d) the tracer content integral(b J) drifts only at
    # truncation level — the consistent (non-telescoping) mapped
    # advection plus the moving boundary sweeping the physical
    # field admit no exact discrete invariant (module docstring);
    # measured relative drift 6.9e-3 over the full morph (n = 8),
    # asserted at 2e-2. Measured: worst per-step divergence
    # 1.9e-14, volume drift 1.4e-16.
    # fixed-iteration mode: pinned for determinism
    model = make_channel_model(MeshVelocityCorrection(),
                               pressure_tolerance=None)
    fields, _ = channel_fields()
    model.set_fields(**fields)
    _, volume0, tracer0 = channel_diagnostics(model)
    u_scale = 0.5
    for _ in range(MORPH_STEPS):
        model.advance(1)
        div, volume, tracer = channel_diagnostics(model)
        assert div < 1e-10 * u_scale
        assert abs(volume - volume0) < 1e-12 * abs(volume0)
        assert abs(tracer - tracer0) < 2e-2 * abs(tracer0)
    assert not model.panicked
    for c in ("u", "v", "w", "b"):
        assert np.isfinite(np.asarray(model.state[c].data)).all()
    # one more step: the SELF_UPDATE writes the geometry at the
    # START-of-step time, so the state sees t = MORPH_T only now —
    # the boundary is genuinely flat, the YN state field 1 exactly
    model.advance(1)
    yn = np.asarray(model.state["YN"].data)
    np.testing.assert_allclose(yn, 1.0, rtol=0, atol=1e-12)


def test_morph_without_ale_runs_to_completion():
    # the deliberate off-switch run (CS-D4): physically wrong
    # during the motion, but it must assemble, run the whole morph,
    # and stay finite
    model = make_channel_model()
    fields, _ = channel_fields()
    model.set_fields(**fields)
    model.advance(MORPH_STEPS)
    assert not model.panicked
    for c in ("u", "v", "w", "b"):
        assert np.isfinite(np.asarray(model.state[c].data)).all()


# ================================================================
#  Gate 4: the oscillating terrain-following column (linear)
# ================================================================
EPS = 0.05
OMEGA = 2.0


def oscillating_depth(x, t):
    """H(x, t): the static slope breathing at 5% amplitude."""
    return depth(x) * (1.0 + EPS * jnp.sin(OMEGA * t))


def make_oscillating_model(device_ids=None, family="nodal"):
    return make_terrain_model(
        MovingGeometry({"H": oscillating_depth}),
        MeshVelocityCorrection(),
        advection=False, device_ids=device_ids, family=family)


def test_oscillating_terrain_compiles_once_and_stays_solenoidal(
        compile_counter):
    # the crucial jit gate: the geometry VALUES sweep every step
    # (schedules are static descriptors, values traced), so the
    # whole run compiles once — zero recompiles after the first
    # advance — while the mapped divergence stays at solver
    # tolerance at the CURRENT geometry
    model = make_oscillating_model()
    fields, _ = terrain_fields()
    model.set_fields(**fields)
    model.advance(2)
    compile_counter.reset()
    model.advance(10)
    assert compile_counter.count == 0
    assert not model.panicked
    state = model.state
    grid = state["u"].grid
    params = mapping_params(state, grid)
    solver = MappedPressureSolver(
        grid, state["p"].function_space, iterations=1,
        weights={"z": 1.0 / DSQR}, params=params)
    div = solver.divergence({
        "x": state["u"], "y": state["v"], "z": state["w"]})
    vel_scale = max(
        float(jnp.abs(state[c].data).max()) for c in "uvw")
    assert float(jnp.abs(div.data).max()) < 1e-8 * max(
        vel_scale, 1e-3)


@pytest.mark.multi_device
def test_oscillating_terrain_is_device_count_invariant():
    # to tight rounding, not bitwise (the C2/C3 precedent): the
    # sharded metric-scaled chains and the CG reductions fuse
    # differently between the 1- and 4-device programs
    def run(device_ids):
        model = make_oscillating_model(device_ids=device_ids)
        fields, _ = terrain_fields()
        model.set_fields(**fields)
        model.advance(10)
        return {c: np.asarray(model.state[c].data)
                for c in ("u", "v", "w", "b", "p")}

    four = run(None)
    one = run((0,))
    for c, want in one.items():
        np.testing.assert_allclose(four[c], want, rtol=0.0,
                                   atol=1e-11)


# ================================================================
#  FV counterparts of the gates (ALE on FV, 2026-07-17)
# ================================================================
# Since the ALE-on-FV closure a moving-geometry model runs on the
# finite-volume family: the MeshVelocityCorrection routes per field on
# its column factor — the conservative flux form on the CellAvg columns
# of b/u/v, the advective form on the wall-normal velocity. These mirror
# the nodal gates above on family="fv".


def _frozen_terrain(family):
    """Return a frozen-schedule (H_dot == 0) terrain model."""
    return make_terrain_model(
        MovingGeometry({"H": lambda x, t: depth(x) + 0.0 * t}),
        family=family)


def test_fv_frozen_motion_reproduces_the_static_run_bitwise():
    # the FV frozen-motion gate: the dynamic pipeline (MovingGeometry
    # state, params= threading, the family-aware ALE flux/advective
    # terms) reproduces the static FV run BITWISE — H_dot == 0 exactly,
    # so the flux-form correction is a bitwise-zero tendency (w == 0 ->
    # G == 0 -> flux_diff(0) == 0), added to the static trajectory.
    static = make_terrain_model(family="fv")
    without_ale = _frozen_terrain("fv")
    with_ale = make_terrain_model(
        MovingGeometry({"H": lambda x, t: depth(x) + 0.0 * t}),
        MeshVelocityCorrection(), family="fv")
    fields, _ = terrain_fields()
    for model in (static, without_ale, with_ale):
        model.set_fields(**fields)
        model.advance(20)
    for c in ("u", "v", "w", "b", "p"):
        want = np.asarray(static.state[c].data)
        assert np.array_equal(
            np.asarray(without_ale.state[c].data), want), c
        assert np.array_equal(
            np.asarray(with_ale.state[c].data), want), c


@pytest.mark.single_device
def test_fv_ale_keeps_the_physical_interpretation_in_place():
    # single_device: see the nodal sign test above (a tall column trips
    # a PRE-EXISTING XLA spmd fault in the static mapped projection).
    # FV counterpart of the manufactured H(t) sign test: with the
    # family-aware flux-form correction the field stays physically put,
    # b(z, T) = f(z * H(T)), and converges at 2nd order. FV reaches its
    # asymptotic regime one refinement later than nodal (a larger error
    # constant; the peak error is INTERIOR — z ~ 0.78 in the shrinking
    # column — not a boundary artifact, so the moving-wall mesh flux is
    # handled), so the rate is measured on the resolved (32, 64) pair:
    # measured 1.62e-3 (n=32), 4.42e-4 (n=64) -> order 1.88.
    steps = 50
    t_final = steps * DT
    errors = []
    for n in (32, 64):
        model = make_ale_model(n, MeshVelocityCorrection(("b",)),
                               family="fv")
        b0, z = ale_initial(n)
        model.set_fields(b=b0)
        model.advance(steps)
        exact = ale_profile(z * (H0 + RATE * t_final))
        errors.append(np.abs(
            np.asarray(model.state["b"].data) - exact).max())
        # the geometry moved far enough that staying frozen is gross
        assert np.abs(b0 - exact).max() > 50 * errors[-1]
    assert errors[0] / errors[1] > 3.4  # 2nd order (4x per level)


def test_fv_morph_with_ale_is_stable_and_consistent():
    # the owner's target experiment on FV. Gates (a)-(c) as the nodal
    # morph: stable; mapped divergence at solver tolerance every step;
    # the discrete volume integral(J) EXACTLY conserved. (d) the NEW
    # tightened tracer bound: the flux-form ALE telescopes, so the
    # tracer-content drift collapses from the nodal 6.9e-3 SPATIAL-
    # truncation drift to a pure TIME-discretization residual (measured
    # worst 1.35e-4 at n = 8; it shrinks with dt — the residual test
    # below — and the semi-discrete budget closes to machine precision
    # at every resolution, tests/model/modules test_fv_telescoping...).
    # Measured: worst per-step divergence 1.7e-14, volume drift 1.4e-16.
    # fixed-iteration mode: pinned for determinism (the nodal twin
    # does the same since the CG tolerance default landed)
    model = make_channel_model(MeshVelocityCorrection(), family="fv",
                               pressure_tolerance=None)
    fields, _ = channel_fields()
    model.set_fields(**fields)
    _, volume0, tracer0 = channel_diagnostics(model)
    u_scale = 0.5
    for _ in range(MORPH_STEPS):
        model.advance(1)
        div, volume, tracer = channel_diagnostics(model)
        assert div < 1e-10 * u_scale
        assert abs(volume - volume0) < 1e-12 * abs(volume0)
        assert abs(tracer - tracer0) < 1e-3 * abs(tracer0)
    assert not model.panicked
    for c in ("u", "v", "w", "b"):
        assert np.isfinite(np.asarray(model.state[c].data)).all()
    # one more step: the SELF_UPDATE writes the geometry at the
    # START-of-step time, so the boundary is genuinely flat now
    model.advance(1)
    yn = np.asarray(model.state["YN"].data)
    np.testing.assert_allclose(yn, 1.0, rtol=0, atol=1e-12)


def test_fv_morph_tracer_drift_is_a_time_residual():
    # the tightened bound is a pure TIME-discretization residual, not
    # the nodal SPATIAL-truncation drift: halving dt over the SAME morph
    # window roughly halves the drift (measured 1.35e-4 -> 6.9e-5),
    # whereas a spatial-truncation error is dt-independent. (The
    # resolution-independence of the underlying invariant is the
    # machine-exact telescoping identity, unit test test_fv_telescoping.)
    def drift(dt, steps):
        model = make_channel_model(
            MeshVelocityCorrection(), family="fv", dt=dt)
        fields, _ = channel_fields()
        model.set_fields(**fields)
        _, _, tracer0 = channel_diagnostics(model)
        model.advance(steps)
        _, _, tracer = channel_diagnostics(model)
        return abs(tracer - tracer0) / abs(tracer0)

    coarse = drift(DT, MORPH_STEPS)          # dt, 60 steps
    fine = drift(DT / 2, 2 * MORPH_STEPS)    # dt/2, 120 steps (same window)
    assert coarse < 1e-3            # tightened vs nodal 6.9e-3
    assert fine < 0.7 * coarse      # shrinks ~linearly with dt


def test_fv_oscillating_terrain_compiles_once_and_stays_solenoidal(
        compile_counter):
    # the FV jit gate: the geometry VALUES sweep every step while the
    # family-aware flux-route reconstruction / flux_diff and the mapped
    # divergence stay at solver tolerance — the whole run compiles once.
    model = make_oscillating_model(family="fv")
    fields, _ = terrain_fields()
    model.set_fields(**fields)
    model.advance(2)
    compile_counter.reset()
    model.advance(10)
    assert compile_counter.count == 0
    assert not model.panicked
    state = model.state
    grid = state["u"].grid
    params = mapping_params(state, grid)
    solver = MappedPressureSolver(
        grid, state["p"].function_space, iterations=1,
        weights={"z": 1.0 / DSQR}, params=params)
    div = solver.divergence({
        "x": state["u"], "y": state["v"], "z": state["w"]})
    vel_scale = max(
        float(jnp.abs(state[c].data).max()) for c in "uvw")
    assert float(jnp.abs(div.data).max()) < 1e-8 * max(
        vel_scale, 1e-3)


@pytest.mark.multi_device
def test_fv_oscillating_terrain_is_device_count_invariant():
    # the FV flux route under forced-4: the one-sided CellAvg -> Outer
    # reconstruction demands the mapped column undistributed, which
    # holds because the periodic x axis shards first (the column z stays
    # device-local). To tight rounding, not bitwise (the sharded
    # metric-scaled chains and CG reductions fuse differently 1 vs 4).
    def run(device_ids):
        model = make_oscillating_model(device_ids=device_ids,
                                       family="fv")
        fields, _ = terrain_fields()
        model.set_fields(**fields)
        model.advance(10)
        return {c: np.asarray(model.state[c].data)
                for c in ("u", "v", "w", "b", "p")}

    four = run(None)
    one = run((0,))
    for c, want in one.items():
        np.testing.assert_allclose(four[c], want, rtol=0.0,
                                   atol=1e-11)
