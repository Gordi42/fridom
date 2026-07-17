"""The split-explicit free surface: barotropic subcycle (H6).

Prefix-mirrored shard of ``hy.modules.free_surface`` covering the
``SplitExplicitFreeSurface`` variant (the explicit variant lives in
``test_free_surface.py``, the implicit in ``test_free_surface_implicit``);
self-contained per the AGENTS oversized-module rule (small builders
duplicated).
"""
import inspect

import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.errors import AssemblyError, LinearOperatorGapError
from fridom.model.implicit import VerticalDiffusion
from fridom.model.io.streams import SnapshotMismatchError
from fridom.model.module import Module
from fridom.model.terms import Treatment
from fridom.spatial.spaces.constant import ConstantSpace

IM = fr.spatial.meshes.IntervalMesh
SEFS = hy.SplitExplicitFreeSurface


# ================================================================
#  Builders (duplicated per the self-contained-shard rule)
# ================================================================
def make_grid(nx, nz, depth=1.0):
    """Return a doubly-periodic horizontal, bounded-vertical grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def make_model(grid, free_surface, *, n2=0.0, csqr=4.0, f0=0.0,
               dt=1e-2, stepper=None):
    """Return a linear hydrostatic model on the given free surface."""
    if stepper is None:
        stepper = fr.model.time_steppers.AdamBashforth(dt, order=3)
    return hy.Model(
        grid=grid, dt=dt, csqr=csqr, free_surface=free_surface,
        stratification=hy.ConstantStratification(n2=n2),
        coriolis=hy.FPlaneCoriolis(f0=f0), advection=False,
        time_stepper=stepper)


def mode_field(model, name, kx, ky, phase, kz=0, depth=1.0):
    """cos/sin(kx x + ky y) [* cos(pi kz z / H)] at name's nodes."""
    fs = model.state[name].function_space
    var = tuple(n for f in fs.bare.factors
                if not isinstance(f, ConstantSpace) for n in f.names)
    trig = np.cos if phase == "cos" else np.sin

    def init(**c):
        base = trig(2 * np.pi * (kx * c["x"] + ky * c["y"]))
        if "z" in c and kz > 0:
            base = base * np.cos(np.pi * kz * c["z"] / depth)
        return base + 0.0 * sum(c.values())
    init.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for n in var])
    return model.grid.create_field(fs, init=init)


def zeroed(model, fields):
    """Return the model's state with the named fields zeroed."""
    st = model.state
    for g in fields:
        st = st.replace(**{g: model.grid.create_field(
            model.state[g].function_space,
            data=0.0 * model.state[g].data)})
    return st


def depth_mean(arr):
    """Barotropic (depth-mean) part of a (nx, ny, nz) numpy array."""
    return arr.mean(axis=2, keepdims=True)


def norm(arr):
    """Euclidean norm of a numpy array."""
    return float(np.sqrt((arr * arr).sum()))


# ================================================================
#  Constructor / properties
# ================================================================
def test_factory_assembles_and_declares_ps_U_V():
    grid = make_grid(8, 4)
    model = make_model(grid, SEFS(substeps=8))
    names = model.state.component_names
    assert {"ps", "U", "V", "ubar_prev", "vbar_prev"} <= set(names)


def test_default_free_surface_is_unchanged():
    # hy.Model()'s default free surface is still the explicit variant:
    # no barotropic transports appear.
    grid = make_grid(8, 4)
    model = hy.Model(grid=grid, advection=False,
                     stratification=hy.ConstantStratification(n2=0.0))
    assert "ps" in model.state.component_names
    assert "U" not in model.state.component_names


def test_properties_and_defaults():
    fs = SEFS(substeps=24)
    assert fs.substeps == 24
    assert fs.sm_filter == (2, 4, 0.18927)
    assert fs.forcing == "increment"
    assert len(fs.weights) == 24


def test_weights_sum_to_one_and_center_near_tau_one():
    fs = SEFS(substeps=32)
    w = np.asarray(fs.weights)
    assert abs(w.sum() - 1.0) < 1e-14
    # discrete first moment (tau_m = m * 2/N) close to 1 (centered)
    tau = np.arange(1, 33) * (2.0 / 32)
    assert abs(float((tau * w).sum()) - 1.0) < 0.02


@pytest.mark.parametrize(
    "substeps",
    [pytest.param(1, id="one"), pytest.param(0, id="zero"),
     pytest.param(-4, id="negative"), pytest.param(2.0, id="float"),
     pytest.param(True, id="bool")],
)
def test_substeps_validation_rejects(substeps):
    with pytest.raises(ValueError, match="substeps"):
        SEFS(substeps=substeps)


@pytest.mark.parametrize(
    "flt",
    [pytest.param((2, 4), id="two-tuple"),
     pytest.param((0, 4, 0.1), id="p-zero"),
     pytest.param((2, -1, 0.1), id="q-negative"),
     pytest.param((2.0, 4, 0.1), id="p-float"),
     pytest.param((2, 4, "x"), id="r-string")],
)
def test_filter_validation_rejects(flt):
    with pytest.raises(ValueError, match="filter"):
        SEFS(substeps=8, filter=flt)


def test_degenerate_filter_has_no_positive_weight():
    # a huge linear correction r drives the whole shape non-positive
    with pytest.raises(ValueError, match="positive weight"):
        SEFS(substeps=8, filter=(2, 4, 100.0))


def test_forcing_validation_rejects():
    with pytest.raises(ValueError, match="forcing"):
        SEFS(substeps=8, forcing="magic")


@pytest.mark.parametrize(
    "horizontal",
    [pytest.param(("x",), id="one"),
     pytest.param(("x", "x"), id="dup"),
     pytest.param(("x", "y", "z"), id="three")],
)
def test_rejects_a_bad_horizontal(horizontal):
    with pytest.raises(TypeError, match="horizontal"):
        SEFS(substeps=8, horizontal=horizontal)


# ================================================================
#  Roles / declarations
# ================================================================
def test_transports_carry_no_velocity_role():
    grid = make_grid(8, 4)
    model = make_model(grid, SEFS(substeps=8))
    table = model._artifacts.field_table

    def is_velocity(record):
        return any(type(role).__name__ == "Velocity"
                   for role in getattr(record, "roles", ()))
    velocity = {r.name for r in table if is_velocity(r)}
    # the baroclinic trio keeps the role; U, V do NOT
    assert velocity == {"u", "v", "w"}
    assert "U" not in velocity
    assert "V" not in velocity


def test_lifecycles():
    grid = make_grid(8, 4)
    model = make_model(grid, SEFS(substeps=8))
    table = model._artifacts.field_table
    life = {r.name: r.lifecycle.name for r in table}
    assert life["ps"] == "PROGNOSTIC"
    assert life["U"] == "PROGNOSTIC"
    assert life["V"] == "PROGNOSTIC"
    assert life["ubar_prev"] == "AUXILIARY"
    assert life["vbar_prev"] == "AUXILIARY"


def test_declares_the_linear_operator_gap():
    grid = make_grid(8, 4)
    split = make_model(grid, SEFS(substeps=8), f0=0.5)
    explicit = make_model(grid, hy.ExplicitFreeSurface(), f0=0.5)
    with pytest.raises(LinearOperatorGapError,
                       match="surface-pressure"):
        fr.model.require_linear_operator(split, consumer="test")
    # the explicit variant passes (its barotropic terms ARE in L)
    fr.model.require_linear_operator(explicit, consumer="test")


# ================================================================
#  Assembly guard: multistep outer drivers only (§5.4)
# ================================================================
@pytest.mark.parametrize(
    "stepper_factory",
    [pytest.param(fr.model.time_steppers.LowStorageRK3, id="rk3"),
     pytest.param(fr.model.time_steppers.ExplicitRungeKutta, id="rk4")],
)
def test_refuses_rk_family_outer_drivers(stepper_factory):
    grid = make_grid(8, 4)
    dt = 1e-2
    with pytest.raises(AssemblyError, match="multistep"):
        make_model(grid, SEFS(substeps=8), dt=dt,
                   stepper=stepper_factory(dt))


@pytest.mark.parametrize(
    "stepper_factory",
    [pytest.param(
        lambda dt: fr.model.time_steppers.AdamBashforth(dt, 2, eps=0.1),
        id="ab2"),
     pytest.param(
        lambda dt: fr.model.time_steppers.AdamBashforth(dt, 3),
        id="ab3"),
     pytest.param(fr.model.time_steppers.CNAB2, id="cnab2"),
     pytest.param(fr.model.time_steppers.SBDF2, id="sbdf2")],
)
def test_accepts_multistep_outer_drivers(stepper_factory):
    grid = make_grid(8, 4)
    dt = 1e-2
    model = make_model(grid, SEFS(substeps=8), csqr=4.0, f0=0.5,
                       dt=dt, stepper=stepper_factory(dt))
    rng = np.random.default_rng(0)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape))
    model.advance(5)
    assert np.isfinite(float(np.abs(np.asarray(
        model.state["u"].data)).max()))


# ================================================================
#  Geostrophic steady state (the correction must not disturb balance)
# ================================================================
def _reduced_tendency(model, fields, kx, ky):
    """Build the reduced tendency operator on one horizontal mode."""
    basis = [(f, p) for f in fields for p in ("cos", "sin")]
    modes = {(f, p): mode_field(model, f, kx, ky, p) for f, p in basis}
    norms = {k: float((v.data * v.data).sum())
             for k, v in modes.items()}
    n = len(basis)
    mat = np.zeros((n, n))
    for i, (f, p) in enumerate(basis):
        st = zeroed(model, ["u", "v", "b", "ps"])
        st = st.replace(**{f: modes[(f, p)]})
        dX = model.tendency(st)
        for j, (g, p2) in enumerate(basis):
            proj = float((np.asarray(dX[g].data)
                          * np.asarray(modes[(g, p2)].data)).sum())
            mat[j, i] = proj / norms[(g, p2)]
    return mat, basis, modes


def test_geostrophic_steady_state_stays_steady():
    grid = make_grid(16, 4, depth=1.0)
    csqr, f0 = 4.0, 0.8
    fields = ["u", "v", "ps"]
    # the exactly-balanced discrete state is the null eigenvector of the
    # EXPLICIT reduced tendency (u, v, ps) — the implicit-variant recipe
    exp = make_model(grid, hy.ExplicitFreeSurface(), csqr=csqr, f0=f0,
                     dt=2e-3)
    mat, basis, modes = _reduced_tendency(exp, fields, 1, 1)
    evals, evecs = np.linalg.eig(mat)
    i0 = int(np.argmin(np.abs(evals)))
    assert abs(evals[i0]) < 1e-12

    m = make_model(grid, SEFS(substeps=16), csqr=csqr, f0=f0, dt=2e-3)
    st = zeroed(m, ["u", "v", "b", "ps"])
    for coeff, (f, p) in zip(evecs[:, i0].real, basis, strict=True):
        st = st.replace(**{f: st[f] + coeff * modes[(f, p)]})
    amp = max(float(np.abs(st[f].data).max()) for f in fields)
    # seed the barotropic transports U, V = H * depth-mean(u, v)
    st = st.replace(
        U=m.grid.create_field(m.state["U"].function_space,
                              data=np.asarray(st["u"].mean("z").data)),
        V=m.grid.create_field(m.state["V"].function_space,
                              data=np.asarray(st["v"].mean("z").data)))
    m.set_state(st)
    u0 = np.asarray(m.state["u"].data).copy()
    p0 = np.asarray(m.state["ps"].data).copy()
    m.advance(20)
    du = np.abs(np.asarray(m.state["u"].data) - u0).max() / amp
    dp = np.abs(np.asarray(m.state["ps"].data) - p0).max() / amp
    assert du < 1e-10
    assert dp < 1e-10


# ================================================================
#  Convergence to the explicit oracle on the SLOW (baroclinic) mode
# ================================================================
def test_converges_to_the_explicit_oracle_on_the_slow_mode():
    # The comparison is honest about the split: the fast barotropic
    # gravity mode is FILTERED (SM2005), so the velocity — which carries
    # that mode — decreases toward a fast-mode-filtering floor as dt->0,
    # while the SLOW baroclinic field, the buoyancy b, converges cleanly
    # to the explicit oracle (measured near second order: rates ~2.3,
    # ~2.9 here). csqr=4 keeps the explicit oracle barotropic-stable at
    # every dt (sqrt(csqr) dt/dx <= 0.8) and the fast-mode imprint on b
    # small enough to resolve the convergence. The fast mode itself is
    # covered by test_barotropic_wave_is_captured_within_a_few_percent.
    nx, nz, csqr, f0, n2 = 16, 8, 4.0, 0.5, 4.0
    grid = make_grid(nx, nz)
    horizon = 0.4
    e_u, e_b = [], []
    for nsteps in (16, 32, 64):
        dt = horizon / nsteps
        ms = make_model(grid, SEFS(substeps=32), n2=n2, csqr=csqr,
                        f0=f0, dt=dt)
        me = make_model(grid, hy.ExplicitFreeSurface(), n2=n2,
                        csqr=csqr, f0=f0, dt=dt)
        for m in (ms, me):
            m.set_fields(u=mode_field(m, "u", 1, 0, "sin", kz=1).data,
                         b=mode_field(m, "b", 1, 0, "cos", kz=1).data)
            m.run(steps=nsteps)
        us = np.asarray(ms.state["u"].data)
        ue = np.asarray(me.state["u"].data)
        bs = np.asarray(ms.state["b"].data)
        be = np.asarray(me.state["b"].data)
        e_u.append(norm(us - ue))
        e_b.append(norm(bs - be) / norm(be))
    e_u = np.asarray(e_u)
    e_b = np.asarray(e_b)
    # the slow baroclinic buoyancy converges (monotone, faster than
    # first order at the finest refinement)
    assert e_b[0] > e_b[1] > e_b[2]
    assert float(np.log2(e_b[-2] / e_b[-1])) > 1.0
    assert e_b[0] < 5e-3
    # the velocity decreases toward the fast-mode-filtering floor
    assert e_u[-1] < e_u[0]


def test_barotropic_wave_is_captured_within_a_few_percent():
    # the FAST barotropic gravity wave: the SM2005-averaged subcycle
    # reproduces the explicit-oracle wave (fine dt) to a few percent —
    # what the filter does to a *resolved* fast mode.
    csqr = 1.0
    grid = make_grid(16, 4)
    ref = make_model(grid, hy.ExplicitFreeSurface(), csqr=csqr, dt=1e-3)
    ref.set_fields(ps=mode_field(ref, "ps", 1, 0, "cos").data)
    ref.run(steps=200)                       # T = 0.2
    ps_ref = np.asarray(ref.state["ps"].data)

    split = make_model(grid, SEFS(substeps=32), csqr=csqr, dt=2e-2)
    split.set_fields(ps=mode_field(split, "ps", 1, 0, "cos").data)
    split.run(steps=10)                      # T = 0.2
    ps_split = np.asarray(split.state["ps"].data)
    rel = norm(ps_split - ps_ref) / norm(ps_ref)
    assert rel < 0.03


# ================================================================
#  Barotropic stability: subcycle sized by the barotropic CFL
# ================================================================
def test_stable_far_beyond_the_explicit_cfl_when_subcycled():
    nx, csqr = 16, 400.0
    dx = 1.0 / nx
    dt = 8.0 * dx / np.sqrt(csqr)     # explicit CFL sqrt(csqr) dt/dx = 8
    assert np.sqrt(csqr) * dt / dx > 7.9
    grid = make_grid(nx, 4)
    # N = 32 -> substep CFL sqrt(csqr) (2 dt / N) / dx = 0.5 < 1: stable
    model = make_model(grid, SEFS(substeps=32), csqr=csqr, f0=0.5,
                       dt=dt)
    rng = np.random.default_rng(1)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape))
    model.advance(100)
    umax = float(np.abs(np.asarray(model.state["u"].data)).max())
    assert np.isfinite(umax)


def test_under_resolved_subcycle_is_caught():
    # N = 4 -> substep CFL = 4 >> 1: the barotropic subcycle is
    # unstable and the NaN seam catches it (a PanicError), never a
    # silent wrong answer.
    from fridom.model.results import PanicError  # noqa: PLC0415
    nx, csqr = 16, 400.0
    dx = 1.0 / nx
    dt = 8.0 * dx / np.sqrt(csqr)
    grid = make_grid(nx, 4)
    model = make_model(grid, SEFS(substeps=4), csqr=csqr, f0=0.5, dt=dt)
    rng = np.random.default_rng(1)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape),
                     v=rng.standard_normal(model.state["v"].shape))
    with pytest.raises(PanicError):
        model.advance(200)


# ================================================================
#  Conservation through the filter
# ================================================================
def test_ps_mean_and_tracer_mass_conserved():
    grid = make_grid(16, 4)
    model = make_model(grid, SEFS(substeps=16), n2=1.0, csqr=4.0,
                       f0=0.5, dt=1e-2)
    rng = np.random.default_rng(2)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))

    def ps_mean(m):
        return float(np.asarray(
            m.state["ps"].mean("x", "y").data).ravel()[0])

    def b_mass(m):
        return float(np.asarray(
            m.state["b"].integrate("x", "y", "z").data).ravel()[0])

    pm0, bm0 = ps_mean(model), b_mass(model)
    model.advance(30)
    assert abs(ps_mean(model) - pm0) < 1e-12
    assert abs(b_mass(model) - bm0) / max(abs(bm0), 1e-30) < 1e-12


# ================================================================
#  The tendency-sums forcing knob (and its IMPLICIT branch)
# ================================================================
def _mix_kappa(_module, _state, _ctx, _name):
    return 0.05


class _VertMix(Module):

    """Minimal in-test implicit vertical mixing on u, v (test-only)."""

    field_references = (
        fr.model.FieldReference("u", hint="core"),
        fr.model.FieldReference("v", hint="core"),
    )

    @fr.model.term(name="mix", treatment=Treatment.IMPLICIT,
                   implicit=VerticalDiffusion("z", ("u", "v"),
                                              _mix_kappa))
    def mix(self, _state, _ctx):
        return {}


def test_tendency_sums_forcing_runs_incl_the_implicit_branch():
    # forcing="tendency_sums" reads ctx's per-treatment sums; under
    # CNAB2 with a vertical-mixing consumer the IMPLICIT sum is
    # populated (the forward apply), exercising that branch.
    grid = make_grid(16, 6)
    model = hy.Model(
        grid=grid, csqr=4.0,
        free_surface=SEFS(substeps=8, forcing="tendency_sums"),
        stratification=hy.ConstantStratification(n2=1.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5), advection=False,
        modules_extra=(_VertMix(),),
        time_stepper=fr.model.time_steppers.CNAB2(1e-2))
    rng = np.random.default_rng(3)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape))
    model.advance(10)
    umax = float(np.abs(np.asarray(model.state["u"].data)).max())
    assert np.isfinite(umax)


def test_tendency_sums_forcing_runs_without_implicit():
    # forcing="tendency_sums" under a purely explicit stepper (AB3):
    # only the EXPLICIT sum is present (the IMPLICIT lookup KeyErrors,
    # handled).
    grid = make_grid(12, 4)
    model = make_model(grid, SEFS(substeps=8, forcing="tendency_sums"),
                       n2=1.0, csqr=4.0, f0=0.5, dt=1e-2)
    rng = np.random.default_rng(4)
    model.set_fields(u=rng.standard_normal(model.state["u"].shape))
    model.advance(6)
    assert np.isfinite(float(np.abs(np.asarray(
        model.state["u"].data)).max()))


# ================================================================
#  Restart fingerprint + own-AUX round-trip
# ================================================================
def _digest(free_surface):
    grid = make_grid(8, 4)
    return make_model(grid, free_surface, csqr=4.0).fingerprint.digest


def test_integrator_statics_change_the_fingerprint():
    base = _digest(SEFS(substeps=16))
    assert _digest(SEFS(substeps=16)) == base               # reproducible
    assert _digest(SEFS(substeps=32)) != base               # substeps
    assert _digest(SEFS(substeps=16, filter=(2, 6, 0.18927))) != base
    assert _digest(SEFS(substeps=16, forcing="tendency_sums")) != base


def test_snapshot_roundtrip_is_bitwise(tmp_path):
    grid = make_grid(8, 4)

    def build():
        return make_model(grid, SEFS(substeps=8), n2=1.0, csqr=4.0,
                          f0=0.5, dt=1e-2)

    rng = np.random.default_rng(5)
    fields = {"u": rng.standard_normal(build().state["u"].shape),
              "v": rng.standard_normal(build().state["v"].shape),
              "b": rng.standard_normal(build().state["b"].shape),
              "ps": rng.standard_normal(build().state["ps"].shape)}
    model = build()
    model.set_fields(**fields)
    model.advance(3)
    model.snapshot(tmp_path / "snap")
    model.advance(4)
    reference = np.asarray(model.state["u"].data)

    resumed = build()
    resumed.load_snapshot(tmp_path / "snap")
    assert int(resumed.clock.it) == 3
    # the own-AUX buffers round-trip with the carry
    assert {"ubar_prev", "vbar_prev"} <= set(
        resumed.state.component_names)
    resumed.advance(4)
    assert np.array_equal(reference,
                          np.asarray(resumed.state["u"].data))


def test_snapshot_refuses_a_different_substep_count(tmp_path):
    grid = make_grid(8, 4)
    model = make_model(grid, SEFS(substeps=8), csqr=4.0)
    model.advance(2)
    model.snapshot(tmp_path / "snap")
    variant = make_model(grid, SEFS(substeps=16), csqr=4.0)
    with pytest.raises(SnapshotMismatchError):
        variant.load_snapshot(tmp_path / "snap")


# ================================================================
#  Forced multi-device (the 2D subcycle under 4 host devices)
# ================================================================
@pytest.mark.multi_device
def test_subcycle_runs_under_forced_devices():
    grid = make_grid(16, 4)
    model = make_model(grid, SEFS(substeps=16), n2=1.0, csqr=9.0,
                       f0=0.5, dt=1e-2)
    rng = np.random.default_rng(7)
    model.set_fields(
        u=rng.standard_normal(model.state["u"].shape),
        v=rng.standard_normal(model.state["v"].shape),
        b=rng.standard_normal(model.state["b"].shape),
        ps=rng.standard_normal(model.state["ps"].shape))
    model.advance(20)
    umax = float(np.abs(np.asarray(model.state["u"].data)).max())
    assert np.isfinite(umax)
    assert umax > 0.0
