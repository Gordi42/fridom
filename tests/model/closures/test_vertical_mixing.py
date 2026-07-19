"""Tests for the shared vertical-mixing closure (closures/vertical_mixing).

Covers construction validation, the published coefficient parameters
(``mixing.vertical_nu`` / ``mixing.vertical_kappa``), the bind-time role
target resolution and its taught assembly errors, the analytic 1D column
decay under CNAB2 / SBDF2 against the DISCRETE modal exact solution (the
tridiagonal solve is exact for the discrete operator, so the residual is
purely the time-discretization), the stiff-kappa unconditional stability,
the EXPLICIT write-once path derived from the operator's own ``apply``,
the same-axis merge (op-level and model-level), the IMPLICIT-under-an-
explicit-stepper assembly error, and the ``extra_halo`` exemption.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.closures.vertical_mixing import (
    VerticalMixing,
    _diffusivity,
    _viscosity,
)
from fridom.model.declarations import Lifecycle
from fridom.model.errors import AssemblyError
from fridom.model.implicit import VerticalDiffusion
from fridom.model.model import _chunk_body
from fridom.model.roles import TRACER, Velocity
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

IM = fr.spatial.meshes.IntervalMesh
NX, NZ, DEPTH = 4, 16, 1.0


# ================================================================
#  Builders
# ================================================================
def make_grid(nx=NX, nz=NZ, depth=DEPTH):
    """Return a doubly-periodic (x, y), bounded-z channel grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def mix_model(*, dt, stepper, mixing, csqr=10.0, f0=0.0, grid=None):
    """Assemble a hydrostatic model carrying a VerticalMixing leg."""
    return hy.Model(
        grid=grid if grid is not None else make_grid(),
        dt=dt, csqr=csqr, advection=False,
        coriolis=hy.FPlaneCoriolis(f0=f0),
        stratification=hy.ConstantStratification(n2=1.0),
        modules_extra=(mixing,), time_stepper=stepper)


def neumann_operator(nz, depth, kappa):
    """Dense Neumann second-difference band ``kappa * d2/dz**2``."""
    dz = depth / nz
    main = np.full(nz, -2.0)
    main[0], main[-1] = -1.0, -1.0
    d2 = (np.diag(main) + np.diag(np.ones(nz - 1), 1)
          + np.diag(np.ones(nz - 1), -1)) / dz ** 2
    return kappa * d2


def b0_column(nz=NZ):
    """Return a z-only initial buoyancy (constant in x, y)."""
    return np.cos((np.arange(nz) + 0.5) * 3 * np.pi / nz)


def broadcast_b(col, nx=NX, nz=NZ):
    """Broadcast a z column across the horizontal plane."""
    return np.broadcast_to(col, (nx, nx, nz)).copy()


def stretched_grid(nx=NX, nz=NZ, depth=DEPTH):
    """Return a doubly-periodic (x, y) channel, STRETCHED in z."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        MappedIntervalMesh(nz, (0.0, depth), lambda s: s ** 2,
                           periodic=False, name="z")))


def terrain_grid(nx=NX, nz=NZ):
    """Return a terrain-following (x, sigma) grid: zp = sigma * H(x)."""
    mx = IM(nx, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = IM(nz, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, height: sigma * height},
        params={"height": lambda x: 1.0 + 0.2 * np.sin(x)})
    return fr.spatial.Grid((mx, ms), mapping=mapping)


# ================================================================
#  Fake field table (unit-level bind)
# ================================================================
class FakeRecord:
    def __init__(self, lifecycle):
        self.lifecycle = lifecycle


class FakeGrid:
    def __init__(self, *, immersed=None):
        self.immersed = immersed


class FakeTable:
    def __init__(self, *, velocity=(), tracer=(), records=None, grid=None):
        self._velocity = velocity
        self._tracer = tracer
        self._records = records or {}
        self.grid = grid

    def select(self, role):
        if role is Velocity:
            return self._velocity
        if role is TRACER:
            return self._tracer
        return ()

    def __getitem__(self, name):
        return self._records[name]


# ================================================================
#  Construction validation
# ================================================================
def test_needs_at_least_one_coefficient():
    with pytest.raises(ValueError, match="at least one coefficient"):
        VerticalMixing()


def test_rejects_a_non_treatment_treatment():
    with pytest.raises(TypeError, match="Treatment"):
        VerticalMixing(kv=1.0, treatment="x")


def test_rejects_a_bad_bottom_slip():
    with pytest.raises(ValueError, match="bottom= must be"):
        VerticalMixing(kv=1.0, bottom="sticky")


def test_rejects_a_bad_top_slip():
    with pytest.raises(ValueError, match="top= must be"):
        VerticalMixing(kv=1.0, top="sticky")


# ================================================================
#  Assembly + published parameters
# ================================================================
def test_publishes_both_coefficient_parameters():
    model = mix_model(
        dt=0.02, mixing=VerticalMixing(kv=0.03, kb=0.05),
        stepper=fr.model.time_steppers.CNAB2(0.02))
    assert "mixing.vertical_nu" in model.parameters
    assert "mixing.vertical_kappa" in model.parameters


def test_kv_only_publishes_only_the_viscosity():
    model = mix_model(
        dt=0.02, mixing=VerticalMixing(kv=0.03),
        stepper=fr.model.time_steppers.CNAB2(0.02))
    assert "mixing.vertical_nu" in model.parameters
    assert "mixing.vertical_kappa" not in model.parameters


def test_kb_only_publishes_only_the_diffusivity():
    model = mix_model(
        dt=0.02, mixing=VerticalMixing(kb=0.05),
        stepper=fr.model.time_steppers.CNAB2(0.02))
    assert "mixing.vertical_kappa" in model.parameters
    assert "mixing.vertical_nu" not in model.parameters


# ================================================================
#  Bind: role target resolution and taught errors
# ================================================================
def test_bind_resolves_velocity_and_tracer_targets():
    mixing = VerticalMixing(kv=0.1, kb=0.2)
    mixing.bind(FakeTable(
        velocity=("u", "v"), tracer=("b",),
        records={"u": FakeRecord(Lifecycle.PROGNOSTIC),
                 "v": FakeRecord(Lifecycle.PROGNOSTIC)}))
    terms = mixing.tendency_terms()
    resolved = {t.name: t.implicit.fields for t in terms}
    assert resolved == {"friction": ("u", "v"), "mixing": ("b",)}


def test_bind_rejects_an_immersed_grid():
    # the implicit column solve assumes uniform dz, so a partial bottom
    # cell would silently solve the wrong operator: bind rejects an
    # immersed grid outright with a taught error naming the plan §5
    # variable-dz deferral, before any target resolution
    table = FakeTable(tracer=("b",), grid=FakeGrid(immersed=object()))
    with pytest.raises(
            NotImplementedError,
            match="immersed_closures_sadourny_plan"):
        VerticalMixing(kb=0.1).bind(table)


# ================================================================
#  Stage 0 — taught gates for the uniform-spacing column contract
# ================================================================
def test_bind_rejects_a_stretched_solve_column():
    # the column band infers one uniform dz from the first two nodes,
    # so a MappedIntervalMesh column would silently solve the wrong
    # d2/dz2 — reject it before assembly
    table = FakeTable(tracer=("b",), grid=stretched_grid())
    with pytest.raises(NotImplementedError, match="uniform spacing"):
        VerticalMixing(kb=0.1).bind(table)


def test_bind_rejects_a_terrain_coupled_solve_column():
    # on a terrain grid the solve axis is base-sigma and the band
    # carries no H(x) Jacobian — reject before assembly
    table = FakeTable(tracer=("b",), grid=terrain_grid())
    with pytest.raises(NotImplementedError, match="terrain"):
        VerticalMixing(kb=0.1, vertical="sigma").bind(table)


def test_uniform_column_model_still_assembles_and_steps():
    # the gates are structural: a plain uniform IntervalMesh column
    # binds, assembles and steps unchanged
    model = mix_model(
        dt=0.02, mixing=VerticalMixing(kv=0.03, kb=0.05),
        stepper=fr.model.time_steppers.CNAB2(0.02))
    model.set_fields(b=broadcast_b(b0_column()))
    model.run(steps=3, progress=False)
    assert np.all(np.isfinite(np.asarray(model.state["b"].data)))


def test_bind_rejects_kb_without_a_tracer():
    with pytest.raises(AssemblyError, match="TRACER"):
        VerticalMixing(kb=0.1).bind(FakeTable(tracer=()))


def test_bind_rejects_kv_without_a_prognostic_velocity():
    # a DIAGNOSTIC-only velocity selection resolves zero targets
    with pytest.raises(AssemblyError, match="PROGNOSTIC"):
        VerticalMixing(kv=0.1).bind(FakeTable(
            velocity=("u",),
            records={"u": FakeRecord(Lifecycle.DIAGNOSTIC)}))


# ================================================================
#  Analytic 1D column decay vs the DISCRETE modal exact solution
# ================================================================
@pytest.mark.parametrize("scheme", ["cnab2", "sbdf2"])
def test_column_decay_matches_the_discrete_exact_solution(scheme):
    # the tridiagonal solve is exact for the discrete operator KB*d2;
    # the residual below is purely the CNAB2/SBDF2 time-discretization
    # against the exact discrete modal decay (measured: cnab2 1.2e-5,
    # sbdf2 7.3e-5), well inside the test_imex 2e-4 tolerance.
    kb, dt, steps = 0.05, 0.02, 50
    matrix = neumann_operator(NZ, DEPTH, kb)
    evals, evecs = np.linalg.eigh(matrix)
    col = b0_column()
    model = mix_model(
        dt=dt, mixing=VerticalMixing(kb=kb),
        stepper=fr.model.time_steppers.IMEXMultistep(dt, scheme=scheme))
    model.set_fields(b=broadcast_b(col))
    model.run(steps=steps, progress=False)
    got = np.asarray(model.state["b"].data)[0, 0, :]
    exact = evecs @ (np.exp(evals * dt * steps) * (evecs.T @ col))
    assert np.max(np.abs(got - exact)) < 2e-4


# ================================================================
#  Stiff-kappa unconditional stability
# ================================================================
def test_stiff_kappa_column_stays_bounded_and_decays():
    # kappa*dt/dz^2 ~ 640 (kb=5, dt=0.5, nz=16): deeply stiff, yet
    # CNAB2 stays finite and decays (measured |b|max ~ 1.5e-3)
    kb, dt = 5.0, 0.5
    col = b0_column()
    model = mix_model(
        dt=dt, mixing=VerticalMixing(kb=kb),
        stepper=fr.model.time_steppers.CNAB2(dt))
    model.set_fields(b=broadcast_b(col))
    model.run(steps=60, progress=False)
    data = np.asarray(model.state["b"].data)
    bmax = np.max(np.abs(data))
    assert np.all(np.isfinite(data))
    assert bmax < 1.0


# ================================================================
#  EXPLICIT treatment — the write-once apply-derived path
# ================================================================
def test_explicit_tendency_is_the_operators_own_apply():
    # under EXPLICIT the tendency IS VerticalDiffusion.apply, i.e. the
    # exact discrete operator L @ b (measured max err ~ 3.5e-15)
    kb, dt = 0.05, 0.001
    col = b0_column()
    model = mix_model(
        dt=dt, mixing=VerticalMixing(kb=kb, treatment=fr.model.EXPLICIT),
        stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))
    model.set_fields(b=broadcast_b(col))
    tendency = np.asarray(model.tendency(model.state)["b"].data)[0, 0, :]
    expected = neumann_operator(NZ, DEPTH, kb) @ col
    assert np.max(np.abs(tendency - expected)) < 1e-12


def test_explicit_treatment_decays_under_adam_bashforth():
    kb, dt = 0.05, 0.001
    col = b0_column()
    model = mix_model(
        dt=dt, mixing=VerticalMixing(kb=kb, treatment=fr.model.EXPLICIT),
        stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))
    model.set_fields(b=broadcast_b(col))
    model.run(steps=200, progress=False)
    data = np.asarray(model.state["b"].data)
    assert np.all(np.isfinite(data))
    assert np.max(np.abs(data)) < np.max(np.abs(col))


# ================================================================
#  Mergeability (same-axis VerticalDiffusion family)
# ================================================================
def test_op_level_velocity_and_buoyancy_legs_merge():
    friction = VerticalDiffusion("z", ("u", "v"), _viscosity)
    mixing = VerticalDiffusion("z", ("b",), _diffusivity)
    merged = friction.merged_with(mixing)
    assert merged.fields == ("u", "v", "b")
    assert friction.merge_key() == mixing.merge_key()


def test_model_level_friction_and_mixing_merge_into_one_solve():
    model = mix_model(
        dt=0.02, f0=0.5, mixing=VerticalMixing(kv=0.03, kb=0.05),
        stepper=fr.model.time_steppers.CNAB2(0.02))
    merged = model._artifacts.schedule.implicit_merged
    assert len(merged) == 1
    operator, _slot = merged[0]
    assert operator.fields == ("u", "v", "b")


# ================================================================
#  IMPLICIT under a purely-explicit stepper is an assembly error
# ================================================================
def test_implicit_under_adam_bashforth_is_an_assembly_error():
    with pytest.raises(AssemblyError, match="not supported"):
        mix_model(
            dt=0.02, mixing=VerticalMixing(kb=0.1),
            stepper=fr.model.time_steppers.AdamBashforth(0.02, order=2))


# ================================================================
#  extra_halo (the EXPLICIT write-once halo-trace exemption)
# ================================================================
def test_explicit_declares_an_extra_halo():
    assert VerticalMixing(
        kb=0.1, treatment=fr.model.EXPLICIT).extra_halo is not None


def test_implicit_declares_no_extra_halo():
    assert VerticalMixing(kb=0.1).extra_halo is None


# ================================================================
#  No-slip (Dirichlet) velocity rows
# ================================================================
def u_profile(nz=NZ):
    """Return a smooth, z-varying velocity column (non-eigenmode)."""
    return np.cos((np.arange(nz) + 0.5) * np.pi / nz)


def test_no_slip_damps_a_constant_velocity_toward_zero():
    # the Dirichlet-Dirichlet band has NO constant nullspace, so a
    # constant velocity column (isolated from the other dynamics:
    # x/y-uniform, f0=0, b=0) decays toward the no-slip solution 0
    kv, dt = 0.1, 0.02
    const = broadcast_b(np.ones(NZ))
    model = mix_model(
        dt=dt,
        mixing=VerticalMixing(kv=kv, bottom="no-slip", top="no-slip"),
        stepper=fr.model.time_steppers.CNAB2(dt))
    model.set_fields(u=const)
    model.run(steps=100, progress=False)
    u = np.asarray(model.state["u"].data)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) < 0.5   # decayed well below the initial 1


def test_free_slip_preserves_a_constant_velocity():
    # the default free-slip (Neumann) band annihilates the constant, so
    # a constant column is preserved — the contrast to the no-slip case
    kv, dt = 0.1, 0.02
    const = broadcast_b(np.ones(NZ))
    model = mix_model(
        dt=dt, mixing=VerticalMixing(kv=kv),
        stepper=fr.model.time_steppers.CNAB2(dt))
    model.set_fields(u=const)
    model.run(steps=100, progress=False)
    u = np.asarray(model.state["u"].data)
    assert np.max(np.abs(u - 1.0)) < 1e-2


def test_no_slip_stays_stable_in_the_stiff_regime():
    # kv*dt/dz^2 ~ 640 (kv=5, dt=0.5, nz=16): deeply stiff, yet the
    # implicit no-slip solve stays finite and decays the profile
    kv, dt = 5.0, 0.5
    prof = broadcast_b(u_profile())
    model = mix_model(
        dt=dt,
        mixing=VerticalMixing(kv=kv, bottom="no-slip", top="no-slip"),
        stepper=fr.model.time_steppers.CNAB2(dt))
    model.set_fields(u=prof)
    model.run(steps=60, progress=False)
    u = np.asarray(model.state["u"].data)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) < np.max(np.abs(prof))


def test_no_slip_velocity_and_neumann_buoyancy_legs_do_not_merge():
    # the velocity leg is Dirichlet-row (no-slip), the buoyancy leg is
    # Neumann (a tracer has no slip): different merge keys, so the
    # composer keeps TWO tridiagonal solve sets, never mixing them
    model = mix_model(
        dt=0.02, f0=0.5,
        mixing=VerticalMixing(kv=0.03, kb=0.05, bottom="no-slip"),
        stepper=fr.model.time_steppers.CNAB2(0.02))
    merged = model._artifacts.schedule.implicit_merged
    assert len(merged) == 2
    covered = {ops.fields for ops, _slot in merged}
    assert covered == {("u", "v"), ("b",)}


# ================================================================
#  Autodiff regression (the differentiability policy)
# ================================================================
def _kernel(model):
    """Return (record, carry, stepper) for the pure step kernel."""
    return (model._artifacts.record, model._carry, model._stepper)


def _state_sq(final):
    """Sum of squares of every final-state field (a smooth loss)."""
    return sum(jnp.sum(f.data ** 2) for f in final.state)


def _leaf_loss(record, carry, stepper, leaf_ref, n_steps):
    """Loss splicing ``leaf_ref`` (found by identity) into the carry."""
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    (idx,) = [i for i, ref in enumerate(leaves) if ref is leaf_ref]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        return _state_sq(_chunk_body(record, n_steps, spliced, stepper))

    return loss


def test_no_slip_run_is_reverse_mode_differentiable():
    # jax.grad of a quadratic loss through a short no-slip run w.r.t.
    # the vertical viscosity is finite and matches a central FD; the
    # Dirichlet row is linear in u_1, so no singularity guard is needed
    dt = 0.02
    model = mix_model(
        dt=dt,
        mixing=VerticalMixing(kv=0.05, bottom="no-slip", top="no-slip"),
        stepper=fr.model.time_steppers.CNAB2(dt))
    model.set_fields(u=broadcast_b(u_profile()))
    record, carry, stepper = _kernel(model)
    kv0 = next(m for m in carry.modules
               if isinstance(m, VerticalMixing)).kv
    loss = _leaf_loss(record, carry, stepper, kv0, n_steps=6)

    grad = float(jax.grad(loss)(kv0))
    assert np.isfinite(grad)
    assert abs(grad) > 0.0   # kv damps u, so the gradient is non-zero
    h = 1e-4 * float(kv0)
    fd = (float(loss(kv0 + h)) - float(loss(kv0 - h))) / (2.0 * h)
    assert grad == pytest.approx(fd, rel=1e-4)
