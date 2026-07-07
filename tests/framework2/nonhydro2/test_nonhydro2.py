"""Basic-correctness + smoke tests for the framework2 nonhydro port.

Parity vs the OLD model is TICK-2's job (the signed-delta suite); these
are the wave-6 acceptance smoke tests: preset == explicit assembly, a
treedef-stable single-compile run, the CONSTRAINT projection driving
div(u) to machine zero, eigenmode biorthogonality/dispersion, the
"provides implies constancy" and Velocity-role rules, and the State
vocabulary.
"""
import jax
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.composed import Divergence
from fridom.framework2.model.model import Model as FrModel
from fridom.framework2.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.framework2.model.roles import ADVECTED, TRACER, Velocity
from fridom.framework2.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.nonhydro2.params import DSQR
from fridom.nonhydro2.state import State

N = 8
DT = 0.02


def make_grid(n=N, length=2 * np.pi):
    return Grid(tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=name)
        for name in ("x", "y", "z")))


def divergence(model):
    st = model.state
    vel = VectorField({c: st[c] for c in ("u", "v", "w")})
    return np.asarray(Divergence()(vel).data)


def grid_coords(n=N, length=2 * np.pi):
    ax = (np.arange(n) + 0.5) * (length / n)
    return np.meshgrid(ax, ax, ax, indexing="ij")


# ================================================================
#  D4 preset test: preset == explicit assembly (identical treedef)
# ================================================================
def test_preset_equals_explicit_assembly_treedef():
    grid = make_grid()
    preset = nh.Model(grid=grid, dt=DT)
    explicit = FrModel(
        grid=grid,
        modules=(
            DynamicalCore(dsqr=1.0, rossby_number=1.0),
            FPlaneCoriolis(f0=1.0),
            ConstantStratification(n2=1.0),
            CenteredAdvection()),
        time_stepper=AdamBashforth(DT, order=3))
    assert (jax.tree_util.tree_structure(preset._carry)
            == jax.tree_util.tree_structure(explicit._carry))


def test_preset_is_a_plain_fr_model():
    model = nh.Model(grid=make_grid(), dt=DT)
    assert isinstance(model, FrModel)
    assert isinstance(model.state, State)


# ================================================================
#  A treedef-stable, single-compile run
# ================================================================
def test_linear_run_is_treedef_stable():
    model = nh.Model(grid=make_grid(), dt=DT, advection=False)
    _, _, z = grid_coords()
    model.set_fields(b=0.01 * np.cos(z))
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(5)
    after = jax.tree_util.tree_structure(model._carry)
    assert before == after


def test_second_advance_compiles_nothing(compile_counter):
    model = nh.Model(grid=make_grid(), dt=DT, advection=False)
    _, _, z = grid_coords()
    model.set_fields(b=0.01 * np.cos(z))
    model.advance(4)
    compile_counter.reset()
    model.advance(4)
    assert compile_counter.count == 0


def test_full_model_advances_treedef_stable():
    model = nh.Model(grid=make_grid(), dt=DT)
    _, y, z = grid_coords()
    model.set_fields(u=0.01 * np.sin(y), b=0.01 * np.cos(z))
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(4)
    assert jax.tree_util.tree_structure(model._carry) == before


# ================================================================
#  The CONSTRAINT projection reduces div(u) to machine zero
# ================================================================
def test_projection_drives_divergence_to_machine_zero():
    model = nh.Model(grid=make_grid(), dt=DT, advection=False)
    x, y, z = grid_coords()
    # a non-divergence-free velocity IC
    model.set_fields(u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x),
                     w=0.2 * np.sin(z))
    model.advance(1)
    assert np.abs(divergence(model)).max() < 1e-12


def test_energy_stays_bounded():
    model = nh.Model(grid=make_grid(), dt=DT, advection=False)
    _, y, z = grid_coords()
    model.set_fields(u=0.05 * np.sin(y), b=0.05 * np.cos(z))
    energies = []
    for _ in range(40):
        model.advance(1)
        e = float(np.asarray(model.diagnostics.ekin().data).sum())
        energies.append(e)
    energies = np.asarray(energies)
    assert np.isfinite(energies).all()
    # inertia-gravity waves oscillate; energy never runs away
    assert energies.max() < 5.0 * (energies[:5].max() + 1e-9)


# ================================================================
#  Eigenmode biorthogonality and dispersion
# ================================================================
def test_eigenmode_biorthogonality():
    em = nh.eigenmodes.from_model(nh.Model(grid=make_grid(), dt=DT))
    mask = np.asarray(em._nonzero_mask())

    def pq(p, q):
        arr = sum(np.conj(np.asarray(p[c])) * np.asarray(q[c])
                  for c in "uvwb")
        return np.asarray(arr)[mask]

    for s in (0, 1, -1):
        d = pq(em.p(s), em.q(s))
        # a projector: <p^s, q^s> is 0 (degenerate mode) or 1
        assert np.abs(d * (d - 1.0)).max() < 1e-9
        assert np.abs(d - 1.0).min() < 1e-9   # some mode is represented
    for s, t in [(0, 1), (0, -1), (1, -1), (1, 0), (-1, 0)]:
        assert np.abs(pq(em.p(s), em.q(t))).max() < 1e-9


def test_eigenmode_dispersion_continuous_limit():
    em = nh.eigenmodes.Eigenmodes(make_grid(), f0=1.0, n2=1.0,
                                  dsqr=1.0)
    # f0 = n2 = dsqr = 1 -> omega = sqrt((kz^2 + kh^2)/k^2) = 1
    om = em.omega_at((0.03, 0.0, 0.03), 1)
    assert abs(om.real - 1.0) < 1e-2
    assert em.omega_at((0.5, 0.3, 0.2), -1) == pytest.approx(
        -em.omega_at((0.5, 0.3, 0.2), 1))


def test_eigenmode_projector_is_idempotent():
    em = nh.eigenmodes.Eigenmodes(make_grid(), f0=1.0, n2=1.0,
                                  dsqr=1.0)
    proj = em.projector(0)
    rng = np.random.default_rng(0)
    shape = np.asarray(em.q(0)["u"]).shape
    z = {c: rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
         for c in "uvwb"}
    once = proj(z)
    twice = proj(once)
    for c in "uvwb":
        assert np.abs(np.asarray(twice[c]) - np.asarray(once[c])).max(
        ) < 1e-9


# ================================================================
#  Provides-implies-constancy and the Velocity-role rules
# ================================================================
def test_fplane_provides_coriolis_f0_betaplane_does_not():
    grid = make_grid()
    fp = nh.Model(grid=grid, dt=DT, advection=False)
    assert CORIOLIS_F0 in fp.parameters
    # from_model succeeds on the f-plane
    nh.eigenmodes.from_model(fp)

    bp = nh.Model(grid=grid, dt=DT, advection=False,
                  coriolis=BetaPlaneCoriolis(f0=1.0, beta=0.5))
    assert CORIOLIS_F0 not in bp.parameters
    with pytest.raises(ValueError, match="constant"):
        nh.eigenmodes.from_model(bp)


def test_velocity_roles_and_pressure_is_role_free():
    table = nh.Model(grid=make_grid(), dt=DT).field_table
    assert Velocity("x") in table["u"].roles
    assert Velocity("z") in table["w"].roles
    assert TRACER in table["b"].roles
    assert ADVECTED in table["b"].roles
    assert table["p"].roles == frozenset()
    sel = table.velocity()
    assert sel.components == ("u", "v", "w")
    assert sel.transverse == ()


def test_stratification_provides_n2_and_dsqr_lives_on_core():
    model = nh.Model(grid=make_grid(), dt=DT)
    assert STRATIFICATION_N2 in model.parameters
    assert DSQR in model.parameters
    assert float(model.parameters[STRATIFICATION_N2]) == 1.0


# ================================================================
#  The State vocabulary
# ================================================================
def test_state_accessors_and_missing_component_hint():
    model = nh.Model(grid=make_grid(), dt=DT)
    st = model.state
    assert st.u is st["u"]
    assert st.b is st["b"]


def test_model_without_stratification_has_no_buoyancy():
    grid = make_grid()
    model = FrModel(
        grid=grid,
        modules=(DynamicalCore(dsqr=1.0),
                 FPlaneCoriolis(f0=1.0),
                 CenteredAdvection()),   # advects w -> coverage lint ok
        time_stepper=AdamBashforth(DT, order=3))
    assert "b" not in model.state.component_names
    with pytest.raises(KeyError, match="stratification module"):
        _ = model.state.b
