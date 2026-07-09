"""Basic-correctness + smoke tests for the framework2 nonhydro port.

Parity vs the OLD model is TICK-2's job (the signed-delta suite); these
are the wave-6 acceptance smoke tests: preset == explicit assembly, a
treedef-stable single-compile run, the CONSTRAINT projection driving
div(u) to machine zero, eigenmode biorthogonality/dispersion, the
"provides implies constancy" and Velocity-role rules, and the State
vocabulary.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.nonhydro2 as nh
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.composed import Divergence
from fridom.framework2.model.eigen import (
    _leray_projector,
    _rest_background,
)
from fridom.framework2.model.model import Model as FrModel
from fridom.framework2.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.framework2.model.roles import ADVECTED, TRACER, Velocity
from fridom.framework2.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.framework2.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
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
def _mode_data(state):
    return {c: np.asarray(state[c].data) for c in "uvwb"}


def _dual_data(q, weights):
    """Rayleigh dual of a data-level column (plus the represented mask)."""
    norm = sum(weights[c] * np.abs(q[c]) ** 2 for c in "uvwb")
    good = norm != 0
    safe = np.where(good, norm, 1.0)
    p = {c: np.where(good, weights[c] * q[c] / safe, 0.0)
         for c in "uvwb"}
    return p, good


def test_eigenmode_biorthogonality():
    # non-trivial dsqr / N^2 so the energy weights genuinely bite
    dsqr, n2 = 2.0, 3.0
    em = nh.eigenmodes.Eigenmodes(make_grid(), f0=1.0, n2=n2, dsqr=dsqr)
    weights = {"u": 1.0, "v": 1.0, "w": dsqr, "b": 1.0 / n2}
    q = {s: _mode_data(em.q(s)) for s in (0, 1, -1)}
    for s in (0, 1, -1):
        p, good = _dual_data(q[s], weights)
        d = sum(np.conj(p[c]) * q[s][c] for c in "uvwb")
        # sum_c conj(p_c) q_c == 1 wherever the mode is represented
        assert np.abs(d[good] - 1.0).max() < 1e-12
        # ... and the degenerate modes are EXACT structural zeros of q
        assert good.any()
        assert (~good).any()
        for c in "uvwb":
            assert np.all(q[s][c][~good] == 0.0)
    for s, t in [(0, 1), (0, -1), (1, -1), (1, 0), (-1, 0)]:
        p, _ = _dual_data(q[s], weights)
        cross = sum(np.conj(p[c]) * q[t][c] for c in "uvwb")
        assert np.abs(cross).max() < 1e-9


def test_eigenmode_projector_reproduces_its_eigenvector():
    # the dual is DERIVED (rayleigh dual under the energy metric), so
    # P(s) q(s) == q(s) — the projector-level replacement of the old
    # raw-entry p-vs-Mq comparison.
    em = nh.eigenmodes.Eigenmodes(make_grid(), f0=1.0, n2=3.0,
                                  dsqr=2.0)
    for s in (0, 1, -1):
        q = em.q(s)
        pq = em.projector(s)(q)
        for c in "uvwb":
            assert np.abs(np.asarray(pq[c].data)
                          - np.asarray(q[c].data)).max() < 1e-12


def test_eigenmode_dispersion_continuum_limit():
    # omega at the lowest resolved wavevector k = (1, 0, 1) approaches
    # the continuum relation (second-order discretization error)
    f0, n2, dsqr = 1.5, 3.0, 2.0
    em = nh.eigenmodes.Eigenmodes(make_grid(n=16), f0=f0, n2=n2,
                                  dsqr=dsqr)
    om = np.broadcast_to(np.asarray(em.omega(1).data), (9, 16, 16))
    kh2 = kz2 = 1.0  # unit fundamental on the 2*pi box
    expect = np.sqrt((f0**2 * kz2 + n2 * kh2) / (dsqr * kh2 + kz2))
    assert abs(om[1, 0, 1] - expect) / expect < 0.05
    # the branches are symmetric and the geostrophic one is zero
    assert np.allclose(np.asarray(em.omega(-1).data), -np.asarray(om))
    assert np.abs(np.asarray(em.omega(0).data)).max() == 0.0


def test_eigenmode_projector_is_idempotent():
    em = nh.eigenmodes.Eigenmodes(make_grid(), f0=1.0, n2=1.0,
                                  dsqr=1.0)
    rng = np.random.default_rng(0)
    template = em.q(0)
    shape = np.asarray(template["u"].data).shape
    z = State({c: template[c].with_data(jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)))
        for c in "uvwb"})
    for s in (0, 1, -1):
        proj = em.projector(s)
        once = proj(z)
        twice = proj(once)
        for c in "uvwb":
            assert np.abs(np.asarray(twice[c].data)
                          - np.asarray(once[c].data)).max() < 1e-9


def test_tendency_eigenrelation_lq_equals_i_omega_q():
    # the strong operator-level check: for the linearized tendency
    # composed with the Leray projector, L q(s) = i omega(s) q(s).
    grid = make_grid()
    model = FrModel(
        grid=grid,
        modules=(
            DynamicalCore(dsqr=1.0, rossby_number=1.0),
            FPlaneCoriolis(f0=1.0),
            ConstantStratification(n2=1.0),
            CenteredAdvection()),
        time_stepper=AdamBashforth(DT, order=3))
    em = nh.eigenmodes.from_model(model)
    kit = em._kit
    lin = fr.linearize(model)
    prog, base0 = _rest_background(lin, 0.0)
    leray = np.asarray(
        _leray_projector(lin, base0, prog, jnp.asarray(0.0)))
    rng = np.random.default_rng(5)
    for s in (0, 1, -1):
        q = em.q(s)
        shape = np.asarray(q["u"].data).shape
        amp = (rng.standard_normal(shape)
               + 1j * rng.standard_normal(shape))
        if s != 0:
            # on a REAL physical field the kx = 0 / Nyquist planes of
            # the rfft layout are Hermitian-mixed with the opposite
            # wave branch; probe the wave branches on interior kx only
            amp[0] = 0.0
            amp[-1] = 0.0
        coeff = State({
            c: q[c].with_data(jnp.asarray(np.asarray(q[c].data) * amp))
            for c in "uvwb"})
        phys = base0.replace(**{
            c: base0[c].with_data(kit.backward(c)(coeff[c]).data)
            for c in "uvwb"})
        # the honest reference: the round-tripped coefficients
        zeta = {c: np.asarray(kit.forward(c)(phys[c]).data)
                for c in "uvwb"}
        tau = lin.tendency(phys, t=0.0, constraints=False)
        tau_hat = np.stack(
            [np.fft.fftn(np.asarray(tau[c].data)) for c in prog],
            axis=-1)
        ptau = np.einsum("...ij,...j->...i", leray, tau_hat)
        omega = np.broadcast_to(np.asarray(em.omega(s).data), shape)
        scale = max(np.abs(zeta[c]).max() for c in "uvwb")
        for i, c in enumerate(prog):
            got = np.asarray(kit.forward(c)(phys[c].with_data(
                jnp.asarray(np.fft.ifftn(ptau[..., i]).real))).data)
            want = 1j * omega * zeta[c]
            assert np.abs(got - want).max() / scale < 1e-11


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


def test_betaplane_advances_with_a_profile_f_of_y():
    # the shared fr.modules beta-plane carries the rotation term as
    # pure field arithmetic (no extra_halo raw-.data bypass); prove it
    # assembles AND advances treedef-stably on a real Profile("y") f(y)
    cor = BetaPlaneCoriolis(f0=1.0, beta=0.5)
    assert cor.extra_halo is None
    model = nh.Model(grid=make_grid(), dt=DT, advection=False,
                     coriolis=cor)
    # the auxiliary Coriolis field genuinely varies in y
    fc = np.asarray(model.state["f_coriolis"].data)
    assert fc.std() > 0.0
    _, y, z = grid_coords()
    model.set_fields(u=0.01 * np.sin(y), b=0.01 * np.cos(z))
    before = jax.tree_util.tree_structure(model._carry)
    model.advance(4)
    assert jax.tree_util.tree_structure(model._carry) == before
    assert np.isfinite(np.asarray(model.state["u"].data)).all()


def test_coriolis_is_the_shared_framework_module():
    assert nh.FPlaneCoriolis is fr.modules.FPlaneCoriolis
    assert nh.BetaPlaneCoriolis is fr.modules.BetaPlaneCoriolis
    model = nh.Model(grid=make_grid(), dt=DT)
    coriolis_modules = [
        m for m in model._carry.modules
        if isinstance(m, fr.modules.FPlaneCoriolis)]
    assert len(coriolis_modules) == 1


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
    assert st.v is st["v"]
    assert st.w is st["w"]
    assert st.b is st["b"]
    # the parameter-free vorticity diagnostic lives on the State
    assert st.rel_vort_z.function_space is not None


def test_bound_diagnostics_evaluate_on_the_carry():
    model = nh.Model(grid=make_grid(), dt=DT)
    epot = model.diagnostics.epot()
    pv = model.diagnostics.linear_pot_vort()
    assert bool(np.all(np.isfinite(np.asarray(epot.data))))
    assert bool(np.all(np.isfinite(np.asarray(pv.data))))


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
