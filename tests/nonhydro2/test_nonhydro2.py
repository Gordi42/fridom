"""Basic-correctness + smoke tests for the nonhydro2 port.

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

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.eigen import (
    _leray_projector,
    _rest_background,
)
from fridom.model.model import Model as FrModel
from fridom.model.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)
from fridom.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.model.roles import ADVECTED, TRACER, Velocity
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
    MeridionalStratification,
)
from fridom.nonhydro2.params import DSQR
from fridom.nonhydro2.state import State
from fridom.spatial.bc import BC
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.composed import Divergence
from fridom.spatial.spaces.nodal import NodeSet

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
    lin = fr.model.linearize(model)
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
#  Even-grid Nyquist strata: the steady decomposition is complete
# ================================================================
@pytest.fixture(scope="module")
def nyquist_setup():
    """One linear periodic model + eigenmodes + Leray probe."""
    model = FrModel(
        grid=make_grid(),
        modules=(
            DynamicalCore(dsqr=2.0, rossby_number=1.0),
            FPlaneCoriolis(f0=1.5),
            ConstantStratification(n2=3.0)),
        time_stepper=AdamBashforth(DT, order=3))
    em = nh.eigenmodes.from_model(model)
    lin = fr.model.linearize(model)
    prog, base0 = _rest_background(lin, 0.0)
    leray = np.asarray(
        _leray_projector(lin, base0, prog, jnp.asarray(0.0)))
    return model, em, lin, prog, base0, leray


def test_nyquist_vortical_columns_are_steady(nyquist_setup):
    # THE per-stratum strong test: every internal vortical column
    # (the horizontal-Nyquist divergence-free stratum merged into
    # the main column, plus the overturning and pure-buoyancy
    # columns of the doubly degenerate strata) is exactly steady
    # under the Leray-projected linearized tendency
    _, em, lin, prog, base0, leray = nyquist_setup
    kit = em.kit
    columns = em._columns(0)
    assert len(columns) == 3
    rng = np.random.default_rng(7)
    template = em.q(0)
    shape = np.asarray(template["u"].data).shape
    for col in columns:
        amp = (rng.standard_normal(shape)
               + 1j * rng.standard_normal(shape))
        coeff = {c: np.broadcast_to(
            np.asarray(col[c].data), shape) * amp for c in "uvwb"}
        state = State({c: template[c].with_data(jnp.asarray(coeff[c]))
                       for c in "uvwb"})
        phys = base0.replace(**{
            c: base0[c].with_data(kit.backward(c)(state[c]).data)
            for c in "uvwb"})
        tau = lin.tendency(phys, t=0.0, constraints=False)
        tau_hat = np.stack(
            [np.fft.fftn(np.asarray(tau[c].data)) for c in prog],
            axis=-1)
        ptau = np.einsum("...ij,...j->...i", leray, tau_hat)
        scale = max(np.abs(coeff[c]).max() for c in "uvwb")
        assert np.abs(ptau).max() / scale < 1e-12


def test_nyquist_columns_are_mutually_m_orthogonal(nyquist_setup):
    # the three steady columns and the wave columns are mutually
    # M-orthogonal on the shared strata (the summed rank-1
    # projectors stay an orthogonal projector)
    _, em, *_ = nyquist_setup
    weights = {"u": 1.0, "v": 1.0, "w": 2.0, "b": 1.0 / 3.0}
    shape = np.asarray(em.q(0)["u"].data).shape
    columns = [{c: np.broadcast_to(np.asarray(col[c].data), shape)
                for c in "uvwb"} for col in em._columns(0)]
    columns += [{c: np.broadcast_to(np.asarray(em.q(s)[c].data),
                                    shape) for c in "uvwb"}
                for s in (1, -1)]
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            na = np.sqrt(sum(weights[c] * np.abs(a[c]) ** 2
                             for c in "uvwb"))
            nb = np.sqrt(sum(weights[c] * np.abs(b[c]) ** 2
                             for c in "uvwb"))
            cross = np.abs(sum(weights[c] * np.conj(a[c]) * b[c]
                               for c in "uvwb"))
            denom = np.where((na * nb) == 0, 1.0, na * nb)
            assert (cross / denom).max() < 1e-13


@pytest.mark.parametrize(("s", "indices"), [
    pytest.param(0, {"x": N // 2, "y": 1, "z": 2},
                 id="steady-nyq-x"),
    pytest.param(0, {"x": 1, "y": N // 2, "z": 0},
                 id="steady-nyq-y-kz0"),
    pytest.param(0, {"x": N // 2, "y": N // 2, "z": 3},
                 id="steady-nyq-xy"),
    pytest.param(0, {"x": N // 2, "y": 2, "z": N // 2},
                 id="steady-nyq-xz"),
    pytest.param(1, {"x": N // 2, "y": 1, "z": 2},
                 id="gravity-nyq-x"),
    pytest.param(-1, {"x": 2, "y": N // 2, "z": 1},
                 id="gravity-nyq-y"),
    pytest.param(1, {"x": 3, "y": 2, "z": N // 2},
                 id="rotational-nyq-z"),
])
def test_nyquist_modes_satisfy_the_strong_eigen_relation(
        nyquist_setup, s, indices):
    # d/dt state(phase) == omega * state(phase + pi/2) through the
    # linearized Leray-projected tendency, per Nyquist stratum
    _, em, lin, _prog, base0, _leray = nyquist_setup
    omega, z0 = em.mode(s, indices)
    _, z1 = em.mode(s, indices, phase=np.pi / 2)
    phys = base0.replace(**{
        c: base0[c].with_data(z0[c].data) for c in "uvwb"})
    tau = lin.tendency(phys, t=0.0, constraints=True)
    residual = max(
        float(np.abs(np.asarray(tau[c].data)
                     - omega * np.asarray(z1[c].data)).max())
        for c in "uvwb")
    assert residual < 1e-12 * (1.0 + abs(omega))
    if s == 0:
        assert omega == 0.0
    else:
        assert omega != 0.0


def test_partition_of_unity_covers_the_nyquist_strata(nyquist_setup):
    # sum_s P(s) == identity on a Leray-projected coefficient state
    # once the pre-existing exclusions (the kh = 0 inertial u/v
    # strata and the k = 0 mean) are removed — in particular the
    # even-grid Nyquist strata are fully covered
    model, em, *_ = nyquist_setup
    kit = em.kit
    rng = np.random.default_rng(5)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in "uvwb"})
    zc = model.constrain(
        State({c: model.state[c] for c in "uvwb"}))
    template = em.q(0)
    coeff = {c: np.array(kit.forward(c)(zc[c]).data) for c in "uvwb"}
    for c in ("u", "v"):
        coeff[c][0, 0, :] = 0.0
    for c in ("w", "b"):
        coeff[c][0, 0, 0] = 0.0
    z = State({c: template[c].with_data(jnp.asarray(coeff[c]))
               for c in "uvwb"})
    out = None
    for s in (0, 1, -1):
        part = em.projector(s)(z)
        out = part if out is None else State(
            {c: out[c] + part[c] for c in "uvwb"})
    scale = max(np.abs(coeff[c]).max() for c in "uvwb")
    for c in "uvwb":
        assert np.abs(np.asarray(out[c].data)
                      - coeff[c]).max() / scale < 1e-12


def test_nyquist_strata_agree_with_the_numeric_eigenpairs(
        nyquist_setup):
    # numeric oracle: on a horizontal-Nyquist mode the constrained
    # spectrum is {-omega, 0, 0, +omega} with the analytic gravity
    # frequency, the analytic steady stratum lies in the numeric
    # zero eigenspace, and the doubly degenerate strata are
    # all-steady
    model, em, *_ = nyquist_setup
    ne = fr.model.numeric_eigenpairs(model)
    omega = np.asarray(ne.omega)
    q = np.asarray(ne.q)
    wts = np.asarray(ne.weights)
    assert ne.components == ("u", "v", "w", "b")
    shape = np.asarray(em.q(0)["u"].data).shape
    table = np.broadcast_to(
        np.real(np.asarray(em.omega(1).data)), shape)
    q0 = {c: np.broadcast_to(np.asarray(em.q(0)[c].data), shape)
          for c in "uvwb"}
    nyq = N // 2
    for pt in ((nyq, 3, 2), (nyq, 0, 2), (nyq, nyq, 1),
               (2, nyq, 0)):
        want = float(table[pt])
        assert want > 0.0
        np.testing.assert_allclose(
            np.sort(omega[pt]), [-want, 0.0, 0.0, want], atol=1e-9)
        # the analytic steady stratum lies in the numeric
        # zero-frequency eigenspace
        zero = np.abs(omega[pt]) < 1e-9
        qz = q[pt][:, zero]
        cand = np.array([q0[c][pt] for c in "uvwb"])
        coef = qz.conj().T @ (wts * cand)
        assert np.abs(cand - qz @ coef).max() \
            < 1e-11 * np.abs(cand).max()
    for pt in ((nyq, 3, N // 2), (nyq, nyq, N // 2)):
        # doubly degenerate: the wave pair collapses, everything
        # is steady (three physical modes + the constraint zero)
        np.testing.assert_allclose(omega[pt], 0.0, atol=1e-9)


def test_doubly_degenerate_wave_modes_are_taught_errors(
        nyquist_setup):
    # on the doubly degenerate strata the wave pair collapses to
    # omega = 0 and its columns vanish structurally: the steady
    # content lives in the vortical family instead
    _, em, *_ = nyquist_setup
    with pytest.raises(ValueError, match="structurally"):
        em.mode(1, {"x": N // 2, "y": 1, "z": N // 2})
    # ... while the vortical accessor exposes the primary
    # (divergence-free) stratum there
    omega, _ = em.mode(0, {"x": N // 2, "y": 1, "z": N // 2})
    assert omega == 0.0


def test_odd_grid_columns_are_bitwise_the_composed_formula():
    # regression: an odd grid has no Nyquist stratum — the vortical
    # family is the single composed column, bitwise
    grid = Grid(tuple(
        IntervalMesh(9, (0.0, 2 * np.pi), periodic=True, name=name)
        for name in ("x", "y", "z")))
    em = nh.eigenmodes.Eigenmodes(grid, f0=1.5, n2=3.0, dsqr=2.0)
    columns = em._columns(0)
    assert len(columns) == 1
    x, y, z = em._axes
    k, kb, a, ab = em.k, em.kb, em.a, em.ab
    old = {
        "u": -(a[x] @ (ab[y] @ k[y]) @ ab[z]),
        "v": a[y] @ (ab[x] @ k[x]) @ ab[z],
        "w": em.omega(0),
        "b": 1.5 * (a[x].magnitude ** 2
                    * a[y].magnitude ** 2 * kb[z]),
    }
    for c in "uvwb":
        assert np.array_equal(np.asarray(old[c].data),
                              np.asarray(columns[0][c].data))


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
    # the shared fr.model.modules beta-plane carries the rotation term as
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
    assert nh.FPlaneCoriolis is fr.model.modules.FPlaneCoriolis
    assert nh.BetaPlaneCoriolis is fr.model.modules.BetaPlaneCoriolis
    model = nh.Model(grid=make_grid(), dt=DT)
    coriolis_modules = [
        m for m in model._carry.modules
        if isinstance(m, fr.model.modules.FPlaneCoriolis)]
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


# ================================================================
#  Walled (rigid-lid) linear runs: topology-driven walls (C8)
# ================================================================
LZ = 1.0


def make_walled_grid(n=N, lz=LZ):
    # x, y periodic; z bounded (rigid lids) — periodicity is the
    # ONLY wall switch, everything else is derived
    meshes = (
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x"),
        IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="y"),
        IntervalMesh(n, (0.0, lz), periodic=False, name="z"),
    )
    return Grid(meshes), meshes


def walled_coords(n=N, lz=LZ):
    hor = (np.arange(n) + 0.5) * (2 * np.pi / n)
    ver = (np.arange(n) + 0.5) * (lz / n)
    return np.meshgrid(hor, hor, ver, indexing="ij")


def make_walled_model(**kwargs):
    grid, _ = make_walled_grid()
    return nh.Model(grid=grid, dt=DT, advection=False, **kwargs)


def test_walled_grid_derives_the_wall_spaces():
    grid, (_, _, mz) = make_walled_grid()
    model = nh.Model(grid=grid, dt=DT, advection=False)
    # w: Dirichlet on its own bounded component axis (impermeability)
    w_z = model.state["w"].function_space.bare.factor("z")
    assert w_z is mz.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    # u, v, p, b stay BC-free (their parities are physics-derived)
    for name in ("u", "v", "p", "b"):
        z = model.state[name].function_space.bare.factor("z")
        assert z is mz.nodal(NodeSet.CENTER, bc=BC.NONE)


def test_walled_linear_run_is_treedef_stable_and_projected():
    model = make_walled_model()
    x, y, _ = walled_coords()
    # a non-divergence-free IC excites the projection genuinely
    model.set_fields(u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x))
    before = jax.tree_util.tree_structure(model._carry)
    for _ in range(5):
        model.advance(1)
        # the CONSTRAINT stage drives div(u) to machine zero
        assert np.abs(divergence(model)).max() < 1e-12
    assert jax.tree_util.tree_structure(model._carry) == before


def test_walled_second_advance_compiles_nothing(compile_counter):
    model = make_walled_model()
    _, _, z = walled_coords()
    model.set_fields(b=0.01 * np.cos(np.pi * z / LZ))
    model.advance(4)
    compile_counter.reset()
    model.advance(4)
    assert compile_counter.count == 0


def test_walled_energy_stays_bounded():
    model = make_walled_model()
    _, y, z = walled_coords()
    model.set_fields(u=0.05 * np.sin(y),
                     b=0.05 * np.cos(np.pi * z / LZ))
    energies = []
    for _ in range(40):
        model.advance(1)
        e = float(np.asarray(model.diagnostics.ekin().data).sum())
        energies.append(e)
    energies = np.asarray(energies)
    assert np.isfinite(energies).all()
    assert energies.max() < 5.0 * (energies[:5].max() + 1e-9)


def test_walled_diagnostics_smoke():
    model = make_walled_model()
    _, y, z = walled_coords()
    model.set_fields(u=0.01 * np.sin(y),
                     b=0.01 * np.cos(np.pi * z / LZ))
    model.advance(1)
    # linear_pot_vort interpolates BC-free staggered derivatives
    # back to centers through the one-sided diagnostics closure
    # (R2, boundary_plan.md 2d)
    for name in ("ekin", "epot", "linear_pot_vort"):
        field = getattr(model.diagnostics, name)()
        assert bool(np.isfinite(np.asarray(field.data)).all())


def test_walled_default_model_assembles_with_advection():
    # the default preset (advection=True: CenteredAdvection, which
    # is walled-capable through the structural-zero wall fluxes)
    # assembles and steps on the rigid-lid grid; the biased schemes
    # keep their taught rejection (test_advection.py)
    grid, _ = make_walled_grid()
    model = nh.Model(grid=grid, dt=DT)  # default advection module
    _, y, z = walled_coords()
    model.set_fields(u=0.05 * np.sin(y),
                     b=0.05 * np.cos(np.pi * z / LZ))
    model.advance(2)
    assert not model.panicked
    assert all(
        bool(np.isfinite(np.asarray(model.state[c].data)).all())
        for c in ("u", "v", "w", "b"))


# ================================================================
#  MeridionalStratification: the varying N^2(y) module type
# ================================================================
def make_walled_y_grid(n=N):
    mx = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, 1.0), periodic=False, name="y")
    mz = IntervalMesh(n, (0.0, 2 * np.pi), periodic=True, name="z")
    return Grid((mx, my, mz))


def test_meridional_stratification_rejects_a_constant():
    with pytest.raises(TypeError, match="ConstantStratification"):
        MeridionalStratification(n2=2.0)


def test_meridional_stratification_declares_the_profile():
    # the two-type precedent: the meridional-profile n2 field is
    # materialized from the callable; provides-implies-constancy
    # means the constant scalar is absent
    model = nh.Model(
        grid=make_walled_y_grid(), dt=DT, advection=False,
        stratification=MeridionalStratification(
            n2=lambda y: 1.0 + 2.0 * y * y))
    assert STRATIFICATION_N2 not in model.parameters
    n2 = model.state["n2"]
    centres = (np.arange(N) + 0.5) / N
    np.testing.assert_allclose(
        np.asarray(n2.data).ravel(), 1.0 + 2.0 * centres ** 2)


def test_meridional_constant_profile_tendency_matches_constant():
    # N^2(y) = n0: the profile broadcast multiplies pointwise, so
    # the coupling terms agree with ConstantStratification bitwise
    n0 = 3.0
    varying = nh.Model(
        grid=make_walled_y_grid(), dt=DT, advection=False,
        stratification=MeridionalStratification(
            n2=lambda y: n0 + 0.0 * y))
    constant = nh.Model(
        grid=make_walled_y_grid(), dt=DT, advection=False,
        stratification=ConstantStratification(n2=n0))
    rng = np.random.default_rng(7)
    fields = {c: rng.standard_normal(
        np.asarray(constant.state[c].data).shape)
        for c in ("u", "v", "w", "b")}
    varying.set_fields(**fields)
    constant.set_fields(**fields)
    tv = varying.tendency(varying.state)
    tc = constant.tendency(constant.state)
    for c in ("u", "v", "w", "b"):
        np.testing.assert_allclose(
            np.asarray(tv[c].data), np.asarray(tc[c].data),
            rtol=0.0, atol=0.0)


# ================================================================
#  function(f, s): scalar functions of the linear operator
# ================================================================
@pytest.fixture(scope="module")
def function_setup():
    """One linear periodic model + eigenmodes for the f(L) tests."""
    model = nh.Model(
        grid=make_grid(), dt=DT, advection=False, dsqr=2.0,
        coriolis=FPlaneCoriolis(f0=1.5),
        stratification=ConstantStratification(n2=3.0))
    return model, nh.eigenmodes.from_model(model)


def _random_coeff_state(em, seed):
    rng = np.random.default_rng(seed)
    template = em.q(0)
    shape = np.asarray(template["u"].data).shape
    return State({c: template[c].with_data(jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)))
        for c in "uvwb"})


@pytest.mark.parametrize("sel", [
    pytest.param(0, id="vortical"),
    pytest.param(1, id="plus"),
    pytest.param((1, -1), id="wave-pair"),
])
def test_function_with_unit_f_reproduces_the_projectors(
        function_setup, sel):
    # f == 1 on a branch selection is exactly the summed projectors
    # (weights 1 on the represented modes, 0 on the structural
    # zeros — where the projector amplitude is exactly 0), bitwise
    _, em = function_setup
    z = _random_coeff_state(em, seed=41)
    branches = (sel,) if isinstance(sel, int) else sel
    want = None
    for s in branches:
        part = em.projector(s)(z)
        want = part if want is None else State(
            {c: want[c] + part[c] for c in "uvwb"})
    got = em.function(np.ones_like, sel)(z)
    for c in "uvwb":
        assert np.array_equal(np.asarray(got[c].data),
                              np.asarray(want[c].data))


def test_function_inverse_wave_strong_test(function_setup):
    # THE STRONG TEST: with invL = function(1/(i omega), (1, -1)),
    # L(invL(z)) == P_wave(z) through the linearized Leray-projected
    # tendency (constraints=True; invL(z) lies in the wave span, so
    # its synthesis is already divergence-free), asserted in
    # coefficient space with the self-conjugate kx planes zeroed
    model, em = function_setup
    kit = em.kit
    lin = fr.model.linearize(model)
    prog, base0 = _rest_background(lin, 0.0)
    rng = np.random.default_rng(42)
    template = em.q(0)
    shape = np.asarray(template["u"].data).shape

    def make_amp():
        amp = (rng.standard_normal(shape)
               + 1j * rng.standard_normal(shape))
        amp[0] = 0.0
        amp[-1] = 0.0
        return amp

    z = State({c: template[c].with_data(jnp.asarray(make_amp()))
               for c in "uvwb"})
    w_hat = em.function(lambda om: 1.0 / (1j * om), (1, -1))(z)
    nodal = {c: kit.backward(c)(w_hat[c]) for c in prog}
    phys = base0.replace(**{
        c: base0[c].with_data(nodal[c].data) for c in prog})
    tau = lin.tendency(phys, t=0.0, constraints=True)
    plus = em.projector(1)(z)
    minus = em.projector(-1)(z)
    scale = max(float(np.abs(np.asarray(z[c].data)).max())
                for c in "uvwb")
    for c in prog:
        got = np.asarray(kit.forward(c)(
            phys[c].with_data(tau[c].data)).data)
        want = np.asarray(plus[c].data) + np.asarray(minus[c].data)
        assert np.abs(got - want).max() / scale < 1e-12


def test_function_inverse_wave_is_real_safe(function_setup):
    # 1/(i omega) satisfies f(-omega) == conj(f(omega)) and (1, -1)
    # is conjugation-closed: the coefficients of a real state stay
    # Hermitian and the backward synthesis stays real
    model, em = function_setup
    kit = em.kit
    rng = np.random.default_rng(43)
    z = State({
        c: kit.forward(c)(model.state[c].with_data(jnp.asarray(
            rng.standard_normal(model.state[c].data.shape))))
        for c in "uvwb"})
    out = em.function(lambda om: 1.0 / (1j * om), (1, -1))(z)
    for c in "uvwb":
        back = np.asarray(kit.backward(c)(out[c]).data)
        scale = float(np.abs(back).max())
        assert np.abs(np.imag(back)).max() < 1e-15 * scale


def test_function_structural_zero_guard(function_setup):
    # the geostrophic branch is represented with omega == 0: a
    # singular f is a taught error — while the wave branches carry
    # their zeros (the k = 0 mean, k_h = 0 strata) as STRUCTURAL
    # zeros of the column, which never reach f, and pass
    _, em = function_setup
    with pytest.raises(ValueError, match=r"s=0"):
        em.function(lambda om: 1.0 / (1j * om), 0)
    assert callable(em.function(lambda om: 1.0 / (1j * om), (1, -1)))
    with pytest.raises(ValueError, match="branches"):
        em.function(np.ones_like, 2)
