"""Discrete shallow-water eigenmodes: dispersion, vectors, projector.

The C6 operator-sourced rewrite: the dispersion is the DISCRETE
relation (validated against the numeric ``eigh(iML, M)`` probe to
machine precision), the eigenvector columns satisfy the strong
operator-level relation ``L q(s) = i omega(s) q(s)`` for the
linearized tendency (shallow water carries no constraint stage), and
the Rayleigh dual under the energy metric is biorthonormal with
exact structural zeros on the interpolation-Nyquist geostrophic
nullspace. The ``k = 0`` mean carries the patched inertial triple
``{geostrophic pressure, (-i s, 1, 0)}`` — complete and M-orthogonal.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.framework2.model.eigen import _rest_background

from .conftest import N, make_grid, make_model

COMPONENTS = ("u", "v", "p")


def _eig(f0=1.0, csqr=1.0):
    model = make_model(csqr=csqr, f0=f0)
    return sw.eigenmodes.from_model(model), model


def _full_omega(em, s, n=N):
    """Unfold the rfft half-lattice omega onto the full fftn lattice."""
    half = np.broadcast_to(np.asarray(em.omega(s).data),
                           (n // 2 + 1, n))
    fold = np.minimum(np.arange(n), n - np.arange(n))
    return half[fold]


def _mode_data(state):
    return {c: np.asarray(state[c].data) for c in COMPONENTS}


# ================================================================
#  Dispersion relation
# ================================================================
def test_geostrophic_mode_has_zero_frequency():
    em, _ = _eig()
    assert float(np.abs(np.asarray(em.omega(0).data)).max()) == 0.0


def test_branches_are_symmetric_and_inertial_at_the_mean():
    f0 = 1.5
    em, _ = _eig(f0=f0, csqr=2.0)
    plus = np.broadcast_to(np.asarray(em.omega(1).data),
                           (N // 2 + 1, N))
    minus = np.broadcast_to(np.asarray(em.omega(-1).data),
                            (N // 2 + 1, N))
    np.testing.assert_allclose(minus, -plus, atol=1e-14)
    # the k = 0 mean is the physical inertial pair +/- f0 (no masking)
    assert plus[0, 0] == pytest.approx(f0)


@pytest.mark.parametrize(("f0", "csqr"), [(1.0, 1.0), (0.5, 4.0)])
def test_dispersion_matches_the_numeric_eigensolve(f0, csqr):
    # the analytic DISCRETE dispersion reproduces the energy-metric
    # eigh(iML, M) spectrum on EVERY mode (k = 0 included) to machine
    # precision.
    em, model = _eig(f0=f0, csqr=csqr)
    ne = fr.numeric_eigenpairs(model)
    omega = np.asarray(ne.omega)
    analytic = np.sort(np.stack(
        [_full_omega(em, s) for s in (-1, 0, 1)], axis=-1), axis=-1)
    assert np.abs(omega - analytic).max() < 1e-9


def test_dispersion_continuum_limit_at_the_fundamental():
    # omega at the lowest resolved wavevector k = (2 pi, 0) approaches
    # the continuum relation (second-order discretization error)
    f0, csqr = 1.5, 2.0
    em, _ = _eig(f0=f0, csqr=csqr)
    om = np.broadcast_to(np.asarray(em.omega(1).data),
                         (N // 2 + 1, N))
    kx = 2.0 * np.pi  # the unit box fundamental
    expect = np.sqrt(f0 ** 2 + csqr * kx ** 2)
    assert abs(om[1, 0] - expect) / expect < 3e-2


# ================================================================
#  The strong operator-level eigenrelation
# ================================================================
def test_tendency_eigenrelation_lq_equals_i_omega_q():
    # for the linearized tendency (no constraint stage in shallow
    # water), L q(s) = i omega(s) q(s) — asserted through physical
    # space on seeded random amplitudes.
    em, model = _eig(f0=1.5, csqr=2.0)
    kit = em._kit
    lin = fr.linearize(model)
    prog, base0 = _rest_background(lin, 0.0)
    assert prog == COMPONENTS
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
        coeff = sw.State({
            c: q[c].with_data(jnp.asarray(np.asarray(q[c].data) * amp))
            for c in prog})
        phys = base0.replace(**{
            c: base0[c].with_data(kit.backward(c)(coeff[c]).data)
            for c in prog})
        # the honest reference: the round-tripped coefficients
        zeta = {c: np.asarray(kit.forward(c)(phys[c]).data)
                for c in prog}
        tau = lin.tendency(phys, t=0.0, constraints=False)
        omega = np.broadcast_to(np.asarray(em.omega(s).data), shape)
        scale = max(np.abs(zeta[c]).max() for c in prog)
        for c in prog:
            got = np.asarray(kit.forward(c)(tau[c]).data)
            want = 1j * omega * zeta[c]
            assert np.abs(got - want).max() / scale < 1e-11


# ================================================================
#  Biorthonormality via the Rayleigh dual
# ================================================================
def test_eigenmode_biorthonormality_and_structural_zeros():
    csqr = 4.0
    em, _ = _eig(csqr=csqr)
    weights = {"u": 1.0, "v": 1.0, "p": 1.0 / csqr}
    q = {s: _mode_data(em.q(s)) for s in (0, 1, -1)}

    def dual(qs):
        norm = sum(weights[c] * np.abs(qs[c]) ** 2 for c in COMPONENTS)
        good = norm != 0
        safe = np.where(good, norm, 1.0)
        p = {c: np.where(good, weights[c] * qs[c] / safe, 0.0)
             for c in COMPONENTS}
        return p, good

    for s in (0, 1, -1):
        p, good = dual(q[s])
        d = sum(np.conj(p[c]) * q[s][c] for c in COMPONENTS)
        # sum_c conj(p_c) q_c == 1 wherever the mode is represented
        assert np.abs(d[good] - 1.0).max() < 1e-12
        if s == 0:
            # the geostrophic column drops the interpolation-Nyquist
            # planes through EXACT structural zeros of every entry
            assert (~good).any()
            for c in COMPONENTS:
                assert np.all(q[s][c][~good] == 0.0)
        else:
            # the patched wave columns are represented at every mode
            assert good.all()
    for s, t in [(0, 1), (0, -1), (1, -1), (1, 0), (-1, 0), (-1, 1)]:
        p, _ = dual(q[s])
        cross = sum(np.conj(p[c]) * q[t][c] for c in COMPONENTS)
        assert np.abs(cross).max() < 1e-9


# ================================================================
#  The projector and the k = 0 inertial triple
# ================================================================
def _random_coeff_state(em, seed=0):
    rng = np.random.default_rng(seed)
    template = em.q(0)
    shape = np.asarray(template["u"].data).shape
    return sw.State({c: template[c].with_data(jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)))
        for c in COMPONENTS})


def test_projector_is_idempotent_and_reproduces_its_eigenvector():
    em, _ = _eig(f0=1.5, csqr=2.0)
    z = _random_coeff_state(em)
    for s in (0, 1, -1):
        proj = em.projector(s)
        once = proj(z)
        twice = proj(once)
        for c in COMPONENTS:
            assert np.abs(np.asarray(twice[c].data)
                          - np.asarray(once[c].data)).max() < 1e-9
        # the dual is DERIVED (rayleigh dual), so P(s) q(s) == q(s)
        q = em.q(s)
        pq = proj(q)
        for c in COMPONENTS:
            assert np.abs(np.asarray(pq[c].data)
                          - np.asarray(q[c].data)).max() < 1e-9


def test_k0_inertial_patch_completes_the_mean_triple():
    f0 = 1.5
    em, _ = _eig(f0=f0, csqr=2.0)
    # the patched inertial entries (u, v, p) = (-i s, 1, 0) at k = 0
    for s in (1, -1):
        q = _mode_data(em.q(s))
        assert q["u"][0, 0] == -1j * s
        assert q["v"][0, 0] == 1.0
        assert q["p"][0, 0] == 0.0
    # the geostrophic mean is the pressure mode
    q0 = _mode_data(em.q(0))
    assert q0["u"][0, 0] == 0.0
    assert q0["v"][0, 0] == 0.0
    assert q0["p"][0, 0] == pytest.approx(f0)
    # M-orthogonality of the k = 0 triple (the +/- pair in particular)
    weights = {"u": 1.0, "v": 1.0, "p": 0.5}
    q = {s: _mode_data(em.q(s)) for s in (0, 1, -1)}
    for s, t in [(1, -1), (0, 1), (0, -1)]:
        inner = sum(weights[c] * np.conj(q[s][c][0, 0]) * q[t][c][0, 0]
                    for c in COMPONENTS)
        assert abs(inner) == 0.0


def test_mode_family_is_complete_off_the_nyquist_nullspace():
    # sum_s P(s) == identity at every mode (the patched k = 0 mean
    # included) except the interpolation-Nyquist planes, where the
    # geostrophic mode is a structural zero of the family.
    em, _ = _eig(f0=1.5, csqr=2.0)
    z = _random_coeff_state(em, seed=3)
    out = None
    for s in (0, 1, -1):
        part = em.projector(s)(z)
        out = part if out is None else sw.State(
            {c: out[c] + part[c] for c in COMPONENTS})
    nyq = N // 2
    mask = np.ones((N // 2 + 1, N), dtype=bool)
    mask[nyq, :] = False
    mask[:, nyq] = False
    for c in COMPONENTS:
        resid = np.abs(np.asarray(out[c].data) - np.asarray(z[c].data))
        assert resid[0, 0] < 1e-12
        assert resid[mask].max() < 1e-12
    # ... and the Nyquist-vortical content is genuinely dropped
    assert max(np.abs(np.asarray(out[c].data)
                      - np.asarray(z[c].data))[~mask].max()
               for c in ("u", "v")) > 1e-2


# ================================================================
#  from_model dispatch: topology first, then parameter validation
# ================================================================
def _walled_model(*, periodic_x=True, coriolis=None):
    mx = fr.grid.meshes.IntervalMesh(8, (0.0, 1.0),
                                     periodic=periodic_x, name="x")
    my = fr.grid.meshes.IntervalMesh(8, (0.0, 1.0), periodic=False,
                                     name="y")
    if coriolis is None:
        coriolis = sw.modules.FPlaneCoriolis(f0=1.0)
    return sw.Model(
        grid=fr.grid.Grid((mx, my)), csqr=1.0, rossby_number=0.2,
        coriolis=coriolis, advection=False,
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


def test_from_model_dispatches_the_walled_channel():
    # a single bounded axis routes to the numeric channel eigenmodes
    # (previously a raw DispatchError out of the analytic kit)
    em = sw.eigenmodes.from_model(_walled_model())
    assert isinstance(em, sw.ChannelEigenmodes)


def test_from_model_routes_a_walled_beta_plane_to_the_channel():
    # topology dispatch precedes the constant-f0 validation: the
    # channel path serves the beta plane numerically
    em = sw.eigenmodes.from_model(_walled_model(
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0)))
    assert isinstance(em, sw.ChannelEigenmodes)


def test_from_model_rejects_a_multi_walled_box():
    with pytest.raises(ValueError,
                       match=r"bounds \('x', 'y'\).*multi-walled"):
        sw.eigenmodes.from_model(_walled_model(periodic_x=False))


# ================================================================
#  from_model structural validation
# ================================================================
def test_from_model_rejects_a_beta_plane():
    grid = make_grid()
    model = sw.Model(
        grid=grid, csqr=1.0,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"coriolis\.f0"):
        sw.eigenmodes.from_model(model)


def test_degenerate_and_non_2d_grids_are_rejected():
    grid = make_grid()
    with pytest.raises(ValueError, match="degenerate"):
        sw.eigenmodes.Eigenmodes(grid, f0=0.0, csqr=0.0)
    mz = fr.grid.meshes.IntervalMesh(4, (0.0, 1.0), periodic=True,
                                     name="z")
    grid3 = fr.grid.Grid((*grid.factors, mz))
    with pytest.raises(ValueError, match="2-D"):
        sw.eigenmodes.Eigenmodes(grid3, f0=1.0, csqr=1.0)
