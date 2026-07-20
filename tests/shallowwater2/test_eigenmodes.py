"""Discrete shallow-water eigenmodes: dispersion, vectors, projector.

The C6 operator-sourced rewrite: the dispersion is the DISCRETE
relation (validated against the numeric ``eigh(iML, M)`` probe to
machine precision), the eigenvector columns satisfy the strong
operator-level relation ``L q(s) = i omega(s) q(s)`` for the
linearized tendency (shallow water carries no constraint stage), and
the Rayleigh dual under the energy metric is biorthonormal. The
``k = 0`` mean carries the patched inertial triple ``{geostrophic
pressure, (-i s, 1, 0)}``; the even-grid interpolation-Nyquist
planes carry the rotation-decoupled steady divergence-free stratum
in the geostrophic column — the family is complete at every mode
(odd grids, which have no Nyquist stratum, stay bitwise on the
composed column).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model.eigen import _rest_background
from fridom.spatial.symbols import rayleigh_dual

from .conftest import N, make_grid, make_model

COMPONENTS = ("u", "v", "p")

#: integer branch -> uniform family-name spelling (the low-level
#: q/omega surface stays integer-indexed; mode() takes families)
FAMILY = {0: "vortical", 1: "wave+", -1: "wave-"}


def _pinned_grid(n=N, *, periodic_x=True, periodic_y=True):
    # device_ids=(0,) keeps every axis local: the analytic eigenmode
    # projections synthesize through the naive (GSPMD) transform, a
    # Tier-1 taught error on a sharded axis (see transform.py). The
    # shared conftest.make_grid is left unpinned (other shallowwater2
    # files run @pytest.mark.multi_device tests on it).
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                        periodic=periodic_x, name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                        periodic=periodic_y, name="y")
    return fr.spatial.Grid((mx, my), device_ids=(0,))


def _eig(f0=1.0, csqr=1.0):
    model = make_model(csqr=csqr, f0=f0, grid=_pinned_grid())
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
    ne = fr.model.numeric_eigenpairs(model)
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
def test_tendency_eigenrelation_lq_equals_minus_i_omega_q():
    # for the linearized tendency (no constraint stage in shallow
    # water), L q(s) = -i omega(s) q(s) — asserted through physical
    # space on seeded random amplitudes.
    em, model = _eig(f0=1.5, csqr=2.0)
    kit = em._kit
    lin = fr.model.linearize(model)
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
            want = -1j * omega * zeta[c]
            assert np.abs(got - want).max() / scale < 1e-11


# ================================================================
#  Biorthonormality via the Rayleigh dual
# ================================================================
def test_eigenmode_biorthonormality_and_completeness():
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
        # every branch is represented at EVERY mode: the k = 0 mean
        # via the inertial patch, the even-grid Nyquist planes via
        # the steady divergence-free stratum of the vortical column
        assert good.all()
        assert np.abs(d - 1.0).max() < 1e-12
    for s, t in [(0, 1), (0, -1), (1, -1), (1, 0), (-1, 0), (-1, 1)]:
        p, _ = dual(q[s])
        cross = sum(np.conj(p[c]) * q[t][c] for c in COMPONENTS)
        assert np.abs(cross).max() < 1e-9


def test_nyquist_steady_stratum_is_the_divergence_free_mode():
    # on the interpolation-Nyquist planes the geostrophic column is
    # the rotation-decoupled steady mode (conj(k_y), -conj(k_x), 0)
    em, _ = _eig(f0=1.5, csqr=2.0)
    x, y = em._axes
    shape = (N // 2 + 1, N)
    ax2 = np.asarray((em.a[x].magnitude ** 2).data)
    ay2 = np.asarray((em.a[y].magnitude ** 2).data)
    mask = np.broadcast_to((ax2 * ay2) == 0.0, shape)
    assert mask.any()
    q0 = _mode_data(em.q(0))
    ky = np.broadcast_to(np.asarray(em.k[y].data), shape)
    kx = np.broadcast_to(np.asarray(em.k[x].data), shape)
    assert np.array_equal(q0["u"][mask], np.conj(ky)[mask])
    assert np.array_equal(q0["v"][mask], -np.conj(kx)[mask])
    assert np.all(q0["p"][mask] == 0.0)


def test_nyquist_steady_stratum_matches_the_numeric_eigenvectors():
    # numeric oracle: at every Nyquist mode the frequency multiset is
    # {-omega, 0, +omega} and the analytic steady stratum spans the
    # numeric zero-frequency eigenvector
    f0, csqr = 1.5, 2.0
    em, model = _eig(f0=f0, csqr=csqr)
    ne = fr.model.numeric_eigenpairs(model)
    omega = np.asarray(ne.omega)
    q = np.asarray(ne.q)
    w = np.asarray(ne.weights)
    q0 = _mode_data(em.q(0))
    nyq = N // 2
    for pt in ((nyq, 3), (nyq, 0), (0, nyq), (nyq, nyq), (2, nyq)):
        om = np.sort(omega[pt])
        khat2 = sum((2.0 * np.sin(np.pi * i / N) * N) ** 2
                    for i in pt)
        gravity = np.sqrt(csqr * khat2)
        np.testing.assert_allclose(
            om, [-gravity, 0.0, gravity], atol=1e-9)
        steady = q[pt][:, np.argmin(np.abs(omega[pt]))]
        cand = np.array([q0[c][pt] for c in COMPONENTS])
        inner = np.sum(w * np.conj(steady) * cand)
        assert np.abs(cand - inner * steady).max() < 1e-12 * np.abs(
            cand).max()


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
    # the patched inertial entries (u, v, p) = (i s, 1, 0) at k = 0
    for s in (1, -1):
        q = _mode_data(em.q(s))
        assert q["u"][0, 0] == 1j * s
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


def test_mode_family_is_a_partition_of_unity_at_every_mode():
    # sum_s P(s) == identity EXACTLY at every mode: the patched
    # k = 0 mean and the even-grid interpolation-Nyquist steady
    # strata included.
    em, _ = _eig(f0=1.5, csqr=2.0)
    z = _random_coeff_state(em, seed=3)
    out = None
    for s in (0, 1, -1):
        part = em.projector(s)(z)
        out = part if out is None else sw.State(
            {c: out[c] + part[c] for c in COMPONENTS})
    for c in COMPONENTS:
        resid = np.abs(np.asarray(out[c].data) - np.asarray(z[c].data))
        assert resid.max() < 1e-12


def test_odd_grid_vortical_column_is_bitwise_the_composed_formula():
    # regression: an odd grid has no Nyquist stratum, so the
    # geostrophic column (and hence the projector action) is
    # bitwise the plain composed pre-Nyquist formula
    n = 9
    em = sw.eigenmodes.Eigenmodes(_pinned_grid(n=n), f0=1.5, csqr=2.0)
    x, y = em._axes
    k, a, ab = em.k, em.a, em.ab
    old = {
        "u": -(a[x] @ (ab[y] @ k[y])),
        "v": a[y] @ (ab[x] @ k[x]),
        "p": 1.5 * (a[x].magnitude ** 2 * a[y].magnitude ** 2),
    }
    new = em._vec_q(0)
    for c in COMPONENTS:
        assert np.array_equal(np.asarray(old[c].data),
                              np.asarray(new[c].data))
    # ... and the projector acts bitwise like the explicit
    # single-column rayleigh-dual formula
    rng = np.random.default_rng(1)
    template = em.q(0)
    shape = np.asarray(template["u"].data).shape
    z = sw.State({c: template[c].with_data(jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)))
        for c in COMPONENTS})
    p = rayleigh_dual(old, {"u": 1.0, "v": 1.0, "p": 0.5})
    amp = sum(jnp.conj(p[c].data) * z[c].data for c in p)
    got = em.projector(0)(z)
    for c in COMPONENTS:
        want = np.asarray(jnp.broadcast_to(old[c].data * amp, shape))
        assert np.array_equal(np.asarray(got[c].data), want)


def test_vortical_projection_is_real_safe_on_the_even_grid():
    # the Nyquist steady stratum is Hermitian-symmetric: the vortical
    # projection of a real physical state stays exactly real through
    # the transform round-trip
    em, model = _eig(f0=1.5, csqr=2.0)
    kit = em._kit
    rng = np.random.default_rng(6)
    z = sw.State({
        c: kit.forward(c)(model.state[c].with_data(jnp.asarray(
            rng.standard_normal(model.state[c].data.shape))))
        for c in COMPONENTS})
    out = em.projector(0)(z)
    for c in COMPONENTS:
        back = kit.backward(c)(out[c])
        data = np.asarray(back.data)
        scale = float(np.abs(data).max())
        assert np.abs(np.imag(data)).max() < 1e-15 * scale
        # Hermitian coefficients: the real synthesis loses nothing
        again = np.asarray(kit.forward(c)(back.real).data)
        assert np.abs(again - np.asarray(out[c].data)).max() \
            < 1e-13 * scale


# ================================================================
#  The mode-indexed accessor (em.mode)
# ================================================================
@pytest.fixture(scope="module")
def mode_setup():
    """One linear periodic model + eigenmodes for the mode tests."""
    model = make_model(csqr=2.0, f0=1.5, advection=False,
                       grid=_pinned_grid())
    return model, sw.eigenmodes.from_model(model)


def _mode_pair(em, s, indices, phase=0.0):
    omega, z0 = em.mode(FAMILY[s], indices, phase=phase)
    _, z1 = em.mode(FAMILY[s], indices, phase=phase + np.pi / 2)
    return omega, z0, z1


@pytest.mark.parametrize(("s", "indices"), [
    pytest.param(1, {"x": 3, "y": 2}, id="plus"),
    pytest.param(-1, {"x": 2, "y": -3}, id="minus-negative-ky"),
    pytest.param(0, {"x": 1, "y": 4}, id="vortical"),
    pytest.param(1, {"x": 0, "y": 0}, id="inertial-mean"),
    pytest.param(0, {"x": N // 2, "y": 3}, id="vortical-nyq-x"),
    pytest.param(0, {"x": 2, "y": N // 2}, id="vortical-nyq-y"),
    pytest.param(0, {"x": N // 2, "y": N // 2},
                 id="vortical-nyq-corner"),
    pytest.param(0, {"x": N // 2, "y": 0}, id="vortical-nyq-axis"),
    pytest.param(1, {"x": N // 2, "y": 3}, id="plus-nyq-x"),
    pytest.param(-1, {"x": N // 2, "y": N // 2},
                 id="minus-nyq-corner"),
])
def test_mode_satisfies_the_strong_eigen_relation(
        mode_setup, s, indices):
    # d/dt state(phase) == omega * state(phase + pi/2) through the
    # REAL model tendency (advection off: the tendency is linear)
    model, em = mode_setup
    omega, z0, z1 = _mode_pair(em, s, indices, phase=0.4)
    tau = model.tendency(z0)
    residual = max(
        float(np.abs(np.asarray(tau[c].data)
                     - omega * np.asarray(z1[c].data)).max())
        for c in COMPONENTS)
    assert residual < 1e-12 * (1.0 + abs(omega))


def test_mode_frequency_matches_the_dispersion_diagonal(mode_setup):
    _, em = mode_setup
    omega, _ = em.mode("wave+", {"x": 3, "y": 2})
    table = np.broadcast_to(np.asarray(em.omega(1).data),
                            (N // 2 + 1, N))
    assert omega == pytest.approx(float(table[3, 2]), rel=1e-14)
    # the mean mode carries the inertial frequency f0
    inertial, _ = em.mode("wave+", {"x": 0, "y": 0})
    assert inertial == pytest.approx(1.5)


def test_mode_projection_keeps_and_annihilates(mode_setup):
    _, em = mode_setup
    _, z = em.mode("wave+", {"x": 3, "y": 2})
    kept = sw.transforms.mode_projection(em, 1)(z)
    assert _absmax_states(kept, z) < 1e-12
    for other in (0, -1):
        killed = sw.transforms.mode_projection(em, other)(z)
        assert max(
            float(np.abs(np.asarray(killed[c].data)).max())
            for c in COMPONENTS) < 1e-12


def _absmax_states(a, b):
    return max(
        float(np.abs(np.asarray(a[c].data)
                     - np.asarray(b[c].data)).max())
        for c in COMPONENTS)


def test_mode_normalization_and_realness(mode_setup):
    # the largest horizontal-velocity envelope is exactly one, the
    # fields are exactly real (Hermitian-closed synthesis), and a
    # pi shift is exactly the negated state
    _, em = mode_setup
    _, z0, z1 = _mode_pair(em, 1, {"x": 3, "y": 2}, phase=0.7)
    peak = max(
        float((np.asarray(z0[c].data) ** 2
               + np.asarray(z1[c].data) ** 2).max())
        for c in ("u", "v"))
    assert peak == pytest.approx(1.0, abs=1e-12)
    for c in COMPONENTS:
        assert not np.iscomplexobj(np.asarray(z0[c].data))
    _, zpi = em.mode("wave+", {"x": 3, "y": 2}, phase=0.7 + np.pi)
    assert _absmax_states(
        zpi, sw.State({c: -z0[c] for c in COMPONENTS})) < 1e-13


def test_mode_standing_wave_on_the_self_conjugate_plane(mode_setup):
    # kx = 0 with interior ky: the Hermitian closure pairs the
    # branch with its mirror — still an exact solution of the
    # tendency eigen-relation
    model, em = mode_setup
    omega, z0, z1 = _mode_pair(em, 1, {"x": 0, "y": 2})
    tau = model.tendency(z0)
    residual = max(
        float(np.abs(np.asarray(tau[c].data)
                     - omega * np.asarray(z1[c].data)).max())
        for c in COMPONENTS)
    assert residual < 1e-12 * (1.0 + abs(omega))


def test_mode_zero_velocity_mean_is_unnormalized(mode_setup):
    # the geostrophic k = 0 mean is the pure-pressure mode: no
    # horizontal velocity to normalize, the raw amplitude stays
    _, em = mode_setup
    omega, z = em.mode("vortical", {"x": 0, "y": 0})
    assert omega == 0.0
    assert float(np.abs(np.asarray(z["u"].data)).max()) == 0.0
    assert float(np.abs(np.asarray(z["v"].data)).max()) == 0.0
    assert float(np.abs(np.asarray(z["p"].data)).max()) > 0.0


def test_mode_errors(mode_setup):
    _, em = mode_setup
    # the standard family is complete, the Nyquist strata included;
    # only degenerate parameters (f0 = 0 empties the geostrophic
    # mean) still hit the structural guard
    degenerate = sw.eigenmodes.Eigenmodes(_pinned_grid(), f0=0.0,
                                          csqr=1.0)
    with pytest.raises(ValueError, match="structurally"):
        degenerate.mode("vortical", {"x": 0, "y": 0})
    with pytest.raises(ValueError, match="half"):
        em.mode("wave+", {"x": -3, "y": 0})
    with pytest.raises(ValueError, match="keyed by the grid axes"):
        em.mode("wave+", {"x": 3})


def test_mode_uniform_family_surface(mode_setup):
    _, em = mode_setup
    # the vocabulary is public and matches the channel tier's shape
    assert dict(em.families) == {"vortical": 0, "wave+": 1,
                                 "wave-": -1}
    assert em.nonphysical_families == ()
    # the unsigned root + branch equals the signed spelling
    ws, zs = em.mode("wave+", {"x": 3, "y": 2})
    wb, zb = em.mode("wave", {"x": 3, "y": 2}, branch=+1)
    assert ws == wb
    for c in COMPONENTS:
        assert np.array_equal(np.asarray(zs[c].data),
                              np.asarray(zb[c].data))


def test_mode_rejects_integer_branches(mode_setup):
    _, em = mode_setup
    with pytest.raises(TypeError, match="family name"):
        em.mode(1, {"x": 3, "y": 2})
    with pytest.raises(TypeError, match="family name"):
        em.mode(0, {"x": 0, "y": 0})


def test_mode_teaches_the_kelvin_gap(mode_setup):
    _, em = mode_setup
    with pytest.raises(ValueError, match="no walls, no Kelvin"):
        em.mode("kelvin+", {"x": 2, "y": 0})


def test_mode_rejects_unknown_families(mode_setup):
    _, em = mode_setup
    with pytest.raises(ValueError, match="unknown mode family"):
        em.mode("rossby", {"x": 2, "y": 0})


# ================================================================
#  from_model dispatch: topology first, then parameter validation
# ================================================================
def _walled_model(*, periodic_x=True, coriolis=None):
    mx = fr.spatial.meshes.IntervalMesh(8, (0.0, 1.0),
                                     periodic=periodic_x, name="x")
    my = fr.spatial.meshes.IntervalMesh(8, (0.0, 1.0), periodic=False,
                                     name="y")
    if coriolis is None:
        coriolis = sw.modules.FPlaneCoriolis(f0=1.0)
    return sw.Model(
        grid=fr.spatial.Grid((mx, my)), csqr=1.0, rossby_number=0.2,
        coriolis=coriolis, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


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
#  sw.eigenbasis: the labeled channel eigenbasis surface
# ================================================================
def test_eigenbasis_returns_the_labeled_channel_eigenmodes():
    model = _walled_model()
    eb = sw.eigenbasis(model)
    assert isinstance(eb, sw.ChannelEigenmodes)
    assert eb.grid is model.grid
    assert (np.asarray(eb.labels) != -1).all()


def test_eigenbasis_dispatches_the_fully_periodic_grid():
    # the uniform entry point: a periodic grid gets the analytic
    # eigenmodes (previously a rejection pointing at from_model)
    em = sw.eigenbasis(make_model())
    assert isinstance(em, sw.eigenmodes.Eigenmodes)


def test_from_model_is_a_thin_alias_of_eigenbasis():
    model = make_model()
    assert isinstance(sw.eigenmodes.from_model(model),
                      sw.eigenmodes.Eigenmodes)
    assert isinstance(sw.eigenmodes.from_model(_walled_model()),
                      sw.ChannelEigenmodes)


def test_eigenbasis_rejects_a_multi_walled_box():
    with pytest.raises(ValueError, match="multi-walled"):
        sw.eigenbasis(_walled_model(periodic_x=False))


# ================================================================
#  from_model structural validation
# ================================================================
def test_from_model_rejects_a_beta_plane():
    grid = make_grid()
    model = sw.Model(
        grid=grid, csqr=1.0,
        coriolis=sw.modules.BetaPlaneCoriolis(f0=1.0, beta=2.0),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"coriolis\.f0"):
        sw.eigenmodes.from_model(model)


def test_degenerate_and_non_2d_grids_are_rejected():
    grid = make_grid()
    with pytest.raises(ValueError, match="degenerate"):
        sw.eigenmodes.Eigenmodes(grid, f0=0.0, csqr=0.0)
    mz = fr.spatial.meshes.IntervalMesh(4, (0.0, 1.0), periodic=True,
                                     name="z")
    grid3 = fr.spatial.Grid((*grid.factors, mz))
    with pytest.raises(ValueError, match="2-D"):
        sw.eigenmodes.Eigenmodes(grid3, f0=1.0, csqr=1.0)


# ================================================================
#  function(f, s): scalar functions of the linear operator
# ================================================================
@pytest.mark.parametrize("sel", [
    pytest.param(0, id="vortical"),
    pytest.param(1, id="plus"),
    pytest.param((1, -1), id="wave-pair"),
])
def test_function_with_unit_f_reproduces_the_projectors(
        mode_setup, sel):
    # f == 1 on a branch selection is exactly the summed projectors
    # (weights 1 on the represented modes, 0 on the structural
    # zeros — where the projector amplitude is exactly 0), bitwise
    _, em = mode_setup
    z = _random_coeff_state(em, seed=41)
    branches = (sel,) if isinstance(sel, int) else sel
    want = None
    for s in branches:
        part = em.projector(s)(z)
        want = part if want is None else sw.State(
            {c: want[c] + part[c] for c in COMPONENTS})
    got = em.function(np.ones_like, sel)(z)
    for c in COMPONENTS:
        assert np.array_equal(np.asarray(got[c].data),
                              np.asarray(want[c].data))


def test_function_inverse_wave_strong_test(mode_setup):
    # THE STRONG TEST: with invL = function(-1/(i omega), (1, -1)),
    # L(invL(z)) == P_wave(z) through the model's linearized
    # tendency, asserted in coefficient space (self-conjugate kx
    # planes zeroed so the physical round-trip is exact)
    model, em = mode_setup
    kit = em._kit
    lin = fr.model.linearize(model)
    prog, base0 = _rest_background(lin, 0.0)
    assert prog == COMPONENTS
    rng = np.random.default_rng(42)
    template = em.q(0)
    shape = np.asarray(template["u"].data).shape

    def make_amp():
        amp = (rng.standard_normal(shape)
               + 1j * rng.standard_normal(shape))
        amp[0] = 0.0
        amp[-1] = 0.0
        return amp

    z = sw.State({c: template[c].with_data(jnp.asarray(make_amp()))
                  for c in COMPONENTS})
    w_hat = em.function(lambda om: -1.0 / (1j * om), (1, -1))(z)
    nodal = {c: kit.backward(c)(w_hat[c]) for c in prog}
    phys = base0.replace(**{
        c: base0[c].with_data(nodal[c].data) for c in prog})
    tau = lin.tendency(phys, t=0.0, constraints=False)
    plus = em.projector(1)(z)
    minus = em.projector(-1)(z)
    scale = max(float(np.abs(np.asarray(z[c].data)).max())
                for c in COMPONENTS)
    for c in prog:
        got = np.asarray(kit.forward(c)(tau[c]).data)
        want = np.asarray(plus[c].data) + np.asarray(minus[c].data)
        assert np.abs(got - want).max() / scale < 1e-12


def test_function_inverse_wave_is_real_safe(mode_setup):
    # -1/(i omega) satisfies f(-omega) == conj(f(omega)) and (1, -1)
    # is conjugation-closed: the coefficients of a real state stay
    # Hermitian and the backward synthesis stays real
    model, em = mode_setup
    kit = em._kit
    rng = np.random.default_rng(43)
    z = sw.State({
        c: kit.forward(c)(model.state[c].with_data(jnp.asarray(
            rng.standard_normal(model.state[c].data.shape))))
        for c in COMPONENTS})
    out = em.function(lambda om: -1.0 / (1j * om), (1, -1))(z)
    for c in COMPONENTS:
        back = np.asarray(kit.backward(c)(out[c]).data)
        scale = float(np.abs(back).max())
        assert np.abs(np.imag(back)).max() < 1e-15 * scale


def test_function_structural_zero_guard(mode_setup):
    # the geostrophic branch is represented with omega == 0: a
    # singular f is a taught error, never a floored division —
    # while the wave branches (inertial k = 0 pair included) carry
    # no represented zero and pass
    _, em = mode_setup
    with pytest.raises(ValueError, match=r"s=0"):
        em.function(lambda om: 1.0 / (1j * om), 0)
    with pytest.raises(ValueError, match="non-finite"):
        em.function(lambda om: 1.0 / (1j * om), (0, 1))
    assert callable(em.function(lambda om: 1.0 / (1j * om), (1, -1)))


def test_function_rejects_bad_branch_selections(mode_setup):
    _, em = mode_setup
    for bad in (2, (), (1, 1), 1.5, "wave"):
        with pytest.raises(ValueError, match="branches"):
            em.function(np.ones_like, bad)
