"""Oracle battery: SW channel eigenmodes vs certified closed forms.

The C4 test: the numeric channel eigenbasis (fr.channel_eigenpairs)
on the walled shallow-water model against the audited analytic
eigenfunctions of the SAME discretization (vortical sines + the
kx = 0 exponential wall-mode pair, boundary-trapped Kelvin pairs,
Poincare waves), plus the shallow-water family labeler
(sw.channel_eigenmodes) on the f-plane, the beta plane, and the
coarse-grid alternating regime.

Index alignment (the reference -> framework2 mapping)
-----------------------------------------------------
The reference y grid is the shifted cell centres INCLUDING the
y = Ly wall: y_j = dy (j + 1), j = 0..N-1. Framework2's v holds the
interior faces dy (1..N-1) only; u and p sit on the centres
dy (0.5..N-0.5). Hence:

- v: reference entry j -> engine v index j (j = 0..N-2); the
  reference's LAST entry (the wall node, exactly zero) is dropped;
- u/p: the reference ave/diff structure functions at index j already
  live on the centre y_j - dy/2 = dy (j + 1/2) -> engine index j,
  all N entries kept.

No inter-segment phase correction is needed: both sides are
per-array (r)fft amplitudes on the same right-face C-grid stagger
(u on Right(x), v on Right(y) interior faces, p collocated).

Time convention: the reference modes evolve as e^{i(kx x - omega t)}
while the engine pairs L q = +i omega q, so a reference mode of
frequency omega_ref sits in the engine column with
omega_eng = -omega_ref (measured overlap 1.0 to machine precision).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.shallowwater2 as sw
from fridom.framework2.model.eigen_channel import (
    UNLABELED,
    ChannelEigenbasis,
)
from fridom.shallowwater2.channel_eigenmodes import (
    FAMILIES,
    KELVIN_MINUS,
    KELVIN_PLUS,
    VORTICAL,
    WAVE_MINUS,
    WAVE_PLUS,
    ChannelEigenmodes,
    label_channel_modes,
)

N = 8
D = 3 * N - 1  # u: N, v: N - 1 (interior faces), p: N
N_KX = N // 2 + 1
LX = LY = 1.0
DX = LX / N
DY = LY / N
CSQR = 0.7
F0 = 1.0
BETA = 2.0
# the coarse-grid alternating regime: 4 c^2 < f0^2 dy^2
CSQR_ALT = 0.01
F0_ALT = 4.0

# the reference y grid: shifted centres INCLUDING the y = Ly wall
Y_REF = DY * jnp.arange(1, N + 1)


# ================================================================
#  Model builders and module-scoped fixtures (compile reuse)
# ================================================================
def make_walled_model(*, csqr=CSQR, coriolis=None):
    """Build the linear walled (channel) shallow-water model."""
    if coriolis is None:
        coriolis = sw.modules.FPlaneCoriolis(f0=F0)
    mx = fr.grid.meshes.IntervalMesh(N, (0.0, LX), periodic=True,
                                     name="x")
    my = fr.grid.meshes.IntervalMesh(N, (0.0, LY), periodic=False,
                                     name="y")
    return sw.Model(
        grid=fr.grid.Grid((mx, my)), csqr=csqr, rossby_number=0.2,
        coriolis=coriolis, advection=False,
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def model():
    """One f-plane channel model shared across the module."""
    return make_walled_model()


@pytest.fixture(scope="module")
def em(model):
    """Label the channel eigenmodes of the shared f-plane model."""
    return ChannelEigenmodes(model)


@pytest.fixture(scope="module")
def beta_em():
    """Labeled eigenmodes with f varying along the dense y axis."""
    return ChannelEigenmodes(make_walled_model(
        coriolis=sw.modules.BetaPlaneCoriolis(f0=F0, beta=BETA)))


@pytest.fixture(scope="module")
def alt_em():
    """Labeled eigenmodes in the coarse-grid alternating regime."""
    return ChannelEigenmodes(make_walled_model(
        csqr=CSQR_ALT,
        coriolis=sw.modules.FPlaneCoriolis(f0=F0_ALT)))


# ================================================================
#  The ported closed forms (jax, f64; complex-safe branches)
# ================================================================
def sym_x(ikx):
    """Zonal derivative/interpolation symbols at integer plane ikx."""
    theta = 2.0 * jnp.pi * ikx / N
    kxh = -1j / DX * (jnp.exp(1j * theta) - 1.0)
    oxh = 0.5 * (jnp.exp(1j * theta) + 1.0)
    return kxh, oxh


def sym_y(n):
    """Meridional symbols for the sine mode number n (ky = pi n/Ly)."""
    ky = jnp.pi * n / LY
    kyh = -1j / DY * (jnp.exp(1j * ky * DY) - 1.0)
    oyh = 0.5 * (jnp.exp(1j * ky * DY) + 1.0)
    return kyh, oyh


def sin_structure(n):
    """Return (phi, ave_phi, diff_phi) of the sine mode number n."""
    ky = jnp.pi * n / LY
    phi = jnp.sin(ky * Y_REF)
    ave = 0.5 * (jnp.sin(ky * Y_REF) + jnp.sin(ky * (Y_REF - DY)))
    diff = (jnp.sin(ky * Y_REF) - jnp.sin(ky * (Y_REF - DY))) / DY
    return phi, ave, diff


def stack_column(u, v, p):
    """Stack reference (u, v, p) onto the engine column layout.

    u/p keep all N centre entries; v drops the reference's wall node
    (its last entry, exactly zero) to land on the N - 1 interior
    faces.
    """
    u = jnp.broadcast_to(jnp.asarray(u, dtype=complex), (N,))
    v = jnp.broadcast_to(jnp.asarray(v, dtype=complex), (N,))
    p = jnp.broadcast_to(jnp.asarray(p, dtype=complex), (N,))
    return jnp.concatenate([u, v[:-1], p])


def ref_vortical(ikx, n, f0=F0):
    """Vortical sine eigenfunction (omega = 0), mode n >= 1."""
    kxh, oxh = sym_x(ikx)
    phi, ave, diff = sin_structure(n)
    return stack_column(-diff, 1j * jnp.conj(kxh) * phi,
                        jnp.conj(oxh) * f0 * ave)


def ref_wall_mode(branch, f0=F0, csqr=CSQR):
    """Return the kx = 0 exponential wall mode (branch +1/-1) and lam.

    The arccosh argument goes NEGATIVE for 4 c^2 < f0^2 dy^2 (the
    Nyquist-modulated alternating regime), so it is evaluated with an
    explicitly complex input and the overflow-guard branch reads
    Re(lam) — the reference's numpy complex-ordering trick
    ``where(-lam > 0, ...)`` is deliberately not ported literally.
    """
    a = 4.0 * csqr
    b = f0 ** 2 * DY ** 2
    arg = jnp.asarray((a + b) / (a - b), dtype=complex)
    lam = branch * jnp.arccosh(arg) / DY
    yoff = jnp.where(jnp.real(lam) < 0, 0.0, LY)

    def psi(y):
        return jnp.exp(lam * (y - yoff))

    phi = psi(Y_REF)
    ave = 0.5 * (phi + psi(Y_REF - DY))
    diff = (phi - psi(Y_REF - DY)) / DY
    return stack_column(-diff, 0.0, f0 * ave), lam


def kelvin_omega(ikx, csqr=CSQR):
    """Return the positive Kelvin frequency c |k_hat_x| at plane ikx."""
    kxh, _ = sym_x(ikx)
    return float(np.sqrt(csqr) * abs(complex(kxh)))


def ref_kelvin(ikx, mode, f0=F0, csqr=CSQR):
    """Kelvin eigenfunction (branch mode = +1/-1) and (omega, lam).

    The log argument goes negative (complex lam, the alternating
    regime) where 2 omega < |conj(oxh) kxh| f0 dy; ported with an
    explicitly complex input and Re-based branching.
    """
    kxh, oxh = sym_x(ikx)
    om = jnp.sqrt(jnp.asarray(csqr * kxh * jnp.conj(kxh),
                              dtype=complex))
    coupling = jnp.conj(oxh) * kxh * f0 * DY
    arg = jnp.asarray((2.0 * om + coupling) / (2.0 * om - coupling),
                      dtype=complex)
    lam = jnp.log(arg) / DY
    yoff = jnp.where(jnp.real(mode * lam) > 0, 0.0, LY)
    struct = jnp.exp(-mode * lam * (Y_REF - yoff))
    col = stack_column(kxh * struct, 0.0, mode * om * struct)
    return col, float(jnp.real(mode * om)), complex(lam)


def poincare_omega(ikx, n, s, f0=F0, csqr=CSQR):
    """Return the Poincare dispersion s sqrt(|ox oy f0|^2 + c^2 |kh|^2)."""
    kxh, oxh = sym_x(ikx)
    kyh, oyh = sym_y(n)
    return float(s * jnp.sqrt(
        jnp.abs(oxh) ** 2 * jnp.abs(oyh) ** 2 * f0 ** 2
        + csqr * (jnp.abs(kxh) ** 2 + jnp.abs(kyh) ** 2)))


def ref_wave(ikx, n, s, f0=F0, csqr=CSQR):
    """Poincare eigenfunction (mode n >= 1, branch s) and omega."""
    kxh, oxh = sym_x(ikx)
    om = poincare_omega(ikx, n, s, f0, csqr)
    phi, ave, diff = sin_structure(n)
    u = oxh * f0 * om * ave - csqr * kxh * diff
    v = -1j * (om ** 2 - csqr * jnp.abs(kxh) ** 2) * phi
    p = csqr * (oxh * jnp.conj(kxh) * f0 * ave - om * diff)
    return stack_column(u, v, p), om


def analytic_spectrum(ikx, f0=F0, csqr=CSQR):
    """Return the sorted analytic frequency multiset of one kx plane."""
    vals = [0.0] * (N + 1 if ikx == 0 else N - 1)
    if ikx != 0:
        vals += [s * kelvin_omega(ikx, csqr) for s in (-1, 1)]
    vals += [poincare_omega(ikx, n, s, f0, csqr)
             for n in range(1, N) for s in (-1, 1)]
    return np.sort(np.asarray(vals))


# ================================================================
#  Engine-side helpers
# ================================================================
def m_inner(em, a, b):
    """Compute the diagonal-metric inner product of stacked columns."""
    metric = np.asarray(em.metric)
    return np.sum(np.conj(np.asarray(a)) * metric * np.asarray(b))


def engine_column(em, ikx, omega_ref):
    """Engine column carrying the reference mode of omega_ref.

    The time-convention flip: the reference evolves as e^{-i omega t}
    while the engine pairs L q = +i omega q, so the match is at
    omega_eng = -omega_ref.
    """
    omega = np.asarray(em.omega[ikx])
    col = int(np.argmin(np.abs(omega + omega_ref)))
    assert abs(omega[col] + omega_ref) < 1e-11
    return col


def column_overlap(em, ikx, col, ref):
    """Phase-free M-overlap of a reference column vs engine column."""
    eng = np.asarray(em.q[ikx])[:, col]
    return abs(m_inner(em, ref, eng)) / np.sqrt(
        abs(m_inner(em, ref, ref)) * abs(m_inner(em, eng, eng)))


def zero_space_membership(em, ikx, ref):
    """|P_0 ref|_M / |ref|_M onto the plane's zero-frequency space.

    The vortical family is degenerate (omega = 0 across the whole
    branch), so single-column overlaps are meaningless there; the
    right oracle is membership of the reference vector in the
    engine's zero eigenspace.
    """
    omega = np.asarray(em.omega[ikx])
    q = np.asarray(em.q[ikx])
    metric = np.asarray(em.metric)
    qz = q[:, np.abs(omega) < 1e-8]
    coef = np.einsum("ij,i,i->j", np.conj(qz), metric,
                     np.asarray(ref))
    proj = qz @ coef
    return float(np.sqrt(abs(m_inner(em, proj, proj))
                         / abs(m_inner(em, ref, ref))))


def family_columns(em, ikx, family):
    """Column indices labeled with the named family at plane ikx."""
    return np.where(
        np.asarray(em.labels[ikx]) == em.families[family])[0]


def eigen_relation_residual(model, em, ikx, col):
    """Drive the real matvec with one column; return (resid, scale).

    Builds the real physical state ``Re(q(y) e^{i kx x})``, applies
    ``model.tendency`` and compares the rfft ``ikx`` plane against
    ``i omega q`` (interior-plane half-spectrum amplitude ``N/2``).
    """
    q_col = np.asarray(em.q[ikx, :, col])
    omega = float(em.omega[ikx, col])
    phase = np.exp(2j * np.pi * ikx * np.arange(N) / N)
    model.set_fields(**{
        name: np.real(
            phase[:, None] * q_col[em.slices[name]][None, :])
        for name in em.components})
    tendency = model.tendency(model.state)

    residual = 0.0
    for name in em.components:
        plane = np.fft.rfft(np.asarray(tendency[name].data),
                            axis=0)[ikx]
        expect = 1j * omega * (N / 2) * q_col[em.slices[name]]
        residual = max(residual, float(np.abs(plane - expect).max()))
    return residual, (1.0 + abs(omega)) * (N / 2)


def make_synthetic_basis(omega, q, n_u=2, n_v=1, n_p=2):
    """Build a hand-crafted ChannelEigenbasis for labeler testing."""
    d = n_u + n_v + n_p
    slices = {"u": slice(0, n_u), "v": slice(n_u, n_u + n_v),
              "p": slice(n_u + n_v, d)}
    return ChannelEigenbasis(
        omega=jnp.asarray(omega, dtype=float),
        q=jnp.asarray(q, dtype=complex),
        components=("u", "v", "p"), slices=slices,
        metric=jnp.ones(d), periodic_axis="x", bounded_axis="y",
        hermiticity_error=0.0)


def unit(d, idx):
    """Return a complex unit vector of length d."""
    e = np.zeros(d, dtype=complex)
    e[idx] = 1.0
    return e


def mix(d, i, j):
    """Return an equal two-entry mixture (v-full on a v index)."""
    return (unit(d, i) + unit(d, j)) / np.sqrt(2.0)


# ================================================================
#  (a) Dispersion: the closed forms at EVERY plane (alignment-free)
# ================================================================
def test_spectrum_matches_the_analytic_multiset(em):
    # counts and values at once: per plane the sorted engine spectrum
    # is the sorted analytic multiset (vortical zeros + Kelvin pair +
    # Poincare set) to near machine precision
    omega = np.asarray(em.omega)
    for ikx in range(N_KX):
        expect = analytic_spectrum(ikx)
        assert np.abs(omega[ikx] - expect).max() < 1e-12


def test_kelvin_pair_matches_c_khat(em):
    # the labeled kelvin+/- columns carry omega = +/- c |k_hat_x|
    omega = np.asarray(em.omega)
    for ikx in range(1, N_KX):
        expect = kelvin_omega(ikx)
        (plus,) = family_columns(em, ikx, "kelvin+")
        (minus,) = family_columns(em, ikx, "kelvin-")
        assert abs(omega[ikx, plus] - expect) < 1e-12
        assert abs(omega[ikx, minus] + expect) < 1e-12


def test_poincare_set_matches_the_closed_form(em):
    # the wave-labeled frequencies are exactly the analytic Poincare
    # set per plane (sorted comparison, alignment-free)
    omega = np.asarray(em.omega)
    labels = np.asarray(em.labels)
    for ikx in range(N_KX):
        wave = np.isin(labels[ikx], (WAVE_PLUS, WAVE_MINUS))
        got = np.sort(omega[ikx][wave])
        expect = np.sort([poincare_omega(ikx, n, s)
                          for n in range(1, N) for s in (-1, 1)])
        assert got.shape == expect.shape
        assert np.abs(got - expect).max() < 1e-12


def test_vortical_columns_are_zero_on_the_f_plane(em):
    omega = np.asarray(em.omega)
    labels = np.asarray(em.labels)
    for ikx in range(N_KX):
        vort = labels[ikx] == VORTICAL
        assert vort.sum() == (N + 1 if ikx == 0 else N - 1)
        assert np.abs(omega[ikx][vort]).max() < 1e-8


# ================================================================
#  (b) The alternating/exponential regime (coarse grid, large f0)
# ================================================================
def test_alternating_regime_spectrum_and_zero_counts(alt_em):
    assert 4 * CSQR_ALT < F0_ALT ** 2 * DY ** 2  # regime active
    omega = np.asarray(alt_em.omega)
    zeros = np.sum(np.abs(omega) < 1e-8, axis=-1)
    assert zeros[0] == N + 1
    assert (zeros[1:] == N - 1).all()
    for ikx in range(N_KX):
        expect = analytic_spectrum(ikx, f0=F0_ALT, csqr=CSQR_ALT)
        assert np.abs(omega[ikx] - expect).max() < 1e-12
    # the labeler stays exact here too
    assert (np.asarray(alt_em.labels) != UNLABELED).all()


@pytest.mark.parametrize("branch", [1, -1])
def test_alternating_wall_modes_match_the_corrected_branch(
        alt_em, branch):
    # 4c^2 < f0^2 dy^2: the arccosh argument is negative, lam picks
    # up the Nyquist modulation Im(lam) = pi/dy, and the corrected
    # complex-safe port still lies in the engine's kx = 0 zero space
    ref, lam = ref_wall_mode(branch, f0=F0_ALT, csqr=CSQR_ALT)
    assert abs(abs(np.imag(complex(lam))) * DY - np.pi) < 1e-12
    assert zero_space_membership(alt_em, 0, ref) > 1 - 1e-12


@pytest.mark.parametrize(("ikx", "alternating"), [
    pytest.param(1, True, id="kx1-alternating"),
    pytest.param(2, True, id="kx2-alternating"),
    pytest.param(3, False, id="kx3-exponential"),
])
def test_alternating_kelvin_columns_match(alt_em, ikx, alternating):
    # near the origin the Kelvin log-argument goes negative (complex
    # lam, alternating decay); past the crossover it is a plain
    # exponential — both regimes match the engine column
    for mode in (1, -1):
        ref, om, lam = ref_kelvin(ikx, mode, f0=F0_ALT,
                                  csqr=CSQR_ALT)
        assert (abs(np.imag(lam)) > 1.0) == alternating
        col = engine_column(alt_em, ikx, om)
        assert column_overlap(alt_em, ikx, col, ref) > 1 - 1e-12
        assert np.asarray(alt_em.labels)[ikx, col] in (
            KELVIN_PLUS, KELVIN_MINUS)


# ================================================================
#  (c) Eigenvector agreement per family (f-plane)
# ================================================================
@pytest.mark.parametrize(("ikx", "n"), [
    pytest.param(0, 1, id="kx0-n1"),
    pytest.param(0, N - 1, id="kx0-n7"),
    pytest.param(1, 1, id="kx1-n1"),
    pytest.param(1, N - 1, id="kx1-n7"),
    pytest.param(3, 4, id="kx3-n4"),
    pytest.param(4, 2, id="nyquist-n2"),
])
def test_vortical_sines_lie_in_the_zero_space(em, ikx, n):
    # the vortical branch is degenerate (omega = 0), so the oracle is
    # membership of the reference sine mode in the zero eigenspace
    ref = ref_vortical(ikx, n)
    assert zero_space_membership(em, ikx, ref) > 1 - 1e-12


@pytest.mark.parametrize("branch", [1, -1])
def test_wall_modes_lie_in_the_kx0_zero_space(em, branch):
    # the kx = 0 exponential wall-mode pair (fine grid: real lam)
    ref, lam = ref_wall_mode(branch)
    assert abs(np.imag(complex(lam))) < 1e-12
    assert zero_space_membership(em, 0, ref) > 1 - 1e-12


@pytest.mark.parametrize("ikx", [1, 2, 4])
@pytest.mark.parametrize("mode", [1, -1])
def test_kelvin_columns_match(em, ikx, mode):
    ref, om, _lam = ref_kelvin(ikx, mode)
    col = engine_column(em, ikx, om)
    assert column_overlap(em, ikx, col, ref) > 1 - 1e-12
    # the reference +omega branch is the engine's kelvin- (the
    # e^{-i omega t} vs L q = +i omega q time-convention flip)
    expect = KELVIN_MINUS if mode == 1 else KELVIN_PLUS
    assert np.asarray(em.labels)[ikx, col] == expect


@pytest.mark.parametrize(("ikx", "n", "s"), [
    pytest.param(0, 1, 1, id="kx0-n1-plus"),
    pytest.param(1, 1, 1, id="kx1-n1-plus"),
    pytest.param(1, 2, -1, id="kx1-n2-minus"),
    pytest.param(3, N - 1, 1, id="kx3-n7-plus"),
    pytest.param(4, 3, -1, id="nyquist-n3-minus"),
])
def test_poincare_columns_match(em, ikx, n, s):
    ref, om = ref_wave(ikx, n, s)
    col = engine_column(em, ikx, om)
    assert column_overlap(em, ikx, col, ref) > 1 - 1e-12


# ================================================================
#  (d) The labeler: exact on the f-plane, graceful under beta
# ================================================================
def test_fplane_labels_are_complete_with_the_expected_counts(em):
    labels = np.asarray(em.labels)
    assert (labels != UNLABELED).all()
    for ikx in range(N_KX):
        counts = {name: int((labels[ikx] == code).sum())
                  for name, code in em.families.items()}
        expect_kelvin = 0 if ikx == 0 else 1
        assert counts == {
            "vortical": N + 1 if ikx == 0 else N - 1,
            "kelvin+": expect_kelvin, "kelvin-": expect_kelvin,
            "wave+": N - 1, "wave-": N - 1}


def test_beta_labels_degrade_gracefully(beta_em):
    labels = np.asarray(beta_em.labels)
    omega = np.asarray(beta_em.omega)
    # kelvin stays labeled: the v-energy criterion is still
    # machine-crisp under beta (v = 0 solves the beta channel too)
    q = np.asarray(beta_em.q)
    metric = np.asarray(beta_em.metric)
    v_slice = beta_em.slices["v"]
    for ikx in range(1, N_KX):
        for family, sign in (("kelvin+", 1), ("kelvin-", -1)):
            (col,) = family_columns(beta_em, ikx, family)
            v_energy = float(np.sum(
                np.abs(q[ikx, v_slice, col]) ** 2 * metric[v_slice]))
            assert v_energy < 1e-20
            assert sign * omega[ikx, col] > 0
    # no mislabels: every vortical column is slow — separated from
    # every wave column by at least the documented gap factor
    for ikx in range(N_KX):
        vort = np.abs(omega[ikx][labels[ikx] == VORTICAL])
        wave = np.abs(omega[ikx][
            np.isin(labels[ikx], (WAVE_PLUS, WAVE_MINUS))])
        assert wave.size == 2 * (N - 1)
        if vort.size and wave.size:
            assert vort.max() * 10.0 <= wave.min()
    # the kx = 0 plane keeps its N + 1 steady columns; the interior
    # planes carry the FULL slow Rossby band (zero steady modes under
    # the v1 Coriolis convention), the Nyquist plane decouples
    assert (labels[0] == VORTICAL).sum() == N + 1
    for ikx in range(1, N_KX - 1):
        vort = omega[ikx][labels[ikx] == VORTICAL]
        assert vort.size == N - 1
        assert (np.abs(vort) > 1e-8).all()
    assert (np.abs(omega[-1][labels[-1] == VORTICAL]) < 1e-8).all()
    # UNLABELED is allowed by contract; at these parameters the
    # spectral gap is clean everywhere and nothing is left over
    assert (labels != UNLABELED).all()


def test_labeler_without_a_clean_gap_leaves_unlabeled():
    # D = 8 (n_u = 3, n_v = 2, n_p = 3), n_wave = 4; plane 1 keeps a
    # kelvin pair and one extra slow-ish column with NO clean gap
    d = 8  # v segment: indices 3, 4
    q0 = np.eye(d, dtype=complex)
    omega0 = [-2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    q1 = np.stack([unit(d, 0), mix(d, 3, 5), mix(d, 4, 6),
                   unit(d, 5), mix(d, 3, 7), mix(d, 4, 5),
                   mix(d, 3, 6), unit(d, 1)], axis=1)
    omega1 = [-4.0, -3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 4.0]
    basis = make_synthetic_basis(
        [omega0, omega1], [q0, q1], n_u=3, n_v=2, n_p=3)
    labels = np.asarray(label_channel_modes(basis))
    # plane 0 (kx = 0): 4 zeros -> vortical, the rest wave by sign
    assert (labels[0] == np.array(
        [WAVE_MINUS, WAVE_MINUS, VORTICAL, VORTICAL, VORTICAL,
         VORTICAL, WAVE_PLUS, WAVE_PLUS])).all()
    # plane 1: kelvin pair found; the 5 remaining nonzero columns
    # exceed n_wave = 4 but mags 0.5 vs 1.0 gives no gap >= 10
    assert (labels[1] == np.array(
        [KELVIN_MINUS, UNLABELED, UNLABELED, VORTICAL, UNLABELED,
         UNLABELED, UNLABELED, KELVIN_PLUS])).all()


def test_labeler_splits_slow_from_wave_across_a_clean_gap():
    # the synthetic beta path: one genuinely slow column, gap >= 10
    d = 8
    q0 = np.eye(d, dtype=complex)
    omega0 = [-2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    q1 = np.stack([unit(d, 0), mix(d, 3, 5), mix(d, 4, 6),
                   unit(d, 5), mix(d, 3, 7), mix(d, 4, 5),
                   mix(d, 3, 6), unit(d, 1)], axis=1)
    omega1 = [-4.0, -3.0, -2.0, 0.0, 0.01, 2.0, 3.0, 4.0]
    basis = make_synthetic_basis(
        [omega0, omega1], [q0, q1], n_u=3, n_v=2, n_p=3)
    labels = np.asarray(label_channel_modes(basis))
    assert (labels[1] == np.array(
        [KELVIN_MINUS, WAVE_MINUS, WAVE_MINUS, VORTICAL, VORTICAL,
         WAVE_PLUS, WAVE_PLUS, KELVIN_PLUS])).all()


def test_labeler_recovers_a_degenerate_kelvin_cluster():
    # eigh mixed the kelvin+ column with a degenerate Poincare
    # column: neither mixture is v-crisp, and the labeler's 2-cluster
    # rotation (the v-energy sub-eigh) recovers the v-free direction.
    # The synthetic basis is genuinely M-orthonormal (metric = 1), so
    # the unitarity of the write-back is observable.
    d = 8  # n_u = 3, n_v = 2 (indices 3, 4), n_p = 3; n_wave = 4
    a = unit(d, 0)                             # v-free (kelvin)
    b = (unit(d, 3) - unit(d, 7)) / np.sqrt(2)  # v-full (poincare)
    q0 = np.eye(d, dtype=complex)
    omega0 = [-2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    q1 = np.stack([unit(d, 1), mix(d, 4, 6),
                   (unit(d, 4) - unit(d, 6)) / np.sqrt(2),
                   unit(d, 5), unit(d, 2),
                   (a + b) / np.sqrt(2), (a - b) / np.sqrt(2),
                   mix(d, 3, 7)], axis=1)
    omega1 = [-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 1.0, 3.0]
    basis = make_synthetic_basis(
        [omega0, omega1], [q0, q1], n_u=3, n_v=2, n_p=3)
    labels = np.asarray(label_channel_modes(basis))
    assert (labels[1] == np.array(
        [KELVIN_MINUS, WAVE_MINUS, WAVE_MINUS, VORTICAL, VORTICAL,
         KELVIN_PLUS, WAVE_PLUS, WAVE_PLUS])).all()
    # the rotation was written back: the kelvin+ column is now the
    # v-free direction, and the unitary mixing kept orthonormality
    rotated = np.asarray(basis.q)[1]
    assert abs(abs(rotated[:, 5] @ np.conj(a)) - 1.0) < 1e-12
    assert float(basis.orthonormality_error()) < 1e-12


def test_labeler_skips_clusters_without_a_v_free_direction():
    # kelvin+ genuinely absent: the only degenerate positive pair
    # spans two v-full directions (Gram = identity, no v-free combo)
    d = 8  # v segment: indices 3, 4
    q0 = np.eye(d, dtype=complex)
    omega0 = [-2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    q1 = np.stack([unit(d, 0), mix(d, 3, 5), mix(d, 4, 6),
                   unit(d, 5), mix(d, 3, 7), unit(d, 3),
                   unit(d, 4), mix(d, 4, 7)], axis=1)
    omega1 = [-4.0, -3.0, -1.0, 0.0, 1.0, 2.0, 2.0, 4.0]
    basis = make_synthetic_basis(
        [omega0, omega1], [q0, q1], n_u=3, n_v=2, n_p=3)
    labels = np.asarray(label_channel_modes(basis))
    assert labels[1][0] == KELVIN_MINUS
    assert labels[1][3] == VORTICAL
    assert (labels[1][[1, 2, 4, 5, 6, 7]] == UNLABELED).all()
    # ... and q was left untouched (no rotation happened)
    assert np.abs(np.asarray(basis.q)[1] - q1).max() == 0.0


def test_labeler_is_defensive_on_unexpected_plane_structure():
    # fewer nonzero columns than the structural wave count: the
    # remainder stays UNLABELED instead of guessing
    d = 5
    basis = make_synthetic_basis(
        [[0.0, 0.0, 0.0, 0.0, 1.0]],
        [np.eye(d, dtype=complex)[:, [1, 3, 4, 0, 2]]])
    labels = np.asarray(label_channel_modes(basis))
    assert (labels[0] == np.array(
        [VORTICAL, VORTICAL, VORTICAL, VORTICAL, UNLABELED])).all()


def test_labeler_override_hook(em):
    # the override receives (basis, automatic labels) and wins
    def drop_plane_zero(_basis, labels):
        return labels.at[0].set(UNLABELED)

    labels = np.asarray(
        label_channel_modes(em.basis, override=drop_plane_zero))
    assert (labels[0] == UNLABELED).all()
    assert (labels[1:] == np.asarray(em.labels)[1:]).all()


def test_labeler_rejects_a_non_2d_basis():
    d = 5
    basis = make_synthetic_basis(
        np.zeros((2, 2, d)),
        np.broadcast_to(np.eye(d, dtype=complex), (2, 2, d, d)))
    with pytest.raises(ValueError, match=r"\(n_kx, D\)"):
        label_channel_modes(basis)


# ================================================================
#  (e) M-orthonormality + the strong per-family eigen relation
# ================================================================
def test_eigenvectors_are_m_orthonormal_across_families(em):
    assert float(em.basis.orthonormality_error()) < 1e-12


@pytest.mark.parametrize(("ikx", "family"), [
    pytest.param(2, "vortical", id="vortical"),
    pytest.param(1, "kelvin+", id="kelvin-plus"),
    pytest.param(1, "kelvin-", id="kelvin-minus"),
    pytest.param(3, "wave+", id="wave-plus"),
    pytest.param(3, "wave-", id="wave-minus"),
])
def test_one_column_per_family_satisfies_the_eigen_relation(
        model, em, ikx, family):
    # the strong test through the real model matvec, one labeled
    # column per family (interior planes: rfft amplitude N/2)
    col = int(family_columns(em, ikx, family)[0])
    residual, scale = eigen_relation_residual(model, em, ikx, col)
    assert residual < 1e-11 * scale


# ================================================================
#  The surface: passthroughs and the family map
# ================================================================
def test_surface_passthroughs_and_family_map(em):
    assert isinstance(em, ChannelEigenmodes)
    assert em.components == ("u", "v", "p")
    assert em.slices["u"] == slice(0, N)
    assert em.slices["v"] == slice(N, 2 * N - 1)
    assert em.slices["p"] == slice(2 * N - 1, D)
    assert em.periodic_axis == "x"
    assert em.bounded_axis == "y"
    assert em.omega.shape == (N_KX, D)
    assert em.q.shape == (N_KX, D, D)
    assert em.labels.shape == (N_KX, D)
    assert np.asarray(em.metric).shape == (D,)
    assert em.families == FAMILIES
    assert em.family_names[em.families["kelvin+"]] == "kelvin+"
    assert set(em.families.values()) == set(em.family_names.keys())
