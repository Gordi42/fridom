"""Oracle battery: nh channel eigenmodes vs certified closed forms.

The labeled 3-D nonhydro channel eigenbasis (walled y, x/z periodic,
the constrained ``P L P`` engine) against the audited analytic
eigenfunctions of the SAME discretization, ported from the
boundary_emission reference (vortical sines + the kx = 0 exponential
wall modes, boundary-trapped internal Kelvin pairs per vertical
wavenumber, Poincaré strata, the kz = 0 buoyancy oscillations), plus
the nonhydro family labeler with its Leray-overlap constraint split
— on the f-plane and the beta plane.

Index alignment (the reference -> framework2 mapping)
-----------------------------------------------------
The reference structure functions live on two y grids: the shifted
centres INCLUDING the y = Ly wall (``Y_FACE``, sine strata) and the
plain cell centres (``Y_CENTER``, Kelvin/buoyancy structures).
Framework2's v holds the interior faces only, so the reference's
last (wall, exactly zero) v entry is dropped; u/w/b keep all N
centre entries. No inter-segment phase correction is needed: both
sides are per-array (r)fft amplitudes on the same right-face C-grid
stagger.

Time convention: the reference modes evolve as
``e^{i(kx x + kz z - omega t)}`` while the engine pairs
``L q = +i omega q``, so a reference mode of frequency ``omega_ref``
sits in the engine column with ``omega_eng = -omega_ref`` (the
labeled branch signs flip accordingly).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.nonhydro2 as nh
from fridom.framework2.model.eigen_channel import (
    UNLABELED,
    ChannelEigenbasis,
)
from fridom.nonhydro2.channel_eigenmodes import (
    CONSTRAINT,
    FAMILIES,
    KELVIN_MINUS,
    KELVIN_PLUS,
    VORTICAL,
    WAVE_MINUS,
    WAVE_PLUS,
    ChannelEigenmodes,
    constraint_overlap,
    label_channel_modes,
)

N = 8
NZH = N // 2 + 1
D = 4 * N - 1  # u: N, v: N - 1 (interior faces), w: N, b: N
F0, N2, DSQR = 1.5, 3.0, 2.0
BETA = 0.5
LX = LZ = 2 * np.pi
LY = 1.0
DX, DY, DZ = LX / N, LY / N, LZ / N

# reference y grids: shifted centres incl. the wall / plain centres
Y_FACE = DY * np.arange(1, N + 1)
Y_CENTER = DY * (np.arange(N) + 0.5)


# ================================================================
#  Model builders and module-scoped fixtures (compile reuse)
# ================================================================
def make_channel(f0=F0, beta=None):
    """Build the linear walled-y nonhydro channel model."""
    mx = fr.grid.meshes.IntervalMesh(N, (0.0, LX), periodic=True,
                                     name="x")
    my = fr.grid.meshes.IntervalMesh(N, (0.0, LY), periodic=False,
                                     name="y")
    mz = fr.grid.meshes.IntervalMesh(N, (0.0, LZ), periodic=True,
                                     name="z")
    coriolis = (nh.FPlaneCoriolis(f0=f0) if beta is None
                else nh.BetaPlaneCoriolis(f0=f0, beta=beta))
    return nh.Model(
        grid=fr.grid.Grid((mx, my, mz)), advection=False, dsqr=DSQR,
        coriolis=coriolis,
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def model():
    """One f-plane channel model shared across the module."""
    return make_channel()


@pytest.fixture(scope="module")
def em(model):
    """Label the channel eigenmodes of the shared f-plane model."""
    return ChannelEigenmodes(model)


@pytest.fixture(scope="module")
def beta_em():
    """Labeled eigenmodes with f varying along the dense y axis."""
    return ChannelEigenmodes(make_channel(beta=BETA))


# ================================================================
#  The expected per-plane family counts (f-plane)
# ================================================================
def expected_counts():
    """Return the per-plane f-plane count matrix of each family."""
    vortical = np.full((N, NZH), N - 1)
    vortical[0, 0] = N          # the kx = kz = 0 mean-flow strata
    vortical[0, 1:] = N + 1     # kx = 0: u columns + constant-p
    vortical[1:, -1] = N + 1    # kz Nyquist: buoyancy decouples
    vortical[N // 2, -1] = 3 * N - 1  # both Nyquists: L = 0
    constraint = np.full((N, NZH), N)
    constraint[0, 0] = N - 1    # the constant pressure drops out
    kelvin = np.zeros((N, NZH), dtype=int)
    kelvin[1:, 1:-1] = 1        # kx != 0, interior kz, per sign
    wave = np.full((N, NZH), N - 1)
    wave[:, 0] = N              # kz = 0: buoyancy strata, per sign
    wave[N // 2, -1] = 0
    assert (vortical + constraint + 2 * kelvin + 2 * wave == D).all()
    return vortical, constraint, kelvin, wave


# ================================================================
#  The ported closed forms (boundary_emission reference)
# ================================================================
def sym(ik, d):
    """Staggered derivative/interp symbols at integer plane ik."""
    theta = 2.0 * np.pi * ik / N
    kh = -1j / d * (np.exp(1j * theta) - 1.0)
    oh = 0.5 * (np.exp(1j * theta) + 1.0)
    return kh, oh


def kelvin_omega(ikx, ikz):
    """Positive Kelvin frequency (f-independent) at (ikx, ikz)."""
    kxh, _ = sym(ikx, DX)
    kzh, ozh = sym(ikz, DZ)
    num = abs(ozh) ** 2 * N2 * abs(kxh) ** 2
    den = DSQR * abs(kxh) ** 2 + abs(kzh) ** 2
    return float(np.sqrt(num / den))


def igw_omega(ikx, ikz, n, s):
    """Poincaré dispersion for the sine stratum n, branch s."""
    kxh, oxh = sym(ikx, DX)
    kzh, ozh = sym(ikz, DZ)
    ky = np.pi * n / LY
    kyh = -1j / DY * (np.exp(1j * ky * DY) - 1.0)
    oyh = 0.5 * (np.exp(1j * ky * DY) + 1.0)
    kh2 = abs(kxh) ** 2 + abs(kyh) ** 2
    f2h = abs(oxh) ** 2 * abs(oyh) ** 2 * F0 ** 2
    n2h = abs(ozh) ** 2 * N2
    return float(s * np.sqrt(
        (f2h * abs(kzh) ** 2 + n2h * kh2)
        / (DSQR * kh2 + abs(kzh) ** 2)))


def buoyancy_omega():
    """Return the kz = 0 buoyancy-oscillation frequency N/delta."""
    return float(np.sqrt(N2 / DSQR))


def sin_structure(n):
    """Return (phi, ave_phi, diff_phi) of the sine stratum n."""
    ky = np.pi * n / LY
    phi = np.sin(ky * Y_FACE)
    ave = 0.5 * (np.sin(ky * Y_FACE) + np.sin(ky * (Y_FACE - DY)))
    diff = (np.sin(ky * Y_FACE) - np.sin(ky * (Y_FACE - DY))) / DY
    return phi, ave, diff


def stack_column(u, v, w, b):
    """Stack reference (u, v, w, b) onto the engine column layout.

    u/w/b keep all N centre entries; v drops the reference's wall
    node (its last entry, exactly zero) to land on the N - 1
    interior faces.
    """
    parts = [np.broadcast_to(np.asarray(a, dtype=complex), (N,))
             for a in (u, v, w, b)]
    return np.concatenate(
        [parts[0], parts[1][:-1], parts[2], parts[3]])


def ref_vortical(ikx, ikz, n):
    """Balanced (thermal-wind) sine eigenfunction, stratum n >= 1."""
    kxh, oxh = sym(ikx, DX)
    kzh, ozh = sym(ikz, DZ)
    phi, ave, diff = sin_structure(n)
    return stack_column(1j * ozh * diff, ozh * np.conj(kxh) * phi,
                        0.0, np.conj(oxh) * kzh * F0 * ave)


def ref_wall_mode(ikz, branch):
    """Return the kx = 0 exponential wall mode (branch +1/-1)."""
    kzh, ozh = sym(ikz, DZ)
    a = 4.0 * abs(ozh) ** 2 * N2
    b = F0 ** 2 * abs(kzh) ** 2 * DY ** 2
    lam = branch * np.arccosh(complex((a + b) / (a - b))) / DY
    yoff = 0.0 if np.real(-lam) > 0 else LY

    def psi(y):
        return np.exp(lam * (y - yoff))

    phi = psi(Y_FACE)
    ave = 0.5 * (phi + psi(Y_FACE - DY))
    diff = (phi - psi(Y_FACE - DY)) / DY
    _, oxh = sym(0, DX)
    return stack_column(1j * ozh * diff, 0.0, 0.0,
                        np.conj(oxh) * kzh * F0 * ave)


def ref_kelvin(ikx, ikz, mode):
    """Kelvin eigenfunction (branch mode = +1/-1) and its omega."""
    kxh, oxh = sym(ikx, DX)
    kzh, ozh = sym(ikz, DZ)
    om = complex(kelvin_omega(ikx, ikz))
    coupling = np.conj(oxh) * kxh * F0 * DY
    lam = np.log((2.0 * om + coupling) / (2.0 * om - coupling)) / DY
    yoff = 0.0 if np.real(mode * lam) > 0 else LY
    struct = np.exp(-mode * lam * (Y_CENTER - yoff))
    u = mode * np.conj(kzh) * om
    w = -mode * np.conj(kxh) * om
    b = 1j * np.conj(ozh) * np.conj(kxh) * N2
    return (stack_column(u * struct, 0.0, w * struct, b * struct),
            float(np.real(mode * om)))


def ref_igw(ikx, ikz, n, s):
    """Poincaré eigenfunction (stratum n >= 1, branch s) and omega."""
    kxh, oxh = sym(ikx, DX)
    kzh, ozh = sym(ikz, DZ)
    om = igw_omega(ikx, ikz, n, s)
    kx2, kz2 = abs(kxh) ** 2, abs(kzh) ** 2
    n2h = abs(ozh) ** 2 * N2
    phi, ave, diff = sin_structure(n)
    u = 1j * (kxh * (n2h - DSQR * om ** 2) * diff
              - oxh * kz2 * F0 * om * ave)
    v = (kx2 * n2h - om ** 2 * (DSQR * kx2 + kz2)) * phi
    w = -1j * kzh * om * (om * diff
                          - oxh * np.conj(kxh) * F0 * ave)
    b = -np.conj(ozh) * kzh * N2 * (om * diff
                                    - oxh * np.conj(kxh) * F0 * ave)
    return stack_column(u, v, w, b), om


def ref_buoyancy(n, s):
    """Kz = 0 buoyancy oscillation (cos stratum n, branch s)."""
    om = buoyancy_omega()
    phi = np.cos(np.pi * n / LY * Y_CENTER)
    return (stack_column(0.0, 0.0, 1j * s * om * phi, N2 * phi),
            s * om)


# ================================================================
#  Engine-side helpers
# ================================================================
def m_inner(em, a, b):
    """Compute the diagonal-metric inner product of stacked columns."""
    metric = np.asarray(em.metric)
    return np.sum(np.conj(np.asarray(a)) * metric * np.asarray(b))


def engine_column(em, ikx, ikz, omega_ref):
    """Engine column carrying the reference mode of omega_ref."""
    omega = np.asarray(em.omega[ikx, ikz])
    col = int(np.argmin(np.abs(omega + omega_ref)))
    assert abs(omega[col] + omega_ref) < 1e-11
    return col


def column_overlap(em, ikx, ikz, col, ref):
    """Phase-free M-overlap of a reference vs an engine column."""
    eng = np.asarray(em.q[ikx, ikz])[:, col]
    return abs(m_inner(em, ref, eng)) / np.sqrt(
        abs(m_inner(em, ref, ref)) * abs(m_inner(em, eng, eng)))


def labeled_membership(em, ikx, ikz, ref, codes):
    """|P ref|_M / |ref|_M onto the span of the labeled columns.

    The steady families are degenerate (omega = 0 across the
    branch), so single-column overlaps are meaningless there; the
    right oracle is membership in the labeled eigenspace — which
    also proves the constraint split: a physical reference must lie
    entirely inside the VORTICAL-labeled columns, with nothing
    leaking into the CONSTRAINT ones.
    """
    labels = np.asarray(em.labels[ikx, ikz])
    cols = np.isin(labels, codes)
    q = np.asarray(em.q[ikx, ikz])[:, cols]
    metric = np.asarray(em.metric)
    coef = np.einsum("dj,d,d->j", np.conj(q), metric,
                     np.asarray(ref))
    proj = q @ coef
    return float(np.sqrt(abs(m_inner(em, proj, proj))
                         / abs(m_inner(em, ref, ref))))


def family_columns(em, ikx, ikz, family):
    """Column indices labeled with the named family at one plane."""
    return np.where(
        np.asarray(em.labels[ikx, ikz]) == em.families[family])[0]


def make_synthetic_basis(omega, q):
    """Build a hand-crafted 3-D ChannelEigenbasis for the labeler.

    Layout: u: 0-1, v: 2 (one interior face), w: 3-4, b: 5-6 —
    a 2-cell bounded axis (n = 2).
    """
    d = 7
    slices = {"u": slice(0, 2), "v": slice(2, 3), "w": slice(3, 5),
              "b": slice(5, d)}
    return ChannelEigenbasis(
        omega=jnp.asarray(omega, dtype=float),
        q=jnp.asarray(q, dtype=complex),
        components=("u", "v", "w", "b"), slices=slices,
        metric=jnp.ones(d), periodic_axis="z", bounded_axis="y",
        hermiticity_error=0.0)


def unit(d, idx):
    """Return a complex unit vector of length d."""
    e = np.zeros(d, dtype=complex)
    e[idx] = 1.0
    return e


# ================================================================
#  (a) The labeled surface and the family counts
# ================================================================
def test_surface_passthroughs_and_family_map(model, em):
    assert isinstance(em, ChannelEigenmodes)
    assert em.components == ("u", "v", "w", "b")
    assert em.slices["u"] == slice(0, N)
    assert em.slices["v"] == slice(N, 2 * N - 1)
    assert em.slices["w"] == slice(2 * N - 1, 3 * N - 1)
    assert em.slices["b"] == slice(3 * N - 1, D)
    assert em.periodic_axis == "z"
    assert em.bounded_axis == "y"
    assert em.omega.shape == (N, NZH, D)
    assert em.q.shape == (N, NZH, D, D)
    assert em.labels.shape == (N, NZH, D)
    assert em.families == FAMILIES
    assert em.family_names[em.families["constraint"]] == "constraint"
    assert em.grid is model.grid
    for name in em.components:
        assert em.spaces[name] is model.state[
            name].function_space.bare


def test_fplane_labels_are_complete_with_expected_counts(em):
    labels = np.asarray(em.labels)
    assert (labels != UNLABELED).all()
    vortical, constraint, kelvin, wave = expected_counts()
    assert ((labels == VORTICAL).sum(axis=-1) == vortical).all()
    assert ((labels == CONSTRAINT).sum(axis=-1) == constraint).all()
    assert ((labels == KELVIN_PLUS).sum(axis=-1) == kelvin).all()
    assert ((labels == KELVIN_MINUS).sum(axis=-1) == kelvin).all()
    assert ((labels == WAVE_PLUS).sum(axis=-1) == wave).all()
    assert ((labels == WAVE_MINUS).sum(axis=-1) == wave).all()


def test_eigenvectors_stay_m_orthonormal_after_label_rotation(em):
    # the zero-space overlap split and the Kelvin recovery rotate
    # basis.q in place — unitarily
    assert float(em.basis.orthonormality_error()) < 1e-12


# ================================================================
#  (b) Dispersion: the closed forms at EVERY plane (alignment-free)
# ================================================================
def analytic_spectrum(ikx, ikz):
    """Return the sorted analytic frequency multiset of one plane."""
    vortical, constraint, _, _ = expected_counts()
    if ikx == N // 2 and ikz == NZH - 1:
        return np.zeros(D)  # both Nyquists: L = 0 on the plane
    vals = [0.0] * int(vortical[ikx, ikz] + constraint[ikx, ikz])
    if ikz == 0:
        vals += [s * buoyancy_omega()
                 for s in (-1, 1) for _ in range(N)]
    else:
        if ikx != 0 and ikz != NZH - 1:
            om = kelvin_omega(ikx, ikz)
            vals += [-om, om]
        vals += [igw_omega(ikx, ikz, n, s)
                 for n in range(1, N) for s in (-1, 1)]
    return np.sort(np.asarray(vals))


def test_spectrum_matches_the_analytic_multiset(em):
    omega = np.asarray(em.omega)
    for ikx in range(N):
        for ikz in range(NZH):
            expect = analytic_spectrum(ikx, ikz)
            got = np.sort(omega[ikx, ikz])
            assert got.shape == expect.shape
            assert np.abs(got - expect).max() < 1e-12, (ikx, ikz)


def test_kelvin_pair_matches_the_closed_form(em):
    # the labeled kelvin+/- columns carry the f-independent Kelvin
    # dispersion (one internal Kelvin pair per vertical wavenumber)
    omega = np.asarray(em.omega)
    for ikx in range(1, N):
        for ikz in range(1, NZH - 1):
            expect = kelvin_omega(ikx, ikz)
            (plus,) = family_columns(em, ikx, ikz, "kelvin+")
            (minus,) = family_columns(em, ikx, ikz, "kelvin-")
            assert abs(omega[ikx, ikz, plus] - expect) < 1e-12
            assert abs(omega[ikx, ikz, minus] + expect) < 1e-12


def test_kz0_nonzero_columns_are_the_buoyancy_oscillations(em):
    # at kz = 0 buoyancy oscillates vertically at N/delta on every
    # stratum (u = v = 0), a wave-family branch — deliberately NOT
    # Kelvin although the wall-normal energy vanishes there too
    omega = np.asarray(em.omega)
    labels = np.asarray(em.labels)
    for ikx in range(N):
        nonzero = np.abs(omega[ikx, 0]) > 1e-8
        assert np.abs(np.abs(omega[ikx, 0][nonzero])
                      - buoyancy_omega()).max() < 1e-12
        assert np.isin(labels[ikx, 0][nonzero],
                       (WAVE_PLUS, WAVE_MINUS)).all()


# ================================================================
#  (c) Eigenvector agreement per family (f-plane)
# ================================================================
@pytest.mark.parametrize(("ikx", "ikz", "mode"), [
    pytest.param(1, 1, 1, id="kx1-kz1-plus"),
    pytest.param(1, 1, -1, id="kx1-kz1-minus"),
    pytest.param(2, 3, 1, id="kx2-kz3-plus"),
    pytest.param(3, 2, -1, id="kx3-kz2-minus"),
    pytest.param(4, 1, 1, id="x-nyquist-plus"),
    pytest.param(7, 3, -1, id="negative-kx-minus"),
])
def test_kelvin_columns_match(em, ikx, ikz, mode):
    ref, om = ref_kelvin(ikx, ikz, mode)
    col = engine_column(em, ikx, ikz, om)
    assert column_overlap(em, ikx, ikz, col, ref) > 1 - 1e-12
    # the reference +omega branch is the engine's kelvin- (the
    # e^{-i omega t} vs L q = +i omega q time-convention flip)
    expect = KELVIN_MINUS if mode == 1 else KELVIN_PLUS
    assert np.asarray(em.labels)[ikx, ikz, col] == expect


@pytest.mark.parametrize(("ikx", "ikz", "n", "s"), [
    pytest.param(1, 1, 1, 1, id="kx1-kz1-n1-plus"),
    pytest.param(1, 1, 3, -1, id="kx1-kz1-n3-minus"),
    pytest.param(2, 2, 5, 1, id="kx2-kz2-n5-plus"),
    pytest.param(0, 1, 2, 1, id="kx0-kz1-n2-plus"),
    pytest.param(3, 4, 2, -1, id="z-nyquist-n2-minus"),
    pytest.param(4, 2, 7, 1, id="x-nyquist-n7-plus"),
])
def test_poincare_columns_match(em, ikx, ikz, n, s):
    ref, om = ref_igw(ikx, ikz, n, s)
    col = engine_column(em, ikx, ikz, om)
    assert column_overlap(em, ikx, ikz, col, ref) > 1 - 1e-12


@pytest.mark.parametrize(("ikx", "ikz", "n"), [
    pytest.param(1, 1, 1, id="kx1-kz1-n1"),
    pytest.param(2, 3, 4, id="kx2-kz3-n4"),
    pytest.param(3, 0, 2, id="kx3-kz0-n2"),
    pytest.param(0, 2, 3, id="kx0-kz2-n3"),
    pytest.param(4, 2, N - 1, id="x-nyquist-n7"),
])
def test_vortical_sines_lie_in_the_labeled_steady_space(
        em, ikx, ikz, n):
    # membership in the VORTICAL-labeled span — nothing may leak
    # into the CONSTRAINT columns of the same (degenerate) zero
    # space, so this also certifies the Leray-overlap split
    ref = ref_vortical(ikx, ikz, n)
    assert labeled_membership(em, ikx, ikz, ref,
                              (VORTICAL,)) > 1 - 1e-12


@pytest.mark.parametrize(("ikz", "branch"), [
    pytest.param(1, 1, id="kz1-plus"),
    pytest.param(1, -1, id="kz1-minus"),
    pytest.param(3, 1, id="kz3-plus"),
])
def test_wall_modes_lie_in_the_kx0_steady_space(em, ikz, branch):
    ref = ref_wall_mode(ikz, branch)
    assert labeled_membership(em, 0, ikz, ref,
                              (VORTICAL,)) > 1 - 1e-12


@pytest.mark.parametrize(("ikx", "n", "s"), [
    pytest.param(1, 0, 1, id="kx1-n0-plus"),
    pytest.param(2, 3, -1, id="kx2-n3-minus"),
    pytest.param(0, 2, 1, id="kx0-n2-plus"),
])
def test_buoyancy_strata_lie_in_the_signed_wave_space(em, ikx, n, s):
    # the kz = 0 branch is N-fold degenerate per sign, so the oracle
    # is membership in the (sign-matching) wave eigenspace; the
    # time-convention flip maps the +omega reference to wave-
    ref, _ = ref_buoyancy(n, s)
    code = WAVE_MINUS if s == 1 else WAVE_PLUS
    assert labeled_membership(em, ikx, 0, ref, (code,)) > 1 - 1e-12


# ================================================================
#  (d) The constraint family: crisp Leray-kernel directions
# ================================================================
def test_kelvin_wall_normal_energy_is_machine_crisp(em):
    # report-grade margins: kelvin columns carry ~1e-30 v-energy,
    # every other nonzero column on the kelvin-eligible planes at
    # least ~1e-3 (measured 9.1e-4) — a twenty-decade gap
    q = np.asarray(em.q)
    labels = np.asarray(em.labels)
    omega = np.asarray(em.omega)
    metric = np.asarray(em.metric)
    v = em.slices["v"]
    energy = np.einsum("xzdj,d->xzj", np.abs(q[:, :, v, :]) ** 2,
                       metric[v])
    kelvin = np.isin(labels, (KELVIN_PLUS, KELVIN_MINUS))
    assert kelvin.any()
    assert energy[kelvin].max() < 1e-25
    eligible = np.zeros_like(kelvin)
    eligible[1:, 1:-1, :] = True
    others = eligible & ~kelvin & (np.abs(omega) > 1e-8)
    assert energy[others].min() > 1e-4


def test_constraint_columns_are_annihilated_by_the_projector(
        model, em):
    # synthesize Re(q e^{i(kx x + kz z)}) for one constraint column
    # and one vortical column of an interior plane: the public Leray
    # matvec must annihilate the former and fix the latter (whose
    # constrained tendency also vanishes — a steady mode)
    ikx = ikz = 1
    q = np.asarray(em.q)
    px = np.exp(2j * np.pi * ikx * np.arange(N) / N)
    pz = np.exp(2j * np.pi * ikz * np.arange(N) / N)

    def synthesize(col):
        q_col = q[ikx, ikz, :, col]
        return {
            name: np.real(px[:, None, None] * pz[None, None, :]
                          * q_col[em.slices[name]][None, :, None])
            for name in em.components}

    (constraint_col,) = family_columns(em, ikx, ikz,
                                       "constraint")[:1]
    model.set_fields(**synthesize(int(constraint_col)))
    projected = model.constrain(model.state)
    scale = max(np.abs(np.asarray(model.state[c].data)).max()
                for c in em.components)
    assert max(np.abs(np.asarray(projected[c].data)).max()
               for c in em.components) < 1e-12 * scale

    (vortical_col,) = family_columns(em, ikx, ikz, "vortical")[:1]
    model.set_fields(**synthesize(int(vortical_col)))
    projected = model.constrain(model.state)
    err = max(np.abs(np.asarray(projected[c].data)
                     - np.asarray(model.state[c].data)).max()
              for c in em.components)
    assert err < 1e-12
    tendency = model.tendency(projected, constraints=True)
    assert max(np.abs(np.asarray(tendency[c].data)).max()
               for c in em.components) < 1e-10


def test_constraint_split_margins_are_machine_crisp(model, em):
    # re-probe the Leray overlap on the labeled (rotated) basis: the
    # diagonal must sit on {0, 1} exactly by label — the measured
    # crispness margin of the zero-space split
    overlap = np.asarray(constraint_overlap(model, em.basis))
    diag = np.einsum("xzii->xzi", overlap).real
    labels = np.asarray(em.labels)
    assert np.abs(diag[labels == CONSTRAINT]).max() < 1e-12
    assert np.abs(diag[labels == VORTICAL] - 1.0).max() < 1e-12


# ================================================================
#  (e) The beta-plane channel
# ================================================================
def test_beta_labels_are_complete_and_kelvin_stays_crisp(beta_em):
    labels = np.asarray(beta_em.labels)
    omega = np.asarray(beta_em.omega)
    assert (labels != UNLABELED).all()
    # kelvin: same planes as the f-plane, machine-crisp v-energy,
    # and the f-INDEPENDENT Kelvin dispersion survives beta exactly
    q = np.asarray(beta_em.q)
    metric = np.asarray(beta_em.metric)
    v = beta_em.slices["v"]
    for ikx in (1, 3, 4, 7):
        for ikz in (1, 3):
            for family, sign in (("kelvin+", 1), ("kelvin-", -1)):
                (col,) = family_columns(beta_em, ikx, ikz, family)
                v_energy = float(np.sum(
                    np.abs(q[ikx, ikz, v, col]) ** 2 * metric[v]))
                assert v_energy < 1e-25
                assert abs(omega[ikx, ikz, col]
                           - sign * kelvin_omega(ikx, ikz)) < 1e-12
    # the constraint family is beta-blind (the Leray kernel does not
    # feel f): same counts as the f-plane
    _, constraint, _, _ = expected_counts()
    assert ((labels == CONSTRAINT).sum(axis=-1) == constraint).all()


def test_beta_vortical_branch_acquires_rossby_frequencies(beta_em):
    # on kx not in {0, Nyquist}, kz != Nyquist planes ALL steady
    # modes become slow westward Rossby modes, split from the wave
    # band across the clean spectral gap (beta = 0.5)
    labels = np.asarray(beta_em.labels)
    omega = np.asarray(beta_em.omega)
    interior_kx = [i for i in range(1, N) if i != N // 2]
    for ikx in interior_kx:
        for ikz in range(NZH - 1):
            vort = omega[ikx, ikz][labels[ikx, ikz] == VORTICAL]
            assert vort.size == N - 1
            assert (np.abs(vort) > 1e-10).all()
    # kx = 0 and kx = Nyquist keep steady columns (the x-average of
    # a Nyquist mode vanishes, so rotation decouples there)
    for ikx in (0, N // 2):
        zeros = np.abs(omega[ikx]) < 1e-8
        steady = labels[ikx] == VORTICAL
        assert (steady == (zeros
                           & (labels[ikx] != CONSTRAINT))).all()


def test_beta_nyquist_strata_overlap_the_slow_band(beta_em):
    # the documented blur: on the kz-Nyquist planes buoyancy
    # decouples and the inertial strata slide into genuinely slow
    # frequencies under beta — they are wave-labeled by exact count,
    # yet overlap the Rossby band, so NO global frequency threshold
    # separates vortical from wave; label-aware predicates are the
    # tool there
    labels = np.asarray(beta_em.labels)
    omega = np.asarray(beta_em.omega)
    wave = np.isin(labels, (WAVE_PLUS, WAVE_MINUS))
    vort = labels == VORTICAL
    nonzero_vort = vort & (np.abs(omega) > 1e-10)
    assert np.abs(omega[wave]).min() < np.abs(
        omega[nonzero_vort]).max()
    # per plane (the labeler's split unit) the gap IS clean
    for ikx in range(N):
        for ikz in range(NZH):
            slow = np.abs(omega[ikx, ikz][nonzero_vort[ikx, ikz]])
            fast = np.abs(omega[ikx, ikz][wave[ikx, ikz]])
            if slow.size and fast.size:
                assert fast.min() >= 10 * slow.max(), (ikx, ikz)


# ================================================================
#  (e') An odd z resolution: no z-Nyquist plane at all
# ================================================================
def test_odd_nz_channel_labels_and_completeness():
    # Nz = 5 (odd): the rfft half spectrum carries no self-conjugate
    # Nyquist slot, the overlap probe recombines only the kz = 0
    # planes, and every interior kz plane carries a Kelvin pair
    nz = 5
    mx = fr.grid.meshes.IntervalMesh(N, (0.0, LX), periodic=True,
                                     name="x")
    my = fr.grid.meshes.IntervalMesh(N, (0.0, LY), periodic=False,
                                     name="y")
    mz = fr.grid.meshes.IntervalMesh(nz, (0.0, LZ), periodic=True,
                                     name="z")
    model = nh.Model(
        grid=fr.grid.Grid((mx, my, mz)), advection=False, dsqr=DSQR,
        coriolis=nh.FPlaneCoriolis(f0=F0),
        stratification=nh.ConstantStratification(n2=N2),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))
    em = ChannelEigenmodes(model)
    labels = np.asarray(em.labels)
    assert em.omega.shape == (N, nz // 2 + 1, D)
    assert (labels != UNLABELED).all()
    assert float(em.basis.orthonormality_error()) < 1e-12
    kelvin = (labels == KELVIN_PLUS).sum(axis=-1)
    assert (kelvin[1:, 1:] == 1).all()
    assert (kelvin[0, :] == 0).all()
    assert (kelvin[:, 0] == 0).all()
    # the physical families still sum to the Leray projector
    rng = np.random.default_rng(3)
    model.set_fields(**{
        c: rng.standard_normal(np.asarray(model.state[c].data).shape)
        for c in em.components})
    z = nh.State({c: model.state[c] for c in em.components})
    parts = [em.projector(sel)(z)
             for sel in ("vortical", "wave", "kelvin")]
    total = {c: sum(np.asarray(p[c].data) for p in parts)
             for c in em.components}
    projected = model.constrain(z)
    assert max(
        float(np.abs(total[c]
                     - np.asarray(projected[c].data)).max())
        for c in em.components) < 1e-12


# ================================================================
#  (f) The labeler: defensive paths on synthetic bases
# ================================================================
def synthetic_setup():
    """Toy 4-plane basis exercising the labeler's branches.

    Plane grid (n_kx = 3, n_kz = 2), D = 7 (n = 2 bounded cells,
    v index 2): the wave counts are 4 on kz = 0 planes and 2 on the
    kz != 0 planes. Every plane is genuinely M-orthonormal
    (metric = 1), so the unitarity of the label rotations is
    observable.
    """
    d = 7
    eye = np.eye(d, dtype=complex)
    omega = np.zeros((3, 2, d))
    q = np.tile(eye, (3, 2, 1, 1))
    diag = np.ones((3, 2, d))

    # (0, 0): kz = 0 plane — 3 zeros (2 constraint + 1 steady),
    # 4 nonzero -> all wave by the kz = 0 count
    omega[0, 0] = [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    diag[0, 0, 2:5] = [0.0, 0.0, 1.0]

    # (1, 1): kelvin- crisp at -1; the +1 pair is a degenerate
    # mixture of a v-free direction (a) and a v-full one (b) ->
    # Gram recovery rotates it apart
    a = unit(d, 5)                                # v-free
    b = (unit(d, 2) - unit(d, 6)) / np.sqrt(2.0)  # v-full
    omega[1, 1] = [-2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    q[1, 1] = np.stack([
        (unit(d, 2) + unit(d, 6)) / np.sqrt(2.0), unit(d, 0),
        unit(d, 1), unit(d, 3), unit(d, 4),
        (a + b) / np.sqrt(2.0), (a - b) / np.sqrt(2.0),
    ], axis=1)
    diag[1, 1, 2:5] = [1.0, 0.0, 1.0]

    # (2, 0): kz = 0 with 6 nonzero columns vs the structural wave
    # count 4 and no clean gap (1 vs 2) -> stays UNLABELED
    omega[2, 0] = [-2.0, -1.0, -1.0, 0.0, 1.0, 1.0, 2.0]
    diag[2, 0, 3] = 1.0

    # (1, 0): a plane WITHOUT zero columns (the empty zero-space
    # branch): 7 nonzero vs the kz = 0 wave count 4, no clean gap
    omega[1, 0] = [-2.0, -1.0, -1.0, -1.0, 1.0, 1.0, 2.0]

    # (2, 1): one non-crisp candidate per sign and no degenerate
    # partner -> the kelvin recovery returns None; the remainder
    # matches the wave count and labels by sign
    omega[2, 1] = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    q[2, 1] = np.stack([
        (unit(d, 2) + unit(d, 3)) / np.sqrt(2.0),
        unit(d, 0), unit(d, 1), unit(d, 4), unit(d, 5), unit(d, 6),
        (unit(d, 2) - unit(d, 3)) / np.sqrt(2.0),
    ], axis=1)
    diag[2, 1, 1:6] = [0.0, 1.0, 1.0, 1.0, 1.0]

    overlap = np.zeros((3, 2, d, d), dtype=complex)
    overlap[..., np.arange(d), np.arange(d)] = diag
    return make_synthetic_basis(omega, q), jnp.asarray(overlap)


def test_labeler_on_the_synthetic_planes():
    basis, overlap = synthetic_setup()
    labels = np.asarray(label_channel_modes(basis, overlap=overlap))
    # (0, 0): eigh sorts the zero-space eigenvalues ascending, so
    # the two constraint columns come first after the rotation
    assert (labels[0, 0] == [WAVE_MINUS, WAVE_MINUS, CONSTRAINT,
                             CONSTRAINT, VORTICAL, WAVE_PLUS,
                             WAVE_PLUS]).all()
    # (1, 1): the recovered kelvin+ column is the v-free direction
    assert (labels[1, 1] == [WAVE_MINUS, KELVIN_MINUS, CONSTRAINT,
                             VORTICAL, VORTICAL, KELVIN_PLUS,
                             WAVE_PLUS]).all()
    rotated = np.asarray(basis.q)[1, 1]
    assert abs(abs(rotated[:, 5] @ np.conj(unit(7, 5))) - 1.0) < 1e-12
    # (2, 0): no clean gap on a kz = 0 plane -> nonzero UNLABELED
    assert (labels[2, 0] == [UNLABELED, UNLABELED, UNLABELED,
                             VORTICAL, UNLABELED, UNLABELED,
                             UNLABELED]).all()
    # (1, 0): an empty zero space is fine; no gap -> all UNLABELED
    assert (labels[1, 0] == UNLABELED).all()
    # (2, 1): kelvin recovery finds nothing; wave labels by sign
    assert (labels[2, 1] == [WAVE_MINUS, CONSTRAINT, VORTICAL,
                             VORTICAL, VORTICAL, VORTICAL,
                             WAVE_PLUS]).all()
    assert float(basis.orthonormality_error()) < 1e-12


def test_labeler_leaves_non_crisp_zero_columns_unlabeled():
    basis, overlap = synthetic_setup()
    smeared = np.asarray(overlap).copy()
    smeared[0, 0, 3, 3] = 0.5  # neither kernel nor fixed point
    labels = np.asarray(label_channel_modes(
        basis, overlap=jnp.asarray(smeared)))
    plane = labels[0, 0]
    assert (plane == UNLABELED).sum() == 1
    assert (plane == CONSTRAINT).sum() == 1
    assert (plane == VORTICAL).sum() == 1


def test_labeler_override_hook():
    basis, overlap = synthetic_setup()

    def drop_plane(_basis, labels):
        return labels.at[0, 0].set(UNLABELED)

    labels = np.asarray(label_channel_modes(
        basis, overlap=overlap, override=drop_plane))
    assert (labels[0, 0] == UNLABELED).all()
    assert (labels[1, 1] != UNLABELED).all()


def test_labeler_rejects_bad_shapes():
    basis, overlap = synthetic_setup()
    with pytest.raises(ValueError, match=r"\(n_kx, n_kz, D\)"):
        label_channel_modes(
            make_synthetic_basis(np.zeros((2, 7)),
                                 np.tile(np.eye(7), (2, 1, 1))),
            overlap=overlap)
    with pytest.raises(ValueError, match=r"one \(D, D\) plane"):
        label_channel_modes(basis, overlap=overlap[..., :3])


def test_constraint_overlap_rejects_a_2d_basis():
    flat = make_synthetic_basis(np.zeros((2, 7)),
                                np.tile(np.eye(7), (2, 1, 1)))
    with pytest.raises(ValueError, match=r"\(n_kx, n_kz, D, D\)"):
        constraint_overlap(object(), flat)
