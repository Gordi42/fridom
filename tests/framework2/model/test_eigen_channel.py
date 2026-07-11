"""Dense-column channel eigenbasis on the walled channel models.

Validates the C3 engine (``fridom.model.eigen_channel``) on
the linear rotating channel: gates, Hermiticity, M-orthonormality, the
per-plane zero-mode counts, and the strong per-column eigen relation
``L Re(q e^{i kx x}) = Re(i omega q e^{i kx x})`` evaluated through the
real model matvec ``model.tendency`` — on the shallow-water f-plane
channel, the beta-plane channel (coefficients varying along the dense
axis), and the CONSTRAINED 3-D nonhydro channel (walled y, the
``P L P`` probe through ``model.constrain`` + the constrained
tendency, cross-checked against the trig-analytic f0 = 0 oracle).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.spatial.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.model import eigen_channel
from fridom.model.eigen_channel import (
    UNLABELED,
    ChannelEigenbasis,
    channel_eigenpairs,
)

N = 8
D = 3 * N - 1  # u: N, v: N - 1 (inner faces), p: N
N_KX = N // 2 + 1
CSQR = 0.7
F0 = 1.0
BETA = 2.0


# ================================================================
#  Model builders and module-scoped fixtures (compile reuse)
# ================================================================
def make_grid(n=N, *, periodic_x=True, periodic_y=False):
    """Return a small channel grid (periodic x, walled y by default)."""
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                     periodic=periodic_x, name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                     periodic=periodic_y, name="y")
    return fr.spatial.Grid((mx, my))


def make_walled_model(grid=None, *, coriolis=None):
    """Build the linear walled shallow-water channel model."""
    if coriolis is None:
        coriolis = sw.modules.FPlaneCoriolis(f0=F0)
    return sw.Model(
        grid=grid if grid is not None else make_grid(),
        csqr=CSQR, rossby_number=0.2,
        coriolis=coriolis, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def model():
    """One walled channel model shared across the module."""
    return make_walled_model()


@pytest.fixture(scope="module")
def basis(model):
    """Compute the channel eigenbasis of the shared model."""
    return channel_eigenpairs(model)


@pytest.fixture(scope="module")
def beta_model():
    """Build a walled channel with f varying along the dense y axis."""
    return make_walled_model(
        coriolis=fr.model.modules.BetaPlaneCoriolis(f0=F0, beta=BETA))


@pytest.fixture(scope="module")
def beta_basis(beta_model):
    """Compute the channel eigenbasis of the beta-plane model."""
    return channel_eigenpairs(beta_model)


def eigen_relation_residual(model, basis, kx, col):
    """Drive the real matvec with one column; return (resid, scale).

    Builds the real physical state ``Re(q(y) e^{i kx x})`` on the true
    component shapes, applies ``model.tendency``, and compares the
    rfft ``kx`` plane against ``i omega q`` (half-spectrum amplitude
    ``N/2``).
    """
    q_col = np.asarray(basis.q[kx, :, col])
    omega = float(basis.omega[kx, col])
    phase = np.exp(2j * np.pi * kx * np.arange(N) / N)
    model.set_fields(**{
        name: np.real(
            phase[:, None] * q_col[basis.slices[name]][None, :])
        for name in basis.components})
    tendency = model.tendency(model.state)

    residual = 0.0
    for name in basis.components:
        plane = np.fft.rfft(np.asarray(tendency[name].data),
                            axis=0)[kx]
        expect = 1j * omega * (N / 2) * q_col[basis.slices[name]]
        residual = max(residual, float(np.abs(plane - expect).max()))
    return residual, (1.0 + abs(omega)) * (N / 2)


# ================================================================
#  Gates: grid topology and constraint stages
# ================================================================
def test_rejects_a_fully_periodic_grid():
    periodic = make_walled_model(make_grid(periodic_y=True))
    with pytest.raises(ValueError, match="numeric_eigenpairs"):
        channel_eigenpairs(periodic)


def test_rejects_two_bounded_axes():
    box = make_walled_model(make_grid(periodic_x=False))
    with pytest.raises(ValueError,
                       match=r"exactly one bounded axis.*'x', 'y'"):
        channel_eigenpairs(box)


def test_a_non_hermitian_pencil_is_a_taught_error(monkeypatch):
    # the residual of iMS is strictly positive in floating point, so a
    # zero tolerance exercises the skew-adjointness assertion
    monkeypatch.setattr(eigen_channel, "_HERMITICITY_TOL", 0.0)
    with pytest.raises(ValueError, match=r"not skew-adjoint"):
        channel_eigenpairs(make_walled_model())


# ================================================================
#  Layout conventions (the C4 labeler / C5 projector contract)
# ================================================================
def test_layout_conventions(basis):
    assert isinstance(basis, ChannelEigenbasis)
    assert basis.components == ("u", "v", "p")
    assert basis.slices["u"] == slice(0, N)
    assert basis.slices["v"] == slice(N, 2 * N - 1)
    assert basis.slices["p"] == slice(2 * N - 1, D)
    assert basis.periodic_axis == "x"
    assert basis.bounded_axis == "y"
    assert basis.omega.shape == (N_KX, D)
    assert basis.q.shape == (N_KX, D, D)
    assert not np.iscomplexobj(np.asarray(basis.omega))


def test_metric_is_weight_times_bounded_measure(basis):
    # u, v carry weight 1, p carries 1/c^2; uniform dy = 1/N on both
    # the N centres (u, p) and the N - 1 interior faces (v)
    metric = np.asarray(basis.metric)
    assert metric.shape == (D,)
    assert np.allclose(metric[: 2 * N - 1], 1.0 / N)
    assert np.allclose(metric[2 * N - 1:], 1.0 / (N * CSQR))


# ================================================================
#  Hermiticity and M-orthonormality
# ================================================================
def test_pencil_is_hermitian_before_symmetrization(basis):
    assert basis.hermiticity_error < 1e-13


def test_eigenvectors_are_m_orthonormal(basis):
    assert float(basis.orthonormality_error()) < 1e-12


# ================================================================
#  The spectrum: zero-mode counts and +/- pairing
# ================================================================
def test_zero_mode_counts_per_plane(basis):
    # kx != 0: N - 1 vortical zeros; kx = 0: N + 1 zeros
    omega = np.asarray(basis.omega)
    zeros = np.sum(np.abs(omega) < 1e-8, axis=-1)
    assert zeros[0] == N + 1
    assert (zeros[1:] == N - 1).all()


def test_spectrum_is_plus_minus_paired(basis):
    # the M-skew-adjoint operator has a +/- symmetric spectrum
    omega = np.asarray(basis.omega)
    assert np.abs(omega + omega[:, ::-1]).max() < 1e-10


# ================================================================
#  The strong test: columns solve the real model's eigen relation
# ================================================================
@pytest.mark.parametrize(("kx", "col"), [
    pytest.param(1, 0, id="kx1-bottom-poincare"),
    pytest.param(2, D // 2, id="kx2-vortical-zero"),
    pytest.param(3, D - 1, id="kx3-top-poincare"),
])
def test_columns_satisfy_the_eigen_relation(model, basis, kx, col):
    residual, scale = eigen_relation_residual(model, basis, kx, col)
    assert residual < 1e-11 * scale


# ================================================================
#  The beta-plane channel: coefficients vary along the dense axis
# ================================================================
def test_beta_pencil_is_hermitian(beta_basis):
    # f(y) = f0 + beta*y varies along the DENSE axis only; the
    # (energy-conserving) rotation does no work for any f profile,
    # so iMS stays Hermitian to machine precision
    assert beta_basis.hermiticity_error < 1e-13


def test_beta_eigenvectors_are_m_orthonormal(beta_basis):
    assert float(beta_basis.orthonormality_error()) < 1e-12


def test_beta_vortical_branch_acquires_rossby_frequencies(beta_basis):
    # on the f-plane every kx != 0 plane carries N - 1 vortical
    # zeros; with beta ALL of them become slow westward Rossby modes
    # (a steady kx != 0 mode would need v = 0 and geostrophic balance
    # at once - contradictory), and the Nyquist plane decouples from
    # the rotation entirely (the x-average of a Nyquist mode is 0)
    omega = np.asarray(beta_basis.omega)
    zeros = np.sum(np.abs(omega) < 1e-8, axis=-1)
    assert zeros[0] == N + 1
    assert (zeros[1:-1] == 0).all()
    assert zeros[-1] == N - 1
    # the slow band is genuinely slow: distinct from the Poincare
    # branches (|omega| >= f0) yet nonzero
    slow = np.abs(omega[1])[np.abs(omega[1]) < F0]
    assert (np.sort(slow) > 1e-3).all()


@pytest.mark.parametrize(("kx", "col"), [
    pytest.param(1, 0, id="kx1-bottom-poincare"),
    pytest.param(2, D // 2, id="kx2-slow-rossby-a"),
    pytest.param(2, D // 2 + 1, id="kx2-slow-rossby-b"),
    pytest.param(3, D - 1, id="kx3-top-poincare"),
])
def test_beta_columns_satisfy_the_eigen_relation(
        beta_model, beta_basis, kx, col):
    # the strong test is coefficient-agnostic: no analytic oracle,
    # just L q = i omega q through the real model matvec
    residual, scale = eigen_relation_residual(
        beta_model, beta_basis, kx, col)
    assert residual < 1e-11 * scale


# ================================================================
#  Chunked probe/eigensolve path (lax.map)
# ================================================================
def test_chunked_run_matches_the_vmapped(model, basis):
    chunked = channel_eigenpairs(model, chunk=5)
    assert np.allclose(chunked.omega, basis.omega, atol=1e-10)
    assert float(chunked.orthonormality_error()) < 1e-12


def test_multi_device_probe_runs_serially_with_a_host_gather(
        basis, monkeypatch):
    # a multi-device decomposition routes the probe through the
    # serial host-gather path (the batched probe cannot thread the
    # impulse axis through the sharded storage contract); faking the
    # device count on one device exercises exactly that path (on a
    # fresh model: linearizing one model over and over trips the
    # module rebind guard)
    monkeypatch.setattr(TensorDecomposition, "device_count",
                        property(lambda _self: 4))
    fresh = make_walled_model()
    serial = channel_eigenpairs(fresh)
    assert np.allclose(serial.omega, basis.omega, atol=1e-10)
    assert float(serial.orthonormality_error()) < 1e-12
    residual, scale = eigen_relation_residual(fresh, serial, kx=1,
                                              col=D - 1)
    assert residual < 1e-11 * scale


# ================================================================
#  The constrained 3-D nonhydro channel (the P L P probe)
# ================================================================
NZ_HALF = N // 2 + 1
D_NH = 4 * N - 1  # u: N, v: N - 1 (inner y faces), w: N, b: N
F0_NH, N2_NH, DSQR_NH = 1.5, 3.0, 2.0


def make_nh_channel(f0=F0_NH, device_ids=None):
    """Walled-y nonhydro channel: x, z periodic, y bounded."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=False, name="y")
    mz = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=True, name="z")
    grid = fr.spatial.Grid((mx, my, mz), device_ids=device_ids)
    return nh.Model(
        grid=grid, advection=False, dsqr=DSQR_NH,
        coriolis=nh.FPlaneCoriolis(f0=f0),
        stratification=nh.ConstantStratification(n2=N2_NH),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def nh_channel():
    """One walled-y nonhydro channel shared across the module."""
    return make_nh_channel()


@pytest.fixture(scope="module")
def nh_basis(nh_channel):
    """Compute the constrained channel eigenbasis (P L P)."""
    return channel_eigenpairs(nh_channel)


def nh_eigen_relation_residual(model, basis, kx, kz, col):
    """Drive ``P L P`` with one column; return (residual, scale).

    Builds the real physical state ``Re(q(y) e^{i(kx x + kz z)})``,
    projects it (``model.constrain``) and applies the constrained
    tendency — together exactly the probed ``P L P`` — then compares
    the rfftn ``(kx, kz)`` plane against ``i omega q``. With two
    periodic axes the half-spectrum amplitude is ``Nx * Nz / 2``
    whenever ``(kx, kz)`` is not self-conjugate (not BOTH indices in
    {0, Nyquist}); the sampled planes respect that.
    """
    q_col = np.asarray(basis.q[kx, kz, :, col])
    omega = float(basis.omega[kx, kz, col])
    px = np.exp(2j * np.pi * kx * np.arange(N) / N)
    pz = np.exp(2j * np.pi * kz * np.arange(N) / N)
    model.set_fields(**{
        name: np.real(px[:, None, None] * pz[None, None, :]
                      * q_col[basis.slices[name]][None, :, None])
        for name in basis.components})
    projected = model.constrain(model.state)
    tendency = model.tendency(projected, constraints=True)
    amp = N * N / 2
    residual = 0.0
    for name in basis.components:
        plane = np.fft.rfftn(np.asarray(tendency[name].data),
                             axes=(0, 2))[kx, :, kz]
        expect = 1j * omega * amp * q_col[basis.slices[name]]
        residual = max(residual, float(np.abs(plane - expect).max()))
    return residual, (1.0 + abs(omega)) * amp


def test_nh_layout_conventions(nh_basis):
    assert nh_basis.components == ("u", "v", "w", "b")
    assert nh_basis.slices["u"] == slice(0, N)
    assert nh_basis.slices["v"] == slice(N, 2 * N - 1)
    assert nh_basis.slices["w"] == slice(2 * N - 1, 3 * N - 1)
    assert nh_basis.slices["b"] == slice(3 * N - 1, D_NH)
    assert nh_basis.periodic_axis == "z"
    assert nh_basis.bounded_axis == "y"
    assert nh_basis.omega.shape == (N, NZ_HALF, D_NH)
    assert nh_basis.q.shape == (N, NZ_HALF, D_NH, D_NH)
    assert not np.iscomplexobj(np.asarray(nh_basis.omega))


def test_nh_pencil_is_hermitian_before_symmetrization(nh_basis):
    # the P L P sandwich is M-skew (probing P L or L P alone would
    # not be); the pre-symmetrization residual is the safety net
    assert nh_basis.hermiticity_error < 1e-13


def test_nh_eigenvectors_are_m_orthonormal(nh_basis):
    assert float(nh_basis.orthonormality_error()) < 1e-12


def test_nh_spectrum_is_plus_minus_paired(nh_basis):
    omega = np.asarray(nh_basis.omega)
    assert np.abs(omega + omega[..., ::-1]).max() < 1e-10


def test_nh_zero_mode_counts_per_plane(nh_basis):
    # P L P per (kx, kz) plane: the div-complement zeros (the discrete
    # pressure-gradient directions: rank(grad) = N, except N - 1 at
    # the kx = kz = 0 plane, where the constant pressure drops out)
    # plus the steady (geostrophic) kernel:
    #   * generic plane: N - 1 balanced strata          -> 2N - 1
    #   * kx = 0, kz != 0: steady set = the full u column (N; the
    #     boundary Kelvin branches join the steady set at kx = 0)
    #     plus the constant-p stratum (b = dp/dz only)  -> 2N + 1
    #   * kz = z-Nyquist: the w<->b interpolation factor cos(kz dz/2)
    #     vanishes, so buoyancy decouples: N steady b-columns plus
    #     one velocity stratum (the y-alternating u with its
    #     compensating w)                               -> 2N + 1
    #   * kx = x-Nyquist AND kz = z-Nyquist: the rotation (x-average)
    #     and buoyancy (z-average) factors both vanish, L = 0 on the
    #     plane and every column is a zero mode         -> D
    #   * kx = kz = 0: N - 1 complement + N mean-flow u -> 2N - 1
    omega = np.asarray(nh_basis.omega)
    zeros = np.sum(np.abs(omega) < 1e-8, axis=-1)
    expected = np.full((N, NZ_HALF), 2 * N - 1)
    expected[0, 1:] = 2 * N + 1
    expected[1:, -1] = 2 * N + 1
    expected[N // 2, -1] = D_NH
    assert (zeros == expected).all()


@pytest.fixture(scope="module")
def nh_norot_basis():
    """Compute the f0 = 0 channel eigenbasis (trig-analytic)."""
    return channel_eigenpairs(make_nh_channel(f0=0.0))


def test_nh_trig_oracle_without_rotation(nh_norot_basis):
    # with f0 = 0 there is no parity obstruction: the walled-y
    # spectrum is trig-analytic, omega^2 = N^2 a_z^2 kh^2 /
    # (dsqr kh^2 + kz^2) with the exact trig y-wavenumber
    # k_y = 2 sin(pi m dy / (2 Ly)) / dy on the cosine strata
    # m = 0..N-1, the staggered-difference periodic wavenumbers
    # k_x/k_z, and the w<->b interpolation factor a_z = cos(kz dz/2)
    # (the same discrete table nh.eigenmodes uses on the walled-z
    # grid, with the y and z roles swapped); each plane additionally
    # carries 2N - 1 exact zeros (N div-complement + N - 1 vortical)
    dx, dy, dz = 2 * np.pi / N, 1.0 / N, 2 * np.pi / N
    kx = 2 * np.pi * np.fft.fftfreq(N, dx)
    kz = np.arange(NZ_HALF, dtype=float)  # Lz = 2 pi
    m = np.arange(N, dtype=float)
    khx = 2 * np.sin(kx * dx / 2) / dx
    khy = 2 * np.sin(np.pi * m * dy / 2) / dy  # Ly = 1
    khz = 2 * np.sin(kz * dz / 2) / dz
    ahz = np.cos(kz * dz / 2)
    kh2 = khx[:, None, None] ** 2 + khy[None, None, :] ** 2
    num = N2_NH * ahz[None, :, None] ** 2 * kh2
    den = DSQR_NH * kh2 + khz[None, :, None] ** 2
    om2 = np.where(den > 0, num / np.where(den > 0, den, 1.0),
                   N2_NH / DSQR_NH)  # the kh = kz = 0 limit
    om = np.sqrt(om2)
    expected = np.sort(np.concatenate(
        [-om, om, np.zeros((N, NZ_HALF, 2 * N - 1))], axis=-1),
        axis=-1)
    got = np.sort(np.asarray(nh_norot_basis.omega), axis=-1)
    assert np.abs(got - expected).max() < 1e-12


@pytest.mark.parametrize(("kx", "kz", "col"), [
    pytest.param(1, 1, 0, id="kx1-kz1-bottom"),
    pytest.param(2, 1, D_NH - 1, id="kx2-kz1-top"),
    pytest.param(3, 2, D_NH // 2, id="kx3-kz2-zero"),
    pytest.param(1, 3, 5, id="kx1-kz3-low-branch"),
    pytest.param(0, 2, D_NH - 1, id="kx0-kz2-top"),
    pytest.param(2, 0, D_NH - 1, id="kx2-kz0-top"),
])
def test_nh_columns_satisfy_the_eigen_relation(
        nh_channel, nh_basis, kx, kz, col):
    residual, scale = nh_eigen_relation_residual(
        nh_channel, nh_basis, kx, kz, col)
    assert residual < 1e-12 * scale


def test_nh_chunked_run_matches_the_vmapped(nh_channel, nh_basis):
    chunked = channel_eigenpairs(nh_channel, chunk=7)
    assert np.allclose(chunked.omega, nh_basis.omega, atol=1e-10)
    assert float(chunked.orthonormality_error()) < 1e-12


def test_nh_multi_device_probe_runs_serially_with_a_host_gather(
        nh_basis, monkeypatch):
    # the constrained probe path shares the serial host-gather
    # fallback (see the sw twin above); faking the device count on
    # one device exercises exactly that branch
    monkeypatch.setattr(TensorDecomposition, "device_count",
                        property(lambda _self: 4))
    fresh = make_nh_channel()
    serial = channel_eigenpairs(fresh)
    assert np.allclose(serial.omega, nh_basis.omega, atol=1e-10)
    assert float(serial.orthonormality_error()) < 1e-12
    residual, scale = nh_eigen_relation_residual(
        fresh, serial, kx=1, kz=1, col=D_NH - 1)
    assert residual < 1e-12 * scale


@pytest.mark.multi_device
def test_nh_constrained_probe_is_device_count_invariant(
        forced_devices):
    # the genuinely sharded gate: the constrained probe (impulse ->
    # constrain -> constrained tendency, serial host gather) matches
    # the explicit one-device grid
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    results = {}
    for tag, device_ids in (("many", None), ("one", (0,))):
        results[tag] = channel_eigenpairs(
            make_nh_channel(device_ids=device_ids))
    assert np.allclose(np.asarray(results["many"].omega),
                       np.asarray(results["one"].omega), atol=1e-10)
    assert float(results["many"].orthonormality_error()) < 1e-12


# ================================================================
#  Varying metric coefficients: csqr(y) (sw) and N^2(y) (nh)
# ================================================================
def csqr_profile(y):
    """Return a varying, strictly positive csqr(y)."""
    return 1.0 + 0.5 * jnp.tanh(4.0 * (y - 0.5))


def n2_profile(y):
    """Return a varying, strictly positive N^2(y)."""
    return 1.0 + 2.0 * y * y


def make_varying_sw(coriolis=None):
    """Walled channel with csqr varying along the dense y axis."""
    return sw.Model(
        grid=make_grid(), csqr=csqr_profile, rossby_number=0.2,
        coriolis=coriolis, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def varying_model():
    """One varying-csqr f-plane channel shared across the module."""
    return make_varying_sw()


@pytest.fixture(scope="module")
def varying_basis(varying_model):
    """Compute the channel eigenbasis of the varying-csqr model."""
    return channel_eigenpairs(varying_model)


@pytest.fixture(scope="module")
def varying_beta_model():
    """Varying csqr combined with a beta-plane f(y)."""
    return make_varying_sw(coriolis=fr.model.modules.BetaPlaneCoriolis(
        f0=F0, beta=BETA, metric_weight="csqr"))


@pytest.fixture(scope="module")
def varying_beta_basis(varying_beta_model):
    """Eigenbasis with BOTH f and csqr varying along y."""
    return channel_eigenpairs(varying_beta_model)


def test_varying_csqr_pencil_is_hermitian(varying_basis):
    # the binding metric diag(c^2, c^2, 1) with the weights sampled
    # exactly as the tendency samples them (csqr.to(u) / csqr.to(v))
    # and the thickness-weighted rotation keep iMS Hermitian to
    # machine precision for a genuinely varying profile
    assert varying_basis.hermiticity_error < 1e-13


def test_varying_csqr_plus_beta_pencil_is_hermitian(
        varying_beta_basis):
    assert varying_beta_basis.hermiticity_error < 1e-13


def test_varying_csqr_eigenvectors_are_m_orthonormal(varying_basis):
    assert float(varying_basis.orthonormality_error()) < 1e-12


def test_varying_metric_diagonal_samples_per_component(
        varying_basis):
    # M[(c, j)] = w_c(y_j) mu_c(j): u carries csqr at the centres
    # (a pointwise broadcast), v carries csqr.to(v) at the interior
    # faces (the tendency's own flux sampling), p carries 1
    metric = np.asarray(varying_basis.metric)
    yc = (np.arange(N) + 0.5) / N
    c2_c = np.asarray(csqr_profile(yc))
    c2_f = 0.5 * (c2_c[:-1] + c2_c[1:])
    assert np.allclose(metric[:N], c2_c / N, rtol=1e-14)
    assert np.allclose(metric[N:2 * N - 1], c2_f / N, rtol=1e-14)
    assert np.allclose(metric[2 * N - 1:], 1.0 / N, rtol=1e-14)


def test_varying_constant_profile_reproduces_the_constant_path(
        basis):
    # csqr(y) = c0 through the varying path: the metric differs from
    # the constant path's diag(1, 1, 1/c^2) by the overall factor
    # c^2 only, so the spectrum agrees to machine precision
    const_var = sw.Model(
        grid=make_grid(), csqr=lambda y: CSQR + 0.0 * y,
        rossby_number=0.2, advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    cv = channel_eigenpairs(const_var)
    assert np.abs(np.asarray(cv.omega)
                  - np.asarray(basis.omega)).max() < 1e-12
    assert float(cv.orthonormality_error()) < 1e-12


def test_varying_csqr_interior_planes_carry_topographic_rossby(
        varying_basis):
    # genuine physics: a geostrophic mode over varying depth has
    # div(c^2 u_g) = c^2'(y) v_g != 0, so the interior-plane steady
    # branch acquires slow topographic Rossby frequencies (the exact
    # beta analogue); the kx = 0 and Nyquist planes keep their
    # structural zeros
    omega = np.asarray(varying_basis.omega)
    zeros = np.sum(np.abs(omega) < 1e-8, axis=-1)
    assert zeros[0] == N + 1
    assert (zeros[1:-1] == 1).all()
    assert zeros[-1] == N - 1
    # the slow band is nonzero yet separated from the gravity waves
    slow = np.abs(omega[1])[np.abs(omega[1]) < 1.0]
    slow = slow[slow > 1e-8]
    assert slow.size == N - 2
    assert slow.max() < 0.2
    fast = np.abs(omega[1])[np.abs(omega[1]) >= 1.0]
    assert fast.min() > 4.0


@pytest.mark.parametrize(("kx", "col"), [
    pytest.param(1, 0, id="kx1-bottom"),
    pytest.param(2, D // 2, id="kx2-mid"),
    pytest.param(3, D - 1, id="kx3-top"),
])
def test_varying_csqr_columns_satisfy_the_eigen_relation(
        varying_model, varying_basis, kx, col):
    # the strong test through the real model matvec validates the
    # whole varying chain: the csqr(y) field, the flux-form gravity,
    # the thickness-weighted rotation, and the sampled metric
    residual, scale = eigen_relation_residual(
        varying_model, varying_basis, kx, col)
    assert residual < 1e-11 * scale


@pytest.mark.parametrize(("kx", "col"), [
    pytest.param(1, 0, id="kx1-bottom"),
    pytest.param(2, D // 2, id="kx2-slow"),
    pytest.param(3, D - 1, id="kx3-top"),
])
def test_varying_csqr_beta_columns_satisfy_the_eigen_relation(
        varying_beta_model, varying_beta_basis, kx, col):
    residual, scale = eigen_relation_residual(
        varying_beta_model, varying_beta_basis, kx, col)
    assert residual < 1e-11 * scale


def test_varying_metric_must_be_positive():
    # a sign-crossing csqr(y) produces an indefinite metric: taught
    model = sw.Model(
        grid=make_grid(), csqr=lambda y: y - 0.5, rossby_number=0.2,
        advection=False,
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match="positive definite"):
        channel_eigenpairs(model)


@pytest.fixture(scope="module")
def nh_varying_channel():
    """Walled-y nonhydro channel with N^2 varying along y."""
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=False, name="y")
    mz = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=True, name="z")
    return nh.Model(
        grid=fr.spatial.Grid((mx, my, mz)), advection=False,
        dsqr=DSQR_NH, coriolis=nh.FPlaneCoriolis(f0=F0_NH),
        stratification=nh.MeridionalStratification(n2=n2_profile),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))


@pytest.fixture(scope="module")
def nh_varying_basis(nh_varying_channel):
    """Compute the constrained eigenbasis with varying N^2(y)."""
    return channel_eigenpairs(nh_varying_channel)


def test_nh_varying_n2_pencil_is_hermitian(nh_varying_basis):
    # the pointwise pairing: N^2 multiplies at the b nodes and the
    # 1/N^2(y) metric weight cancels it there, leaving the plain
    # measure-weighted interpolation adjointness — Hermitian for any
    # profile, through the P L P sandwich included
    assert nh_varying_basis.hermiticity_error < 1e-13


def test_nh_varying_n2_eigenvectors_are_m_orthonormal(
        nh_varying_basis):
    assert float(nh_varying_basis.orthonormality_error()) < 1e-12


def test_nh_varying_n2_keeps_the_structural_zero_counts(
        nh_varying_basis):
    # balance survives varying N^2: the steady branch has w = 0, so
    # the N^2(y) restoring never enters it — the zero counts equal
    # the constant-stratification expectation exactly (see
    # test_nh_zero_mode_counts_per_plane for the census)
    omega = np.asarray(nh_varying_basis.omega)
    zeros = np.sum(np.abs(omega) < 1e-8, axis=-1)
    expected = np.full((N, NZ_HALF), 2 * N - 1)
    expected[0, 1:] = 2 * N + 1
    expected[1:, -1] = 2 * N + 1
    expected[N // 2, -1] = D_NH
    assert (zeros == expected).all()


def test_nh_varying_constant_profile_reproduces_the_constant_path(
        nh_basis):
    # N^2(y) = n0 through the varying path is bitwise the constant
    # coupling (the profile broadcast multiplies pointwise), so the
    # spectrum agrees to machine precision
    mx = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=True, name="x")
    my = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0),
                                     periodic=False, name="y")
    mz = fr.spatial.meshes.IntervalMesh(N, (0.0, 2 * np.pi),
                                     periodic=True, name="z")
    model = nh.Model(
        grid=fr.spatial.Grid((mx, my, mz)), advection=False,
        dsqr=DSQR_NH, coriolis=nh.FPlaneCoriolis(f0=F0_NH),
        stratification=nh.MeridionalStratification(
            n2=lambda y: N2_NH + 0.0 * y),
        time_stepper=fr.model.time_steppers.AdamBashforth(5e-3, order=3))
    cv = channel_eigenpairs(model)
    assert np.abs(np.asarray(cv.omega)
                  - np.asarray(nh_basis.omega)).max() < 1e-12


@pytest.mark.parametrize(("kx", "kz", "col"), [
    pytest.param(1, 1, 0, id="kx1-kz1-bottom"),
    pytest.param(2, 1, D_NH - 1, id="kx2-kz1-top"),
    pytest.param(3, 2, D_NH // 2, id="kx3-kz2-zero"),
    pytest.param(2, 0, D_NH - 1, id="kx2-kz0-top"),
])
def test_nh_varying_columns_satisfy_the_eigen_relation(
        nh_varying_channel, nh_varying_basis, kx, kz, col):
    residual, scale = nh_eigen_relation_residual(
        nh_varying_channel, nh_varying_basis, kx, kz, col)
    assert residual < 1e-12 * scale


# ================================================================
#  Labels: empty by default, model packages fill them
# ================================================================
def test_labels_start_empty_and_label_with_fills(basis):
    assert basis.labels.shape == basis.omega.shape
    assert (np.asarray(basis.labels) == UNLABELED).all()
    basis.label_with(
        lambda b: jnp.where(jnp.abs(b.omega) < 1e-8, 0, 1))
    labels = np.asarray(basis.labels)
    assert set(labels.ravel()) == {0, 1}
    assert (labels == 0).sum(axis=-1)[0] == N + 1


def test_label_with_rejects_a_wrong_shape(basis):
    with pytest.raises(ValueError, match="one integer label per mode"):
        basis.label_with(lambda _b: jnp.zeros(3, dtype=jnp.int32))


def test_label_with_rejects_a_non_integer_labeler(basis):
    with pytest.raises(ValueError, match="small integers"):
        basis.label_with(lambda b: jnp.zeros_like(b.omega))
