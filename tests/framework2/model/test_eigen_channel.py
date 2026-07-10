"""Dense-column channel eigenbasis on the walled shallow water model.

Validates the C3 engine (``fridom.framework2.model.eigen_channel``) on
the linear rotating channel: gates, Hermiticity, M-orthonormality, the
per-plane zero-mode counts, and the strong per-column eigen relation
``L Re(q e^{i kx x}) = Re(i omega q e^{i kx x})`` evaluated through the
real model matvec ``model.tendency`` — on both the f-plane channel and
the beta-plane channel (coefficients varying along the dense axis).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework2 as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.framework2.grid.decomposition.tensor import (
    TensorDecomposition,
)
from fridom.framework2.model import eigen_channel
from fridom.framework2.model.eigen_channel import (
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
    mx = fr.grid.meshes.IntervalMesh(n, (0.0, 1.0),
                                     periodic=periodic_x, name="x")
    my = fr.grid.meshes.IntervalMesh(n, (0.0, 1.0),
                                     periodic=periodic_y, name="y")
    return fr.grid.Grid((mx, my))


def make_walled_model(grid=None, *, coriolis=None):
    """Build the linear walled shallow-water channel model."""
    if coriolis is None:
        coriolis = sw.modules.FPlaneCoriolis(f0=F0)
    return sw.Model(
        grid=grid if grid is not None else make_grid(),
        csqr=CSQR, rossby_number=0.2,
        coriolis=coriolis, advection=False,
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))


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
        coriolis=fr.modules.BetaPlaneCoriolis(f0=F0, beta=BETA))


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


def test_rejects_a_constraint_stage():
    # the walled 3-D nonhydro carries the pressure CONSTRAINT stage;
    # its channel eigenbasis is a later phase (taught error)
    mx = fr.grid.meshes.IntervalMesh(4, (0.0, 1.0), periodic=True,
                                     name="x")
    my = fr.grid.meshes.IntervalMesh(4, (0.0, 1.0), periodic=True,
                                     name="y")
    mz = fr.grid.meshes.IntervalMesh(4, (0.0, 1.0), periodic=False,
                                     name="z")
    model = nh.Model(
        grid=fr.grid.Grid((mx, my, mz)), advection=False,
        coriolis=nh.FPlaneCoriolis(f0=F0),
        time_stepper=fr.time_steppers.AdamBashforth(5e-3, order=3))
    with pytest.raises(ValueError, match=r"CONSTRAINT.*later phase"):
        channel_eigenpairs(model)


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
