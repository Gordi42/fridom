"""Tests for the lifted banded (tridiagonal) solve primitive.

Covers the Neumann second-difference band assembly against a dense
numpy reference (including its constant-mode nullspace), the batched
apply along a chosen storage axis, the batched solve as the exact
inverse of the apply, and the per-column Thomas kernel
(``tridiagonal_solve_along_axis``) against a dense per-column solve
(batched, axis-agnostic, reverse-mode differentiable).
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.operators.banded import (
    _resolve_tridiagonal_method,
    apply_along_axis,
    second_difference_matrix,
    solve_along_axis,
    tridiagonal_apply_along_axis,
    tridiagonal_solve_along_axis,
    validate_boundary_conditions,
    validate_tridiagonal_method,
)

N = 12

#: whether the default jax backend is a GPU (gates the cusparse kernel)
_ON_GPU = jax.default_backend() == "gpu"
#: the cusparse kernel skipped off a GPU backend
_CUSPARSE = pytest.param(
    "cusparse",
    marks=pytest.mark.skipif(
        not _ON_GPU, reason="cusparse needs a CUDA GPU jax backend"))
#: every kernel exercised on the current backend
_ALL_METHODS = ["scan", "pcr", _CUSPARSE]


def _dense_tridiagonal(lower, diag, upper):
    """Dense (N, N) matrix from a single column's three bands."""
    return (np.diag(np.asarray(diag))
            + np.diag(np.asarray(upper)[:-1], 1)
            + np.diag(np.asarray(lower)[1:], -1))


def _random_bands(rng, shape):
    """Diagonally dominant symmetric-ish bands (non-singular columns)."""
    diag = jnp.asarray(rng.uniform(4.0, 6.0, shape))
    lower = jnp.asarray(rng.uniform(-1.0, 1.0, shape))
    upper = jnp.asarray(rng.uniform(-1.0, 1.0, shape))
    # Neumann ends: lower[0] and upper[-1] unused
    lower = lower.at[0].set(0.0)
    upper = upper.at[-1].set(0.0)
    return lower, diag, upper


def dense_reference(nodes):
    """Neumann second-difference / dz^2 as a plain numpy matrix."""
    nodes = np.asarray(nodes).reshape(-1)
    size = nodes.shape[0]
    dz = nodes[1] - nodes[0]
    main = np.full(size, -2.0)
    main[0] = -1.0
    main[-1] = -1.0
    return (np.diag(main) + np.diag(np.ones(size - 1), 1)
            + np.diag(np.ones(size - 1), -1)) / dz ** 2


# ================================================================
#  Band assembly
# ================================================================
def test_second_difference_matches_dense_reference():
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    band = second_difference_matrix(jnp.asarray(nodes))
    assert band.shape == (N, N)
    assert np.allclose(np.asarray(band), dense_reference(nodes))


def test_second_difference_annihilates_the_constant_mode():
    # the zero-flux (Neumann) boundary rows preserve constants
    nodes = jnp.asarray((np.arange(N) + 0.5) * (2.0 / N))
    band = second_difference_matrix(nodes)
    const = jnp.ones((N,))
    assert float(jnp.abs(band @ const).max()) < 1e-10


def test_second_difference_flattens_a_column_node_field():
    # the assembly reads the 1D line, so a column-shaped (N, 1) node
    # field flattens to the same band
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    column = jnp.asarray(nodes).reshape(N, 1)
    band = second_difference_matrix(column)
    assert band.shape == (N, N)
    assert np.allclose(np.asarray(band), dense_reference(nodes))


def test_default_bc_is_neumann_neumann():
    # the default reproduces the historical zero-flux band exactly
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    default = second_difference_matrix(jnp.asarray(nodes))
    explicit = second_difference_matrix(
        jnp.asarray(nodes), ("neumann", "neumann"))
    assert np.allclose(np.asarray(default), np.asarray(explicit))


# ================================================================
#  Per-side Dirichlet (no-slip) boundary rows
# ================================================================
def test_dirichlet_corners_are_minus_three():
    # the no-slip corner is the odd-mirror ghost u_{-1} = -u_1 across
    # the half-cell: the wall-adjacent row becomes (u_2 - 3 u_1)/dz^2
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    dz = 1.0 / N
    band = np.asarray(
        second_difference_matrix(jnp.asarray(nodes),
                                 ("dirichlet", "dirichlet")))
    assert band[0, 0] == pytest.approx(-3.0 / dz ** 2)
    assert band[-1, -1] == pytest.approx(-3.0 / dz ** 2)
    # the off-diagonal coupling is unchanged (still +1/dz^2)
    assert band[0, 1] == pytest.approx(1.0 / dz ** 2)
    assert band[-1, -2] == pytest.approx(1.0 / dz ** 2)
    # the interior rows are the plain -2 second difference
    assert band[1, 1] == pytest.approx(-2.0 / dz ** 2)


def test_mixed_bc_sets_each_corner_independently():
    # a Dirichlet bottom (-3) and Neumann top (-1): per-side selection
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    dz = 1.0 / N
    band = np.asarray(
        second_difference_matrix(jnp.asarray(nodes),
                                 ("dirichlet", "neumann")))
    assert band[0, 0] == pytest.approx(-3.0 / dz ** 2)
    assert band[-1, -1] == pytest.approx(-1.0 / dz ** 2)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_dirichlet_band_acts_as_the_discrete_sine_eigenvalue(k):
    # the Dirichlet-Dirichlet band diagonalizes the cell-centered sine
    # modes v_i = sin((i+0.5) k pi / N): band @ v = lambda_k v with the
    # exact discrete eigenvalue lambda_k = -(2 - 2 cos(k pi / N))/dz^2
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    dz = 1.0 / N
    band = second_difference_matrix(
        jnp.asarray(nodes), ("dirichlet", "dirichlet"))
    theta = k * np.pi / N
    mode = jnp.asarray(np.sin((np.arange(N) + 0.5) * theta))
    eigval = -(2.0 - 2.0 * np.cos(theta)) / dz ** 2
    assert np.allclose(np.asarray(band @ mode),
                       eigval * np.asarray(mode), atol=1e-9)


# ================================================================
#  Boundary-condition validation (the single source of truth)
# ================================================================
def test_validate_boundary_conditions_passes_valid_pairs():
    assert validate_boundary_conditions(
        ("dirichlet", "neumann")) == ("dirichlet", "neumann")


@pytest.mark.parametrize(
    "bad", [("robin", "neumann"), ("neumann",),
            ("neumann", "neumann", "neumann"), ("free", "no")])
def test_validate_boundary_conditions_rejects_bad_pairs(bad):
    with pytest.raises(ValueError, match="low, high"):
        validate_boundary_conditions(bad)


def test_second_difference_rejects_a_bad_bc():
    nodes = (np.arange(N) + 0.5) * (1.0 / N)
    with pytest.raises(ValueError, match="low, high"):
        second_difference_matrix(jnp.asarray(nodes), ("robin", "robin"))


# ================================================================
#  Batched apply / solve
# ================================================================
@pytest.mark.parametrize("axis_index", [0, 1, 2])
def test_apply_along_axis_matches_a_manual_contraction(axis_index):
    rng = np.random.default_rng(0)
    operator = jnp.asarray(rng.standard_normal((N, N)))
    data = jnp.asarray(rng.standard_normal((N, N, N)))
    out = apply_along_axis(operator, data, axis_index)
    expected = np.moveaxis(
        np.tensordot(np.asarray(operator), np.asarray(data),
                     axes=([1], [axis_index])),
        0, axis_index)
    assert out.shape == data.shape
    assert np.allclose(np.asarray(out), expected)


@pytest.mark.parametrize("axis_index", [0, 1])
def test_solve_is_the_inverse_of_apply(axis_index):
    rng = np.random.default_rng(1)
    # a well-conditioned SPD-ish system
    a = rng.standard_normal((N, N))
    system = jnp.asarray(a @ a.T + N * np.eye(N))
    x = jnp.asarray(rng.standard_normal((N, N)))
    rhs = apply_along_axis(system, x, axis_index)
    recovered = solve_along_axis(system, rhs, axis_index)
    assert np.allclose(np.asarray(recovered), np.asarray(x))


def test_solve_matches_numpy_linalg_solve():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((N, N))
    system = jnp.asarray(a @ a.T + N * np.eye(N))
    rhs = jnp.asarray(rng.standard_normal((N, 4)))
    solved = solve_along_axis(system, rhs, 0)
    expected = np.linalg.solve(np.asarray(system), np.asarray(rhs))
    assert np.allclose(np.asarray(solved), expected)


# ================================================================
#  Per-column Thomas solve (distinct tridiagonal per column)
# ================================================================
def test_tridiagonal_solve_matches_dense_per_column():
    # a distinct tridiagonal per off-axis column: the Thomas kernel
    # must reproduce a dense jnp.linalg.solve column by column
    rng = np.random.default_rng(0)
    shape = (N, 3, 4)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))
    solved = tridiagonal_solve_along_axis(lower, diag, upper, rhs, 0)
    for i in range(shape[1]):
        for j in range(shape[2]):
            dense = _dense_tridiagonal(
                lower[:, i, j], diag[:, i, j], upper[:, i, j])
            expected = np.linalg.solve(
                dense, np.asarray(rhs[:, i, j]))
            assert np.allclose(np.asarray(solved[:, i, j]), expected)


def test_tridiagonal_solve_is_the_inverse_of_the_band_apply():
    # T x = b then solve(T, b) recovers x, batched over columns; the
    # forward apply is the promoted tridiagonal_apply_along_axis
    rng = np.random.default_rng(1)
    shape = (N, 5)
    lower, diag, upper = _random_bands(rng, shape)
    x = jnp.asarray(rng.standard_normal(shape))
    b = tridiagonal_apply_along_axis(lower, diag, upper, x, 0)
    recovered = tridiagonal_solve_along_axis(lower, diag, upper, b, 0)
    assert np.allclose(np.asarray(recovered), np.asarray(x))


# ================================================================
#  Per-column tridiagonal apply (the forward band stencil)
# ================================================================
def test_tridiagonal_apply_matches_a_manual_stencil():
    # lower q_{c-1} + diag q_c + upper q_{c+1}, ends against a zero
    # neighbour (lower[0] / upper[-1] multiply a zero, hence unused)
    rng = np.random.default_rng(2)
    shape = (N, 4)
    lower, diag, upper = _random_bands(rng, shape)
    q = jnp.asarray(rng.standard_normal(shape))
    got = tridiagonal_apply_along_axis(lower, diag, upper, q, 0)
    qn = np.asarray(q)
    manual = np.asarray(diag) * qn
    manual[:-1] += np.asarray(upper)[:-1] * qn[1:]
    manual[1:] += np.asarray(lower)[1:] * qn[:-1]
    assert np.allclose(np.asarray(got), manual)


def test_tridiagonal_apply_matches_a_dense_matmul():
    # the band stencil equals a dense per-column tridiagonal matmul
    rng = np.random.default_rng(3)
    shape = (N, 6)
    lower, diag, upper = _random_bands(rng, shape)
    q = jnp.asarray(rng.standard_normal(shape))
    got = tridiagonal_apply_along_axis(lower, diag, upper, q, 0)
    for j in range(shape[1]):
        dense = _dense_tridiagonal(lower[:, j], diag[:, j], upper[:, j])
        assert np.allclose(np.asarray(got[:, j]),
                           dense @ np.asarray(q[:, j]))


@pytest.mark.parametrize("axis_index", [0, 1, 2])
def test_tridiagonal_apply_is_axis_agnostic(axis_index):
    rng = np.random.default_rng(4)
    base = (N, 3, 4)
    lower, diag, upper = _random_bands(rng, base)
    data = jnp.asarray(rng.standard_normal(base))
    ref = tridiagonal_apply_along_axis(lower, diag, upper, data, 0)
    lo = jnp.moveaxis(lower, 0, axis_index)
    di = jnp.moveaxis(diag, 0, axis_index)
    up = jnp.moveaxis(upper, 0, axis_index)
    d = jnp.moveaxis(data, 0, axis_index)
    out = tridiagonal_apply_along_axis(lo, di, up, d, axis_index)
    assert np.allclose(
        np.asarray(jnp.moveaxis(out, axis_index, 0)), np.asarray(ref))


def test_tridiagonal_apply_ignores_the_unused_ends():
    # garbage in lower[0] / upper[-1] cannot change the result (they
    # multiply the zero-filled out-of-range neighbours)
    rng = np.random.default_rng(5)
    shape = (N, 4)
    lower, diag, upper = _random_bands(rng, shape)
    data = jnp.asarray(rng.standard_normal(shape))
    clean = tridiagonal_apply_along_axis(lower, diag, upper, data, 0)
    dirty = tridiagonal_apply_along_axis(
        lower.at[0].set(1e3), diag, upper.at[-1].set(-1e3), data, 0)
    assert np.allclose(np.asarray(clean), np.asarray(dirty))


def test_tridiagonal_apply_broadcasts_the_bands():
    # a column-uniform band (shape (N, 1)) broadcasts over the batch
    rng = np.random.default_rng(6)
    lower, diag, upper = _random_bands(rng, (N, 1))
    data = jnp.asarray(rng.standard_normal((N, 5)))
    got = tridiagonal_apply_along_axis(lower, diag, upper, data, 0)
    dense = _dense_tridiagonal(lower[:, 0], diag[:, 0], upper[:, 0])
    expected = dense @ np.asarray(data)
    assert np.allclose(np.asarray(got), expected)


def test_tridiagonal_apply_is_reverse_mode_differentiable():
    # pure shifted arithmetic (no custom_vjp): grad flows and matches FD
    rng = np.random.default_rng(7)
    shape = (N, 3)
    lower, diag, upper = _random_bands(rng, shape)
    data = jnp.asarray(rng.standard_normal(shape))

    def loss(scale):
        y = tridiagonal_apply_along_axis(
            lower, diag, upper, scale * data, 0)
        return jnp.sum(y ** 2)

    grad = jax.grad(loss)(2.0)
    eps = 1e-6
    fd = (loss(2.0 + eps) - loss(2.0 - eps)) / (2 * eps)
    assert bool(jnp.isfinite(grad))
    assert abs(float(grad) - float(fd)) <= 1e-4 * abs(float(fd))


@pytest.mark.parametrize("axis_index", [0, 1, 2])
def test_tridiagonal_solve_is_axis_agnostic(axis_index):
    # solving along any storage axis matches solving along axis 0
    rng = np.random.default_rng(3)
    base = (N, 3, 4)
    lower, diag, upper = _random_bands(rng, base)
    rhs = jnp.asarray(rng.standard_normal(base))
    ref = tridiagonal_solve_along_axis(lower, diag, upper, rhs, 0)
    lo = jnp.moveaxis(lower, 0, axis_index)
    di = jnp.moveaxis(diag, 0, axis_index)
    up = jnp.moveaxis(upper, 0, axis_index)
    r = jnp.moveaxis(rhs, 0, axis_index)
    out = tridiagonal_solve_along_axis(lo, di, up, r, axis_index)
    assert np.allclose(
        np.asarray(jnp.moveaxis(out, axis_index, 0)), np.asarray(ref))


def test_tridiagonal_solve_broadcasts_the_bands():
    # a column-uniform band (shape (N, 1)) broadcasts over the batch
    rng = np.random.default_rng(4)
    lower, diag, upper = _random_bands(rng, (N, 1))
    rhs = jnp.asarray(rng.standard_normal((N, 6)))
    solved = tridiagonal_solve_along_axis(lower, diag, upper, rhs, 0)
    dense = _dense_tridiagonal(lower[:, 0], diag[:, 0], upper[:, 0])
    expected = np.linalg.solve(dense, np.asarray(rhs))
    assert np.allclose(np.asarray(solved), expected)


def test_tridiagonal_solve_is_reverse_mode_differentiable():
    # scan Thomas (no custom_vjp): jax.grad flows and matches an FD
    rng = np.random.default_rng(5)
    shape = (N, 3)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))

    def loss(scale):
        x = tridiagonal_solve_along_axis(
            lower, diag, upper, scale * rhs, 0)
        return jnp.sum(x ** 2)

    grad = jax.grad(loss)(2.0)
    eps = 1e-6
    fd = (loss(2.0 + eps) - loss(2.0 - eps)) / (2 * eps)
    assert bool(jnp.isfinite(grad))
    assert abs(float(grad) - float(fd)) <= 1e-4 * abs(float(fd))


# ================================================================
#  Kernel selection: scan / pcr / cusparse (auto dispatch)
# ================================================================
@pytest.mark.parametrize("method", _ALL_METHODS)
def test_tridiagonal_kernel_matches_dense_solve_along_axis(method):
    # a column-uniform band: every kernel must reproduce the dense
    # batched reference solve_along_axis (broadcast bands over the batch)
    rng = np.random.default_rng(10)
    lower, diag, upper = _random_bands(rng, (N, 1))
    rhs = jnp.asarray(rng.standard_normal((N, 6)))
    solved = tridiagonal_solve_along_axis(
        lower, diag, upper, rhs, 0, method=method)
    dense = jnp.asarray(
        _dense_tridiagonal(lower[:, 0], diag[:, 0], upper[:, 0]))
    expected = solve_along_axis(dense, rhs, 0)
    assert np.allclose(np.asarray(solved), np.asarray(expected))


@pytest.mark.parametrize("method", _ALL_METHODS)
@pytest.mark.parametrize("size", [7, 12, 100])
def test_tridiagonal_kernel_matches_per_column_dense(method, size):
    # per-column-VARYING bands and non-power-of-two sizes (7, 100 — the
    # PCR edge): each kernel solves each column's distinct tridiagonal
    rng = np.random.default_rng(size)
    shape = (size, 3, 4)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))
    solved = tridiagonal_solve_along_axis(
        lower, diag, upper, rhs, 0, method=method)
    for i in range(shape[1]):
        for j in range(shape[2]):
            dense = _dense_tridiagonal(
                lower[:, i, j], diag[:, i, j], upper[:, i, j])
            expected = np.linalg.solve(dense, np.asarray(rhs[:, i, j]))
            assert np.allclose(np.asarray(solved[:, i, j]), expected)


@pytest.mark.parametrize("method", _ALL_METHODS)
@pytest.mark.parametrize("axis_index", [0, 1, 2])
def test_tridiagonal_kernel_is_axis_agnostic(method, axis_index):
    # solving along any storage axis matches solving along axis 0
    rng = np.random.default_rng(30 + axis_index)
    base = (N, 3, 4)
    lower, diag, upper = _random_bands(rng, base)
    rhs = jnp.asarray(rng.standard_normal(base))
    ref = tridiagonal_solve_along_axis(
        lower, diag, upper, rhs, 0, method=method)
    lo = jnp.moveaxis(lower, 0, axis_index)
    di = jnp.moveaxis(diag, 0, axis_index)
    up = jnp.moveaxis(upper, 0, axis_index)
    r = jnp.moveaxis(rhs, 0, axis_index)
    out = tridiagonal_solve_along_axis(lo, di, up, r, axis_index,
                                       method=method)
    assert np.allclose(
        np.asarray(jnp.moveaxis(out, axis_index, 0)), np.asarray(ref))


@pytest.mark.parametrize("other", ["pcr", _CUSPARSE])
def test_tridiagonal_kernels_agree_with_scan(other):
    # cross-kernel agreement to near machine precision on a
    # well-conditioned diagonally dominant system
    rng = np.random.default_rng(40)
    shape = (N, 5)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))
    ref = tridiagonal_solve_along_axis(
        lower, diag, upper, rhs, 0, method="scan")
    got = tridiagonal_solve_along_axis(
        lower, diag, upper, rhs, 0, method=other)
    assert np.allclose(np.asarray(got), np.asarray(ref),
                       atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_tridiagonal_kernel_ignores_the_unused_ends(method):
    # contract: garbage in the unused lower[0] / upper[-1] must not
    # change the answer, for every kernel (they are zeroed internally)
    rng = np.random.default_rng(50)
    shape = (N, 4)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))
    clean = tridiagonal_solve_along_axis(
        lower, diag, upper, rhs, 0, method=method)
    dirty = tridiagonal_solve_along_axis(
        lower.at[0].set(1e3), diag, upper.at[-1].set(-1e3), rhs, 0,
        method=method)
    assert np.allclose(np.asarray(clean), np.asarray(dirty))


# ----------------------------------------------------------------
#  Host-side method dispatch (backend query monkeypatched)
# ----------------------------------------------------------------
def test_resolve_auto_picks_pcr_off_gpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert _resolve_tridiagonal_method("auto") == "pcr"


def test_resolve_auto_picks_cusparse_on_gpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert _resolve_tridiagonal_method("auto") == "cusparse"


@pytest.mark.parametrize("method", ["pcr", "scan", "cusparse"])
def test_resolve_passes_explicit_methods_on_gpu(method, monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert _resolve_tridiagonal_method(method) == method


def test_resolve_explicit_pcr_and_scan_off_gpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert _resolve_tridiagonal_method("pcr") == "pcr"
    assert _resolve_tridiagonal_method("scan") == "scan"


def test_resolve_cusparse_off_gpu_raises(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(ValueError,
                       match="cusparse tridiagonal kernel requires"):
        _resolve_tridiagonal_method("cusparse")


def test_resolve_rejects_unknown_method():
    with pytest.raises(ValueError,
                       match="tridiagonal method must be one of"):
        _resolve_tridiagonal_method("thomas")


def test_validate_tridiagonal_method_returns_and_rejects():
    assert validate_tridiagonal_method("pcr") == "pcr"
    with pytest.raises(ValueError,
                       match="tridiagonal method must be one of"):
        validate_tridiagonal_method("bogus")


def test_cusparse_kernel_runs_on_any_backend(monkeypatch):
    # force the cusparse wrapper regardless of the real backend: jax's
    # tridiagonal_solve has a CPU lowering in 0.10.2, so the helper
    # executes and stays correct even off the GPU — this covers the
    # cusparse code path under JAX_PLATFORMS=cpu, where the gpu-gated
    # parametrizations skip
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    rng = np.random.default_rng(60)
    shape = (N, 5)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))
    solved = tridiagonal_solve_along_axis(
        lower, diag, upper, rhs, 0, method="cusparse")
    for j in range(shape[1]):
        dense = _dense_tridiagonal(lower[:, j], diag[:, j], upper[:, j])
        expected = np.linalg.solve(dense, np.asarray(rhs[:, j]))
        assert np.allclose(np.asarray(solved[:, j]), expected)


# ----------------------------------------------------------------
#  Reverse-mode differentiability of the swapped kernels
# ----------------------------------------------------------------
@pytest.mark.parametrize("method", ["pcr", _CUSPARSE])
def test_tridiagonal_kernel_grad_matches_scan(method):
    # jax.grad wrt the rhs AND the diag band is finite, matches the
    # reference scan gradient, and matches a central finite difference
    rng = np.random.default_rng(70)
    shape = (8, 3)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))

    def loss(dia, right, kernel):
        x = tridiagonal_solve_along_axis(
            lower, dia, upper, right, 0, method=kernel)
        return jnp.sum(x ** 2)

    g_scan = jax.grad(lambda d, r: loss(d, r, "scan"),
                      argnums=(0, 1))(diag, rhs)
    g_new = jax.grad(lambda d, r: loss(d, r, method),
                     argnums=(0, 1))(diag, rhs)
    for scan_grad, new_grad in zip(g_scan, g_new, strict=True):
        assert bool(jnp.all(jnp.isfinite(new_grad)))
        assert np.allclose(np.asarray(new_grad), np.asarray(scan_grad),
                           atol=1e-10, rtol=1e-8)
    # central finite difference in a scalar rhs scale (policy check)
    eps = 1e-6
    scale_grad = jax.grad(
        lambda s: loss(diag, s * rhs, method))(2.0)
    fd = (loss(diag, (2.0 + eps) * rhs, method)
          - loss(diag, (2.0 - eps) * rhs, method)) / (2 * eps)
    assert abs(float(scale_grad) - float(fd)) <= 1e-4 * abs(float(fd))


# ----------------------------------------------------------------
#  HLO structure: pcr is while-free, scan is not (perf-guard gap B)
# ----------------------------------------------------------------
def _isolated_solve_hlo(method):
    """Compile the ISOLATED banded solve and return its compiled HLO."""
    rng = np.random.default_rng(80)
    shape = (N, 4)
    lower, diag, upper = _random_bands(rng, shape)
    rhs = jnp.asarray(rng.standard_normal(shape))
    return jax.jit(
        lambda lo, di, up, r: tridiagonal_solve_along_axis(
            lo, di, up, r, 0, method=method),
    ).lower(lower, diag, upper, rhs).compile().as_text()


def _has_while_opcode(hlo_text):
    """Return True if the HLO carries a ``while`` control-flow op."""
    # the opcode spelling ``while(%...)``; the ``while/body`` op_name
    # metadata and the ``%while.N_computation`` names have no ``(`` right
    # after ``while``, so this matches the instruction, not the label
    return bool(re.search(r"while\(", hlo_text))


def test_pcr_kernel_lowers_without_a_while_loop():
    # gap-B HLO guard: pcr is host-unrolled cyclic reduction
    # (ceil(log2 N) passes emitted straight-line), so the isolated
    # banded solve carries NO HLO while — the flat lowering the CG solve
    # depends on (a silent revert to scan-Thomas costs 9-18x). Scoped to
    # the banded kernel alone: the outer CG/Krylov iteration may
    # legitimately lower to while/scan and is not compiled here.
    assert not _has_while_opcode(_isolated_solve_hlo("pcr"))


def test_scan_kernel_lowers_with_a_while_loop():
    # the detector's positive control: the reference scan-Thomas kernel
    # IS a pair of lax.scan calls, which lower to HLO while — so a
    # while-absence assert genuinely distinguishes the two kernels
    assert _has_while_opcode(_isolated_solve_hlo("scan"))
