"""Tests for the pure stencil kernels (grid/operators/stencil_kernels)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.operators import stencil_kernels as sk

ORDERS = [2, 4, 6]


# ================================================================
#  Naive reference implementation (permanent oracle)
# ================================================================
def naive_staggered_diff_weights(order):
    """Independent weights oracle: float Vandermonde solve (numpy)."""
    offsets = np.arange(order) - (order - 1) / 2
    lhs = np.vander(offsets, increasing=True).T
    rhs = np.zeros(order)
    rhs[1] = 1.0
    return np.linalg.solve(lhs, rhs)


def naive_staggered_diff(arr, axis, spacing, order):
    """Loop-based staggered-difference oracle (numpy)."""
    arr = np.moveaxis(np.asarray(arr), axis, 0)
    weights = naive_staggered_diff_weights(order)
    out = np.zeros((arr.shape[0] - order + 1, *arr.shape[1:]))
    for i in range(out.shape[0]):
        for j in range(order):
            out[i] += weights[j] * arr[i + j]
    return np.moveaxis(out / spacing, 0, axis)


def naive_linear_interp(arr, axis):
    """Loop-based two-point-average oracle (numpy)."""
    arr = np.moveaxis(np.asarray(arr), axis, 0)
    out = np.zeros((arr.shape[0] - 1, *arr.shape[1:]))
    for i in range(out.shape[0]):
        out[i] = 0.5 * (arr[i] + arr[i + 1])
    return np.moveaxis(out, 0, axis)


# ================================================================
#  Coefficients
# ================================================================
def test_weights_order_2():
    assert sk.staggered_diff_weights(2) == (-1.0, 1.0)


def test_weights_order_4():
    assert sk.staggered_diff_weights(4) == (
        1 / 24, -27 / 24, 27 / 24, -1 / 24)


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_weights_match_numpy_oracle(order):
    np.testing.assert_allclose(
        sk.staggered_diff_weights(order),
        naive_staggered_diff_weights(order),
        rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_weights_zero_sum_and_antisymmetric(order):
    weights = np.array(sk.staggered_diff_weights(order))
    assert weights.sum() == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(weights, -weights[::-1], atol=1e-15)


def test_weights_are_python_floats():
    # static constants baked into the jaxpr, never traced arrays
    assert all(
        type(w) is float for w in sk.staggered_diff_weights(4))


@pytest.mark.parametrize("order", [-2, 0, 1, 3, 5])
def test_weights_invalid_order_raises(order):
    with pytest.raises(ValueError, match="even order"):
        sk.staggered_diff_weights(order)


# ================================================================
#  Polynomial exactness (order p exact up to degree p)
# ================================================================
@pytest.mark.parametrize("order", ORDERS)
def test_staggered_diff_polynomial_exactness_1d(order):
    dx = 0.5
    m = 12
    x = np.arange(m) * dx
    x_mid = (np.arange(m - order + 1) + (order - 1) / 2) * dx
    for degree in range(order + 1):
        poly = np.polynomial.Polynomial(np.arange(1.0, degree + 2))
        got = sk.staggered_diff(jnp.asarray(poly(x)), 0, dx, order)
        np.testing.assert_allclose(
            np.asarray(got), poly.deriv()(x_mid),
            rtol=1e-10, atol=1e-9,
            err_msg=f"order {order} not exact on degree {degree}")


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_staggered_diff_polynomial_exactness_3d(order, axis):
    rng = np.random.default_rng(42)
    shape = [8, 7, 6]
    dx = 0.25
    n_axis = shape[axis]
    x = np.arange(n_axis) * dx
    x_mid = (np.arange(n_axis - order + 1) + (order - 1) / 2) * dx
    poly = np.polynomial.Polynomial(np.arange(1.0, order + 2))

    coord_shape = [1, 1, 1]
    coord_shape[axis] = n_axis
    mod_shape = list(shape)
    mod_shape[axis] = 1
    modulation = rng.normal(size=mod_shape)
    field = poly(x).reshape(coord_shape) * modulation

    got = sk.staggered_diff(jnp.asarray(field), axis, dx, order)
    coord_shape[axis] = len(x_mid)
    expected = poly.deriv()(x_mid).reshape(coord_shape) * modulation
    np.testing.assert_allclose(
        np.asarray(got), expected, rtol=1e-10, atol=1e-10)


def test_linear_interp_polynomial_exactness():
    # two-point averaging is exact on polynomials up to degree 1
    dx = 0.5
    x = np.arange(10) * dx
    x_mid = (np.arange(9) + 0.5) * dx
    for degree in range(2):
        poly = np.polynomial.Polynomial(np.arange(1.0, degree + 2))
        got = sk.linear_interp(jnp.asarray(poly(x)), 0)
        np.testing.assert_allclose(
            np.asarray(got), poly(x_mid), rtol=1e-14, atol=1e-14)


# ================================================================
#  Oracle comparison on random data
# ================================================================
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_staggered_diff_matches_naive_oracle(order, axis):
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(12, 11))
    got = sk.staggered_diff(jnp.asarray(arr), axis, 0.25, order)
    expected = naive_staggered_diff(arr, axis, 0.25, order)
    np.testing.assert_allclose(
        np.asarray(got), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_linear_interp_matches_naive_oracle(axis):
    rng = np.random.default_rng(8)
    arr = rng.normal(size=(12, 11))
    got = sk.linear_interp(jnp.asarray(arr), axis)
    np.testing.assert_allclose(
        np.asarray(got), naive_linear_interp(arr, axis),
        rtol=1e-14, atol=1e-15)


# ================================================================
#  Convergence (measured order on smooth profiles)
# ================================================================
RESOLUTIONS = [16, 32, 64, 128]

PROFILES = {
    "sin": (
        lambda x: np.sin(2 * np.pi * x),
        lambda x: 2 * np.pi * np.cos(2 * np.pi * x),
    ),
    "exp": (
        lambda x: np.exp(4 * x),
        lambda x: 4 * np.exp(4 * x),
    ),
}


def _measured_slope(errors):
    """Log-log slope of the error against the resolution."""
    return np.polyfit(np.log(RESOLUTIONS), np.log(errors), 1)[0]


@pytest.mark.parametrize("profile", list(PROFILES))
@pytest.mark.parametrize("order", ORDERS)
def test_staggered_diff_convergence(order, profile):
    func, deriv = PROFILES[profile]
    errors = []
    for n in RESOLUTIONS:
        h = 1.0 / n
        x = np.arange(n + order - 1) * h
        x_mid = (np.arange(n) + (order - 1) / 2) * h
        got = sk.staggered_diff(jnp.asarray(func(x)), 0, h, order)
        errors.append(np.max(np.abs(np.asarray(got) - deriv(x_mid))))
    assert _measured_slope(errors) == pytest.approx(-order, abs=0.3)


@pytest.mark.parametrize("profile", list(PROFILES))
def test_linear_interp_convergence(profile):
    func, _ = PROFILES[profile]
    errors = []
    for n in RESOLUTIONS:
        h = 1.0 / n
        x = np.arange(n + 1) * h
        x_mid = (np.arange(n) + 0.5) * h
        got = sk.linear_interp(jnp.asarray(func(x)), 0)
        errors.append(np.max(np.abs(np.asarray(got) - func(x_mid))))
    assert _measured_slope(errors) == pytest.approx(-2, abs=0.3)


# ================================================================
#  Shape contracts (rules doc section 3.5)
# ================================================================
@pytest.mark.parametrize("order", ORDERS)
def test_staggered_diff_shape_1d(order):
    n = 10
    out = sk.staggered_diff(jnp.zeros(n), 0, 1.0, order)
    assert out.shape == (n - order + 1,)


def test_shape_contract_section_3_5():
    # order-2 diff on a bounded axis with n cells:
    n = 8
    # Outer(n + 1) -> Center(n)
    assert sk.staggered_diff(jnp.zeros(n + 1), 0, 1.0).shape == (n,)
    # Center(n) -> Inner(n - 1)
    assert sk.staggered_diff(jnp.zeros(n), 0, 1.0).shape == (n - 1,)
    # periodic Center -> Right: the caller passes the one-cell
    # wrap-filled halo window (n + 1 entries) and gets n back
    assert sk.staggered_diff(jnp.zeros(n + 1), 0, 1.0).shape == (n,)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_staggered_diff_shape_3d(axis):
    arr = jnp.zeros((9, 8, 7))
    out = sk.staggered_diff(arr, axis, 1.0, 4)
    expected = list(arr.shape)
    expected[axis] -= 3
    assert out.shape == tuple(expected)


def test_linear_interp_shape():
    assert sk.linear_interp(jnp.zeros((5, 9)), 1).shape == (5, 8)


def test_too_short_axis_raises():
    with pytest.raises(ValueError, match=r"shorter than the .*stencil"):
        sk.staggered_diff(jnp.zeros(3), 0, 1.0, order=4)


# ================================================================
#  Generic stencil application
# ================================================================
def test_apply_stencil_identity():
    rng = np.random.default_rng(9)
    arr = jnp.asarray(rng.normal(size=(6, 5)))
    out = sk.apply_stencil(arr, 0, (1.0,))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(arr))


def test_apply_stencil_weighted_sum():
    arr = jnp.asarray(np.arange(6.0))
    out = sk.apply_stencil(arr, 0, (1.0, -2.0, 1.0))
    # second difference of a linear ramp vanishes
    np.testing.assert_allclose(np.asarray(out), np.zeros(4), atol=1e-15)


# ================================================================
#  Spacing handling (scalar fold vs measure array)
# ================================================================
def test_spacing_array_matches_scalar():
    rng = np.random.default_rng(10)
    arr = jnp.asarray(rng.normal(size=12))
    scalar = sk.staggered_diff(arr, 0, 0.25, 2)
    array = sk.staggered_diff(arr, 0, jnp.full(11, 0.25), 2)
    np.testing.assert_allclose(
        np.asarray(array), np.asarray(scalar), rtol=1e-15)


def test_spacing_array_nonuniform():
    rng = np.random.default_rng(11)
    arr = rng.normal(size=9)
    spacing = rng.uniform(0.5, 1.5, size=8)
    got = sk.staggered_diff(jnp.asarray(arr), 0, jnp.asarray(spacing), 2)
    expected = (arr[1:] - arr[:-1]) / spacing
    np.testing.assert_allclose(
        np.asarray(got), expected, rtol=1e-14, atol=1e-15)


# ================================================================
#  jit-cleanliness (one trace across repeated same-shape calls)
# ================================================================
def test_staggered_diff_traces_once(compile_counter):
    kernel = jax.jit(
        sk.staggered_diff, static_argnames=("axis", "order"))
    rng = np.random.default_rng(12)
    arrs = [jnp.asarray(rng.normal(size=(16, 8))) for _ in range(3)]
    compile_counter.reset()
    results = [
        kernel(arr, axis=0, spacing=0.1, order=4) for arr in arrs]
    jax.block_until_ready(results)
    assert compile_counter.count == 1


def test_staggered_diff_traces_once_with_measure_array(compile_counter):
    kernel = jax.jit(
        sk.staggered_diff, static_argnames=("axis", "order"))
    rng = np.random.default_rng(13)
    arrs = [jnp.asarray(rng.normal(size=16)) for _ in range(3)]
    spacing = jnp.asarray(rng.uniform(0.5, 1.5, size=15))
    compile_counter.reset()
    results = [
        kernel(arr, axis=0, spacing=spacing, order=2) for arr in arrs]
    jax.block_until_ready(results)
    assert compile_counter.count == 1


def test_linear_interp_traces_once(compile_counter):
    kernel = jax.jit(sk.linear_interp, static_argnames=("axis",))
    rng = np.random.default_rng(14)
    arrs = [jnp.asarray(rng.normal(size=(16, 8))) for _ in range(3)]
    compile_counter.reset()
    results = [kernel(arr, axis=1) for arr in arrs]
    jax.block_until_ready(results)
    assert compile_counter.count == 1
