"""
Tests for the jax-sharding based domain decomposition.

Description
-----------
The tests in this file run on however many devices jax provides. In a
default test session this is a single device. To also exercise real
multi-device sharding, `test_multi_device` reruns this file in a
subprocess with ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
(the flag must be set before jax initializes, hence the subprocess).
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework as fr

FORCED_DEVICES_ENV = "FRIDOM_TEST_FORCED_DEVICES"


# ================================================================
#  Helper functions
# ================================================================
def padded_shape(domain):
    """Compute the expected shape of a padded array.

    The sharded axis (axis 0) carries one halo region per device
    shard, all other axes carry a single halo region.
    """
    shape = [n + 2 * domain.halo for n in domain.shape]
    shape[0] = domain.shape[0] + 2 * domain.halo * domain.n_devices
    return tuple(shape)


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=[0, 1, 2], ids=["halo_0", "halo_1", "halo_2"])
def halo(request):
    return request.param


@pytest.fixture(params=[(32, 32), (32, 16), (16, 16, 8)],
                ids=["2d_32x32", "2d_32x16", "3d_16x16x8"])
def shape(request):
    return request.param


@pytest.fixture
def domain(halo, shape):
    return fr.domain_decomposition.JaxDecomposition(shape=shape, halo=halo)


@pytest.fixture
def u(domain):
    return domain.create_random_array(seed=42, pad=False)


# ================================================================
#  Construction
# ================================================================
def test_construction(domain, halo, shape):
    assert domain.shape == shape
    assert domain.halo == halo


def test_one_dimensional_shape_raises():
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        fr.domain_decomposition.JaxDecomposition(shape=(32,), halo=1)


def test_too_large_halo_raises():
    with pytest.raises(ValueError, match="smaller than halo"):
        fr.domain_decomposition.JaxDecomposition(shape=(4, 4), halo=8)


# ================================================================
#  Padding
# ================================================================
def test_pad_unpad_roundtrip(domain, u):
    u_padded = domain.pad(u)

    assert u_padded.shape == padded_shape(domain)

    u_unpadded = domain.unpad(u_padded)
    assert u_unpadded.shape == u.shape
    assert (np.asarray(u_unpadded) == np.asarray(u)).all()


def test_flat_axis_padding(halo):
    shape = (32, 32)
    domain = fr.domain_decomposition.JaxDecomposition(shape=shape, halo=halo)
    u = domain.create_array(pad=False, topo=(True, False))

    flat_axes = [1]
    u_padded = domain.pad(u, flat_axes)

    # the flat axis is not padded
    assert u_padded.shape == (padded_shape(domain)[0], 1)

    u_unpadded = domain.unpad(u_padded, flat_axes)
    assert u_unpadded.shape == u.shape


def test_spectral_paddings_are_not_supported(domain, u):
    for method in (domain.pad_extend, domain.unpad_extend, domain.pad_trim):
        with pytest.raises(NotImplementedError, match="not supported"):
            method(u)


# ================================================================
#  Halo exchange
# ================================================================
def shard_blocks(domain, u_synced):
    """Split the synced array into the per-device blocks along axis 0."""
    return np.split(u_synced, domain.n_devices, axis=0)


def test_halo_exchange_periodic(domain, u, halo, shape):
    u_synced = np.asarray(domain.sync(domain.pad(u)))

    if halo == 0:
        assert (u_synced == np.asarray(u)).all()
        return

    # axis 0 is sharded: every device block carries its own halo
    # regions, which must contain the neighbor blocks' edge values
    # (this verifies the ppermute exchange at every shard boundary)
    blocks = shard_blocks(domain, u_synced)
    n_blocks = len(blocks)
    for i, block in enumerate(blocks):
        left_neighbor = blocks[(i - 1) % n_blocks]
        right_neighbor = blocks[(i + 1) % n_blocks]

        assert (block[:halo] == left_neighbor[-2 * halo:-halo]).all()
        assert (block[-halo:] == right_neighbor[halo:2 * halo]).all()

    # all other axes are not sharded: a single periodic halo wrap
    for axis in range(1, len(shape)):
        left_halo = [slice(None)] * len(shape)
        left_halo[axis] = slice(0, halo)
        right_inside = [slice(None)] * len(shape)
        right_inside[axis] = slice(-2 * halo, -halo)
        assert (u_synced[tuple(left_halo)]
                == u_synced[tuple(right_inside)]).all()

        right_halo = [slice(None)] * len(shape)
        right_halo[axis] = slice(-halo, None)
        left_inside = [slice(None)] * len(shape)
        left_inside[axis] = slice(halo, 2 * halo)
        assert (u_synced[tuple(right_halo)]
                == u_synced[tuple(left_inside)]).all()


def test_halo_exchange_nonperiodic(halo, shape):
    domain = fr.domain_decomposition.JaxDecomposition(
        shape=shape, halo=halo, periods=(False,) * len(shape))
    u = domain.create_random_array(seed=42, pad=False)
    u_synced = np.asarray(domain.sync(domain.pad(u)))

    if halo == 0:
        return

    # on the sharded axis, only the two domain-boundary halos are
    # zero; interior shard boundaries still exchange their halos
    blocks = shard_blocks(domain, u_synced)
    assert (blocks[0][:halo] == 0).all()
    assert (blocks[-1][-halo:] == 0).all()
    for i, block in enumerate(blocks[:-1]):
        right_neighbor = blocks[i + 1]

        assert (block[-halo:] == right_neighbor[halo:2 * halo]).all()
        assert (right_neighbor[:halo] == block[-2 * halo:-halo]).all()

    # all other axes have zero halo regions
    for axis in range(1, len(shape)):
        left_halo = [slice(None)] * len(shape)
        left_halo[axis] = slice(0, halo)
        right_halo = [slice(None)] * len(shape)
        right_halo[axis] = slice(-halo, None)

        assert (u_synced[tuple(left_halo)] == 0).all()
        assert (u_synced[tuple(right_halo)] == 0).all()


# ================================================================
#  Transforms
# ================================================================
def test_fft(domain, u):
    u_hat_expected = jnp.fft.fftn(np.asarray(u))

    forward = domain.parallel_forward_transform(jnp.fft.fftn)
    u_hat = forward(domain.pad(u))

    assert np.allclose(domain.gather(u_hat, spectral=True), u_hat_expected)

    backward = domain.parallel_backward_transform(jnp.fft.ifftn)
    w = backward(u_hat)

    assert np.allclose(np.asarray(domain.unpad(w)), np.asarray(u))


@pytest.mark.parametrize("axes", [
    pytest.param((0,), id="axis_0"),
    pytest.param((1,), id="axis_1"),
])
def test_fft_along_single_axis(axes):
    shape = (32, 16)
    domain = fr.domain_decomposition.JaxDecomposition(shape=shape, halo=1)
    u = domain.create_random_array(seed=42, pad=False)
    u_hat_expected = jnp.fft.fftn(np.asarray(u), axes=axes)

    forward = domain.parallel_forward_transform(jnp.fft.fftn)
    u_hat = forward(domain.pad(u), axes=axes)

    assert np.allclose(domain.gather(u_hat, spectral=True), u_hat_expected)

    backward = domain.parallel_backward_transform(jnp.fft.ifftn)
    w = backward(u_hat, axes=axes)

    assert np.allclose(np.asarray(domain.unpad(w)), np.asarray(u))


# ================================================================
#  Gather
# ================================================================
def test_gather(domain, u):
    u_synced = domain.sync(domain.pad(u))
    gathered = domain.gather(u_synced)

    assert gathered.shape == u.shape
    assert np.allclose(gathered, np.asarray(u))


def test_gather_with_slice(domain, u, shape):
    u_synced = domain.sync(domain.pad(u))
    slc = (slice(0, 4),) + (slice(None),) * (len(shape) - 1)
    gathered = domain.gather(u_synced, slc=slc)

    assert gathered.shape == (4, *shape[1:])
    assert np.allclose(gathered, np.asarray(u)[slc])


# ================================================================
#  Array creation
# ================================================================
def test_create_array(domain):
    arr = domain.create_array()

    assert arr.shape == padded_shape(domain)
    assert arr.dtype == fr.utils.dtype_real()
    assert (np.asarray(arr) == 0).all()


def test_create_array_spectral(domain, shape):
    arr = domain.create_array(spectral=True)

    # spectral arrays are complex and are not padded
    assert arr.shape == shape
    assert arr.dtype == fr.utils.dtype_comp()


def test_create_array_with_topography(halo):
    shape = (32, 32)
    domain = fr.domain_decomposition.JaxDecomposition(shape=shape, halo=halo)
    arr = domain.create_array(topo=(True, False))

    # the flat axis has size one and is not padded
    assert arr.shape == (padded_shape(domain)[0], 1)


def test_create_random_array(domain):
    arr1 = domain.create_random_array(seed=5)
    arr2 = domain.create_random_array(seed=5)
    arr3 = domain.create_random_array(seed=6)

    assert (np.asarray(arr1) == np.asarray(arr2)).all()
    assert not (np.asarray(arr1) == np.asarray(arr3)).all()


def test_create_random_array_spectral(domain, shape):
    arr = domain.create_random_array(seed=5, spectral=True)

    assert arr.shape == shape
    assert arr.dtype == fr.utils.dtype_comp()


def test_create_meshgrid(halo):
    shape = (32, 16)
    domain = fr.domain_decomposition.JaxDecomposition(shape=shape, halo=halo)
    x = jnp.arange(shape[0], dtype=fr.utils.dtype_real())
    y = jnp.arange(shape[1], dtype=fr.utils.dtype_real())

    x_mesh, y_mesh = domain.create_meshgrid(x, y)
    assert x_mesh.shape == padded_shape(domain)
    assert y_mesh.shape == padded_shape(domain)

    x_mesh, y_mesh = domain.create_meshgrid(x, y, pad=False)
    x_expected, y_expected = jnp.meshgrid(x, y, indexing="ij")
    assert np.allclose(np.asarray(x_mesh), x_expected)
    assert np.allclose(np.asarray(y_mesh), y_expected)


# ================================================================
#  Array operations
# ================================================================
def test_reductions(domain, u):
    u_np = np.asarray(u)

    assert np.isclose(float(domain.sum(u)), u_np.sum())
    assert np.isclose(float(domain.max(u)), u_np.max())
    assert np.isclose(float(domain.min(u)), u_np.min())

    partial_sum = domain.sum(u, axes=(0,))
    assert np.allclose(np.asarray(partial_sum), u_np.sum(axis=0))


# ================================================================
#  Sharding helpers
# ================================================================
def test_sharding_roundtrip(domain, u):
    u_alt = domain.to_alterative_sharding(u)
    u_main = domain.to_main_sharding(u_alt)

    assert (np.asarray(u_main) == np.asarray(u)).all()


def test_shard_map(domain, u):
    double = domain.shard_map(lambda x: 2 * x)

    assert np.allclose(np.asarray(double(u)), 2 * np.asarray(u))


# ================================================================
#  Multi-device runs
# ================================================================
def test_forced_device_count():
    """Inside a multi-device subprocess, check the forced device count."""
    forced_devices = os.environ.get(FORCED_DEVICES_ENV)
    if forced_devices is None:
        pytest.skip("only relevant in a multi-device subprocess")

    # guards against XLA_FLAGS being set too late to take effect
    assert jax.device_count() == int(forced_devices)


@pytest.mark.parametrize("n_devices", [4])
def test_multi_device(n_devices):
    """Rerun this test file on multiple (forced host) devices."""
    if os.environ.get(FORCED_DEVICES_ENV) is not None:
        pytest.skip("already running in a multi-device subprocess")

    env = os.environ.copy()
    env[FORCED_DEVICES_ENV] = str(n_devices)
    xla_flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = (
        f"{xla_flags} --xla_force_host_platform_device_count={n_devices}")

    result = subprocess.run(  # noqa: S603 (runs this very test file)
        [sys.executable, "-m", "pytest", __file__, "-q",
         "-p", "no:cacheprovider"],
        env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
