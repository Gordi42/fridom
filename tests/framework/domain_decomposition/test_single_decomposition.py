import pytest
import fridom.framework as fr
from copy import deepcopy

# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=[0, 1, 4], ids=["halo_0", "halo_1", "halo_4"])
def halo(request):
    return request.param

@pytest.fixture(params=[(64, ), (65, ), (32, 32), (32, 33), (16, 16, 7)],
                ids=["1d_64", "1d_65", "2d_32x32", "2d_32x33", "3d_16x16x7"])
def shape(request):
    return request.param

# ================================================================
#  Test construction
# ================================================================

def test_construction(backend, halo, shape):
    fr.config.set_backend(backend)
    domain = fr.domain_decomposition.SingleDecomposition(
        shape=shape, halo=halo, shared_axes=[0])
    assert domain.shape == shape
    assert domain.halo == halo

def test_deepcopy(backend):
    domain = fr.domain_decomposition.SingleDecomposition(
        shape=(64, 64), halo=0, shared_axes=[0])
    domain_copy = deepcopy(domain)
    assert domain != domain_copy
    assert domain.shape == domain_copy.shape

# ================================================================
#  Test padding
# ================================================================
def test_padding(backend, halo, shape):
    fr.config.set_backend(backend)
    domain = fr.domain_decomposition.SingleDecomposition(
        shape=shape, halo=halo, shared_axes=[0])
    u = fr.utils.random_array(shape)
    u_padded = domain.pad(u)

    # correct shape
    padded_shape = tuple([n + 2 * halo for n in shape])
    assert u_padded.shape == padded_shape

    # correct values in the interior
    if halo == 0:
        interior = slice(None)
    else:
        interior = tuple(slice(halo, -halo) for _ in shape)
    assert u_padded[interior].shape == u.shape
    assert (u_padded[interior] == u).all()

    # correct values in the halo
    if halo == 0:
        return
    for axis in range(len(shape)):
        # left halos
        left_halo = list(slice(None) for _ in shape)
        left_halo[axis] = slice(0, halo)
        right_inside = list(slice(None) for _ in shape)
        right_inside[axis] = slice(-2*halo, -halo)
        assert (u_padded[tuple(left_halo)] == u_padded[tuple(right_inside)]).all()
        # right halos
        right_halo = list(slice(None) for _ in shape)
        right_halo[axis] = slice(-halo, None)
        left_inside = list(slice(None) for _ in shape)
        left_inside[axis] = slice(halo, 2*halo)
        assert (u_padded[tuple(right_halo)] == u_padded[tuple(left_inside)]).all()

def test_unpad(backend, halo, shape):
    fr.config.set_backend(backend)
    domain = fr.domain_decomposition.SingleDecomposition(
        shape=shape, halo=halo, shared_axes=[0])
    u = fr.utils.random_array(shape)
    u_padded = domain.pad(u)
    u_unpadded = domain.unpad(u_padded)
    assert u_unpadded.shape == u.shape
    assert (u_unpadded == u).all()

def test_nonperiodic_padding(backend, halo):
    fr.config.set_backend(backend)
    shape = (32, 32, 32)
    domain = fr.domain_decomposition.SingleDecomposition(
        shape=shape, halo=halo, shared_axes=[0], periods=(False, True, False))
    u = fr.utils.random_array(shape=shape)
    u_padded = domain.pad(u)

    # check correct shape
    padded_shape = tuple([n + 2 * halo for n in shape])
    assert u_padded.shape == padded_shape

    # check correct values in the halo regions
    if halo == 0:
        return

    # nonperiodic axes
    assert (u_padded[:halo] == 0).all()
    assert (u_padded[-halo:] == 0).all()
    assert (u_padded[:, :, :halo] == 0).all()
    assert (u_padded[:, :, -halo:] == 0).all()
    # periodic axes
    assert (u_padded[:, :halo, :] == u_padded[:, -2*halo:-halo, :]).all()
    assert (u_padded[:, -halo:, :] == u_padded[:, halo:2*halo, :]).all()

@pytest.mark.parametrize("flat_axes", [(0,), (1,), (0, 1)])
def test_flat_axis_padding(backend, halo, flat_axes):
    fr.config.set_backend(backend)
    shape = (32, 32, 32)
    domain = fr.domain_decomposition.SingleDecomposition(
        shape=shape, halo=halo, shared_axes=[0])
    # create the shape of the array with flat axes
    arr_shape = list(shape)
    for axis in flat_axes:
        arr_shape[axis] = 1
    u = fr.utils.random_array(shape=arr_shape)
    u_padded = domain.pad(u, flat_axes=flat_axes)

    # check if the padded shape is correct
    padded_shape = list(arr_shape)
    for axis in range(len(shape)):
        if axis in flat_axes:
            continue
        padded_shape[axis] += 2 * halo
    assert u_padded.shape == tuple(padded_shape)

    # check if unpadded values are correct
    u_unpadded = domain.unpad(u_padded, flat_axes=flat_axes)
    assert (u_unpadded == u).all()


# ================================================================
#  Test halo exchange
# ================================================================

def test_halo_exchange(backend, halo, shape):
    fr.config.set_backend(backend)
    domain = fr.domain_decomposition.SingleDecomposition(
        shape=shape, halo=halo, shared_axes=[0])
    u = fr.utils.random_array(shape)
    u_padded = domain.pad(u)
    ics = domain._inner_slice
    v = fr.utils.random_array(shape, seed=312)
    u_padded = fr.utils.modify_array(u_padded, ics, v)
    u_padded = domain.sync(u_padded)

    # correct values in the halo
    if halo == 0:
        return
    for axis in range(len(shape)):
        # left halos
        left_halo = list(slice(None) for _ in shape)
        left_halo[axis] = slice(0, halo)
        right_inside = list(slice(None) for _ in shape)
        right_inside[axis] = slice(-2*halo, -halo)
        assert (u_padded[tuple(left_halo)] == u_padded[tuple(right_inside)]).all()
        # right halos
        right_halo = list(slice(None) for _ in shape)
        right_halo[axis] = slice(-halo, None)
        left_inside = list(slice(None) for _ in shape)
        left_inside[axis] = slice(halo, 2*halo)
        assert (u_padded[tuple(right_halo)] == u_padded[tuple(left_inside)]).all()

# ================================================================
#  Test fft
# ================================================================
def test_fft(backend, halo, shape):
    fr.config.set_backend(backend)
    ncp = fr.config.ncp
    domain = fr.domain_decomposition.SingleDecomposition(
        shape=shape, halo=halo, shared_axes=[0])
    u = fr.utils.random_array(shape)
    u_hat = ncp.fft.fftn(u)

    v = domain.pad(u)
    forward = domain.parallel_forward_transform(ncp.fft.fftn)
    v_hat = forward(v)

    assert (v_hat == u_hat).all()

    backward = domain.parallel_backward_transform(ncp.fft.ifftn)
    w = backward(v_hat)
    w_test = domain.pad(ncp.fft.ifftn(u_hat))
    assert (w == w_test).all()


    