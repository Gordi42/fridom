import pytest

from fridom.framework import config
from fridom.framework.grid.cartesian import FFT
from fridom.framework.domain_decomposition import DomainDecomposition, ParallelFFT

@pytest.fixture(params=[10, 11], ids=["nx=10", "nx=11"])
def nx(request):
    return request.param

@pytest.fixture(params=[1.0, 3.1], ids=["lx=1.0", "lx=3.1"])
def lx(request):
    return request.param

@pytest.fixture(params=[1, 3, 9], ids=["kx=1", "kx=3", "kx=9"])
def kx(request):
    return request.param

@pytest.mark.mpi_skip
def test_dct1D(backend, nx, lx, kx):
    fft = FFT((False, ))
    ncp = config.ncp

    # prepare the domain
    dx = lx / nx
    k_test = kx * ncp.pi / lx
    x = ncp.linspace(0, lx, nx, endpoint=False) + dx / 2.0

    # set the physical space
    u = ncp.cos(k_test * x)

    # transform to spectral space and back
    u_hat = fft.forward(u)
    v = fft.backward(u_hat)

    # check that the result is the same
    assert ncp.allclose(u, v)

    # check that u_hat is zero for all but the kx mode
    assert u_hat[kx] != 0
    u_hat[kx] = 0
    assert ncp.allclose(u_hat, 0)

@pytest.fixture(params=[(True, True, True), 
                        (False, True, True),
                        (True, False, True),
                        (True, True, False),
                        (False, False, True),
                        (False, True, False),
                        (True, False, False),
                        (False, False, False)])
def periodic(request):
    return request.param

@pytest.mark.mpi_skip
def test_fft3D(backend, nx, periodic):
    fft = FFT(periodic)
    ncp = config.ncp

    # prepare the domain
    n = [nx] * 3

    # set the physical space
    u = ncp.random.rand(*n)

    # transform to spectral space and back
    u_hat = fft.forward(u)
    v = fft.backward(u_hat)

    # check that the result is the same
    assert ncp.allclose(u, v)

@pytest.fixture(params=[[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]])
def axes(request):
    return request.param

@pytest.mark.mpi_skip
def test_fft3D_axes(backend, nx, axes, periodic):
    fft = FFT(periodic)
    ncp = config.ncp

    # prepare the domain
    n = [nx] * 3

    # set the physical space
    u = ncp.random.rand(*n)

    # transform to spectral space and back
    u_hat = fft.forward(u, axes=axes)
    v = fft.backward(u_hat, axes=axes)

    # check that the result is the same
    assert ncp.allclose(u, v)

# -------------------------------------------------------------------
# MPI Tests
# -------------------------------------------------------------------
@pytest.fixture(params=[(32, 32), 
                        (32, 33), 
                        (33, 32), 
                        (33, 33)], ids=["32x32", "32x33", "33x32", "33x33"])
def n(request):
    return request.param

@pytest.fixture(params=[(True, True), 
                        (False, True), 
                        (True, False), 
                        (False, False)], 
                ids=["PerTT", "PerFT", "PerTF", "PerFF"])
def periodic2d(request):
    return request.param

@pytest.fixture(params=[0, 1], ids=["NoHalo", "Halo"])
def halo(request):
    return request.param

@pytest.mark.mpi(max_size=32)
def test_fft2D_mpi(backend, n, periodic2d, halo):
    domain_decomp = DomainDecomposition(n, halo, shared_axes=[0])
    fft = FFT(periodic2d)
    ncp = config.ncp
    pfft = ParallelFFT(domain_decomp)

    # set the physical space
    u = ncp.random.rand(*domain_decomp.my_subdomain.shape)
    domain_decomp.sync(u)

    # transform to spectral space and back
    u_hat = pfft.forward_apply(u, fft.forward)
    v = pfft.backward_apply(u_hat, fft.backward).real

    # check that the result is the same
    assert ncp.allclose(u, v)
