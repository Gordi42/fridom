import pytest
import numpy as np
from copy import deepcopy
import fridom.framework as fr

@pytest.mark.parametrize("in_halo", [0, 1, 4])
@pytest.mark.parametrize("out_halo", [0, 0, 3])
@pytest.mark.parametrize("n_global", [(64, 64), (64, 68)])
def test_fft2D(backend, in_halo, out_halo, n_global):
    ncp = fr.config.ncp
    # create domain decomposition
    domain_ph = fr.domain_decomposition.DomainDecomposition(
        n_global=n_global, shared_axes=[0], halo=in_halo)
    subdom_ph = domain_ph.my_subdomain

    # create fft transformer
    pfft = fr.domain_decomposition.ParallelFFT(domain_ph, halo_out=out_halo)
    domain_sp = pfft.domain_out
    subdom_sp = domain_sp.my_subdomain

    # create random array
    u_global = fr.utils.random_array(domain_ph.n_global)

    u = ncp.zeros(domain_ph.my_subdomain.shape)
    u = fr.utils.modify_array(
        u, subdom_ph.inner_slice, u_global[subdom_ph.global_slice])
    u = domain_ph.sync(u)

    # forward fft
    v = pfft.forward(u)
    v_global = ncp.fft.fftn(u_global)

    inner = subdom_sp.inner_slice
    glob = subdom_sp.global_slice
    assert ncp.allclose(v[inner], v_global[glob])

    # backward fft
    w = pfft.backward(v).real
    assert ncp.allclose(w, u)

@pytest.mark.parametrize("in_halo", [0, 1, 4])
@pytest.mark.parametrize("out_halo", [0, 0, 3])
@pytest.mark.parametrize("n_global", [(64, 64, 64), (65, 64, 63)])
@pytest.mark.parametrize("shared_axes", [[0], [0, 1]])
def test_fft3D(backend, in_halo, out_halo, n_global, shared_axes):
    ncp = fr.config.ncp
    # create domain decomposition
    domain_ph = fr.domain_decomposition.DomainDecomposition(
        n_global=n_global, shared_axes=shared_axes, halo=in_halo)
    subdom_ph = domain_ph.my_subdomain

    # create fft transformer
    pfft = fr.domain_decomposition.ParallelFFT(domain_ph, halo_out=out_halo)
    domain_sp = pfft.domain_out
    subdom_sp = domain_sp.my_subdomain

    # create random array
    u_global = fr.utils.random_array(domain_ph.n_global)

    u = ncp.zeros(domain_ph.my_subdomain.shape)
    u = fr.utils.modify_array(
        u, subdom_ph.inner_slice, u_global[subdom_ph.global_slice])
    u = domain_ph.sync(u)

    # forward fft
    v = pfft.forward(u)
    v_global = ncp.fft.fftn(u_global)

    inner = subdom_sp.inner_slice
    glob = subdom_sp.global_slice
    assert ncp.allclose(v[inner], v_global[glob])

    # backward fft
    w = pfft.backward(v).real
    assert ncp.allclose(w, u)

def test_deepcopy(backend):
    domain = fr.domain_decomposition.DomainDecomposition(
        n_global=(64, 64), shared_axes=[0], halo=0)
    pfft = fr.domain_decomposition.ParallelFFT(domain, halo_out=0)
    pfft_copy = deepcopy(pfft)
    assert pfft != pfft_copy
