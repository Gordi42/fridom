import pytest
from mpi4py import MPI
from copy import deepcopy
import fridom.framework as fr

@pytest.fixture(params=[0, 1, 4], ids=["halo_0", "halo_1", "halo_4"])
def halo(request):
    return request.param

@pytest.mark.parametrize("n_global", [(64, 64), (64, 65), (64, 64, 63)])
def test_construction(backend, halo, n_global):
    fr.config.set_backend(backend)
    domain = fr.domain_decomposition.DomainDecomposition(
        n_global=n_global, shared_axes=[0], halo=halo)
    assert domain.n_global == n_global
    assert domain.halo == halo
    n_proc = MPI.COMM_WORLD.Get_size()
    assert len(domain.all_subdomains) == n_proc
    if n_proc != 1:
        assert domain.my_subdomain.shape != tuple(n_global)

def test_deepcopy(backend):
    domain = fr.domain_decomposition.DomainDecomposition(
        n_global=(64, 64), shared_axes=[0], halo=0)
    domain_copy = deepcopy(domain)
    assert domain != domain_copy
    assert domain.n_global == domain_copy.n_global

@pytest.fixture(params=[0, 1])
def flat_axis(request):
    return request.param

def test_sync_topo_array(backend, flat_axis):
    ncp = fr.config.ncp
    domain = fr.domain_decomposition.DomainDecomposition(
        n_global=(64, 64), shared_axes=[0], halo=2)
    shape = list(domain.my_subdomain.shape)
    shape[flat_axis] = 1
    u = fr.utils.random_array(shape)
    domain.sync(u, flat_axes=[flat_axis])

@pytest.mark.parametrize("halo", [0, 1, 2])
def test_sync_1d_array(backend, halo):
    ncp = fr.config.ncp
    n_global = (64, 1 , 32)
    domain = fr.domain_decomposition.DomainDecomposition(
        n_global=n_global, shared_axes=[0, 1], halo=halo)
    u = fr.utils.random_array(domain.my_subdomain.shape)
    assert u.shape[1] == 1 + 2 * halo
    domain.sync(u)

@pytest.mark.parametrize("halo", [1, 2, 3])
@pytest.mark.parametrize("ndims", [1, 2, 3])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", ["left", "right"])
def test_apply_boundary_conditions(backend, ndims, halo, axis, side):
    if axis >= ndims:
        return
    ncp = fr.config.ncp
    domain = fr.domain_decomposition.DomainDecomposition(
        n_global=tuple([64]*ndims), halo=halo)
    
    u = ncp.ones(shape=domain.my_subdomain.shape)
    domain.sync(u)
    assert ncp.allclose(u, 1.0)
    bc_shape = list(u.shape)
    bc_shape[axis] = 1
    u_bc = ncp.zeros(shape=bc_shape)
    u = domain.apply_boundary_condition(u, u_bc, axis, side=side)
    if side == "left":
        is_edge = domain.my_subdomain.is_left_edge[axis]
        edge = slice(0, halo)
    else:
        is_edge = domain.my_subdomain.is_right_edge[axis]
        edge = slice(-halo, None)
    if is_edge:
        edge_sl = tuple(edge if i == axis else slice(None) 
                        for i in range(ndims))
        assert ncp.allclose(u[edge_sl], 0.0)
        assert not ncp.allclose(u, 1.0)
    else:
        assert ncp.allclose(u, 1.0)

    



    