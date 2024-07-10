import pytest
from copy import deepcopy
import fridom.framework as fr

@pytest.mark.parametrize("in_halo", [0, 1, 4])
@pytest.mark.parametrize("out_halo", [0, 0, 3])
@pytest.mark.parametrize("n_global", [(64, 128), (64, 64, 64)])
@pytest.mark.parametrize("shared_axes_out", [None, [0], [1]])
@pytest.mark.parametrize("shared_axes_in", [None, [0], [1]])
def test_transformer(backend, in_halo, out_halo, n_global, 
                     shared_axes_in, shared_axes_out):
    # construct domain decompositions
    domain_x = fr.domain_decomposition.DomainDecomposition(
        n_global=n_global, shared_axes=shared_axes_in, halo=in_halo)
    domain_y = fr.domain_decomposition.DomainDecomposition(
        n_global=n_global, shared_axes=shared_axes_out, halo=out_halo)
    # construct transformer
    transformer = fr.domain_decomposition.Transformer(domain_x, domain_y)
    # create a random array
    u = fr.config.ncp.random.rand(*domain_x.my_subdomain.shape)
    domain_x.sync(u)

    # test forward transformation
    v = transformer.forward(u)
    assert v.shape == domain_y.my_subdomain.shape

    # test backward transformation
    w = transformer.backward(v)
    assert fr.config.ncp.allclose(u, w)

def test_deepcopy(backend):
    domain_1 = fr.domain_decomposition.DomainDecomposition(
        n_global=(64, 64), shared_axes=[0], halo=0)
    domain_2 = fr.domain_decomposition.DomainDecomposition(
        n_global=(64, 64), shared_axes=[0], halo=0)
    transformer = fr.domain_decomposition.Transformer(domain_1, domain_2)
    transformer_copy = deepcopy(transformer)
    assert transformer != transformer_copy