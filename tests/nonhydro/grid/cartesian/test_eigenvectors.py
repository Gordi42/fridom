"""test_eigenvectors.py - Test the eigenvectors of the nonhydro model."""

import pytest

import fridom.nonhydro as nh

# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture(params=[1, -1, 0, "d"])
def mode1(request):
    return request.param

@pytest.fixture(params=[1, -1, 0, "d"])
def mode2(request):
    return request.param

@pytest.fixture(params=[True, False])
def use_discrete(request):
    return request.param

@pytest.fixture(
    params=[
        ((15, 21, 5), (3, 1, 4), 1, 1, 0.1),
        ((15, 21, 5), (1000, 1000, 20), 1e-4, 50e-4, 1),
        ((15, 21, 5), (500e3, 500e3, 200), 1e-4, 50e-4, 1),
    ],
    ids=["Scaled model", "Unscaled small model", "Unscaled large model"],
)
def mset(request):
    grid_shape, domain_extent, f0, n_squared, aspect_ratio = request.param
    grid = nh.grid.cartesian.Grid(N=grid_shape,
                                  L=domain_extent,
                                  periodic_bounds=(True, True, True))
    return nh.ModelSettings(grid, f0=f0, N2=n_squared, dsqr=aspect_ratio**2).setup()

# ================================================================
#  Tests
# ================================================================

def test_pq_is_kronecker_product(
        mode1, mode2, use_discrete, mset: nh.ModelSettings):
    """Test that p @ q is 1 if they correspond to the same mode."""
    grid = mset.grid
    q = grid.vec_q(s=mode1, use_discrete=use_discrete)
    p = grid.vec_p(s=mode2, use_discrete=use_discrete)

    pq = p @ q
    expected = 1 if mode1 == mode2 else 0

    # We need to mask the k = 0 mode
    kx, ky, kz = grid.K
    mask = (kx ** 2 + ky ** 2 + kz ** 2) > 0

    # We also need to mask the nyquist frequency

    assert nh.config.ncp.allclose(pq.arr[mask], expected)

# TODO(Silvano): Add test with even grid sizes
# TODO(Silvano): Add test with nonperiodic boundaries
# TODO(Silvano): Add test that include a model run (This will also test the eigenvalues)

