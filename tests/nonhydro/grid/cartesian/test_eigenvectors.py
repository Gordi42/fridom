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

@pytest.fixture(params=["vec_p", "vec_q"])
def vector(request):
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
    return nh.ModelSettings(grid, f0=f0, stratification_n2=n_squared, dsqr=aspect_ratio**2).setup()

@pytest.fixture(
    params=[
        ((3, 1, 4), 1, 1, 0.1),
        ((1000, 1, 20), 1e-4, 2.5e-5, 1),
        ((500e3, 1, 200), 1e-4, 2.5e-5, 1),
    ],
    ids=["Scaled model", "Unscaled small model", "Unscaled large model"],
)
def mset_2d(request):
    domain_extent, f0, n_squared, aspect_ratio = request.param
    grid = nh.grid.cartesian.Grid(N=(128, 1, 128),
                                  L=domain_extent,
                                  periodic_bounds=(True, True, True))
    return nh.ModelSettings(grid, f0=f0, stratification_n2=n_squared, dsqr=aspect_ratio**2).setup()

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

    assert nh.config.ncp.allclose(pq.arr[mask], expected)

@pytest.mark.parametrize(*(
    "f0, beta, n_squared",
    [
        pytest.param(0, 0, 0, id="all zero"),
        pytest.param(1, 1, 1, id="beta nonzero"),
    ],
))
def test_invalid_model_settings(f0, beta, n_squared, mode1, vector):
    grid = nh.grid.cartesian.Grid(N=(3, 3, 3), L=(1, 1, 1))
    nh.ModelSettings(grid, f0=f0, beta=beta, stratification_n2=n_squared).setup()

    vec_constructor = getattr(grid, vector)

    # the divergent mode should work fine
    if mode1 == "d":
        vec = vec_constructor(s=mode1)
        assert isinstance(vec, nh.State)
        return

    with pytest.raises(ValueError):  # noqa: PT011
        vec_constructor(s=mode1)

@pytest.mark.parametrize(*(
    "periodic_boundaries, should_pass",
    [
        pytest.param((False, True, True), False, id="x-boundary"),
        pytest.param((True, False, True), False, id="y-boundary"),
        pytest.param((True, True, False), True, id="z-boundary"),
        pytest.param((False, False, True), False, id="hor-boundary"),
        pytest.param((False, False, False), False, id="all-boundaries"),
    ],
))
def test_nonperiodic_boundaries(mode1, periodic_boundaries, should_pass):
    # divergent modes should always work
    if mode1 == "d":
        should_pass = True
    grid = nh.grid.cartesian.Grid(N=(15, 15, 5),
                                  L=(1, 1, 1),
                                  periodic_bounds=periodic_boundaries)
    nh.ModelSettings(grid, f0=1, stratification_n2=1).setup()

    if not should_pass:
        with pytest.raises(ValueError):  # noqa: PT011
            grid.vec_p(s=mode1)
        with pytest.raises(ValueError):  # noqa: PT011
            grid.vec_q(s=mode1)
        return

    p = grid.vec_p(s=mode1)
    q = grid.vec_q(s=mode1)

    pq = p @ q
    expected = 1  # scalar product should be 1, since both modes are the same

    # We need to mask the k = 0 mode
    kx, ky, kz = grid.K
    mask = (kx ** 2 + ky ** 2 + kz ** 2) > 0

    assert nh.config.ncp.allclose(pq.arr[mask], expected)

@pytest.mark.parametrize("grid_shape", [(3, 3, 4), (3, 4, 3), (4, 4, 3)])
def test_even_grid_size(mode1, mode2, grid_shape):
    grid = nh.grid.cartesian.Grid(N=grid_shape, L=(1, 1, 1))
    nh.ModelSettings(grid, f0=1, stratification_n2=1).setup()

    p = grid.vec_p(s=mode1)
    q = grid.vec_q(s=mode2)

    pq = p @ q

    expected = 1 if mode1 == mode2 else 0

    # We need to create a mask for the k = 0 mode
    kx, ky, kz = grid.K
    mask = (kx ** 2 + ky ** 2 + kz ** 2) > 0

    # We also need to mask the nyquist modes
    for i, ni in enumerate(grid_shape):
        if ni % 2 == 0:
            k_nyquist = grid.k_global[i][ni // 2]
            mask &= (grid.K[i] != k_nyquist)

    assert nh.config.ncp.allclose(pq.arr[mask], expected)

def test_model_run(mset_2d: nh.ModelSettings, use_discrete):
    ncp = nh.config.ncp
    grid = mset_2d.grid
    steps = 100
    # construct a wave using the eigenvector
    q = grid.vec_q(s=1, use_discrete=use_discrete)
    # we test a wave that fits twice in x and once in z
    lx, _ly, lz = grid.L
    kx = 4 * ncp.pi / lx
    kz = 2 * ncp.pi / lz
    # construct a mask to select the mode
    k_loc = ncp.isclose(grid.K[0], kx) & ncp.isclose(grid.K[2], kz)
    mask = ncp.where(k_loc, 1, 0)
    # construct the initial condition
    z_ini = (q * mask).ifft()
    # get the frequency of the mode
    om = grid.omega(k=(kx, 0, kz), use_discrete=use_discrete)
    # set the time_step to a fraction of the period
    period = 2 * ncp.pi / om.real
    dt = float(0.001 * period)
    mset_2d.time_stepper.dt = dt
    # compute the time discretization effect on that frequency
    omt = mset_2d.time_stepper.time_discretization_effect(om)
    # check that the growth rate is negative
    assert omt.imag <= 0

    # integrate the model
    model = nh.Model(mset_2d)
    model.z = z_ini
    model.run(steps=steps)
    z_res = model.z

    # compute the expected solution
    fac = ncp.exp(-1j * omt * steps * dt)
    z_exp = (q * mask * fac).ifft()

    # compute the tolerance based on whether discrete values are used or not
    tol = 0.0001 if use_discrete else 0.1

    # compare the result with the expected solution
    assert z_res.norm_of_diff(z_exp) < tol
