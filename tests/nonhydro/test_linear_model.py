import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro as nh


@pytest.mark.parametrize("runlen", [1, 6, 24])  # in hours
def test_linear_model(runlen):
    f0 = 1e-4
    N2 = (50 * f0) ** 2
    N = tuple([16] * 3)
    L = (10_000, 10_000, 100)

    grid = nh.grid.cartesian.Grid(shape=N, domain_size=L)
    mset = nh.ModelSettings(grid, f0=f0, stratification_n2=N2)
    mset.time_stepper.dt = np.timedelta64(2, "m")
    mset.tendencies.advection.disable()
    mset.setup()

    _X, Y, Z = grid.x_mesh
    _Lx, Ly, Lz = grid.domain_size

    z = nh.State(mset)
    z.u.arr = (jnp.exp(-(Y - Ly/2)**2 / (0.2*Ly)**2)
               * jnp.exp(-(Z - Lz/2)**2 / (0.2*Lz)**2))
    z.sync()

    initial_total_energy = z.etot.integrate().value

    model = nh.Model(mset)
    model.z = z
    model.run(runlen=np.timedelta64(runlen, "h"))

    final_total_energy = model.z.etot.integrate().value

    assert jnp.abs(1 - final_total_energy / initial_total_energy) < 1e-3

@pytest.mark.parametrize("periodic_bounds",
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ])
def test_boundary_conditions(periodic_bounds):
    f0 = 1e-4
    N2 = (50 * f0) ** 2
    N = tuple([16] * 3)
    L = (10_000, 10_000, 100)

    grid = nh.grid.cartesian.Grid(shape=N, domain_size=L,
                                  periodic_bounds=periodic_bounds)
    mset = nh.ModelSettings(grid, f0=f0, stratification_n2=N2)
    mset.time_stepper.dt = np.timedelta64(20, "s")
    mset.tendencies.advection.disable()
    mset.setup()

    X, Y, Z = grid.x_mesh
    Lx, Ly, Lz = grid.domain_size

    z = nh.State(mset)
    width = 0.05
    z.b.arr = 0.1 * jnp.exp(-(Y - 3*Ly/4)**2 / (width*Ly)**2) * \
                    jnp.exp(-(Z - 1*Lz/4)**2 / (width*Lz)**2) * \
                    jnp.exp(-(X - 1*Lx/4)**2 / (width*Lx)**2)

    z.sync()

    initial_total_energy = z.etot.integrate().value

    model = nh.Model(mset)
    model.z = z
    model.run(runlen=np.timedelta64(6, "h"))

    final_total_energy = model.z.etot.integrate().value

    assert jnp.abs(1 - final_total_energy / initial_total_energy) < 1e-2
