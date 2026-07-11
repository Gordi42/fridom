"""
Diffusion against analytic decaying modes (task 1.7).

Manual explicit stepping on the bounded mesh for both BC
structures: Dirichlet (DST route) and Neumann (DCT route, including
the DCT-I outer variant) — exercising the Neumann shape decision
end-to-end — plus a spectral variant on the periodic mesh via
Fourier + ``SpectralDerivative`` with spectral accuracy.

Iteration-1 note: ``FiniteDifference`` grounds BC-free spaces only
(BC-structured nodal stencils are designed-for), so the bounded
runs use the seeded trig-transform Laplacian; time stepping stays
manual/explicit.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.spatial.spaces.nodal import NodeSet

N = 32
LENGTH = 2.0
NU = 0.1
DT = 1e-3
N_STEPS = 100


def rk4_march(u0, tendency):
    @jax.jit
    def step(u):
        k1 = tendency(u)
        k2 = tendency(u + (DT / 2) * k1)
        k3 = tendency(u + (DT / 2) * k2)
        k4 = tendency(u + DT * k3)
        return u + (DT / 6) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    u = u0
    for _ in range(N_STEPS):
        u = step(u)
    return u


def transform_laplacian(grid, space):
    t = grid.dispatch.resolve("transform", space)

    def tendency(u):
        return NU * t.backward(t.forward(u).diff("x").diff("x"))

    return tendency


@pytest.fixture
def bounded():
    mesh = fr.spatial.meshes.IntervalMesh(
        N, (0.0, LENGTH), periodic=False, name="x")
    return fr.spatial.Grid((mesh,)), mesh


def decay(t_end, wavenumber=np.pi / LENGTH):
    return float(np.exp(-NU * wavenumber**2 * t_end))


# ================================================================
#  Dirichlet structure (Center origin -> DST-II)
# ================================================================
def test_dirichlet_decaying_mode(bounded):
    grid, mesh = bounded
    space = mesh.nodal(NodeSet.CENTER, bc=fr.spatial.BC.DIRICHLET)
    # Dirichlet drops no Center DOF: n cells -> n DOFs, n sine modes
    assert space.shape == (N,)
    u0 = grid.create_field(
        space, init=lambda x: jnp.sin(jnp.pi * x / LENGTH))
    u = rk4_march(u0, transform_laplacian(grid, space))
    exact = decay(N_STEPS * DT) * u0.data
    # the sine mode is an exact eigenfunction: machine precision
    assert float(jnp.abs(u.data - exact).max()) < 1e-12
    # the Dirichlet mode ladder starts at k = pi / L
    t = grid.dispatch.resolve("transform", space)
    k = grid.wavenumbers(t.forward(u0).function_space).data
    assert jnp.allclose(k[0], jnp.pi / LENGTH)


# ================================================================
#  Neumann structure (Center origin -> DCT-II; the shape decision)
# ================================================================
def test_neumann_decaying_mode_and_preserved_mean(bounded):
    grid, mesh = bounded
    space = mesh.nodal(NodeSet.CENTER, bc=fr.spatial.BC.NEUMANN)
    # Neumann keeps every Center DOF: n DOFs and n cosine modes
    # including the k = 0 constant (the Neumann shape decision)
    assert space.shape == (N,)
    t = grid.dispatch.resolve("transform", space)
    u0 = grid.create_field(
        space,
        init=lambda x: 1.0 + jnp.cos(jnp.pi * x / LENGTH))
    coeff = t.forward(u0)
    assert coeff.shape == (N,)
    k = grid.wavenumbers(coeff.function_space).data
    assert float(k[0]) == 0.0  # the constant mode exists
    u = rk4_march(u0, transform_laplacian(grid, space))
    exact = 1.0 + decay(N_STEPS * DT) * (u0.data - 1.0)
    assert float(jnp.abs(u.data - exact).max()) < 1e-12
    # the mean (zero mode) is preserved under Neumann diffusion.
    # iteration-1 gap: f.integrate/f.mean have no rows on
    # BC-structured nodal spaces, so the mean is read off the
    # k = 0 cosine coefficient instead
    zero_mode0 = t.forward(u0).data[0]
    zero_mode = t.forward(u).data[0]
    assert float(jnp.abs(zero_mode - zero_mode0)) < 1e-13


def test_neumann_outer_dct1_variant(bounded):
    grid, mesh = bounded
    space = mesh.nodal(NodeSet.OUTER, bc=fr.spatial.BC.NEUMANN)
    # boundary nodes stay true DOFs under Neumann: n + 1 DOFs
    assert space.shape == (N + 1,)
    u0 = grid.create_field(
        space, init=lambda x: jnp.cos(jnp.pi * x / LENGTH))
    u = rk4_march(u0, transform_laplacian(grid, space))
    exact = decay(N_STEPS * DT) * u0.data
    assert float(jnp.abs(u.data - exact).max()) < 1e-12


# ================================================================
#  Spectral variant on the periodic mesh (Fourier)
# ================================================================
def test_periodic_spectral_diffusion_is_spectrally_accurate():
    mesh = fr.spatial.meshes.IntervalMesh(N, (0.0, 1.0), name="x")
    grid = fr.spatial.Grid((mesh,))
    u0 = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x)
        + 0.5 * jnp.cos(4 * jnp.pi * x))
    nu = 5e-3
    t = grid.dispatch.resolve("transform", mesh.center)

    def tendency(u):
        # Fourier + SpectralDerivative Laplacian
        return nu * t.backward(t.forward(u).diff("x").diff("x"))

    u = rk4_march(u0, tendency)
    t_end = N_STEPS * DT
    x = grid.evaluation_nodes(mesh.center).data
    exact = (np.exp(-nu * (2 * np.pi) ** 2 * t_end)
             * jnp.sin(2 * jnp.pi * x)
             + 0.5 * np.exp(-nu * (4 * np.pi) ** 2 * t_end)
             * jnp.cos(4 * jnp.pi * x))
    # spectral in space; RK4 time error at machine level for
    # these scales (measured ~1e-13)
    assert float(jnp.abs(u.data - exact).max()) < 1e-11
