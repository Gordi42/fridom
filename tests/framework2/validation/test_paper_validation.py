"""
Numerical realization of the ``05_validation.md`` checks (task 1.7).

Every section-6 check the iteration-1 surface supports, exercised
through user-facing spellings:

- 6.1 Fourier x Fourier: state living permanently in coefficient
  space, no interpolation on the collocated spectral grid,
  ``init_coeff``/random spectra construction, dealiased products
  via the padded transform, and volume penalization through the
  boolean immersed mask.
- 6.2 uniform x Chebyshev: mixed-mesh grid with FD along x and the
  Chebyshev coefficient derivative along z, identity conversions,
  and the per-mesh decomposition (z stays local). The blocked
  pieces (Chebyshev geometry accessors, per-factor quadrature,
  operator eigenvalue access, Shen bases) are asserted to raise.
- Staggered-pair identities of the FD/interp defaults (summation
  by parts, self-adjoint two-point interpolation).

6.3 (sphere), 6.4 (unstructured), and 6.5 (terrain-following) are
design-for only: their meshes/mappings are stubs, so they have no
numerical realization in iteration 1.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework2 as fr


def rk4(u, dt, tendency):
    k1 = tendency(u)
    k2 = tendency(u + (dt / 2) * k1)
    k3 = tendency(u + (dt / 2) * k2)
    k4 = tendency(u + dt * k3)
    return u + (dt / 6) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ================================================================
#  6.1 Fourier x Fourier (collocated spectral Galerkin)
# ================================================================
def test_state_lives_permanently_in_coefficient_space():
    # linear advection-diffusion stepped entirely on Fourier
    # coefficients; nodal space is touched only at the ends
    n = 32
    mx = fr.grid.meshes.IntervalMesh(n, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    u0 = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x)
        + 0.5 * jnp.cos(4 * jnp.pi * x))
    t = grid.dispatch.resolve("transform", mx.center)
    u_hat0 = t.forward(u0)
    c, nu, dt, n_steps = 1.0, 5e-3, 1e-3, 100

    def tendency(u_hat):
        du = u_hat.diff("x")  # SpectralDerivative dispatch
        return -c * du + nu * du.diff("x")

    @jax.jit
    def step(u_hat):
        return rk4(u_hat, dt, tendency)

    u_hat = u_hat0
    for _ in range(n_steps):
        u_hat = step(u_hat)
    assert u_hat.function_space.bare is u_hat0.function_space.bare
    u = t.backward(u_hat)
    t_end = n_steps * dt
    x = grid.evaluation_nodes(mx.center).data
    exact = (np.exp(-nu * (2 * np.pi) ** 2 * t_end)
             * jnp.sin(2 * jnp.pi * (x - c * t_end))
             + 0.5 * np.exp(-nu * (4 * np.pi) ** 2 * t_end)
             * jnp.cos(4 * jnp.pi * (x - c * t_end)))
    # spectral in space, RK4 in time (measured ~1.3e-10)
    assert float(jnp.abs(u.data - exact).max()) < 1e-8


def test_collocated_spectral_grid_needs_no_interpolation():
    # a single un-staggered Fourier space: conversions onto the own
    # space are the identity — no interpolation, no phase shift
    mx = fr.grid.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    f = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    t = grid.dispatch.resolve("transform", mx.center)
    f_hat = t.forward(f)
    assert f_hat.to(f_hat.function_space) is f_hat
    assert f.to(f.function_space) is f


def test_spectral_native_construction():
    # init_coeff assigns coefficients verbatim (assignment, not
    # projection); grid.random draws directly on the coefficient
    # space and transforms back to a real nodal field
    mx = fr.grid.meshes.IntervalMesh(32, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    t = grid.dispatch.resolve("transform", mx.center)
    coeff_space = t.forward(grid.create_field()).function_space
    f = grid.create_field(
        coeff_space,
        init_coeff=lambda kx: jnp.where(
            jnp.abs(kx - 2 * jnp.pi) < 1e-9, 1.0, 0.0))
    k = grid.wavenumbers(coeff_space).data
    expected = jnp.where(jnp.abs(k - 2 * jnp.pi) < 1e-9, 1.0, 0.0)
    assert jnp.array_equal(f.data, expected.astype(f.dtype))
    r = grid.random.normal(coeff_space, seed=3)
    assert r.function_space.bare is coeff_space.bare
    back = t.backward(r)
    assert back.function_space.bare is mx.center
    assert not jnp.iscomplexobj(back.data)


def test_dealiased_product_via_the_padded_transform():
    # cos(2 pi k0 x)^2 has a 2 k0 = 20 component; on n = 32 the
    # plain collocation product aliases it onto k = 12, the padded
    # (3/2-rule) route removes it
    n, k0 = 32, 10
    mx = fr.grid.meshes.IntervalMesh(n, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    f = grid.create_field(
        init=lambda x: jnp.cos(2 * jnp.pi * k0 * x))
    plain = grid.dispatch.resolve("transform", mx.center)
    prod_plain = plain.forward(f * f)
    padded = fr.grid.operators.fourier.Fourier(
        grid, pad=fr.grid.operators.dealias.degree(2))
    fine = padded.backward(plain.forward(f))
    assert fine.function_space.bare.mesh.refined_from is mx
    prod_dealiased = padded.forward(fine * fine)
    alias_slot = n - 2 * k0  # rfft index of the aliased mode
    assert float(jnp.abs(prod_plain.data[alias_slot])) > 0.2
    assert float(jnp.abs(prod_dealiased.data[alias_slot])) < 1e-14
    # both keep the exact mean 1/2 of cos^2
    assert jnp.allclose(prod_dealiased.data[0], 0.5)


def test_volume_penalization_with_the_boolean_mask():
    # masked domains have no spectral route except penalization: a
    # mask-weighted relaxation forcing in the tendency (6.1); the
    # iteration-1 boolean immersed subset supplies the mask
    mx = fr.grid.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    my = fr.grid.meshes.IntervalMesh(16, (0.0, 1.0), name="y")
    grid = fr.grid.Grid((mx, my))
    grid.with_immersed(fr.grid.ImmersedDomain(
        lambda x, y: ((x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.04)
        .astype(jnp.float64)))
    center = grid.create_field().function_space.bare
    mask = grid.immersed.mask(center)
    u0 = grid.create_field(init=lambda x, y: 1.0 + 0.0 * (x + y))
    eta, dt = 5e2, 1e-3

    @jax.jit
    def step(u):
        return u + dt * (-eta) * ((1.0 - mask) * u)

    u = u0
    for _ in range(40):
        u = step(u)
    solid = np.asarray(mask.data) == 0.0
    assert solid.any()
    assert float(np.abs(np.asarray(u.data))[solid].max()) < 1e-6
    # the fluid region is untouched (tendency exactly zero there)
    assert np.array_equal(np.asarray(u.data)[~solid],
                          np.asarray(u0.data)[~solid])


# ================================================================
#  6.2 uniform x Chebyshev (mixed tensor grid)
# ================================================================
@pytest.fixture
def mixed():
    mx = fr.grid.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    mz = fr.grid.meshes.ChebyshevMesh(16, (0.0, 1.0), name="z")
    return fr.grid.Grid((mx, mz)), mx, mz


def lobatto_nodes(n, lo, hi):
    ref = -np.cos(np.pi * np.arange(n + 1) / n)
    return lo + (ref + 1.0) * (hi - lo) / 2


def test_mixed_grid_derivatives_per_factor(mixed):
    grid, mx, mz = mixed
    space = mx.center * mz.lobatto
    x = np.asarray(
        grid.evaluation_nodes(mx.center * mz.constant).data)
    z = lobatto_nodes(16, 0.0, 1.0)
    f = grid.create_field(space, data=jnp.asarray(
        np.sin(2 * np.pi * x) * (z**3)[None, :]))
    # f.diff("x") is an FD stencil on the uniform factor
    dfx = f.diff("x")
    assert dfx.function_space.bare is mx.right * mz.lobatto
    x_r = np.asarray(
        grid.evaluation_nodes(mx.right * mz.constant).data)
    exact_dfx = (2 * np.pi * np.cos(2 * np.pi * x_r)
                 * (z**3)[None, :])
    assert float(jnp.abs(dfx.data - exact_dfx).max()) < (
        (2 * np.pi) ** 3 * mx.dx**2)
    # d/dz is a Chebyshev coefficient recurrence (exact on cubics)
    cheb = fr.grid.operators.chebyshev.Chebyshev(grid, axes=("z",))
    dfz = cheb.backward(cheb.forward(f).diff("z"))
    assert dfz.function_space.bare is space
    exact_dfz = np.sin(2 * np.pi * x) * (3 * z**2)[None, :]
    assert float(jnp.abs(dfz.data - exact_dfz).max()) < 1e-12
    # "interpolate to space S" is the identity when S is right
    assert f.to(space) is f


def test_mixed_grid_decomposition_keeps_z_local(
        mixed, forced_devices):
    grid, _, _ = mixed
    layout = grid.decomposition.default_layout
    assert layout.is_local("z")
    if forced_devices is not None:
        # the uniform factor absorbs the sharding (halo strategy)
        assert not layout.is_local("x")


def test_mixed_grid_designed_for_gaps(mixed):
    grid, mx, mz = mixed
    # Chebyshev geometry accessors arrive in a later wave: no
    # init=/evaluation_nodes/measure on the Lobatto factor yet
    with pytest.raises(NotImplementedError, match="ChebyshevMesh"):
        grid.evaluation_nodes(mz.lobatto)
    with pytest.raises(NotImplementedError, match="ChebyshevMesh"):
        grid.measure(mz.lobatto)
    # per-factor quadrature (Clenshaw-Curtis weights) is therefore
    # blocked too: integrate along z raises
    f = grid.create_field(mx.center * mz.lobatto)
    with pytest.raises(NotImplementedError, match="ChebyshevMesh"):
        f.integrate("z")
    # implicit vertical solves need operator eigenvalue access,
    # designed-for until the Symbol algebra lands
    coeff = mz.chebyshev(mz.lobatto)
    diff_op = grid.dispatch.resolve("diff", coeff)
    with pytest.raises(fr.grid.operators.EigenbasisError):
        diff_op.eigenvalues(None, coeff)
    # Shen/Galerkin BC bases (n - 1 free modes) are designed-for:
    # the galerkin factory raises in iteration 1
    with pytest.raises(NotImplementedError, match="designed-for"):
        mz.galerkin(bc=fr.grid.BC.DIRICHLET)


# ================================================================
#  Staggered-pair identities (FD + interp defaults)
# ================================================================
def test_staggered_pair_summation_by_parts():
    # discrete integration by parts is exact for the staggered
    # two-point pair on the periodic mesh:
    # sum f (dg) dx + sum (df) g dx = 0 (telescoping)
    mx = fr.grid.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    f = grid.random.normal(mx.center, seed=1)
    g = grid.random.normal(mx.right, seed=2)
    i1 = (f * g.diff("x")).integrate("x").data.ravel()[0]
    i2 = (f.diff("x") * g).integrate("x").data.ravel()[0]
    assert float(jnp.abs(i1 + i2)) < 1e-13


def test_staggered_interpolation_is_self_adjoint():
    # the two-point mean is its own adjoint under the uniform
    # measure: <f, Ig> = <If, g> exactly
    mx = fr.grid.meshes.IntervalMesh(16, (0.0, 1.0), name="x")
    grid = fr.grid.Grid((mx,))
    f = grid.random.normal(mx.center, seed=1)
    g = grid.random.normal(mx.right, seed=2)
    i1 = (f * g.to(mx.center)).integrate("x").data.ravel()[0]
    i2 = (f.to(mx.right) * g).integrate("x").data.ravel()[0]
    assert float(jnp.abs(i1 - i2)) < 1e-13
