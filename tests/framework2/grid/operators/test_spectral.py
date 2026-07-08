"""Coefficient-space operator tests (derivative, shifts)."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.chebyshev import ChebyshevMesh
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import EigenbasisError
from fridom.framework2.grid.operators.chebyshev import Chebyshev
from fridom.framework2.grid.operators.fourier import Fourier
from fridom.framework2.grid.operators.spectral import (
    PhaseShift,
    SincShift,
    SpectralDerivative,
    fourier_wavenumbers,
)
from fridom.framework2.grid.operators.symbol import Symbol
from fridom.framework2.grid.operators.trig import Cosine, Sine
from fridom.framework2.grid.spaces.nodal import NodeSet

TWO_PI = 2.0 * jnp.pi
N = 16


@pytest.fixture
def periodic():
    mesh = IntervalMesh(N, (0.0, 1.0), name="x")
    return Grid((mesh,)), mesh


@pytest.fixture
def bounded():
    mesh = IntervalMesh(8, (0.0, 1.0), periodic=False, name="x")
    return Grid((mesh,)), mesh


# ================================================================
#  Wavenumbers
# ================================================================
def test_fourier_wavenumbers_half_layout(periodic):
    _, mesh = periodic
    k = fourier_wavenumbers(mesh.fourier(origin=mesh.center))
    assert k.shape == (N // 2 + 1,)
    assert jnp.allclose(k, TWO_PI * jnp.arange(N // 2 + 1))


def test_fourier_wavenumbers_full_layout(periodic):
    _, mesh = periodic
    space = mesh.fourier(origin=mesh.center.as_complex())
    k = fourier_wavenumbers(space)
    assert k.shape == (N,)
    assert k[1] == TWO_PI
    assert k[-1] == -TWO_PI


# ================================================================
#  SpectralDerivative codomain signatures
# ================================================================
def test_fourier_derivative_preserves_the_space(periodic):
    _, mesh = periodic
    space = mesh.fourier(origin=mesh.center)
    assert SpectralDerivative().codomain(space) is space


def test_sine_cosine_bc_flip_signatures(bounded):
    _, mesh = bounded
    dirichlet = mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    neumann = mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    d = SpectralDerivative()
    assert d.codomain(mesh.sine(dirichlet)) is mesh.cosine(neumann)
    assert d.codomain(mesh.cosine(neumann)) is mesh.sine(dirichlet)


def test_chebyshev_derivative_preserves_the_space():
    mesh = ChebyshevMesh(8, (-1.0, 1.0), name="z")
    space = mesh.chebyshev(mesh.lobatto)
    assert SpectralDerivative().codomain(space) is space


def test_nodal_domains_raise(periodic):
    _, mesh = periodic
    with pytest.raises(SpaceMismatchError, match="FiniteDifference"):
        SpectralDerivative().codomain(mesh.center)


def test_i_type_pair_signatures(bounded):
    _, mesh = bounded
    inner = mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    outer = mesh.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    d = SpectralDerivative()
    assert d.codomain(mesh.sine(inner)) is mesh.cosine(outer)
    assert d.codomain(mesh.cosine(outer)) is mesh.sine(inner)


def test_requirements_and_dispatch_kind(periodic):
    _, mesh = periodic
    d = SpectralDerivative()
    req = d.requirements(mesh.fourier(origin=mesh.center))
    assert req.halo == 0
    assert req.layout == "local"
    assert d.dispatch_kind == "diff"
    assert PhaseShift().dispatch_kind == "interpolate"
    assert SincShift().dispatch_kind == "interpolate"


def test_spectral_derivative_eigenvalues_is_the_ik_diagonal(periodic):
    grid, mesh = periodic
    space = mesh.fourier(origin=mesh.center)
    sym = SpectralDerivative().eigenvalues(grid, space)
    assert sym.space is space
    assert sym.codomain is space  # spectral diff does not stagger
    k = fourier_wavenumbers(space)
    # exact i k, the even-n Nyquist mode annihilated (like _apply)
    assert jnp.allclose(sym.data[:-1], 1j * k[:-1])
    assert sym.data[-1] == 0


def test_spectral_derivative_eigenvalues_raise_off_fourier(bounded):
    grid, mesh = bounded
    sine = mesh.sine(mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))
    # the sine/cosine derivative is an index-shifted diagonal the
    # Hadamard Symbol cannot carry (deferred to the block layer)
    with pytest.raises(EigenbasisError, match="index-shifted"):
        SpectralDerivative().eigenvalues(grid, sine)


def test_spectral_derivative_eigenvalues_raise_on_chebyshev():
    mesh = ChebyshevMesh(8, (-1.0, 1.0), name="z")
    grid = Grid((mesh,))
    with pytest.raises(EigenbasisError):
        SpectralDerivative().eigenvalues(
            grid, mesh.chebyshev(mesh.lobatto))


# ================================================================
#  SpectralDerivative exactness
# ================================================================
def test_fourier_derivative_is_spectrally_exact(periodic):
    grid, mesh = periodic
    f = grid.create_field(
        init=lambda x: jnp.sin(TWO_PI * x)
        + 0.25 * jnp.cos(3 * TWO_PI * x))
    ft = Fourier(grid)
    deriv = ft.backward(SpectralDerivative()(ft.forward(f)))
    x = grid.evaluation_nodes(mesh.center).data
    exact = (TWO_PI * jnp.cos(TWO_PI * x)
             - 0.75 * TWO_PI * jnp.sin(3 * TWO_PI * x))
    assert jnp.max(jnp.abs(deriv.data - exact)) < 1e-12


def test_fourier_derivative_annihilates_the_nyquist_mode(periodic):
    grid, _ = periodic
    # sin at the Nyquist mode samples to (-1)^j at centers: the
    # coefficient is exactly 1 and the derivative annihilates it
    f = grid.create_field(
        init=lambda x: jnp.sin(TWO_PI * (N // 2) * x))
    ft = Fourier(grid)
    coeff = ft.forward(f)
    assert jnp.abs(coeff.data[-1] - 1.0) < 1e-14
    deriv = SpectralDerivative()(coeff)
    assert jnp.max(jnp.abs(deriv.data)) < 1e-14


def test_fourier_derivative_metadata_resets(periodic):
    grid, _ = periodic
    f = grid.create_field(name="u",
                          init=lambda x: jnp.sin(TWO_PI * x))
    ft = Fourier(grid)
    deriv = SpectralDerivative()(ft.forward(f))
    assert deriv.name != "u"


def test_trig_derivative_is_spectrally_exact(bounded):
    grid, mesh = bounded
    dirichlet = mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    x = grid.evaluation_nodes(dirichlet).data
    f = grid.create_field(
        dirichlet,
        data=jnp.sin(2 * jnp.pi * x) + 3 * jnp.sin(jnp.pi * x))
    coeff = Sine(grid).forward(f)
    deriv = Cosine(grid).backward(SpectralDerivative()(coeff))
    exact = (2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
             + 3 * jnp.pi * jnp.cos(jnp.pi * x))
    assert jnp.max(jnp.abs(deriv.data - exact)) < 1e-12


def test_sine_top_mode_is_annihilated(bounded):
    grid, mesh = bounded
    dirichlet = mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    n = dirichlet.shape[0]
    top = jnp.zeros(n).at[n - 1].set(1.0)
    coeff = grid.create_field(mesh.sine(dirichlet), data=top)
    deriv = SpectralDerivative()(coeff)
    assert jnp.max(jnp.abs(deriv.data)) < 1e-14


def test_cosine_constant_is_annihilated(bounded):
    grid, mesh = bounded
    neumann = mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    n = neumann.shape[0]
    constant = jnp.zeros(n).at[0].set(1.0)
    coeff = grid.create_field(mesh.cosine(neumann), data=constant)
    deriv = SpectralDerivative()(coeff)
    assert jnp.max(jnp.abs(deriv.data)) < 1e-14
    # and the top sine slot is never populated
    assert deriv.data.shape == (n,)
    assert deriv.data[n - 1] == 0.0


def test_cosine_derivative_index_map(bounded):
    grid, mesh = bounded
    neumann = mesh.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    x = grid.evaluation_nodes(neumann).data
    f = grid.create_field(neumann, data=jnp.cos(3 * jnp.pi * x))
    coeff = Cosine(grid).forward(f)
    deriv = Sine(grid).backward(SpectralDerivative()(coeff))
    exact = -3 * jnp.pi * jnp.sin(3 * jnp.pi * x)
    assert jnp.max(jnp.abs(deriv.data - exact)) < 1e-12


def test_i_type_sine_derivative_is_spectrally_exact(bounded):
    grid, mesh = bounded
    n = mesh.n_cells
    inner = mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    x = grid.evaluation_nodes(inner).data
    f = grid.create_field(
        inner,
        data=jnp.sin(2 * jnp.pi * x) + 3 * jnp.sin(jnp.pi * x))
    coeff = Sine(grid).forward(f)
    deriv = SpectralDerivative()(coeff)
    # lands on the DCT-I family: n + 1 modes, k = 0 and k = n empty
    assert deriv.function_space.bare is mesh.cosine(
        mesh.nodal(NodeSet.OUTER, bc=BC.NEUMANN))
    assert deriv.data.shape == (n + 1,)
    assert deriv.data[0] == 0.0
    assert deriv.data[n] == 0.0
    back = Cosine(grid).backward(deriv)
    x_outer = jnp.linspace(0.0, 1.0, n + 1)
    exact = (2 * jnp.pi * jnp.cos(2 * jnp.pi * x_outer)
             + 3 * jnp.pi * jnp.cos(jnp.pi * x_outer))
    assert jnp.max(jnp.abs(back.data - exact)) < 1e-12


def test_i_type_cosine_derivative_is_spectrally_exact(bounded):
    grid, mesh = bounded
    n = mesh.n_cells
    outer = mesh.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    x_outer = jnp.linspace(0.0, 1.0, n + 1)
    f = grid.create_field(
        outer,
        data=0.5 + jnp.cos(3 * jnp.pi * x_outer))
    coeff = Cosine(grid).forward(f)
    deriv = SpectralDerivative()(coeff)
    assert deriv.function_space.bare is mesh.sine(
        mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    assert deriv.data.shape == (n - 1,)
    back = Sine(grid).backward(deriv)
    inner = mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    x = grid.evaluation_nodes(inner).data
    exact = -3 * jnp.pi * jnp.sin(3 * jnp.pi * x)
    assert jnp.max(jnp.abs(back.data - exact)) < 1e-12


def test_i_type_cosine_annihilations(bounded):
    grid, mesh = bounded
    n = mesh.n_cells
    outer = mesh.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    # the constant k = 0 and the Nyquist cosine k = n both map to
    # zero (the Nyquist sine image vanishes at the interior faces)
    ends = jnp.zeros(n + 1).at[0].set(1.0).at[n].set(2.0)
    coeff = grid.create_field(mesh.cosine(outer), data=ends)
    deriv = SpectralDerivative()(coeff)
    assert jnp.max(jnp.abs(deriv.data)) < 1e-14


def test_chebyshev_derivative_is_exact_with_extent_scaling():
    # physical interval (0, 2): d/dx = (2/L) d/dxi = d/dxi
    for extent, scale in (((-1.0, 1.0), 1.0), ((0.0, 4.0), 0.5)):
        mesh = ChebyshevMesh(8, extent, name="z")
        grid = Grid((mesh,))
        xi = -jnp.cos(jnp.pi * jnp.arange(9) / 8)
        f = grid.create_field(mesh.lobatto,
                              data=4 * xi ** 3 - 3 * xi)  # T_3
        ch = Chebyshev(grid)
        deriv = ch.backward(SpectralDerivative()(ch.forward(f)))
        exact = scale * (12 * xi ** 2 - 3)
        assert jnp.max(jnp.abs(deriv.data - exact)) < 1e-12


def test_derivative_binds_like_any_separable_kernel():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    my = IntervalMesh(8, (0.0, 1.0), name="y")
    grid = Grid((mx, my))
    f = grid.create_field(
        init=lambda x, y: jnp.sin(TWO_PI * x) * jnp.cos(TWO_PI * y))
    ft = Fourier(grid)
    coeff = ft.forward(f)
    deriv = ft.backward(SpectralDerivative()["y"](coeff))
    xs = grid.evaluation_nodes(f.function_space, name="x").data
    ys = grid.evaluation_nodes(f.function_space, name="y").data
    exact = -TWO_PI * jnp.sin(TWO_PI * xs) * jnp.sin(TWO_PI * ys)
    assert jnp.max(jnp.abs(deriv.data - exact)) < 1e-12


# ================================================================
#  PhaseShift
# ================================================================
def test_phase_shift_center_to_right(periodic):
    grid, mesh = periodic
    ft = Fourier(grid)
    below_nyquist = lambda x: (jnp.sin(TWO_PI * x)  # noqa: E731
                               + jnp.cos(5 * TWO_PI * x))
    center_hat = ft.forward(grid.create_field(init=below_nyquist))
    right_hat = ft.forward(
        grid.create_field(mesh.right, init=below_nyquist))
    shifted = PhaseShift(to=NodeSet.RIGHT)(center_hat)
    assert shifted.function_space is right_hat.function_space
    assert jnp.allclose(shifted.data, right_hat.data, atol=1e-13)


def test_phase_shift_zeroes_the_even_n_nyquist_mode(periodic):
    grid, _ = periodic
    ft = Fourier(grid)
    nyquist = lambda x: jnp.cos(TWO_PI * (N // 2) * x)  # noqa: E731
    center_hat = ft.forward(grid.create_field(init=nyquist))
    shifted = PhaseShift(to=NodeSet.RIGHT)(center_hat)
    # the one non-exact DOF (section 3.2 caveat)
    assert jnp.max(jnp.abs(shifted.data)) < 1e-14


def test_phase_shift_integer_offsets_keep_the_nyquist_mode():
    mesh = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    ft = Fourier(grid)
    f = grid.create_field(
        mesh.left, init=lambda x: jnp.cos(TWO_PI * 4 * x))
    shifted = PhaseShift(to=NodeSet.RIGHT)(ft.forward(f))
    direct = ft.forward(grid.create_field(
        mesh.right, init=lambda x: jnp.cos(TWO_PI * 4 * x)))
    assert jnp.allclose(shifted.data, direct.data, atol=1e-13)


def test_phase_shift_is_identity_on_matching_origins(periodic):
    grid, _ = periodic
    ft = Fourier(grid)
    coeff = ft.forward(
        grid.create_field(init=lambda x: jnp.sin(TWO_PI * x)))
    shifted = PhaseShift(to=NodeSet.CENTER)(coeff)
    assert shifted.function_space is coeff.function_space
    assert jnp.array_equal(shifted.data, coeff.data)


def test_phase_shift_target_validation():
    with pytest.raises(ValueError, match="CENTER/LEFT/RIGHT"):
        PhaseShift(to=NodeSet.OUTER)
    with pytest.raises(ValueError, match="CENTER/LEFT/RIGHT"):
        PhaseShift(to="center")


def test_phase_shift_rejects_average_origins(periodic):
    _, mesh = periodic
    with pytest.raises(SpaceMismatchError, match="SincShift"):
        PhaseShift().codomain(mesh.fourier(origin=mesh.cell_avg))


def test_phase_shift_rejects_nodal_domains(periodic):
    _, mesh = periodic
    with pytest.raises(SpaceMismatchError, match="Fourier"):
        PhaseShift().codomain(mesh.center)


# ================================================================
#  SincShift
# ================================================================
def test_sinc_shift_cell_avg_to_center(periodic):
    grid, mesh = periodic
    dx = mesh.dx
    ft = Fourier(grid)
    # exact cell averages of sin(2 pi x): sinc(pi dx) factor
    avg = grid.create_field(
        mesh.cell_avg,
        init=lambda x: jnp.sin(TWO_PI * x) * jnp.sinc(dx))
    center_hat = ft.forward(
        grid.create_field(init=lambda x: jnp.sin(TWO_PI * x)))
    converted = SincShift(to=NodeSet.CENTER)(ft.forward(avg))
    assert converted.function_space is center_hat.function_space
    assert jnp.allclose(converted.data, center_hat.data, atol=1e-13)


def test_sinc_shift_face_avg_composes_the_phase(periodic):
    grid, mesh = periodic
    dx = mesh.dx
    ft = Fourier(grid)
    # face averages of sin around right faces: same sinc factor,
    # sampled at the right nodes
    avg = grid.create_field(
        mesh.face_avg,
        init=lambda x: jnp.sin(TWO_PI * x) * jnp.sinc(dx))
    center_hat = ft.forward(
        grid.create_field(init=lambda x: jnp.sin(TWO_PI * x)))
    converted = SincShift(to=NodeSet.CENTER)(ft.forward(avg))
    assert converted.function_space is center_hat.function_space
    assert jnp.allclose(converted.data, center_hat.data, atol=1e-13)


def test_sinc_shift_rejects_nodal_origins(periodic):
    _, mesh = periodic
    with pytest.raises(SpaceMismatchError, match="PhaseShift"):
        SincShift().codomain(mesh.fourier(origin=mesh.center))


def test_sinc_shift_target_validation():
    with pytest.raises(ValueError, match="CENTER/LEFT/RIGHT"):
        SincShift(to=NodeSet.INNER)


def test_odd_n_fourier_derivative_has_no_nyquist_slot():
    mesh = IntervalMesh(9, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    f = grid.create_field(init=lambda x: jnp.sin(TWO_PI * 4 * x))
    ft = Fourier(grid)
    deriv = ft.backward(SpectralDerivative()(ft.forward(f)))
    x = grid.evaluation_nodes(mesh.center).data
    exact = TWO_PI * 4 * jnp.cos(TWO_PI * 4 * x)
    assert jnp.max(jnp.abs(deriv.data - exact)) < 1e-12


def test_phase_shift_is_exact_on_complex_origins(periodic):
    grid, mesh = periodic
    space = mesh.center.as_complex()
    x = grid.evaluation_nodes(mesh.center).data
    xr = grid.evaluation_nodes(mesh.right).data
    wave = lambda x: (jnp.exp(1j * TWO_PI * x)  # noqa: E731
                      + jnp.exp(-1j * TWO_PI * (N // 2) * x))
    ft = Fourier(grid)
    center_hat = ft.forward(grid.create_field(space, data=wave(x)))
    right_hat = ft.forward(grid.create_field(
        mesh.right.as_complex(), data=wave(xr)))
    shifted = PhaseShift(to=NodeSet.RIGHT)(center_hat)
    assert shifted.function_space is right_hat.function_space
    # exact and unitary on the full spectrum, Nyquist included
    assert jnp.allclose(shifted.data, right_hat.data, atol=1e-13)


def test_shift_target_properties():
    assert PhaseShift(to=NodeSet.RIGHT).to is NodeSet.RIGHT
    assert SincShift(to=NodeSet.LEFT).to is NodeSet.LEFT


def test_phase_shift_is_exact_on_odd_n_real_origins():
    mesh = IntervalMesh(9, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    ft = Fourier(grid)
    wave = lambda x: (jnp.sin(TWO_PI * x)  # noqa: E731
                      + jnp.cos(TWO_PI * 4 * x))
    center_hat = ft.forward(grid.create_field(init=wave))
    right_hat = ft.forward(grid.create_field(mesh.right, init=wave))
    shifted = PhaseShift(to=NodeSet.RIGHT)(center_hat)
    # odd n: no Nyquist mode, the shift is exact on every mode
    assert jnp.allclose(shifted.data, right_hat.data, atol=1e-13)


# ================================================================
#  PhaseShift / SincShift eigenvalue symbols
# ================================================================
def test_phase_shift_eigenvalues_matches_the_apply(periodic):
    grid, _ = periodic
    ft = Fourier(grid)
    wave = lambda x: (jnp.sin(TWO_PI * x)  # noqa: E731
                      + jnp.cos(5 * TWO_PI * x))
    center_hat = ft.forward(grid.create_field(init=wave))
    op = PhaseShift(to=NodeSet.RIGHT)
    sym = op.eigenvalues(grid, center_hat.function_space)
    assert isinstance(sym, Symbol)
    assert sym.codomain.origin.node_set is NodeSet.RIGHT
    assert jnp.allclose(sym(center_hat).data,
                        op(center_hat).data, atol=1e-13)


def test_sinc_shift_eigenvalues_matches_the_apply(periodic):
    grid, mesh = periodic
    dx = mesh.dx
    ft = Fourier(grid)
    avg = grid.create_field(
        mesh.cell_avg,
        init=lambda x: jnp.sin(TWO_PI * x) * jnp.sinc(dx))
    avg_hat = ft.forward(avg)
    op = SincShift(to=NodeSet.CENTER)
    sym = op.eigenvalues(grid, avg_hat.function_space)
    assert sym.codomain.origin.node_set is NodeSet.CENTER
    assert jnp.allclose(sym(avg_hat).data, op(avg_hat).data,
                        atol=1e-13)
