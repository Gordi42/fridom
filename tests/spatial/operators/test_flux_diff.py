"""Tests for fridom.spatial.operators.flux_diff."""
import jax.numpy as jnp
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.base import (
    Dispatched,
    EigenbasisError,
    SeparableComposite,
)
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.flux_diff import (
    DualFluxDifference,
    FaceDifference,
    FluxDifference,
    FVDerivative,
)
from fridom.spatial.operators.reconstruct import (
    LinearReconstruction,
)
from fridom.spatial.operators.spectral import fourier_wavenumbers
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def flux():
    return FluxDifference()


@pytest.fixture
def dual():
    return DualFluxDifference()


@pytest.fixture
def face():
    return FaceDifference()


# ================================================================
#  Static surface and codomain tables
# ================================================================
def test_dispatch_kinds(flux, dual, face):
    assert flux.dispatch_kind == "flux_diff"
    assert dual.dispatch_kind == "flux_diff"
    assert face.dispatch_kind == "face_diff"


def test_requirements(flux, dual, face, mx):
    for op in (flux, dual, face):
        assert op.requirements(mx.cell_avg).halo == 1
        assert op.requirements(mx.cell_avg).layout == "any"


def test_flux_codomains(flux, mx, my):
    assert flux.codomain(mx.right) is mx.cell_avg
    assert flux.codomain(my.outer) is my.cell_avg
    assert flux.codomain(my.inner) is my.cell_avg
    assert flux.codomain(mx.right.as_complex()) is (
        mx.cell_avg.as_complex())


def test_flux_right_is_periodic_only(flux, my):
    with pytest.raises(SpaceMismatchError, match="periodic-only"):
        flux.codomain(my.right)


def test_flux_accepts_dirichlet_inner_walled(flux, my):
    # F4: a Dirichlet-tagged interior face closes the flux divergence
    # with the exact-zero wall value; the codomain is the BC-free CellAvg
    inner_dir = my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    assert flux.codomain(inner_dir) is my.cell_avg


def test_flux_rejects_neumann_inner_walled(flux, my):
    # a Neumann tag claims no wall value, so the divergence cannot close
    inner_neu = my.nodal(NodeSet.INNER, bc=BC.NEUMANN)
    with pytest.raises(SpaceMismatchError, match="Neumann"):
        flux.codomain(inner_neu)


def test_flux_rejects_center_domains(flux, mx):
    with pytest.raises(SpaceMismatchError,
                       match="DualFluxDifference"):
        flux.codomain(mx.center)


def test_dual_codomains(dual, mx, my):
    assert dual.codomain(mx.center) is mx.face_avg
    assert dual.codomain(mx.cell_avg) is mx.face_avg
    assert dual.codomain(my.center) is my.face_avg
    assert dual.codomain(my.cell_avg) is my.face_avg


def test_dual_rejects_face_domains(dual, my):
    with pytest.raises(SpaceMismatchError,
                       match="FluxDifference"):
        dual.codomain(my.outer)


def test_face_diff_codomains(face, mx, my):
    assert face.codomain(mx.cell_avg) is mx.right
    assert face.codomain(my.cell_avg) is my.inner


def test_face_diff_rejects_nodal_domains(face, mx):
    with pytest.raises(SpaceMismatchError,
                       match="no face_diff signature"):
        face.codomain(mx.center)


# ================================================================
#  Fourier symbols (eigenvalues) — scoping study gap G1
# ================================================================
def _symbol_matches_operator(op, mesh, dom_space, seed=5):
    """Symbol in coeff space == coeff image of the physical output."""
    grid = Grid((mesh,))
    axis = mesh.names[0]
    f = grid.random.normal(dom_space, seed=seed)
    t_in = grid.dispatch.resolve("transform", dom_space)
    t_out = grid.dispatch.resolve("transform", op.codomain(dom_space))
    fhat = t_in.forward(f)
    sym = op[axis].eigenvalues(grid, fhat.function_space.bare)
    got = sym(fhat).data
    want = t_out.forward(op[axis](f)).data
    return float(jnp.abs(got - want).max())


@pytest.mark.parametrize("n", [8, 9, 16, 17])
def test_flux_diff_symbol_matches_operator(flux, n):
    # the divergence leg on periodic uniform meshes, both parities
    mesh = IntervalMesh(n, (0.0, 1.3), name="x")
    assert _symbol_matches_operator(flux, mesh, mesh.right) < 1e-12


@pytest.mark.parametrize("n", [8, 9, 16, 17])
def test_face_diff_symbol_matches_operator(face, n):
    mesh = IntervalMesh(n, (0.0, 1.3), name="x")
    assert _symbol_matches_operator(face, mesh, mesh.cell_avg) < 1e-12


@pytest.mark.parametrize("n", [8, 9])
@pytest.mark.parametrize("dom", ["center", "cell_avg"])
def test_dual_flux_diff_symbol_matches_operator(dual, n, dom):
    mesh = IntervalMesh(n, (0.0, 1.3), name="x")
    assert _symbol_matches_operator(
        dual, mesh, getattr(mesh, dom)) < 1e-12


def test_flux_diff_symbol_is_the_ik_hat_retagging_symbol(flux, mx):
    grid = Grid((mx,))
    sym = flux["x"].eigenvalues(grid, mx.fourier(origin=mx.right))
    # retags Fourier(Right) -> Fourier(CellAvg)
    assert sym.space.origin is mx.right
    assert sym.codomain.origin is mx.cell_avg
    k = fourier_wavenumbers(mx.fourier(origin=mx.right))
    dx = mx.dx
    # i k_hat with the Right-origin half-cell phase e^{-i k dx/2}
    expected = (1j * 2.0 * jnp.sin(k * dx / 2.0) / dx
                * jnp.exp(-1j * k * 0.5 * dx))
    assert jnp.allclose(sym.data, expected)
    # the Nyquist leaf is kept and snapped real: i k_hat e^{-iπ/2} =
    # 2/dx (representable, matching staggered_diff — so the pressure
    # projection reaches machine zero at Nyquist too)
    assert sym.data.ravel()[-1] == 2.0 / dx
    assert sym.data.ravel()[-1].imag == 0.0


def test_fv_legs_match_the_nodal_stencil_numbers(flux, face, mx):
    # scoping study §1: the FV divergence/gradient legs are bitwise the
    # nodal FiniteDifference numbers, differing only in the codomain tag
    grid = Grid((mx,))
    fd = FiniteDifference()
    div_fv = flux["x"].eigenvalues(grid, mx.fourier(origin=mx.right))
    div_nod = fd["x"].eigenvalues(grid, mx.fourier(origin=mx.right))
    assert jnp.array_equal(div_fv.data, div_nod.data)
    assert div_fv.codomain.origin is mx.cell_avg  # not Center
    grad_fv = face["x"].eigenvalues(grid, mx.fourier(origin=mx.cell_avg))
    grad_nod = fd["x"].eigenvalues(grid, mx.fourier(origin=mx.center))
    assert jnp.array_equal(grad_fv.data, grad_nod.data)


def test_eigenvalues_thread_bare_or_fourier_factor(flux, face, mx):
    # layout-faithful threading: a bare periodic origin (nodal for
    # flux, average for face) yields the same symbol as its Fourier
    # coefficient factor (fv_fourier_partner resolves both)
    grid = Grid((mx,))
    for op, origin in ((flux, mx.right), (face, mx.cell_avg)):
        bare = op["x"].eigenvalues(grid, origin)
        coeff = op["x"].eigenvalues(grid, mx.fourier(origin=origin))
        assert coeff.space is bare.space
        assert coeff.codomain is bare.codomain
        assert jnp.array_equal(coeff.data, bare.data)


def test_codomain_retags_a_fourier_factor(flux, dual, face, mx):
    # the eigenvalue chain resolves codomains on Fourier factors
    assert flux.codomain(mx.fourier(origin=mx.right)) is (
        mx.fourier(origin=mx.cell_avg))
    assert dual.codomain(mx.fourier(origin=mx.center)) is (
        mx.fourier(origin=mx.face_avg))
    assert dual.codomain(mx.fourier(origin=mx.cell_avg)) is (
        mx.fourier(origin=mx.face_avg))
    assert face.codomain(mx.fourier(origin=mx.cell_avg)) is (
        mx.fourier(origin=mx.right))


def test_eigenvalues_raise_on_bounded(flux, dual, face, my):
    # average families have no diagonalizing basis on a walled mesh
    grid = Grid((my,))
    with pytest.raises(EigenbasisError, match="periodic"):
        flux["y"].eigenvalues(grid, my.outer)
    with pytest.raises(EigenbasisError, match="periodic"):
        dual["y"].eigenvalues(grid, my.center)
    with pytest.raises(EigenbasisError, match="periodic"):
        face["y"].eigenvalues(grid, my.cell_avg)


def test_eigenvalues_raise_on_mapped(flux, face, mapped_periodic):
    # a stretched mesh's non-constant metric breaks translation
    # invariance: no diagonal symbol (scoping study G5/F5)
    mesh = mapped_periodic
    grid = Grid((mesh,))
    with pytest.raises(EigenbasisError, match="periodic"):
        flux["w"].eigenvalues(grid, mesh.right)
    with pytest.raises(EigenbasisError, match="periodic"):
        face["w"].eigenvalues(grid, mesh.cell_avg)


def test_eigenvalues_raise_on_non_fourier_coefficient(face, my):
    # a trig coefficient factor is not a periodic Fourier basis
    grid = Grid((my,))
    sine = my.sine(my.nodal(NodeSet.CENTER, bc=BC.DIRICHLET))
    with pytest.raises(EigenbasisError, match="sine/cosine"):
        face["y"].eigenvalues(grid, sine)


def test_fv_laplacian_symbol_is_the_real_neg_khat2(mx):
    # FluxDifference @ FaceDifference: the CellAvg pressure Laplacian —
    # real -k_hat**2, the honest discrete div @ grad (Nyquist kept)
    grid = Grid((mx,))
    lap = FluxDifference() @ FaceDifference()
    coeff = mx.fourier(origin=mx.cell_avg)
    sym = lap["x"].eigenvalues(grid, coeff)
    assert sym.space.origin is mx.cell_avg
    assert sym.codomain.origin is mx.cell_avg
    k = fourier_wavenumbers(coeff)
    dx = mx.dx
    khat2 = -2.0 * (1.0 - jnp.cos(k * dx)) / dx ** 2
    assert float(jnp.max(jnp.abs(sym.data.imag))) < 1e-12
    assert jnp.allclose(sym.data.real, khat2)


def test_walled_fv_grad_div_trig_symbols(face, flux, my):
    # F4: on a walled axis the FV pressure gradient / flux divergence
    # diagonalize in the sine/cosine basis. The grad leg (Neumann
    # CellAvg cosine -> Dirichlet Inner sine) is -2 sin(k dz/2)/dz; the
    # div leg (Dirichlet Inner sine -> Neumann CellAvg cosine) is +2
    # sin(k dz/2)/dz -- the same magnitude, no sinc (bitwise the nodal
    # Center<->Inner numbers), with the family-flip sign
    grid = Grid((my,))
    n = my.n_cells
    dz = (my.extent[1] - my.extent[0]) / n
    cos = my.cosine(my.average(CellAvg, bc=BC.NEUMANN))  # DCT-II domain
    sin = my.sine(my.nodal(NodeSet.INNER, bc=BC.DIRICHLET))  # DST-I
    grad = face["y"].eigenvalues(grid, cos)
    div = flux["y"].eigenvalues(grid, sin)
    # cross-family codomains
    assert face["y"].codomain(cos) is sin
    assert flux["y"].codomain(sin) is cos
    length = my.extent[1] - my.extent[0]

    def khat(m):
        return 2.0 * jnp.sin(m * jnp.pi * dz / (2 * length)) / dz

    grad_expect = jnp.array([-khat(m) for m in range(1, n)])  # sine 1..n-1
    div_expect = jnp.array([khat(m) for m in range(n)])       # cosine 0..n-1
    assert float(jnp.abs(grad.data - grad_expect).max()) < 1e-13
    assert float(jnp.abs(div.data - div_expect).max()) < 1e-13


def test_walled_fv_laplacian_symbol_matches_composed_operator(my):
    # gate 6 (F0 pattern): the composed FV walled Laplacian symbol
    # FluxDifference @ FaceDifference is the real -k_hat**2, and it
    # matches the composed FIELD operator (grad, Dirichlet-mid retag,
    # div) applied to the cosine cell-average eigenmodes to ~1e-14
    grid = Grid((my,))
    n = my.n_cells
    length = my.extent[1] - my.extent[0]
    dz = length / n
    cos = my.cosine(my.average(CellAvg, bc=BC.NEUMANN))
    sym = (FluxDifference() @ FaceDifference())["y"].eigenvalues(grid, cos)
    assert float(jnp.abs(sym.data.imag).max()) < 1e-13
    khat = jnp.array([2.0 * jnp.sin(m * jnp.pi * dz / (2 * length)) / dz
                      for m in range(n)])
    assert float(jnp.abs(sym.data.real - (-khat ** 2)).max()) < 1e-12
    # the field operator on each cosine eigenmode: lap p_m = -khat_m^2 p_m
    inner_dir = my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)

    def cosine_mode(m):
        def init(y):
            return jnp.cos(m * jnp.pi * y / length)
        return init

    for m in range(1, n):  # the non-constant modes
        p = grid.create_field(
            my.average(CellAvg, bc=BC.NEUMANN), init=cosine_mode(m))
        gp = FaceDifference()["y"](p).retag(inner_dir)
        lap = FluxDifference()["y"](gp)
        target = -float(khat[m]) ** 2 * jnp.asarray(p.data)
        assert float(jnp.abs(lap.data - target).max()) < 1e-12


def test_fv_derivative_symbol_is_the_wide_centered_difference(mx):
    # FVDerivative = flux_diff @ reconstruct (CellAvg -> CellAvg): the
    # collocated wide difference i sin(k dx)/dx (the phases cancel)
    grid = Grid((mx,))
    sym = FVDerivative(LinearReconstruction())["x"].eigenvalues(
        grid, mx.fourier(origin=mx.cell_avg))
    k = fourier_wavenumbers(mx.fourier(origin=mx.cell_avg))
    dx = mx.dx
    assert jnp.allclose(sym.data, 1j * jnp.sin(k * dx) / dx)


# ================================================================
#  The FV pressure solve (SpectralSolve gate, plan stage F0)
# ================================================================
@pytest.mark.parametrize("n", [16, 17])
def test_spectral_solve_drives_1d_divergence_to_machine_zero(n):
    # the FV pressure operator (flux_diff of the face-difference
    # gradient) solves a Poisson problem so the discrete divergence
    # after projection is machine zero on a periodic box
    mesh = IntervalMesh(n, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    u = grid.random.normal(mesh.right, seed=4)   # face-normal velocity
    flux = FluxDifference()["x"]
    face = FaceDifference()["x"]
    lap = FluxDifference() @ FaceDifference()    # CellAvg -> CellAvg
    div = flux(u)
    solve = SpectralSolve(lap, grid, div.function_space)
    p = solve(div)
    assert p.function_space.bare is mesh.cell_avg
    corrected_div = flux(u - face(p))
    assert float(jnp.abs(div.data).max()) > 1.0     # non-trivial start
    assert float(jnp.abs(corrected_div.data).max()) < 1e-11


@pytest.mark.parametrize("n", [12, 16])
def test_spectral_solve_projects_a_2d_box_divergence_free(n):
    mx = IntervalMesh(n, (0.0, 1.0), name="x")
    my = IntervalMesh(n, (0.0, 2.0), name="y")
    grid = Grid((mx, my))
    cell = mx.cell_avg * my.cell_avg
    ux = grid.random.normal(mx.right * my.cell_avg, seed=1)
    uy = grid.random.normal(mx.cell_avg * my.right, seed=2)
    fx, fy = FluxDifference()["x"], FluxDifference()["y"]
    gx, gy = FaceDifference()["x"], FaceDifference()["y"]
    div = fx(ux) + fy(uy)
    lap = ((FluxDifference() @ FaceDifference())["x"]
           + (FluxDifference() @ FaceDifference())["y"])
    solve = SpectralSolve(lap, grid, cell)
    p = solve(div)
    corrected = fx(ux - gx(p)) + fy(uy - gy(p))
    assert float(jnp.abs(div.data).max()) > 1.0
    assert float(jnp.abs(corrected.data).max()) < 1e-11
    # residual: the Laplacian recovers the mean-free divergence
    lap_p = ((FluxDifference() @ FaceDifference())["x"](p)
             + (FluxDifference() @ FaceDifference())["y"](p))
    assert float(jnp.abs((lap_p - (div - div.mean())).data).max()) < 1e-11


# ================================================================
#  Exactness: telescoping / conservation (the section-3.9 contract)
# ================================================================
def test_bounded_flux_diff_telescopes_to_boundary_fluxes(flux, my):
    grid = Grid((my,))
    f = grid.random.normal(my.outer, seed=1)
    d = flux["y"](f)
    assert d.function_space.bare is my.cell_avg
    total = d.integrate("y").data[0]
    assert jnp.allclose(total, f.data[-1] - f.data[0])


def test_inner_flux_diff_is_homogeneous(flux, my):
    grid = Grid((my,))
    f = grid.random.normal(my.inner, seed=1)
    d = flux["y"](f)
    # zero boundary fluxes: the total integral telescopes to zero
    assert jnp.allclose(d.integrate("y").data[0], 0.0)
    # and the edge cells difference against an exact zero, never
    # against the BC-free extrapolation ghost
    dy = my.dx
    assert jnp.allclose(d.data[0], f.data[0] / dy)
    assert jnp.allclose(d.data[-1], -f.data[-1] / dy)


def test_periodic_flux_diff_is_conservative(flux, mx):
    grid = Grid((mx,))
    f = grid.random.normal(mx.right, seed=1)
    d = flux["x"](f)
    assert d.function_space.bare is mx.cell_avg
    assert jnp.allclose(d.integrate("x").data[0], 0.0)


def test_flux_diff_is_exact_on_linear_fluxes(flux, my):
    grid = Grid((my,))
    f = grid.create_field(my.outer, init=lambda y: 5.0 * y)
    d = flux["y"](f)
    assert jnp.allclose(d.data, jnp.full(8, 5.0))


def test_dual_flux_diff_telescopes_to_outer_centers(dual, my):
    grid = Grid((my,))
    f = grid.random.normal(my.center, seed=1)
    d = dual["y"](f)
    assert d.function_space.bare is my.face_avg
    total = d.integrate("y").data[0]
    assert jnp.allclose(total, f.data[-1] - f.data[0])


def test_dual_flux_diff_exact_ftc_on_centers(dual, my):
    grid = Grid((my,))
    f = grid.create_field(my.center, init=lambda y: 2.0 * y + 1.0)
    d = dual["y"](f)
    assert jnp.allclose(d.data, jnp.full(7, 2.0))


def test_face_diff_is_the_exact_two_point_gradient(face, my):
    grid = Grid((my,))
    p = grid.create_field(my.cell_avg, init=lambda y: 3.0 * y)
    g = face["y"](p)
    assert g.function_space.bare is my.inner
    assert jnp.allclose(g.data, jnp.full(7, 3.0))


def test_results_carry_default_metadata(flux, mx):
    grid = Grid((mx,))
    f = grid.create_field(mx.right, name="F", units="m/s")
    assert flux["x"](f).name == "unnamed"  # new quantity


# ================================================================
#  FVDerivative (the ("diff", CellAvg) default)
# ================================================================
def test_fv_derivative_is_a_separable_composite():
    chain = FVDerivative()
    assert isinstance(chain, SeparableComposite)
    assert isinstance(chain.factors[0], FluxDifference)
    assert chain.factors[1] is Dispatched("reconstruct")


def test_fv_derivative_accepts_an_explicit_reconstruction():
    recon = LinearReconstruction()
    chain = FVDerivative(recon)
    assert chain.factors[1] is recon


def test_grid_seeds_a_concrete_fv_derivative(mx):
    grid = Grid((mx,))
    op = grid.dispatch.resolve("diff", mx.cell_avg)
    assert isinstance(op, SeparableComposite)
    assert isinstance(op.factors[0], FluxDifference)
    assert isinstance(op.factors[1], LinearReconstruction)
    # the summed chain halo drives the provisional negotiation
    assert op.requirements(mx.cell_avg).halo == 2
    assert grid.decomposition.halo["x"] == 2


def test_fv_diff_converges_at_second_order():
    errors = []
    for n in (16, 32):
        mesh = IntervalMesh(n, (0.0, 1.0), name="x")
        grid = Grid((mesh,))
        f = grid.create_field(
            mesh.cell_avg, init=lambda x: jnp.sin(2 * jnp.pi * x))
        df = f.diff("x")
        assert df.function_space.bare is mesh.cell_avg
        x = grid.evaluation_nodes(mesh.cell_avg).data
        errors.append(
            jnp.abs(df.data - 2 * jnp.pi
                    * jnp.cos(2 * jnp.pi * x)).max())
    assert errors[0] / errors[1] > 3.0


def test_fv_diff_is_conservative(mx, my):
    periodic = Grid((mx,))
    f = periodic.random.normal(mx.cell_avg, seed=2)
    assert jnp.allclose(f.diff("x").integrate("x").data[0], 0.0)
    bounded = Grid((my,))
    g = bounded.random.normal(my.cell_avg, seed=3)
    # bounded default reconstructs onto Inner: homogeneous fluxes
    assert jnp.allclose(g.diff("y").integrate("y").data[0], 0.0)


def test_fv_diff_on_a_2d_average_product(mx, my):
    # chain-safe application: the stored-unbound factors receive
    # the axis from the composite on a multi-axis operand
    grid = Grid((mx, my))
    f = grid.create_field(
        mx.cell_avg * my.cell_avg,
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) + 0.0 * y)
    df = f.diff("x")
    assert df.function_space.bare is mx.cell_avg * my.cell_avg
    x = grid.evaluation_nodes(
        mx.cell_avg * my.cell_avg, name="x").data
    exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
    assert jnp.abs(df.data - exact).max() < 1.0


# ================================================================
#  Mapped meshes: measure-field denominators (stage C0)
# ================================================================
def _tanh_map(s):
    return jnp.tanh(2.0 * s) / jnp.tanh(2.0)


def _wavy_map(s):
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


@pytest.fixture
def mapped_bounded():
    return MappedIntervalMesh(8, (0.0, 1.0), _tanh_map, name="v")


@pytest.fixture
def mapped_periodic():
    return MappedIntervalMesh(8, (0.0, 1.0), _wavy_map,
                              periodic=True, name="w")


def test_mapped_flux_diff_telescopes_exactly(flux, mapped_bounded):
    mesh = mapped_bounded
    grid = Grid((mesh,))
    f = grid.create_field(mesh.outer,
                          init=lambda v: v**3 + 0.5 * v)
    d = flux["v"](f)
    # discrete Gauss: the measure-weighted sum telescopes to the
    # boundary fluxes on the stretched mesh too
    total = d.integrate("v").data.squeeze()
    assert jnp.allclose(total, 1.5)


def test_mapped_flux_diff_is_exact_on_linear_fluxes(
        flux, mapped_bounded):
    mesh = mapped_bounded
    grid = Grid((mesh,))
    f = grid.create_field(mesh.outer, init=lambda v: 3.0 * v)
    d = flux["v"](f)
    assert jnp.allclose(d.data, jnp.full(8, 3.0))


def test_mapped_inner_flux_diff_pads_exact_zero_fluxes(
        flux, mapped_bounded):
    mesh = mapped_bounded
    grid = Grid((mesh,))
    f = grid.random.normal(mesh.inner, seed=7)
    d = flux["v"](f)
    # homogeneous no-normal-flow: conservation to the wall fluxes 0
    assert jnp.allclose(d.integrate("v").data.squeeze(), 0.0)
    # first/last cells divide the wall-adjacent flux by their own
    # primal width
    w = grid.measure(mesh.cell_avg, name="v").data
    assert jnp.allclose(d.data[0], f.data[0] / w[0])
    assert jnp.allclose(d.data[-1], -f.data[-1] / w[-1])


def test_mapped_periodic_flux_diff_is_conservative(
        flux, mapped_periodic):
    mesh = mapped_periodic
    grid = Grid((mesh,))
    f = grid.random.normal(mesh.right, seed=11)
    d = flux["w"](f)
    assert jnp.allclose(d.integrate("w").data.squeeze(), 0.0)


def test_mapped_face_diff_divides_by_the_dual_measure(
        face, mapped_bounded):
    mesh = mapped_bounded
    grid = Grid((mesh,))
    p = grid.create_field(mesh.cell_avg, init=lambda v: 3.0 * v)
    g = face["v"](p)
    assert g.function_space.bare is mesh.inner
    # exact two-point gradient of a linear profile: the dual
    # center-to-center spacing cancels
    assert jnp.allclose(g.data, jnp.full(7, 3.0))


def test_mapped_dual_flux_diff_exact_ftc(dual, mapped_bounded):
    mesh = mapped_bounded
    grid = Grid((mesh,))
    f = grid.create_field(mesh.center, init=lambda v: 2.0 * v + 1.0)
    d = dual["v"](f)
    assert d.function_space.bare is mesh.face_avg
    assert jnp.allclose(d.data, jnp.full(7, 2.0))


def test_mapped_fv_diff_converges_at_second_order():
    errors = []
    for n in (16, 32):
        mesh = MappedIntervalMesh(n, (0.0, 1.0), _wavy_map,
                                  periodic=True, name="w")
        grid = Grid((mesh,))
        f = grid.create_field(
            mesh.cell_avg,
            init=lambda w: jnp.sin(2 * jnp.pi * w))
        df = f.diff("w")
        x = grid.evaluation_nodes(mesh.cell_avg).data
        errors.append(
            jnp.abs(df.data - 2 * jnp.pi
                    * jnp.cos(2 * jnp.pi * x)).max())
    assert errors[0] / errors[1] > 3.0
