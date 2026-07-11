"""Tests for fridom.spatial.operators.flux_diff."""
import jax.numpy as jnp
import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.base import (
    Dispatched,
    EigenbasisError,
    SeparableComposite,
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


def test_eigenvalues_designed_for(flux, mx):
    with pytest.raises(EigenbasisError):
        flux.eigenvalues(None, mx.right)


def test_flux_codomains(flux, mx, my):
    assert flux.codomain(mx.right) is mx.cell_avg
    assert flux.codomain(my.outer) is my.cell_avg
    assert flux.codomain(my.inner) is my.cell_avg
    assert flux.codomain(mx.right.as_complex()) is (
        mx.cell_avg.as_complex())


def test_flux_right_is_periodic_only(flux, my):
    with pytest.raises(SpaceMismatchError, match="periodic-only"):
        flux.codomain(my.right)


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
