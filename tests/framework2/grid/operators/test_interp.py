"""Tests for fridom.framework2.grid.operators.interp."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import EigenbasisError
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def interp():
    return LinearInterp()


# ================================================================
#  Construction and static surface
# ================================================================
def test_target_validation():
    assert LinearInterp().target is None
    assert LinearInterp(target=NodeSet.OUTER).target is NodeSet.OUTER
    with pytest.raises(TypeError, match="NodeSet"):
        LinearInterp(target="outer")


def test_dispatch_kind_and_requirements(interp, mx):
    assert interp.dispatch_kind == "interpolate"
    assert interp.requirements(mx.center).halo == 1
    assert interp.requirements(mx.center).layout == "any"


# ================================================================
#  Per-factor signatures (codomain table)
# ================================================================
def test_codomain_periodic(interp, mx):
    assert interp.codomain(mx.center) is mx.right
    assert interp.codomain(mx.right) is mx.center


def test_codomain_bounded(interp, my):
    assert interp.codomain(my.center) is my.inner
    assert interp.codomain(my.outer) is my.center
    assert interp.codomain(my.inner) is my.center


def test_codomain_preserves_scalars(interp, mx):
    assert interp.codomain(mx.right.as_complex()) is (
        mx.center.as_complex())


def test_codomain_retags_a_fourier_factor(interp, mx):
    # layout-faithful eigenvalue threading (decision 3): the codomain of
    # a Fourier factor retags through its staggered origin, preserving
    # the Körper (so a chain threads coefficient spaces)
    src = mx.fourier(origin=mx.center)
    assert interp.codomain(src) is mx.fourier(origin=mx.right)
    csrc = mx.fourier(origin=mx.center).as_complex()
    assert interp.codomain(csrc) is (
        mx.fourier(origin=mx.right).as_complex())


def test_codomain_outer_variant(my):
    outer = LinearInterp(target=NodeSet.OUTER)
    assert outer.codomain(my.center) is my.outer


def test_codomain_outer_variant_rejects_other_domains(mx, my):
    outer = LinearInterp(target=NodeSet.OUTER)
    with pytest.raises(SpaceMismatchError, match="Center -> Outer"):
        outer.codomain(mx.center)  # periodic has no Outer
    with pytest.raises(SpaceMismatchError, match="Center -> Outer"):
        outer.codomain(my.inner)


def test_codomain_rejects_average_and_unlisted(interp, mx, my):
    with pytest.raises(SpaceMismatchError, match="reconstruct"):
        interp.codomain(mx.cell_avg)
    with pytest.raises(SpaceMismatchError,
                       match="no interpolate signature"):
        interp.codomain(my.left)


# ================================================================
#  Application
# ================================================================
def test_periodic_center_to_right_wraps(interp, mx):
    grid = Grid((mx,))
    f = grid.create_field(data=jnp.arange(8.0))
    g = interp["x"](f)
    assert g.function_space.bare is mx.right
    expected = 0.5 * (jnp.arange(8.0)
                      + jnp.roll(jnp.arange(8.0), -1))
    assert jnp.allclose(g.data, expected)


def test_bounded_center_to_inner(interp, my):
    grid = Grid((my,))
    f = grid.create_field(data=jnp.arange(8.0))
    g = interp["y"](f)
    assert g.function_space.bare is my.inner
    assert jnp.allclose(g.data, jnp.arange(7) + 0.5)


def test_bounded_outer_variant_extrapolates_boundary_faces(my):
    grid = Grid((my,))
    outer = LinearInterp(target=NodeSet.OUTER)
    f = grid.create_field(init=lambda y: 2.0 * y + 1.0)
    g = outer["y"](f)
    assert g.function_space.bare is my.outer
    # exact for linear data, including the extrapolated boundaries
    y_outer = grid.evaluation_nodes(my.outer).data
    assert jnp.allclose(g.data, 2.0 * y_outer + 1.0)


def test_metadata_is_preserved(interp, mx):
    grid = Grid((mx,))
    f = grid.create_field(name="u", units="m/s")
    g = interp["x"](f)
    assert g.metadata == f.metadata  # same-quantity rule


# ================================================================
#  Eigenvalue symbol (the one_hat averaging diagonal)
# ================================================================
def test_eigenvalues_matches_the_apply(interp, mx):
    grid = Grid((mx,))
    ft = grid.dispatch.resolve("transform", mx.center)
    f = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x)
        + jnp.cos(3 * 2 * jnp.pi * x))
    center_hat = ft.forward(f)
    sym = interp["x"].eigenvalues(grid, mx.center)
    # retags Fourier(Center) -> Fourier(Right)
    assert sym.space.origin.node_set is NodeSet.CENTER
    assert sym.codomain.origin.node_set is NodeSet.RIGHT
    op_hat = ft.forward(interp["x"](f))
    assert jnp.allclose(sym(center_hat).data, op_hat.data, atol=1e-12)


def test_eigenvalues_nyquist_is_an_exact_structural_zero(interp, mx):
    grid = Grid((mx,))
    sym = interp["x"].eigenvalues(grid, mx.center)
    # cos(pi/2) = 0 exactly: the snapped Nyquist leaf is a structural
    # zero, so ``Symbol.inverse`` regularizes it (no ~1e-17 residue)
    assert sym.data.ravel()[-1] == 0.0


def test_eigenvalues_raise_on_the_wrong_boundary(interp, my, mx):
    # bounded meshes diagonalize in the sine/cosine basis
    with pytest.raises(EigenbasisError, match="periodic"):
        interp["y"].eigenvalues(Grid((my,)), my.center)
    # the target= variant has no diagonalizing symbol in iteration 1
    with pytest.raises(EigenbasisError, match="target="):
        LinearInterp(target=NodeSet.OUTER)["x"].eigenvalues(
            Grid((mx,)), mx.center)
