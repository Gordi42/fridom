"""Tests for the coefficient spaces (framework2/grid/spaces/coefficient.py)."""
import pytest

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.meshes.chebyshev import ChebyshevMesh
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.coefficient import (
    ChebyshevSpace,
    CoefficientSpace,
    CosineSpace,
    SineSpace,
)
from fridom.framework2.grid.spaces.nodal import NodeSet

N = 8


@pytest.fixture
def periodic():
    return IntervalMesh(N, (0, 1), name="x")


@pytest.fixture
def bounded():
    return IntervalMesh(N, (0, 1), periodic=False, name="x")


# ================================================================
#  The origin is constitutive
# ================================================================
def test_origin_property(periodic):
    space = periodic.fourier(origin=periodic.center)
    assert isinstance(space, CoefficientSpace)
    assert space.origin is periodic.center


def test_distinct_origins_are_distinct_spaces(periodic):
    of_center = periodic.fourier(origin=periodic.center)
    of_right = periodic.fourier(origin=periodic.right)
    of_cell_avg = periodic.fourier(origin=periodic.cell_avg)
    assert of_center is not of_right
    assert of_center is not of_cell_avg
    assert of_center is periodic.fourier(origin=periodic.center)


def test_bc_is_the_origins(bounded):
    origin = bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    space = bounded.sine(origin=origin)
    assert space.bc is origin.bc


def test_scalars_follow_the_origin(periodic):
    real = periodic.fourier(origin=periodic.center)
    assert real.scalars is Scalars.REAL
    complex_space = periodic.fourier(
        origin=periodic.center.as_complex())
    assert complex_space.scalars is Scalars.COMPLEX


# ================================================================
#  Shapes: the rfft half-spectrum rule
# ================================================================
def test_fourier_real_origin_is_the_half_spectrum(periodic):
    space = periodic.fourier(origin=periodic.center)
    assert space.shape == (N // 2 + 1,)


def test_fourier_complex_origin_is_the_full_spectrum(periodic):
    space = periodic.fourier(origin=periodic.center.as_complex())
    assert space.shape == (N,)


def test_as_complex_changes_the_shape(periodic):
    # never a dtype flag flip: the origin is complexified
    real = periodic.fourier(origin=periodic.center)
    complexified = real.as_complex()
    assert complexified.shape == (N,)
    assert complexified.origin is periodic.center.as_complex()
    assert complexified is periodic.fourier(
        origin=periodic.center.as_complex())


def test_as_complex_roundtrip(periodic):
    real = periodic.fourier(origin=periodic.center)
    assert real.as_complex().as_real() is real
    assert real.as_real() is real
    assert real.as_complex().as_complex() is real.as_complex()


# ================================================================
#  Sine / cosine / Chebyshev shapes
# ================================================================
def test_dst_i_of_dirichlet_inner(bounded):
    origin = bounded.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    space = bounded.sine(origin=origin)
    assert type(space) is SineSpace
    assert space.shape == (N - 1,)


def test_dst_ii_of_dirichlet_center(bounded):
    origin = bounded.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    assert bounded.sine(origin=origin).shape == (N,)


def test_dct_ii_of_neumann_center(bounded):
    origin = bounded.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    space = bounded.cosine(origin=origin)
    assert type(space) is CosineSpace
    assert space.shape == (N,)


def test_dct_i_of_neumann_outer(bounded):
    # Neumann never reduces the origin shape: n + 1 nodes make the
    # DCT-I shape-honest (n + 1 cosine modes k = 0..n)
    origin = bounded.nodal(NodeSet.OUTER, bc=BC.NEUMANN)
    assert origin.shape == (N + 1,)
    space = bounded.cosine(origin=origin)
    assert type(space) is CosineSpace
    assert space.shape == (N + 1,)


def test_chebyshev_shape():
    mesh = ChebyshevMesh(N, (0, 1), name="z")
    space = mesh.chebyshev(origin=mesh.lobatto)
    assert type(space) is ChebyshevSpace
    assert space.shape == (N + 1,)


# ================================================================
#  Layout protocol
# ================================================================
def test_layout_variants(periodic):
    space = periodic.fourier(origin=periodic.center)
    laid_out = space.with_layout("L0")
    assert laid_out is space.with_layout("L0")
    assert laid_out.bare is space
    assert laid_out.origin is space.origin


# ================================================================
#  Repr
# ================================================================
def test_repr(periodic, bounded):
    assert repr(periodic.fourier(origin=periodic.center)) == (
        "Fourier(x, origin=Center)")
    assert repr(periodic.fourier(origin=periodic.right)) == (
        "Fourier(x, origin=Right)")
    origin = bounded.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    assert repr(bounded.sine(origin=origin)) == (
        "Sine(x, origin=Inner, bc=(DIRICHLET, DIRICHLET))")
