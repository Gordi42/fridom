"""Tests for fridom.spatial.operators.integrate.

Also covers the two seams this cluster wires for it: the
``grid.measure`` accessor and the ``f.integrate`` / ``f.mean``
forwarders on ``ScalarField``.
"""
import jax.numpy as jnp
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def integral():
    return Integral()


# ================================================================
#  Static surface and codomain
# ================================================================
def test_dispatch_kind(integral):
    assert integral.dispatch_kind == "integrate"


def test_requirements(integral, mx):
    req = integral.requirements(mx.center)
    assert req.halo == 0
    assert req.layout == "any"
    assert req.collective is True


def test_codomain(integral, mx):
    assert integral.codomain(mx.center) is mx.constant
    assert integral.codomain(mx.cell_avg) is mx.constant
    assert integral.codomain(mx.constant) is mx.constant
    assert integral.codomain(mx.center.as_complex()) is (
        mx.constant.as_complex())


def test_codomain_rejects_coefficient_factors(integral, mx):
    fourier = mx.fourier(origin=mx.center)
    with pytest.raises(SpaceMismatchError, match="transform back"):
        integral.codomain(fourier)


def test_codomain_rejects_non_factor_domains(integral, mx, my):
    with pytest.raises(SpaceMismatchError,
                       match="nodal and average"):
        integral.codomain(mx.center * my.center)


# ================================================================
#  grid.measure (the quadrature-weight seam)
# ================================================================
def test_measure_is_the_uniform_cell_width(mx):
    grid = Grid((mx,))
    w = grid.measure(mx.center)
    assert w.function_space.bare is mx.center
    assert jnp.allclose(w.data, jnp.full(8, mx.dx))
    assert grid.measure(mx.cell_avg).data.shape == (8,)
    assert jnp.allclose(grid.measure(mx.right).data, mx.dx)


def test_measure_halves_the_boundary_dual_cells(my):
    grid = Grid((my,))
    w = grid.measure(my.outer)
    dy = my.dx
    expected = jnp.full(9, dy).at[0].set(dy / 2).at[-1].set(dy / 2)
    assert jnp.allclose(w.data, expected)
    # the trapezoid weights sum to the domain length
    assert jnp.allclose(w.data.sum(), 2.0)
    # face-family interior measures stay full dual cells
    assert jnp.allclose(grid.measure(my.inner).data, dy)
    assert grid.measure(my.face_avg).data.shape == (7,)


def test_measure_tags_the_querying_space(mx, my):
    grid = Grid((mx, my))
    w = grid.measure(mx.center * my.center, name="x")
    factors = w.function_space.bare.factors
    assert factors[0] is mx.center
    assert isinstance(factors[1], ConstantSpace)
    assert w.name == "dx"
    # broadcasts against the querying space
    f = grid.create_field(mx.center * my.center)
    assert (f * w).function_space.bare is mx.center * my.center


def test_measure_name_resolution_errors(mx, my):
    grid = Grid((mx, my))
    with pytest.raises(ValueError, match="ambiguous"):
        grid.measure(mx.center * my.center)
    with pytest.raises(ValueError, match="constant"):
        grid.measure(mx.constant * my.center, name="x")
    with pytest.raises(ValueError, match="coefficient"):
        grid.measure(mx.fourier(origin=mx.center) * my.center,
                     name="x")


def test_measure_drops_bc_constrained_boundary_dofs(my):
    grid = Grid((my,))
    space = my.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    w = grid.measure(space)
    # both boundary DOFs are dropped, like the space shape drops them
    assert w.data.shape == space.shape
    assert jnp.allclose(w.data, my.dx)


def test_measure_is_iteration_1_interval_only():
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="s")
    grid = Grid((cheb,))
    with pytest.raises(NotImplementedError, match="IntervalMesh"):
        grid.measure(cheb.outer)


# ================================================================
#  Integral application
# ================================================================
def test_integral_of_sin_vanishes_on_a_periodic_axis(mx):
    grid = Grid((mx,))
    f = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    total = f.integrate("x")
    assert total.function_space.bare is mx.constant
    assert jnp.allclose(total.data[0], 0.0, atol=1e-12)


def test_integral_is_exact_on_cell_averages(my):
    # exact cell averages of y^2 over [0, 2]: sum(avg * dy) = 8/3
    grid = Grid((my,))
    edges = jnp.linspace(0.0, 2.0, 9)
    averages = (edges[1:] ** 3 - edges[:-1] ** 3) / (3 * my.dx)
    f = grid.create_field(my.cell_avg, data=averages)
    assert jnp.allclose(f.integrate("y").data[0], 8.0 / 3.0)


def test_integral_is_the_trapezoid_rule_on_outer(my):
    grid = Grid((my,))
    f = grid.create_field(my.outer, init=lambda y: 3.0 * y)
    # trapezoid is exact on linears: int_0^2 3y dy = 6
    assert jnp.allclose(f.integrate("y").data[0], 6.0)


def test_integral_result_has_default_metadata(mx):
    grid = Grid((mx,))
    f = grid.create_field(name="q", units="kg")
    assert f.integrate("x").name == "unnamed"  # new quantity


# ================================================================
#  f.integrate forwarder
# ================================================================
def test_partial_reduction_keeps_the_other_factors(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) + y)
    fx = f.integrate("x")
    factors = fx.function_space.bare.factors
    assert isinstance(factors[0], ConstantSpace)
    assert factors[1] is my.center
    assert fx.shape == (1, 8)


def test_no_names_reduces_every_factor(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(init=lambda x, y: 1.0 + 0.0 * x * y)
    total = f.integrate()
    assert all(isinstance(factor, ConstantSpace)
               for factor in total.function_space.bare.factors)
    assert jnp.allclose(total.data, 1.0 * 2.0)  # the domain volume


def test_reduction_along_a_constant_factor_is_identity(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(mx.center * my.constant)
    assert f.integrate("y") is f


def test_integrate_broadcasts_back(mx):
    # f - f.integrate("x") stays in the strict algebra (rules 3.3)
    grid = Grid((mx,))
    f = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x) + 5.0)
    residual = f - f.integrate("x")
    assert residual.function_space.bare is mx.center
    assert jnp.allclose(residual.integrate("x").data[0], 0.0,
                        atol=1e-12)


def test_integrate_validates_and_dedupes_names(mx):
    grid = Grid((mx,))
    f = grid.create_field()
    with pytest.raises(ValueError, match="no factors named"):
        f.integrate("z")
    once = f.integrate("x")
    twice = f.integrate("x", "x")
    assert jnp.allclose(once.data, twice.data)


def test_integrate_coefficient_factor_says_transform_back(mx):
    grid = Grid((mx,))
    f = grid.create_field(mx.fourier(origin=mx.center))
    with pytest.raises(DispatchError, match="transform back"):
        f.integrate("x")


# ================================================================
#  f.mean forwarder
# ================================================================
def test_mean_of_a_constant_is_the_constant(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(init=lambda x, y: 3.0 + 0.0 * x * y)
    assert jnp.allclose(f.mean().data, 3.0)


def test_mean_removes_the_oscillation(mx):
    grid = Grid((mx,))
    f = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x) + 4.0)
    assert jnp.allclose(f.mean("x").data[0], 4.0, atol=1e-12)


def test_mean_uses_the_trapezoid_measure_on_outer(my):
    grid = Grid((my,))
    f = grid.create_field(my.outer, init=lambda y: 3.0 * y)
    # mean of 3y over [0, 2] is 3 (trapezoid exact on linears)
    assert jnp.allclose(f.mean("y").data[0], 3.0)


def test_mean_on_an_all_constant_space_is_identity(mx):
    grid = Grid((mx,))
    f = grid.create_field(mx.constant)
    assert f.mean() is f
