"""End-to-end tests for ROADMAP task 1.2 (sketch-4.1 flow).

The integration surface of the Wave-2 merge: seeded registry, field
forwarders, staggered stencil operators over the halo-filled
single-device decomposition, and the operator algebra applied to
real fields.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.spaces.nodal import NodeSet

TWO_PI = 2.0 * jnp.pi


def _periodic_grid(n):
    mesh = IntervalMesh(n, (0.0, 1.0), name="x")
    return Grid((mesh,)), mesh


def _sin_field(grid):
    return grid.create_field(init=lambda x: jnp.sin(TWO_PI * x))


# ================================================================
#  Sketch 4.1: staggered derivative, .to, strict algebra
# ================================================================
def test_diff_lands_on_right_with_second_order_convergence():
    errors = []
    for n in (16, 32, 64):
        grid, mesh = _periodic_grid(n)
        f = _sin_field(grid)
        g = f.diff("x")
        assert g.function_space.bare is mesh.right
        x_right = grid.evaluation_nodes(mesh.right).data
        exact = TWO_PI * jnp.cos(TWO_PI * x_right)
        errors.append(float(jnp.max(jnp.abs(g.data - exact))))
    # second order: each doubling divides the error by ~4
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.1)


def test_to_interpolates_back_with_the_right_codomain():
    grid, mesh = _periodic_grid(32)
    f = _sin_field(grid)
    g = f.diff("x")
    h = g.to(f)
    assert h.function_space is f.function_space
    x_center = grid.evaluation_nodes(mesh.center).data
    exact = TWO_PI * jnp.cos(TWO_PI * x_center)
    assert jnp.allclose(h.data, exact, atol=TWO_PI ** 3 / 32 ** 2)
    _ = f + h  # same space now: the strict algebra accepts it


def test_cross_space_add_still_raises():
    grid, _ = _periodic_grid(16)
    f = _sin_field(grid)
    g = f.diff("x")
    with pytest.raises(SpaceMismatchError, match="x"):
        _ = f + g


def test_products_route_through_the_real_registry():
    grid, mesh = _periodic_grid(32)
    f = _sin_field(grid)
    g = f.diff("x")
    u = g * f.to(g)
    assert u.function_space.bare is mesh.right
    x_right = grid.evaluation_nodes(mesh.right).data
    exact = (TWO_PI * jnp.cos(TWO_PI * x_right)
             * jnp.sin(TWO_PI * x_right))
    assert jnp.allclose(u.data, exact, atol=0.1)
    # and the quotient/power kinds resolve on the same rows
    _ = u / (2.0 + f.to(g))
    _ = f ** 2
    _ = abs(f)


# ================================================================
#  Bounded mesh: BC-structured fill under the stencils
# ================================================================
def test_bounded_diff_is_exterior_free_and_gated():
    mesh = IntervalMesh(16, (0.0, 1.0), periodic=False, name="y")
    grid = Grid((mesh,))
    f = grid.create_field(init=lambda y: y * (1.0 - y))
    df = f.diff("y")
    assert df.function_space.bare is mesh.inner
    y_inner = grid.evaluation_nodes(mesh.inner).data
    assert jnp.allclose(df.data, 1.0 - 2.0 * y_inner)
    # the second derivative needs the wall faces, which a BC-free
    # bounded space does not define (R1, boundary_plan.md): the
    # default Inner -> Center row does not exist — declare BC
    # structure or opt into boundary="one_sided" (the one-sided d2
    # numerics live in test_boundary_closures)
    with pytest.raises(DispatchError, match="diff"):
        df.diff("y")


def test_dirichlet_structured_storage_gets_the_odd_extension():
    mesh = IntervalMesh(8, (0.0, 1.0), periodic=False, name="y")
    # one device pinned: the assertions index the single-shard
    # storage frame (the blocked variant lives in test_multi_device)
    grid = Grid((mesh,), device_ids=(0,))
    space = mesh.nodal(NodeSet.CENTER, bc=BC.DIRICHLET)
    f = grid.create_field(space,
                          init=lambda y: jnp.sin(jnp.pi * y))
    # consumption-side contract (task 1.8): created fields claim
    # zero ghost validity; the BC-structured fill appears with the
    # sync (forced explicitly here to inspect the storage frame)
    f = grid.sync(f)
    storage = np.asarray(f._data)
    data = np.asarray(f.data)
    width = grid.decomposition.halo["y"]
    # odd extension, ghost slot k mirrors the k-th interior DOF
    assert storage[width - 1] == pytest.approx(-data[0])
    assert storage[width - 2] == pytest.approx(-data[1])
    assert storage[-width] == pytest.approx(-data[-1])
    assert storage[-width + 1] == pytest.approx(-data[-2])


# ================================================================
#  Operator algebra end to end
# ================================================================
def test_separable_composite_second_derivative():
    grid, mesh = _periodic_grid(64)
    f = _sin_field(grid)
    fd = FiniteDifference(order=2)
    chain = fd["x"] @ fd["x"]
    # halo 2 honored: the chain declares the summed requirement
    assert chain.requirements(mesh.center).halo == 2
    d2 = chain(f)
    assert d2.function_space.bare is mesh.center
    # identical to the synced two-step route
    two_step = f.diff("x").diff("x")
    assert jnp.allclose(d2.data, two_step.data)
    x_center = grid.evaluation_nodes(mesh.center).data
    exact = -(TWO_PI ** 2) * jnp.sin(TWO_PI * x_center)
    assert jnp.allclose(d2.data, exact,
                        atol=TWO_PI ** 4 / 64 ** 2)


# ================================================================
#  jit: one trace across same-shape calls
# ================================================================
def test_tendency_expression_traces_once(compile_counter):
    grid, _ = _periodic_grid(16)
    a = _sin_field(grid)
    b = grid.create_field(init=lambda x: jnp.cos(TWO_PI * x))
    c = grid.create_field(init=lambda x: x * (1.0 - x))

    @jax.jit
    def tendency(u, v):
        du = u.diff("x")
        return du * v.to(du) - 0.5 * du + du ** 2

    tendency(a, b).block_until_ready()  # compile once
    compile_counter.reset()
    tendency(b, c).block_until_ready()
    tendency(c, a).block_until_ready()
    assert compile_counter.count == 0
