"""Tests for fridom.framework2.grid.operators.select."""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.errors import (
    GridMismatchError,
    SpaceMismatchError,
)
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.registry import DispatchError
from fridom.framework2.grid.operators.select import Where


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def where():
    return Where()


# ================================================================
#  Static surface and codomain
# ================================================================
def test_dispatch_kind(where):
    assert where.dispatch_kind == "select"


def test_codomain_returns_the_common_space(where, mx):
    assert where.codomain(mx.right, mx.right, mx.right) is mx.right
    assert where.codomain(
        mx.cell_avg, mx.cell_avg, mx.cell_avg) is mx.cell_avg


def test_codomain_rejects_mismatched_branches(where, mx):
    with pytest.raises(SpaceMismatchError, match="branches"):
        where.codomain(mx.right, mx.right, mx.cell_avg)


def test_codomain_rejects_a_mismatched_condition(where, mx):
    with pytest.raises(SpaceMismatchError, match="condition"):
        where.codomain(mx.cell_avg, mx.right, mx.right)
    # iteration 1: no implicit real -> complex lift of the condition
    with pytest.raises(SpaceMismatchError, match="condition"):
        where.codomain(mx.right, mx.right.as_complex(),
                       mx.right.as_complex())


def test_select_is_strictly_ternary(where, mx):
    grid = Grid((mx,))
    f = grid.create_field(mx.center)
    with pytest.raises(TypeError):
        where(f, f)  # binary call: no ternary codomain
    with pytest.raises(TypeError):
        where(f, f, f, f)  # quaternary call


# ================================================================
#  Application
# ================================================================
def test_where_matches_jnp_where(where, mx):
    grid = Grid((mx,))
    rng = np.random.default_rng(21)
    a = grid.create_field(mx.center,
                          data=jnp.asarray(rng.normal(size=8)))
    b = grid.create_field(mx.center,
                          data=jnp.asarray(rng.normal(size=8)))
    cond = grid.create_field(
        mx.center, data=jnp.asarray([1., 0., 1., 1., 0., 0., 1., 0.]))
    out = where(cond, a, b)
    assert out.function_space.bare is mx.center
    expected = jnp.where(cond.data, a.data, b.data)
    assert np.array_equal(np.asarray(out.data), np.asarray(expected))


def test_nonzero_condition_entries_select_the_true_branch(where, mx):
    # plain where semantics: any nonzero condition value picks `a`
    grid = Grid((mx,))
    a = grid.create_field(mx.center, data=jnp.full(8, 2.0))
    b = grid.create_field(mx.center, data=jnp.full(8, -3.0))
    cond = grid.create_field(
        mx.center,
        data=jnp.asarray([0.5, -1.0, 0.0, 2.0, 0.0, 1e-8, 0.0, 7.0]))
    out = where(cond, a, b)
    expected = np.where(
        np.asarray(cond.data) != 0.0, 2.0, -3.0)
    assert np.array_equal(np.asarray(out.data), expected)


def test_result_keeps_the_first_branch_metadata(where, mx):
    grid = Grid((mx,))
    a = grid.create_field(mx.center, name="flux", units="m/s")
    b = grid.create_field(mx.center, name="other")
    cond = grid.create_field(mx.center, data=jnp.ones(8))
    assert where(cond, a, b).name == "flux"


def test_operands_must_share_the_grid(where, mx):
    grid_a = Grid((mx,))
    grid_b = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    f = grid_a.create_field(mx.center)
    g = grid_b.create_field(grid_b.factors[0].center)
    with pytest.raises(GridMismatchError):
        where(f, f, g)


def test_where_is_device_count_invariant():
    def compute(device_ids):
        mesh = IntervalMesh(16, (0.0, 1.0), name="x")
        grid = Grid((mesh,), device_ids=device_ids)
        a = grid.create_field(
            mesh.center, init=lambda x: jnp.sin(2.0 * jnp.pi * x))
        b = grid.create_field(mesh.center, init=lambda x: x - 0.5)
        cond = grid.create_field(
            mesh.center, init=lambda x: (x > 0.5).astype(jnp.float64))
        return Where()(cond, a, b).data

    assert np.array_equal(np.asarray(compute(None)),
                          np.asarray(compute((0,))))


# ================================================================
#  Registry rows: ("select", nodal/CellAvg/FaceAvg) -> Where()
# ================================================================
def test_select_rows_are_seeded_on_nodal_and_average_factors(mx):
    grid = Grid((mx,))
    spaces = (mx.center, mx.right, mx.cell_avg, mx.face_avg)
    resolved = {grid.dispatch.resolve("select", space)
                for space in spaces}
    resolved |= {grid.dispatch.resolve("select", space.as_complex())
                 for space in spaces}
    assert len(resolved) == 1  # one shared instance (form-2 rule)
    assert isinstance(next(iter(resolved)), Where)


def test_select_rows_cover_bounded_meshes():
    my = IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")
    grid = Grid((my,))
    for space in (my.center, my.outer, my.inner, my.cell_avg,
                  my.face_avg):
        assert isinstance(grid.dispatch.resolve("select", space),
                          Where)


def test_select_has_no_coefficient_space_row(mx):
    grid = Grid((mx,))
    fourier = mx.fourier(origin=mx.center)
    with pytest.raises(DispatchError):
        grid.dispatch.resolve("select", fourier)
