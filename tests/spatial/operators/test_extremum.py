"""Tests for fridom.spatial.operators.extremum.

The operator-level surface of the ``Maximum`` / ``Minimum``
reductions; the ``f.max`` / ``f.min`` field forwarders are covered
by ``tests/spatial/fields/test_scalar_field_reductions.py``.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.extremum import Maximum, Minimum
from fridom.spatial.space_patterns import Staggered

TWO_PI = 2.0 * np.pi


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture(params=[Maximum, Minimum],
                ids=["maximum", "minimum"])
def op(request):
    return request.param()


# ================================================================
#  Static surface and codomain
# ================================================================
def test_dispatch_kind():
    assert Maximum().dispatch_kind == "amax"
    assert Minimum().dispatch_kind == "amin"


def test_interning():
    assert Maximum() is Maximum()
    assert Minimum() is Minimum()
    assert Maximum() is not Minimum()


def test_requirements(op, mx):
    req = op.requirements(mx.center)
    assert req.halo == 0
    assert req.layout == "any"
    assert req.collective is True


def test_codomain(op, mx, my):
    assert op.codomain(mx.center) is mx.constant
    assert op.codomain(mx.cell_avg) is mx.constant
    assert op.codomain(my.outer) is my.constant
    assert op.codomain(mx.constant) is mx.constant


def test_codomain_rejects_coefficient_factors(op, mx):
    fourier = mx.fourier(origin=mx.center)
    with pytest.raises(SpaceMismatchError, match="transform back"):
        op.codomain(fourier)


def test_codomain_rejects_complex_factors(op, mx):
    with pytest.raises(SpaceMismatchError, match="no order"):
        op.codomain(mx.center.as_complex())


def test_codomain_rejects_non_factor_domains(op, mx, my):
    with pytest.raises(SpaceMismatchError,
                       match="nodal and average"):
        op.codomain(mx.center * my.center)


# ================================================================
#  Values: the reduction is the un-weighted true-DOF extremum
# ================================================================
def test_extremum_on_center(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(
        init=lambda x, y: jnp.sin(TWO_PI * x) + y)
    true = np.asarray(f.xr)
    assert np.allclose(f.max().item(), true.max())
    assert np.allclose(f.min().item(), true.min())


def test_extremum_on_cell_averages(mx, my):
    grid = Grid((mx, my))
    space = mx.cell_avg * my.cell_avg
    f = grid.create_field(space, init=lambda x, y: x - 2.0 * y)
    true = np.asarray(f.xr)
    assert np.allclose(f.max().item(), true.max())
    assert np.allclose(f.min().item(), true.min())


def test_boundary_outer_nodes_are_genuine_dofs(my):
    # the trapezoid outer nodes include the interval ends: an
    # endpoint extremum must win (measure halves, but does not
    # drop, boundary-member dual cells)
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mx, my))
    space = mx.center * my.outer
    f = grid.create_field(space, init=lambda x, y: y + 0.0 * x)
    assert np.allclose(f.max().item(), 2.0)
    assert np.allclose(f.min().item(), 0.0)


def test_masked_rows_never_win(my):
    # all-negative field: the zero-filled halo / padding / claimed
    # wall rows in storage would fake a 0 maximum without the
    # measure-positivity mask
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mx, my))
    v_space = Staggered("y").resolve(grid)
    v = grid.create_field(
        v_space, init=lambda x, y: -1.0 - jnp.sin(np.pi * y / 2.0) + 0.0 * x)
    assert bool((np.asarray(v.storage) == 0.0).any())
    true = np.asarray(v.xr)
    assert v.max().item() == pytest.approx(true.max())
    assert v.max().item() < 0.0
    w = grid.create_field(
        v_space, init=lambda x, y: 1.0 + jnp.sin(np.pi * y / 2.0) + 0.0 * x)
    assert w.min().item() == pytest.approx(np.asarray(w.xr).min())
    assert w.min().item() > 0.0


def test_result_has_default_metadata(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(init=lambda x, y: x + y, name="f",
                          units="m")
    top = f.max()
    assert top.name != "f"
    assert top.metadata.units != "m"
