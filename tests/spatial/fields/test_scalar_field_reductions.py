"""Tests for fridom.spatial.fields.scalar_field — max/min reductions.

Prefix-mirrored shard of the oversized ``test_scalar_field.py``
(testing-policy exception): the ``f.max(*names)`` / ``f.min(*names)``
forwarders onto the seeded ``("amax"/"amin", factor)`` rows.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodeSet

TWO_PI = 2.0 * np.pi


@pytest.fixture
def grid():
    return Grid((IntervalMesh(8, (0.0, 1.0), name="x"),
                 IntervalMesh(6, (0.0, 2.0), periodic=False,
                              name="y")))


@pytest.fixture
def f(grid):
    return grid.create_field(
        init=lambda x, y: jnp.sin(TWO_PI * x) * (1.0 + y))


# ================================================================
#  Values
# ================================================================
def test_full_reduction_matches_numpy(f):
    true = np.asarray(f.xr)
    assert f.max().item() == pytest.approx(true.max())
    assert f.min().item() == pytest.approx(true.min())


def test_partial_reduction_matches_numpy(f):
    true = np.asarray(f.xr)  # dims (x, y)
    assert np.allclose(np.asarray(f.max("x").data).ravel(),
                       true.max(axis=0))
    assert np.allclose(np.asarray(f.min("y").data).ravel(),
                       true.min(axis=1))


def test_sequential_equals_full(f):
    assert f.max("x").max("y").item() == pytest.approx(
        f.max().item())


def test_min_is_negated_max(f):
    assert f.min().item() == pytest.approx(-((-f).max().item()))


def test_abs_max_chain(f):
    assert abs(f).max().item() == pytest.approx(
        np.abs(np.asarray(f.xr)).max())


# ================================================================
#  Spaces and the strict algebra
# ================================================================
def test_reduction_lands_on_constant_factors(f):
    top = f.max()
    assert all(isinstance(factor, ConstantSpace)
               for factor in top.function_space.bare.factors)
    part = f.max("y")
    assert isinstance(part.function_space.bare.factor("y"),
                      ConstantSpace)
    assert not isinstance(part.function_space.bare.factor("x"),
                          ConstantSpace)


def test_reduction_along_a_constant_factor_is_identity(f):
    part = f.max("y")
    again = part.max("y")
    assert again.function_space == part.function_space
    assert np.allclose(np.asarray(again.data),
                       np.asarray(part.data))


def test_broadcast_after_reduce(f):
    normalized = f / f.max()
    assert normalized.function_space == f.function_space
    assert np.asarray(normalized.xr).max() == pytest.approx(1.0)
    shifted = f - f.min("x")
    assert shifted.function_space == f.function_space
    assert np.asarray(shifted.xr).min() >= -1e-12


def test_result_has_default_metadata(grid):
    f = grid.create_field(init=lambda x, y: x + y, name="f")
    assert f.max().name != "f"


# ================================================================
#  Errors
# ================================================================
def test_unknown_name_raises(f):
    with pytest.raises(ValueError, match="z"):
        f.max("z")


def test_complex_field_raises(grid):
    f = grid.create_field(
        init=lambda x, y: x + y).as_complex()
    with pytest.raises(TypeError, match="no order"):
        f.max()
    with pytest.raises(TypeError, match="no order"):
        f.min()


def test_coefficient_space_raises(grid):
    # a real (cosine) coefficient factor reaches the dispatch guard
    # (a Fourier factor would trip the complex TypeError first)
    my = grid.factors[1]
    origin = my.nodal(NodeSet.CENTER, bc=BC.NEUMANN)
    space = grid.factors[0].center * my.cosine(origin=origin)
    f = grid.create_field(space)
    with pytest.raises(DispatchError, match="transform back"):
        f.max("y")


# ================================================================
#  Sharded axes (must pass on any device count) and autodiff
# ================================================================
def test_reduction_along_the_decomposed_axis(grid):
    # the leading axis carries the default decomposition; the
    # reduction must be correct however many devices back the array
    f = grid.create_field(init=lambda x, y: jnp.cos(TWO_PI * x) - y)
    assert f.max("x").function_space == f.max(
        "x").function_space  # smoke: stable resolve
    assert np.allclose(np.asarray(f.max("x").data).ravel(),
                       np.asarray(f.xr).max(axis=0))
    assert f.max().item() == pytest.approx(np.asarray(f.xr).max())


def test_grad_is_one_hot_at_the_argmax(grid):
    # unique argmax: cotangent lands on exactly that entry (and on
    # no halo copy of it), matching the central finite difference
    f = grid.create_field(
        init=lambda x, y: jnp.sin(1.3 + TWO_PI * x) * (1.0 + y))

    def loss(data):
        return f.with_data(data).max().data.sum()

    grad = np.asarray(jax.grad(loss)(f.data))
    assert np.isfinite(grad).all()
    assert (grad != 0.0).sum() == 1
    assert grad.sum() == pytest.approx(1.0)
    # central finite difference at the argmax entry
    eps = 1e-4
    hot = np.unravel_index(np.argmax(grad), grad.shape)
    up = np.asarray(f.data).copy()
    up[hot] += eps
    down = np.asarray(f.data).copy()
    down[hot] -= eps
    fd = (loss(jnp.asarray(up)) - loss(jnp.asarray(down))) / (2 * eps)
    assert float(fd) == pytest.approx(float(grad[hot]), rel=1e-4)
