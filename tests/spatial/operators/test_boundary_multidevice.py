"""Multi-device tests for the boundary operators (forced-4 suite).

Compares an auto-negotiated (genuinely sharded under
``XLA_FLAGS=--xla_force_host_platform_device_count=4
FRIDOM_TEST_FORCED_DEVICES=4``) grid against an explicit one-device
grid: trace / embed / scatter with the traced axis LOCAL and the
horizontals sharded, and — the reshard-local path — with the traced
axis itself sharded. Pointwise selections / scatters stay bitwise.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.verbs import scatter_add, scatter_set
from fridom.spatial.spaces.trace import Side


def bitwise(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def _build(device_ids):
    # x periodic (sharded by default), y periodic, z bounded (local)
    mx = IntervalMesh(16, (0.0, 1.0), name="x")
    my = IntervalMesh(16, (0.0, 2.0), name="y")
    mz = IntervalMesh(16, (0.0, 3.0), periodic=False, name="z")
    return Grid((mx, my, mz), device_ids=device_ids)


def _cell_space(grid):
    mx, my, mz = grid.factors
    return mx.center * my.center * mz.center


def _init(x, y, z):
    return jnp.sin(2.0 * jnp.pi * x) + jnp.cos(y) + z * x


@pytest.fixture
def grids(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    return _build(None), _build((0,))


def _gathered(field, grid):
    return np.asarray(
        grid.decomposition.gather(field._data, field.function_space))


# ================================================================
#  Traced axis LOCAL, horizontals sharded (the H7 layout)
# ================================================================
def test_trace_is_device_count_invariant(grids):
    many, one = grids
    f_many = many.create_field(_cell_space(many), init=_init)
    f_one = one.create_field(_cell_space(one), init=_init)
    for side in (Side.LOW, Side.HIGH):
        assert bitwise(_gathered(f_many.trace("z", side), many),
                       f_one.trace("z", side).data)


@pytest.mark.multi_device
def test_trace_keeps_the_horizontal_sharding(grids):
    many, _one = grids
    f_many = many.create_field(_cell_space(many), init=_init)
    # the collapsed trace keeps the parent's horizontal sharding
    assert f_many.trace("z", Side.HIGH)._data.sharding.spec[0] == "devices"


def test_embed_is_device_count_invariant(grids):
    many, one = grids
    f_many = many.create_field(_cell_space(many), init=_init)
    f_one = one.create_field(_cell_space(one), init=_init)
    a = f_many.trace("z", Side.HIGH).embed("z")
    b = f_one.trace("z", Side.HIGH).embed("z")
    assert bitwise(_gathered(a, many), b.data)


def test_scatter_is_device_count_invariant(grids):
    many, one = grids

    def compute(grid):
        f = grid.create_field(_cell_space(grid), init=_init)
        base = grid.create_field(_cell_space(grid),
                                 init=lambda x, y, z: 1.0 + x + y + z)
        t = f.trace("z", Side.HIGH)
        return scatter_add(base, t * 2.0), scatter_set(base, t)

    for a, b in zip(compute(many), compute(one), strict=True):
        assert bitwise(_gathered(a, many), b.data)


# ================================================================
#  Traced axis SHARDED: the reshard-local path
# ================================================================
@pytest.mark.multi_device
def test_trace_reshards_a_sharded_axis_local(grids):
    many, one = grids
    # move the devices onto the horizontals so z becomes sharded
    z_sharded = many.decomposition.layout_for(("x", "y"))
    assert not z_sharded.is_local("z")
    f_many = many.create_field(
        _cell_space(many), init=_init).reshard(z_sharded)
    f_one = one.create_field(_cell_space(one), init=_init)
    for side in (Side.LOW, Side.HIGH):
        traced = f_many.trace("z", side)
        # the collapsed result is handed back in the operand's layout
        assert traced.function_space.layout == z_sharded
        assert bitwise(_gathered(traced, many),
                       f_one.trace("z", side).data)
