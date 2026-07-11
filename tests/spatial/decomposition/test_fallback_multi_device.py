"""Multi-device forced-4 gate for the graded ``Fallback`` operator.

Stage F6, plan ``design/plans/active/fallback_operator_plan.md`` section 8:
drop ``layout="local"`` -- make the graded reconstruction bitwise-correct
when the BOUNDED axis it grades is genuinely SHARDED across devices. The
wall rows are patched behind the decomposition seam
``patch_physical_ends`` (a ``shard_map`` + ``axis_index`` mask mirroring
``_exchange_block``), so ``Fallback.requirements`` now declares
``layout="any"`` and negotiation is free to distribute the bounded axis.

Lives in the decomposition suite because the forced-4 CI job
(``.github/workflows/tests.yml:35-42``) only globs
``tests/spatial/decomposition`` -- a forced-4 gate must live
here to be run genuinely sharded (four host devices via
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` +
``FRIDOM_TEST_FORCED_DEVICES=4``).

Reuses the shared ``forced_devices`` fixture
(``tests/conftest.py``) and the None-vs-(0,)
device-count-invariance convention from
``test_multi_device.py:38-44`` / ``test_weno.py:459-478``.
"""
import jax
import jax.numpy as jnp
import numpy as np

from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.fallback import graded_reconstruction


# ================================================================
#  A genuinely-sharded bounded axis (1-D bounded IntervalMesh)
# ================================================================
def _graded_on_sharded_bounded_axis(device_ids):
    # 16 cells over 4 forced devices => 4 cells/shard; WENO-5 halo 3
    # fits (cells >= width + 1 = 4), so the bounded axis is genuinely
    # distributed -- the two physical-wall shards each patch their K
    # reduced faces, interior shards run the wide pass unpatched.
    mesh = IntervalMesh(16, (0.0, 1.0), periodic=False, name="y")
    grid = Grid((mesh,), device_ids=device_ids)
    grid.negotiate(halo=HaloSpec({"y": 3}))
    f = grid.create_field(
        mesh.cell_avg,
        init=lambda y: jnp.sin(2.0 * jnp.pi * y))
    g = graded_reconstruction(5, "left")["y"](f)
    return grid, f, g


def test_graded_sharded_bounded_axis_is_device_count_invariant(
        forced_devices):
    if forced_devices is not None:
        # the forced suite must genuinely see the devices (fail, not
        # skip, if the XLA flag did not take effect)
        assert jax.device_count() == forced_devices

    grid_many, _, g_many = _graded_on_sharded_bounded_axis(None)
    _, _, g_one = _graded_on_sharded_bounded_axis((0,))

    # the decisive precondition: with >1 device the bounded wall axis
    # is GENUINELY distributed (F6 dropped the local requirement)
    if forced_devices and forced_devices > 1:
        assert not grid_many.decomposition.default_layout.is_local("y")

    # finite everywhere and bitwise device-count invariant, both
    # per-shard (.data) and after gathering the many-device run
    assert bool(jnp.all(jnp.isfinite(g_many.data)))
    assert np.array_equal(np.asarray(g_many.data),
                          np.asarray(g_one.data))
    gathered = grid_many.decomposition.gather(
        g_many._data, g_many.function_space)
    assert np.array_equal(np.asarray(gathered),
                          np.asarray(g_one.data))


def test_graded_sharded_bounded_axis_reads_interior_only(forced_devices):
    # NaN-poison PHYSICAL-EXTERIOR-halo gate UNDER SHARDING: poison
    # only the two physical-wall exterior halos (shard 0's leading
    # ghosts, the last shard's trailing ghosts) and claim the ghosts
    # valid so the consumption-side sync leaves them. Interior
    # shard-boundary ghosts are left intact (the legitimate wide
    # interior pass reads them). The graded output must stay finite --
    # proving the two wall shards' patches read only interior (true)
    # cells, never the poisoned exterior halo.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    grid, f, _ = _graded_on_sharded_bounded_axis(None)
    width = grid.decomposition.halo["y"]
    storage = f._data
    total = storage.shape[0]
    devices = jax.device_count()
    mask = np.zeros(total, dtype=bool)
    if grid.decomposition.default_layout.is_local("y"):
        # single block (1 device): the two ends ARE the exterior halos
        mask[:width] = True
        mask[total - width:] = True
    else:
        # blocked: only shard 0's leading ghosts and the last shard's
        # trailing (ghost + stagger) slots are physical-exterior halos
        block = total // devices
        cells = grid.factors[0].n_cells // devices
        mask[:width] = True                              # left wall
        last = (devices - 1) * block
        mask[last + width + cells:last + block] = True   # right wall
    poisoned = jnp.where(jnp.asarray(mask), jnp.nan, storage)
    f._data = poisoned
    f._halo_valid = HaloSpec({"y": width})

    g = graded_reconstruction(5, "left")["y"](f)
    assert bool(jnp.all(jnp.isfinite(g.data)))


# ================================================================
#  Kept: the local-bounded-axis path (bounded axis stays on device
#  because the periodic x axis absorbs the sharding) is still valid.
# ================================================================
def _graded_on_local_bounded_axis(device_ids):
    mx = IntervalMesh(16, (0.0, 1.0), name="x")                   # periodic
    my = IntervalMesh(16, (0.0, 1.0), periodic=False, name="y")   # walls
    grid = Grid((mx, my), device_ids=device_ids)
    # the default layout shards x (first GHOST-shardable name), so the
    # bounded y axis stays device-local and the graded pass runs the
    # single-block wall patch
    grid.negotiate(halo=HaloSpec({"x": 3, "y": 3}))
    space = mx.cell_avg * my.cell_avg
    f = grid.create_field(
        space,
        init=lambda x, y: jnp.sin(2.0 * jnp.pi * x)
        + jnp.sin(2.0 * jnp.pi * y))
    g = graded_reconstruction(5, "left")["y"](f)
    return grid, g


def test_graded_local_bounded_axis_is_device_count_invariant(
        forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    grid_many, g_many = _graded_on_local_bounded_axis(None)
    _, g_one = _graded_on_local_bounded_axis((0,))

    # y kept undistributed in the default layout; the graded pass
    # matches the one-device run bitwise, per-shard and gathered
    assert grid_many.decomposition.default_layout.is_local("y")
    assert np.array_equal(np.asarray(g_many.data),
                          np.asarray(g_one.data))
    gathered = grid_many.decomposition.gather(
        g_many._data, g_many.function_space)
    assert np.array_equal(np.asarray(gathered),
                          np.asarray(g_one.data))
