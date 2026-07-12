"""Multi-device forced-4 gate for the NODAL graded near-wall closure.

The twin of ``test_fallback_multi_device.py`` for the *nodal* C-grid
family (the walled biased advection schemes of ``nonhydro2``): both
drive the shared ``operators.graded`` wall patch behind the
decomposition seam ``patch_physical_ends``, so both must be bitwise
device-count invariant when the BOUNDED axis they grade is genuinely
SHARDED across devices.

The nodal family adds the ``shift = 1`` cell frame the FV family never
exercises: the dual ``Inner -> Center`` direction, whose lattice cells
are the mesh faces and whose two wall cells are the operand's
homogeneous-Dirichlet boundary values. Its right-wall block indexes
those cells off the LOCAL true count, which only lines up because the
decomposition's last shard absorbs the staggered true-count deficit —
exactly the invariant a forced-4 run pins.

Lives in the decomposition suite because the forced-4 CI job
(``.github/workflows/tests.yml``) only globs
``tests/spatial/decomposition`` (see ``test_fallback_multi_device.py``).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.nonhydro2.modules.advection import (
    _BiasedFaceReconstruction,
)
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.nodal import NodeSet

ORDER = 5
WIDTH = ORDER // 2 + 1


def _smooth(y):
    # vanishes at both walls, so the Dirichlet Inner operand is legal
    return jnp.sin(jnp.pi * y) * (1.0 + 0.4 * jnp.cos(3 * jnp.pi * y))


def _graded_on_sharded_bounded_axis(device_ids, node_set, bias,
                                    weighting):
    # 16 cells over 4 forced devices => 4 cells/shard; the order-5 halo
    # (3) fits, so the bounded axis is genuinely distributed and the two
    # physical-wall shards each patch their K reduced faces.
    mesh = IntervalMesh(16, (0.0, 1.0), periodic=False, name="y")
    grid = Grid((mesh,), device_ids=device_ids)
    grid.negotiate(halo=HaloSpec({"y": WIDTH}))
    src = (mesh.center if node_set is NodeSet.CENTER
           else mesh.nodal(NodeSet.INNER, bc=BC.DIRICHLET))
    f = grid.create_field(src, init=_smooth)
    op = _BiasedFaceReconstruction(ORDER, bias, weighting, "graded")
    return grid, f, op["y"](f)


@pytest.mark.parametrize("weighting", ["linear", "weno"])
@pytest.mark.parametrize("bias", ["left", "right"])
@pytest.mark.parametrize("node_set", [NodeSet.CENTER, NodeSet.INNER])
def test_graded_nodal_rows_are_device_count_invariant(
        forced_devices, node_set, bias, weighting):
    if forced_devices is not None:
        # the forced suite must genuinely see the devices (fail, not
        # skip, if the XLA flag did not take effect)
        assert jax.device_count() == forced_devices

    grid_many, _, g_many = _graded_on_sharded_bounded_axis(
        None, node_set, bias, weighting)
    _, _, g_one = _graded_on_sharded_bounded_axis(
        (0,), node_set, bias, weighting)

    if forced_devices and forced_devices > 1:
        # the decisive precondition: the bounded wall axis is genuinely
        # distributed (nothing pins it local)
        assert not grid_many.decomposition.default_layout.is_local("y")

    assert bool(jnp.all(jnp.isfinite(g_many.data)))
    assert np.array_equal(np.asarray(g_many.data),
                          np.asarray(g_one.data))
    gathered = grid_many.decomposition.gather(
        g_many._data, g_many.function_space)
    assert np.array_equal(np.asarray(gathered),
                          np.asarray(g_one.data))


@pytest.mark.parametrize("node_set", [NodeSet.CENTER, NodeSet.INNER])
def test_graded_nodal_rows_read_interior_only_under_sharding(
        forced_devices, node_set):
    # NaN-poison PHYSICAL-EXTERIOR-halo gate UNDER SHARDING: poison only
    # the two physical-wall halos (shard 0's leading ghosts, the last
    # shard's trailing ghost + stagger slots) and claim the ghosts valid
    # so the consumption-side sync leaves them. Interior shard-boundary
    # ghosts stay intact (the legitimate wide interior pass reads them).
    # The graded output must stay finite -- the two wall shards' patches
    # read only true cells and SYNTHESIZE the exact-zero wall values.
    if forced_devices is not None:
        assert jax.device_count() == forced_devices

    grid, f, _ = _graded_on_sharded_bounded_axis(
        None, node_set, "left", "linear")
    width = grid.decomposition.halo["y"]
    storage = f._data
    total = storage.shape[0]
    devices = jax.device_count()
    mask = np.zeros(total, dtype=bool)
    if grid.decomposition.default_layout.is_local("y"):
        mask[:width] = True
        mask[total - width:] = True
    else:
        block = total // devices
        cells = f.function_space.bare.factor("y").shape[0] // devices
        mask[:width] = True                              # left wall
        last = (devices - 1) * block
        mask[last + width + cells:last + block] = True   # right wall
    f._data = jnp.where(jnp.asarray(mask), jnp.nan, storage)
    f._halo_valid = HaloSpec({"y": width})

    op = _BiasedFaceReconstruction(ORDER, "left", "linear", "graded")
    assert bool(jnp.all(jnp.isfinite(op["y"](f).data)))
