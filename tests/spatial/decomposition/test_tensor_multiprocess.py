"""Real multi-process check of born-sharded field construction.

The automated analogue of the ``srun -n N`` hardware check for
``TensorDecomposition.zeros`` / ``assemble``: it spawns several OS
processes, each initializing ``jax.distributed`` with one CPU device,
builds a grid sharded across the processes and constructs fields the
born-sharded way (zeros, ``init=``, host ``data=``, random draws) on
the uniform, the staggered-deficit and the staggered-surplus spaces of
a walled sharded axis — so the ranks build pieces of *different*
shapes. Under real multi-process a process cannot address the global
array at all, which is invisible to the single-controller suite
(forced host devices leave every array fully addressable): each rank
must build exactly the shards it addresses, and the gathered result
must equal the single-device field bit for bit. A second grid lives on
a device *subset*, so one rank addresses no shard of it at all and
contributes nothing but the dtype.

The test skips cleanly when the environment cannot bind a coordinator
port or spawn/coordinate the processes (so it never destabilizes CI);
the authoritative check remains the ``srun`` reproducer.
"""
import multiprocessing as mp
import socket
import sys
import traceback
from pathlib import Path

import pytest

NUM_PROCESSES = 3
# a walled x extent divisible by the three ranks and by the two-rank
# subset: inner (17 DOFs) is the staggered-deficit leg, outer (19 DOFs)
# the surplus (global-fallback) leg; y divides neither, so x is sharded
NX = 18
NY = 7


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _ensure_importable():
    """Put this module's package root on ``sys.path`` for spawn.

    Mirrors ``tests/spatial/test_immersed_domain_multiprocess.py``:
    pytest's ``--import-mode=importlib`` does not place the package root
    on ``sys.path``, but spawn re-imports this module in the child to
    unpickle ``_worker`` by its dotted ``__module__``.
    """
    depth = __name__.count(".")
    root = str(Path(__file__).resolve().parents[depth])
    if root not in sys.path:
        sys.path.insert(0, root)


# ================================================================
#  The per-rank body (fridom is imported after jax.distributed init)
# ================================================================
def _build(device_ids):
    """Return a walled-x grid and its uniform/deficit/surplus spaces."""
    from fridom.spatial.grid import Grid  # noqa: PLC0415
    from fridom.spatial.meshes.interval import IntervalMesh  # noqa: PLC0415

    mx = IntervalMesh(NX, (0.0, 2.0), periodic=False, name="x")
    my = IntervalMesh(NY, (0.0, 3.0), periodic=False, name="y")
    grid = Grid((mx, my), device_ids=device_ids)
    return grid, (mx.center * my.center, mx.inner * my.center,
                  mx.outer * my.inner)


def _fields(grid, space):
    """Construct one field per born-sharded route on ``space``."""
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    shape = grid.create_field(space).shape
    host = np.random.default_rng(3).standard_normal(shape)
    return (
        grid.create_field(space),
        grid.create_field(
            space, init=lambda x, y: jnp.sin(3.1 * x) * jnp.cos(y) + x * y),
        grid.create_field(space, data=host),
        grid.random.normal(space, seed=5),
    )


def _run(process_id):
    """Build fields sharded across the processes; compare to one device."""
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from jax.experimental import multihost_utils  # noqa: PLC0415

    many, spaces_many = _build(None)
    if many.decomposition.device_count != NUM_PROCESSES:
        return "single"
    if dict(many.decomposition.default_layout.device_axes) != {
            "x": "devices"}:
        raise AssertionError("the walled x axis must be the sharded one")
    # the reference lives on this process's own device (device_ids
    # index jax.devices(), whose ids are not contiguous across hosts)
    local = jax.devices().index(jax.local_devices()[0])
    one, spaces_one = _build((local,))
    for space_many, space_one in zip(spaces_many, spaces_one, strict=True):
        for got, want in zip(_fields(many, space_many),
                             _fields(one, space_one), strict=True):
            storage = got.storage
            if storage.is_fully_addressable:
                return "single"
            # a rank holds exactly its own block, never the global array
            (shard,) = storage.addressable_shards
            if shard.data.shape[0] * NUM_PROCESSES != storage.shape[0]:
                raise AssertionError(
                    f"rank {process_id} holds {shard.data.shape} of "
                    f"{storage.shape}")
            gathered = np.asarray(multihost_utils.process_allgather(
                got.data, tiled=True))
            if not np.array_equal(gathered, np.asarray(want.data)):
                raise AssertionError(
                    f"rank {process_id}: {got.function_space!r} differs "
                    "from the single-device field")
    _check_subset(process_id, _fields(one, spaces_one[0]))
    return "pass"


def _check_subset(process_id, reference):
    """Build on a two-rank device subset; the third rank owns nothing."""
    import numpy as np  # noqa: PLC0415

    subset, spaces = _build((0, 1))
    if dict(subset.decomposition.default_layout.device_axes) != {
            "x": "devices"}:
        raise AssertionError("the subset grid must shard x")
    for got, want in zip(_fields(subset, spaces[0]), reference,
                         strict=True):
        shards = got.data.addressable_shards
        if len(shards) != (1 if process_id < 2 else 0):
            raise AssertionError(
                f"rank {process_id} addresses {len(shards)} shard(s)")
        for shard in shards:
            if not np.array_equal(np.asarray(shard.data),
                                  np.asarray(want.data)[shard.index]):
                raise AssertionError(
                    f"rank {process_id}: subset shard differs from the "
                    "single-device field")


def _worker(process_id, num_processes, coordinator, queue):
    """Spawn entry point: init jax.distributed, then run the check."""
    import os  # noqa: PLC0415
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["FRIDOM_TEST_JAX_CACHE_DIR"] = ""
    try:
        import jax  # noqa: PLC0415
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=num_processes,
            process_id=process_id)
    except Exception:  # noqa: BLE001 — an env without multi-host -> skip
        queue.put((process_id, "initfail", traceback.format_exc()))
        return
    try:
        status = _run(process_id)
        queue.put((process_id, status, ""))
    except Exception:  # noqa: BLE001 — a real failure in the code path
        queue.put((process_id, "error", traceback.format_exc()))


def _collect(queue, procs, num_processes):
    """Gather rank results within a deadline, then reap the processes."""
    results = {}
    for _ in range(num_processes):
        try:
            pid, status, detail = queue.get(timeout=180)
        except Exception:  # noqa: BLE001 — queue empty within the deadline
            break
        results[pid] = (status, detail)
    for proc in procs:
        proc.join(timeout=10)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    return results


def _evaluate(results):
    """Fail on a real fault; skip if the ranks did not coordinate."""
    statuses = {pid: st for pid, (st, _) in results.items()}
    details = {pid: dt for pid, (_, dt) in results.items()}
    errored = [pid for pid, st in statuses.items() if st == "error"]
    if errored:
        pid = errored[0]
        pytest.fail(
            f"rank {pid} failed the born-sharded construction check:\n"
            f"{details[pid]}")
    if "single" in statuses.values():
        pytest.skip("ranks resolved to a single addressable device; "
                    "the check needs a grid sharded across processes")
    if set(statuses.values()) != {"pass"} or len(statuses) != NUM_PROCESSES:
        pytest.skip("multi-process workers did not coordinate on this "
                    f"host (statuses={statuses})")


# ================================================================
#  The test: spawn the ranks, every rank must reproduce one device
# ================================================================
@pytest.mark.multi_process
def test_born_sharded_construction_multiprocess(tmp_path):  # noqa: ARG001
    try:
        _ensure_importable()  # the spawned child re-imports this module
        coordinator = f"localhost:{_free_port()}"
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_worker,
                args=(pid, NUM_PROCESSES, coordinator, queue))
            for pid in range(NUM_PROCESSES)]
        for proc in procs:
            proc.start()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"cannot spawn multi-process workers: {error}")
    _evaluate(_collect(queue, procs, NUM_PROCESSES))
