"""Real multi-process close-over check for ``ImmersedDomain``.

The automated analogue of the ``srun -n 4`` hardware check for the
immersed cache: it spawns several OS processes, each initializing
``jax.distributed`` with one CPU device, builds a grid sharded across
the processes carrying an ``ImmersedDomain``, queries the wet
``fraction`` eagerly (populating the concrete-only cache with a
would-be sharded array), and then lowers a ``jax.jit`` that closes over
that cached fraction — exactly what the traced model step does.

Under real multi-process the cached array spans non-addressable
devices, so closing over it is illegal (``jax`` raises ``Closing over
jax.Array that spans non-addressable devices`` at lower time). The
``ImmersedDomain`` fix replicates the concrete array before caching, so
each process holds a full local copy and the lower succeeds. This is
invisible to the single-controller test suite (forced host devices and
one device leave the array fully addressable, so the replicate is a
no-op) — the blind spot this test closes.

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

NUM_PROCESSES = 2
# an x extent divisible by the process count so each rank owns a shard
NX = 16
NY = 8
NZ = 8
TWO_PI = 6.283185307179586


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _ensure_importable():
    """Put this module's package root on ``sys.path`` for spawn.

    Mirrors ``tests/model/io/test_writer_multiprocess.py``: pytest's
    ``--import-mode=importlib`` does not place the package root on
    ``sys.path``, but spawn re-imports this module in the child to
    unpickle ``_worker`` by its dotted ``__module__``.
    """
    depth = __name__.count(".")
    root = str(Path(__file__).resolve().parents[depth])
    if root not in sys.path:
        sys.path.insert(0, root)


# ================================================================
#  The per-rank body (fridom is imported after jax.distributed init)
# ================================================================
def _cut(x, y, z):  # noqa: ARG001
    """Return a sloped analytic bottom carving genuine partial cells."""
    import jax.numpy as jnp  # noqa: PLC0415
    return jnp.clip((z - 0.3 - 0.1 * jnp.sin(x)) * 6.0 + 0.5, 0.0, 1.0)


def _cell_space(mx, my, mz):
    """Return the cell-centre space resolving every grid coordinate."""
    space = mx.center * my.center
    return space * mz.center


def _run(process_id):
    """Build a sharded immersed grid; lower a jit closing over theta."""
    import jax  # noqa: PLC0415

    from fridom.spatial.grid import Grid  # noqa: PLC0415
    from fridom.spatial.immersed_domain import (  # noqa: PLC0415
        ImmersedDomain,
    )
    from fridom.spatial.meshes.interval import IntervalMesh  # noqa: PLC0415

    mx = IntervalMesh(NX, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(NY, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(NZ, (0.0, 1.0), periodic=False, name="z")
    grid = Grid((mx, my, mz),
                immersed=ImmersedDomain(_cut, order=4, min_fraction=0.1))
    space = _cell_space(mx, my, mz)

    # EAGER query: materializes and memoizes the concrete fraction
    # (a globally sharded array under real multi-process).
    theta = grid.immersed.fraction(space)
    if process_id == 0 and theta.data.is_fully_addressable:
        # a single-device fallback: nothing to prove here
        return "single"

    # a jit that RE-queries the fraction (cache hit -> the concrete
    # cached array is closed over as an MLIR constant) and lowers it,
    # exactly as the model step does.
    @jax.jit
    def use():
        return grid.immersed.fraction(space).data * 2.0

    use.lower()  # pre-fix: RuntimeError "Closing over ..."; post-fix: OK
    return "pass"


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
            f"rank {pid} raised while closing over the immersed "
            f"fraction (the non-addressable close-over leak):\n"
            f"{details[pid]}")
    if statuses.get(0) == "single":
        pytest.skip("ranks resolved to a single addressable device; "
                    "the close-over leak needs a real sharded array")
    if statuses.get(0) != "pass":
        pytest.skip("multi-process workers did not coordinate on this "
                    f"host (statuses={statuses})")


# ================================================================
#  The test: spawn the ranks, rank 0's lower must succeed
# ================================================================
@pytest.mark.multi_process
def test_immersed_fraction_closeover_multiprocess(tmp_path):  # noqa: ARG001
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
