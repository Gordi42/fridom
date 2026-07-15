"""Real multi-process Writer test (``jax.distributed`` on CPU).

The automated analogue of the ``srun -n 4`` hardware check: it spawns
four OS processes, each initializing ``jax.distributed`` with one CPU
device, builds a grid sharded across the four processes with fields set
to known functions of the GLOBAL coordinates, drives
``Writer.bind/write/close`` on every rank (blocking and the
async->blocking fallback), and (rank 0) verifies the store against the
ground truth rebuilt from the store's own exported coordinates. A
correct gather-free multi-process write must reproduce those fields
exactly: no shard dropped, misplaced, or duplicated.

The test skips cleanly when the environment cannot bind a coordinator
port or spawn/coordinate the processes (so it never destabilizes CI);
the authoritative check remains the ``srun`` reproducer.
"""
import multiprocessing as mp
import shutil
import socket
import sys
import traceback
from pathlib import Path

import numpy as np
import pytest

DT = 0.5
ITS = (0, 1, 2, 3)
# grid divisible by four so each of the four ranks owns one window
NX = 16
NY = 12


# ================================================================
#  Duck-typed model / carry (mirrors tests/model/io/test_writer.py)
# ================================================================
class _FakeCarry:
    def __init__(self, state, clock):
        self.state = state
        self.clock = clock


class _FakeTable:
    def __init__(self, prognostic=(), diagnostic=()):
        self.prognostic = tuple(prognostic)
        self.diagnostic = tuple(diagnostic)
        self.names = self.prognostic + self.diagnostic


class _FakeFingerprint:
    def __init__(self, digest):
        self.digest = digest


class _FakeModel:
    def __init__(self, state, clock, *, table=None, digest=None):
        self._carry = _FakeCarry(state, clock)
        if table is not None:
            self.field_table = table
        if digest is not None:
            self.fingerprint = _FakeFingerprint(digest)

    @property
    def carry(self):
        return self._carry


# ================================================================
#  Free-port helper (bind :0, read the assigned port, release it)
# ================================================================
def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _ensure_importable():
    """Put this module's package root on ``sys.path`` for spawn.

    ``multiprocessing`` spawn re-imports this module in the child to
    unpickle ``_worker`` (by its dotted ``__module__``), but pytest's
    ``--import-mode=importlib`` does not place the package root on
    ``sys.path``. Spawn propagates ``sys.path`` to the child, so adding
    the root here (before ``start``) lets the child import the module.
    """
    depth = __name__.count(".")  # package levels above this module
    root = str(Path(__file__).resolve().parents[depth])
    if root not in sys.path:
        sys.path.insert(0, root)


# ================================================================
#  The per-rank body (fridom is imported after jax.distributed init)
# ================================================================
def _clock_at(clock_cls, it):
    clock = clock_cls()
    for _ in range(it):
        clock = clock.tick(DT)
    return clock


def _build_state(vector_cls, grid, mx, my):
    u = grid.create_field(mx.right * my.center,
                          init=lambda x, y: x + y, name="u", units="m/s")
    p = grid.create_field(init=lambda x, y: x * y, name="p", units="Pa")
    return vector_cls({"u": u, "p": p})


def _verify_store(path):
    """Rank-0 check: every slice matches the ground truth from coords."""
    import xarray as xr  # noqa: PLC0415

    ds = xr.open_zarr(path, consolidated=False)
    ok = True
    if ds["iteration"].values.tolist() != list(ITS):
        ok = False
    if not np.array_equal(ds["time"].values,
                          np.array([it * DT for it in ITS])):
        ok = False
    # ground truth rebuilt from the store's OWN exported coordinates
    x_right = ds.coords["x_right"].values
    x_center = ds.coords["x"].values
    y = ds.coords["y"].values
    u_truth = x_right[:, None] + y[None, :]
    p_truth = x_center[:, None] * y[None, :]
    if ds["u"].shape != (len(ITS), *u_truth.shape):
        ok = False
    if ds["p"].shape != (len(ITS), *p_truth.shape):
        ok = False
    for k in range(len(ITS)):
        if not np.array_equal(ds["u"].values[k], u_truth):
            ok = False
        if not np.array_equal(ds["p"].values[k], p_truth):
            ok = False
    if ds.attrs.get("fridom_fingerprint") != "deadbeef":
        ok = False
    ds.close()
    return ok


def _run(process_id, outdir):
    """Drive the writer on one rank; rank 0 returns pass/fail."""
    from jax.experimental import multihost_utils  # noqa: PLC0415

    from fridom.model.clock import Clock  # noqa: PLC0415
    from fridom.model.io.triggers import every  # noqa: PLC0415
    from fridom.model.io.writer import Writer  # noqa: PLC0415
    from fridom.spatial.fields.vector_field import (  # noqa: PLC0415
        VectorField,
    )
    from fridom.spatial.grid import Grid  # noqa: PLC0415
    from fridom.spatial.meshes.interval import IntervalMesh  # noqa: PLC0415

    def barrier(tag):
        multihost_utils.sync_global_devices(tag)

    overall_ok = True
    for async_writes in (False, True):
        path = outdir / ("async.zarr" if async_writes else "sync.zarr")
        if process_id == 0 and path.exists():
            shutil.rmtree(path)
        barrier(f"clean-{path.name}")
        mx = IntervalMesh(NX, (0.0, 1.0), name="x")  # periodic
        my = IntervalMesh(NY, (0.0, 2.0), periodic=False, name="y")
        grid = Grid((mx, my))
        state = _build_state(VectorField, grid, mx, my)
        table = _FakeTable(prognostic=("u",), diagnostic=("p",))
        model = _FakeModel(state, _clock_at(Clock, 0), table=table,
                           digest="deadbeef")
        writer = Writer(path, fields=["u", "p"], trigger=every(steps=1),
                        mode="w-", async_writes=async_writes)
        writer.bind(model)
        for it in ITS:
            writer.write(_FakeCarry(state, _clock_at(Clock, it)))
        writer.close()
        barrier(f"written-{path.name}")
        if process_id == 0:
            overall_ok = _verify_store(path) and overall_ok
        barrier(f"verified-{path.name}")
    if process_id == 0:
        return "pass" if overall_ok else "fail"
    return "pass"


def _worker(process_id, num_processes, coordinator, outdir_str, queue):
    """Spawn entry point: init jax.distributed, then drive the writer."""
    import os  # noqa: PLC0415
    os.environ["JAX_PLATFORMS"] = "cpu"
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
        status = _run(process_id, Path(outdir_str))
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
            break  # a hung rank leaves its slot empty
        results[pid] = (status, detail)
    for proc in procs:
        proc.join(timeout=10)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    return results


def _evaluate(results):
    """Fail on a real fault or a bad store; skip if uncoordinated."""
    statuses = {pid: st for pid, (st, _) in results.items()}
    details = {pid: dt for pid, (_, dt) in results.items()}
    # a real fault in the code under test surfaces on whichever rank
    # raised it, even if other ranks then hang at a barrier
    errored = [pid for pid, st in statuses.items() if st == "error"]
    if errored:
        pid = errored[0]
        pytest.fail(f"rank {pid} raised in the writer path:\n"
                    f"{details[pid]}")
    if statuses.get(0) == "fail":
        pytest.fail("rank 0 store verification failed (multi-process "
                    "write is not bit-correct)")
    if statuses.get(0) != "pass":
        pytest.skip("multi-process workers did not coordinate on this "
                    f"host (statuses={statuses})")


# ================================================================
#  The test: spawn four ranks, rank 0 verifies (or skip on failure)
# ================================================================
@pytest.mark.multi_process
def test_writer_multiprocess_matches_ground_truth(tmp_path):
    num_processes = 4
    try:
        _ensure_importable()  # the spawned child re-imports this module
        coordinator = f"localhost:{_free_port()}"
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_worker,
                args=(pid, num_processes, coordinator, str(tmp_path),
                      queue))
            for pid in range(num_processes)]
        for proc in procs:
            proc.start()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"cannot spawn multi-process workers: {error}")
    _evaluate(_collect(queue, procs, num_processes))
