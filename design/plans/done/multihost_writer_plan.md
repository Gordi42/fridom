# Multi-host writer plan

> **Shipped 2026-07-15** (`dev` merge `8b6642bf feat/multihost-writer`,
> commits `d445dd6f` writer coordination + `c0f9cbb2` `export_layout`
> label gather). Both gaps closed exactly as designed below: the
> `_process_index`/`_process_count`/`_barrier` seams and the
> rank-0-owns-metadata / per-rank-disjoint-shard write path landed in
> `writer.py`, and `_host_labels` (the conditional `process_allgather`)
> landed in `export.py`. `async_writes` falls back to blocking under
> `process_count() > 1` (v1); async multi-host stays a follow-up.
> Single-process output is bit-identical (the whole existing
> `test_writer.py` suite stays green, plus the new monkeypatched
> multi-process seam tests and the registered `multi_process` marker).
> The `srun -n 4` hardware recipe is in AGENTS.md.

Make `fr.io.Writer` (`src/fridom/model/io/writer.py`) correct under a
real multi-process run (`srun -n N` + `jax.distributed.initialize()`,
one device per process). Today it works only single-controller (one
process addressing every device via GSPMD); it was never wired for the
multi-host case that the spec flags as a follow-up (io_ops.md:434,
672-673). The gather-free per-shard write core is already correct — the
gaps are the *bind* coordinate fetch and the *store creation / metadata*
coordination.

## Empirical findings (2026-07-15, real 4x A100, srun -n 4)

Test harness: session scratchpad `writer_dist_test.py` — a 2-D Grid
sharded across all processes, fields set to known functions of the
GLOBAL coordinates, driven through `Writer.bind/write/close`, then
rank-0 reopens the store and checks every slice against ground truth
rebuilt from the store's own exported coords.

- **Sharding is correct.** Each rank owns one distinct global window
  (`P('devices', None)`, `addressable_shards == 1`,
  windows `0:32 / 32:64 / 64:96 / 96:128` for 128 over 4).
  `decomposition.shard_writes` (walks `addressable_shards`, dedups
  `replica_id != 0`) gives each rank exactly its process-local tiles.

- **Gap 1 — `bind()` host-fetches the distributed coords.**
  `Writer.bind -> _layout -> export_layout` at `spatial/export.py:240`
  does `np.asarray(vector.data)` on the distributed 1-D coordinate node
  vector (`grid.evaluation_nodes`) ->
  `RuntimeError: Fetching value for jax.Array that spans non-addressable
  devices`. The layout is values-free w.r.t. the field (it reads only
  `field._data.dtype`); only the coordinate *labels* need materializing.

- **Gap 2 — store creation is uncoordinated.** Shim past gap 1 and all
  ranks race on `_open_store -> _write_coords -> _create_array`
  (`create=True, delete_existing=True`) -> `ValueError: ALREADY_EXISTS`
  writing `.../x_right/.zarray`. No rank-0 guard, no barrier between
  skeleton-create and shard-write; `_write_json` is a non-atomic write;
  the per-firing `_grow` resize and time/iteration label writes are also
  unguarded.

`export_layout` is the ONLY distributed host-fetch in the writer path;
every other path (`shard_writes`, the `clock.time/it` scalars, the
`iteration` reads in reopen/truncate) is process-local.

## Coordination design (de-risked under srun -n 4)

Probe `ts_concurrency_probe.py` validated the tensorstore behaviour:

- Concurrent writes of **disjoint data chunks** to one pre-created zarr
  array are safe.
- A non-rank-0 handle opened with `recheck_cached_metadata: true` sees a
  resize performed by rank 0 (no stale-shape out-of-bounds).
- The destructive `create + delete_existing` is the only real hazard
  (that is gap 2).

**Rule: rank 0 owns every metadata mutation (create, resize, the
time/iteration label writes); every rank writes only its own disjoint
data chunks.** Single-process (`process_count() == 1`) collapses to
today's exact path — barriers are no-ops, no gather, bit-identical
output. This is the invariant the existing `tests/model/io/test_writer.py`
suite must keep proving.

### Seams (new, mockable)

Add to `writer.py`, wrapping jax so tests can simulate multi-process:

```python
def _process_index() -> int:
    return jax.process_index()

def _process_count() -> int:
    return jax.process_count()

def _barrier(tag: str) -> None:
    if _process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(tag)
```

The `Writer` reads `self._rank0 = _process_index() == 0` and
`self._distributed = _process_count() > 1` at `bind`.

### Gap 1 — `spatial/export.py`

Replace the bare fetch at line 240 with a conditional gather:

```python
def _host_labels(arr):
    # arr is the coord node vector (grid.evaluation_nodes(...).data)
    if isinstance(arr, jax.Array) and not arr.is_fully_addressable:
        from jax.experimental import multihost_utils
        return np.asarray(
            multihost_utils.process_allgather(arr, tiled=True))
    return np.asarray(arr)
```

`process_allgather` is a collective, but it fires only when the array is
non-addressable, i.e. only in a genuine multi-process context where
`export_layout` is already called symmetrically on every rank (via
`Writer.bind`, and via `scalar_to_dataarray`/`field.xr` which already
gather the values collectively through `decomposition.gather`). When
fully addressable (single process or replicated) it is a plain
`np.asarray` — single-process output stays bit-identical. `tiled=True`
is required (bare `process_allgather` errors on a sharded array).

### Gap 2 — `writer.py` creation + per-firing coordination

- `_open_store`: build the templates/`_spatial` on **all** ranks (the
  `export_layout` gather is collective). Guard the disk skeleton to
  rank 0: `mkdir`, `_write_group`, `_write_coords`, `_write_time_axis`,
  `_write_variables` run on rank 0 only; then `_barrier("skeleton")`;
  then non-rank-0 ranks **open** (not create) `time`, `iteration`, and
  each variable array via `_open_array(..., recheck=True)`, populating
  `self._time/_iteration/_vars/_spatial`. The mode (`w-`/`w`/`a`)
  existence check + `FileExistsError` is evaluated identically on every
  rank (shared FS) before any mutation, so all ranks agree.

- `_open_array`: add `recheck` so non-rank-0 handles carry
  `recheck_cached_metadata: true`.

- `write()`: rank 0 grows `time`/`iteration`/every var to `nt+1`;
  `_barrier("resized")`; every rank writes its own shard chunks
  (`_write_shards`, unchanged — disjoint); `_barrier("data")`; rank 0
  writes `time[nt]` then `iteration[nt]` (labels last, the commit
  marker). Non-rank-0 handles see the grown shape via recheck.

- `truncate_after` / `_reopen`: the down-resize and the trailing-sentinel
  trim are rank-0-only, bracketed by a barrier; non-rank-0 ranks re-open
  / recheck to the trimmed length. The `iteration.read()` used to find
  `_committed_length` is a process-local read (fine on every rank, but
  only rank 0 needs to act on it).

- `close()`: `_barrier("close")` before dropping handles so no rank
  races ahead of another's final commit.

### `async_writes` under multi-process

The blocking path is coordinated by the per-firing barriers. Deferring
writes past a global barrier breaks the "one firing in flight"
invariant, so **`async_writes` falls back to blocking when
`process_count() > 1`** (v1). Async multi-host is a follow-up. Document
this in the `async_writes` docstring; single-process async is unchanged.

## Testing

1. **Single-process regression (the invariant):** the whole existing
   `tests/model/io/test_writer.py` must stay green unchanged — the
   guards are no-ops at `process_count() == 1`. Add unit tests that
   monkeypatch `_process_index`/`_process_count`/`_barrier` to assert the
   rank-0 guard (non-rank-0 opens instead of creates; only rank 0 writes
   labels) and that `_host_labels` takes the plain-`np.asarray` branch on
   an addressable array.

2. **Real multi-process (CPU subprocess test):** a test that spawns N
   OS processes (`multiprocessing` "spawn") each calling
   `jax.distributed.initialize(coordinator_address="localhost:<port>",
   num_processes=N, process_id=i)` on CPU (one host device each), builds
   the sharded grid + fields, drives the writer, and (rank 0) verifies
   the store against the ground truth. Mark it (e.g.
   `@pytest.mark.multi_process`) and skip by default / when spawning is
   unavailable, so it does not destabilize the single-process CI; it is
   the automated analogue of the `srun` hardware check.

3. **Hardware proof:** the `srun -n 4` reproducer on the 4x A100 node
   (recipe in AGENTS.md) — the authoritative check.

## Merge gate

`tests/model/io/test_writer.py` + `tests/spatial/test_export.py` green,
`ruff check src tests` clean, single-process output bit-identical, and
the real `srun -n 4` reproducer PASSes for both sync and (blocking-
fallback) async before landing on dev.
