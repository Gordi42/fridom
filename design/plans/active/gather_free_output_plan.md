# Gather-free output: shard-wise tensorstore writes

Charter: `design/roadmap/open.md` ("Decomposed / gather-free output").
The iteration-1 `Writer` funnels every firing through
`decomposition.gather` (inside `scalar_to_dataarray`,
`src/fridom/spatial/export.py:149-153`), replicating the full true
field to every device and pulling it to host before tensorstore writes
it. That cannot fit large grids (768³+). The designed-for fix — the
decomposed-slice sink swap keyed by true-DOF indices
(`design/specs/model/classes/io_ops.md`, Writer section;
`design/research/d4_3_io_seams.md`) — is implemented here: each jax
shard writes its own true-DOF tile straight into the zarr store.

## The padding problem, precisely

Storage frame per blocked axis (`TensorDecomposition._axis_storage`,
`tensor.py:405-426`): every shard stores a uniform block

    block = cells + 1 + 2*width,   cells = ceil(n_cells / shards)

(`+1` stagger reserve, `width` halo ghosts each side). True DOFs of
shard `s` occupy the in-block slice `[width, width + t_s)` where, with
`bounds[s] = min(s*cells, n)` and `bounds[shards] = n`
(`_block_bounds`, `tensor.py:351-366`):

    t_s = bounds[s+1] - bounds[s]

- `t_s = cells` for `s < shards-1`;
- last shard: `n - (shards-1)*cells` — *smaller* under mild cell
  padding (non-divisible `n_cells`), or `cells + 1` for the staggered
  `n = shards*cells + 1` spaces (`Outer`, periodic `FaceAvg`).
- unblocked axis (`shards == 1`): single block `n + 2*width`, true
  region `[width, width + n)`.

Everything outside `[width, width + t_s)` — leading/trailing ghosts,
the stagger reserve, cell padding — is storage artifact and must never
reach the output file. The file has exactly the true global shape.

**Do not `unpad()` on device.** Slicing the padded-even storage down to
true size on device is exactly the reblock that can force a
collective-permute (see `design/plans/done/uneven_shard_padding_plan.md`)
and materializes a second array. The gather-free writer slices each
shard's *host copy* instead: fetch the shard block (one contiguous D2H
per shard), apply the `[width, width + t_s)` numpy view, write.

## Chunk alignment (why no chunk is ever read back)

Default spatial zarr chunks are the storage grid: `cells` along a
blocked axis, `n` along unblocked axes, time chunk 1. Then shard `s`'s
target window `[bounds[s], bounds[s+1])` is chunk-aligned: interior
boundaries `s*cells` are chunk multiples, and every chunk of the `[nt]`
time slice is fully covered by exactly one shard's write. Consequences:

- no two writes touch the same chunk → no cross-write serialization;
- every write fully covers the in-bounds region of its chunks → no
  read-modify-write;
- the trailing chunk is a normal zarr edge chunk (short last shard) or
  one extra edge chunk (the staggered `+1` DOF) — both single-writer.

This holds for every divisibility case ("independent of divisibility",
per the charter). User-supplied `chunks=` overrides stay *correct*
(single-process tensorstore serializes per-chunk internally) but may
read-modify-write; same for `mode="a"` appends onto a store created
under a different device count. Both are documented, not forbidden.

## API

### spatial: `Decomposition.shard_writes` (new, abstract + tensor impl)

```python
def shard_writes(self, arr, space, layout=None):
    """
    Iterate the locally-owned true-DOF tiles of a storage array.

    Returns a tuple of ``(index, values)`` pairs: ``index`` is a
    tuple of slices (one per storage axis) in global true-DOF
    coordinates, ``values`` the matching host numpy block with halo
    ghosts, stagger reserve, and cell padding stripped. Only
    replica-0 addressable shards are yielded, so across all
    processes every true DOF appears exactly once and writing all
    tiles reproduces ``gather(arr)`` tile-by-tile — without ever
    forming the global array.
    """
```

Implementation (`TensorDecomposition`): iterate
`arr.addressable_shards`, skip `shard.replica_id != 0` (this dedupes
replicated factors and fully replicated arrays for free), and per axis
record `(name, n, factor, shards, width, block, total)` from
`_geometry(space, layout)`:

- `shards == 1` → target `slice(0, n)`, source `slice(width, width+n)`;
- blocked → `s = shard.index[axis].start // block`, target
  `slice(bounds[s], bounds[s+1])`, source
  `slice(width, width + t_s)`.

`values = np.asarray(shard.data)[source]` — one contiguous D2H of the
block, then a host view. (Per-axis overhead `(2*width+1)/cells` of
transferred-but-dropped bytes is a few percent at production sizes;
device-side pre-slicing is a rejected micro-optimization — it burns
device memory and kernel launches to save PCIe bytes.)

### spatial: `Decomposition.chunk_hint` (new)

```python
def chunk_hint(self, space, layout=None) -> tuple[int, ...]
```

Per storage axis: `cells` if blocked else `n` — the write-aligned
chunk grid above. The writer keys it by dim name; explicit user
`chunks=` wins.

### spatial: export layout helper (refactor, no behavior change)

Extract the label/coord/attr construction out of `scalar_to_dataarray`
(`export.py`) into a values-free helper (plain dataclass: name, dims,
per-dim 1-D coord arrays, attrs, dtype, true shape — no xarray
import). `scalar_to_dataarray` becomes layout + `gather` + xarray
assembly, byte-identical output. Coordinate node vectors are 1-D;
their internal gather is trivial and stays.

### model/io: the Writer sink swap

- `bind`: build templates from the layout helper — **no gather, no
  xarray** anywhere in the writer (xarray drops from the writer's
  runtime requirements; it stays dev-only for reading). Complex/
  coefficient rejection keeps its current error + `.data` pointer,
  now keyed off the field dtype/space directly.
- variable creation: default spatial chunks from `chunk_hint`.
- `write`: per output, evaluate → `Field`, `_grow` the variable, then

  ```python
  for target, values in decomp.shard_writes(field._data, space):
      futures.append(var[(nt, *target)].write(values))
  ```

  All shards × all variables issue as async tensorstore writes; block
  on every future before returning (per-firing barrier — "partial
  output must survive" close semantics unchanged). Cross-boundary
  async (overlapping writes with the next jit chunk) stays a noted
  follow-up seam, as in `d4_3_io_seams.md`.
- Axis mapping: `shard_writes` yields one slice per *storage* axis; if
  the export layout drops/squeezes any axis (constant factors), the
  writer drops the same positions from `target` and squeezes `values`.
  Time/iteration axes, `truncate_after`, `_reopen`, sidecar JSON, and
  the `OutputStream` protocol are untouched.

## Tests

- `tests/spatial/decomposition/…` (mirroring `tensor.py`'s test home):
  tiling property `assemble(shard_writes(arr)) == gather(arr)` —
  single-device and `forced_devices` multi-device, over
  divisible/mild-padded `n_cells` (e.g. 8, 7, 10 on P=4) × spaces
  center / outer / inner / bounded FaceAvg; replicated-factor dedupe
  (each index yielded once); values-are-numpy; layout override.
- `tests/model/io/test_writer.py`: the existing suite must pass
  unchanged (bitwise-equal slices, dims, coords, CF attrs — the file
  format is invariant under this change). Add: device-count invariance
  (multi-device store == single-device store, mirroring
  `test_export.py:245-269`), file shape == true shape on a
  non-divisible grid (padding provably absent), chunk default ==
  `chunk_hint`, and a no-`decomposition.gather`-on-write guard
  (monkeypatch gather to raise after bind; write must succeed).
- Multi-device runs ride the standard forced-host invocation
  (`XLA_FLAGS=--xla_force_host_platform_device_count=4`,
  `FRIDOM_TEST_FORCED_DEVICES=4`).

## Phases

1. `spatial`: `shard_writes` + `chunk_hint` (+ abstract contract) +
   decomposition tests.
2. `spatial/export` + `model/io`: layout helper refactor; writer
   template off the helper (independent of phase 1).
3. `model/io`: the sink swap in `write` + writer tests (needs 1 + 2).
4. Records: io_ops.md / 04_run_loop_io.md sink paragraphs, roadmap
   open → done entry; this plan → `design/plans/done/`.

## Out of scope

- Multi-host: single-controller stands (`local_slice` stub unchanged);
  `shard_writes`' replica-0 / addressable-shard contract is already
  the multi-host-correct shape, so the future backend swaps under the
  same API.
- Snapshots (separate leaf-blob store by design), TimeSeries (already
  gather-free), async-across-boundaries, `.xr` / `scalar_to_dataarray`
  public gather path (interactive convenience, stays).
