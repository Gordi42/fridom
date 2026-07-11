# D4.3 — The IO seams

Research report (see [`README.md`](README.md) for status).

## 1. Where outputs are evaluated: chunk-boundary (adopted)

**Iteration 1 evaluates all outputs host-side on the synced carry at
chunk boundaries, with chunk boundaries derived from the union of
trigger times. No `io_callback` in the traced step.**

Against in-trace `io_callback`: ordered callbacks thread a
sequencing token pinning D2H copies into the scan's dependency chain
and constraining donation; each firing stalls the stream for
synchronous Python; `lax.cond` triggers make every writer/derived
set part of the *step trace* — **adding a diagnostic to a writer
would recompile the physics** (the decisive seam argument: under
chunk-boundary evaluation the compiled step is IO-free and 2.6 can
iterate on sinks forever); callbacks force gathers/shard decisions
into the trace on multi-device; `io_callback` doesn't compose with
`vmap` (ensemble futures). The cadence-not-dividing-chunk problem is
dissolved by inversion: **chunks are derived from triggers**.
Degenerate every-step output → chunk length 1 → accepted as a
debugging mode; the in-trace *capture* variant (same output
callables evaluated under a traced trigger into scan outputs,
flushed by the same sink) is the designed-for escape 2.6 may add
**without touching any D1–D4 contract** — the frozen contract is
D2.3's: an output is a pure `(model_state) -> Field | scalar`
callable, named, attached to a trigger and a sink.

## 2. Triggers and chunk alignment

Declarative frozen host-side data — never callbacks, never traced
(it-1): `fr.every(steps=|hours=|walltime=)`, `fr.at(times=[...])`,
unions via `|`, `after=`/`until=` windows. Lowering at run planning:
model-time triggers reduce to sorted step-index sets against the
fixed-dt step grid, **snapping up** with realized times logged (and
recorded on the time axis — no phantom times). `every()` includes
step 0 (IC written — the `execute_at_start` successor). **Chunk
boundaries = union of all trigger step sets ∪ snapshot steps ∪ last
step**, subdivided by a `max_chunk` knob; a trigger fires iff its
step equals the boundary. Walltime triggers only on
snapshots/actions — rejected on data writers (nondeterministic
output grids). Because triggers are pure data, a traced lowering for
the capture path is derivable later from the same objects.

## 3. The writer seam

```python
fr.io.Writer(path, fields=(...), derived={name: callable},
             trigger=..., mode="w|w-|a", chunks=None, attrs={...})
```

- Attached at `run(outputs=...)` (and/or standing config) —
  post-assembly host infrastructure: never in the carry, treedef, or
  fingerprint. This concretely retires the constructor `diagnostics=`
  kwarg *(reconciled: the constructor `io=` slot holds standing
  config naming unbound primitives, bound at assembly; run-time
  `outputs=` accepts bound `model.diagnostics.*`)*.
- `fields=` resolves against the FieldTable; **default: all
  PROGNOSTIC + DIAGNOSTIC; AUXILIARY opt-in by name** (D1.1).
  Unknown names → hinted bind-time error; bind-time dry evaluation
  validates names/shapes/scalar-ness and creates the store eagerly.
- **Layout contract** (builds on the grid notes' `.xr` export): the
  written zarr opens in xarray/xgcm with no post-processing —
  xgcm-style axis-position dims from the component's space,
  coordinates from `grid.evaluation_nodes`, attrs from
  `FieldMetadata` (derived variables from the diagnostic's
  `with_metadata`), CF time axis (`seconds since <start_date>`,
  host-side calendar) + an `iteration` coordinate, global attrs
  incl. the **fingerprint digest + version** (provenance).
  Coefficient-space/complex fields raise (inherits the `.xr` it-1
  restriction).
- **Sinks receive Fields, not numpy** — the gather
  (`decomposition.gather`, rank 0 writes) happens *inside* the it-1
  sink, so the ROADMAP decomposed-slice follow-up is a sink swap
  (per-process region writes keyed by `local_slice` in global
  true-DOF indices). Subsetting is composition on the callable
  (`sel=` convenience later). File splitting: designed-for
  `split=fr.every(...)`.
- **Resume contract (load-bearing)**: on restart resume, writers
  open `mode="a"` and **`truncate_after(snapshot_time)`** — a
  crashed segment may have written outputs past the snapshot;
  skipping this forks the time axis.

## 4. Restart / snapshot

**A dumb leaf blob, not zarr-via-the-writer** — the snapshot must
round-trip the entire carry bitwise (all lifecycles, module leaves,
stepper ring buffers + warm-up counter, float64 clock, panic flag);
forcing it through the human-facing writer layout risks lossiness
exactly where restarts must be exact. *Writers are for humans;
snapshots are for the machine.*

- Format: snapshot **directory** = `manifest.json` + leaf arrays
  (~50 lines, no orbax dependency; orbax-shaped for later; versioned
  manifest; multi-host later = per-process leaf files + rank-0
  manifest). *(Reconciled with d4_4: leaves are stored true-shape/
  gathered so restarts are device-count-portable; decomposed
  snapshots are the same follow-up as decomposed output.)*
- **Manifest stores the fingerprint digest AND its source record**
  (declarations, module tuple, per-term treatments, stepper statics,
  parameter specs incl. Ramp shapes) — which is what makes the
  mandated "named error, never silent reuse" achievable: the
  mismatch error *diffs* the records ("stepper statics differ:
  cnab2 → sbdf2"). IO config is deliberately excluded (changing
  output cadence across a resubmit is legal).
- Fingerprint computed **once at the end of assembly**
  (`model.fingerprint`); `load_snapshot` compares.
- **API: assemble-then-load** — `model.snapshot(path)` /
  `model.load_snapshot(path)` (fingerprint check, overwrite all
  carry leaves). **`Model.restore(path)` classmethod rejected**: it
  implies the snapshot contains the recipe — the dill pattern
  reborn. Scripts are the recipe.
- Run config: `fr.io.Snapshots(path, trigger=fr.every(walltime=...)
  | fr.every(days=...), keep=N, resume=True,
  on_walltime=fr.io.resubmit())` — resume scans for the newest
  *complete* snapshot (atomic tmp-dir + rename), fingerprint-checks,
  loads, recomputes remaining steps from the stored clock against
  the absolute target. The old SLURM auto-detection
  (`scontrol` → `sbatch`) survives as `fr.io.resubmit()`.
- Async checkpointing: not built; the seam allows a
  background-thread store (host copy of the boundary-synced carry,
  joined before the next snapshot).

## 5. The dill successor — confirmed

`Model.save/load` (dill) dies with no direct successor: it pickled
code (version-brittle, grid-amputation hack, recipe hidden in a
blob). Persistence = **script re-assembly + leaf snapshots** — one
code path, one fingerprint rule, zero pickled bytecode. Resuming
without the original script is deliberately dropped (the manifest's
source record keeps mismatches diagnosable). Proposed 02_rules
entry: *"No pickled models. Anything not reconstructible from
(script, snapshot) is a design bug."*

## 6. The TimeSeries sink and the OutputStream protocol

`fr.io.TimeSeries(path, columns={name: callable}, trigger=...)` —
scalar outputs (0-d, replicated on multi-device → single-scalar
fetch, no gather), rows `(time, iteration, *values)`, flushed per
boundary (tail-able). Bind-time scalar-ness check with a hinted
error. The one interface D4 owns and 2.6 builds behind:

```python
class OutputStream(Protocol):        # Writer, TimeSeries, future capture streams
    trigger: Trigger
    def bind(self, model): ...           # resolve names, dry-evaluate, create store
    def write(self, model_state): ...    # at each firing boundary, deterministic order
    def truncate_after(self, time): ...  # restart-resume alignment
    def close(self): ...                 # run end / abort (partial output survives)
```

## 7. Risks / open questions

1. Every-step output → chunk 1 dispatch overhead: accepted; capture
   path is the escape.
2. Multi-host gather OOM: decomposed-slice sink is the fix; schedule
   early in 2.6 for large runs.
3. Walltime granularity: a single chunk exceeding the remaining
   allocation still dies — `max_chunk` margin (D4.2 knob).
4. `truncate_after` awkward for CSV (rewrite tail) — pinned in the
   protocol so sinks must answer it.
5. Async sink ordering: serialize per store — sink-internal.
6. `at(times=)` under a later dt change lands on different steps —
   plan-time logging + realized times mitigate; consider a tolerance
   warning.
7. Fingerprint scope: IC differences are invisible (leaves, not
   structure) — correct by design; one sentence in 02_rules.

## 8. Sketch (the full flow)

The `run.py` script assembles, sets ICs, attaches a zarr Writer
(10-model-hour cadence, derived pv) + a TimeSeries (1-hour etot) +
`Snapshots(walltime="7.5h", resume=True, on_walltime=resubmit)`.
Plan: triggers → boundary set (60-step chunks); bind: stores
created, IC written at step 0. At ~7.5 h wall: atomic snapshot dir,
rotation, `sbatch`, exit 0. **Resubmitted job**: same script,
identical assembly/fingerprint, `set_fields` runs harmlessly,
`run()` finds the snapshot, loads (stepper history bitwise), writers
open `mode="a"` + `truncate_after(t_snap)`, remaining steps
re-planned. Editing the script to SBDF2 first →
`SnapshotMismatchError: stepper statics differ` — never silent.

**Iteration-1 build list (D4-owned)**: triggers + plan-time
lowering; OutputStream protocol; Writer with the zarr-append gather
sink; TimeSeries CSV sink; leaf-blob snapshot store +
manifest/fingerprint; Snapshots config + SLURM resubmit. **Behind
the seams for 2.6**: TensorStoreWriter backend, decomposed-slice
writes, async writes, capture streams, spectral export, splitting,
`sel=`.
