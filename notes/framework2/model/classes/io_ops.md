# Model layer redesign — Class designs: IO & ops

Part of the model-layer class designs; see [`README.md`](README.md)
for the document map, the shared template, and the cross-cluster
seam anchors. Normative law for this cluster:
[`../04_run_loop_io.md`](../04_run_loop_io.md) §6.3 (the run loop,
including the coupling-round amendments) and §6.4 (the IO seams);
[`../02_rules.md`](../02_rules.md) (restart-fingerprint scope,
no-pickled-models, the extended `set_aux` / three-operation matrix);
[`../09_coupling_designfor.md`](../09_coupling_designfor.md) §11.2
decision 3 (the full Session spec) and §11.3 (**CS-3..11 and CS-18
bind these specs**); [`../06_validation.md`](../06_validation.md)
§8.4 (io binding/collision resolutions). Research archive (pointers,
never re-argued):
[d4_3](../research/d4_3_io_seams.md) (the io seams — the primary
transcription source), [d4_2](../research/d4_2_run_loop.md) (the run
internals the Session absorbs),
[c3](../research/c3_coupling_architecture.md) (the H1–H5 hooks),
[c2](../research/c2_coupled_walk.md) (the walk behind the CS list).

Classes owned: the Trigger family + the plan-time lowering function,
the `OutputStream` protocol, `fr.io.Writer`, `fr.io.TimeSeries`,
`fr.io.Snapshots` + the snapshot store + `SnapshotManifest`,
`fr.io.resubmit`/`fr.slurm`, **`fr.ops.Session`**, the normative ops
protocols (`WalltimeGuard`, `ProgressReporter`), and the reserved
`PendingAdvance`. `AdvanceResult`/`RunResult`/`PanicError` and the
`advance`/`run`/`snapshot`/`load_snapshot` methods are
**[`model.md`](model.md)-owned** and only cross-referenced here.

---

## Module placement and cluster-wide rules

```
framework2/
    io/
        __init__.py     # fr.io namespace (lazypimp re-exports)
        triggers.py     # Trigger nodes, every/at factories, lower_trigger
        streams.py      # OutputStream protocol, binding helpers, IO errors
        writer.py       # Writer (zarr-append gather sink behind it, 2.6)
        timeseries.py   # TimeSeries (CSV sink, 2.6)
        snapshots.py    # Snapshots config, SnapshotManifest, store functions
        slurm.py        # fr.slurm helpers; resubmit
    ops/
        __init__.py     # fr.ops namespace
        session.py      # Session
        protocols.py    # WalltimeGuard, ProgressReporter, ChunkStats
        pending.py      # PendingAdvance (reserved)
```

`fr.every` / `fr.at` are additionally re-exported at top level (the
acceptance-surface spelling, sketch 7.7); everything else is reached
as `fr.io.*` / `fr.ops.*` / `fr.slurm.*`. The package split follows
the repo lazypimp pattern; the file layout is spec-level, not
normative.

Rules governing every class in this file:

- **Host infrastructure, never traced.** Nothing here is a pytree,
  enters the carry or the treedef, or is hashed into the restart
  fingerprint (IO config is deliberately excluded from the manifest
  check — changing output cadence across a resubmit is legal).
- **Outputs are evaluated host-side on the synced carry at chunk
  boundaries; no `io_callback` in the traced step** (§6.4's decisive
  seam argument: the compiled step is IO-free forever — adding a
  diagnostic to a writer never recompiles the physics). The frozen
  output contract is D2.3's: an output is a pure
  `(model_state) -> Field | scalar` callable, named, attached to a
  trigger and a sink. The in-trace *capture* variant is the 2.6
  designed-for escape, reachable behind `OutputStream` without any
  contract change.
- **Boundary sequence** (normative, §6.3): sync → panic check →
  writer flush → progress → walltime check. Boundary IO fires on the
  committed post-chunk carry *before* control returns to the caller,
  so driver-side `set_aux` writes/resets (the S6 accumulation
  idiom's host-read-then-host-reset, 02_rules) never race an output
  read.
- **Writers are for humans, snapshots are for the machine** (§6.4):
  the human-facing layout contract and the bitwise machine blob are
  different classes on purpose; snapshots are never zarr-via-the-
  writer.
- **CS-11**: no class here holds time-integral state — Session and
  stream state is configuration plus transient counters, discarded
  at close/exit; anything with time-integral semantics (window-mean
  fluxes, accumulators) is a declared field in some model's carry.
- **`advance` is the primitive; everything here is layered sugar**
  (README seam anchor), with the bitwise equivalence test extended:
  `run()` ≡ a user-written single-model Session loop.

Member tags follow the README template's *task* column: `# 2.4`
(run/Session/snapshots), `# 2.6` (Writer/TimeSeries backends),
`# 3.2` (Session multi-model maturation), `# designed-for`.

---

### Trigger family — `fr.every`, `fr.at`, unions, windows

Declarative frozen host-side data — never callbacks, never traced
(§6.4).

| Aspect | Value |
|--------|-------|
| Kind | frozen dataclasses; `every`/`at` factory functions are the API |
| Pytree | host object, not a pytree |
| Task | 2.4 |
| Design refs | 04 §6.3–6.4 (triggers, chunking), d4_3 §2, CS-7 |

```python
"""Declarative output/action triggers: frozen data, never callbacks."""
from __future__ import annotations

from dataclasses import dataclass

import fridom.framework2 as fr


@dataclass(frozen=True)
class Trigger:
    """Base of the trigger algebra; immutable declarative data."""

    def __or__(self, other: Trigger) -> Trigger:               # 2.4
        """Union: the composed trigger fires when any operand fires."""
        ...

    @property
    def has_walltime(self) -> bool:                            # 2.4
        """Whether any component is walltime-based (snapshot/action
        only; data streams reject such triggers at bind)."""
        ...


def every(                                                     # 2.4
    *,
    steps: int | None = None,
    seconds: float | None = None,
    minutes: float | None = None,
    hours: float | None = None,
    days: float | None = None,
    walltime: str | float | np.timedelta64 | None = None,
    after: float | np.timedelta64 | None = None,
    until: float | np.timedelta64 | None = None,
) -> Trigger:
    """Periodic trigger; exactly one cadence kwarg. Model-time and
    step cadences include step 0 (the execute_at_start successor)."""
    ...


def at(                                                        # 2.4
    times: Sequence[float | np.timedelta64],
    *,
    after: float | np.timedelta64 | None = None,
    until: float | np.timedelta64 | None = None,
) -> Trigger:
    """Fire at explicit model times; validated at run planning."""
    ...
```

Notes:

- **Vocabulary** (d4_3 §2, fixed): `fr.every(steps=|seconds=|
  minutes=|hours=|days=|walltime=)`, `fr.at(times=[...])`, unions
  via `|`, `after=`/`until=` windows. Model times are float seconds
  or `np.timedelta64`; `walltime=` additionally accepts the string
  spelling (`"7.5h"`, sketch 7.7). Exactly one cadence kwarg per
  `every()` call — composition is `|`, never multiple cadences in
  one node.
- **Walltime triggers are snapshot/action-only** — rejected on data
  writers at bind (nondeterministic output grids); they are
  boundary-evaluated by the `WalltimeGuard`, never lowered to steps.
- **Windows**: `after=`/`until=` filter the firing set; the
  endpoints are lowered to step indices by the same step-space rule
  as everything else (below), so windows are sign-agnostic too.
- The concrete node classes (`Every`, `At`, `Union`, `Window`) are
  public in `triggers.py` but not re-exported; the factories are the
  user surface. Node naming is spec-level (the notes name only the
  factories and the algebra).
- Because triggers are pure data, a *traced* lowering for the 2.6
  capture path is derivable later from the same objects (d4_3 §2) —
  no contract change reserved beyond that sentence.

---

### `fr.io.lower_trigger` — the plan-time lowering function

The pure, reusable trigger→step-set lowering (CS-7: importable by
drivers, no model or clock access).

| Aspect | Value |
|--------|-------|
| Kind | pure module-level function |
| Pytree | n/a (host function) |
| Task | 2.4 |
| Design refs | 04 §6.3–6.4 (step-space lowering, amended V-S1), CS-7, d4_3 §2 |

```python
def lower_trigger(                                             # 2.4
    trigger: Trigger,
    *,
    t0: float,
    dt: float,
    n_steps: int,
) -> tuple[int, ...]:
    """Lower a trigger to a sorted step-index set in step space:
    k = (t - t0) / dt, snapped up with ceil(k) — sign-agnostic by
    construction. Walltime components lower to the empty set."""
    ...
```

Notes:

- **Step space, sign-agnostic** (amended V-S1): for `dt < 0` the
  division maps decreasing times to increasing `k`; the same
  `ceil` snap applies. No branch on the dt sign exists anywhere in
  the lowering.
- `every()` cadences enumerate firing times in the run direction and
  snap up; step 0 is included. `fr.at` times outside the run's time
  interval (`ceil(k) ∉ [0, n_steps]`) **error at planning** — never
  silently dropped.
- **Realized times are logged and written**: the snapped time
  `t0 + k·dt` is what lands on the output time axis (no phantom
  times); the logging duty sits with the caller (bind/plan), not
  with this pure function.
- **Chunk-boundary derivation is not this function**: boundaries =
  union of all trigger step sets ∪ snapshot steps ∪ the last step,
  subdivided by `max_chunk` (auto ~256) — that planning is
  Session/`run()` internals (§6.3). CS-7 requires exactly the
  trigger→step-set piece to be public, and this is it.
- The function name `lower_trigger` is a spec-level choice (the
  notes fix the semantics and the reusability requirement, not the
  spelling).

---

### `OutputStream` (protocol)

The one interface D4 owns; 2.6 builds behind it (d4_3 §6).

| Aspect | Value |
|--------|-------|
| Kind | `typing.Protocol` (structural) |
| Pytree | host object, not a pytree |
| Task | 2.4 (protocol); implementations tagged individually |
| Design refs | 04 §6.4 (protocol, binding split, IOCollisionError), 06 §8.4, d4_3 §6 |

```python
"""The output-stream protocol: Writer, TimeSeries, future capture
streams all implement it."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OutputStream(Protocol):
    """A trigger-driven output stream bound to one model."""

    trigger: Trigger

    def bind(self, model: fr.Model) -> None:                   # 2.4
        """Resolve names against the model, dry-evaluate every
        output, create the store eagerly. Called at RUN START,
        never at assembly."""
        ...

    def write(self, model_state) -> None:                      # 2.4
        """Write from the boundary-synced carry at a firing
        boundary; streams fire in deterministic (binding) order."""
        ...

    def truncate_after(self, iteration: int) -> None:          # 2.4
        """Drop output past the given iteration coordinate
        (restart-resume alignment; iteration-keyed per V-S1)."""
        ...

    def close(self) -> None:                                   # 2.4
        """Run end or abort; partial output must survive."""
        ...
```

Semantics, invariants, error behavior:

- **The binding split** (amended V-C5, §6.4): diagnostic-*name*
  resolution happens at assembly (for constructor `io=` standing
  config naming *unbound* package primitives) or immediately (for
  bound `model.diagnostics.*` spellings in `run(outputs=...)`);
  `bind()` — store creation, dry evaluation — happens at **run
  start**, never at assembly. Presets never create files for models
  that never run. The union of standing and per-run streams binds at
  run start.
- **Collision rule** (06 §8.4): streams dedupe by resolved path; two
  *distinct* streams targeting one path raise `IOCollisionError` at
  bind.
- `write` receives the boundary-synced carry (`model_state` — the
  02_rules carry naming): fields plus clock and iteration, which is
  what the time/iteration axes are written from.
- **`truncate_after` is iteration-keyed** — a deliberate deviation
  from d4_3 §6's `truncate_after(time)` sketch, per the applied V-S1
  amendment: the iteration coordinate is monotone in both time
  directions (backward runs invert the time axis; decreasing-time CF
  axes are legal). Skipping the truncation on resume forks the axis
  — a crashed segment may have written outputs past the snapshot.
- `close()` on abort keeps partial output readable (zarr appends and
  flushed CSV rows survive by construction).
- Walltime-bearing triggers are rejected at `bind` on data streams
  (hinted error pointing at `Snapshots`).
- **Implementers**: `Writer`, `TimeSeries` (below); designed-for 2.6:
  capture streams (in-trace evaluation of the same output callables
  under a traced trigger into scan outputs, flushed by the same
  sink) and post-assembly writer attach — both reachable with **no
  change to this protocol** (07_open_threads).

Error types owned by this cluster (registered in
[`model.md`](model.md)'s error-type registry):

```python
class IOCollisionError(Exception):                             # 2.4
    """Two distinct output streams resolved to the same path."""


class SnapshotMismatchError(Exception):                        # 2.4
    """Snapshot fingerprint mismatch; the message DIFFS the source
    records ('stepper statics differ: cnab2 -> sbdf2')."""
```

---

### `fr.io.Writer`

Gridded output: xarray/xgcm-openable zarr, one stream per store.

| Aspect | Value |
|--------|-------|
| Kind | concrete, implements `OutputStream` |
| Pytree | host object, not a pytree |
| Task | 2.6 (front kwargs fixed here; sink backends behind the seam) |
| Design refs | 04 §6.4 (Writer seam, layout contract, resume), d4_3 §3, 06 §8.4 |

```python
"""The zarr writer: human-facing gridded output."""
from __future__ import annotations

import fridom.framework2 as fr


class Writer:
    """Trigger-driven gridded output stream (zarr, xarray/xgcm-openable)."""

    def __init__(                                              # 2.6
        self,
        path: str | Path,
        *,
        fields: Sequence[str] | None = None,
        derived: Mapping[str, Callable] | None = None,
        trigger: Trigger,
        mode: Literal["w", "w-", "a"] = "w-",
        chunks: Mapping[str, int] | None = None,
        attrs: Mapping[str, str] | None = None,
    ) -> None:
        """Configure the stream; no file IO happens here."""
        ...

    @property
    def trigger(self) -> Trigger:                              # 2.6
        """The stream's firing trigger (protocol slot)."""
        ...

    def bind(self, model: fr.Model) -> None:                   # 2.6
        """Resolve fields/derived, dry-evaluate, create the store."""
        ...

    def write(self, model_state) -> None:                      # 2.6
        """Append one time slice from the synced carry."""
        ...

    def truncate_after(self, iteration: int) -> None:          # 2.6
        """Drop appended slices past the iteration coordinate."""
        ...

    def close(self) -> None:                                   # 2.6
        """Flush and close the store; partial output survives."""
        ...
```

Semantics, invariants, error behavior:

- **`fields=` lifecycle defaults** (D1.1, §6.4): names resolve
  against the FieldTable; the default is **all PROGNOSTIC +
  DIAGNOSTIC; AUXILIARY is opt-in by name**. Unknown names raise a
  hinted bind-time error; bind-time dry evaluation validates
  names/shapes and creates the store eagerly.
- **`derived=` and the binding split**: a derived entry is a named
  pure output callable. In the constructor `io=` slot it names
  *unbound* package primitives
  (`nh.diagnostics.pot_vort` — name resolution at assembly, which
  resolves D2.3's chicken-and-egg); in `run(outputs=...)` /
  Session `outputs=` it accepts *bound*
  `model.diagnostics.*` spellings (resolved immediately). Either
  way the store is created only at `bind()`. Derived variables carry
  the diagnostic's `with_metadata` annotation (the 02_rules
  diagnostics-metadata rule); evaluation happens only at output
  cadence.
- **Layout contract** (summary — the full label/gather rules are the
  grid cluster's export section,
  [`../../classes/grid.md`](../../classes/grid.md) §4, which this
  writer consumes exactly as `f.xr` does): the written zarr opens in
  xarray/xgcm with no post-processing — xgcm-style axis-position
  dims derived from each component's function space, coordinates
  from `grid.evaluation_nodes`, per-variable attrs from
  `FieldMetadata`, a CF time axis (`seconds since <start_date>`,
  host-side calendar) **plus an `iteration` coordinate**, and global
  attrs including the **fingerprint digest + version** (provenance).
  Coefficient-space and complex fields raise (inheriting the `.xr`
  it-1 restriction, with a pointer to `.data`).
- **Sinks receive Fields, not numpy** (§6.4): the gather
  (`decomposition.gather`, rank 0 writes) happens *inside* the it-1
  sink, so decomposed-slice output is a **sink swap** keyed by
  `local_slice` in global true-DOF indices — an internal seam, not a
  public kwarg in it-1. Behind the seam for 2.6: TensorStoreWriter
  backend, decomposed-slice writes, async writes (serialize per
  store), file splitting (`split=fr.every(...)`), `sel=` subsetting
  (meanwhile: composition on the callable).
- **`mode` / resume semantics**: `"w-"` create-and-fail-if-exists,
  `"w"` overwrite, `"a"` append. The `"w-"` *default* is a
  spec-level choice (failing loudly beats silently clobbering a
  previous segment); the notes fix the vocabulary, not the default.
  On snapshot resume the run/Session machinery flips bound writers
  to `mode="a"` and calls `truncate_after(it_snap)` — the
  load-bearing resume contract (§6.4).
- Never in the carry, the treedef, or the fingerprint; a Writer is
  reusable config until `bind`, single-store after it.

---

### `fr.io.TimeSeries`

Scalar rows to a tail-able CSV.

| Aspect | Value |
|--------|-------|
| Kind | concrete, implements `OutputStream` |
| Pytree | host object, not a pytree |
| Task | 2.6 |
| Design refs | 04 §6.4, d4_3 §6, D2.3 (integrated scalars) |

```python
class TimeSeries:
    """Trigger-driven scalar time series (CSV, tail-able)."""

    def __init__(                                              # 2.6
        self,
        path: str | Path,
        *,
        columns: Mapping[str, Callable],
        trigger: Trigger,
    ) -> None:
        """Configure named scalar expressions; no file IO here."""
        ...

    # trigger / bind / write / truncate_after / close: the
    # OutputStream protocol, same binding-split rules as Writer.
```

Notes:

- **Columns are scalar outputs**: each callable returns a 0-d value
  (0-d Field or scalar). On multi-device the 0-d result is
  replicated, so the fetch is a **single-scalar read — no gather**.
  Bind-time dry evaluation checks scalar-ness with a hinted error.
- Rows are `(time, iteration, *values)`, flushed at each firing
  boundary — the file is tail-able during the run. Integrated
  scalars (`total_energy`) are the same D2.3 primitive at rank 0
  feeding this sink.
- `truncate_after(iteration)` for a CSV sink means rewriting the
  tail keyed on the iteration column; the protocol pins that the
  sink *must* answer the call — the concrete strategy is a parked
  2.6 residual (see Open questions).

---

### `fr.io.Snapshots` (run config)

The one home for snapshot trigger + rotation + resume + resubmit
(§6.3 reconciliation 4).

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen config object |
| Pytree | host object, not a pytree |
| Task | 2.4 |
| Design refs | 04 §6.3 (walltime), §6.4 (Snapshots-run-only, resume), d4_3 §4, 06 §8.4 |

```python
class Snapshots:
    """Run-config for restart snapshots: where, when, how many,
    whether to resume, and what to do on walltime."""

    def __init__(                                              # 2.4
        self,
        path: str | Path,
        *,
        trigger: Trigger,
        keep: int | None = None,
        resume: bool = True,
        on_walltime: Callable[[], None] | None = None,
    ) -> None:
        """Pure configuration; consumed by run()/Session."""
        ...
```

Semantics, invariants, error behavior:

- **Run-config only** (amended, 06 §8.4): `Snapshots` is accepted by
  `run(snapshots=)` / `Session(snapshots=)` **only**; the model
  constructor `io=` slot and the `outputs=` tuple **reject** it with
  a hinted error — one resume path, never two.
- `trigger` may mix model-time components (lowered to steps, joining
  the chunk-boundary union) and a walltime component
  (`fr.every(walltime="7.5h")`), which is evaluated **predictively**
  at boundaries via the `WalltimeGuard` — the run stops *before* the
  chunk that would blow the budget, snapshots, then fires
  `on_walltime`.
- **`keep=N` rotation**: after each successful (committed) snapshot,
  complete snapshots beyond the newest N are deleted; in-progress
  tmp dirs never count as snapshots.
- **`resume=True`**: at run start, scan `path` for the newest
  *complete* snapshot (the atomic rename is the commit marker); read
  the manifest **header only** (no leaf IO); fingerprint-check
  (mismatch → `SnapshotMismatchError` with the source-record diff);
  load all carry leaves (bitwise — stepper history and warm-up
  counter included; bitwise applies to the leaf round-trip and to
  continuation at unchanged device count on the same compiled
  path — see the device-count note in the store section below);
  **re-plan the remaining steps from the stored
  clock against the absolute target** (`end_time=` is the natural
  resumable spelling); flip bound writers to `mode="a"` +
  `truncate_after(it_snap)`. If no snapshot exists, start fresh —
  the driver script's `set_fields` runs harmlessly either way
  (d4_3 §8).
- `on_walltime` is a plain callable invoked once, on the walltime
  exit path only, after the triggered snapshot has committed
  (`fr.io.resubmit()` is the shipped one). It is **not** called on
  normal completion or on `PanicError`.
- Coupled runs: snapshots only at window boundaries (CS-18) —
  enforced mechanically by the Session's planning (below), not by
  this config object.

---

### The snapshot store + `SnapshotManifest`

The dumb leaf blob and its machine-readable header (CS-10).

| Aspect | Value |
|--------|-------|
| Kind | frozen dataclasses + pure module functions |
| Pytree | host objects, not pytrees |
| Task | 2.4 |
| Design refs | 04 §6.4 (leaf blobs, manifest), 02_rules (fingerprint scope, dt-in-manifest, no-pickled-models), CS-10/CS-11, d4_3 §4–5 |

```python
"""The snapshot store: manifest + true-shape leaf arrays."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeafEntry:
    """Index entry for one stored carry leaf."""

    key: str                     # carry tree path                 2.4
    file: str                    # leaf file, relative to the dir  2.4
    dtype: str                                                   # 2.4
    shape: tuple[int, ...]       # TRUE (gathered) global shape    2.4


@dataclass(frozen=True)
class SnapshotManifest:
    """Versioned snapshot header + leaf index; the header is
    machine-readable WITHOUT leaf IO (CS-10)."""

    version: int                                               # 2.4
    model_name: str | None                                     # 2.4
    time: float                  # float64 clock                  2.4
    iteration: int                                             # 2.4
    provided_parameters: Mapping[str, float]  # >= {"TIME_STEP"}  2.4
    fingerprint: str             # digest                         2.4
    fingerprint_record: Mapping[str, object]  # the source record 2.4
    leaves: tuple[LeafEntry, ...]                              # 2.4

    @property
    def dt(self) -> float:                                     # 2.4
        """Shorthand for provided_parameters['TIME_STEP']."""
        ...

    def fingerprint_diff(self, record: Mapping) -> str | None: # 2.4
        """Human-readable structural diff against a model's source
        record ('stepper statics differ: cnab2 -> sbdf2');
        None when the records match."""
        ...


def write_snapshot(path, manifest, leaves) -> Path:            # 2.4
    """Atomic commit: write manifest + leaf files into a tmp dir,
    then rename (the rename IS the completeness marker)."""
    ...


def read_manifest(path) -> SnapshotManifest:                   # 2.4
    """Header + index only — never touches leaf files (CS-10)."""
    ...


def read_leaves(path, manifest) -> Mapping[str, object]:       # 2.4
    """Load the true-shape leaf arrays named by the manifest."""
    ...


def find_latest(directory) -> Path | None:                     # 2.4
    """Newest COMPLETE snapshot in a Snapshots directory."""
    ...


def rotate(directory, keep: int) -> None:                      # 2.4
    """Delete complete snapshots beyond the newest ``keep``."""
    ...
```

Semantics, invariants, error behavior:

- **A dumb leaf blob, not zarr-via-the-writer** (§6.4): the snapshot
  round-trips the *entire carry* bitwise — all lifecycles, module
  leaves, stepper ring buffers + warm-up counter, float64 clock,
  panic flag. Format: snapshot **directory** = versioned
  `manifest.json` + leaf arrays; ~50 lines of store code,
  orbax-shaped but with **no orbax dependency**.
- **Leaves are stored true-shape, gathered** — restarts are
  device-count-portable by construction (CI cases: 4→1, 1→4,
  permuted device ids; no device identity anywhere in the record).
  Scope of "bitwise" (2026-07-08): the leaf round-trip is bitwise,
  and continuation at **unchanged device count on the same compiled
  path** is bitwise; a device-count-changed resume is valid but
  continues within ~2 ulp/step (phase-1 finding 1) — never bitwise.
  Decomposed/multi-host snapshots (per-process leaf files + rank-0
  manifest) are the same follow-up as decomposed writer output.
- **The manifest header is machine-readable without leaf IO**
  (CS-10): clock time, iteration, dt (via provided-parameter
  values), fingerprint digest — this is what the coupled
  cross-model consistency assertion reads on resume (CS-18).
- **Provided-parameter values are recorded** — at least
  `fr.params.TIME_STEP` (§8.6-5): `load_snapshot` **errors on a dt
  sign mismatch and warns on magnitude** (the successor of
  `run_backward`'s sign re-forcing; 02_rules).
- **Fingerprint digest AND source record**: the record holds field
  declarations (names/spaces/lifecycles), the module tuple (types +
  order), per-term treatments, stepper statics (order/eps/tableau,
  plus module-owned ADVANCE-stage integrator statics), and parameter
  *specs* (a Ramp's shape is structure; endpoints are leaves).
  Mismatches **diff, never silently reuse**
  (`SnapshotMismatchError`). Fingerprint scope (02_rules, refined):
  structure only, never leaves — IC differences are deliberately
  invisible; and the record's "spaces" are the **bare declared
  spaces / patterns** — `Layout`, negotiation fingerprints, and
  device topology never enter it.
- **CS-11**: the snapshot never embeds run-loop or driver state — no
  session counters, stream positions, or walltime state. Everything
  time-integral is a carry leaf and is therefore already inside.
- IO config is deliberately excluded from the fingerprint check
  (changing output cadence across a resubmit is legal).
- **Atomicity**: tmp-dir + rename; `find_latest` only ever sees
  committed snapshots; a crash mid-write leaves a tmp dir that is
  ignored and garbage-collected by the next rotation.
- Consumers: `model.snapshot(path)` / `model.load_snapshot(path)`
  ([`model.md`](model.md) — assemble-then-load;
  `Model.restore(path)` classmethod **rejected**: the dill pattern
  reborn, see d4_3 §4–5) and the `Snapshots` resume path. The dill
  successor rule (02_rules): *no pickled models — persistence is
  script re-assembly + leaf snapshots; anything not reconstructible
  from (script, snapshot) is a design bug.*
- Async checkpointing is not built; the seam allows a
  background-thread store later (host copy of the boundary-synced
  carry, joined before the next snapshot).
- The function-level store surface (vs. a store class) and all
  function/field names here are spec-level; the manifest *contents*
  and atomicity rules are normative.

---

### `fr.io.resubmit` / `fr.slurm`

The `restart_module` successor, reduced to a plain function
(§6.3, d4_2 §5).

| Aspect | Value |
|--------|-------|
| Kind | module-level functions |
| Pytree | n/a (host functions) |
| Task | 2.4 |
| Design refs | 04 §6.3 (walltime), d4_2 §5, d4_3 §4 |

```python
# framework2/io/slurm.py — also surfaced as fr.slurm
def in_job() -> bool:                                          # 2.4
    """Whether the process runs inside a SLURM allocation."""
    ...


def job_id() -> str | None:                                    # 2.4
    """The current SLURM job id, if any."""
    ...


def resubmit_current() -> None:                                # 2.4
    """Re-submit the current job: the old scontrol -> sbatch
    auto-detection, kept as a plain function."""
    ...


# framework2/io/__init__.py
def resubmit() -> Callable[[], None]:                          # 2.4
    """Factory for the on_walltime action wrapping
    fr.slurm.resubmit_current (sketch 7.7 spelling)."""
    ...
```

Notes: rank-0-only submission guard on multi-process; the
walltime/interrupt *consensus* across processes is the parked
3.2/3.3 residual (Open questions). Helper names are spec-level; the
`fr.io.resubmit()` spelling is the acceptance surface's.

---

### `fr.ops.Session`

The ops front door: a context manager composing the normative
protocol-level utilities over N models (§6.3 hybrid, §11.2
decision 3b).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final; context manager |
| Pytree | host object, not a pytree |
| Task | 2.4 (single-model, `run()` substrate); 3.2 (multi-model maturation) |
| Design refs | 04 §6.3 (the hybrid), 09 §11.2-3b, CS-3/4/6/7/9/10/11/18, d4_2 (absorbed internals) |

```python
"""fr.ops.Session — plans, binds, advances, and cleans up."""
from __future__ import annotations

import fridom.framework2 as fr


class Session:
    """Context manager over N models + outputs/snapshots/progress;
    composes the walltime, progress, and stream-binding protocols."""

    def __init__(                                              # 2.4
        self,
        models: fr.Model | Sequence[fr.Model],
        *,
        outputs: Sequence[OutputStream] = (),
        snapshots: Snapshots | None = None,
        progress: bool | ProgressReporter = True,
        max_chunk: int | None = None,
        jit: bool = True,
        debug_nan: bool = False,
    ) -> None:
        """PROVISIONAL SIGNATURE (normative note, §6.3): the exact
        constructor is provisional until 2.4/3.2 experience; the
        protocols and the __enter__/advance/__exit__ duties below
        are the commitment."""
        ...

    @property
    def models(self) -> Mapping[str, fr.Model]:                # 2.4
        """The bound models, keyed by ``model.name``."""
        ...

    def __enter__(self) -> Session:                            # 2.4
        """Plan triggers, bind outputs, resume-check snapshots,
        install the interrupt handler, start progress/walltime."""
        ...

    @property
    def active(self) -> bool:                                  # 2.4
        """Keep-going predicate: no panic, no interrupt received,
        and predictive walltime headroom remains."""
        ...

    def advance(                                               # 2.4
        self,
        plan: Mapping[fr.Model | str, int] | None = None,
        /,
        **steps_by_name: int,
    ) -> Mapping[str, AdvanceResult]:
        """Advance each named model by its step count —
        dispatch-then-sync built in; s.advance(atm=10, ocn=1) is
        the sanctioned multi-model spelling."""
        ...

    @property
    def result(self) -> Mapping[str, RunResult]:               # 2.4
        """Aggregated per-model results (shared status enum)."""
        ...

    def __exit__(self, exc_type, exc, tb) -> bool:             # 2.4
        """Cleanup guarantees on every exit path; never suppresses
        (always returns False)."""
        ...
```

Semantics, invariants, error behavior:

- **Constructor.** Models are keyed by `model.name` (§6.1's log
  attribution slot); duplicate or missing names in a multi-model
  session raise a hinted error; a lone unnamed model gets the
  default key `"model"` (provisional detail). `outputs=` rejects a
  `Snapshots` instance with a hinted error (one resume path — the
  `snapshots=` kwarg is the only home). Multi-model output routing
  (which stream binds to which model) is a 3.2 maturation item; the
  single-model form is what 2.4 ships and what `run()` uses.
  `jit`/`debug_nan`/`max_chunk` mirror `run()`'s knobs (they are
  chunk-dispatch policy, which the Session owns).
- **`__enter__` duties** (normative order):
  1. **Plan**: lower every trigger through the public CS-7 planner
     (`lower_trigger`); build per-model boundary plans — union of
     trigger step sets ∪ snapshot steps ∪ last step, subdivided by
     `max_chunk` (auto ~256, documented meaning: host-sync
     granularity).
  2. **Bind outputs**: dedupe by resolved path (`IOCollisionError`),
     reject walltime triggers on data streams, dry-evaluate, create
     stores.
  3. **Resume-check snapshots**: find the newest complete snapshot,
     header-only manifest read, fingerprint check per model, load
     leaves, re-plan remaining steps against the absolute target,
     flip writers to `mode="a"` + `truncate_after(it_snap)`. **CS-18
     is enforced mechanically here**: on a multi-model resume the
     cross-model clock assertion runs (all restored clocks must sit
     on a common window boundary; the window index is derived from
     the clocks, never stored separately), and snapshot steps are
     planned onto **common window boundaries only** (snapshots of
     coupled runs happen only at window boundaries).
  4. Install the interrupt handler (restored at exit) and start the
     `ProgressReporter` (`on_run_start`) and the `WalltimeGuard`.
  AOT compilation (`lower().compile()`, per `(chunk_len, treedef)`,
  with the peak-memory report) happens at enter or on a model's
  first advance — an implementation choice; the commitment is the
  compile-vs-run accounting separation in `result`.
- **`active`** encapsulates exactly three conditions (§6.3): panic
  polling (cheap `model.panicked` host reads, CS-6), the graceful
  first-Ctrl-C flag, and the predictive walltime check
  (`WalltimeGuard.should_stop()`). It does **not** track target
  exhaustion — loop termination on a step/time target is the
  caller's condition (`run()` tracks its own remaining steps; a
  coupled driver loops `while s.active: s.advance(atm=10, ocn=1)`
  until its own criterion).
- **`advance` — dispatch-then-sync built in** (the correct
  concurrency order becomes the easy one, c3): all models' chunks
  are dispatched **before** any panic-flag read; the per-model panic
  reads (the primitive's only host synchronization, CS-4) happen
  after every dispatch. Overlap on disjoint devices then arrives
  free via jax async dispatch; the Session does this internally —
  the public non-blocking spelling stays reserved (`PendingAdvance`,
  below). Per call:
  - In a multi-model session the per-call step counts define **one
    coupling window**; the Session asserts post-window clock
    agreement (exact float64 equality — guaranteed by the CS-9
    master-dt rule: one designated master dt, every other model's dt
    derived by exact division, windows exact step counts in every
    participant).
  - Each model's quantum is subdivided at its planned boundaries; at
    each boundary the normative sequence runs: sync → panic check →
    writer flush (streams fire in binding order) → progress
    `on_chunk` → walltime check; a firing snapshot step writes the
    store (atomic) and rotates.
  - On interrupt or walltime the call returns early
    (`AdvanceResult.steps` < requested; `active` flips) — the
    in-flight chunk completes, the carry is consistent, zero steps
    lost, no exception on first Ctrl-C.
  - On panic the primitive raises `PanicError` (carrying the model
    `name=`) at the abort boundary, after the host NaN report
    (first-failure it/time, per-component non-finite counts with the
    chunk-end caveat). The error propagates through `advance` into
    `__exit__`.
  - **Exchange-then-advance is the caller's duty**: the Session
    cannot see `set_aux` exchanges; the CS-18 coupler protocol
    *page* (doc, not class) owns that rule (primes window 0;
    self-heals staleness). The Session's boundary ordering merely
    guarantees driver-side exchange writes never race output reads.
- **`result`** aggregates per-model `RunResult`s — plain,
  aggregatable, sharing the status enum
  (`COMPLETED|NAN_ABORT|WALLTIME|INTERRUPTED`) per CS-3/H4, with
  compile vs run seconds separated.
- **`__exit__` guarantees**, on *every* exit path (normal return,
  `PanicError`, `KeyboardInterrupt`, any other exception), in this
  order: **wait for in-flight device work → writer flush + close
  (partial output survives) → triggered snapshot → `on_walltime`** —
  each stage per its own firing condition (`on_walltime` fires only
  when the walltime guard caused the stop; it never fires on normal
  completion or panic). **On `PanicError`: writers are flushed, but
  there is NO automatic crash snapshot — the carries are left in
  memory for autopsy** (an `on_error=` policy is a possible later
  option, leave-open). The interrupt handler and progress reporter
  are torn down last (`on_run_end`).
- **Exception ordering**: `__exit__` always returns `False` — the
  Session never suppresses an exception. A cleanup failure while an
  exception is in flight is chained (`__context__`) and logged; the
  in-flight exception always wins and propagates.
- **Reentrancy / bare-use guards**: `advance`/`active`/`result`
  outside the `with` block raise a hinted `RuntimeError`; a Session
  is **single-use** — `__enter__` on an already-entered or
  already-exited Session raises; entering a Session over a model
  already bound to another active Session raises (store-collision
  protection; spec-level guard).
- **State = config + transient counters only** (CS-11): boundary
  plans, chunk-rate EMA, the interrupt flag, wall-clock and
  steps-done accounting — all discarded at exit. Nothing
  time-integral lives here; everything time-integral lives in the
  carries. Discarding a Session loses nothing that matters.
- **`run()` is reimplemented as a single-model Session loop**
  ([`model.md`](model.md) owns `run()`); the bitwise equivalence
  test extends: `run() ≡ user-written Session loop`, chunk for
  chunk. Notebooks keep `run()` and never see the `with`.
  `run(raise_on_nan=False)` catches the Session's `PanicError` and
  returns `RunResult(NAN_ABORT)` (notebooks want the carry).

---

### The normative ops protocols — `WalltimeGuard`, `ProgressReporter`

Free-standing protocol-level utilities (CS-3: `run()` must be
implementable purely on `advance` + these; they are the normative
spec the Session composes — its signature is provisional, these are
not).

| Aspect | Value |
|--------|-------|
| Kind | concrete utility (`WalltimeGuard`) + `typing.Protocol` (`ProgressReporter`) |
| Pytree | host objects, not pytrees |
| Task | 2.4 |
| Design refs | 04 §6.3 (predictive walltime, boundary sequence), d4_2 §4–5, CS-3 |

```python
"""Protocol-level ops utilities: importable by any driver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class WalltimeGuard:
    """Predictive walltime budget check."""

    def __init__(                                              # 2.4
        self,
        budget: float,
        *,
        snapshot_margin: float | None = None,
    ) -> None:
        """budget in wall seconds; snapshot_margin reserves store
        time (measured from the first snapshot when None)."""
        ...

    def on_chunk(self, wall_seconds: float) -> None:           # 2.4
        """Feed one chunk's wall time into the smoothed rate."""
        ...

    def should_stop(self) -> bool:                             # 2.4
        """elapsed + predicted_next_chunk + snapshot_margin >
        budget — stops BEFORE the chunk that would blow it."""
        ...


@dataclass(frozen=True)
class ChunkStats:
    """Per-chunk observation (payload provisional, grows
    compatibly; the three reporter hook NAMES are normative)."""

    name: str | None             # model attribution              2.4
    iteration: int                                             # 2.4
    time: float                                                # 2.4
    steps_done: int                                            # 2.4
    wall_seconds: float                                        # 2.4
    steps_per_second: float                                    # 2.4


class ProgressReporter(Protocol):
    """The progress protocol; ``progress=False | Reporter()``."""

    def on_run_start(self, *, models, n_steps: int | None) -> None:
        """Called once at __enter__ / run start."""            # 2.4
        ...

    def on_chunk(self, stats: ChunkStats) -> None:             # 2.4
        """Called at each chunk boundary (after the panic sync)."""
        ...

    def on_run_end(self, results) -> None:                     # 2.4
        """Called once on every exit path (teardown)."""
        ...
```

Notes:

- **WalltimeGuard**: the budget comes from the Snapshots walltime
  trigger; prediction uses the smoothed (EMA) chunk rate — the same
  observation stream the reporter renders. A single chunk exceeding
  the remaining allocation still dies — the `max_chunk` margin is
  the documented mitigation (d4_3 risk 3). Multi-process consensus
  (rank-0 broadcast at boundaries) is the parked 3.2/3.3 residual.
- **ProgressReporter**: the old ProgressBar module dies; this
  host-side protocol replaces it. Per-chunk cadence — steps/sec,
  model time, ETA — with **one scalar D2H piggybacking on the panic
  sync** (zero extra host syncs). The default renderer ports the
  old tqdm/StringIO file-output handling as host code, with a
  rank-0-only guard. In-trace `io_callback` progress ticks are
  rejected (archive, d4_2 §4). The three hook names are the
  normative commitment; the payload dataclass is provisional and
  grows compatibly.

---

### `PendingAdvance` (reserved)

The reserved non-blocking `advance` handle (CS-5); **shape only —
full semantics leave-open**.

| Aspect | Value |
|--------|-------|
| Kind | concrete, final (reserved — not built) |
| Pytree | host object, not a pytree |
| Task | designed-for |
| Design refs | 04 §6.3 (reservation), CS-4/CS-5, c3 (H1/H2) |

```python
class PendingAdvance:
    """RESERVED: handle returned by ``advance(steps, sync=False)``."""

    def wait(self) -> AdvanceResult:                    # designed-for
        """Block until the dispatched steps commit; the panic read
        happens here. (Indicative only — semantics leave-open.)"""
        ...
```

Notes:

- What is reserved (and only this): the primitive's signature is
  `advance(steps, *, sync=True)` from day one; `sync=False` returns
  a `PendingAdvance` handle instead of an `AdvanceResult`. Full
  semantics — chaining, error-surfacing timing, interaction with
  boundary IO — are **leave-open** (§11.2/CS-5); the `wait()` sketch
  above is indicative, not normative.
- Why it exists: under the **no-hidden-sync invariant** (CS-4: the
  panic read is `advance`'s only host synchronization; the committed
  carry may hold pending, future-valued arrays), overlap of two
  models on disjoint devices needs only dispatch-then-sync ordering
  — the reservation keeps that a pure addition. The Session already
  implements the ordering internally without the public handle;
  nothing concurrent is *built*.

---

### Cross-owned types (pointers, not specs)

Owned by [`model.md`](model.md) and only consumed here — do not
respec:

- **`AdvanceResult`** — minimal frozen return of the primitive:
  steps done, panic `(flag, it)`, wall seconds (everything else is
  `RunResult`'s).
- **`RunResult`** — plain, aggregatable; status enum
  `COMPLETED|NAN_ABORT|WALLTIME|INTERRUPTED` shared with
  `AdvanceResult`'s consumers (CS-3/H4); steps, final it/time,
  compile vs run seconds, rates.
- **`PanicError`** — typed abort exception carrying the model
  `name=`; raised at the abort boundary; running a panicked carry
  raises with guidance.
- **`model.panicked`** (CS-6, cheap host-readable) and
  **`model.carry`** (CS-12, the opaque in-memory snapshot read) —
  Session consumes both.
- **`model.snapshot(path)` / `model.load_snapshot(path)`** — the
  assemble-then-load API over this cluster's store functions.

---

## Open questions

Parked residuals only ([`../07_open_threads.md`](../07_open_threads.md)
item 4 / §6.9); decided questions are not reopened.

1. **`truncate_after` for CSV** (TimeSeries): the tail-rewrite
   strategy keyed on the iteration column — pinned in the protocol
   so the sink must answer it; concrete mechanics decided in 2.6.
2. **Capture streams and post-assembly writer attach**: both are
   2.6 designed-fors behind the `OutputStream` protocol (the
   in-trace capture variant reuses the same output callables and
   trigger objects; attach needs only a bind-at-boundary entry
   point) — neither requires a contract change; the API spelling is
   2.6's.
3. **Multi-process walltime/interrupt consensus**: a rank-0
   consensus broadcast at chunk boundaries (walltime stop, Ctrl-C,
   resubmit-once) — deferred to 3.2/3.3 with the multi-host
   snapshot follow-up.
4. **`fr.at(times=)` under a later dt change** lands on different
   realized steps across resubmits; plan-time logging + realized
   times written to the axis mitigate — a tolerance warning is a
   possible 2.4 nicety (d4_3 risk 6).
