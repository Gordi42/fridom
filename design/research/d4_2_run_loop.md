---
status: frozen
date: 2026-07-07
---

# D4.2 — The run loop (ROADMAP 2.4)

Research report (see [`README.md`](README.md) for status).

> **Reconciliation note**: this report assumed in-trace writer
> triggers; the resolved design adopts d4_3's chunk-boundary
> evaluation — the stated fallback here (chunk boundaries derived
> from trigger cadences) is what applies. Everything else survives.

## 1. The `run()` surface

`run(steps=None, *, runlen=None, end_time=None, chunk=None,
walltime=..., progress=True, jit=True, debug_nan=False,
raise_on_nan=False) -> RunResult` *(walltime/snapshot config moved
into `fr.io.Snapshots` per d4_3)*.

- **`steps=` primary**; `runlen`/`end_time` reduce to
  `ceil((end − t0)/dt)` with an epsilon guard — the old overshoot
  semantics. Exactly one of the three. `start_step`/`start_time`
  die (the clock is carry; resetting it is a lifecycle method).
- **Repeated `run()` continues from the carry** and is bitwise
  identical to one uninterrupted run, warm-up included.
- **`run_backward` stays dead**: flip the dt leaf, then `run()`.
- **`RunResult`**: status (`COMPLETED|NAN_ABORT|WALLTIME|
  INTERRUPTED`), steps done, final it/time, `compile_s`/`run_s`,
  steps/sec. Scripts and SLURM drivers branch on it; `run()` never
  `sys.exit`s.
- **Panic gate at entry**: running a panicked carry raises with
  guidance; `set_fields`/`set_state`/`load_snapshot` clear the flag.
- **Compile-time reporting** replaces `_execute_first_time_step`:
  AOT `jitted_chunk.lower(carry).compile()` before the loop — exact
  compile/run separation, plus `memory_analysis()` (peak device
  memory — new capability). Executables cached per
  `(chunk_len, treedef)`; the D3 warm-up counter guarantees one
  compile per length. Document the persistent compilation cache for
  SLURM resubmits.

## 2. Chunk mechanics

```python
@partial(jax.jit, static_argnames=("n",), donate_argnums=(0,))
def _run_chunk(carry, n):
    carry, _ = lax.scan(lambda c, _: (composed_step(c), None),
                        carry, xs=None, length=n)
    return carry
```

Loop-invariant stepper/schedule pytrees are non-donated arguments
(not closure captures) so leaf updates never retrace.

- **Remainder: two compiled lengths `{C, 1}`** — a distinct scan
  length is a full recompile (rejected per-r compiles); pad-and-mask
  rejected (reimports the no-op machinery §3 rejects, masks clock
  and triggers). `chunk(1)` doubles as the small-run path.
- **Chunk length**: user knob, auto default ~256; documented meaning
  "host-sync granularity" (progress, Ctrl-C latency, walltime
  checks, NaN-abort waste, snapshot boundaries). Under d4_3's model,
  boundaries derive from trigger unions with `max_chunk`
  subdivision.
- **Donation: yes** (`donate_argnums=(0,)`) — the carry dominates
  memory (state + ring buffers); treedefs match by construction.
  Rules: the driver assigns the output at dispatch (never holds the
  pre-chunk carry); `debug_nan` replay copies first (the exception).
  No pipelined dispatch: the panic check needs one scalar sync per
  chunk anyway — that read is *the* host sync point.

## 3. NaN / panic mechanism

**Per-step flag write + chunk-boundary abort; NO `lax.cond` no-op
wrapper.**

- Cond-wrapper cost analysis: a data-dependent HLO Conditional in
  the scan body forces a per-step predicate sync on GPU (pipeline
  bubble) and blocks CUDA-graph capture/whole-loop fusion; both
  branches compile regardless. It taxes every healthy step to skip
  ≤C−1 steps of garbage in the one bad chunk. The select variant
  does all the work plus a carry-sized copy. Rejected; recorded as
  an opt-in retrofit (the flag is already in the carry).
- The per-step write is kept and cheap: S5 computes
  `bad = OR(any(~isfinite(x)))` over PROGNOSTIC leaves — one
  fusable reduction. `isfinite`, not `isnan` (catches the Inf that
  precedes NaN). Carry record: `panic.flag` (sticky) + `panic.it`
  (first-failure iteration).
- Why per-step, not chunk-boundary-only: **NaN laundering** — a
  positivity clamp (`where(c > 0, c, 0)`: NaN>0 is False) erases
  NaNs before the boundary sees them; the boundary check also loses
  the failure iteration and lets garbage flow into S6 accumulators.
- **Abort flow**: read `panic.flag` per chunk (the sync); on panic
  stop, build the host-side report (first-failure it/time,
  per-component non-finite counts with the chunk-end caveat),
  return `RunResult(NAN_ABORT)` — no exception by default
  (notebooks want the carry); `raise_on_nan=True` for scripts.
  **Opt-in `debug_nan=True`**: keep a chunk-start carry copy and
  replay to the exact first-bad step with `chunk(1)`.

## 4. Progress and the interrupt contract

- The ProgressBar module dies; a host-side `ProgressReporter` owned
  by `run()` replaces it — per-chunk cadence (steps/sec, model
  time, ETA; one scalar D2H piggybacking on the panic sync); the
  old tqdm/StringIO file-output handling ports as host code;
  rank-0-only guard. In-trace `io_callback` progress ticks
  rejected. `progress=False|Reporter()`
  (`on_run_start/on_chunk/on_run_end` protocol).
- **Ctrl-C contract**: the in-flight chunk completes; the carry is
  consistent at that boundary; **zero steps lost**; no exception on
  first Ctrl-C (`RunResult(INTERRUPTED)`); second Ctrl-C re-raises
  with the carry still valid once the device drains.

## 5. Walltime stopping (restart_module successor)

Trigger/snapshot/reload/resubmit separate (d4_3 owns
snapshot/reload): the budget check is **predictive** —
`elapsed + predicted_next_chunk + snapshot_margin > budget` at each
boundary, using the reporter's smoothed chunk rate — it stops
*before* the chunk that would blow the budget. Resubmission is a
plain callback (`fr.slurm.resubmit` keeping the old
`scontrol`→`sbatch` auto-detection as a function). Driver-script
pattern: assemble → `restore-if-snapshot else ICs` →
`run(end_time=..., ...)` — `end_time` is the natural resumable
spelling.

## 6. Timers, profiling, debug mode

The old `mset.timer` is impossible under one jitted scan (no host
boundary between modules; fusion erases the concept). Three tiers:

1. **Per-chunk wall accounting (always on)**: RunResult +
   end-of-run stats — answers what the timer was actually used for.
2. **`jax.profiler`** with **`jax.named_scope(f"{module}/{term}")`**
   stamped by the composer at compose time — profile rows carry
   module names (fusion caveat documented). Unconditional,
   zero runtime cost. Sugar: `run(profile="logdir")`.
3. **`run(jit=False)` eager mode — feasible by construction and
   load-bearing**: the composed step is plain Python over fields
   (the halo tracer *requires* it), so a Python for-loop over the
   eager step works; the composer's wrapper can then time each
   term/stage (`block_until_ready`) and print the per-module table
   — the timer's true successor, on demand. Also: pdb,
   `jax_debug_nans` (faults at the first NaN *operation* — the
   sharp companion to §3's per-step flag). Document "orders of
   magnitude slower; steps<=10". Debugging ladder: chunk stats →
   NaN replay → jit=False table → profiler.

## 7. Logging under trace

Assembly-time (rich: tables, schedule, compile notice + memory) →
chunk-boundary (run stats, notices) → in-trace only via
`jax.debug.print` as the documented escape hatch ("debugging
sessions, never shipped modules"); shipped step-cadence monitoring
is a DIAGNOSTIC accumulator or writer expression, not logging.

## 8. Risks / open questions

1. The GPU conditional-cost claim decides only an optional feature's
   default — benchmark once during 2.4 and record.
2. S5 isfinite fusion assumed — if >~2% on a real model, fall back
   to strided/every-k masking (measure-then-decide).
3. Chunk default 256 and `{C,1}`: revisit if notebooks show pain.
4. **Donation vs `model.state` reads**: boundary reads are safe;
   users keeping live views across a later `run()` may hit
   deleted-buffer errors — test and document (possibly
   copy-on-read).
5. Boundary sequence: sync → panic check → writer flush → progress
   → walltime check.
6. Time targets under future adaptive dt: only the reduction
   changes; `end_time=` accommodates.
7. Multi-process: walltime/interrupt need a rank-0 consensus
   broadcast at boundaries — 3.2/3.3 note.
8. `update_parameters` mid-session: the run loop only requires
   treedef preservation (rewarm is d4_4's).

## 9. Sketch

(Host-side `run()` pseudocode and the notebook session — see the
report body in the consolidated design; key lines: assign donated
output at dispatch; one scalar sync per chunk; `chunk(1)` remainder;
predictive walltime check; `RunResult` return.)
