---
status: frozen
date: 2026-07-16
---

# Single-GPU memory ceiling — why 1024x512x512 OOMs after 512^3 fits

512^3 f64 nonhydro2 runs on one A100-80GB with a peak below 40 GB, so
2x the cells (1024x512x512, ~57 GB peak) "should" fit — but the
advective step OOMs at every `XLA_PYTHON_CLIENT_MEM_FRACTION`. This
note records the attribution (the ceiling is allocator *fragmentation*,
not capacity), the measured fixes, and the options rejected. Campaign
2026-07-16, one A100-80GB, jax/jaxlib 0.10.2, `bench_step.py`'s flat
triply-periodic nonhydro cases, f64, AB3, `chunk_size` explicit.

## 1. What actually runs out

XLA emits the whole chunk's transients as **one monolithic
`preallocated-temp` allocation** that must be contiguous. Measured via
`compiled.memory_analysis()` and the buffer-assignment dumps:

| n | carry (donated args) | temp arena adv | temp lin | peak adv |
|---|---|---|---|---|
| 256^3 | 1.70 GiB | 1.96 | 1.30 | 3.66 |
| 512^3 | 13.31 GiB | 15.33 | 9.19 | 28.64 |
| 768x512^2 | 19.91 GiB | 22.94 | 13.75 | 42.85 |
| 1024x512^2 | 26.51 GiB | **30.55** | 20.35 | 57.07 |

Everything is linear in cell count; peak = carry + temp because the
chunk output fully aliases the donated carry (13 padded f64 cubes:
5 state u,v,w,b,p + 8 AB3-ring tendencies; `donate_argnums=(2,)` in
`_compile_chunk` — already optimal). The eigenvalue diagonal is *not*
materialized (largest constant in the module: 4 KiB; only 1-D
wavenumber vectors appear — XLA fuses the broadcast sum + reciprocal
into the Hadamard divide), and exactly one c128 spectral cube is live
at a time. Two census hypotheses died here: no diagonal cube, no
missing donation.

At 1024x512^2 the 57 GiB peak fits the 73.6 GiB pool (fraction 0.92),
yet the first chunk execution dies:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 30.55GiB.
```

with ~47 GiB free — **no contiguous hole >= 30.55 GiB**. The BFC
allocator cannot compact (buffers are pinned), so placement, not
capacity, is the ceiling.

## 2. Where the fragmentation comes from

Setup churn scatters the 13 long-lived carry cubes through the pool:

| rank | source | transient at 1024x512^2 |
|---|---|---|
| 1 | `_canonicalize` at construction — non-donating jitted identity over the full carry | +18.4 GiB (peak 44.9) |
| 2 | `_canonicalize` at every `set_fields` commit — full 14-cube copy | +26.5 GiB (peak 59.2) |
| 3 | `model.carry` copy-on-read property (harness-side; use `model.state`) | +26.5 GiB per read |
| 4 | set-IC meshgrid + IC evaluation | ~+6.2 GiB |

## 3. Why unroll=3 sets the arena size (and must stay)

The advective arena holds ~15 co-live 2.04 GiB cube slots at
`scan_unroll=3` vs exactly 10 at unroll=1 (20.39 GiB — which fits and
runs at 185.8 ms/step vs 152.9 at unroll=3, i.e. unroll=1 costs +23%
at this size and +11% at 512^3). The extra co-living is
**dataflow-forced by the AB3 ring**: with unroll = order = 3 the
spliced step bodies' fresh tendencies must stay live until the later
bodies consume them (up to 2 levels x 4 prognostic cubes). The HLO
shows XLA already serializes the bodies (control-predecessors; one FFT
transpose live at peak), and no scheduler flag changes the arena
(`--xla_gpu_memory_limit_slop_factor` 95/80/50,
latency-hiding-scheduler off: temp byte-identical). 30.55 GiB is the
dataflow minimum for unroll=3 — so the fix must target *placement*.

## 4. Fixes (landed on `perf/gpu-memory-ceiling`)

1. **`_canonicalize` donates its carry** (`model.py`): the jitted
   identity aliases in -> out, removing both full-carry setup spikes
   (churn ranks 1-2). Safe because `_rehome` always re-pads incoming
   values through `decomposition.pad` (user buffers never become carry
   leaves) and all public accessors return copies. Two hazards are
   guarded by a host-side `_prepare_for_donation`: (a) the AB ring is
   warmed as `(zeros,) * (order-1)` — the SAME buffer in every slot —
   and jax 0.10.2 raises "Attempt to donate the same buffer twice" on
   aliased donated leaves (duplicates get a copy); (b) `assemble()`
   does not clone modules, so module parameter leaves ARE carry
   leaves, and jax donates even 0-d scalars — sub-1-MiB leaves are
   copied so a caller-held module handle is never invalidated. Only
   the big unique cubes are donated; that is where the win is.
   Measured: construction peak 44.87 -> 26.51 GiB (canonicalize
   transient 0.000), set-IC peak 59.18 -> 36.67 GiB at 1024x512^2.
2. **One-time carry defragmentation before the first chunk**
   (`model.py`): armed at construction and every `_commit`, `advance`
   sequentially copies each >= 1 MiB carry leaf, blocks, and deletes
   the original (`FRIDOM_DISABLE_DEFRAG=1` opts out; GPU backends
   only). Sequential copy-with-forced-free migrates cubes into small
   holes and coalesces the freed stretches — a bulk `tree_map` copy
   allocates the whole second set before freeing anything and does
   NOT defragment. Bitwise value-preserving; one-off cost tens of ms.

Validated (patched worktree, one A100): 1024x512x512 advective
unroll=3 under BFC `MEM_FRACTION=0.92` fits 3/3 with no harness-side
tricks — the 30.55 GiB arena places (`largest_alloc` confirms), peak
59.1 GiB, 153.0 ms/step. 256^3 one-chunk output bitwise identical to
unpatched dev; per-step perf unchanged (512^3 adv 75.47 / lin 50.01 /
256^3 adv 9.35 vs baselines 75.4 / 50.0 / 9.2). The stock 0.75
memory fraction (60 GiB pool) still cannot hold the 59 GiB workload +
contiguous arena — very large single-GPU runs keep needing
`XLA_PYTHON_CLIENT_MEM_FRACTION~0.92`, exactly as 512^3 already did.

## 5. The env-only alternative: `cuda_async`

`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` fits 1024x512^2 with **zero**
measured per-step cost at 256^3/512^3 (<0.1%, inside noise): the
cudaMallocAsync pool is VMM-backed, so a "contiguous" 30.55 GiB
virtual range maps physically scattered pages — fragmentation stops
existing. Caveats that kept it from becoming a default: keep
preallocation ON (`PREALLOCATE=false` remaps pages every chunk: +19%
at 512^3, +48% at 1024x512^2); pair with `MEM_FRACTION~0.9` (the 0.75
default fit with ~0.3 GiB margin over the set-IC peak, pre-fix);
NCCL/multi-GPU interaction unvalidated (single GPU only in this
campaign); the test suite's `PREALLOCATE=false` sharing would make it
slow there. Recommended env for very large single-GPU runs; revisit as
a default after a multi-GPU validation.

## 6. Rejected / dead ends (measured)

- **Scheduler flags to shrink the arena**: none change a byte (§3).
- **Global unroll=1**: fits but −11% at 512^3, −23% at 1024x512^2.
- **`chunk_size=1`**: length-1 scan inlines straight-line, arena
  ~unchanged (30.59 GiB) — no help.
- **Raising MEM_FRACTION**: 0.95 adds pool but no 30.55 GiB hole.
- **`XLA_PYTHON_CLIENT_ALLOCATOR=platform`**: grow-on-demand inflates
  the true high-water ~2.2x over packed BFC; still OOMs.
- **Arena-content source fix**: the co-live cubes are the AB3 ring's
  dataflow, not waste; `tendency_sums` is dead-code-eliminated (an
  earlier census hypothesis) and pad/ghost-fill copies are absorbed.

Raw logs/dumps: session scratch (`~/.claude/jobs/c4ef6662/tmp/`) —
`logs/`, `dump/d{512,1024}adv_c7/`, `p2_repack.py`, `p3_churn.py`.

## 7. The 4-GPU signature (2026-07-17): the same fix, not remat

The Oceananigans comparison's 4-GPU max-fit probe
(`benchmarks/comparison/fridom_multi/bench_maxfit.py`, `cmp_maxfit`)
recorded, **2026-07-16 10:01 UTC**, that 1024x1024x512 fits (~30.7
GiB/GPU) but **1024x1024x768 "dies in compile (remat)"**, quoting an
`hlo_rematerialization.cc` line. That attribution is wrong and the death
is already closed.

Config (matched protocol, z-walled flat nonhydro): `(2*pi, 2*pi, 1)`,
(Periodic, Periodic, Bounded-z), `CenteredAdvection`, `FPlaneCoriolis
f0=1`, `dsqr=0.25`, `dt=1e-3`, `chunk_size=20`; single-process GSPMD over
4x A100-80GB; `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`.
Footprint via `peak_bytes_in_use` after syncing the LIVE `_carry` — never
`model.state` (copy-on-read). Driver: session scratch (`driver.py`).

### Verdict

1024x1024x768 **fits on current dev** at ~44 GiB/GPU steady. The original
death was per-device **BFC arena fragmentation** — the same kind as the
single-GPU ceiling (§1-2), one rung larger — **not** rematerialization.
It was closed by the SAME fix (§4: carry donation + pre-chunk defrag, dev
`b6b24644`), which merged **2026-07-16 15:30 CEST, 5.5 h AFTER the
observation**. The `hlo_rematerialization.cc` line is a **non-fatal
warning**, off the failure path.

### Buffer census (per device, `compiled.memory_analysis()`)

| shape | arg (carry) | temp arena | footprint | measured peak GPU0 |
|---|---|---|---|---|
| 1024x1024x512  | 13.40 | 11.35 | 24.75 | 29.28 |
| 1024x1024x768  | 20.06 | 18.36 | 38.42 | 43.88 |
| 1024x1024x1024 | ~26.7 | 24.71 | ~51.4 | 54.44 (OOM) |

(GiB.) Output aliases the donated carry, so footprint = arg + temp.
Everything is linear in per-device cells (768/512 = 1.5x: arg x1.50, temp
x1.62). GPU0 carries ~4 GiB of replicated buffers the sharded devices do
not (25.76 vs 29.28 at 512; 39.92 vs 43.88 at 768). The 768 footprint
(38-44 GiB) sits well under the 60 GiB pool (`MEM_FRACTION` 0.75) — no
capacity or remat problem.

### The decisive knob: allocator/placement flips it (-> fragmentation)

| # | code | allocator / knob | 768 outcome |
|---|---|---|---|
| 1 | current dev | BFC 0.75 (default) | FIT 43.88 GiB |
| 2 | current dev | BFC 0.75 + copy-on-read churn | FIT 43.88 |
| 3 | current dev | BFC 0.75, `DISABLE_DEFRAG=1` + churn | FIT 43.88 |
| 4 | current dev | BFC 0.75, **no** fusion flag | FIT 43.88 (arena =) |
| 5 | **pre-fix `4025f121~1`** | **BFC 0.75** | **OOM** |
| 6 | pre-fix `4025f121~1` | cuda_async 0.75 | FIT 44.74 |

Row 5 reproduces the original death directly — revert only the
donation+defrag commit and 1024x1024x768 raises, at chunk **execution**:

```
bfc_allocator.cc:514] Allocator (GPU_0_bfc) ran out of memory trying to
allocate 18.36GiB ... If the cause is memory fragmentation maybe the
environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve
the situation.  [free map: *___***___****...]
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 18.36GiB.
[executable_name='jit__chunk_body']
```

The failing 18.36 GiB is exactly the `temp_size` arena: it cannot be
placed as one contiguous block in a pool fragmented by the non-donating
`_canonicalize` set-IC churn (peak 44.7 GiB reached in a 60 GiB pool, no
18.36 GiB hole). XLA itself names fragmentation and points at cuda_async.
Row 6 confirms it — pre-fix + cuda_async (VMM maps the "contiguous" arena
onto scattered pages) fits, which also **validates cuda_async on
multi-GPU**, the open caveat of §5. Rows 2-4 show the current-dev fit is
robust: copy-on-read churn, defrag-disabled (donation alone suffices),
and the fusion flag toggled all fit, arena byte-identical.

The original JSON captured only the (non-fatal) remat WARNING, truncated
at "down from 62..."; the fatal `RESOURCE_EXHAUSTED` that followed it in
stderr was dropped — which is why the death was misnamed "remat". On
current dev the warning still fires on a COLD compile (identical text:
budget 57.47 GiB = 0.95 x 0.75 x 80, vs an "irreducible" estimate of
61.94 GiB) yet the compile proceeds and the program runs at a true ~44
GiB peak — the remat estimate is a pessimistic upper bound ~1.4x the real
buffer-assignment peak. Served warm from `.jax_cache` the warning never
appears and the verdict is unchanged: it is not on the failure path.

### Lever for a user who needs 1024x1024x768 on 4 GPUs

None — it fits out of the box at the default BFC 0.75 on current dev.

The next rung, 1024x1024x1024 (~51 GiB/GPU footprint), is a harder wall
and BFC does **not** clear it at either fraction: 0.75 OOMs on the 24.71
GiB arena with GPU0 at 54 GiB in the 60 GiB pool, and **0.92 OOMs too** —
GPU0 (which carries the ~4-8 GiB of replicated buffers the sharded
devices do not) cannot place the contiguous 24.71 GiB arena even in the
73.6 GiB pool (fragmented free-map `***___***___***`). So this rung is
`cuda_async`'s (VMM has no contiguous-arena requirement; row 6 validates
it multi-GPU), not a MEM_FRACTION bump. Operational note: the single-
device OOM did not fail cleanly — it left the other three ranks **hung on
the NCCL clique barrier** (`rendezvous.cc: ... may be stuck`), the same
dead-participant-blocks-the-collective mode AGENTS.md flags for `srun`;
guard large single-process multi-device probes under `timeout`.
