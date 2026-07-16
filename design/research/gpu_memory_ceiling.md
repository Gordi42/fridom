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
