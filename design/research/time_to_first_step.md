---
status: frozen
date: 2026-07-16
---

# Time-to-first-step — attribution and validated fixes

Why a cold nonhydro2 run takes ~8–11 s to produce its first step, where
every second goes, and three measured mitigations. Campaign goal: total
compile < 2 s for the centered scheme with zero per-step regression.

Canonical config throughout: `bench_step.py`'s flat triply-periodic
nonhydro, `CenteredAdvection`, FPlane `f0=1`, `dsqr=0.25`, AB3,
`chunk_size=50`, one A100, jax/jaxlib 0.10.2, cold process, no
persistent compilation cache unless stated.

## 1. The headline number was a metric artifact

The fridom-vs-Oceananigans REPORT's "fridom compile s" column (the
source of the "11 s at 512³" reading) is the raw wall time of the first
`model.advance(chunk)` — which *executes the whole first chunk*
(200–1000 steps) after compiling, while the Oceananigans column times a
single first `time_step!`. Subtracting a steady-state repetition of the
same chunk from the committed results gives the honest, nearly
size-independent compile cost:

| scheme | pure compile, 1 GPU | 2/4 GPU |
|---|---|---|
| linear | ~1.5 s | ~2.9–3.1 s |
| centered | ~2.0–2.3 s | ~3.4–3.6 s |
| upwind5 | ~3.5–4.2 s | — |
| weno5 | ~8.5–10 s | ~12–14 s |

At 512³ the centered row's "16.3 s" is ~14.2 s of stepping plus ~2.1 s
of compile. **Action for the comparison suite (out-of-tree):** report
compile separately — `fr.model.model._CHUNK_COMPILE_LOG` already records
per-executable compile seconds; `first_advance_s` should be split into
`chunk_compile_s` + execution. Note the reverse caveat: the Oceananigans
2.2 s metric excludes *its* model construction, and fridom's
`first_advance_s` excludes fridom's (see §2). A node-cold first CUDA
process additionally pays 2–3× on everything (one measured 26.6 s
end-to-end); quote warm-node numbers.

## 2. True attribution of a cold run (256³, warm node, ~8.4 s)

| phase | s | nature |
|---|---|---|
| `import jax` + backend init | 1.2 | fixed |
| lazypimp symbol resolve | 0.55 | fixed |
| grid construction | 0.004 | negligible |
| **model construction** | **3.5** | **113 throwaway eager XLA compiles** |
| `set_fields` (ICs) | 1.1 | data-bound (0.04 s at 64³) |
| first `advance(50)` | 2.0 | **one chunk compile, 1.45 s XLA** + exec |
| second `advance(50)` | 0.375 | steady state (7.50 ms/step) |

Total XLA compile ≈ 4.3–5.1 s, split into two buckets. Everything is
XLA-backend time: tracing + StableHLO lowering are < 0.2 s per bucket,
and CPU-platform compile times match GPU (the cost is XLA-general, not
GPU codegen). Both buckets are grid-size-independent (identical jaxpr
and HLO at 64³ vs 256³; only baked constants scale).

**Bucket 1 — construction, 113 compiles, ~2.7–3.0 s XLA.** Root cause:
`TendencyComposer.dry_run` (`model/composer.py`) eagerly executes every
term/stage/solver hook once on a zero-valued full-size state; every
eager primitive dispatches its own one-shot compile (~24 ms each).
Sub-costs (cProfile): spectral eigenvalue materialization inside the dry
projection (`spectral_solve.py` `_materialize`, built afresh and
discarded) 1.40 s; the dry solve 0.69 s; dry tendency terms 1.16 s;
`.to()` transforms 0.89 s. None of these kernels are reused by the step.

**Bucket 2 — the chunk executable, 1.45 s XLA.** Exactly one step
compile (no retracing; the AOT `(record, n, treedef, sigs)` cache
works). Composition: ~0.9 s single-step body ≈ 0.8 s framework floor
(pressure + halo seal + AB combine + scan) + 0.46 s advection; the AB3
`scan_unroll=3` replication adds ~0.4–0.54 s (opt-HLO 1645→4672 ops).
Advection is 45% of the step graph (452/999 deep jaxpr eqns): 4 advected
fields × 3 axes, Python-unrolled, each pair ≈ 2 interpolations +
product + difference. Frame plumbing is the hidden bulk: ~70 `pad`
(storage↔true round trips), ~60 `dynamic_update_slice` (carry ghost
seal), ~12 `take` (index-map fills) per body, ×3 unrolled.

**Scheme scaling (64³, cold):** compile tracks unoptimized-HLO volume at
~O(ops^1.35) through the same XLA passes — centered 1.58 s / 1921 ops,
upwind5 2.58 s / 2733, weno5 7.11 s / 5829. Construction grows too
(113 → 150 → 165 compiles): the dry pass executes the heavier
reconstruction kernels eagerly. Nothing weno-specific is pathological;
it is simply more HLO.

**Dead ends (measured, do not revisit without new evidence):**
`--xla_gpu_autotune_level=0` saves < 0.25 s;
`--xla_gpu_force_compilation_parallelism` no help;
`--xla_gpu_enable_llvm_module_compilation_parallelism` unsupported on
jaxlib 0.10.2; construction compiles run serially (CPU/wall 0.71) and
the chunk compile is only ~2.3×-threaded.

## 3. Validated fixes (prototyped, measured; patches preserved)

Patches: `~/work/fridom/compile-investigation-2026-07-16/{dryrun,
async_compile}.patch`. **Update, same day: 3a and 3b are LANDED on dev**
(merge `7842242b`: `model: validate dry_run by abstract evaluation`,
`fridom: enable the persistent compilation cache by default` — the
cache lives in `src/fridom/_compile_cache.py`, env knobs
`FRIDOM_DISABLE_COMPILE_CACHE` / `FRIDOM_JAX_CACHE_DIR`, per-rank
subdirs under an initialized `jax.distributed`, and the test suite's
conftest keeps precedence). Post-merge 64³ GPU: cold TTFS 7.5→5.05 s,
warm 2.83 s, per-step unchanged. **3c LANDED 2026-07-18** behind the
default-off `Model(async_chunk_compile=True)` knob (the shipped version
drops the `eager1` mode — measured strictly worse — and serves the lenC
unroll-1 tier).

### 3a. dry_run under `jax.eval_shape` — kills bucket 1 (~3 s → ~0.1 s)

One hunk in `model/composer.py`: move the dry pass's array-touching body
into a zero-argument closure and run `jax.eval_shape(_evaluate)`; keep
the two Python-side lints outside. Measured: construction compiles
111 → 12 (the 12 residuals are state/carry allocation elsewhere in
assembly, size-independent), 256³ GPU build 10.1 → 1.5 s under
contention (~3.5 → ~1.0 s uncontended), dry_run itself 3.46 s → 0.09 s.
10-step state **bitwise identical**; 216 composer/assembly/model/
nonhydro2 tests pass; ruff clean.

Why it is sound: dry_run consumes only component *names* and *spaces*
(`frozenset(result)` keys, `function_space.bare` comparisons, the
advances string-set check) — never host values — and every hook is
already trace-safe because `compose()` is jitted into the chunk. The
zero-arg closure is load-bearing: it also captures the grid-derived
eigenvalue build (the 1.40 s), which a state-only abstraction would
leave eager. Caveat: a hook branching on traced values would now raise
`TracerBoolConversionError` instead of running — but such a hook already
cannot survive the jitted step, so no user-facing regression.

### 3b. Persistent compilation cache — warm runs (needs threshold 0)

`jax_compilation_cache_dir` + **`jax_persistent_cache_min_compile_time_secs=0`**:
warm-start TTFS 8.85 → 4.61 s at 256³ (construction 3.7 → 0.9 s, first
advance 2.3 → 0.8 s), 7.54 → 3.10 s at 64³; per-step unchanged;
~1–2 MB and 116 entries per configuration. At jax's default threshold
(1.0 s) only the chunk executable is cached and construction stays at
~3.5 s — the threshold is the whole trick. **No cross-size reuse**
(shapes/constants are baked into the key): each grid size pays one cold
write. Correctness is safe (keyed on HLO + jaxlib + backend + flags;
atomic writes; silent fallback on cache errors, so NFS is low-risk);
the only care is unbounded growth (`jax_compilation_cache_max_size`
defaults to no cap). Proposal: enable at import in `fridom/__init__.py`
next to the x64 setup — XDG cache dir, `FRIDOM_JAX_CACHE_DIR` override,
`FRIDOM_DISABLE_COMPILE_CACHE=1` opt-out. Complementary to 3a/3c: disk
cache serves reruns; the others serve the genuinely-cold first run.

### 3c. Two-tier async chunk compile — hides part of bucket 2

`lowered.compile()` releases the GIL (a 37 s background compile inflated
a main-thread Python loop by ×1.01), so: on a chunk-cache miss, lower
the full unroll-3 chunk on the main thread (must happen *before* the
carry is donated — a Lowered holds HLO, not buffers), compile an
unroll-1 variant synchronously (~1.0 s), serve it while the unroll-3
compile runs in a daemon thread, swap atomically at a chunk boundary.
Measured (lenC1 variant): first-advance 2.13 → 1.61 s (−24%) at 256³,
1.87 → 1.29 s (−31%) at 64³; steady per-step back to baseline 7.50 ms
after the swap (~1.4 s wall, independent of size); transient and final
states **bitwise identical** to pure-unroll-3 on this config (single
GPU; multi-device FP-reassoc could differ — re-verify there). Serving
the n=1 tail executable in a loop instead ("eager1") is strictly worse.
Assessment: correct and modest (~0.5–0.6 s); worth shipping only behind
a default-off flag for interactive use; the ~0.9 s step-body compile is
the floor no unroll trick beats. Sharp edges recorded in the patch:
donation ordering, out_shardings pin shared by both tiers (swap is
reshard-free), background-failure propagation on next call.

**LANDED 2026-07-18** behind the default-off knob
`Model(async_chunk_compile=True)` (`step_chunk(..., async_compile=)`,
`_TwoTier` holder, `_lower_chunk`/`_compile_chunk` split, module-level
`_CHUNK_LOCK` guarding the swap). The shipped version drops the `eager1`
mode (measured strictly worse) and serves the lenC unroll-1 tier only;
the prototype's global `_ASYNC_CFG` config is gone (the knob rides the
`Model`). Miss dispatch: async only when `async_compile` and the
natural unroll > 1 and n > 1, else the legacy synchronous single-tier
compile; the length-1 tails always take the sync path. Forced-4 CPU
kept the bitwise-equality invariant (no `single_device` mark needed).
Tests: `tests/model/test_step_chunk_async.py`.

### Rejected: permanent lower unroll

unroll=2 on the *advective* config costs **+10.5%/step** (8.27 vs
7.49 ms at 256³; the earlier ~4% figure was linear-512³) for only
0.56 s of one-time compile — violates the no-regression constraint.
(Oddity: unroll=2 compiles faster than unroll=1, 0.83 vs 1.03 s.)

## 4. Where this leaves the goal

Centered, cold, warm node, with 3a applied: total XLA compile ≈ 0.3 s
(residual construction) + 1.45 s (chunk) ≈ **1.75 s < 2 s — goal met**;
with 3b every rerun of the same configuration is well under it. The
user-visible TTFS becomes import/backend-bound (~1.7 s fixed) plus
data-bound `set_fields`. Remaining follow-up levers, in value order:

1. **HLO-volume reduction in the step body** — the only lever for
   weno5's 7.8 s chunk compile (and it buys every scheme): the 12×
   Python-unrolled flux kernels, ~70 frame pads, ~60 seal DUS per body.
   Any reduction must clear the step suite on one device (the
   2026-07-14 lesson: standalone-kernel wins can regress the in-model
   step).
2. **Comparison-suite metric fix** (§1) — cheap, corrects the public
   story: fridom centered is at Oceananigans-parity on honest compile,
   not 5–8× worse.
3. lazypimp resolve (0.55 s) and jax import — minor, fixed costs.

Wave-1/2 harnesses (phase breakdown, census, AOT split, scheme sizing,
async prototype) lived in the session scratchpad (volatile); the
methodology is reproducible from this note + the two patches.
