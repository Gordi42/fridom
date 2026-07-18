# Benchmarks

Timing suites for the `fridom.benchmarking` harness, plus manual
accuracy benchmarks and recorded results.

## Layout

| path | what |
|---|---|
| `bench_*.py` (top level) | **old-stack** cases (`fridom.framework` / `nonhydro` / `shallowwater`); kept for old-vs-new comparisons until the cutover |
| `spatial/` | new-stack kernel and operator-application cases |
| `model/bench_step.py` | **the performance guard**: `model.advance` on the production configurations (flat / walled / terrain-following nonhydro, flat / sphere shallow water) |
| `model/bench_balance.py` | manual accuracy benchmark (not a timing case; run directly) |
| `baselines/` | checked-in reference results per device configuration |
| `results/`, `*/results/` | local run outputs (gitignored) |
| `RESULTS.md`, `nonhydro_shallowwater_new_vs_old.md` | recorded measurement campaigns |

## Running a suite

```bash
uv run python -m fridom.benchmarking list benchmarks/model
uv run python -m fridom.benchmarking run benchmarks/model --filter nh_flat
```

Each case instance runs in a fresh subprocess (independent jit caches
and peak-memory counters); results are written as JSON.

## The performance guard (`model/bench_step.py`)

The step suite exists so that "did this change slow the model down?"
has a mechanical answer:

```bash
# 1 GPU
CUDA_VISIBLE_DEVICES=0 uv run python -m fridom.benchmarking \
    run benchmarks/model -o /tmp/step-gpu1.json
uv run python -m fridom.benchmarking compare \
    benchmarks/baselines/step-gpu1.json /tmp/step-gpu1.json \
    --fail-on-regression

# 4 GPUs — the XLA flag is REQUIRED (see traps below)
XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion \
    uv run python -m fridom.benchmarking \
    run benchmarks/model -o /tmp/step-gpu4.json
uv run python -m fridom.benchmarking compare \
    benchmarks/baselines/step-gpu4.json /tmp/step-gpu4.json \
    --fail-on-regression
```

Baselines are re-recorded with the same commands (`-o
benchmarks/baselines/step-gpu<N>.json`) when a change *intentionally*
moves the numbers; the commit message states the before/after and the
device (baselines are machine-specific — the checked-in ones are from
a DKRZ 4x A100-80GB node). Repeated 50-step chunks; the per-step
number is `wall / extras["steps"]`.

### How `compare` decides

`compare` hardens the raw threshold check three ways so an on-device
comparison is honest:

1. **Environment guard.** Before comparing any case it checks that the
   two files agree on `backend`, `device_count`, `device_kind`, and
   `jax_version` (a missing field counts as a mismatch). A mismatch is
   a hard error with a nonzero exit and *no* comparison output — a cpu
   run against a gpu baseline otherwise prints meaningless deltas
   silently. `--allow-env-mismatch` downgrades this to a prominent
   warning and proceeds.
2. **Min estimator.** The compared statistic is the **minimum** of each
   case's `wall_times`, not the median: environmental noise is
   one-sided (it only ever adds time), so the minimum is the
   least-contaminated estimate of the true cost (Chen & Revels, HPEC
   2016). Per-run tables still show the median. Committed baselines
   need no re-record — the full `wall_times` lists are stored.
3. **Per-case tolerance.** The effective band is `max(--threshold,
   3 * CoV_base)`, where `CoV_base = std/mean` of the baseline's own
   samples. A stable large case is held to the flat `--threshold`; a
   jittery small case (measured A100 jitter runs 2-6% on the small
   cases) is given its own noise-derived band, so the flat 5% is
   neither too tight nor too loose. The report shows the effective
   tolerance and which rule set it (`global` vs `noise`) per case.

CI smoke-runs every suite (`--first-only --reps 1`, smallest sizes,
cpu) so the case files cannot go stale. **No timing assertions run in
CI** — regression checks against the baselines are a manual,
on-device step.

## Measurement traps

Learned the hard way; every one of these silently corrupts a
measurement:

1. **Chunk size.** `model.advance(n)` with `n < chunk_size` executes
   n chunks of length 1 — the scan unroll never engages and per-step
   time reads 2-3x worse. Always advance exactly `chunk_size` steps
   per timed repetition (`bench_step.py` passes `chunk_size=STEPS`).
2. **Rotation is opt-in** (since the 2026-07 geometry merge).
   `coriolis=None` installs no Coriolis module at all — a
   before/after comparison across that default change reports a
   spurious speedup. Pass an explicit `coriolis=`.
3. **4-GPU runs need**
   `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion` until
   jax-ml/jax#39100 is fixed: without it XLA:GPU miscompiles the
   4-device step (silently wrong physics — it *times* fine). The
   flag costs nothing measurable.
4. **Old-stack cases are not device-count invariant.** On a
   multi-GPU node, pin them to one device
   (`CUDA_VISIBLE_DEVICES=0`); the tiny grids are not divisible by
   4 devices and error at setup.
5. **jax dispatches asynchronously.** Every timed call must block on
   its output (`jax.block_until_ready`); the harness does this, ad
   hoc scripts must too.
