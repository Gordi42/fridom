# Oceananigans.jl reference benchmark (RTX 3060 Laptop, float64)

Matched-protocol re-establishment of the July 2026 A100 comparison, rebuilt on
this machine from the documented matched config. Mirror of fridom's
`design/research/stencil_lowering/microbench/phase3/p3b_model.py` build_model +
timing harness. Fresh Julia process per (scheme, n). No changes made to the
fridom repo; all work under this scratchpad.

## (a) Environment

| Component | Version |
|---|---|
| Machine | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM, 55 W cap, sm_86 |
| Driver | 595.71 (NVML 13.0.0+595.71.5) |
| Julia | 1.12.6 (non-official / distro build — see caveats) |
| Oceananigans.jl | 0.105.3 (pinned; resolved cleanly, no drift) |
| CUDA.jl | 5.11.3 |
| CUDA artifact toolkit | runtime 13.2, compiler 13.3 (artifact installation) |
| CUDA libs | CUBLAS 13.4.0, CUFFT 12.2.0, CUSOLVER 12.2.0, CUSPARSE 12.7.10, CURAND 10.4.2 |
| GPUCompiler / GPUArrays / KernelAbstractions | 1.17.1 / 11.5.8 / 0.9.42 |
| LLVM | 18.1.7 |
| Precision | float64 (Oceananigans default); consumer Ampere f64 ~1/64 rate |
| GPU memory available at start | 5.656 GiB / 6.000 GiB |

`CUDA.functional() == true`.

## Matched config (as built)

- Grid: `RectilinearGrid(GPU(); size=(n,n,n), x=(0,10000), y=(0,10000),
  z=(0,100), topology=(Periodic,Periodic,Periodic))` — triply periodic (default
  z would be Bounded; explicitly overridden).
- Model: `NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2,
  advection=SCHEME, tracers=:b, buoyancy=BuoyancyTracer(),
  coriolis=FPlane(f=1e-4))`. Default FFT pressure solver. No background fields,
  no closure, no output writers, no Simulation wrapper. (`grid` is a positional
  argument in 0.105.3, not a keyword — see caveats.)
- ICs (match p3b): kx=ky=2π/10000, kz=2π/100; u=0.2·sin(kx·x)·cos(ky·y),
  v=0.06·cos(kx·x), b=1e-4·cos(kz·z), w=0. Applied via `set!` with functions.
- Schemes: `Centered(order=2)`, `UpwindBiased(order=5)`, `WENO(order=5)`
  (applies to momentum and tracers).
- Timing: dt=20.0; 50 steps/chunk bounded by `CUDA.synchronize()`; 1 warmup
  chunk discarded (absorbs compile), then 6 timed chunks of 50 steps. Median
  and min ms/step over the 6 chunks. Finiteness checked on u at end — all runs
  finite, so dt stayed at 20.0 everywhere (no NaN, no dt reduction).

## (b) Results: median / min ms/step

| scheme | n=96 | n=128 | n=160 | n=192 |
|---|---|---|---|---|
| Centered(2)      | 6.63 / 6.46   | 15.36 / 15.24 | 31.84 / 31.67 | 53.93 / 53.69 |
| UpwindBiased(5)  | 9.83 / 9.62   | 23.00 / 22.92 | 47.33 / 47.06 | 82.29 / 81.17 |
| WENO(5)          | 31.99 / 31.70 | 73.27 / 71.62 | 145.43 / 143.57 | — (skipped) |

- All runs `finite=true` at dt=20.0. First-chunk compile times were 2.7–8.9 s
  (Oceananigans/CUDA precompilation was already warmed during setup).
- n=192 was run only for Centered and UpwindBiased per spec; both fit in 6 GB
  (peak well under the 5.656 GiB budget — no OOM). WENO(5) n=192 was not
  attempted (spec: n=192 only for the two cheaper schemes). n=256 not attempted.
- Cost ordering as expected: WENO(5) ≈ 3–5× Centered(2); UpwindBiased(5) ≈
  1.5× Centered(2). Scaling with n is roughly ~n³ (e.g. Centered 128→160 is
  (160/128)³≈1.95× vs measured 2.07×).

## (c) Thermal drift

No meaningful drift / no throttling observed. Snapshots
(`clocks.sm, temp, power`) taken immediately before and after each run; because
they are taken after `CUDA.synchronize()` the GPU has already returned to its
210 MHz idle clock in every snapshot but one (upwind5 n128 AFTER caught 667 MHz
in the split second before idle — a sampling artifact, not sustained state).
GPU temperature rose modestly over the session: BEFORE snapshots ~49–53 °C,
AFTER snapshots 53–63 °C (highest after the back-to-back WENO n128/n160 runs,
63 °C). Power AFTER 17–21 W — nowhere near the 55 W cap, consistent with the
f64-bound (not power-bound) workload on this laptop part.

Direct drift quantifier — UpwindBiased(5) n=128 re-run at the end of the
session vs. its original mid-session run:

| run | median ms/step | min ms/step | temp before |
|---|---|---|---|
| original (mid-session) | 22.996 | 22.924 | 52 °C |
| repeat (end-of-session) | 22.836 | 22.768 | 50 °C |

The repeat is 0.7% *faster*, not slower — timings are stable and there is no
thermal degradation across the ~30-run session; the small delta is within
run-to-run noise (and if anything tracks the slightly cooler start).

## (d) Caveats

- **Stratification term (config delta vs fridom):** fridom's matched config
  additionally carries a `ConstantStratification` linear tendency
  (N²=(50·1e-4)²=2.5e-5). Oceananigans here has no equivalent term — one cheap
  linear term, negligible for step-time comparison but not identical. Not fixed
  (per instructions). This makes the Oceananigans numbers a marginal *under*count
  of what an exactly-matched config would cost, but the effect is below the
  run-to-run noise floor.
- **Timestepper:** QuasiAdamsBashforth2 = 1 RHS evaluation/step, matched to
  fridom's AB3 (also 1 RHS/step). QAB2's very first step is an Euler step;
  absorbed by the discarded warmup chunk.
- **Version drift from target:** none. Oceananigans 0.105.3 (the July target)
  resolved cleanly on Julia 1.12.6; no fallback to a nearest 0.10x was needed.
- **dt changes:** none. Every run was finite at dt=20.0; the dt=5.0 fallback
  path was never triggered.
- **`grid` positional in 0.105.3:** `NonhydrostaticModel` takes `grid` as a
  positional argument, not the keyword shown in the task brief
  (`NonhydrostaticModel(grid; ...)`). Adapted; caught during CPU validation.
- **Non-official Julia build:** CUDA.jl emits "You are using a non-official
  build of Julia. This may cause issues with CUDA.jl." (`/usr/bin/julia`, distro
  build). `CUDA.functional()` is true and all runs completed cleanly; flagged
  for the record only.
- **Shared-GPU coordination:** another agent was benchmarking fridom on the same
  single GPU during this session. Every timed run was gated on the GPU being
  free (compute-apps empty for ≥2 consecutive minutes, polled every 30 s); no
  timed run overlapped foreign GPU compute.

## Files

- Julia project: `ocean/` (Project.toml + Manifest.toml, pinned deps)
- Benchmark script: `bench.jl` (one fresh process per scheme,n)
- CPU validation: `cpu_validate.jl`
- Driver: `run_all.sh` (GPU-free gating + thermal snapshots)
- Per-run results: `results/ocean_<scheme>_n<N>.json` (+ `_drift`)
- Thermal snapshots: `results/thermal_log.txt`
- Per-run logs: `results/run_<scheme>_n<N>.log`, `results/driver.log`
