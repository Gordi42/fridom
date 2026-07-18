# Step-benchmark performance guard (`benchmarks/ci/`)

`step_guard.sbatch` is the on-device timing guard: one SLURM batch job
that runs the model step suite (`benchmarks/model`) on 1 GPU and on
4 GPUs of a DKRZ Levante A100-80GB node, compares each leg against the
committed baseline in `benchmarks/baselines/`, and writes a GREEN/RED
verdict. It is the timing half of the performance guard designed in
[`design/plans/active/perf_guard_plan.md`](../../design/plans/active/perf_guard_plan.md)
(G3, §4.3); the deterministic fast-path assertions (G1) are the CI
teeth and live in `tests/`.

## How to run it

```bash
sbatch benchmarks/ci/step_guard.sbatch    # submit from the repo root
```

Submit from the repository root: the script resolves the repo from
`$SLURM_SUBMIT_DIR`. Watch the job's output
(`benchmarks/results/step-guard-<jobid>.out`); there is no alerting —
the terminal output and the marker file are the record (owner ruling
§5.2).

**Run from a quiescent checkout.** The job imports the repo state at
each case's subprocess launch, so parallel sessions merging onto
`dev` (or dirtying the tree — including an in-flight baseline
re-record) mid-run corrupt the comparison. The result JSON records
`commit` and `dirty`; a `dirty=True` run is not evidence. Lesson from
the first run, 2026-07-18: two guard runs interleaved with a parallel
session's baseline re-record and read a shifted tree.

The submitting checkout's `.venv` must carry the CUDA jax plugin:
`uv sync --extra dev --extra cuda`. This bites **worktrees** in
particular — a fresh worktree synced with `--extra dev` alone gets
CPU-only jax, and both legs then die in seconds with `Backend 'cuda'
is not in the list of known backends` and a spurious RED marker
(observed 2026-07-18, job 26346286).

Local sanity check without touching SLURM or GPUs:

```bash
DRY_RUN=1 bash benchmarks/ci/step_guard.sbatch   # echoes each command
```

## When to run it (owner-batched checkpoints — never per-merge)

Guard runs are **not** a merge requirement. Silvano batches them: he
decides when a checkpoint is due (for example after ~10 merges), runs
the guard against the **last-guarded baselines**, and on RED studies
the accumulated batch — bisecting within it if needed — before either
fixing the regression or accepting the movement and re-recording the
baselines (which makes the new state the reference for the next
checkpoint).

**Agents: never submit this script — or any GPU job — on your own
initiative.** A merge being "perf-sensitive" is not authorization;
codifying a per-merge guard requirement was tried on 2026-07-18 and
retracted the same day after agents began submitting guard runs
unprompted (owner ruling: GPU submissions happen only when Silvano
explicitly asks in chat; see `design/plans/active/perf_guard_plan.md`
§5).

## Manual submission ONLY

This script must be **submitted by hand, every time.** Do **not** wire
it into scrontab, cron, GitHub Actions, or any auto-submitter (owner
ruling, `perf_guard_plan.md` §5.1). Reason: unattended GPU submissions
silently consume the project's DKRZ cluster usage limits. Every run is
a deliberate human (or agent) action against a specific commit.

## Where results land (and that they accumulate)

Everything is written into `benchmarks/results/` (gitignored — nothing
here reaches the public repo), each artifact stamped with a shared
run timestamp:

- `step-gpu{1,4}-<ts>.json` — the fresh measurements;
- `report-gpu{1,4}-<ts>.md` — the `compare` markdown tables;
- `step-guard-<ts>.{GREEN,RED}` — the verdict marker (contents point at
  the run's JSON and reports).

The script **never deletes or overwrites** prior output (retention
ruling §5.4): the timestamped files accumulate into the private result
series that later change-point detection (G4) can consume. Prune the
directory by hand if it grows too large.

## SLURM header values

Confirmed from `sinfo` on Levante: partition `gpu` carries
`gpu:a100_80:4` (the 80GB A100 nodes — `a100_40` also lives in `gpu`,
so the script pins the 80GB GRES to match the baselines' hardware,
`NVIDIA A100-SXM4-80GB`, node `l50109.lvt.dkrz.de`). The job asks for
1 node, 1 task, all 4 GPUs, `--exclusive` (clean timing), 4 h wall.

**`--account` is TODO-marked in the script.** `sacctmgr show assoc`
lists `uo0780` / `uo0780_gpu` (and `ka1125` / `ka1125_gpu`); the work
dir is `/work/uo0780/...`, so the script defaults to `uo0780_gpu` as
the best guess for the GPU budget. Confirm and adjust if the job is
rejected on the account. `LEG_TIMEOUT` (per-leg `timeout`, default
7200 s), `RESULTS_DIR`, and `DRY_RUN` are overridable environment
variables at the top of the script.

## Baseline lifecycle

The committed baselines (`benchmarks/baselines/step-gpu{1,4}.json`) are
re-recorded **only** when a change *intentionally* moves the numbers,
with the same commands, on the same node, and a commit message stating
the before/after and device. The full re-record workflow is in
[`benchmarks/README.md`](../README.md) ("The performance guard"). The
guard legs mirror that record invocation exactly (plain `run
benchmarks/model -o <out>` with default reps/warmup — no CI smoke
flags) so the comparison is apples-to-apples.

## On RED

A RED marker means a leg failed to run or a case regressed past the
threshold. Do **not** re-baseline to make it green. Follow the perf
methodology: re-measure the flagged case(s) to rule out A100 thermal
jitter (small-`n` and `sw_sphere` cases carry 2–6% noise — a single
reading can collapse on re-measure), then attribute the change to the
responsible commit before deciding whether it is a real regression or
an intended, to-be-re-baselined move.

**Node-to-node variance is a confirmed false-RED cause.** Observed
2026-07-18: `sw_flat[n=1024]` read 14.5 ms (l50051) and 15.4 ms
(l50163) on identical code, tight reps each — a +7% wholesale shift
past the 5% global tolerance from the node alone. When a RED does not
attribute to a commit, re-run pinned to the node the baseline was
recorded on (`sbatch -w <node> benchmarks/ci/step_guard.sbatch`); the
baseline JSON records its node in `metadata.hostname`, matching the
"same node" rule of the baseline lifecycle below. See
[`design/plans/active/perf_guard_plan.md`](../../design/plans/active/perf_guard_plan.md)
(§1.3, §4.3) and `benchmarks/README.md` "Measurement traps".

> Note: a `compare` hardening (environment guard, min estimator,
> per-case tolerance) is landing in parallel on branch
> `feat/bench-compare-hardening`. This guard uses only plain
> `--fail-on-regression`, which behaves identically before and after
> that change.
