---
title: Performance guard — closing the CI-gate roadmap item
status: active (G1–G3 code on dev 2026-07-18; open: one green manual A100 guard run + gpu-marked legs)
created: 2026-07-18
owner: Silvano
---

# Performance guard — closing the CI-gate roadmap item

Closure design for the roadmap item *"Performance guard — wire the
benchmark harness as a CI gate"* (`design/roadmap/open.md`). The item
has two halves: (1) the A/B step harness exists but CI only smoke-runs
it, so a timing regression stays green; (2) only the distributed-solve
fast paths are asserted — production can silently fall off any other
fast path and pass the suite.

The central finding: **the item as literally titled should not be
built.** GitHub CI cannot honestly gate on the committed baselines
(§1.2, §2), and no comparable project PR-gates on wall-clock (§1.4).
The honest closure is three-legged: deterministic fast-path assertions
as the *real* CI gate (§4.1), a hardened `compare` (§4.2), and the
timing guard wired where it can honestly run — the DKRZ A100 node, as
a **manually triggered** pre-merge protocol (owner ruling §5.1: no
automated cluster submissions), never a PR-blocking check (§4.3).

## 1. What the investigation established

### 1.1 The gate is half-wired

`python -m fridom.benchmarking compare --fail-on-regression` is
implemented, documented, and was self-checked green at the T7
re-record — but nothing invokes it. The CI `benchmark-smoke` job
(`.github/workflows/tests.yml:98-123`) runs `--first-only --reps 1
--warmup 1` with no `compare`; its only failure mode is a case that
*raises*. The header comment says it outright: "No timing assertions".

### 1.2 CI cannot reproduce the baselines' environment

- Both committed baselines (`benchmarks/baselines/step-gpu{1,4}.json`)
  are machine-specific: DKRZ 4×A100-80GB node, jax 0.10.2, with the
  mandatory `--xla_disable_hlo_passes=multi_output_fusion` workaround
  on the 4-GPU leg. All CI runners are GitHub-hosted `ubuntu-22.04`,
  CPU-only; no cron, no `workflow_dispatch`, no self-hosted runner.
- On CPU the `ON_GPU` size gating drops every large case, leaving only
  the tiny `n=31/32/64` instances — the noisiest ones.
- `compare` matches cases by **name only** and never checks the
  metadata block (backend, device_count, device_kind, jax_version): a
  CPU run compared against a GPU baseline produces meaningless deltas
  with no warning.

### 1.3 The shipped harness is weaker than the campaign discipline

- The trustworthy campaign numbers came from **bracketed
  interleaving** (baseline before AND after every candidate, delta
  against the bracket mean — cancels thermal drift). That lives in
  `design/research/upwind5_revisit/flags/interleave.py`, not in
  `src/fridom/benchmarking/measure.py`, which times one isolated
  subprocess per case.
- The gate is a **single global 5% threshold** on the median
  (`compare.py:12,81-121`). Measured A100 jitter: <0.3 ms std on the
  mixed-transform cases, but 2–6% on small-n and `sw_sphere` cases (a
  +6.5% reading collapsed to +1.9% on re-measure). A flat 5% both
  false-positives on the jittery cases and is blind to sub-5%
  regressions on the stable large ones.

### 1.4 External evidence (2024–2026 survey)

- Shared-runner noise: ~2.66% CoV on GitHub-hosted runners (CodSpeed,
  independently corroborated by Quansight); a 2% wall-clock gate
  false-alarms ~45% of runs, and ~7% is the smallest threshold with
  ~1% false positives — above the regressions our baselines track.
  Runner microarchitecture varies *between jobs on the same pinned
  image*.
- **No surveyed project PR-gates on wall-clock**: JAX/XLA (correctness
  CI + reactive bisection), PyTorch (on-demand TorchBench + nightly
  A100 fleet feeding HUD dashboards and auto-filed issues), Julia
  (Nanosoldier on-demand bot on a dedicated cluster), NumPy/SciPy/
  pandas (asv scheduled runs + dashboards + post-merge culprit
  comments), and Oceananigans.jl (the closest analog: `benchmarking/`
  dir → scheduled runs → GitHub-Pages dashboard).
- **GitHub explicitly warns against self-hosted runners on public
  repos** (twice, in the runner and security docs): fork PRs can
  execute arbitrary code on the runner host. For us the host would be
  a DKRZ login/compute node.
- Statistics: MongoDB (ICPE 2020) abandoned fixed-percent pairwise
  gates for change-point detection over the result series; Chen &
  Revels (HPEC 2016) establish the **minimum** as the best estimator
  under one-sided environmental noise; Kalibera & Jones argue
  effect-size confidence intervals over significance tests; the HPC
  standard (ReFrame/CSCS) is per-system reference values with
  **per-case tolerance bands**, submitted as SLURM jobs.

### 1.5 Fast-path guard coverage (the second half of the item)

The repo already owns six deterministic non-timing assertion
techniques, all in `tests/`, ranked by backend-stability:

1. **Compile-count / cache-size counters** (`tests/conftest.py`
   `compile_counter`) — fully backend-stable.
2. **Seam spies** (monkeypatch the resolver, assert on the returned
   plan object / call count) — the `test_distributed_projection.py`
   and `test_exchange_counts.py` pattern; backend-stable.
3. **Plan/property asserts** (host-side, no compile) — e.g.
   `distributed_forward_plan(...) is not None`; backend-stable.
4. **Collective-family opcode sum, differential** — the
   `test_reblock_step_collectives.py` template: sum
   all-to-all/collective-permute/all-gather/all-reduce/reduce-scatter
   over the compiled step and compare two configs on the *same*
   backend; backend-agnostic by construction.
5. **HLO opcode presence/absence and op-counts on isolated
   operators** — absence checks robust; presence checks must match by
   substring (GPU async `-start/-done` spellings). jaxpr op-counts
   (pre-lowering) are fully CPU-stable.
6. **Byte-for-byte HLO goldens** — CPU-forced-4-only by design; do
   not generalize.

Coverage today: the distributed-solve geometries (seam spy + HLO),
the reblock collective budget (whole-step collective sum), halo
exchange lowering, the map+carry seal, storage-frame claims + sync
elision, and the eager multi-device kernel jit are guarded.
**Unasserted or under-asserted** (ranked by silent-failure risk):

| gap | path | why it matters | proxy (technique #) |
|---|---|---|---|
| A | multigrid **vertical-line** smoother (`mapped_pressure.py`, `immersed_pressure.py`) | a point-smoother swap looks *faster* per V-cycle but stops converging on steep mapped/immersed — the cases multigrid exists for | `isinstance(level.smoother, VerticalLineJacobi)` per level (3) + anisotropic iters-to-tol (line ≈15, point stalls) |
| B | tridiagonal kernel wiring (`banded.py` auto → cusparse/pcr vs scan-Thomas) | 9–18× on the CG solve; unit dispatch tested, but nothing asserts the *built pressure solver* carries it end to end | seam/property assert on the built smoother (2,3); HLO `while`-absence for pcr (5, CPU-testable); gpu-marked cusparse leg |
| C | WENO selected-input (`advection.py` `_face_value` override) | +66%@256³/+87%@512³ if silently reverted to both-then-select; bitwise parity tests cannot see it | jaxpr `div`-count == ½ of both-then-select on the isolated operator (5, CPU-stable) |
| D | chunk-body donation + `out_shardings` pin (`model/model.py`) | lost donation = 2× carry peak → OOM at 1024×512²; lost pin = silent recompiles | `input.is_deleted()` after call + `compile_counter == 0` on second advance (1); do **not** pin `memory_analysis()` bytes on CPU |
| E | FV `flux_diff @ reconstruct` fusion on **uniform** rows | FV=nodal parity (0.997–1.003×) rests on full fusion; a lowering change could quietly unfuse the default path | differential compiled-op-count FV vs nodal sibling on the *periodic* case, same backend (4,5) |
| F | scalar-`dx` fold on uniform meshes (`flux_diff.py` `_windowed_diff`) | uniform meshes must fold spacing to a static scalar, not divide by a measure field | jaxpr: no `div` by a materialized measure array on a uniform mesh (5, CPU-stable) |

The walled/mapped FV op-count gap is **ratcheted, not equalized**
(owner ruling §5.3): record today's FV-vs-nodal op-count relationship
on the walled/mapped rows and gate against *worsening*, with a
tolerance band. Caveat, encoded in the test itself: this pins a
compiler artifact, so a jax/jaxlib upgrade may move the counts for
reasons unrelated to fridom — the failure message must direct the
reader to re-measure and re-baseline the ratchet (single documented
regen knob) before blaming the triggering change. The separate
"FV-vs-nodal step-time gap" roadmap item still aims to eliminate the
gap; when it does, the ratchet tightens to parity.

## 2. What would be bad practice (rejected options)

1. **Wall-clock threshold in GitHub CI** — noise floor (≥~7% usable
   threshold) sits above the regressions the baselines track; the gate
   is either flaky or blind. Rejected on the evidence in §1.4.
2. **CPU baselines for GitHub runners** — heterogeneous runner
   hardware between jobs + only the noisiest tiny cases survive the
   `ON_GPU` gating. Measures nothing we care about.
3. **Self-hosted runner on DKRZ for a public repo** — fork PRs =
   arbitrary code execution on the cluster; GitHub's own docs forbid
   the pattern. Results must flow cluster→GitHub, never CI→cluster.
4. **Generalizing the byte-for-byte HLO golden** as the guard —
   backend-locked, fragile across jax upgrades; keep it scoped to the
   forced-4 CPU leg it was built for.
5. **Gating `memory_analysis()` aliasing bytes on CPU CI** —
   backend-dependent; donation is guarded behaviorally instead (gap D).
6. **A uniform fixed-percent threshold as the long-term statistic** —
   false-positives on 2–6%-jitter cases, blind on <0.3 ms-std cases.
   Tolerances must be derived per case from the baseline's own spread.

## 3. Design position

The PR gate and the timing guard are **different instruments**:

- The regressions that cost 2× at scale (falling off a fast path) are
  *discrete and deterministic* — a resolver returns `None`, a kernel
  unfuses, a collective family changes. These are exactly what CPU CI
  *can* gate, byte-noise-free, via the six techniques in §1.5. This is
  the real CI gate, and it is the higher-value half of the item.
- Continuous timing drift (fusion quality, kernel latency) is only
  measurable on the A100 node. It gets a *manual protocol*, not a PR
  check (per universal field practice, §1.4) and not a schedule
  (owner ruling, §5.1).

## 4. The proposal

### 4.1 G1 — fast-path assertion sweep (test-only; the real CI teeth)

Close gaps A–F of §1.5 with the mapped techniques. Placement follows
the mirrored-test rule: A,B in
`tests/spatial/operators/test_mapped_pressure*.py` /
`test_immersed_pressure*.py` / `test_banded.py`; C in
`tests/model/modules/test_advection_selected.py`; D in
`tests/model/test_step_chunk.py` / `test_model_memory.py`; E,F in
`tests/spatial/operators/test_flux_diff*.py` (E's differential
op-count may need the forced-4 leg if the single-device compiled text
is uninformative — decide at implementation). E carries both the
uniform-parity assert and the walled/mapped ratchet (§1.5, §5.3);
the ratchet baselines regenerate behind one env knob, following the
`FRIDOM_REGEN_HLO_GOLDEN` precedent. All CPU-stable except B's
cusparse leg (gpu-marked, runs on the A100 suite) and possibly E.
Estimated ~6 focused test additions; no `src/` changes.

### 4.2 G2 — harden `compare` (small `src/fridom/benchmarking` change)

1. **Environment guard**: `compare` fails (override:
   `--allow-env-mismatch`) unless backend, device_count, device_kind,
   and jax_version match between baseline and candidate metadata.
2. **Min estimator**: compare on `min(wall_times)` (one-sided-noise
   argument, §1.4). Stored baselines already carry the full
   `wall_times` lists — no re-record needed.
3. **Per-case tolerance**: `tol_case = max(--threshold,
   k·CoV_base(case))` with k≈3, CoV from the baseline's own samples;
   report which rule fired per case. Keeps the CLI contract, fixes the
   flat-5% weak spot.

Mirrored tests in `tests/benchmarking/`. Explicitly *not* building:
interleaved A/B inside the harness (commit-vs-commit interleave needs
two checkouts; the bracketed-interleave research script remains the
campaign tool) and change-point detection (needs a result series
first — see G4).

### 4.3 G3 — wire the timing guard where it can honestly run (DKRZ)

**Manual-trigger only** (owner ruling §5.1): no scrontab, no cron, no
CI-triggered submission, nothing that submits GPU jobs without a
deliberate human/agent action — automated submissions can silently
consume the owner's cluster usage limits.

1. **`benchmarks/ci/step_guard.sbatch`** (+ short README): one SLURM
   job on the A100 partition that runs the gpu1 and gpu4 legs with the
   required XLA flags, then `compare --fail-on-regression` against the
   committed baselines, writing JSON + markdown next to a red/green
   marker. Guarded by `timeout`. Invoked **by hand** as the
   **pre-merge protocol**: binding for perf-sensitive merges
   (anything touching step-path lowering: `spatial/operators/`,
   `spatial/decomposition/`, `model/time_steppers/`, tendency
   modules, `model/model.py`) — one AGENTS.md line under the merge
   gate, codifying what the campaigns already do by hand.
2. **No alerting infrastructure**: the operator watches the run; the
   red/green marker + report in the results dir are the record.
3. **Result retention** (owner ruling §5.4): every guard run appends
   its result JSON to an untracked results dir on DKRZ (the script
   simply never deletes output). Nearly free, keeps machine-specific
   numbers off the public repo, and accumulates the series that
   change-point detection needs later (§4.4).
4. **Baseline lifecycle** stays as documented (re-record only on
   intentional movement, same node, commit message with before/after);
   the G2 environment guard turns "jax upgraded / wrong node" from a
   silent nonsense-compare into a hard error.

### 4.4 G4 — deferred

Change-point detection over the retained series (once it is long
enough to be informative) and any public dashboard. Not part of
closing the roadmap item; a dashboard would publish machine-specific
numbers and is postponed indefinitely.

## 5. Owner rulings (2026-07-18)

1. **No automated GPU submissions to Levante — ever.** No scrontab,
   no scheduled jobs, no auto-submission scripts: they can silently
   eat cluster usage limits. The timing guard is manual-trigger only,
   as the pre-merge protocol in §4.3. (This supersedes the proposal's
   "protocol + weekly scrontab" recommendation.)
2. **No alerting infrastructure** — follows from ruling 1: with no
   unattended runs there is nothing to alert on; the run's terminal
   output and the results-dir report suffice.
3. **Gap E: ratchet the walled/mapped gap** in addition to the
   uniform-parity guard (owner chose broader coverage over the
   spurious-red risk; the compiler-artifact caveat and the regen knob
   are encoded in the test, §1.5/§4.1).
4. **Retain results privately**: guard runs keep their JSON in an
   untracked results dir on DKRZ; no public series, no dashboard.

## 6. Closure criteria for the roadmap item

The item moves to `done.md` when:

- G1 assertions are on `dev` (all six gaps, including the E ratchet,
  green on CPU default + forced-4 leg; gpu-marked legs green on the
  A100 suite);
- G2 `compare` hardening is on `dev` with mirrored tests;
- G3 script + AGENTS.md merge-gate line are on `dev` and one full
  `step_guard.sbatch` run has been executed green on the A100 node
  (manually submitted, per §5.1);
- the roadmap entry is rewritten to record the §2/§3/§5 rulings (PR
  CI gates structure, DKRZ gates time, manual-trigger only) so the
  "wire it into GitHub CI" framing does not resurface.
