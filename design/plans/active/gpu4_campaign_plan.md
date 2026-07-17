---
status: active
date: 2026-07-17
---

# 4-GPU campaign — work list + agent prompt

Everything on [`../../roadmap/open.md`](../../roadmap/open.md) that is
gated on **real multi-GPU hardware** (a DKRZ node with 4× A100), bundled
so one session on such a node clears the lot. The section below the
rule is the prompt to hand to that session verbatim. Written against
dev `b77f8582` (2026-07-17); the prompt tells the agent to re-check the
roadmap first in case items shipped in between.

---

You are working in the FRIDOM repo (`AGENTS.md` is binding — read it
first) on a DKRZ GPU node with 4× NVIDIA A100-SXM4-80GB. Your job is to
clear every open-roadmap item that is gated on real multi-GPU hardware,
in this one session. Before starting, re-read
`design/roadmap/open.md` and skip any task below that has already moved
to `done.md`.

Work the tasks in the order given — T7 (baseline re-record) must run
**last**, on the final merged dev state, so the committed numbers match
what actually landed. If a task blocks (cannot reproduce, hardware
issue), write the negative result into the relevant research record and
move on; do not stall the campaign.

## Ground rules (read before the first GPU run)

1. **Verify hardware first**: `nvidia-smi` — expect 4× A100-SXM4-80GB.
2. **Two parallelisms — never conflate them** (AGENTS.md):
   *single-process 4-device* GSPMD (`JAX_PLATFORMS=cuda`, one python,
   all 4 GPUs visible — what benchmarks and `multi_device` tests use)
   vs *real multi-process* (`srun -n 4 --gpu-bind=none`, script calls
   `jax.distributed.initialize()` **before** importing fridom). Use
   `--gpu-bind=none`, never `--gpus-per-task=1`. Guard every `srun`
   under `timeout` — a dead rank blocks the others at the coordination
   barrier. In multi-process code, never `np.asarray` a global array;
   gather with
   `jax.experimental.multihost_utils.process_allgather(arr, tiled=True)`.
3. **Known upstream miscompile**: jax 0.10.x `multi_output_fusion`
   silently corrupts 4-GPU new-stack physics (reported as jax#39100).
   For every correctness-validation run set
   `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion` (measured
   ~free; do NOT use priority-fusion off — 25%–3.6× slower). Check
   first whether the harness/tests already set it
   (`grep -rn multi_output_fusion src tests benchmarks`). Before
   attributing any 4-GPU physics mismatch to fridom, re-run with the
   workaround; without it you may be staring at the jax bug.
4. **Backend gotcha**: bitwise/HLO multi-device tests are
   backend-specific — a red *forced-CPU* multi-device suite is not a
   regression; judge each test on the backend it was written for
   (`tests/` use a backend-aware `invariant` helper).
5. **Memory-measurement traps**
   (`design/research/gpu_memory_ceiling.md`): never measure footprint
   through copy-on-read host fetches of the carry — sync the live
   `_carry` instead; `cuda_async` breaks at
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` (0.75 is the safe setting);
   donation+defrag is the shipped configuration.
6. **Git**: anything touching `src/`, `tests/`, or `benchmarks/` goes
   on a `<type>/<kebab-topic>` branch, merged `--no-ff` into dev, the
   branch deleted in the same session. Design-record-only updates may
   go direct to dev. Merge gate per AGENTS.md: mirrored tests for
   edited files + `uv run ruff check src tests`. Do **not** run the
   full test suite.
7. **Roadmap hygiene (binding)**: the moment a task ships, move its
   record to `design/roadmap/done.md` in the same change and trim the
   `open.md` entry to what remains.

## Tasks

### T1 — FV default: validate on real 4 GPUs

Roadmap "Finite-volume nonhydro" (record:
`design/plans/active/fv_nonhydro_scoping.md`, validation note near
§7/line 468). The distributed solve on **average origins** has run only
1-GPU; forced-4 CPU asserts the walled + mapped FV fast paths, but real
multi-GPU has not.

- Run the `multi_device` tests of
  `tests/nonhydro2/test_distributed_projection.py` on real 4 GPUs
  (`JAX_PLATFORMS=cuda`, single process seeing all 4 devices).
- Physics smoke: an FV-default nonhydro case (periodic and walled-x)
  stepped N steps, 1-GPU vs 4-GPU, compared `allclose` at a stated
  tolerance — with the fusion workaround (rule 3).
- Gate: tests green on GPU, 1-vs-4 match. Update the scoping record and
  the roadmap bullet.

### T2 — Immersed PCG paths: validate on real 4 GPUs

Roadmap "Immersed partial cells — residuals" (record:
`design/plans/active/immersed_partial_cells_plan.md`,
"Distributed correctness" bullet). Forced-4 CPU is asserted
(`test_immersed_step_is_device_count_invariant` in
`tests/nonhydro2/test_distributed_projection.py`); real multi-GPU is
not.

- Enumerate the three immersed masked-PCG paths from the plan record
  and run their multi-device assertions on real 4 GPUs.
- Add one genuine-partial-fractions smoke (not just the face-aligned
  {0,1} box): 1-GPU vs 4-GPU device-count invariance at test tolerance.
- No committed baselines change (per the plan's perf note). Update the
  plan residuals bullet and the roadmap.

### T3 — 4-GPU memory signature: attribute the 1024×1024×768 death

Roadmap "Gaps against the Oceananigans reference comparison", second
bullet. Known state: 1024×1024×512 fits (~31 GiB/GPU steady);
1024×1024×768 dies **in compile** (remat). The single-GPU ceiling story
is closed (`design/research/gpu_memory_ceiling.md`) — this needs its
own attribution: per-device arena/fragmentation vs genuine remat.

- Reproduce both sizes on current dev, single-process 4-GPU GSPMD. The
  comparison harness is out-of-tree; write a small driver (reuse the
  `benchmarks/model` harness patterns, including the sync-live-`_carry`
  measurement rule).
- Separate the hypotheses: capture the exact failure (compile-time OOM
  message? XLA remat diagnostics?), sweep allocator config within known
  safe bounds (BFC vs `cuda_async` at ≤0.75, MEM_FRACTION), inspect
  peak-live via XLA buffer-assignment stats / `--xla_dump_to` /
  `jax.profiler.device_memory_profile`.
- Gate: a written attribution with evidence (which knob changes the
  outcome, peak-live numbers), appended to `gpu_memory_ceiling.md` or a
  sibling record; the roadmap bullet trimmed to whatever lever remains.

### T4 — WENO selected-input walled path under real multi-process

Roadmap "WENO selected-input follow-ups" (records:
`design/research/stencil_lowering.md`,
`design/research/multidevice_test_faults.md`). Single-controller
forced-4 is exercised; real multi-process is not.

- Script per rule 2 (`jax.distributed.initialize()` before importing
  fridom): a walled nonhydro config with `weno5` advection that
  exercises the selected-input walled path; launch
  `JAX_PLATFORMS=cuda timeout <t> srun -n 4 --gpu-bind=none ...`.
- Verify clean completion (no crash, no hang) and physics against a
  single-process reference — gather shards via `process_allgather`.
- Gate: match recorded in `stencil_lowering.md`; roadmap bullet
  trimmed. (The forced-4 knife-edge divergence flip is already
  understood — kernel-shape roundoff, see `multidevice_test_faults.md`
  — do not re-litigate it.)

### T5 — Channel-eigenmode GPU lowering fault: minimal repro + mitigation

Roadmap "Channel eigenmodes are broken on multi-device" (record:
`design/research/multidevice_test_faults.md`). Upstream-attributed
XLA:GPU/GSPMD fault: a c64 FFT-norm constant is synthesized against the
c128 cuFFT output inside the large sharded projection module, so the
HLO verifier kills the projection on real multi-GPU. **Not** covered by
the `multi_output_fusion` workaround. The traced jaxpr is clean (zero
complex64) — the earlier fridom-side dtype reading is refuted; do not
reopen it.

- Reproduce on current dev, real multi-GPU.
- Reduce to a **minimal standalone repro** (ideally pure jax/XLA, no
  fridom imports) ready to file upstream. Do **not** file — filing
  needs the owner's go-ahead; leave the repro script + a drafted issue
  body in the research record's directory and link them from it.
- Fridom-side mitigation: try keeping the FFT norm scaling **outside**
  the fused sharded kernel. If that clears the verifier with correct
  results, ship it (with mirrored tests) on a `fix/` branch. If not,
  implement the taught multi-device skip on the channel eigenbasis so
  the projection fails loudly instead of dying in the HLO verifier —
  also with mirrored tests.
- If time permits: the sibling forced-CPU `sort` segfault repro (exit
  139) needs no GPU and can be minimized in the same session.
- Gate: repro artifacts committed, mitigation or taught skip merged,
  record + roadmap updated.

### T6 — CG tolerance: step-level re-measure on A100 (piggyback)

Roadmap "Mapped-solve residual levers", last bullet (research:
`design/research/cg_stopping_criterion.md`). Needs an A100, not 4 of
them — do it while on the node. The 3× forward win (8.2 vs 24.6 ms) is
**CPU micro-timing on a standalone solve**; XLA:GPU buffer assignment
inside the chunked step has reversed such wins before (padded-carry
experiment, krylov docstring).

- `benchmarks/model/bench_step.py`, `nh_mapped` cases, tolerance
  opt-in vs default fixed-iteration, single-GPU (optionally also
  gpu4). Report per-case deltas.
- Gate: numbers appended to `cg_stopping_criterion.md` and the roadmap
  lever bullet updated. Do **not** flip the default — that is an owner
  call, additionally gated on operational floor-trap experience.

### T7 — Re-record the step baselines (LAST, on final merged dev)

Roadmap "Finite-volume nonhydro", last bullet. The committed
`benchmarks/baselines/step-gpu4.json` predates the nodal sibling cases
(`nh_flat_periodic_nodal` / `nh_flat_advective_nodal` exist only in
gpu1) **and** the FV default flip; the walled cases in both baselines
predate the flip.

- First run `bench_step.py --fail-on-regression` against the old
  baselines and **attribute** any large delta before re-recording (FV
  and nodal are expected same-speed; an unexplained regression means
  bisect first — never blame the newest commit by default, and never
  cache values measured under trace).
- Re-record `step-gpu4.json` in full (adding the nodal sibling cases)
  and the walled rows of `step-gpu1.json`, on a **clean checkout of
  merged dev** (the metadata records commit + dirty flag — record
  clean). Commit on a `perf/` branch.
- Gate: baselines committed, regression guard green against
  themselves, roadmap bullet moved to `done.md`.

## Explicitly out of scope for this session

- The Oceananigans comparison-suite re-run (out-of-tree harness,
  single-GPU, largely re-checked 2026-07-17).
- CI perf-gate wiring (no GPU needed).
- The hydrostatic `im_centered` ≥512² instability lead (single-GPU
  repro; separate investigation).
- Filing the upstream jax issues (owner go-ahead required — prepare,
  don't file).

## Wrap-up

Per AGENTS.md: no leftover branches or worktrees. Roadmap hygiene
applied for every shipped item. Final report: one verdict + the key
numbers per task (T1–T7), plus anything that failed to reproduce and
where the negative result is recorded.
