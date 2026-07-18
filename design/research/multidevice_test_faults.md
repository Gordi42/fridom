---
status: frozen
date: 2026-07-16
---

# Pre-existing multi-device faults surfaced by whole-dir forced-4 runs

Research report (see [`README.md`](README.md) for status). Catalogued
while validating the indivisible-shard fixes
([`../plans/done/indivisible_shard_plan.md`](../plans/done/indivisible_shard_plan.md));
**every item below reproduces on pre-campaign dev `0139577d`**, so none
was introduced by that campaign. They surfaced only because those
validation runs were the first to execute the whole `tests/nonhydro2`
directory under forced-4 devices — CI runs single-device, and the
multi-device gate covers the decomposition tests only.

## 1. Channel eigenmodes are broken on multi-device (real bug)

`tests/nonhydro2/test_transforms.py::test_channel_projection_is_device_count_invariant`
fails on every multi-device configuration, by two *independent*
mechanisms:

- **CPU, forced-4** (`XLA_FLAGS=--xla_force_host_platform_device_count=4`):
  hard **segfault** (exit 139) of the interpreter while jax 0.10.2
  lowers the `sort` primitive of the eigenvalue ordering —
  `eigen_channel.py:611 _generalized_eigh_diag` →
  `lax.sort` comparator lowering (`_sort_lower` →
  `trace_to_jaxpr_dynamic` → crash in `core.typeof`). Deterministic,
  fires with the single test in isolation. Killed pytest-xdist
  workers (`[gw2] node down`) in whole-dir runs, producing collateral
  `F` marks on unrelated tests — rerun individual files before
  believing failures from a run that lost a worker.
- **GPU, real 4 devices** (needs
  `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`): the
  projection itself dies in the HLO verifier —
  `Binary op multiply with different element types: c64[] and c128[]`
  in `jit(fft)`, reached via `_eigenbasis._contract_planes`
  (`_eigenbasis.py:706`) → `Transform.forward` →
  `Fourier._forward_fused_kernel` (`fourier.py:208`). A genuine dtype
  bug: the eigenbasis feeds a c64 operand into a kernel with c128
  constants on the multi-device path (the single-device branch of the
  same test passes).

Consequence: the channel eigenbasis / VorticalProjection **cannot run
multi-device at all** today. The single-device path is fine. Needs its
own fix campaign: the dtype bug is fridom-side and fixable; the sort
segfault is a jax/jaxlib 0.10.2 fault that wants a minimal upstream
repro plus a fridom-side mitigation or a CPU-multi-device skip.

## 2. Forced-4 advection-test sensitivities (test-infrastructure)

Eight tests fail identically on dev and on any campaign branch under
forced-4 CPU (verified same-set both sides, 2026-07-16):

- `test_advection.py::test_old_stack_tendency_parity[upwind-3|upwind-5|weno-3|weno-5]`
- `test_advection_background.py::test_old_stack_background_parity_at_ro_one[upwind3|weno5]`
- `test_advection_background.py::test_background_none_reduction_is_bitwise`
- `test_advection_walls.py::test_walled_biased_projected_tendency_stays_divergence_free[channel-and-lid-upwind5]`

Two known mechanisms plausibly cover them: **old-stack parity** tests
instantiate the old stack, which is *not* device-count invariant
(`framework/domain_decomposition` shards axis 0 across all visible
devices and requires divisibility), so under forced-4 the reference
itself changes; **bitwise** comparisons on forced-CPU hit
FP-reassociation (the documented backend gotcha that the
backend-aware `invariant` helper exists for). Each test needs the
appropriate guard (`single_device` mark for old-stack-parity, the
backend-aware helper for bitwise) — triage per test, not blanket.

Also long-known:
`test_weno.py::test_weno_reconstruct_is_device_count_invariant`
(forced-CPU FP-reassoc; fails on dev).

## 3. Status of the whole-dir forced-4 run as a gate

With item 1 deselected, `tests/nonhydro2` forced-4 on the campaign tip
is **581 passed / 8 failed (the item-2 set) — identical to dev**. The
whole-dir forced-4 configuration is therefore *usable as a
delta-vs-dev check* but is **not green** and must not be treated as a
pass/fail gate until items 1–2 are fixed.

## Repro

```bash
# segfault (CPU forced-4)
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
FRIDOM_TEST_FORCED_DEVICES=4 uv run pytest \
  "tests/nonhydro2/test_transforms.py::test_channel_projection_is_device_count_invariant"
# dtype bug (4 GPUs)
XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion JAX_PLATFORMS=cuda \
uv run pytest \
  "tests/nonhydro2/test_transforms.py::test_channel_projection_is_device_count_invariant"
```

Related: the per-worktree cold-compilation-cache trap (fresh worktrees
start with an empty `.jax_cache`; share it via
`FRIDOM_TEST_JAX_CACHE_DIR`) amplified all of the above into
hour-scale, crash-riddled suite runs before the mechanisms were
separated.

## Corrections and resolutions (2026-07-16, follow-up sweep)

Both item-1 faults were re-verified on the dev tip the same day (the
segfault: exit 139 in 56 s; the GPU error: identical HLO-verifier
message). Two findings amend the record above:

- **Item 1, GPU mechanism — re-attributed upstream.** The "genuine
  fridom-side c64/c128 dtype bug" reading above is refuted by three
  probes on dev (4x A100): (i) operand dtypes on the failing
  multi-device path are identical to the passing single-device path
  (state float64; the crashing forward is the z-axis `rfftn`,
  in=float64 / out=complex128 on both paths; the sharded axis is 0,
  the transform axis 2); (ii) `jax.make_jaxpr` of the projection
  contains **zero** complex64 nodes on either path (1091- vs 631-line
  jaxprs) — no fridom array, eigenvector constant, or metric is c64;
  (iii) an isolated `rfftn` over the same sharding lowers and runs
  cleanly, eager and jitted. The offending `c64[]` is the FFT norm
  scale constant (jax `_fft_core`'s `1/prod(s)`), synthesized at
  **XLA:GPU/GSPMD HLO lowering** of the large sharded projection
  module while the cuFFT custom-call output stays c128; the HLO
  verifier then rejects the mixed multiply. An upstream XLA:GPU
  lowering fault in the same family as jax#39100 — and **not**
  covered by the `multi_output_fusion` workaround, which was active
  in every repro run. Consequence: both item-1 mechanisms (CPU `sort`
  segfault, GPU c64 constant) are jax/XLA-side; the fridom work is a
  minimal upstream repro for each, plus a mitigation (e.g. keeping
  the norm scaling outside the fused sharded FFT) or a taught
  multi-device skip on the channel eigenbasis.
- **Item 2, path correction:** the WENO invariance test lives at
  `tests/spatial/operators/test_weno.py`, not
  `tests/nonhydro2/test_weno.py`.
- **Item 2 — resolved** (2026-07-16, `test/forced4-triage`, merged
  `0d139fc5`): `single_device` marks on the six old-stack-parity
  tests (the failures are O(0.1) divergence of the old-stack
  *reference* under forced devices, not FP noise), and the
  backend-aware `invariant` helper from the decomposition suite
  (duplicated locally per the self-contained-test-file convention) on
  the two bitwise asserts. The ninth case — the walled
  divergence-free gate — turned out **not to be a forced-4
  sensitivity at all**: the 1.066e-13 overshoot reproduces bit-for-bit
  on single-device CPU (the 8³ walled grid is too small to shard, so
  forced-4 runs the same single-device program; the overshoot is
  CPU-backend FP reassociation, also seen as 1.42e-13 on CI after the
  native-DCT landing) and was fixed independently the same day by
  bounding the residual relative to the tendency (`cbfc032a`). The
  bound is scheme-generic: the `weno5` parametrization that the
  selected-input kernel-shape roundoff later tipped is covered by the
  same fix (verified 2026-07-18, all 12
  `test_walled_biased_projected_tendency_stays_divergence_free` cases
  green under forced-4 CPU on dev). Gates
  on the four files: forced-4 CPU 281 passed / 6 skipped / 0 failed;
  single-device 287 passed, the marked tests running. With item 1's
  test deselected, the whole-dir `tests/nonhydro2` forced-4 run is
  green again and usable as a gate: 583 passed / 6 skipped /
  1 deselected / 0 failed on the merged dev (`0d139fc5`), consistent
  with the 581 / 8 record above (six marked tests skip, two pass).

## Item 1, CPU mechanism — re-attributed (2026-07-17, T5b)

The forced-CPU exit-139 crash is **not** a jax 0.10.2 `sort`-lowering
bug. Minimized on current dev (`c0259f74`, jax/jaxlib 0.10.2) to a
fridom-free repro; the root cause is a **heap corruption inside
jaxlib's batched CPU eigendecomposition**, and the `sort` lowering is
only where the already-poisoned heap happens to be walked next (heap
corruption aliases the crash site — it wanders run to run between
`_sort_lower`/`shaped_abstractify`, `_standard_weak_type_rule`,
`mlir.make_ir_context`, and glibc `malloc` detection).

- **Minimal repro exists** (no fridom, no `sort`, no sharding, one
  device): `jnp.linalg.eigh` on a batch of 144 `63x63` f64 symmetric
  matrices deterministically corrupts the heap (SIGABRT `corrupted
  size vs. prev_size`, 4/4). Batch <= 128 is fine; a single `63x63`
  is fine. `gdb` puts the abort in
  `jaxlib/cpu/_lapack.so`
  `jax::EigenvalueDecompositionSymmetric::Kernel` under
  `jax::ParallelBatchMap` — jaxlib fans the batch across the Eigen
  intra-op threadpool while each worker's `?syevd` also spins up
  OpenBLAS threads, and on this 256-thread node (2× EPYC 7763) the
  nested oversubscription overruns OpenBLAS's precompiled
  `NUM_THREADS` (`OpenBLAS warning: precompiled NUM_THREADS
  exceeded, adding auxiliary array`). fridom hits batch = 16*9 = 144:
  one `63x63` generalized-Hermitian pencil per Fourier mode of the
  `16^3` channel.
- **Trigger conditions:** CPU backend, `jax_enable_x64=True`, a
  many-core host (256 logical here), batched `eigh` with batch beyond
  OpenBLAS's precompiled thread budget. Independent of device count
  and of sharding — a plain single-device `eigh` crashes too, so the
  record's "single-device path is fine" holds only on low-core hosts
  (CI runners); it does **not** hold on this node. `--xla_force_host_
  platform_device_count=4` makes it fire regardless, and surfaces it
  in the following op's lowering (hence the original `sort` blame).
- **Mitigation:** `OPENBLAS_NUM_THREADS<=32` clears the isolated
  single-device eigh (2/2 OK); it does **not** clear the forced-4
  configuration (each forced device drives the eigh concurrently).
- **One-line minimization:** started from the fridom channel test →
  the crashing `argsort` operand aval is byte-identical to a
  non-crashing isolated `argsort` → removing the `sort` still crashes
  → `_generalized_eigh_diag`'s batched `eigh` alone crashes → a bare
  `jnp.linalg.eigh` on a 144-batch crashes; the `sort` is neither
  necessary nor sufficient.
- **Artifacts:** `artifacts/channel_sort_segfault/`
  (`repro_batched_eigh_heap_corruption.py` — the minimal cause;
  `repro_forced4_sort_alias.py` — reproduces the record's exact
  `sort`-lowering exit-139 symptom to demonstrate the aliasing;
  `issue_draft.md` — drafted jax-ml/jax issue, **not filed**, awaiting
  owner go-ahead).

## Item 1, GPU mechanism — re-attributed + mitigated (2026-07-17, T5)

Reproduced on current dev (branch `fix/channel-eigen-gpu-lowering`, off
dev `e0b96107`; jax/jaxlib 0.10.2) on real 4× A100, single-process
GSPMD, with `--xla_disable_hlo_passes=multi_output_fusion` active:
`test_channel_projection_is_device_count_invariant`'s `many`-device
`proj(z)` dies in the HLO verifier with the recorded
`Binary op multiply with different element types: c64[] and c128[]`
(reached via `fourier.py:256` `Fourier._forward_fused_kernel`
`jnp.fft.fftn`, the full-spectrum transform over the sharded x-axis).

**The 2026-07-16 "c64 FFT-norm constant" attribution is REFUTED.** The
offending `complex64` scalar is *not* jax `_fft_core`'s `1/prod(s)`
normalization. Evidence:

- A fridom-free repro reproduces the exact verifier error **with
  `norm=None`** (no `_fft_norm`, no `1/prod(s)` anywhere) — so the c64
  is not the norm constant. It also reproduces with `norm="forward"`;
  moving the norm outside the FFT (my first mitigation attempt) does
  **not** clear it.
- The SPMD-partitioned HLO (`--xla_dump_to`,
  `*.after_spmd_partitioner.txt`) shows the real source: XLA:GPU lowers
  a **sharded-transform-axis** FFT through its distributed Cooley-Tukey
  decomposition — a `while` loop `fft_collective_permute_body` doing
  local `fft` chunks + `collective-permute` + `all-to-all`. Its
  **twiddle-factor** constants are synthesized at `complex64` while the
  data is `complex128`:
  `%constant.1 = c64[] constant((0, -1.57079637))` (= −i·π/2),
  `%multiply = c64[] multiply(%constant.1, %get-tuple-element.4)`
  (c64 × c128), `%exponential = c64[] exponential(...)`, and
  `%constant.4 = c64[] constant((0, -0.392699093))` (= −i·2π/16). These
  are the distributed-FFT phase factors `exp(-2πik/N)`, emitted at c64.

**Minimization (fridom-free, pure jax/XLA).** The fault is *contextual*:
an isolated sharded FFT compiles fine (GSPMD all-gathers the axis → one
local cuFFT); the fault needs a downstream consumer (an einsum/dot) that
keeps the transform axis sharded, forcing the distributed decomposition.
Discriminated: (i) shard the **transform axis** + einsum → **crashes**;
(ii) shard a **non-transform axis** (transform axis local) + einsum →
**OK**; (iii) FFT alone, sharded transform axis → **OK**. So the trigger
is precisely *a sharded FFT axis whose result is contracted while still
sharded*. Artifacts:
`artifacts/channel_fftnorm_gpu/repro_distributed_fft_c64_twiddle.py`
(the three trials above) and `issue_draft.md` (drafted jax-ml/jax issue,
**not filed**, awaiting owner go-ahead).

**Fridom-side mitigation shipped — taught skip.** Task 3a ("keep the FFT
norm scaling outside the fused sharded kernel") is inapplicable: the c64
is a distributed-FFT twiddle, not the norm, and the norm-outside variant
still fails. Shipped the taught skip (3b) instead:
`_eigenbasis._reject_sharded_projection` guards `_contract_planes` and
raises a taught `NotImplementedError` when the grid's default layout
shards a **periodic (Fourier) axis** (`Layout.is_local(name)` per axis),
so the projection fails loudly with an actionable message
(`device_ids=(0,)`) instead of dying in the verifier. Precise: fires
only when a transform axis is genuinely sharded — never on single-device
or `device_ids=(0,)` grids, and never on a many-device grid too small to
shard (the collapsed case that the existing `n=8` channel tests exercise
green on 4 GPU). Test:
`tests/nonhydro2/test_transforms.py::test_channel_projection_rejects_a_sharded_periodic_axis`
(GPU-scoped — skips on the CPU backend so the n=16 eigenbasis's
batch-144 `eigh` does not hit the T5b heap-corruption crash; on 4 GPU
the `many` branch asserts the taught error and the `device_ids=(0,)`
branch asserts the projection still runs and is idempotent). The single-
device GPU projection is unchanged (validated: 176 passed / 3 skipped
across `test_eigenbasis` + `test_transforms` +
nh/sw `test_channel_eigenmodes` on 1 GPU).

**A real fix (deferred).** Making the projection *run* multi-device is
possible and correct: forcing the transform axis device-local before the
FFT (a `with_sharding_constraint` to replicate it, or a slab/pencil
reshard so each axis is local when its FFT runs) gives one local cuFFT
per shard and is **bit-for-bit identical** to the single-device result
(verified in pure jax, max abs diff 0.0). The channel `_contract_planes`
uses the plain `Transform.forward/backward` (naive GSPMD path); the
proper fix routes it through the slab distributed-transform lowering the
spectral solver already owns (`operators/distributed_solve.py`) — wider
blast radius, left as follow-up (see `roadmap/open.md`).

**Anomaly vs the frozen record.** Two corrections to the header of this
document: (1) the GPU c64 is the distributed-FFT twiddle, not the
FFT-norm constant (refuted with `norm=None`); (2) the failing transform
is the `fftn` full-spectrum stage over the sharded x-axis
(`fourier.py:256`), not `rfftn` — the earlier note cited the `rfftn`
z-stage.

## Item 1, GPU mechanism — real fix shipped (2026-07-18)

The deferred "real fix" landed (merge `e60259de`): the projection now
**runs** on a grid sharding a periodic axis, via a fused
`jax.shard_map` lowering
(`spatial/operators/distributed_contract.py`) that keeps every FFT
axis device-local when its transform runs — the distributed-FFT
lowering (and its c64 twiddles) is never emitted. Not the generic
slab planner: on the channel's two-stage periodic transform
`_distributed_geometry` would run fully complex (`h=None`),
incompatible with the engine's half-spectrum `q` frame, so the
lowering pins the half axis to the engine's `rfftn` frame and
transposes on the half **coefficient** extent (`n//2+1`, padded, empty
trailing pad shards allowed — transient lanes, zero-padded `q`/`w`).
Measured on 4× A100 (n=16 nonhydro channel): many-vs-`device_ids=(0,)`
max abs diff 1.08e-14, idempotency 4.9e-15, HLO all-to-all only,
0 warm recompiles, finite reverse-mode gradients. The taught skip
(`_reject_sharded_projection`) narrows to the remainder the lowering
declines: 2-D channel, the half axis itself sharded, non-1-D mesh.
The upstream fault itself is unchanged (the fridom-free repro still
fires on jax 0.10.2, re-verified 2026-07-18) — the two issue drafts
still await the owner's go-ahead.

**Two further pre-existing item-1 faults surfaced by the validation**
(both reproduce on the pre-merge dev tip; the whole-file 4-GPU
eigen-surface run fails identically with and without the fix): (a) a
**setup `GridFrozenError`** — on small sharded grids the
`linearize(model)` probe's halo demand exceeds the frozen halo, so
`channel_eigenpairs` cannot build the basis at all; (b) a **`mode()` /
synthesis crash** — the backward-only synthesis paths (`mode`,
`channel_random_state`) still hit the sharded-FFT fault, as they do
not route through the fused contraction. Tracked in
`roadmap/open.md`; the shipped projection fix is validated by
dedicated tests on non-freezing grids (n=12/16).

## Closure (2026-07-18, same day)

Both faults above are fixed on dev: the `GridFrozenError` was a
negotiate/verify cap asymmetry (merge `8a787452`), and the synthesis
paths route through the fused backward-only contraction on 3-D
channels (merge `e316987d`). Mechanisms and follow-on invariant fixes:
[`halo_sharding_invariants.md`](halo_sharding_invariants.md); campaign
record:
[`gspmd_naive_transform_illegality.md`](gspmd_naive_transform_illegality.md).
