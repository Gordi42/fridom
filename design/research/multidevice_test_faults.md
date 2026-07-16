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
  bounding the residual relative to the tendency (`cbfc032a`). Gates
  on the four files: forced-4 CPU 281 passed / 6 skipped / 0 failed;
  single-device 287 passed, the marked tests running. With item 1's
  test deselected, the whole-dir `tests/nonhydro2` forced-4 run is
  green again and usable as a gate: 583 passed / 6 skipped /
  1 deselected / 0 failed on the merged dev (`0d139fc5`), consistent
  with the 581 / 8 record above (six marked tests skip, two pass).
