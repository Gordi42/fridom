# Framework2 kernel-optimization results (Wave 4B)

Machine caveat: all numbers are **CPU wall-time medians on one
8-core desktop machine** (single jax cpu device, float64). Only
*relative* comparisons are meaningful, and the run-to-run noise
floor is high (unchanged-code cases swing +-10-30% at n <= 1024
between sessions); the old-vs-new pairs below come from one
exclusive back-to-back session (`results/final-oldstack.json` /
`results/final.json`, 2026-07-07). GPU tuning is out of scope;
nothing GPU-hostile was introduced (static shapes, no host
callbacks, no gather/scatter, no roll).

Suites: `benchmarks/framework2/` (kernel cases in
`bench_kernels.py`, application-path cases in `bench_operators.py`)
vs the old stack's `benchmarks/bench_operators.py`, both run through
`fridom.benchmarking` at identical sizes (n x n grids).
`results/pre-opt.json` / `results/post-opt.json` are the harness
runs bracketing the (reverted) storage-frame-arithmetic experiment
of log entry 1.

## Old vs new per kernel family (wall median, jit-compiled)

The old stack's `bench_diff` is `field.diff(0)` — kernel plus a thin
wrapper, **no sync**. Its framework2 kernel-level counterpart is
`bench_staggered_diff[order=2]`; `bench_diff` (framework2) is the
full application path including the contractual post-op sync.

| case | n | old `field.diff` | f2 kernel | f2 application (synced) |
|---|---|---|---|---|
| FD derivative (order 2) | 256 | 73 us | 48 us | 99 us |
| | 1024 | 1.93 ms | 1.94 ms | 3.34 ms |
| | 4096 | 34.5 ms | 34.4 ms | 72.0 ms |
| WENO-5 upwind pair | 256 | - | 131 us | 469 us (field pair + select) |
| | 1024 | - | 2.40 ms | 15.5 ms |
| | 4096 | - | 60.0 ms | 271 ms |

(The old stack has no kernel-level WENO benchmark case; parity of
the WENO numerics vs `weno_interpolation.py` is a Wave-4A test.)

The pure kernels are at old-stack speed: the slice-window fused
stencils were already written to the kernel rules, and the HLO shows
one fused loop per kernel. An order-2 formula variant
(`(x1 - x0) * c` vs the weighted sum) measured 18.9 vs 19.2 ms at
4096^2 — memory-bound, no change made.

## Application path, old vs new (wall median, jit)

Semantics when reading this table: **every framework2 application
returns a synced field** (iteration-1 contract); the old stack
returns unsynced results and syncs separately. The semantically
equal pair is old `diff` + `sync` vs framework2 `diff`.

| case | n = 1024 | old stack | framework2 |
|---|---|---|---|
| diff (unsynced vs synced) | | 1.93 ms | 3.34 ms |
| diff + sync (two jits) vs diff | | 6.72 ms | 3.34 ms |
| sync alone | | 4.79 ms | 4.66 ms |
| fft (forward) vs roundtrip/2 | | 19.4 ms | 19.6 ms |
| add / mul (synced in f2) | | ~2.2 ms (unsynced)* | 5.57 / 4.66 ms |
| tendency (interp+product+3 diffs)* | | 12.4 ms | 17.2-19.5 ms |

\* the old-stack `add`/`mul`/`tendency` rows come from an earlier
same-session profile script (the old suite has no such cases); the
f2 tendency range spans that session (17.2) and the final harness
run (19.5).

## Where the time goes (measured)

- Under jit the framework2 stencil arithmetic fuses into the
  halo-fill assembly (HLO-verified); the per-application overhead is
  the **post-application sync** — the slice+concatenate ghost-fill
  costs ~one extra full-array pass per application (diff: 1.9 ms
  kernel -> 3.3 ms application at 1024^2). At equal semantics
  (synced result) one framework2 `diff` beats the two
  separately-jitted old-stack calls `diff` + `sync` (3.34 vs
  6.72 ms) because the ghost fill fuses with the kernel.
- The remaining old-vs-new gap on chained tendencies (~1.4x at
  1024^2) is the sync-after-every-operator contract: the old stack
  syncs once per chain, framework2 once per operator. Eliding
  redundant syncs along chains is the **designed-for** optimization
  (fields.md storage contract) and was deliberately not implemented
  in this pass.
- Eager (interactive) applications are dominated by jax eager
  primitive dispatch (~70% of wall in cProfile at 256^2), not by
  framework Python; the framework-side static-structure resolution
  is now cached (below).
- Transforms: framework2's Fourier path is at parity per transform
  (19.6 vs 19.4 ms) and was measured *faster* than the old
  fft+ifft round trip in a same-session profile (35 vs 51 ms at
  1024^2); untouched.

## Optimization log (every change measured; losers reverted)

1. **Storage-frame linear arithmetic** (`+`/`-`/scalar ops on
   `_data` + sync, skipping the unpad->pad round trip): **REVERTED**.
   Isolated `add` at 1024^2: 5.8 -> 4.2 ms (-27%, same-process A/B,
   bitwise-equal results). But in composed chains the true-shape
   materialization boundary is what stops XLA from re-fusing the
   upstream stencil into every downstream halo-fill concatenate
   piece: the representative tendency went 18.7 -> 30.9 ms (+65%)
   with the HLO growing from 153 to 247 instructions (14 -> 23
   fusions, 16 -> 28 concatenates). Chains are the realistic
   workload; reverted (a note in `scalar_field._linear_combine`
   records the measurement).
2. **In-place sync fill** (`.at[].set` ghost writes instead of the
   concatenate assembly): **REJECTED at prototype stage**.
   Standalone sync at 1024^2: 7.8 -> 4.4 ms, but fused-after-kernel
   applications regress (diff 3.3 -> 4.2 ms, mul 4.4 -> 6.2 ms):
   XLA CPU materializes a copy for the dynamic-update-slice while
   the concatenate fuses with the producer kernel. Concatenate kept.
3. **Per-(space, layout) geometry/sharding caches in
   `TensorDecomposition`** (`_geometry`, cached `sharding`):
   **KEPT**. The static-structure resolution that ran per operator
   application now runs once per interned key: `storage_shape`
   6.5 -> 1.0 us/call, `sharding` 7.2 -> 0.4 us/call (20k-rep
   microbench). End-to-end this is a few percent of eager/trace time
   (below the machine's noise floor); the compiled path is provably
   unchanged (bit-identical jaxprs, see gates).
4. **WENO fold onto `apply_fv_staggered`** (wave-4A-flagged tail
   duplication): **KEPT**. `apply_fv_staggered` gained an `align=`
   parameter for biased (odd-size) kernels;
   `WenoReconstruction._apply_factor` now reuses it instead of
   duplicating the slice/pad/reach-check tail. Pure dedup: jaxprs
   bit-identical, equal-results tests green.
5. **Single-device `device_put` elision in `pad`/`zeros`**:
   **REJECTED** — the decomposition contract (class doc + tests)
   pins outputs committed to the negotiated sharding; the measured
   saving was ~0.15 ms/call on the eager path only.

## Gates

- Jaxpr fingerprints of every touched application path (WENO both
  biases/orders, diff, FV chain, add, scale, sync) are
  **bit-identical** between the final tree and the pre-change tree —
  numerics, compiled runtime, and memory provably unchanged.
- Fusion sanity check on the optimized HLO: stencil kernels land in
  fusions; no gather/scatter/host custom-calls (fft's custom-call is
  the sanctioned exception on the transform path).
- Compile-counter tests (no retraces across same-shape calls) green.
- Both suite variants green: plain (1368 passed) and forced-4
  (`XLA_FLAGS=--xla_force_host_platform_device_count=4
  FRIDOM_TEST_FORCED_DEVICES=4`); `tests/framework/utils` green;
  ruff clean.

## Conclusion

The framework2 kernels are at old-stack speed on cpu — nothing was
left on the table at the kernel level. At equal (synced) semantics
the application path beats the old stack's diff+sync; the remaining
gap on chained tendencies is the sync-after-every-operator contract,
whose chain-level elision is the designed-for feature explicitly out
of scope for this pass.
