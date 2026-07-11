# Framework2 kernel-optimization results (Wave 4B)

> **Naming:** these measurements predate the 2026-07-11 package split; "framework2" is today's `fridom.spatial` + `fridom.model`.

Machine caveat: all numbers are **CPU wall-time medians on one
8-core desktop machine** (single jax cpu device, float64). Only
*relative* comparisons are meaningful, and the run-to-run noise
floor is high (unchanged-code cases swing +-10-30% at n <= 1024
between sessions); the old-vs-new pairs below come from one
exclusive back-to-back session (`results/final-oldstack.json` /
`results/final.json`, 2026-07-07). GPU tuning is out of scope;
nothing GPU-hostile was introduced (static shapes, no host
callbacks, no gather/scatter, no roll).

Suites: `benchmarks/spatial/` + `benchmarks/model/` (kernel cases in
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

---

# Sync-strategy redo (task 1.8, 2026-07-07)

Consumption-side sync with halo-validity tracking replaced the
iteration-1 sync-after-every-operator placement (branch
`framework2-sync-redo`; plan and mechanism in
`design/plans/done/sync_redo_plan.md`). This closes the "remaining
gap on chained tendencies" the conclusion above left open — the
chain-level elision is now the shipped contract, generalized to the
whole composed step.

Comparison: `results/final.json` (pre-1.8, `dfb57113+`) vs
`results/post-syncredo.json` (`11ef4377+`), same machine, exclusive
run. 48 cases: 28 faster, 9 within noise, 11 slower.

| case (n=4096 rows shown) | pre-1.8 | post-1.8 | d wall |
|---|---|---|---|
| bench_tendency | 327.16 ms | 203.45 ms | **-37.8%** (n=1024: -52.3%) |
| bench_add | 108.88 ms | 40.03 ms | **-63.2%** (memory -100%) |
| bench_mul | 82.10 ms | 39.98 ms | **-51.3%** (memory -100%) |
| bench_weno_pair_field | 271.16 ms | 136.62 ms | **-49.6%** (memory -66%) |
| bench_fv_diff | 116.66 ms | 94.40 ms | -19.1% |
| bench_fft_roundtrip | 862.43 ms | 687.34 ms | -20.3% |
| bench_sync | 77.92 ms | 73.17 ms | -6.1% |
| bench_diff | 72.04 ms | 97.49 ms | **+35.3%** (see below) |

Reading:

- The composed cases carry the point: per-arithmetic syncs are gone
  (`add`/`mul` no longer allocate or exchange at all — the -100%
  memory), and the representative tendency drops 38-52%.
- `bench_diff` (+34-43% across sizes) is the one real regression: an
  *isolated first consumption* now pays its entry sync serially
  before the kernel (a full-buffer materialization on the critical
  path) where the old contract fused the sync into the kernel
  output. In a composed step this entry sync is paid once and
  amortized across every consumer (memoization) and chained op —
  which is exactly what `bench_tendency` measures. Accepted as the
  microbench artifact of the contract.
- Multi-device (not in this table): the forced-4 framework2 suite
  wall time dropped 23:47 -> 4:02 on this machine.
- Small-n WENO scatter (+47%/-34% at n=256) is the documented
  +/-10-30% noise floor at small sizes.

---

# Balance benchmark: propagate+rebalance (T5, 2026-07-11)

`bench_balance.py` — an **accuracy** benchmark (not a timing case):
the classic diagnosed-imbalance protocol (Chouksey et al. 2023 JFM
971 A2 style) for `fr.transforms.BalanceExpansion` orders 0-3 vs
`fr.transforms.OptimalBalance`, run manually once per release of the
method. Protocol: balance a geostrophically dominated IC
(`sw.random_vortical`, seed 123, amplitude = Ro on a 32^2 doubly
2-pi periodic f-plane `sw.Model`, c = f = 1, AB3 dt = 0.02),
integrate the full nonlinear model one eddy turnover `T = 1/Ro`,
rebalance with the same method; imbalance =
`||(I - P_div)(z_f - M(z_f))||_M / ||z_f||_M`. The Rossby number
lives in the state amplitude (`scaling.rossby` stays 1) so that
OptimalBalance's hard-wired 0 -> 1 rossby ramp targets exactly the
model's nominal nonlinearity. Deterministic; full run 51 s on one
cpu device (this machine).

## Propagate+rebalance imbalance (structural complement excluded)

| method | Ro=0.05 | Ro=0.1 | Ro=0.2 |
|---|---|---|---|
| BalanceExpansion(0) | 1.49e-02 | 2.83e-02 | 5.56e-02 |
| BalanceExpansion(1) | 2.41e-04 | 8.19e-04 | 3.34e-03 |
| BalanceExpansion(2) | 9.06e-06 | 3.18e-05 | 2.14e-04 |
| BalanceExpansion(3) | 8.39e-06 | 1.69e-05 | 4.75e-05 |
| OptimalBalance(ramp=2IP, max_it=2) | 3.64e-04 | 5.98e-04 | 8.60e-04 |

The excluded content is the raw structural floor (5.1e-3 relative
at every Ro and every method, bitwise dt-independent): the aliased
Sadourny advection pumps residual into the interpolation-Nyquist
planes where the staggered geostrophic column is a structural zero
(`DivergenceProjection` content, outside the eigenmode span). Every
method annihilates it identically, so it is an artifact of the
diagnosis at 32^2, not wave imbalance. The walled-channel column
has **no** such floor (raw == excluded below) — the dense channel
eigenbasis labels every column.

## residual_fast at t = 0 (the cheap differential diagnostic)

| order | Ro=0.05 | Ro=0.1 | Ro=0.2 |
|---|---|---|---|
| 0 | 3.18e-02 | 6.36e-02 | 1.27e-01 |
| 1 | 4.89e-04 | 1.96e-03 | 7.81e-03 |
| 2 | 8.38e-06 | 6.70e-05 | 5.36e-04 |
| 3 | 2.70e-07 | 4.32e-06 | 6.90e-05 |

**Ranking agreement (the key deliverable): AGREE at every Ro** —
the cheap `residual_fast` diagnostic orders the methods exactly
like the expensive propagate+rebalance protocol (3 < 2 < 1 < 0 at
all three Rossby numbers). This is what justifies never running
the propagation protocol in CI: the tests' epsilon-slope ladder and
`residual_fast` carry the ranking information.

## Walled channel (f-plane, walls in y) at Ro=0.1

| method | imbalance | raw | residual_fast(t=0) |
|---|---|---|---|
| BalanceExpansion(0) | 2.22e-02 | 2.22e-02 | 6.78e-02 |
| BalanceExpansion(1) | 6.36e-04 | 6.36e-04 | 2.28e-03 |
| BalanceExpansion(2) | 4.57e-05 | 4.57e-05 | 8.39e-05 |

## Against the plan's acceptance targets

Chouksey et al. 2023 diagnosed-imbalance ballpark at Ro = 0.1
(orders 0/1/2/3 ~ 5e-2 / 2e-3 / 3e-5 / 5e-6): measured
2.8e-2 / 8.2e-4 / 3.2e-5 / 1.7e-5 — orders 0-2 agree within a
factor ~2 (order 2 essentially exact), order 3 sits ~3x above its
target and is visibly floor-limited (orders 2 and 3 nearly
coincide at Ro = 0.05, ~8-9e-6 — the protocol floor of this
32^2 / one-turnover configuration, consistent with the literature's
1e-6..1e-7 numeric floors). Slopes between Ro = 0.1 and 0.2:
order 1 ~ 2.0, order 2 ~ 2.75, order 3 ~ 1.5 (floor-limited) —
against the ideal Ro^(N+1). No order inversion anywhere, matching
the literature (and confirming the historical "worse at order 3+"
was an implementation bug). OptimalBalance at deliberately modest
settings (2 inertial periods per leg, max_it = 2, fixed-point error
~2e-4) lands between orders 1 and 2; the literature's "order ~4
comparable to OB" refers to converged OB — cranking `--ob-ramp` /
`--ob-max-it` buys more.
