---
status: done
date: 2026-07-16
---

# Indivisible-extent sharding — the multi-device performance hole

> **Shipped 2026-07-16.** All four phases landed on `dev` the day
> after the plan was written (merges `b6a5cc47` Phase 1, `1114b5ab`
> Phase 2, `aa7d0a40` Phase 3 stepper spelling, `2adae87b` Phase 3
> shard-axis ordering, plus the baseline re-record). Measured
> outcomes (4×A100, nh 256³ linear, ms/step): **x-walled 15.62 →
> 3.36** (was 2.3× slower than 1 GPU, now ~2× faster, incl. the
> shard-axis reorder), **xyz-walled 16.69 → 4.34**, **prime 257³
> 9.18 → 4.02** (4-GPU now 2.14× faster than 1-GPU, vs 2.8× *slower
> than 256³* before; residual vs divisible = cuFFT Bluestein, paid on
> one device too). Bonus from the excursion hygiene: **1-GPU nh_flat
> 11–17% faster** across 256³/512³, bitwise-identical. Phase
> sections below record what was *planned*; deviations are flagged
> inline. Follow-ups that outlived the plan: the surplus (`n+1`) leg
> stays on the global reblock path (no hot-loop consumer; documented
> at the gate), multi-host validation is still open (roadmap), and
> the validation campaign surfaced **pre-existing** multi-device
> faults catalogued in
> [`../../research/multidevice_test_faults.md`](../../research/multidevice_test_faults.md).

**One line.** When a field's extent along the **sharded** axis is not
divisible by the device count `P`, the step pays collectives on almost
every operation — a **2.8×–4.8× slowdown** at 256³ on 4×A100. Two
triggers, now both root-caused and with validated fixes: a **walled
sharded axis** (the staggered velocity leg becomes `n_cells−1`,
indivisible → per-op reblock collectives) and an **indivisible domain
size** (any `n % P ≠ 0`, e.g. prime 257 → the distributed solve
declines and replicates). A `(p, p, p)` prime domain hits it on every
axis, so no choice of sharding escapes.

Owner flagged it "needs to be fixed basically now" (2026-07-15).
Probe evidence is frozen in
[`../../research/indivisible_shard_probes.md`](../../research/indivisible_shard_probes.md)
(referenced below as **P1–P4**).

## Why it matters

- The decomposition silently shards **axis 0** (`default_layout =
  layouts[0]`; `_shardable_names` qualifies by **cell-count**
  divisibility only, blind to staggering). So the worst case is also
  the *default* case whenever axis 0 is walled.
- The distributed spectral solve **declines** to the replicated
  all-gather cube when the split axes are indivisible — an indivisible
  domain loses the whole distributed-transform win.
- Production runs will not always pick `P`-divisible sizes; 257³ must
  be a first-class citizen.

## Evidence (2026-07-15, 4×A100-80GB, jax 0.10.x, nh 256³ linear, ms/step)

Env: `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`
(jax-ml/jax#39100), f64, 50-step chunks, `block_until_ready`, median.

### 1. The wall sweep is bimodal on whether the SHARDED axis (x) is walled

| config | x (sharded) walled? | 1-GPU | 4-GPU | 4-GPU vs 1-GPU |
|---|---|---|---|---|
| none | no | 5.80 | 3.28 | **1.77× faster** |
| y | no | 6.64 | 3.43 | 1.94× faster |
| z | no | 6.68 | 3.30 | 2.02× faster |
| yz | no | 7.43 | 4.79 | 1.55× faster |
| **x** | **yes** | 6.83 | **15.69** | **2.30× slower** |
| **xy** | **yes** | 7.48 | **15.82** | 2.11× slower |
| **xz** | **yes** | 7.67 | **16.27** | 2.12× slower |
| **xyz** | **yes** | 9.75 | **16.77** | 1.72× slower |

The 1-GPU column is the control: walls cost the same on any axis
without sharding. The 4-GPU asymmetry is purely a sharding artifact.

### 2. Whole-step compiled collectives (forced-4 CPU, backend-independent)

| config | `u` extent on sharded axis | all-to-all | collective-permute | all-gather | dynamic-slice |
|---|---|---|---|---|---|
| none / z / yz | 64 (`÷4` ✓) | 10 | ~132 | 0 | ~21 |
| **x / xyz** | **63 (`n−1`, ✗)** | **245** | **~745** | **20** | **~525** |

### 3. Indivisible domain size (prime) is slow regardless of walls

Triple-periodic, 4-GPU:

| N | factor | `÷4`? | ms/step | vs 256 | how |
|---|---|---|---|---|---|
| 256 | 2⁸ | ✓ | 3.28 | — | distributed (10 all-to-all) |
| **257** | **prime** | ✗ | **9.18** | **2.8×** | solve **replicates**: all-to-all → 0, +5 all-gather |
| 260 | 4·65 | ✓ | 3.59 | 1.1× | distributed |

It is divisibility by `P`, not powers of two.

## Root cause — proven (P1)

The stencil operators and the solve are innocent. The causal chain for
the walled sharded axis:

1. **The hot loop makes true-frame excursions.** The AB3 stepper's
   `_weighted` (`f.with_data(w * f.data)`, `adam_bashforth.py:563`),
   the coriolis space-lifts (`scalar_field.py` `_lift_field` /
   `_broadcast_factor` / `.to`), and per-chunk scrub/isfinite all
   round-trip fields through `unpad → arithmetic → pad`.
2. **The reblock gate excludes the walled staggered leg.**
   `_build_local_reblock` (`decomposition/tensor.py:525`) returns
   `None` for the *divisible staggered-deficit* case
   (`n = P·cells − 1` with `n_cells % P == 0` — exactly the walled
   face velocity), so its pad/unpad take the legacy **global
   slice/concat** path. (The non-divisible mild case, e.g. center 257,
   already uses the fast padded-even plan — the walled leg is the one
   residue left behind.)
3. **The global path replicates.** A `P·cells − 1` true array cannot
   block-align with the storage blocks; `unpad` materializes it
   **replicated** and `pad` re-scatters — collectives per excursion,
   per step, independent of N.

For the indivisible domain, the chain is separate and simpler: the
transform's `_distributed_geometry` (`transform.py:794–801`) requires
the sharded axis `a` **and** the transpose partner `b` to divide `P`;
257 fails on every axis → `build_distributed_plan` returns `None` →
the replicated composite chain all-gathers the cube per solve.

## What the probes established (P1–P4)

- **The gate fix works and is exact.** Routing the deficit leg through
  the existing padded-even plan is byte-for-byte identical in storage
  placement, `unpad∘pad == id`, and collapses the whole x-walled step
  to the periodic baseline (collective-permute 274→34, all-gather
  4→0) in a monkeypatch A/B.
- **A padded all-to-all transpose handles indivisible split axes with
  zero extra collectives** — local `jnp.pad` before, local slice
  after, FFTs at true length; machine-precision vs the replicated
  reference; ~2× faster than replicating at meaningful sizes (P2).
- **"Pad without copying" does not exist, and is not needed.** XLA
  buffers are immutable; donation aliases only same-shape outputs. But
  a pad consumed inside one jit **fuses to zero cost**, and an uneven
  true-extent global array is **unrepresentable** in jax anyway
  (`device_put`/slicing refuse). Doctrine: allocate padded once, true
  extent is metadata, the hot loop never leaves the padded frame; a
  pad/unpad across a jit boundary is a full-bandwidth copy (P3).
- **Pad-and-mask over a balanced all-to-all is the industry idiom**
  (GSPMD internals, P3DFFT `USEEVEN`); jax has no practical
  alltoallv (`ragged_all_to_all` is MoE-shaped/experimental), and
  `with_sharding_constraint` on indivisible dims silently replicates
  (jax#26946) — never rely on it (P4).

## The fix plan

### Phase 1 — reblock gate: fix the walled sharded axis

**Change.** `_build_local_reblock` (`tensor.py:525`): reject only
`n > shards * cells`; let the divisible-deficit leg build the
padded-even plan (`surplus = 1`). Extend the plan to the **surplus
leg** (`n = n_cells + 1`, Neumann outer, which still falls through) —
alignment arithmetic says ≤1 collective-permute via the same
machinery (P1), but this case is *not yet empirically validated*:
prove storage byte-identity + `unpad∘pad == id` for it during
implementation, as `mechanism_05_H2` did for the deficit.

**Tests** (mirrored: `tests/spatial/decomposition/`):
- storage byte-identity fast-vs-global for the ±1 legs across
  `P ∈ {2,3,4}`, `n_cells` residues `{0,1,P−1} mod P`, with/without
  ghost widths; `unpad(pad(x)) == x`.
- whole-step collective-count regression: x-walled == periodic
  (forced-4, small N) — the P1 attribution harness distilled to a
  test.
- existing divisible-path HLO goldens unchanged
  (`tests/spatial/decomposition/golden/`), center/outer
  collective-free guarantee intact.

**Gates.** Mirrored tests + `ruff` (merge gate); bitwise vs dev on
divisible configs; A100 sweep re-run: **x/xy/xz/xyz must collapse onto
their y/z/yz mirrors** (~15.7 → ~3.4 ms/step); step-suite baselines
(`benchmarks/baselines/step-gpu{1,4}.json`) — no regression on any
divisible case, then re-record the walled cases.

Small, contained; one session including GPU validation.

### Phase 2 — distributed transform on indivisible split axes: fix primes

**Change.** Accept indivisible `a`/`b` in
`Transform._distributed_geometry` (`transform.py:794–801`) and lower
the SlabPlan's transposes (`distributed_solve.py` all_to_all sites)
with the P2 spelling: local pad of the split axis to `ceil(n/P)·P`
before `lax.all_to_all(tiled=True)`, local slice after; every FFT/trig
stage runs at **true** length (the relevant axis is always locally
full). Eigenvalue divide is per-shard local; mask/ignore pad lanes
(they never mix — P2 verified zero pad-lane leak, but the inverse
must re-zero them before the return transpose if any stage could
write there).

Planner policy: prefer a divisible partner axis `b` when one exists
(keeps today's byte-identical fast path); pad only the axes that need
it. Divisible operands must produce **byte-identical programs** to
today (the `distributed_transform_plan.md` gate). Mixed
Fourier⊗Sine/Cosine stages compose unchanged — stages are local
kernels; only the transpose spelling changes.

**Tests.** Plan-eligibility units (257 accepted, plan describes padded
frames); parity distributed-vs-replicated ≤1e-14 at 257³ and a mixed
walled+indivisible case; solve HLO: all-to-all only, **no
all-gather**; 1-device program bitwise unchanged; compile-count
regression (plans memoized).

**Gates.** A100: 257³ 4-GPU beats 1-GPU and lands near 260³
(≈3.6 ms/step; residual gap = cuFFT Bluestein at prime length, which
1-GPU pays too); 768³-class memory ceiling preserved (no replicated
cube). Re-record baselines; add a 257³ case to
`benchmarks/model/bench_step.py` so this never regresses silently.

Medium; 1–2 sessions. Interacts with
[`distributed_transform_plan.md`](distributed_transform_plan.md) —
the padded transpose belongs in the plan lowering, not as a bypass.

### Phase 3 — optional hygiene (after 1+2 land)

- **Storage-frame `_weighted` + excursion audit.** Spell AB3
  `_weighted` as `with_storage(w * f._data)` (bitwise-identical, P1)
  and audit the coriolis lifts. Defense in depth plus a small win for
  *divisible* runs (fewer reblock round-trips). Must pass the 1-GPU
  step suite — the 2026-07-14 lesson: storage-frame carry changes can
  regress single-device XLA fusion; the committed suite gates it.
- **Shard-axis selection respecting staggering** (`_shardable_names`
  ordering): prefer an axis whose staggered legs stay divisible.
  Demoted from "fix" to heuristic — after Phase 1 the walled axis is
  no longer catastrophic; this only shaves the ≤1-permute residues.
  Check consistency with the transform planner's `b`-partner
  preference before reordering (`transform.py:783` resolves `a` from
  `default_layout`).

### Non-goals / rejected

- **`jax.lax.ragged_all_to_all`** — MoE-shaped, HLO-only, GPU-
  experimental, still divisibility-constrained on offsets (P4).
- **Pad-everything-to-P storage rework** — unnecessary; the existing
  padded-even storage already is the invariant, Phase 1 merely stops
  excluding one residue from it.
- **Uneven true-extent sharding** — unrepresentable in jax (P2);
  never materialize a global true-extent array on the mesh.
- **`with_sharding_constraint` band-aids** — silent replication
  footgun on indivisible dims (jax#26946).

## Open questions

- **Multi-host** (real `srun -n P`): confirm Phase 1+2 behaviour under
  a genuine multi-process launch — forced-4 is single-controller, and
  the padded transpose + reblock plans must not host-fetch true-extent
  arrays (they don't by construction, but verify).
- The `n_cells ≡ 1 (mod P)` deficit residue keeps ≤1
  collective-permute per reblock (neighbour shift). Accepted; Phase 3's
  excursion audit reduces how often it is paid.
- Does any wall/BC operator *write* into the pad lanes of the
  transpose frames (Phase 2 masking assumption)? Verify with the
  pad-lane-leak check from P2 on the real solve.

## Method / repro

nh linear, `AdamBashforth(3)`, advection off, f64, 50-step chunks,
`block_until_ready`, median; 4-GPU needs
`XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`. Collective
counting and HLO attribution method: see the research record's method
notes (P1). Counting on CPU vs GPU differs in spelling (async
`-start/-done`, all-to-all decomposition into permutes) — compare like
with like.

## Related records

- [`../../research/indivisible_shard_probes.md`](../../research/indivisible_shard_probes.md)
  — P1–P4 probe evidence (frozen).
- [`distributed_transform_plan.md`](distributed_transform_plan.md) —
  the planner Phase 2 extends; its divisible-path byte-identity gate
  is binding.
- [`../done/uneven_shard_padding_plan.md`](../done/uneven_shard_padding_plan.md)
  — the padded-even storage contract Phase 1 completes.
- [`perf_geometry_merge_plan.md`](perf_geometry_merge_plan.md) — §8b's
  walled-4-GPU fusion effect is a different, smaller mechanism.
- Memory: `uneven-shard-reblock-collective`,
  `new-stack-gpu-performance`.
