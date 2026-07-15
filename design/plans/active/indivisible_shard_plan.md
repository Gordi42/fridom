---
status: active
date: 2026-07-15
---

# Indivisible-extent sharding — the multi-device performance hole

**One line.** When a field's extent along the **sharded** axis is not
divisible by the device count `P`, the whole step pays collectives on
almost every operation — a **2.8×–4.8× slowdown** at 256³ on 4×A100.
Two triggers, one root cause: a **walled sharded axis** (the staggered
velocity leg becomes `n_cells−1`, indivisible) and an **indivisible
domain size** (any `n` with `n % P ≠ 0`, e.g. a prime). A `(p, p, p)`
prime domain hits it on **every** axis, so no choice of sharding
escapes.

This is a first-class correctness-of-performance hole, not a
walled-grid corner case: production runs will not always pick
`P`-divisible sizes. Owner flagged it "needs to be fixed basically
now" (2026-07-15).

## Why it matters

- The decomposition silently shards **axis 0** (`default_layout =
  layouts[0]`, and `_shardable_names` iterates `x, y, z` picking the
  first qualifying axis by **cell-count** divisibility — blind to
  staggering and to walls). So the worst case is also the *default*
  case whenever axis 0 is walled.
- The distributed spectral solve **declines** (falls back to the
  replicated all-gather cube) when the operand's sharded axis is
  indivisible — so an indivisible domain loses the whole distributed
  transform win and pays an all-gather per solve on top.
- It compounds with the walls: a prime **and** walled domain suffers
  both the reblock explosion and the replicated solve.

## Evidence (2026-07-15, 4×A100-80GB, jax 0.10.x, nh 256³ linear, ms/step)

Env: `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`
(jax-ml/jax#39100), f64, one 50-step chunk, `block_until_ready`,
median. Harnesses in the perf scratchpad (session 696a6b5a):
`ab_sweep.py`, `prime_probe.py`, `mechanism_probe.py`,
`collective_probe.py`.

### 1. The wall sweep is bimodal on whether the SHARDED axis (x) is walled

| config | walled | x (sharded) walled? | 1-GPU | 4-GPU | 4-GPU vs 1-GPU |
|---|---|---|---|---|---|
| none | — | no | 5.80 | 3.28 | **1.77× faster** |
| y | y | no | 6.64 | 3.43 | 1.94× faster |
| z | z | no | 6.68 | 3.30 | 2.02× faster |
| yz | y, z | no | 7.43 | 4.79 | 1.55× faster |
| **x** | x | **yes** | 6.83 | **15.69** | **2.30× slower** |
| **xy** | x, y | **yes** | 7.48 | **15.82** | 2.11× slower |
| **xz** | x, z | **yes** | 7.67 | **16.27** | 2.12× slower |
| **xyz** | x, y, z | **yes** | 9.75 | **16.77** | 1.72× slower |

The **1-GPU column is the control**: with no sharding, walling x, y, or
z each costs the same (~6.7 ms, monotone in wall count). The 4-GPU
asymmetry is therefore *purely* a sharding artifact — identical
one-wall physics is fast when the wall is off the shard axis (y/z:
3.3 ms) and 4.8× slower when it is on it (x: 15.7 ms).

### 2. The mechanism — whole-step compiled collectives (forced-4 CPU, N=64)

Collective counts are backend-independent (a function of program +
mesh), so forced-4 CPU equals the GPU program; N small for compile
speed, same divisibility residue as 256.

| config | `u` extent on sharded axis 0 | all-to-all | collective-permute | all-gather | dynamic-slice |
|---|---|---|---|---|---|
| none / z / yz | 64 (`÷4` ✓) | 10 | ~132 | 0 | ~21 |
| **x / xyz** | **63 (`n−1`, ✗)** | **245** | **~745** | **20** | **~525** |

Only `u` (the x-face-staggered velocity) goes to 63 on the walled
sharded axis; `v, w, b` stay 64. That single indivisible leg mismatches
the cell-centered storage blocks (`ceil(n_cells/P)`), and the
uneven-shard reblock machinery
([`decomposition/tensor.py`](../../../src/fridom/spatial/decomposition/tensor.py),
memory `uneven-shard-reblock-collective`) inserts collectives on ~every
field op across the step. The pressure solve is **not** implicated: its
2 all-to-alls are a rounding error next to 245, and it is byte-identical
across all wall configs (verified: `collective_probe.py` — every config
issues exactly 1 all-to-all forward, 1 backward, no hidden
all-gather/all-reduce, ~135 MB transposed either way, since rfft halves
the count but c128 doubles the bytes and fully-walled moves f64 on the
full cube).

### 3. Indivisible domain size (prime) is slow regardless of walls

Triple-**periodic**, 4-GPU:

| N | factor | `u.shape[0]` | `÷4`? | ms/step | vs 256 | how |
|---|---|---|---|---|---|---|
| 256 | 2⁸ | 256 | ✓ | 3.28 | — | distributed (10 all-to-all) |
| **257** | **prime** | 257 | ✗ | **9.18** | **2.8×** | solve **replicates**: all-to-all → 0, +5 all-gather |
| 260 | 4·65 | 260 | ✓ | 3.59 | 1.1× | distributed |

N=260 (even, not a power of two) is fast — it is **divisibility by `P`**,
not powers of two. N=257 is 2.8× slower: the cell-centered pressure is
indivisible on every axis, the distributed slab declines (its tiled
all_to_all needs an exactly divisible split axis), and the solve
all-gathers the cube. Shard-axis selection cannot save this — a prime
domain has no divisible axis.

## Root cause, precisely

1. **Shard-axis choice is divisibility-blind to staggering.**
   `_shardable_names`
   ([`decomposition/decomposition.py`](../../../src/fridom/spatial/decomposition/decomposition.py)
   ~L891) qualifies an axis on the **cell** count only; `default_layout`
   returns `layouts[0]` (the first qualifying axis). A walled axis whose
   cell count divides `P` is chosen even though its face-staggered leg
   (`n_cells−1`) does not divide `P`.
2. **The reblock is not collective-free for the `n_cells−1` residue.**
   The padded-even fast reblock is collective-free for the aligned
   (divisible) case; a staggered leg that is `n_cells−1` against
   `ceil(n_cells/P)` blocks forces a full reblock (all-to-all +
   all-gather), not the "one collective-permute" the isolated-reblock
   note describes — in the full step every op touching `u` re-pays it.
3. **The distributed solve declines on an indivisible sharded axis** and
   falls back to the replicated (all-gather) cube.

## Fix options (ranked; none implemented — this is the open work)

1. **Shard-axis selection that respects staggering (cheap, partial).**
   Choose the default sharded axis as one on which **every** field
   (cell *and* staggered) stays divisible by `P` — in practice a
   periodic axis, or an axis whose staggered legs keep an even extent.
   Rescues **x / xy / xz for free** (each keeps ≥1 periodic axis).
   *Does nothing for a prime domain* (no divisible axis exists). Small,
   contained change to `_shardable_names` ordering / `default_layout`.
   Validate by re-running the sweep: x/xy/xz should collapse onto their
   y/z/yz mirrors.
2. **Collective-free reblock for the `n_cells±1` staggered residue
   (general for walls).** Make the pad/unpad genuinely collective-free
   when only the staggered leg is off by one — align the staggered
   field to the cell block layout with a local pad rather than a
   cross-shard reblock. Fixes the walled sharded axis even when *no*
   divisible axis exists (i.e. fully-walled xyz), which option 1 cannot.
3. **Distributed solve on an indivisible sharded axis (general for
   primes).** Let the slab transpose handle a non-divisible split axis
   (pad-to-`ceil` all_to_all, mask the tail) instead of declining to the
   replicated cube. This is the only lever that helps a prime periodic
   domain. Heaviest; interacts with the distributed-transform planner
   ([`distributed_transform_plan.md`](distributed_transform_plan.md)).
4. **Pad the sharded axis to a multiple of `P` at storage level.**
   Store every field padded up on the sharded axis so all legs align by
   construction; trades memory + edge masking for zero reblock
   collectives. Blunt but uniform; covers both triggers.

Likely shipping order: **1** (immediate, rescues the common
walled-with-a-periodic-axis case), then **2** + **3** as the general
fix for fully-walled and prime domains. **4** is the fallback if 2/3
prove intractable.

## Open questions

- Does option 1's axis reorder interact with the distributed-transform
  planner's own `b`-axis preference (it wants a Fourier axis local for
  the rfft)? The solve resolves `a` from the transform plan, not the
  default layout — check they stay consistent.
- For option 2, is the staggered `n_cells−1` always a *local* pad away
  from the cell block layout, or are there stencils that read across the
  shard boundary into the padded slot?
- Multi-host (real `srun -n P`) behaviour: the forced-4 / single-
  controller measurements here are single-process; confirm the reblock
  explosion and the replicated-solve fallback reproduce (and are not
  worse) under a genuine multi-process launch.

## Method / repro

nh 256³ linear, `AdamBashforth(3)`, advection off, f64, one 50-step
chunk (`chunk_size = 50` so scan-unroll engages), `block_until_ready`,
median. 4-GPU needs `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`.
Collective counts via `_CHUNK_EXECUTABLES` (whole step) and
`plan._forward/_backward/_build_solve(...).lower(...).compile().as_text()`
(solve), counting instruction-definition lines and folding async
`-start/-done` pairs. GPU async collectives spell as `all-to-all-start`;
a naive substring count on CPU over-counts — count opcodes.

## Related records

- [`distributed_transform_plan.md`](distributed_transform_plan.md) — the
  distributed slab solve this sits beside (the solve itself is *not* the
  bottleneck here).
- [`perf_geometry_merge_plan.md`](perf_geometry_merge_plan.md) — the
  perf line; §8b's "walled 4-GPU regression" is a *different* effect
  (fusion re-derivation of the fill), not this reblock explosion.
- Memory: `uneven-shard-reblock-collective`,
  `new-stack-gpu-performance` (the 2026-07-15 investigation entry).
