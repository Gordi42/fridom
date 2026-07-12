---
status: done
date: 2026-07-12
---

# Uneven-shard padding — general non-divisible ghost sharding

**Shipped 2026-07-12** (`dev` merge `6bf2c032`). All stages landed as
specified; Option A settled (§6). Byte-for-byte no-op on divisible
extents proven by HLO golden (`tests/spatial/decomposition/golden/`);
device-count invariance verified on non-divisible grids (23, 257) across
1/2/4 devices; center/outer collective-free, misaligned inner one
`collective-permute` (Option A). An adversarial review found no
negotiate-reachable correctness defect; its three findings (a
device-count-rigid golden test, a missing `sync` golden, and an
unreachable hand-built empty-wall-shard edge — now a loud
`patch_physical_ends` guard) were fixed. Decomposition suite 206 passed
(forced-4); `ruff` clean. Follow-on (separate): the decomposed/gather-free
IO write (ROADMAP 2.6) and the distributed-transform reconciliation that
depends on this capability.

Generalize the decomposition layer so a **ghost-sharded factor axis
whose cell count `n_cells` does not divide the device count `P`** shards
by padding to a uniform per-shard storage block, instead of being
rejected. Today `TensorDecomposition._cells_per_shard`
(`tensor.py:317`) raises unless `n_cells % P == 0`; the only uneven case
handled is the staggered pair (n vs n±1) on top of an evenly-divisible
cell count. This plan removes the divisibility requirement itself.

This is the decomposition-layer capability the distributed-transform
reconciliation ([`distributed_transform_reconciliation.md`](distributed_transform_reconciliation.md))
and the ≥4-GPU / 1024³ pencil work depend on. It is **purely the
decomposition layer**: 1-D device mesh only, no transform changes (see
[Non-goals](#non-goals)).

Owning specs: [`../../specs/grid/04_decomposition.md`](../../specs/grid/04_decomposition.md)
§5 (staggered pairs / padding as the sanctioned, decomposition-owned,
invisible-above mitigation) and
[`../../specs/grid/classes/decomposition.md`](../../specs/grid/classes/decomposition.md).
Memory: [[uneven-shard-reblock-collective]].

---

## 1. Design: generalize the stagger mechanism to arbitrary surplus

The spec already sanctions **padding to a uniform storage shape** as the
decomposition-owned, invisible-above-the-layer mitigation for staggered
pairs (04 §5). The existing implementation realizes it as: cell-aligned
blocks of capacity `cells + 1` slots, with the **last shard absorbing
the surplus/deficit** (`_block_bounds`, `tensor.py:328`). The whole plan
is to make `cells` the **ceiling** rather than the exact quotient, so the
same "last shard absorbs the remainder" machinery covers arbitrary
non-divisible extents. **One uniform blocking story, not a parallel
one** — exactly the spec's intent.

### 1.1 Blocking (the one change that unlocks everything)

For a blocked axis over `P` shards define

```
cells = ceil(n_cells / P)          # was: n_cells // P, required to divide
block = cells + 1 + 2 * width      # UNCHANGED (the +1 stagger capacity)
storage(axis) = P * block          # evenly divisible by P — always
```

`_block_bounds(n, shards, cells)` (`tensor.py:328`) is **already correct
for this**: `bounds[s] = min(s*cells, n)`, `bounds[P] = n`. With ceil
cells:

| space | n | per-shard true counts (n_cells=257, P=4, cells=65) |
|---|---|---|
| center / cell_avg | n_cells = 257 | 65, 65, 65, **62** |
| outer | n_cells+1 = 258 | 65, 65, 65, **63** |
| inner / face_avg | n_cells−1 = 256 | 65, 65, 65, **61** |

Only the **last** shard differs from `cells` — the exact shape the
staggered mechanism already assumes. So the whole per-shard-count story
(`_block_bounds`, `_scatter_axis`, `_gather_axis`, `_exchange_block`'s
`t = where(s == last, n − (P−1)·cells, cells)`, `patch_physical_ends`'s
`t_in`/`t_out`) generalizes **with no change to those methods** — they
already parametrize on `cells` and the bounds.

Verified: the true-DOF values round-trip correctly through
`pad`/`unpad`/`gather` for center, outer, and inner on a 257-over-4 grid
(scratchpad probe reproducing the exact scatter/gather bodies).

### 1.2 Mild vs heavy padding (the one new invariant)

The "only the last shard is short" property holds **iff**

```
(P − 1) · cells < n_cells        # "mild padding": last shard holds ≥ 1 cell
```

When it fails ("heavy padding" — tiny grids sharded too finely, e.g.
`n_cells=5` over `P=4` gives `cells=2`, `(P−1)·cells = 6 ≥ 5`), trailing
shards go **empty or negative**, and the last-shard-only machinery breaks
(`t = n − (P−1)·cells` goes negative). Heavy padding is **rejected**:

- at negotiation (§Stage 3, `_shardable_names`), so it never reaches the
  backend from a real grid;
- defensively in the blocking helper (§Stage 1), so a hand-constructed
  `TensorDecomposition` fails loudly (replacing today's "must divide"
  guard, which was the same defensive role).

Divisible extents are always mild (`(P−1)·(n_cells/P) = n_cells − cells <
n_cells`), so nothing changes for them.

### 1.3 The fast `shard_map` path — padded-true frame

`_build_local_reblock` (`tensor.py:479`) currently returns `None` for any
blocked axis with `n != P·cells`, dropping `pad`/`unpad` to the global
per-block re-assembly (which the SPMD partitioner lowers to all-to-alls).
Extend it to the non-divisible case via a **padded-true** intermediate:

- the storage array `(P·block,)` is **always evenly sharded** (block per
  shard) — no divisibility problem there;
- the fast reblock operates on a **padded-true** array of length
  `P·cells` (evenly divisible), whose last shard's `cells − last` trailing
  slots are **inert** (like ghost/stagger slots — never addressed);
- the per-shard `shard_map` bodies are **identical to today's**
  (`scatter_local(piece) = jnp.pad(piece, ((width, block−width−cells),))`,
  `gather_local(block) = block[width : width+cells]`) — uniform, because
  every shard carries `cells` padded-true slots;
- the true↔padded-true conversion is a **trailing trim outside the
  shard_map**: `pad` does `jnp.pad(true, (0, P·cells − n))` before the
  scatter; `unpad` does `[:n]` after the gather.

Verified collective-free (4 forced devices, compiled-HLO inspection of
the `pad(unpad(storage))` round trip) for **center and outer** on a
257-over-4 grid — zero all-to-all / all-gather / all-reduce /
collective-permute.

### 1.4 The one residual: misaligned `n = n_cells−1` spaces

The trailing trim is collective-free **iff** jax's canonical sharding of
the true `(n,)` array (which uses `ceil(n/P)` blocks) matches the
storage's `cells = ceil(n_cells/P)` blocks. This holds for **every
center (state-field) and outer space always**. It fails for exactly one
family: `n = n_cells − 1` spaces (`inner`, `face_avg`) when
**`n_cells ≡ 1 (mod P)`** (e.g. n_cells=257 → inner=256, jax shards
256 as 64-blocks ≠ storage 65-blocks). There the trim rebalances via a
single **`collective-permute`** (a neighbour shift — **not** the
all-to-all pathology, **not** all-gather/all-reduce). jax cannot
block-align an indivisible true array with the cell-aligned storage
blocks (a `NamedSharding` on an `Auto` mesh even refuses indivisible
sizes outright), so this residual is intrinsic, not an implementation
miss.

**Settled: Option A** (owner, 2026-07-12; see [§6](#6-decision-settled-option-a)).
Keep these spaces on the fast path (one collective-permute, strictly
cheaper than the global all-to-all fallback); scope the collective-free
acceptance gate to center + outer (the state-field / perf-critical
spaces). The residual is **effectively test/edge-only**: `inner`/
`face_avg` are intermediate reconstruction / double-diff codomains, the
hot loop keeps them as storage arrays under `shard_map`, and reductions
over them go through the operator dispatch on the **storage frame**
(clean all-reduce, no permute — §1.5). The single collective-permute
appears only when a *misaligned-inner* true-shape array is materialized
as a distributed value — the synthetic `pad(unpad(storage))` round trip,
or an explicit `.data` used element-wise — neither of which the stepping
or reduction paths do.

### 1.5 True-shape materialization on a non-divisible grid (verified)

Because jax cannot shard an indivisible-length axis (`NamedSharding`
raises `IndivisibleError`; a forced indivisible distributed array is
**replicated**, not unevenly sharded — measured), a **true-shape `(n,)`
array of a non-divisible grid is never a genuinely sharded array**. This
sounds alarming but is inert in practice, because nothing on the hot or
reduction paths materializes it:

- **Stepping loop**: operates entirely on the **storage frame**
  (`ScalarField._data`, evenly `P·block`-sharded); same-space arithmetic
  uses the storage seam (`scalar_field.py` `_binary`), operators run
  `shard_map` on storage, `store` is pad-only. No true-shape array is
  built.
- **Reductions / diagnostics** (`sum`, `mean`, `integrate`, `has_nan`):
  **do not gather**. Verified HLO — `isnan(f.data).any()` and
  `sum(f.data)` fuse to a **shard-local reduce + a scalar all-reduce**;
  the replicated array is never materialized. `integrate`/`mean` reduce
  on the storage frame through the operator dispatch, so they are clean
  `all-reduce` even for misaligned `inner`.
- **IO / `gather`**: replicates the true field to host — but that is the
  **pre-existing iteration-1 IO contract** (ROADMAP 2.6; the sink gathers
  to rank 0, with decomposed-slice writes designed-for), **independent of
  this change**. A non-divisible grid gathers exactly like a divisible
  one. The scaling fix (shard-wise TensorStore write) is tracked there,
  not here.
- **Only** an explicit `f.data` used as an element-wise value all-gathers
  (replicates) — an intentional "give me the whole array" request, at a
  boundary, never in the loop.

So the plan's collective-free guarantee is about the **storage↔storage**
path (what runs hot); true-shape materialization is a boundary concern
that jax forces to replicate and this change neither adds to nor worsens.

---

## 2. What does NOT change (the no-op surface to protect)

- **Divisible extents**: `cells = n_cells // P` exactly; the new blocking
  branch (`n_cells % P != 0`) is never entered; the fast-path predicate
  stays `n == P·cells` for center and `None` (global fallback) for
  divisible outer/inner — **byte-for-byte identical**. The 512³/256³
  multi-GPU slab-FFT benchmark and every single-device program are
  untouched. This is the §4-constraints no-op the distributed-transform
  reconciliation depends on.
- **Single device** (`P = 1`): `shards == 1` everywhere; no blocking,
  `padded == n`, trims are no-ops. Unchanged.
- `sync` / `_exchange_block` / `patch_physical_ends` / `_block_bounds`
  method bodies: **unchanged** (they already parametrize on `cells`).
- `sharding` / `local_slice`: the storage is always evenly block-sharded,
  and `local_slice` stays the global true extent `slice(0, n)` — so the
  random draw and per-DOF keying remain device-count invariant by
  construction (04 §5).

---

## 3. Staged implementation

Each stage is independently testable and mergeable behind the
[Merge gate](#5-merge-gate). Stages 1–2 are unit-testable via **direct
`TensorDecomposition` construction** (as `test_indivisible_blocking_is_
rejected_at_use` already does), because negotiation only starts producing
non-divisible layouts at Stage 3.

Follow the AGENTS.md git workflow: one short-lived branch
`feat/uneven-shard-padding` + worktree; land with `git merge --no-ff`
onto `dev`; delete branch and worktree in the same session.

### Stage 0 — Capture the no-op golden (before any edit)

Prove-don't-assert scaffolding, committed first so the baseline is the
pre-change tip.

- **Change**: add `test_divisible_path_is_byte_for_byte_unchanged` to
  `tests/spatial/decomposition/test_multi_device.py`: on a **divisible**
  grid (16 over 4), lower+compile `pad`, `unpad`, `sync`, `zeros`, and
  the `pad(unpad(storage))` round trip; assert the compiled HLO text
  equals a **golden string checked in from the current tip** (captured
  now, before Stage 1). Also assert exact gathered-value equality vs a
  1-device grid (already covered, re-anchored here).
- **Test**: the new test, run on the current code — it must pass as a
  tautology now, then **keep passing unchanged after every later stage**.
- **Gate**: golden captured; test green on `HEAD`.

### Stage 1 — Blocking generalization + global path (correctness)

- **Change** (`tensor.py`):
  - `_cells_per_shard`: return `-(-n_cells // P)` (ceil). Replace the
    `n_cells % P` guard with the **heavy-padding** guard: raise (message
    naming padding, not divisibility) when
    `(P − 1) · cells >= n_cells` or `n_cells is None`.
  - Nothing else: `_axis_storage`, `_block_bounds`, `_scatter_axis`,
    `_gather_axis`, `_exchange_block`, `patch_physical_ends`, `pad`/
    `unpad` global branches inherit ceil-cells unchanged.
  - `_build_local_reblock` still returns `None` for `n != P·cells`, so
    **non-divisible axes use the global (collective) fallback in this
    stage** — correct but not yet fast. Fast path is Stage 2.
- **Tests** (`test_multi_device.py`, direct construction over
  `jax.device_count()`):
  - non-divisible round trip `pad`→`gather`/`unpad` for center, outer,
    inner on n_cells=257 (or `4·k+1` for forced-4) — **device-count
    invariant** (bitwise vs a 1-device decomp), values correct.
  - `storage_shape` arithmetic: `P·(cells+1+2·width)` on a non-div axis.
  - `sync` on a non-div sharded axis matches the 1-device halo fill
    (extend `test_blocked_ghosts_match_the_single_device_fill` with a
    non-divisible mesh).
  - heavy padding (n_cells=5 over 4) raises (repoint/rename
    `test_indivisible_blocking_is_rejected_at_use` → heavy-padding).
- **Gate**: non-div correctness + invariance through the global path;
  Stage 0 golden still green (divisible untouched); mirrored tests +
  ruff.

### Stage 2 — Fast `shard_map` path for padded-even (performance)

- **Change** (`tensor.py`):
  - Extend `_build_local_reblock` so a blocked axis takes the fast path
    when `n == P·cells` (existing divisible-center) **or**
    `n_cells % P != 0` (new padded-even). For the padded-even case record
    the per-axis trailing trim (`pre_pad = P·cells − n`, `post_slice =
    slice(0, n)`).
  - Extend `_ReblockPlan` so `scatter`/`gather` apply the trim **around**
    the (unchanged) jit-wrapped `shard_map` callables:
    `scatter(true) = sm_scatter(jnp.pad(true, pre_pad))`,
    `gather(storage) = sm_gather(storage)[post_slice]`. **When there is
    no cell-padding (`pre_pad == 0` on every axis), build the exact
    current callables** (no wrapping op) so the divisible HLO is
    bit-identical.
- **Tests** (`test_multi_device.py`, direct construction):
  - `test_padded_reblocking_compiles_without_collectives`: on a
    non-divisible grid, `pad(unpad(storage))` for **center and outer**
    compiles with no all-to-all / all-gather / all-reduce /
    collective-permute (the §1.3 guarantee).
  - `_local_reblock` returns a plan (not `None`) for non-div center /
    outer; assert `pad_widths` / `true_slices` / trim fields.
  - warm re-run adds **0 compiles** (extend
    `test_warm_eager_reblocking_adds_zero_compiles` to a non-div grid) —
    the trim closures must be built once and jit-wrapped.
  - `inner`/`face_avg` at `n_cells ≡ 1 mod P` (Option A, §6): assert the
    fast plan **exists** (not `None`) and that a **storage-frame
    reduction** (`sum`/`any`) over such a field compiles to a clean
    `all-reduce` with **no all-gather** (§1.5). The synthetic
    `pad(unpad(storage))` round trip's single `collective-permute` is
    documented as expected for these spaces — asserted as `≤ 1
    collective-permute, no all-to-all/all-gather`, not as collective-free
    (that gate is center + outer only).
- **Gate**: padded-even round trip collective-free (center/outer); 0
  recompiles; Stage 0 golden still green; mirrored tests + ruff.

### Stage 3 — Negotiation admits mild non-divisible axes

- **Change** (`decomposition.py`):
  - `_shardable_names` (`:783`): drop the `n_cells % devices` skip;
    compute `cells = ceil(n_cells/devices)`,
    `last = n_cells − (devices−1)·cells`; a name qualifies when
    `last >= max(min_local, width + 1)` (the shortest — last — shard must
    still hold a full exchange edge + BC-fill depth). This subsumes the
    mild-padding invariant (`last >= 1`) and so **auto-rejects heavy
    padding**. For divisible axes `last = cells`, so the check is
    identical to today.
  - `_cap_for_sharding` (`:728`): drop the `n_cells % devices` skip;
    `cap = last − 1` with the same `cells`/`last`. For divisible axes
    `last = cells`, so `cap = cells − 1` — identical to today.
- **Tests** (`test_negotiate.py`): a grid with a non-divisible cell count
  negotiates a sharded default layout (was: fell back to 1 device /
  raised on explicit ids); a heavy-padding grid still falls back / raises;
  divisible negotiation unchanged; the width-cap path fires correctly on
  a non-divisible axis.
- **Gate**: end-to-end `negotiate` produces the non-div sharded layout;
  divisible negotiation byte-identical; mirrored tests + ruff.

### Stage 4 — End-to-end invariance + suite + no-op proof

- **Change**: extend `test_multi_device.py`'s device-count-invariance
  fixture to also build a **non-divisible** grid (e.g. `IntervalMesh(257,
  …)` or a size that is `4·k+1` for the forced-4 suite) and run the
  existing invariance battery on it (create/init, random.normal, diff
  chains periodic+bounded, staggered-pair diffs, interpolation/products,
  reductions, reshard round trip). Add the multi-device blocked-storage
  assertion for the non-div case.
- **Tests**: the full
  `tests/spatial/decomposition/test_multi_device.py` under the forced-4
  suite; one model smoke (`tests/nonhydro/test_linear_model.py`, since
  decomposition is core machinery).
- **Gate (the no-op proof)**:
  1. **Structural**: the non-div branch is gated on `n_cells % P != 0`;
     divisible extents execute the pre-change path.
  2. **Empirical HLO**: Stage 0's golden test is green on the final tip
     (divisible `pad`/`unpad`/`sync`/`zeros`/round-trip HLO
     byte-identical to the pre-change capture).
  3. **Empirical values**: the existing bitwise divisible gates
     (`test_tensor.py`, `test_multi_device.py`) pass unchanged.
  4. **Device-count invariance**: non-div 1-vs-4 bitwise (≤1e-12) via the
     extended invariance battery.
  5. **CI**: full multi-device suite green; ruff-clean.

---

## 4. Test matrix (per §3, consolidated)

| property | test | stage |
|---|---|---|
| divisible HLO byte-for-byte unchanged | golden HLO gate | 0, re-checked 4 |
| divisible values unchanged | existing bitwise gates | 0/4 |
| non-div blocking arithmetic | `storage_shape` on 257 | 1 |
| non-div correctness + 1v4 invariance (global path) | round-trip + battery | 1 |
| non-div sync == 1-device fill | extended blocked-ghosts test | 1 |
| heavy padding rejected (direct + negotiation) | renamed reject test + negotiate | 1, 3 |
| padded-even fast path collective-free (center/outer) | new collectives test | 2 |
| padded-even 0 recompiles | extended warm-reblock test | 2 |
| misaligned inner/face_avg residual pinned | new test (per §6 decision) | 2 |
| negotiation admits mild non-div | negotiate tests | 3 |
| non-div 1v4 invariance end-to-end | extended invariance battery | 4 |
| full multi-device suite + model smoke + ruff | CI | 4 |

---

## 5. Merge gate

Per AGENTS.md: mirrored tests for every edited source file
(`tests/spatial/decomposition/test_tensor.py`,
`test_multi_device.py`, `test_negotiate.py`) + the forced-4 multi-device
suite + `tests/nonhydro/test_linear_model.py` smoke +
`uv run ruff check src tests`, all green. The Stage 0 golden HLO test is
the binding no-op proof. Land `git merge --no-ff feat/uneven-shard-padding`
onto `dev`; delete branch + worktree same session. No GitHub PR unless
Silvano asks.

---

## 6. Decision (settled): Option A

**How to handle the `n = n_cells − 1` spaces (`inner`, `face_avg`) at
`n_cells ≡ 1 (mod P)`**, whose trailing trim cannot block-align with the
cell storage (§1.4). **Chosen: Option A** (owner sign-off, 2026-07-12).

- **(A, chosen)** Keep them on the fast path; accept one
  `collective-permute` (ring shift) *only* when a misaligned-inner
  true-shape array is materialized as a distributed value (§1.5 shows the
  loop and reductions never do this); scope the collective-free
  acceptance gate to center + outer. Strictly cheaper than the global
  all-to-all fallback, and the residual is test/edge-only. Where a test
  would otherwise trip on it, the test reduces on the storage frame
  (clean `all-reduce`) rather than round-tripping through the true shape.
- **(B, rejected)** Route these spaces to the global fallback: simpler
  predicate, but *worse* HLO (all-to-all > collective-permute) —
  contradicts "must not drop to the collective path for the uneven case."

Under A, center (the 512³-class state field) and outer are fully
collective-free, correctness / device-count invariance are identical, and
the misaligned-inner permute is pinned (not gated) by the Stage-2 test.

---

## Non-goals

Explicitly out of scope (each a separate context):

- The **2-D device mesh / pencil** decomposition — `_build_device_mesh`
  stays 1-D (`tensor.py:169`).
- Any **Fourier / transform** change; the distributed-transform
  reconciliation ([`distributed_transform_reconciliation.md`](distributed_transform_reconciliation.md))
  is planned in parallel and *depends on* this capability — this plan
  only guarantees the §4-constraints no-op it protects.
- **Inhomogeneous ghost fill**, Robin data paths, graph decomposition —
  untouched.
