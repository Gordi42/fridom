---
status: active
date: 2026-07-12
---

# Distributed transform reconciliation — slab FFT vs the §5.1 layout design

The multi-GPU pressure-solve speedup landed on `dev`
(`2f324783 Merge perf/distributed-slab-fft`) as a **deviation from the
agreed decomposition design**. This note records the deviation, what
must be preserved when reconciling it, and the open questions the
reconciliation plan must answer. **It gates the push**: `dev` is not to
be pushed until the merged distributed solve is reconciled with the
design (owner decision, 2026-07-12).

This file is *findings + constraints* — the seed for the plan, not the
plan. The plan (how exactly it works, staged, with acceptance gates) is
authored next, in its own context.

## 1. The deviation

The design (`specs/grid/04_decomposition.md` §5.1 "Layout is part of the
function space", and `specs/grid/classes/operators_transforms.md`)
specifies distributed FFT as an **emergent property of the ordinary
`Fourier` transform**:

- `Layout` is a defining attribute of the function space; fields live on
  laid-out spaces; the join requires layout equality (no implicit
  reshard in field arithmetic, ever).
- `Reshard` and `Sync` are first-class operators; a `Reshard`'s kernel
  is a transpose/`redistribute`.
- **Requirements-driven lowering inserts `Reshard` stages into the
  transform plan** (`forward_plan`/`backward_plan`), chosen by shortest
  path over the negotiated layout graph; the layout is carried through
  each spectral pencil in the coefficient space's *type*. The codomain
  is the schedule's final pencil.
- Consequence, stated normatively (`operators_transforms.md` line ~220,
  `04_decomposition.md` §5): the transform API is rich enough that
  **"solvers no longer bypass it" — the explicit `RFFTPressureSolver`
  lesson.**

What landed instead (`src/fridom/spatial/operators/slab_fft.py`, wired
into `SpectralSolve`):

- A **solve-scoped, bespoke** pipeline: `SlabPlan`/`SlabSolve` resolved
  only inside `SpectralSolve` (the pressure solve). A standalone
  `field.fft()` on several devices still replicates / falls back.
- The whole forward → spectral divide → backward runs as **raw
  `jax.shard_map` bodies with one `jax.lax.all_to_all`**, hand-written —
  the `Reshard` operator and the transform planner's multi-device stages
  are **not** used.
- The intermediate spectral pencils are **not** laid-out
  `FunctionSpace`s; they exist only as `PartitionSpec`s inside the
  shard_map body plus a private `_internal_coeff` space the plan owns.
- It *is* compatible with layout-in-the-space **at the field boundary**:
  it reads the grid's negotiated `decomposition.default_layout` /
  `device_mesh` / `device_count`, reuses the single-device
  `transform.forward_plan(bare).stages` for axis order, and is
  sharding-neutral (input layout == output layout). It just does not use
  the design's type system or lowering **internally**.

So it is precisely the solver-bypass §5 forbids, delivering the win
outside the algebra.

## 2. Why it diverged (the real blocker)

The transform's `forward` delivers each 1D stage's output through
`store` (`spatial/fields/storage.py`), which materializes on the
coefficient space's **device-local** storage (iteration-1 storage
contract). Running the ordinary `Fourier.forward` distributed would
therefore **re-replicate at every intermediate `store`**, defeating the
distribution. The design's answer is layout-aware coefficient spaces +
`store`/`_deliver` that respect a pencil layout; that multi-device
planner is the part `operators_transforms.md`/`transform.py` explicitly
defer ("the multi-device planner slots into
`forward_plan`/`backward_plan` later"). The subagent, rather than build
that, realized the whole pipeline **one level up inside a single
shard_map** so nothing stores between stages.

## 3. What already exists vs what is missing

**Built** (the §5.1 scaffolding is real): `Layout`
(`decomposition/layout.py`), `space.with_layout`/`.bare`/`.layout`,
`Reshard`/`Sync` operators (`operators/movement.py`), `field.reshard()`,
the layout-equality join with `SpaceMismatchError`, the negotiated
layout graph (`decomposition/graph.py`), and the **single-device**
transform planner (grid-coordinate order, zero reshard stages).

**Missing** (the gap the slab module papers over):

1. the **multi-device transform planner** — `forward_plan`/
   `backward_plan` inserting `Reshard` stages over the negotiated layout
   graph;
2. **layout-carrying coefficient spaces** through each pencil (the
   intermediate typed spaces);
3. a **`store`/`_deliver` path that respects a pencil layout** instead
   of re-replicating to device-local storage — the actual blocker;
4. a lowering that compiles the multi-stage plan into an **efficient
   program** (see §5, the central risk).

## 4. Hard constraints — what reconciliation MUST NOT regress

These are measured on the merged `dev` tip (A100, matched nonhydro
config; the slab is the multi-GPU path). The reconciled implementation
must match them within noise, proven by re-measurement **and HLO diff**:

| property | current value (must hold) |
|---|---|
| 4-GPU 512³ linear | 20.97 ms/step (was 37.5 replicated) |
| 4-GPU 512³ advective | 27.75 ms/step (was 44.6) |
| 4-GPU 256³ linear | 3.54 ms/step |
| single-device program | **byte-for-byte unchanged** (fallback) |
| distributed vs replicated solve | rel ≤ 1e-11 (measured 6.5e-16) |
| 4-dev vs 1-dev, 20 steps | drift ≤ 1e-11 (measured 2.2e-12) |
| compile stability | **0 recompiles** across warm `advance` on 4 dev |
| HLO | contains `all-to-all`, **no `all-gather`/`all-reduce`** |
| largest grid on 4×A100 | 768³ fits (43.7 GiB/dev); 1024³ OOMs |
| multi-GPU flag | still runs under
  `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`
  ([[xla-gpu-fusion-bug]], jax#39100) |

Also preserve the full multi-device gate (1241 tests) and ruff-clean.

## 5. The central technical risk (and the question the plan must answer)

The slab pipeline is fast for **one specific reason**: the entire
forward → divide → backward is **one `shard_map` region with a single
`all_to_all` and no intermediate `store`/materialization**. The naïve
"design-correct" lowering — each transform stage a separate operator
with a `Reshard` (and a `store`) between them — risks either (a)
re-replicating at each `store` (the very bug), or (b) emitting a
separate shard_map / materialized intermediate per stage (extra memory,
lost fusion, the 768³ ceiling regresses).

**So the reconciliation is not "delete slab_fft and call `Reshard`."**
The plan must show how the lowered, layout-typed transform plan compiles
down to the *same* efficient program the hand-written slab emits —
ideally reusing the slab shard_map kernels as the **lowering target**
(the thing `Reshard`+stage nodes lower into), not a parallel path. The
acceptance test is an **HLO diff**: reconciled vs current slab, same
collective (`all-to-all`) count, no `all-gather`, equal per-device peak.

## 6. Open questions for the plan

1. **Lowering granularity.** Does the whole transform plan (stages +
   `Reshard`s) lower to **one** `shard_map`, or per-stage regions XLA
   then fuses? What guarantees the fusion / no intermediate replication?
2. **`store`/`_deliver` under a pencil layout.** How does per-stage
   delivery avoid re-replication — a layout-aware `store`, or does the
   distributed plan bypass per-stage `store` by construction (deliver
   only at the final pencil)?
3. **Typed intermediate pencils.** Are the intermediate coefficient
   spaces real laid-out `FunctionSpace`s (§5.1), and does interning /
   dispatch cost stay bounded? Half-spectrum-factor determinism under
   the schedule (the real-FFT stage order) must be preserved.
4. **Reuse vs rebuild.** Can `slab_fft.py`'s shard_map kernels become
   the `Reshard`/stage lowering target (lowest perf risk), or must they
   be rederived? What is retired vs kept?
5. **Trace stability.** The design's plan objects must be built once and
   memoized like the current `SlabPlan` (the 553ad876 idiom) — where
   does the cache live so warm `advance` stays at 0 recompiles?
6. **Scope of the first cut.** 1-D slab only (match today), or build the
   general N-D pencil path (§5.1 closed-vocabulary graph) now — since
   pencil is separately wanted for 1024³+ / >4 GPUs? Recommend slab-first
   parity, pencil as a follow-on, to de-risk the perf-preservation gate.
7. **Generalization dividend.** Once distribution lives in the
   transform, which other consumers (spectral filters, diagnostics,
   `field.fft()`) become distributed for free, and are any newly
   exercised paths a correctness risk?

## 7. Reconciliation direction (input, not the decision)

Preferred shape to evaluate in the plan: implement the **multi-device
transform planner** (§5.1) so `Fourier.forward_plan`/`backward_plan`
carry `Reshard` stages and layout-typed pencils; make the plan lower to
the existing slab shard_map kernels (kept as the lowering target, not a
separate solve path); make `SpectralSolve` obtain distribution by
composing the ordinary transform (no `SlabSolve` special case), so the
"solvers don't bypass the transform" rule holds. Prove
performance-neutrality by HLO diff + re-benchmark against §4 before
retiring the `SlabSolve` dispatch. Keep 1-D slab as the first
parity-locked cut; N-D pencil is a scoped follow-on.

Related: [`phase2_grid_followups.md`](phase2_grid_followups.md),
`specs/grid/04_decomposition.md` §5.1,
`specs/grid/classes/operators_transforms.md`. Memory:
[[new-stack-gpu-performance]].
