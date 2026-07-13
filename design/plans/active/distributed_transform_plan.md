---
status: active
date: 2026-07-12
---

# Distributed transform reconciliation — staged plan

Companion to the findings/constraints doc
[`distributed_transform_reconciliation.md`](distributed_transform_reconciliation.md)
(the deviation, the §4 hard constraints, the §6 open questions, and the
§8 multi-host record). **This file is the plan** that answers §6 and
gates the push. Target decided (owner, 2026-07-12): distributed
**transform**, `SpectralSolve` obtains distribution by composing the
ordinary transform, and the solve-scoped `SlabSolve` is retired.

## 0. Decisions locked

- **Distribution lives in the transform, not the solve.** `Fourier.forward`
  / `backward` distribute on a sharded operand; `SpectralSolve` =
  `backward @ inverse @ forward` with **no** `SlabSolve` special case
  (`spectral_solve.py:296–325,424` deleted). Satisfies the §5 "solvers
  no longer bypass the transform" rule.
- **Lowering shape: C** (typed plan → per-stage `shard_map` kernels).
  The typed layout-annotated plan is the *spec*; a multi-device lowering
  compiles it to `shard_map` kernels with `lax.all_to_all` at reshard
  nodes. The **all-Fourier instance reuses the existing slab kernel
  bodies** (`slab_fft.py` `_forward_local`/`_backward_local`) as the
  lowering target — no throwaway parallel path. Chosen over a monolithic
  slab kernel (A) because only C extends to mixed transforms
  (`Fourier(x,y) ⊗ Chebyshev(z)`), and over pure-jax composition (B)
  because B is not viable (see §2).
- **1-D slab (single-node) is this plan's scope.** N-D pencil (2-D mesh)
  and multi-host are follow-ons (§5). `decomposition/graph.py` stays a
  stub; the flat `layouts` vocabulary + `layout_for` suffice.
- **Uneven-shard padding is a landed dependency** (`6bf2c032`): any factor
  axis — nodal or coefficient — shards over any device count, so pencil
  delivery works for arbitrary extents and slab-first stays on the clean
  divisible fast path.

## 1. Design (mechanics)

Running example throughout: real 3-D field, coordinates x/y/z, operand
on the negotiated nodal default layout **sharded on z**; the planner
puts the rfft on **x** (x carries the Hermitian half spectrum, stays
local), transpose partner **y**. Codomain: `Fourier(x,half) ⊗
Fourier(y,full) ⊗ Fourier(z,full)`, sharded on **y**.

### 1.1 The typed multi-device plan (§6 Q1)

Minimal data-structure change — annotate the existing stage with the
layout it runs under; let the plan's codomain carry the final pencil:

```python
@dataclass(frozen=True)
class TransformStage:            # transform.py:197
    axis, index, half, nodal, coeff
    layout: Layout | None = None     # NEW: the layout this stage runs under

@dataclass(frozen=True)
class TransformPlan:             # transform.py:234
    domain, codomain, stages         # codomain laid-out on multi-device
```

Single device: every `layout` is `None` → the `forward` loop and the
emitted program are **byte-for-byte unchanged**. Reshards are *derived*
from layout changes between consecutive stages (no separate node type).

The multi-device planner fires in `forward_plan` when `domain.layout`
shards a coordinate and `device_count > 1`. **The plan is keyed on the
operand's layout**, not just the bare space (a z-pencil operand and an
x-pencil operand yield different reshards); the single-device key stays
bare (`transform.py:528`).

1. **Stage order** — initially-*local* axes first, initially-*sharded*
   axis last: the rfft lands on a local axis and the half-spectrum axis
   stays local (divisibility). Deterministic (grid-coordinate order
   within each group).
2. **Partner `b`** — the first full-spectrum stage axis ≠ `a` that is
   shardable, exactly `slab_fft._slab_geometry` (`slab_fft.py:578`).
   Target `Layout({b: "devices"})`, constructed directly and checked
   against the negotiated vocabulary (member iff `b` shardable).
3. **Layout assignment** — the initially-local block (x-rfft, y-fft) runs
   under `Layout({z})`; after the reshard the z-stage runs under
   `Layout({y})`. Codomain layout = the last stage's layout = `Layout({y})`.
   Nothing reshards back to the nodal layout.
4. **Reshard lowering params** are derived from `(src, dst)`: on a 1-D
   mesh, `concat_axis` = array index of the name sharded in `src` but
   local in `dst` (z), `split_axis` = index of the name sharded in `dst`
   but local in `src` (y) — precisely `all_to_all(split_axis=b,
   concat_axis=a)` (`slab_fft.py:235`). The node carries only
   `(src_layout, dst_layout)`.

`backward_plan` mirrors: reversed stages, reversed reshards, ending on
`Layout({z})` (the solve round-trip returns to the operand's nodal
pencil).

### 1.2 Typed intermediate pencil spaces (§6 Q3)

At each block boundary the data lives on a **partial-transform space**:
axes transformed so far are coefficient factors (half/full rule:
`mesh.fourier(origin=factor)` for the half stage, `origin=factor.as_complex()`
later — `slab_fft._internal_coeff:592`), the rest still nodal, carrying
that block's layout:

```
after block 1:  Fourier(x,half) ⊗ Fourier(y,full) ⊗ Nodal(z)   layout {z}
codomain:       Fourier(x,half) ⊗ Fourier(y,full) ⊗ Fourier(z) layout {y}
```

These are the incremental states of `forward_plan`'s existing `mapping`
accumulation (`transform.py:535–546`) plus `.with_layout(block.layout)`.
They are **real interned laid-out `FunctionSpace`s** (memoized per plan,
bounded interning), so the plan, the `codomain`, and `field.fft()` are
honestly typed. Whether a runtime **field** materializes at a boundary
is a *lowering* choice: inside a fused `shard_map` span the boundary is
a `lax.all_to_all` and the partial space is a **type only**; where the
lowering breaks a span (separate regions, the forward→divide→backward
seam, or a user's explicit `reshard`) a real field materializes on it.

### 1.3 Delivery — coeff fields are storage-replicated (revised 2026-07-12)

**Correction (recon, 2026-07-12).** The original §1.3 had `_deliver`
stamp the pencil and route the coeff output through `store`. That is
**wrong**: the storage layer *actively replicates* coefficient factors —
`_n_shards` returns 1 and `sharding` emits `None` for any
`CoefficientSpace`, **regardless of layout** (`tensor.py:312,592`). So
delivering a `shard_map`'s y-sharded coeff output through `store` would
reshard it to *replicated* — an `all-gather`, the very thing we remove.
Making coeff factors genuinely shard is **new decomposition machinery**
("**Path X**": lift those guards and replace the nodal-cell block formula
`tensor.py:336–342,422–426` with the Fourier extent) — not a validation
item.

Consequence: the distributed **solve materializes no coeff field at
all.** The fused lowering (1.4) runs forward → divide → backward in one
`shard_map` and returns the nodal result via `with_data` (slab-style,
`slab_fft.py:431`) — no `store` on a coeff space, no gather. A
*standalone* distributed `forward` (which must return a coeff field) is
the `field.fft()` follow-on that needs Path X (§5); it is **not** on the
parity path.

### 1.4 The lowering (revised 2026-07-12)

The distributed **solve** lowers to a **single** `jax.shard_map` region =
the slab's `_build_solve` body (`slab_fft.py:341`: local FFTs +
`lax.all_to_all` + the `in_specs`-sliced diagonal multiply + the mirrored
backward), **driven by the planner's geometry (1.1)** instead of slab's
`_slab_geometry`/`_internal_coeff`. The reshards are the two `all_to_all`
inside the region; no coeff field is materialized (1.3). The kernel
bodies are lifted from `slab_fft.py` to a shared home so the slab module
retires. `Fourier.forward`/`backward` keep the single-device global
`rfftn`/`fftn` path unchanged. The *decomposed* form (per-stage regions,
materialized coeff fields, standalone `field.fft()`) is the Path-X
follow-on (§5) — collective-identical (§2 A3≡A5) but needing shardable
coeff storage.

### 1.5 `SpectralSolve` recomposition (§6 Q4; revised 2026-07-12)

Delete `_resolve_slab`, the `SlabSolve` imports, `self._slab`, and the
`__call__` slab branch; `SpectralSolve` builds the composite
`backward @ inverse @ forward` unconditionally.

**Diagonal handling (correction, recon 2026-07-12).** The original claim
— `eigenvalues` on a laid-out coeff space yields a y-sharded diagonal
*field* — is false: `Symbol.__init__` strips the layout to bare and holds
a plain **replicated** array (`symbol.py:88,748`); `eigenvalues` never
touches `store`. So the inverse diagonal is a replicated full cube. Two
ways to consume it:

- **Fused lowering (parity path, default):** slice the replicated
  diagonal per shard via the `shard_map` `in_specs`, exactly as the slab
  does (`slab_fft.py:337–348`). No coeff field, no per-device full
  diagonal — byte-for-byte the slab.
- **Three-region composition (follow-on):** broadcast the replicated
  diagonal against the sharded forward output. Correct, but every device
  holds the whole inverse diagonal (~1 GB/dev at 512³ — a memory
  regression vs the sliced slab) and it needs Path X for the coeff
  fields. Deferred.

Dividend (§6 Q7): `single_precision`'s `_CastMap` chain composes with the
distributed path once the three-region form lands (out of scope for
parity).

### 1.6 Trace stability / caching (§6 Q5)

Mirror the slab's proven idiom (`slab_fft.py:182,193`), relocated:

1. **Plans** — `forward_plan`/`backward_plan` already memoize in
   `self._plans` (`transform.py:352`); extend the key to include the
   operand layout. Built once.
2. **Per-stage `shard_map` kernels** — the cardinal rule: built **once
   and stashed** (stable identity → jax trace-cache hits; a per-call
   closure rebuild = guaranteed recompile). Home on the **Transform,
   keyed by plan** (the static grid-bound plan owner) — the `SlabPlan.__init__`
   idiom, relocated.
3. **Composite + inverse diagonal** — built once at `SpectralSolve`
   construction (already true).

Treedef stays stable: coeff fields are **halo-0** (`_halo_valid` trivial
→ no phase2-item-12 carry churn); the pencil layout is static in the
space identity; reshards lowered as `lax.all_to_all` never touch
`Reshard.__call__`/`halo_valid` (`movement.py:166`). Gate:
`test_warm_eager_solve_adds_zero_compiles` (`test_slab_fft.py:420`),
re-pointed at the composite.

### 1.7 Determinism + Hermitian projection (§6 Q3)

- **`_project` is not a hazard.** `hermitian_project` fires only when
  `flat_hermitian_applies` — exactly one complex-carrying factor
  (`storage.py:114`). The 3-D coeff space has three Fourier factors →
  **no-op**; the rfftn/staged kernels already emit a valid
  conjugate-paired spectrum. A 1-D real transform fires it as a **local
  per-shard mask** on the kept-local half axis — no collective.
- **Half-spectrum axis** = rfft axis = an initially-local axis, chosen
  statically from the negotiated default layout → deterministic per
  grid, stable across steps, reproducible. May differ from the
  single-device grid-first convention (so the multi-device coeff *type*
  can carry the half spectrum on a different axis) — design-sanctioned;
  the solve stays correct because eigenvalues are queried on the actual
  coeff space (the slab already depends on this, `slab_fft.py:206`).
  `backward_plan` reads the half axis unambiguously from the coeff space
  (the sole real-origin Fourier factor).

### 1.8 Multi-host / at-scale

Out of scope here; recorded in the findings doc §8. SPMD makes the
lowering topology-agnostic (runs across nodes unchanged); the gaps
(manual `jax.distributed.initialize`, sharded-array construction/IO,
topology-aware mesh mapping) are the decomposition/IO layer, scoped with
the pencil follow-on, gated by a ≥2-node smoke test.

## 2. Evidence base

GSPMD probe (jax 0.10.2, forced 4-device CPU; collective *structure* is
platform-independent), recorded in [[distributed-transform-gspmd-lowering]]:

- Plain `jnp.fft.fftn/rfftn` over **local** axes on a sharded array →
  **all-gather** (jnp.fft is monolithic; GSPMD replicates the cube). ⇒
  pure-jax composition (B) is not viable; manual `shard_map` mandatory.
- `shard_map` local FFT → **zero collectives**; a full forward as
  `shard_map` (rfft + fft + all_to_all + fft) → all-to-all, no gather —
  the slab kernel works standalone as a *forward*.
- **Per-stage/per-block `shard_map` regions compose with no extra
  collectives** vs one fused region (A3≡A5) ⇒ the lowering can be
  per-stage kernels, and the fused-span-vs-per-stage choice is a
  benchmark knob, not a correctness fork.
- A **non-FFT stage (matmul, DST/DCT/Chebyshev stand-in)** composes
  inside `shard_map` identically ⇒ C generalizes to mixed transforms.

## 3. Staging (land-order; revised 2026-07-12 after recon)

**STATUS (2026-07-12): all four stages landed** on
`feat/distributed-transform-planner`
(`702dc3ca`..`c942d35e`), CPU-verified (forced-4-device + full
operators dir 1009 passed + ruff + >=97% branch coverage on the two
touched modules) and A100-verified at Stage 3 (below). Implementation
nuance vs the plan below: Stage 2 **reused** the slab `SlabPlan`/`SlabSolve`
kernel driven by the planner geometry (rather than lifting it), and the
physical relocation of the kernel into `distributed_solve.py` + deletion
of `slab_fft.py` folded into Stage 4. Not yet merged to `dev`.

Each stage is one short-lived branch (`<type>/<topic>`), mirrored tests +
ruff green before merge (AGENTS.md). The slab path stays **live** until
Stage 4. **Revision:** the recon (§1.3/§1.5 corrections) collapsed the
old "make `Fourier.forward` distributed, then compose three regions" into
a single **fused distributed-solve** lowering — the parity path never
materializes a coeff field, so the coeff-field/`store` work (Path X) and
standalone `field.fft()` move to the follow-on.

**Stage 1 — planner (types). DONE (2026-07-12).**
`feat/distributed-transform-planner`. Added `TransformStage.layout` and
the multi-device `distributed_forward_plan`/`distributed_backward_plan`
(layout-annotated stages, codomain = final pencil, memoized; the
single-device `forward_plan`/`backward_plan` and slab's consumption of
them are untouched). *Verified:* `test_transform.py` — the planner's
`_distributed_geometry` and codomain coeff frame match the live slab
(`resolve_slab_plan`/`_slab_geometry`) byte-for-byte on forced-4-device;
single-device plans carry no layout; ruff clean.

**Stage 2 — fused distributed-solve lowering.**
`feat/distributed-solve-lowering`. Lift the slab kernel bodies
(`_forward_local`/`_backward_local`/`_build_solve`, `slab_fft.py:228–348`)
to a shared home and **drive them from the planner geometry (Stage 1)**
instead of `_slab_geometry`/`_internal_coeff`: one `jax.shard_map` region
= forward local FFTs + `all_to_all` + `in_specs`-sliced diagonal multiply
+ mirrored backward. Kernel built once and cached (1.6). Slab dispatch
**not** touched yet (live solve unchanged); the new lowering is exercised
by new tests only. *Verify:* the fused lowering's HLO == the baseline
(**exactly 2 `all-to-all`, no gather** — §2 golden ref); result ==
replicated composite ≤1e-11; 0 recompiles warm.

**Stage 3 — switch the solve (GATE).** `refactor/spectral-solve-compose`.
`SpectralSolve` builds `backward @ inverse @ forward` and lowers the
distributed case to the Stage-2 fused region (a composite lowering, not a
SlabSolve dispatch); delete `_resolve_slab`/`self._slab`/the `__call__`
branch. *Verify — the full §4 acceptance gate.* Only if every gate passes
does Stage 4 proceed. (The only per-step consumer is the pressure solve;
the eigenbasis/diagnostic single-axis consumers stay on the gather path —
no newly-exercised paths, per recon §2.)

**Stage 4 — retire slab.** `refactor/retire-slab-fft`. Delete
`slab_fft.py`'s `SlabSolve`/`SlabPlan`/`resolve_slab_plan`/`_slab_geometry`
(the kernel bodies now live in the shared home).
*Verify:* full multi-device gate (1241 tests), ruff, benchmark unchanged.

**Follow-on (separate plan) — Path X + decomposed transforms.** Teach the
storage layer to shard coefficient factors (`tensor.py:312,592,336–342,
422–426`); then standalone distributed `Fourier.forward`/`backward`
(materialized coeff fields), the three-region composition, `field.fft()`,
and the eigenbasis/diagnostic distribution dividend (recon §2) all follow;
plus N-D pencil (2-D mesh) and multi-host (findings §8). Not required for
the push.

## 4. Acceptance gates

Measured on the merged `dev` tip (A100, matched nonhydro config), proven
by **HLO diff + re-benchmark**, not tests alone:

| property | must hold |
|---|---|
| 4-GPU 512³ linear | 20.97 ms/step |
| 4-GPU 512³ advective | 27.75 ms/step |
| 4-GPU 256³ linear | 3.54 ms/step |
| single-device program | byte-for-byte unchanged |
| distributed vs replicated solve | rel ≤ 1e-11 (slab: 6.5e-16) |
| 4-dev vs 1-dev, 20 steps | drift ≤ 1e-11 (slab: 2.2e-12) |
| warm `advance`, 4 dev | **0 recompiles** |
| HLO | `all-to-all` present, **no `all-gather`/`all-reduce`** |
| largest grid on 4×A100 | 768³ fits (43.7 GiB/dev) |
| XLA flag | runs under `--xla_disable_hlo_passes=multi_output_fusion` |

Plus: full multi-device gate (1241 tests) + ruff clean. **HLO nuance:** a
`collective-permute` from the uneven-shard reblock is *acceptable* (a
neighbor exchange, not a gather) — it appears only for non-divisible
sharded axes ([[uneven-shard-reblock-collective]]), never on the
divisible benchmark grids, whose HLO stays all-to-all only.

**Verified (2026-07-12, 4×A100-80GB, fusion flag on).** The reconciled
solve is **bitwise-identical to the slab** at 256³/512³/768³ (the
component that changed is provably the same kernel), with equal
wall-time (512³ solve: new 8.13 ms fused / 15.4 ms eager ≡ slab) and
**768³ still fits** (no OOM). HLO `all-to-all` present, no
`all-gather`/`all-reduce` (`test_slab_fft`/`test_distributed_solve`
green on GPU). 0 recompiles; distributed-vs-1-device ≤1e-12. Caveat: the
exact **20.97 ms full-*step*** figure was not reproduced — no committed
512³ full-model harness exists (`bench_nonhydro.py` is the old stack) —
but the gate's intent (no regression vs the slab) holds by the solve's
bitwise identity. A committed nonhydro2 512³ step benchmark is a useful
follow-up before the release merge.

## 5. Scope boundaries / follow-ons

- **This plan:** 1-D slab (single-node), all-Fourier transforms. Delivers
  distributed `Fourier.forward`/`backward` + composed `SpectralSolve`.
- **Follow-on (pencil):** N-D pencil, the 2-D device mesh
  (`tensor.py:206` `NotImplementedError`), topology-aware mesh mapping,
  and multi-host (init docs + sharded-array construction/IO, findings §8)
  — gated by a ≥2-node smoke test.
- **Enabled but not implemented here:** mixed transforms (`Fourier ⊗
  Sine/Cosine/Chebyshev`). The lowering supports non-FFT stages (§2 A4);
  wiring the trig/Chebyshev per-stage kernels is separate work.
- `decomposition/graph.py` stays a stub; the flat `layouts` vocabulary +
  `layout_for` are sufficient at slab scale.

## 6. Open risks

- **Divide fusion — largely retired by the fused lowering.** Because the
  parity solve is one `shard_map` region with the diagonal multiply
  *inside* it (slab `_build_solve`), the Hadamard fuses by construction —
  no cross-region boundary to worry about. Residual: confirm the lifted
  kernel, driven by the planner geometry, emits the **same** HLO as the
  slab (the Stage-2 golden-ref diff).
- **Coeff-space sharding is Path X, not a check (recon 2026-07-12).**
  `store`/`sharding` replicate `CoefficientSpace` factors regardless of
  layout (`tensor.py:312,592`), so *any* path that materializes a
  distributed coeff field (standalone `field.fft()`, three-region
  composition) needs real decomposition work — deferred to the follow-on.
  The parity solve sidesteps it (no coeff field).
- **Replicated inverse diagonal (three-region only).** The composed form
  broadcasts a full replicated diagonal per device (~1 GB at 512³) — a
  memory regression the fused parity path avoids by `in_specs`-slicing.
  Relevant only when the follow-on lands the three-region composition.
- **New-path consumers — minimal under fused-first (recon 2026-07-12).**
  The fused solve touches only the pressure solve; the eigenbasis /
  diagnostic single-axis Fourier consumers stay on the gather path (no
  regression, no dividend). They switch — and need re-validation of their
  device-invariance + 0-recompile gates — only when Path X distributes
  standalone `Fourier.forward`.

## Related

Findings/constraints:
[`distributed_transform_reconciliation.md`](distributed_transform_reconciliation.md).
Design: `specs/grid/04_decomposition.md` §5.1,
`specs/grid/classes/operators_transforms.md`. Landed dependency:
`design/plans/done/uneven_shard_padding_plan.md`. Memory:
[[distributed-transform-gspmd-lowering]],
[[uneven-shard-reblock-collective]], [[new-stack-gpu-performance]],
[[xla-gpu-fusion-bug]].
