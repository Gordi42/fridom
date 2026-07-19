---
status: active
date: 2026-07-18
---

# Multigrid coarse-grid agglomeration — replicate the deep levels

A plan to stop sharding the coarsest multigrid levels. Below a
per-shard-extent threshold a level's V-cycle work is replicated onto
every device (the hypre coarse-grid-agglomeration / PETSc `redundant`
pattern), reusing the existing local-axis halo path so it issues **no**
collectives. Two independent drivers, one mechanism.

Grounded in the shipped MG stack
([`multigrid_pathway_plan.md`](multigrid_pathway_plan.md), MG-D5 already
names the replicated-below-the-floor fallback) and the two same-day
measurement records that turned it from a designed-for into a driven
lever.

## 1. Motivation (two-fold)

*The capability driver below was the plan's original hypothesis; §4
records that it did **not** reproduce (MG-D5 already replicates below
the shardability floor at any P, so floor depth is reached at forced
16/32 devices with no depth cap). The reproduced driver is latency
only. Read this bullet as the framing at planning time, corrected in
§4.*

- **Capability.** A P-device sharded axis cannot coarsen below P cells:
  the V-cycle depth is capped by the device count, not by the problem.
  [`../../research/multigrid_depth_scaling.md`](../../research/multigrid_depth_scaling.md)
  showed that capping depth breaks h-independence — iterations climb
  **10 -> 27** at 512^3 when the hierarchy cannot reach its floor. On
  large device counts this bites at ever larger base sizes. Replicating
  the deep levels lets the hierarchy coarsen all the way to the
  4-cell floor regardless of P, restoring flat 10-iteration
  convergence.
- **Latency.** The collective census
  ([`../../research/multigrid_kernel_study.md`](../../research/multigrid_kernel_study.md)
  Addendum 3; full report
  [`../../research/artifacts/multigrid_gspmd_validation/census_collective_report.md`](../../research/artifacts/multigrid_gspmd_validation/census_collective_report.md))
  found that at 128^3 on 4 A100s the two coarsest levels (x = 8 and 4
  cells total -> **2 and 1 planes per shard**) fire **~33% of the halo
  collective-permutes** — ~86 of 262 per V-cycle, ~860 sub-kilobyte
  latency-only permutes per step — each moving <1 KB at the ~10 us
  NVLink small-message latency floor. That is **~9-14 ms of the ~34 ms**
  4-GPU-minus-1-GPU overhead at 128^3, spent sharding levels that have
  almost nothing left to shard. This is the structural reason
  mg-cuSPARSE is 0.37x spectral at 128^3 on 4 GPUs (Addendum 2).

## 2. Design

**Coarse-grid agglomeration by full replication below a threshold.**

- **Switch rule.** Descend the hierarchy; agglomerate at the first level
  where the per-shard extent along **any** sharded axis drops below
  `tau` planes (`tau ~= 4`, to be measured) **AND** the total level size
  is `<= a` bytes (a guard so a huge-P case never full-replicates a
  still-large level). At P = 4 the guarded and unguarded rules coincide,
  because the levels that undershoot `tau` are already tiny; the guard
  only matters at large P, where a level can undershoot `tau` per shard
  while still being globally large — that **middle regime is PARTIAL
  agglomeration onto a mesh sub-group, explicitly DEFERRED** (§5) until a
  measurement demands it.
- **Mechanism.** Levels at or below the switch carry a **replicated
  layout** (every axis local — `Layout({})` over the same device mesh,
  the MG-D5 fallback). Their halo exchange reuses the **existing
  local-axis path** — no new comm pattern, no new kernel. Restriction
  *across* the switch boundary (fine sharded -> coarse replicated)
  inserts **one small all-gather / reshard per V-cycle**; prolongation
  back (replicated -> sharded) is a **local slice**. Below the switch,
  every device does the same redundant compute on the full small level
  and issues zero collectives.
- **Knob.** A model setting in the existing house style (a sibling of
  `multigrid_levels`), fingerprint-static. **Default OFF in the
  prototype**; making it default-on is a later measurement-backed owner
  decision (Phase 4).
- **Differentiability.** This is step-path code, so per AGENTS.md the
  change ships one cheap `jax.grad` regression test through a short run
  (Phase 2). The transfers are linear; the reshard is a pure relayout —
  no new singular divide is introduced, but the gate is required
  regardless.

The replicated-below-the-floor fallback already exists structurally
(MG-D5, `negotiate(..., allow_replicated=True)`, exercised by the
`x12->6` GB-5 shard). This plan **chooses** it deliberately for deep
levels rather than only falling into it when an axis stops dividing P,
and adds the threshold rule + the cross-boundary reshard.

## 3. Phases

- **Phase 0 — probe (cheap, CPU forced-device).** Characterize the
  current behavior when a level's axis extent `< device count` under
  forced 4/8/16 host devices: does floor-depth coarsening stop early,
  does jax pad with empty shards, does the negotiator replicate or
  raise, what breaks. No repo edits — reuses the existing hierarchy
  builder. Establishes the exact status-quo the agglomerated path must
  match and the failure it must remove.
- **Phase 1 — implement.** Replicated-below-threshold in the hierarchy
  builder (`nonhydro2/modules/multigrid_hierarchy.py`) + the switch rule
  + the cross-boundary reshard on the transfers, behind the new knob.
  Reuses `Grid.coarsened` with the replicated device set and the
  existing local-axis halo lowering.
- **Phase 2 — parity gates.** Agglomerated vs status-quo results
  **machine-precision identical** (the V-cycle is the same operator, only
  its layout changed) on walled / mapped / immersed grids and on the
  semicoarsening fallback (`multigrid_coarsen_vertical=False`), under
  forced 4 / 8 / 16 devices. Plus a **capability test**: a small grid on
  16 forced devices whose floor depth is reachable *only* with
  agglomeration (status quo caps depth and the iteration count degrades;
  agglomerated restores flat convergence). Plus the autodiff regression
  test (Phase 1 change is step-path).
- **Phase 3 — 4-GPU GB-2 measurement.** Real 4x A100: 128^3 and 512^3
  (+ immersed), agglomerated vs status-quo mg vs spectral, ms/step under
  the GB-2 protocol. Threshold sweep `tau in {2, 4, 8}` to pin the
  switch. Confirms the ~9-14 ms recovery projected at 128^3 and checks
  it does not regress 512^3 (where the coarse levels are a smaller
  fraction). **Agents never submit GPU jobs unless Silvano asks in
  chat** (AGENTS.md).
- **Phase 4 — default decision (owner gate).** With the sweep in hand,
  the owner decides whether agglomeration becomes the default for
  multi-device mg (and at what `tau` / byte guard).

## 4. Prototype results

Shipped on `feat/multigrid-agglomeration` (2026-07-19). Phases 0-3
complete. Phase 3 (4-GPU wall-clock) **ran** (owner-authorized, job
`26355284`) and found **no `tau` is a wall-clock win** — the projected
~9-14 ms/step recovery did not reproduce and the immersed case
regresses — so **default OFF stands** (Phase 4 owner decision now
data-backed; see below).

### Phase 0 — status quo at high forced device count (CPU)

Probed forced 16 and 32 host devices, mg floor depth (`multigrid_levels
= None`), on small mapped / immersed / semicoarsen grids. **Two
corrections to the motivation's framing fall out:**

- **No crash, no early stop, no empty shards — floor depth is reached
  today at every device count.** The hierarchy builder coarsens to the
  four-cell floor independent of P, and each coarse grid negotiates its
  *own* valid layout, so `check_level_shardability` never fires. The
  literal capability claim (§1, "a P-device sharded axis cannot coarsen
  below P cells; depth capped by the device count") does **not**
  reproduce: below the shardability floor the ordinary negotiation
  already replicates (MG-D5 `allow_replicated`), and that is a universal
  safety net. So the depth-cap / h-independence break is **not** a
  device-count effect in this stack — it was the `multigrid_levels=5`
  cap (already fixed, floor-depth default `b8b165f1`).
- **The real, reproduced driver is latency.** In the semicoarsening /
  immersed hierarchy the divisible full vertical stays shardable, so as
  the horizontal axes coarsen below shardability the default sharded
  axis **flips from x to the vertical z**, which then stays sharded at
  **2 planes per shard** (extent 2 < P) all the way to the coarsest
  level — exactly the census's sub-KB latency regime, reproduced at
  P = 16. In full-3D coarsening the coarse levels instead replicate
  naturally (no axis divides P), so the finest coarse levels are the
  only tiny-shard exposure there. Agglomeration's value is therefore
  **making the replication deliberate and threshold-driven** — replacing
  every below-`tau` sharded coarse level (not only the strictly
  unshardable ones) with a replicated one — rather than curing a crash.

### Phase 1 — implementation (files + seam)

Landed in **`src/fridom/spatial/operators/multigrid_hierarchy.py`** (the
builder was promoted to `spatial` under GM-D5, not
`nonhydro2/modules/`), reusing the MG-D5 replicated layout end to end —
no new communication pattern:

- `decomposition.negotiate(..., force_replicated=True)` returns the
  replicated-only `Layout({})` over the **full** device set even when a
  factor would still shard (the capped halo geometry matches a naturally
  replicated level).
- `Grid.__init__(_force_replicated=)` threads it to both negotiation
  calls; `Grid.coarsened(replicated=)` sets it and extends the memo key.
- `coarsen_levels(agglomerate=tau)` peeks at each ordinary coarse level
  and, via `_should_agglomerate`, crosses the switch at the first level
  whose shortest would-be per-shard extent `< tau` **and** whose
  replicated per-device footprint `<= _AGGLOMERATE_MAX_BYTES` (a module
  constant, **4 MiB** float64); that level and every level below it are
  built `replicated=True`. `validate_agglomerate` guards the knob.
- The cross-boundary reshard is **not hand-written**: the transfer
  re-enters storage through `Grid.create_field` -> `store` -> `pad`,
  whose `jax.device_put(..., target_sharding)` is the all-gather on
  restrict (sharded fine -> replicated coarse) and the local slice on
  prolong (replicated -> sharded). Below the switch every operator /
  smoother / projection runs on a replicated array and issues zero
  collectives (the local-axis halo path y/z already use).
- Knob `multigrid_agglomerate: int | None = None` on `nh.Model` ->
  `DynamicalCore` -> `MappedPressureSolver` / `ImmersedPressureSolver`,
  house-style alongside `multigrid_levels`.

### Phase 2 — gates (forced 4 and 16 host devices, CPU; all green)

- **Parity (ON vs OFF, and vs the 1-device reference).** Identical CG
  iteration counts in every case (mapped 10 = 10, immersed 17 = 17 =
  17). The converged solutions are **not bitwise** ON vs OFF — the
  coarse mean projection reduces over a replicated array (local sum)
  instead of a sharded all-reduce, so the last bits reassociate — but
  the difference sits far below the `1e-8` solve tolerance and **ON is
  no less accurate than OFF against the 1-device truth**:
  - mapped (full-3D and semicoarsen): machine precision — 6-step model
    drift ON-vs-1dev `3.4e-16`, ON-vs-OFF `3.0e-16`.
  - immersed (semicoarsen): at the reduction-reassociation floor —
    single-solve ON-vs-1dev `2.2e-10`; 6-step model ON-vs-1dev
    `1.7e-11` **<** OFF-vs-1dev `3.5e-11`.
  On one device the knob is a bitwise no-op (`Layout({})` either way).
  The plan's "machine-precision identical" is therefore precise for
  mapped and should read, for immersed, **"identical iterations +
  agreement below the solve tolerance, no less accurate than OFF."**
- **Capability.** At forced devices OFF keeps a sharded coarse level
  whose per-shard extent is below the device count (the semicoarsen
  vertical at 2 planes/shard); ON replicates it — and every level below
  — at the **same floor depth** (same level count), leaving the
  above-`tau` levels sharded. Tested on immersed + mapped-full-3D +
  mapped-semicoarsen and on the plain-nodal hierarchy.
- **Autodiff.** `jax.grad` of a quadratic loss through a short run with
  the knob **ON** (immersed multigrid solve) is finite and central-FD
  matched to `rtol 1e-4` — the reshard is a pure relayout, no new
  singular divide.
- Mirrored tests for every touched file + `ruff check src tests` clean.

### Phase 3 — GPU wall-clock: RUN (owner-authorized 2026-07-19)

*An earlier attempt was skipped (the named allocation `jobid 26350895`
was dead at run time). Silvano then authorized one job in chat: job
`26355284`, node l50193, `--exclusive`, 4x A100-80GB, dev pinned at
`8752170a`, ~59 min, exit 0. Full data + census verdicts:
[`../../research/artifacts/multigrid_agglomeration_phase3/`](../../research/artifacts/multigrid_agglomeration_phase3/).*

**Matrix.** Mapped GB-2 steep terrain `n in {128,256,512}` ×
{spectral, mg-off, mg-t2, mg-t4, mg-t8}; immersed slope `n in {128,256}`
× {spectral, mg-off, mg-t4}. ms/step (median 6x20, compile excluded) +
CG iterations + a GPU HLO collective census (mg-off vs mg-t4, mapped
n=128).

**The projected ~9-14 ms/step recovery at 128^3 did NOT reproduce; no
`tau` is a wall-clock win, and default OFF stands.** Mapped mg-off is
0.405x spectral at 128^3 (73.9 vs 29.9 ms); the best `tau` (t8) reaches
only 0.439x (68.1 ms) — a **5.8 ms** recovery, and `tau=4` (the
structural default) recovers **0.6 ms**, both short of the 9-14 ms
projection and nowhere near closing the gap to spectral. At 512^3 mg-off
already wins (1.211x) and `tau` widens it only to 1.228x (t8, +1.4%),
still below the 1.5x bar. On mapped, larger `tau` is monotone-better
(t8 best: +8.5 / +5.7 / +1.4% vs off at 128/256/512), but the gains sit
within the run-to-run thermal spread and cost ~3x compile. On
**immersed** — the first 4-GPU immersed measurement, the worst
latency-bound case (mg-off 0.339x spectral at 128^3) — `tau=4`
**regresses** off at both sizes (−0.9%, −4.3%), so **no `tau` is
universally non-regressive**. CG iterations are IDENTICAL ON vs OFF in
every case (mapped 10, immersed 20/21); maxu ON-vs-OFF agrees to
~1e-15–1e-16; peak memory is unchanged.

**Census (mapped n=128 full-3D, the hierarchy GB-2 runs).** (a) `tau=4`
did **not** make the sub-KB coarse permutes vanish: it removed only the
L8 720-B halos (real/V-cycle 33->9) and **left the coarsest L4 permutes
untouched** (128-B/576-B, 24+22 both OFF and ON) — 262->241 CP per
V-cycle (−8%), 21 of the 86 flagged. The CPU immersed-semicoarsen
removal (halo 24->0, all-to-all 6->0) does not appear here: this
hierarchy has no line-smoother all-to-alls (A2A=0 both). (b) The GPU
partitioner does **not** fold the replicated-level reductions — it
re-partitions them like CPU: all-reduce rose 288->321/step, so the
total collective count barely moves (3219->3022/step, −6%), tracking the
null timing.

### Collective-count confirmation (CPU forced-4 HLO)

The GPU census is unrun, but dumping the **optimized** HLO of the
immersed semicoarsen solve at forced-4 host devices (n=32, `scan`
smoother) and running the census parser confirms the structural change
— per V-cycle (one real CG trip):

| kind | OFF | ON | note |
|---|---|---|---|
| collective-permute | 219 | 189 | the sub-KB coarsest z-halos removed |
| all-to-all | 6 | **0** | vertical-line-smoother column transposes |
| all-reduce | 12 | 34 | replicated-level reductions + reshard |
| all-gather | 32 | 44 | the switch-boundary reshards |

- **The flagged latency collectives are gone.** In the semicoarsen /
  immersed hierarchy OFF shards the *vertical* at the coarse levels, so
  the `scan` vertical-line smoother must transpose each column to solve
  along z — **6 small all-to-alls per V-cycle** (`f64[·,4,·,8]`, 256 B)
  — and the coarsest levels fire the census's **sub-KB z-halo permutes**
  (`f64[4,4,2]`×16 + `f64[4,4,3]`×8 = 24 tiny permutes/trip, 256-384 B).
  ON replicates those levels: **every column transpose and every
  sub-KB coarse z-halo disappears** (all-to-all 6 -> 0; the `f64[4,4,·]`
  permutes 24 -> 0).
- **The offset is a CPU-GSPMD folding gap.** ON adds restrict/prolong
  reshard permutes plus all-gather/all-reduce because XLA on CPU
  lowers the replicated coarse levels' reductions as partitioned
  all-reduces rather than folding them to local sums, and the
  switch-boundary all-gather is explicit. So the *total* collective
  count is roughly flat on this backend/size — the win is the removal
  of the **largest-payload (all-to-all) and tiniest-latency (sub-KB
  permute) coarse-level collectives**, not the raw count. Whether a
  `with_sharding_constraint` hint folds the replicated reductions on
  GPU (and the net ms/step) is the unrun Phase 3 question.

### Follow-ups surfaced by the prototype

- **Replicated-reduction folding — measured, unfolded on GPU.** Phase 3
  answered the "does GPU fold" question: it does **not** — the GPU
  partitioner re-partitions the replicated-level projection sums into
  all-reduces exactly like CPU (all-reduce 288->321/step). A
  `with_sharding_constraint` on the replicated level (or a manual local
  reduce) remains untried, but is moot given the null wall-clock result.
- **Phase 3 GPU sweep — RUN** (job `26355284`, 2026-07-19):
  `tau in {2,4,8}` × mapped `{128,256,512}` + immersed `{128,256}`,
  ms/step + CG iters + GPU HLO census. No `tau` is a wall-clock win;
  immersed regresses; default OFF stands. Only open Phase-3 residual is
  the **owner default decision** (Phase 4), now data-backed.
- The "machine-precision identical" gate wording should be relaxed to
  "identical iterations + agreement below the solve tolerance" for the
  immersed reassociation floor (see the parity numbers above).

## 5. Open questions

- **Partial agglomeration onto a mesh sub-group** (the large-P middle
  regime where a level is per-shard-tiny but globally large): deferred
  until a measurement demands it. Would agglomerate onto a *subset* of
  devices rather than replicating on all — the Fast(er)PM / PETSc
  sub-communicator pattern. Not built first; the byte guard keeps the
  full-replication rule from mis-firing there.
- **2-D multigrid (`ImplicitFreeSurface`)** shares this machinery: its
  2-D V-cycle hits the same coarse-level sharding floor, and the same
  replicate-below-threshold lever applies later with no new design.
- **`tau` and the byte guard** are the two tunables; Phase 3's sweep
  fixes `tau`, and the byte guard only becomes load-bearing at large P
  (Phase 4+ / the partial-agglomeration follow-up).

## 6. Relation to the existing record

MG-D5 ([`multigrid_pathway_plan.md`](multigrid_pathway_plan.md)) already
specifies "one device set for all levels; replicate below the
shardability floor" as the correctness fallback. This plan is the
**perf/capability promotion** of that fallback: choosing replication
deliberately for deep levels, adding the threshold + cross-boundary
reshard, and driving it with the census + depth-scaling evidence. The
sibling roadmap item lives under "Multigrid preconditioner — follow-up
measurements" in [`../../roadmap/open.md`](../../roadmap/open.md).
