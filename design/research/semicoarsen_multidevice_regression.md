---
status: frozen
date: 2026-07-18
---

# Semicoarsening V-cycle multi-device break — root cause and closure

Companion records: the discovery evidence is
[`stretched_terrain_combined.md`](stretched_terrain_combined.md) (§GPU
validation addendum + §Re-check); the invariant fixes that close it
are [`halo_sharding_invariants.md`](halo_sharding_invariants.md)
(found and landed independently the same day by the invariants
campaign — the connection between the two was established by this
investigation).

## Symptom

Discovered 2026-07-18 on the stretched+terrain GPU leg: the
semicoarsening multigrid hierarchy (horizontal-only coarsening +
full-vertical line smoother) produced wrong numbers on >= 2 devices —
1-device clean, 4-device V-cycle asymmetry `3.5e-3`, preconditioned
solve rel error up to `0.96` (solve rel-residual `7.09` where 1-device
reaches `5.5e-9`) — bit-identically on forced-CPU-4 and real GPU-4,
kernel-independent. Victims: the 4 stretched-column multigrid tests,
`converges_under_both_coarsenings[False]`,
`full_and_semi_coarsening_agree_on_the_solution`; pre-GM-D9 also the
GB-5 parity cases.

## Bisect (forced-CPU-4, isolated worktree + own venv)

Predicate: `test_forced4_multigrid_solve_matches_single_device
[replicated-x12-coarse6]` (0.959 vs the 1e-8 gate when broken).
Endpoints verified: T8 `5af2e370` PASS, `c4497db5` FAIL. First-parent
probe over the 49-commit window, then inner-branch bisect of the
first-bad merge:

**First-bad commit: `4ca61a96` "spatial: two-sided (interval) halo
accounting"** (inside merge `40a24df8`); its parent `a810c16d` is
clean, and the failure signature is bit-identical at every bad point.
`uv.lock` is unchanged across the window (not a jax change).

## Mechanism — a layout flip exposing a latent bounded-axis sync hole

Two ingredients, both required:

**(1) The negotiation layout flip (introduced by `4ca61a96`).**
Side-by-side per-level negotiation at the flip commit vs its parent:

| level (config) | n_cells | layout @ a810c16d (good) | layout @ 4ca61a96 (bad) |
|---|---|---|---|
| x12 L0 | x=12, sigma=8 | {x: devices} | {x: devices} |
| x12 L1 | x=6, sigma=8 | **replicated {}** | **{sigma: devices}** |
| x16 L0 | x=16, sigma=8 | {x: devices} | {x: devices} |
| x16 L1 | x=8, sigma=8 | **replicated {}** | {x: devices} |
| x16 L2 | x=4, sigma=8 | **replicated {}** | **{sigma: devices}** |

The interval accounting halved the registry-visible widths (symmetric
`2` -> interval `(1,1)`, `HaloSpec[name] = max(lo, hi) = 1`), so the
shardability check `last >= width + 1` newly passed on
2-cells-per-shard axes: coarse levels that had correctly fallen to the
MG-D5 **replicated** layout began to shard — and the deepest level,
whose horizontals no longer qualify, sharded the only remaining axis,
**sigma**: the vertical that semicoarsening keeps full (the
line-smoother axis, and on these grids the *bounded* axis). The test
id `replicated-x12-coarse6` encoded the expected replication; the
commit broke exactly that expectation. Semicoarsening is structurally
exposed because it retains a full-size (shardable) vertical at every
level; full coarsening halves the vertical too, so its coarsest level
has nothing shardable and replicates — which is why
`converges_under_both_coarsenings[True]` stayed green throughout and
why the GM-D9 full-coarsening default was never affected.

**(2) The latent corrupt op (pre-existing, exposed by the flip).**
Per-stage 1-dev-vs-4-dev instrumentation on the broken state pinned
the first divergence to the coarse-level mapped elliptic **operator
apply** — the staggered flux-divergence along the sharded bounded
sigma axis (`mapped_pressure.py` `self._div[base]`) inside the
`VerticalLineJacobi` sweeps: clean at sweep 0 (pure `T^-1`), corrupt
by sweep 7 (rel 3.2), with the error landing **exactly on the sigma
shard-boundary cell pairs** ([3|4], [7|8], [11|12] of 16 cells;
interior cells clean). The tridiagonal line solve itself is
device-invariant even along the sharded axis (`6e-14`). This is the
`halo_sharding_invariants.md` §3 class — bounded staggered kernels
published a cancelled `(0,0)` exterior reach, so the inter-shard halo
sync was silently skipped on sharded bounded axes (the same hole
behind the walled diffusion/friction silent-wrong-physics and the
lone sharded bounded 1-D `MappedIntervalMesh` failures in
`test_stretched_mesh.py`). The "involuntary rematerialization /
roll-gather resharding" XLA warnings in the discovery record were the
downstream lowering symptom of that mis-declared reach, not the
cause.

## Closure

Both ingredients were repaired independently the same day by the
invariants campaign (`halo_sharding_invariants.md`): `b57e3e78`
(bounded staggered ops publish the per-shard `footprint_reach` — the
sharded bounded-axis sync fires again; the operative cure here) and
`94786a7c` (cap floor = per-application stencil reach; neutral for
the reach-1 Poisson operator — it does **not** re-collapse these
coarse levels, which remain sigma-sharded by choice on current dev).

Post-fix verification (this investigation, dev `95010353` lineage):

- forced-CPU-4: all 5 semicoarsening multigrid tests PASS (incl. the
  two dev-red ones); `test_mapped_pressure_stretched.py` +
  `test_stretched_mesh.py` **23/23 PASS** (all 4 stretched multigrid
  victims + both lone-1D bounded-mesh failures cured).
- **Real 4x A100** (jax#39100 workaround set): the full three-file
  battery (`test_mapped_pressure_multigrid.py` +
  `test_mapped_pressure_stretched.py` + `test_stretched_mesh.py`)
  **47 passed, 1 skipped** — the same files scored 9 failed / 38
  passed / 1 skipped the previous run at `33707661`; all nine
  victims flip green, including
  `test_multigrid_hlo_grows_with_the_level_count` (whose 4-device
  HLO-structure failure disappeared with the corrected lowering).
- Latent-hazard probe: verdict **reachable-but-correct**. The
  sigma-sharded coarsest level is still what every real
  semicoarsening config negotiates (16^3 -> (4,4,16) sigma 4/shard;
  16x16x64 -> (4,4,64) sigma 16/shard; stretched 16x16x64 likewise;
  only indivisible-horizontal + tiny-sigma configs replicate). Parity
  is now machine-precision on all of them: vcycle rel `1.6e-15` /
  `1.6e-14` / `5.2e-14`, short-solve residuals identical to all
  printed digits, nodal and stretched alike. One coarse sweep lowers
  to 181 collective-permutes, 0 all-gathers — the distributed
  line-solve and bounded-axis halos partition cleanly.

## Residue (all perf/hardening; no correctness item remains)

1. **Coarse-level replication preference (owner call).** A
   sigma-sharded coarsest level is correct but pays ~181
   collective-permutes per coarse sweep (x8 sweeps x pre+post x CG
   iterations) on the smallest, most latency-bound grid; the
   replicated layout MG-D5 intended is likely faster and structurally
   immune to this seam class. Options, smallest first: demote or
   exclude the hierarchy `vertical` in the coarse-level
   `_shardable_names` ranking (thread the name through
   `Grid.coarsened` -> `negotiate`, sibling to `_allow_replicated`);
   or replicate below a cell-count floor. Measure the coarse-level
   timing before choosing (`multigrid_kernel_study.md` methods).
2. **Defense in depth:** a hierarchy-builder warning (or A5-gate
   check) when a level's layout shards the line-smoother axis — any
   future negotiation-policy drift fails loudly instead of silently.
3. **Stretched-base eager pre-warm (perf).** Build the coarse chain
   host-side at solver construction so `Grid.coarsened`'s memo makes
   the trace-time rebuild a memo hit — lifts the
   `MappedIntervalMesh`-ctor jit-incompatibility, letting stretched
   columns take the GM-D9 full-coarsening default (faster, and its
   coarsest level replicates naturally).
4. **Test hygiene:** `multi_device` markers for the parity victims
   (unmarked 4-device-only failures breach the house device-count
   rule; single-device CI never sees them).
