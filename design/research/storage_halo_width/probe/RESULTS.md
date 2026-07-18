# Storage-halo width probe — results

Empirical probe of forcing the negotiated storage-halo width *below* the
traced demand. Single device, triperiodic, CPU (`JAX_PLATFORMS=cpu`),
`nonhydro2` (`import fridom.nonhydro2 as nh`), `family="nodal"`.

Scripts in this directory; run with `uv run` from the worktree, e.g.

```
JAX_PLATFORMS=cpu uv run python e1_parity.py 32 10
```

## Forcing mechanism (`_force_halo.py`)

Passing an explicit `halo=` spec to `negotiate` can only *raise* the
width (`merge_max`), so it cannot force a narrower store. The clean
lever is to clamp the negotiated spec. `force_halo(cap)` is a context
manager that patches two seams to `min(width, cap)` per name:

* `fridom.spatial.decomposition.decomposition._negotiated_halo` — the
  spec stored into the `TensorDecomposition` (the actual storage width).
* `Grid._demanded_halo` — the freshly-traced demand a **frozen** grid
  verifies against its recorded fingerprint. Clamped identically, else
  `_verify_frozen` sees demand 4 > record 3 and raises
  `GridFrozenError`.

Both patches must stay active across assembly + freeze + run/compile, so
each variant is wrapped in one `with force_halo(cap): ...` block.
Verified effective: `grid.decomposition.halo` and each field's
`.storage.shape` drop from `n+2*4` to `n+2*3` (upwind5) and `n+2*2` to
`n+2*1` (centered).

## E1 — parity of final state (n=32^3, 10 steps)

`e1_parity.py 32 10`. Per-field `max|forced - baseline|`:

**UpwindAdvection(5)** — baseline width {x:4,y:4,z:4} vs forced {3,3,3}:

| field | max\|diff\| | max\|base\| | rel |
|---|---|---|---|
| u | 0.000e+00 | 3.213 | 0.0 |
| v | 0.000e+00 | 3.310 | 0.0 |
| w | 0.000e+00 | 3.641 | 0.0 |
| b | 0.000e+00 | 4.059 | 0.0 |

**CenteredAdvection** — baseline width {2,2,2} vs forced {1,1,1}:

| field | max\|diff\| | max\|base\| | rel |
|---|---|---|---|
| u | 1.332e-15 | 3.392 | 3.9e-16 |
| v | 1.332e-15 | 3.557 | 3.7e-16 |
| w | 1.332e-15 | 3.803 | 3.5e-16 |
| b | 8.882e-16 | 4.219 | 2.1e-16 |

Upwind5 forced-3 is **bit-identical** (exactly 0.0); centered forced-1
differs only at fp-reassociation level (~1e-15). **Identical physics.**
(n=16^3/5 steps agrees: upwind5 0.0, centered ~4e-16.)

## E2 — sync behaviour (n=16^3)

`e2_sync.py 16`. Count of `Grid.sync` calls (each = one halo-exchange
node inserted into the traced graph). Monkeypatched at
`Grid.sync` (the `tests/spatial/test_exchange_counts.py` approach).

| variant | advection-only syncs | full-step syncs |
|---|---|---|
| upwind5 baseline (w4) | **7** | **14** |
| upwind5 forced-3 | **16** | **26** |
| centered baseline (w2) | **7** | **14** |
| centered forced-1 | **16** | **26** |

**This is the headline finding: forcing the width narrower is NOT free
— it inserts extra mid-chain syncs.** In the baseline the 4 state
components each pay one entry sync (+3 more) and the biased/centered
reconstruction ∘ flux-difference chain elides via the kernels' validity
claims. When forced narrower, the entry sync stamps `halo_valid` only to
the (reduced) negotiated width, which is **below** the depth the
consumption-side accounting attributes to `flux_diff ∘ reconstruct`
(interval-arithmetic sum: reconstruct 3/interp 1, +1 for the trailing
difference = 4/2). So `_ensure_valid`
(`operators/base.py:1626`) finds `valid < required` on the reconstructed
intermediate and inserts a fresh sync to refill the ghost layer before
the difference. The extra syncs fire on the **reconstructed intermediate
`unnamed` fields with partial `halo_valid`** (e.g. `{x:0,y:3,z:3}`,
`{x:3,y:0,z:3}`, ... one per advected-component × differenced-axis
combination) — never on the entry state components. The refill keeps the
physics identical (hence E1 = 0.0), trading storage bytes for exchanges.

Note the runtime accounting uses the interval-arithmetic **sum** (4),
not the tight composite footprint (3/side) established analytically —
that is exactly why width 3 storage suffices for correctness (E1) yet
still trips the mid-chain refill (E2). Recovering the storage win
*without* the extra syncs would require teaching the consumption-side
requirement of `flux_diff ∘ reconstruct` the tight composite depth.

## E3 — compiled-step memory (upwind5, chunk_size=1)

`e3_memory.py <n>`. Compiled length-1 chunk `memory_analysis()` plus
per-field storage bytes (float64). Predicted byte ratio
(n+6)^3/(n+8)^3.

**n=64^3** (predicted −8.10%):

| | baseline w4 (72^3) | forced-3 (70^3) | ratio |
|---|---|---|---|
| per-field storage | 2,985,984 B | 2,744,000 B | 0.9190 |
| sum state storage | 11,943,936 B | 10,976,000 B | 0.9190 |
| temp_size_in_bytes | 116,456,897 | 107,019,521 | 0.9190 |
| argument_size_in_bytes | 38,817,861 | 35,672,069 | 0.9190 |
| output_size_in_bytes | 38,818,029 | 35,672,237 | 0.9190 |

**n=96^3** (predicted −5.66%):

| | baseline w4 (104^3) | forced-3 (102^3) | ratio |
|---|---|---|---|
| per-field storage | 8,998,912 B | 8,489,664 B | 0.9434 |
| sum state storage | 35,995,648 B | 33,958,656 B | 0.9434 |
| temp_size_in_bytes | 350,962,625 | 331,101,953 | 0.9434 |
| argument_size_in_bytes | 116,985,925 | 110,365,701 | 0.9434 |
| output_size_in_bytes | 116,986,093 | 110,365,869 | 0.9434 |

`generated_code_size_in_bytes = 0` on both (CPU backend). Every reported
byte figure tracks the storage ratio (n+6)^3/(n+8)^3 to 4 s.f. — the
saving is exactly the ghost-shell volume and **shrinks with n** (8.1% at
64, 5.7% at 96; asymptotically →0 as the ghost shell becomes a vanishing
fraction of the cube).

## E4 — guard check (upwind5 forced width 2)

`e4_guard.py 16`. Forcing width 2 (below the biased-5 reach of 3/side)
raises during assembly's `eval_shape` dry-run:

```
TermEvaluationError: UpwindAdvection/advection: evaluation failed with
ValueError: the negotiated halo width 2 along 'x' is too small for the
5-point stencil of _BiasedFaceReconstruction; renegotiate with a
registry that declares the wider requirement
```

Fires via `composer._dry_terms` → `schedule.evaluate_entry` → the
`reconstruct.py:281` reach guard. Confirms width 3 is the true floor for
upwind5 and width 2 is correctly rejected (loud, not silent).

## E5 — indicative CPU wall-time (upwind5 n=64^3, chunk_size=1)

`e5_walltime.py 64 base` / `... 64 3`, fresh process each, median over 25
single steps after warmup. **INDICATIVE ONLY — login-node CPU, noisy.**

| variant | median/step | min | max |
|---|---|---|---|
| baseline (w4) | 61.95 ms | 51.43 | 71.38 |
| forced-3 | 53.77 ms | 44.61 | 71.67 |

Forced-3 is ~13% faster **on a single device** despite the extra E2
syncs, because a single-device "sync" is a cheap in-array periodic-wrap
copy (no inter-device communication) and the smaller (n+6)^3 arrays cut
arithmetic + memory traffic. **Caveat:** on multi-device the extra E2
mid-chain syncs become real halo *exchanges* (communication), which
would erase or reverse this single-device gain — so this number does not
generalize to the sharded regime.

## Summary / surprises

1. Width 3 for upwind5 (and width 1 for centered) is **physically
   sufficient** — bit-identical / fp-identical final states (E1),
   confirming the analytic 3-cell composite footprint.
2. But it is **not free under the current consumption-side sync
   accounting**: the narrower store drops `halo_valid` below the
   interval-arithmetic requirement of `flux_diff ∘ reconstruct`, so the
   runtime auto-inserts extra mid-chain refill syncs (7→16 advection
   syncs, 14→26 per full step) to keep the physics correct (E2). Storage
   bytes are traded for halo exchanges.
3. Memory saving is exactly the ghost-shell volume and modest / shrinking
   with n (−8.1% at 64^3, −5.7% at 96^3) (E3).
4. Width 2 (below reach) is correctly rejected loudly at assembly (E4).
5. On single-device CPU the smaller arrays win wall-time (~−13%,
   indicative); on multi-device the extra syncs (E2) would likely cancel
   or reverse that (E5).

The real win would need the consumption-side requirement of
`flux_diff ∘ reconstruct` to carry the tight composite depth (3), so the
narrower store *and* the single entry sync coexist — otherwise narrowing
storage just relocates the cost from memory to communication.
