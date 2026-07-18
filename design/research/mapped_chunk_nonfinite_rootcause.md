---
status: complete
date: 2026-07-18
---

# Mapped + advection + chunked scan non-finite — root cause

Resolution of the open-roadmap item "Mapped + advection + chunked
scan goes non-finite on GPU" (recorded 2026-07-17 from the CG-tolerance
GPU measurement, dev `b77f8582`). Verdict up front: **the fault was
real, was never a GPU/compiler problem, and is already fixed on dev**
— accidentally, by `44b5cb8d` (2026-07-17 17:20, "guard mapped
velocity-correction divide VJP"), a commit whose message explicitly
believed the forward path was untouched. The remaining open work is
hardening, not fixing (see the residuals entry in
[`../roadmap/open.md`](../roadmap/open.md)).

## Symptom re-established (both endpoints, A100 + CPU)

Repro: the `bench_step.py` mapped-advective recipe — 3-D nonhydro2,
`zp = z·H(x)`, `H = 1 + 0.2 sin x`, x/y periodic, z walled, centered
advection, f-plane, `dsqr=0.25`, 30 pressure iterations, jet IC,
dt = 0.005.

| dev commit | chunk=1 | chunk≥2 |
|---|---|---|
| `b77f8582` (item recorded) | finite to it=23, physics blowup panic at it=**24** | panic at it=**2** (state tame, \|u\|≈0.5) |
| HEAD `b773e119` (today) | identical bit-for-bit trajectory, panic at it=24 | **bit-identical to chunk=1** through it=22 (every component), same physics panic at 24 |

Three separate facts untangle from the one recorded observation:

1. **The cadence fault (the real item):** at `b77f8582`, chunk≥2
   panics at iteration 2 with u/v/w/p non-finite in **every** cell
   (b still finite) — at n=64 as at n=256, on **CPU and GPU
   bit-identically**. Not physics (the state is tame), not a
   miscompile (backend-independent, deterministic), not
   size-dependent.
2. **The "on GPU" attribution was an artifact** — CPU was simply
   never tried at the failing config (the roadmap entry itself listed
   the CPU leg as open work). The `multi_output_fusion` workaround
   not helping is thereby explained: there was no compiler fault to
   work around.
3. **A genuine nonlinear instability** kills this unclosed inviscid
   centered config at it≈24 (t≈0.12) at 256³ — smooth exponential
   ramp from it≈5, per-step squaring from it≈18 (the quadratic
   advective feedback), cadence- and backend-independent, and
   bit-identical between `b77f8582` and HEAD. The original session's
   chunk=1 "control" ran finite only because it never reached 24
   steps. Any future long advective measurement on this bench config
   needs a closure or a shorter horizon; the CG-record's linear-only
   measurement is unaffected.

## Bisect

CPU probe (n=64, chunk=2, 6 steps; broken = panic it≤4, fixed =
finite), `git bisect` over `b77f8582..679e822b`:

- `30f4a624` (design-only, 07-17) — **broken** (panic it=2)
- `44b5cb8d` (child of `30f4a624`) — **fixed** (finite)

Adjacent-commit resolution; every later probed commit
(`8edf5cd1`, `c05f68c3`, `627caace`, `09c096ba`, `51db9ba6`, …,
HEAD) probes fixed.

## Mechanism

`44b5cb8d` sealed the mapped velocity correction's `flux / J`
(`nonhydro2/modules/mapped_pressure.py`, `_divide_by_jacobian`) with
the double-`jnp.where` guard: valid cells bitwise unchanged, and the
**never-valid storage padding now holds 0 instead of `x/0 = inf`**
(J is zero-filled there). The commit targeted the reverse-mode NaN
(`0·inf` in the VJP) and stated "the forward projection is
untouched" — but the padding `inf` *was* the forward fault, via the
chunk cadence:

- The storage-frame design tolerates arbitrary **finite** garbage in
  never-valid lanes: claims mark them invalid, consumers
  `_ensure_valid`-sync what they read, and wall closures may combine
  them only under a zero mask (`0·finite = 0`).
- `inf` breaks exactly that last assumption: `0·inf = NaN`.
- `_scrub_ghost_storage` (unpad → zero-repad) runs once per **chunk**,
  so at chunk=1 every step boundary physically cleanses the pad lanes
  and the inf never survives into a consumer. Inside a chunk≥2 scan
  the carry keeps raw storage; `_seal_carry_ghosts` refills only the
  **negotiated halo lanes**, never the structural padding. Step N's
  correction infs reach step N+1's advective wall arithmetic →
  `0·inf = NaN` in a masked slot → one bad value in the divergence
  RHS → the CG dot products globalize it in one iteration → p, then
  u/v/w via the correction, 100 % non-finite; b (updated with the
  still-finite pre-correction velocity) survives one step more.
- Selector checklist: mapped-only (flat has no `/J`), advection-only
  (the linear path never mask-combines pad lanes), chunk≥2-only
  (scrub cadence), dt-independent, size-independent,
  backend-independent. All observed selectors reproduced and
  explained.

## The latent hazard class

The fix was aimed at a different symptom; nothing structural prevents
recurrence. The class is: **any unguarded storage-frame divide by a
zero-padded factor plants `inf` in never-valid lanes, and the
per-chunk scrub cadence lets it detonate under any masked combine in
the next in-chunk step.** Known members: the `MetricScaled` divides
(`spatial/operators/mapped.py:219-222`, already flagged in the
roadmap as unguarded but "empirically reverse-safe" — the same
empirically applies forward today, HEAD chunk parity is bitwise at
256³, but one new composition away). The suite's only
mapped + chunked test file pins `chunk_size=1`
(`tests/nonhydro2/test_fv_fusion_guards.py`), so no existing test
would catch a recurrence. Hardening options are tracked in the
roadmap residuals entry.

## Provenance

Investigation 2026-07-18 (single A100-80GB node l50000 + CPU legs,
jax 0.10.2). Probe scripts (growth curve, fault matrix driver, bisect
probe) were session-scratch; the fault matrix and growth curves are
reproduced in full above and in the roadmap entry. Related records:
[`cg_stopping_criterion.md`](cg_stopping_criterion.md) (GPU addendum
— where the item was first observed; its linear-only measurement
stands), the differentiability policy in `AGENTS.md` (the
masked-singularity poison this is the *forward* twin of).

## Addendum (2026-07-18): hardening landing

- **Regression floor n=8.** The chunk-cadence parity regression
  (`test_mapped_advection_chunk_cadence_parity`,
  `tests/model/test_step_chunk.py`) red-checks by reverting the
  `_divide_by_jacobian` guard: it raises `PanicError` at it=2 already
  at **n=8**, a smaller detonation floor than the n=64 recorded in the
  fault matrix above.
- **HEAD-parity claim is backend-qualified.** The "HEAD chunk parity
  is bitwise at 256³" statement is a **GPU** measurement. On **CPU**
  the two scan-length groupings reassociate FP at ~1e-15, so the
  shipped test asserts finite + `allclose (rtol 1e-12, atol 1e-13)`,
  not bitwise. Bitwise parity is a GPU-only property here.
- **Site-attribution matrix.** The 4-combination red-check (guard on/off
  × `_divide_by_jacobian` / `MetricScaled`) localizes the fault site
  to **`_divide_by_jacobian`**: reverting only its guard reproduces
  the detonation, while reverting only the `MetricScaled` seal keeps
  the parity test green. The `MetricScaled` seal is therefore
  **defensive-only** — nothing routes the historical fault through it
  today.
- **D4 hold — resolved (2026-07-19).** The hold was overtaken the
  same evening: the differentiability campaign disproved the
  defensive-only premise for the **reverse** pass (a chart sw2
  IC-grad NaNs through the very same divides; the forward analysis
  above stands) and shipped its own seal, dev merge `7fdbc900`
  (`_sealed_metric_divide` on both branches, 4 operator tests +
  sphere-chart autodiff regression), with a measured cost proof of
  exactly 0.000% added FLOPs/bytes on the mapped PCG step (the
  static-geometry mask constant-folds, the singular quotient is
  DCE'd). The owner ratified the seal 2026-07-19. The held local
  branch `fix/pad-inf-hardening` was verified fully redundant (same
  double-`where` seal at the same two sites; its tests a subset) and
  deleted (tip `69b3a9e8`). No hardening residual remains.
