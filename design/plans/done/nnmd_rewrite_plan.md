---
status: done
date: 2026-07-13
---

# NNMD rewrite plan (shipped)

The nonlinear normal-mode decomposition rewrite. **Shipped
2026-07-11** as `fr.transforms.BalanceExpansion`
(`src/fridom/model/transforms/balance_expansion.py`, private core
`_slaving.py`); T5 benchmark landed with it. This closes the "NNMD
descoped, future rewrite" line of
[`projection_eigenmode_roadmap.md`](../active/projection_eigenmode_roadmap.md)
and [`cutover_parity_plan.md`](../active/cutover_parity_plan.md). ROADMAP 2.8
still says "NNMD deferred to its own future rewrite" — that sentence
is stale.

Design: [`../../specs/nnmd/nnmd_design_note.md`](../../specs/nnmd/nnmd_design_note.md).
Literature: [`../../research/nnmd_literature.md`](../../research/nnmd_literature.md).
Port archaeology: [`../../research/d5_3_family_ports.md`](../../research/d5_3_family_ports.md) §4.

## Phases (all complete)

- **M1/M2 + P0** done — formulation (generalized unscaled recursion,
  declared slow projector V, telescoping-vs-direct analysis) and the
  owner-reviewed surface: `../../specs/nnmd/nnmd_design_note.md`
  (85960a5).
- **R1** done — literature sweep, record:
  `../../research/nnmd_literature.md` (f3048f9).
- **R2 / T1** done — forensic replication in the toy harness
  (Lorenz–Krishnamurthy, detuned triad); both schemes side by side
  (86d807b, `tests/model/transforms/test_slaving.py`).
- **P1** done — generic slaving core + toy harness,
  `model/transforms/_slaving.py` (86d807b).
- **P2** done — `em.function(f, sel)` applicator on all three
  eigenmode tiers, quadraticity lint, bilinear form (0338f4e).
- **P3 + P4** done — `fr.transforms.BalanceExpansion`: periodic
  analytic tier, channel/walled tier, predicate slow space
  (β-plane Rossby band); shipped as one slice (859eab3).
- **P5** done — T5 propagate+rebalance benchmark
  (`benchmarks/model/bench_balance.py`, results in
  `benchmarks/RESULTS.md`; 10ce626).

Test ladder T2 (ε-slopes), T3 (`residual_fast`), T4 live in
`tests/model/transforms/test_balance_expansion.py`; T5 is a manual
benchmark, not a test.

## Results at shipping

- sw2 ε-slopes exactly N+1 (1.999 / 2.999).
- Walled f-plane channel residuals 7.8e-3 / 4.9e-5 / 4.5e-7
  (orders 0/1/2); β-plane predicate band r0 7.6e-3 → r1 1.4e-4.
- v1 shallow-water NNMD regression: 3.3e-16 / 3.3e-16 / 5.5e-14
  (closures first differ at order 3).
- T5 (32² periodic SW, one eddy turnover, Ro = 0.1): imbalance
  2.8e-2 / 8.2e-4 / 3.2e-5 / 1.7e-5 for orders 0–3 vs 6.0e-4 for
  `OptimalBalance` — within a factor ~2 of the Chouksey et al. 2023
  targets, order 3 floor-limited, no order inversion. The cheap
  `residual_fast` diagnostic ranks the methods exactly like the
  expensive protocol at every Ro (the key deliverable: the
  propagate+rebalance protocol never needs to run in CI).

## What the rewrite settled

- **The v1 sign flip was shallow water.** `framework/projection/nnmd.py`
  hard-codes the SW pairing, so v1 NNMD was correct for SW and broken
  for nonhydro; `tests/framework/projection/test_nnmd.py` imports only
  shallowwater, which is why it never surfaced. Moot for the new stack,
  whose eigenmodes are pinned by the strong `Lq = iωq` test.
- **"Telescoping performs worse at order ≥ 3" was an implementation
  bug**, not physics. The telescoping series is an exact term-by-term
  resummation of the direct recursion: consistent implementations
  coincide to machine precision at every order (1–5), both closures,
  both toys, with and without detuning. Only the direct recursion ships;
  both stay in the private core as the proof.
- **The projector formulation collapses to the v1 one** (verified): the
  coordinate-free recursion `φ_{n+1} = A⁻¹W(φ_n^{(1)} − Σ B(φ_k,φ_{n−k}))`
  is componentwise identical to the v1 `(i/λ_j)(∂_T z − p·I)` form. It
  needs only V, W, `A⁻¹|_W` and B — which the eigenbasis tiers supply on
  periodic *and* walled domains, hence the new walled capability.
- **Closure:** the leading-order closure saturates at residual slope 3
  from order 3 on (detuned triad); the order-consistent closure is
  uniformly N+1 through order 4 and is the shipping default.
- **Dropped:** the v1 `use_model` finite-difference slow-derivative
  branch (analytic derivatives only), and the never-implemented
  `enable_dealiasing` flag (dealiasing now runs through `PadFactor`).
- **Spec correction found in P3:** the planned T3 differential-residual
  formula was dimensionally off (the difference quotient carries the
  O(1) slow part v̇); the shipped `residual_fast` applies W to the
  central quotient and reports relative to ‖z_b‖_M.
- The `nonhydro2` walled-advection gap flagged during P4 was closed
  separately (4b4bc85, structural-impermeability centered advection);
  the walled nh channel test now runs the real advection module.

## Not done here

- Reader-facing docs / gallery example for `BalanceExpansion`
  (deferred by the owner at shipping). Not an NNMD item any more: it
  belongs to the general transforms documentation in
  [`docs_examples_plan.md`](../active/docs_examples_plan.md) — no `docs/` page
  covers any state transform today.
- ROADMAP 2.8's "NNMD deferred to its own future rewrite" needs a
  one-line correction.
