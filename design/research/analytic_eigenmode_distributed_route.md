# Distributed route for analytic eigenmode consumers

**Date:** 2026-07-19 · **Status:** design ratified by the campaign
mandate ("do the effort now — analytic all-periodic eigenmodes and
balance/NNMD"); implementation dispatched the same day.
**Predecessors:** `gspmd_naive_transform_illegality.md` (Tier-1 guard),
the Phase-3 core (`8752170a`: `DistributedTransform`, `Channel2DPlan`),
the consumer wave (`b3712959`: `apply_diagonal`).

Two research tracks (mechanism: the analytic eigenmode machinery;
demand: the deferred consumers' dataflow) were run on 2026-07-19;
their full briefs are condensed here. File/line cites refer to dev at
`b3712959`.

## 1. Demand map — what actually needs serving

- **BalanceExpansion / NNMD** (same module,
  `model/transforms/balance_expansion.py`): the slaving recursion holds
  only *physical* state between orders; the seam is
  `_AnalyticOperator.__call__` (`:157-166`), which materializes a full
  coefficient VectorField per V/W/L_w⁻¹ call around a per-mode
  component-mixing matrix. One fused region per operator call
  eliminates the materialization entirely.
- **Analytic projections** (nh2/sw2 `transforms.py::_project`,
  `em.projector`/`em.function`): same shape — forward all components,
  per-mode matrix, backward, `.real`. Nothing couples across k; the
  cross-k physics lives in the physical bilinear outside the seam.
- **Analytic random-state / `mode()`**
  (`model/eigenstates.py::prescribed_spectra_coefficients`,
  `Eigenmodes.mode`): synthesize-only — coefficient arrays built from
  wavenumber formulas, then a lone `kit.backward`. Needs the backward
  half of the fused region plus frame-local gain construction.
- **OptimalBalance**: no seam of its own — a step-path `Propagator`
  ramp whose only transform is the injected `base_projection`. Served
  for free once projections are.
- **ETDRK4** (`model/time_steppers/exponential.py`): a *channel*
  consumer that rolls its own per-axis Fourier + einsum. Whole-step
  fusion is impossible (the physical tendency runs between RK stages);
  the amplitudes are raw sharded arrays spanning the four stages (no
  storage-contract violation). Fits as split project-half /
  synthesize-half fused regions: 5 projects + 4 synthesizes per step
  with shard-local phi/exp arithmetic between.
- **Not consumers:** `EnergyMetric` (J-weighted integrate / Parseval
  sum — never rides the seam); the numeric eigen build probes (already
  gather-guarded host paths).

## 2. Mechanism map — what the analytic engine is

- Per wavenumber the operator is **rank-1 per branch**:
  `out[c] = Σ_s q^s[c] · conj(p^s[c']) · M · z[c']` — a per-mode D×D
  matrix (`D` = component count: nh 4, sw2 3), pointwise in k. The
  columns are tag-checked compositions of operator eigenvalue symbols
  (`kit.diff/interp`), **jnp-traced and evaluable on any coefficient
  frame** — a frame-local rebuild is cheap. The dual row is
  `rayleigh_dual` (M-weighted pseudo-inverse).
- The distributed internal frame re-designates the half axis
  (verified: single-device `(9,16,16)` → internal `(16,16,9)`, x full,
  z halved) — the same trick the numeric engine uses. The shipped
  `TransposeGeometry`/`transpose_forward/backward` machinery (two
  all-to-alls, half-axis re-designation, padded-even indivisible
  shards, `.real` synthesis, per-shard array threading via `in_specs`)
  is **sufficient as-is**.
- The one missing piece of plumbing: `Eigenmodes.__init__` hard-builds
  its kit on the grid's default single-device rfftn frame and exposes
  **no frame hook** — there is no path today to evaluate
  `q, p, omega` on `dt.coeff.bare`.
- Degeneracies all fold into a pre-assembled matrix: nh Nyquist
  strata / variable-rank supplement columns (built host-gated with
  `np.any` today → build unconditionally with jnp masks); the sw2
  inertial DC patch (`_patch_mean`'s static corner write → a
  k==0-masked matrix entry); k=0 structural zeros (already the
  double-`where` pattern in `_dual` — shard-safe).
- Latent VJP hazards: `Symbol.magnitude`/`sqrt` are unguarded at
  structural zeros (`symbol.py:241,244`). The analytic surfaces are
  host-tier (exempt from the step-path policy), but the fused route
  should not *add* hazards; seal where cheap, following the numeric
  channel's grad-through-contraction precedent.

## 3. The design

**Primitive — `DistributedTransform.apply_matrix(fields, matrix)`**
(and the split halves `project`/`synthesize`): one shard_map region
that transposes-forward all D component fields, stacks them on a
trailing axis, applies the per-mode D×D matrix with one einsum
(`"...jd,...d->...j"`), transposes-backward, takes `.real`. The
matrix (shape `(*coeff_bare, D, D)`) threads through `in_specs`
sharded on the transform-axis plane, exactly like `apply_diagonal`'s
diagonal and `Channel2DPlan._q_spec`. Projection, `f(L)`, and the
balance operators are all the same call with different pre-assembled
matrices; random-state is the synthesize half fed frame-locally built
coefficient arrays.

**Frame-parametrized builder** on the analytic `Eigenmodes`: assemble
`Σ_s q^s (p^s)^H M` (with optional per-branch weights `f(ω_s)`, and
selection masks for projections) as a single `(*modes, D, D)` jnp
array **on an arbitrary coefficient frame** — the `symbol_factory`
analog. All host `np.any` gates lifted into masks; the DC patch and
Nyquist supplements folded in; the region body stays branch-free.

**Routing:** an analytic sibling of the numeric engine's
`resolve_distributed_contraction` — when the operand's layout shards
a stage axis and the space is plain Fourier, `_project`/`em.function`/
`_AnalyticOperator`/`prescribed_spectra_coefficients` route through
the fused region; single-device/replicated paths stay bit-identical.
The Tier-1 taught error narrows accordingly.

**ETDRK4 (independent work item):** expose the channel plans'
project/synthesize halves as standalone fused ops exchanging raw
sharded amplitude arrays; reroute `ETDRK4._forward/_backward` through
them; per-mode phi/exp arithmetic stays shard-local.

## 4. Phasing

- **Wave A (owner-named, dispatched):** the primitive + builder +
  fully-periodic analytic reroutes (projections, balance/NNMD,
  random-state) + test conversions (the sw2 sharded-random-state
  taught-error test flips to an invariance gate; nh2/sw2 projections,
  balance expansion gain device-count-invariance tests; HLO
  no-all-gather; a grad-through-projection regression).
- **Wave C (dispatched, parallel):** ETDRK4 via channel-plan halves +
  a device-count-invariance test for `test_exponential.py`.
- **Wave B (follow-up, recorded not dispatched):** the walled-vertical
  analytic tier — `ComposedTransform` (trig z stage) is declined by
  `resolve_distributed_transform`; needs a `ContractPlan`-shaped
  region absorbing the bounded axis + `ModeChart` embed/restrict into
  the stacked column. The owner named all-periodic; walled-vertical
  analytic grids keep the taught error until Wave B. [Update
  2026-07-19: shipped — `WalledVerticalTransform`, merge `4367d2f9`;
  fused synthesize is architecturally impossible on walled frames, so
  `can_synthesize=False` routes `synthesize_columns` to the
  replicated fallback. See done.md.]
- **`mode()` mirror-write:** the Hermitian mirror-index placement is
  cross-shard inside a region; route it through the synthesize half
  with the pair assembled in the (frame-local) coefficient arrays, as
  the numeric channel's `mode()` does via `ContractPlan.synthesize`.

## 5. What this closes

With Waves A and C landed, the Tier-2 escape list shrinks to the
irreducible pair (Chebyshev-vertical solve, mismatched-layout
composite) plus the recorded Wave-B tier — the state the owner set as
the precondition for the Tier-2 illegality decision.
