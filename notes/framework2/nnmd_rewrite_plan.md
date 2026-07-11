# NNMD rewrite plan (framework2)

**Status: research synthesis + workplan, 2026-07-11.** Closes the
"NNMD descoped, future rewrite"
([`projection_eigenmode_roadmap.md:156`](projection_eigenmode_roadmap.md),
[`cutover_parity_plan.md:13`](cutover_parity_plan.md)). Port design
archaeology retained in
[`model/research/d5_3_family_ports.md`](model/research/d5_3_family_ports.md)
§4.

## 0. Forensics (settled 2026-07-11)

### 0.1 The v1 sign flip — it was shallow water

Under the v1 convention `∂t z = −i A z`, the two models disagree:

- **nonhydro** `vec_q(s=+1)` satisfies `A q = +ω q` (standard); pinned
  by a propagation-direction test
  (`tests/nonhydro/grid/cartesian/test_eigenvectors.py:165`).
- **shallowwater** `vec_q(s=+1)` satisfies `A q = −ω q` (flipped,
  oscillates as `e^{+i|ω|t}`); no propagation test exists for it.

`framework/projection/nnmd.py:777` hard-codes the shallow-water pairing
(`zip([1,2],[1,-1])` sign factors), so v1 NNMD was correct for SW and
broken for nonhydro — and `tests/framework/projection/test_nnmd.py`
imports only shallowwater, which is why it never surfaced. **Moot for
the port**: framework2 eigenmodes were convention-verified by the
strong `Lq = iωq` test (three drafted-formula sign errors caught).
The rewrite must inherit that test pattern, not the convention debate.

### 0.2 The "telescoping performs worse at order ≥ 3" result is unreliable

In `../boundary_emission`:

- `src/boundary_emission/nnmd.py` (the **direct** recursion, not the
  telescoping one) is broken: the `order_derivative == 0` branch
  (`nnmd.py:126-130`) loops `for k in range(n)` with `n ≡ 0` and never
  reads the series order, so `self[m,0]` returns **zero for every
  m ≥ 1**. `__call__` only requests `self[j,0]`, so the class returns
  the *linear* vortical projection at every order — the real recursion,
  interaction terms, and inverse operator are dead code as invoked.
- `super_balance.py` holds the telescoping formulation (its
  `get_order_series_field` matches the v2-note formula, binomials and
  `A^{-(n+1)}` included).
- Both share a `+i/ω` (s=+1) inverse-operator sign, opposite to the
  validated v1 fridom NNMD — a latent secondary bug.
- **No comparison script/notebook is committed anywhere in that repo.**

So the observed "telescoping worse at order 3+" plausibly compared a
correct-ish telescoping against a baseline frozen at order 0, or ran
with a flipped inverse sign, or lives in an uncommitted notebook.
Treat the question as **open**, to be settled cleanly (M2, T1).

### 0.3 The projector formulation collapses to the v1 one (verified)

The "Stationary Eigenspace" vault note's recursion

```
φ_{n+1} = A⁻¹ W ( φ_n^{(1)} − Σ_{k=0}^n B(φ_k, φ_{n−k}) )
```

is componentwise identical to the implemented
`z_{j,n} = (i/λ_j)(∂_T z_{j,n−1} − p_j·I_{n−1})`: with
`A_note = −i A_v1`, the wave mode s=±1 has `A_note`-eigenvalue `∓iλ`,
so `A⁻¹` restricted to it is multiplication by `±i/λ` — exactly the v1
`-1j * sign * one_over_omega` factors, derivative recursion included.
The projector form is the coordinate-free statement of the same
method. It needs only **V, W, A⁻¹|_W, and the bilinear form B** — all
of which the framework2 eigenbasis tiers provide or nearly provide,
on periodic *and* walled domains.

## 1. Formulation work (Phase M — math, before any code)

- **M1 — design note, generalized unscaled formulation.** Write the
  method for `∂t φ = Lφ + B(φ,φ)` with a *declared* slow projector V
  (not necessarily ker L), W = I−V, `L_w = WLW` invertible on ran W.
  The only change vs the stationary case: the slow time derivative of
  the leading term keeps the slow linear rotation,

  ```
  φ_0^{(k)} = L φ_0^{(k−1)} + V Σ_m C(k−1,m) B(φ_0^{(m)}, φ_0^{(k−1−m)})
  φ_{n+1}   = L_w⁻¹ W ( φ_n^{(1)} − Σ_k B(φ_k, φ_{n−k}) )
  ```

  (this is the v2 note's `ε⁻¹Aφ_0 δ_{m,0}` term, made operational by
  never splitting off ε). Bookkeeping assumption to state explicitly:
  `|ω_slow| ≲ ε·|ω_fast|` and nonlinear rate ≲ ε·|ω_fast| — i.e. the
  frequency-gap predicate projectors from the channel eigenbasis
  (β-plane Rossby band) define V. Prove the collapse to §0.3 when
  `LV = 0`, and state the higher-derivative recursion
  `φ_n^{(k)}` including the n=0 correction term.
- **M2 — telescoping vs direct, settled analytically.** Both
  truncations should agree to `O(ε^{N+1})` if consistent; derive the
  remainder difference (candidate mechanism: `A^{−n−1}` amplifies
  high slow-derivative terms whose accuracy is lowest). Combined with
  the asymptotic-series view (the expansion is divergent; optimal
  truncation order grows as ε shrinks — Vanneste), decide whether
  telescoping is worth carrying at all. Empirical confirmation in T1.
  Default expectation: ship the direct recursion only.

## 2. Research tasks (Phase R — parallel with M)

- **R1 — literature sweep** (web search + 3–5 papers, extract test
  protocols and any prior projector-form NNMD):
  - *Implicit normal-mode initialization* (Temperton 1988/1989; Daley,
    "Atmospheric Data Analysis" NMI chapters): the operational
    community's projector/elliptic-inversion NMI for limited-area /
    bounded domains — the closest published relative of the projector
    formulation, and prior art for walls.
  - Warn, Bokhove, Shepherd, Vallis 1995 (slaving; ordering).
  - Vanneste 2013 Annu. Rev. (exponential asymptotics, optimal
    truncation ~ divergence at fixed ε — directly relevant to any
    "higher order got worse" observation).
  - Chouksey, Eden et al. (2018, 2022/23) balance comparisons — they
    ran exactly the balance→propagate→rebalance protocol; mine their
    diagnostics and ε-scaling figures for acceptance targets.
  - Masur & Oliver (2020+) optimal balance — cross-method reference;
    OB already exists as `fr.transforms.OptimalBalance`.
- **R2 — forensic replication.** Reproduce the direct-vs-telescoping
  comparison *correctly* inside the toy harness (T1); settles §0.2.

## 3. Test strategy (cheap → expensive)

The v1 protocol (balance → integrate an eddy turnover → rebalance →
residual) stays as **one benchmark**, not a test. The ladder:

- **T1 — toy ODE harness (primary correctness tool).** Implement the
  core recursion generically (operators injected: `V, W, invLw, B`,
  optional `L_slow`), so it runs on low-dimensional systems with
  machine-precision reference slow manifolds: Lorenz-86 /
  Lorenz–Krishnamurthy five-component, a single resonant triad with
  slow detuning δ ≠ 0 (exercises the non-stationary slow space of M1
  exactly). Millisecond unit tests; both formulations (R2) side by
  side.
- **T2 — ε-slope test (the PDE workhorse).** On tiny grids (32² SW,
  16³ nh), sweep Ro over a decade, assert
  `residual(order N) ∝ ε^{N+1}` in log–log. Residual = norm of the
  next-order correction `‖φ_{N+1}‖` (internal, zero extra machinery)
  plus the fast-tendency residual `‖W F(φ_b)‖` after subtracting the
  slaved prediction. Single tendency evaluations, no time stepping.
  A sign/convention bug flattens the slope immediately — this is the
  cheap replacement for the propagation test's *ranking* role.
- **T3 — differential balance residual (instantaneous emission).**
  `ρ = ‖W F(φ_b) − (balance(v + h·v̇) − balance(v))/h‖` with
  `v̇ = V F(φ_b)`: two balance calls + two tendency calls, no
  integration. This is the t→0 derivative of the run-and-rebalance
  residual — same ordering of methods, ~100× cheaper. Validate the
  equivalence once against T5.
- **T4 — short-burst oscillation test.** Integrate only 1–2 fast
  periods; measure the fast-oscillation amplitude of the wave
  component around its slaved value (std over the window). Captures
  actual emission without eddy-turnover integrations.
- **T5 — the full propagate+rebalance benchmark** (once per release of
  the method, both nh2 and sw2, plus an `OptimalBalance` cross-check
  on the same IC) in `benchmarks/`, not `tests/`.

## 4. Implementation phases

- **P0** — design note = M1 + M2 + user-surface sketch; owner reviews
  formulation and surface only (per standing preference).
- **P1** — generic core + toy harness (T1, R2). Location: core in
  `framework2/transforms/` (private module), toys in its test file.
- **P2** — framework2 surface gaps (the two flagged in the d5_3
  archaeology, generalized):
  - an **apply-f(L)-on-a-subspace** applicator on all three eigenmode
    tiers — analytic tier: per-s `1/ω` arrays (`em.omega_field`, the
    deferred S3 item) assembled as `Σ_s P_s · f(ω_s)`; channel tier:
    `Q diag(f(ω)·mask) Q^H M` next to `_project_masked`; numeric probe
    tier: same contraction. `f = 1/(iω)` (with structural-zero guard)
    is the NNMD instance; `f = ω ↦ (iω)` gives the forward operator
    for free.
  - **quadraticity lint** on the `nonlinear=` term filter (the method
    silently mis-balances non-quadratic terms; warn once).
  - **bilinear form** `S(z1,z2) = ½(N(z1+z2) − N(z1) − N(z2))` with
    `N(z) = model.variant(term_filter=~fr.terms.linear).tendency(z)`,
    dealiasing wired through the existing `PadFactor` machinery (v1's
    `enable_dealiasing` flag was never implemented — do it for real or
    drop it explicitly).
- **P3** — `NNMD` as a `StateTransform` (ctor shape mirroring
  `OptimalBalance`: `NNMD(model, order=3, *, nonlinear=~fr.terms.linear,
  at_time=0.0, ...)`), periodic analytic tier first; T2 slopes green;
  regression vs v1 **shallow-water** NNMD (the correct one) on a
  periodic SW case.
- **P4** — channel/walled tier + predicate slow space (β-plane Rossby
  band as V): the genuinely new capability. Walled demo as the
  showcase; multi-device sharding same condition as the eigenbasis
  projectors.
- **P5** — T5 benchmark, docs, close the cutover-parity NNMD line.

## 5. Decisions

1. **DECIDED (owner, 2026-07-11): drop** the v1 `use_model`
   finite-difference slow-derivative branch (Euler step + recursive
   sub-NNMD). Analytic derivatives only; revisit only if T2/T5 show
   the analytic recursion limiting.
2. Telescoping variant — decide AFTER M2/R2: ship as a second
   selectable scheme in framework2 (only if the corrected toy
   comparison shows it winning somewhere), or keep it toy-harness-only
   and document the conclusion (default).
3. **DECIDED (owner, 2026-07-11): `fr.transforms.BalanceExpansion`**
   (framework-level only, no package factories; NNMD stays in the
   docstring as the literature name). Slow space via family/predicate
   (`eb.projector` grammar); f(L) via general `em.function(f, sel)`;
   slow-tendency closure = leading-order with an empirical
   toy-harness gate. Full surface: `nnmd_design_note.md` §2.
4. **DECIDED (owner, 2026-07-11): P4 (walled/β predicate slow space)
   ships in the first slice** — P3 and P4 are one deliverable, not
   sequential releases. Plan the generic core (P1) and the f(L)
   applicator (P2) for all tiers from the start.
