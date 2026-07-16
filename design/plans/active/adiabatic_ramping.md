---
status: active
date: 2026-07-16
---

# Generalized adiabatic ramping — implementation plan (ROADMAP 3.8)

**Goal (owner, 2026-07-16):** generalize the adiabatic-ramping machinery
so a model can be continuously deformed between two operator
configurations — a *reference* system `L(0)` and a *target* system
`L(1)` with `L(s) = (1-rho(s)) L_ref + rho(s) L_target` — following the
JFM draft *Fast-slow splittings for geophysical flows via the adiabatic
theorem* (Rosenau, Chouksey, Eden, Koul & Oliver;
`~/OneDrive/PhD/PaperProjects/adiabatic/main.tex`). Optimal balance's
nonlinear ramp is the special case reference = linearization. Two hard
requirements. (i) **Never compute shared terms twice**: the convex
combination is an analytic statement, not an implementation — terms
present in both operators must be evaluated once, unscaled. (ii) **All
four propagator legs**: reference→target and target→reference, each
forward or backward in time. The paper's Coriolis ramp
`f(y,t) = f0 + beta * rho(t/tau) * y` is the first new consumer; the
staggered double-ramp protocol and the adiabatic projector (paper
appendix B) are the acceptance demos.

This upgrades the 2026-07-13 idea record in place. That record scoped
3.8 as an ergonomics/factoring task, parked "until a second consumer
appears" — the paper is that consumer, and it adds one genuinely new
capability the idea record did not cover: **field-valued blends** (the
Coriolis profile deforms, not just a scalar).

Companion records:
[`../roadmap/open.md`](../roadmap/open.md) §"Generalized adiabatic
ramping" and §"Time-dependent parameters and time-dependent fields"
(level 1 and the affine-blend subset of level 2 are absorbed here;
arbitrary time-dependent fields stay open there);
[`../specs/model/08_state_transforms.md`](../specs/model/08_state_transforms.md)
§10.5 (Propagator / OptimalBalance);
[`../research/exponential_stepper.md`](../research/exponential_stepper.md)
§5 (ETDRK4 fallback);
[`../research/d5_2_variants.md`](../research/d5_2_variants.md)
(term predicates).

## 1. Decisions (proposed — need owner sign-off)

**AR-D1 — Contract: smooth operator path, convex combination when
affine.** The framework guarantees a smooth path `L(lambda)` with
`L(0) = L_ref`, `L(1) = L_target` and `lambda(t) = rho((t-t0)/tau)`
whose endpoint time-derivatives vanish (inherited from the ramp by the
chain rule). That is all the adiabatic theorem needs. The literal
convex combination `(1-lambda) L_ref + lambda L_target` is reproduced
exactly wherever the lambda-coupling is *affine* — which covers every
planned consumer (Coriolis `f`, nonlinear scaling, stratification,
topography). Three blend mechanisms, by cost:

| term relation across endpoints | mechanism | extra cost |
|---|---|---|
| identical | none — term untouched | zero |
| affine in a parameter | blended parameter, `p(lambda) = p_ref + lambda*(p_target - p_ref)`, read at stage time | zero (scalar) / one fused multiply-add on a profile (field) |
| one-sided (e.g. `rho * N(z)`) | stage-time scaling parameter on the term (the shipped `scaling.rossby` pattern, `src/fridom/shallowwater2/modules/sadourny.py:603`) | one multiply of that term's output |
| non-affine or structurally disjoint | two term instances weighted `lambda`, `1-lambda` | that term twice — never the full operator |

**AR-D2 — Field blends are stage-time affine combinations of
assembly-materialized profiles; the Coriolis family is the first
instance.** Today `f_coriolis` is an AUXILIARY field frozen at assembly
(`src/fridom/model/modules/coriolis.py:386`
`jnp.full(space.shape, self.f0)`), so `FPlaneCoriolis(f0=Ramp(...))`
raises a bare `TypeError`. We do **not** build general time-dependent
fields (no SELF_UPDATE rewrite, no `(coords, t)` recompute contract —
those stay with the open roadmap entry). Instead: when a Coriolis
parameter is `TimeDependent`, the module declares the *static*
ingredient profiles at assembly (a `Profile("y")` holding `y`) and the
rotation term computes `f = f0(t) + beta(t) * y` in-term from
stage-time scalars (`ctx.params`). AUXILIARY treedefs stay scan-stable,
no halo traffic is added (pointwise, no stencil), lambda-sweeps never
recompile (`Ramp` leaves are dynamic). The static-parameter path is
kept bit-identical to today's assembly. This *is* the paper's
`f(y,t) = f0 + beta*rho(t/tau)*y`, and the same contract extends later
to stratification/topography blends (§7).

**AR-D3 — The blend lives in the model's clock-aware tendency, driven
by `ctx.params`; the transform algebra stays clock-free.** The
transform surface deliberately forbids Ramp-scaled transforms
(`src/fridom/model/transforms/base.py:287`). Deformations therefore
enter exclusively as Ramp-valued *parameter updates* on an internal
model variant (`Propagator(updates={param: Ramp(...)})`,
`src/fridom/model/transforms/propagator.py:70`), evaluated at stage
time inside the schedule. No new tendency-loop machinery.

**AR-D4 — `AdiabaticRamping(StateTransform)` is the named base
surface; parameters only; `OptimalBalance` becomes a subclass carrying
only balancing policy.** Constructor takes the model, a deformation
dict `{param_key: (v_ref, v_target)}`, `ramp_period`, `curve`, and
builds the internal `Propagator`. Two orthogonal involutions give the
four legs: `.reversed` swaps the endpoints (lambda path 1→0),
`.backward` retraces the same lambda path with `dt < 0`
(`params.TIME_STEP` sign flip + `Ramp.reversed()`, both shipped:
`propagator.py:84`, `src/fridom/model/time_dependent.py:281`).
`FixedPoint` stays optimal-balance policy, not base machinery (idea
record's open question — resolved as *policy*). Per the C4 precedent,
`MovingGeometry` is *not* a consumer: parameters only.

**AR-D5 — Staggered protocols are transform composition, not
interleaved update windows.** The double ramp is
`lin_down @ nl_down @ free @ nl_up @ lin_up` — five legs, each its own
`AdiabaticRamping`/`Propagator` with its own step count and a stepper
re-warm at every leg boundary (matches the paper: chained ramps are
"easier to test and reason about" than interleaved ones). Staggered
`t0` windows inside one leg remain possible via the `updates` dict but
are not the documented surface.

**AR-D6 — Backward legs refuse irreversible terms unless filtered.**
Backward diffusion is ill-posed; the literature either disables
dissipation during ramping or ramp-scales it with a sign flip (§6).
Policy: `AdiabaticRamping(backward=True)` (and OB's backward leg,
which already has the `backward_filter` seam,
`src/fridom/model/transforms/optimal_balance.py:141`) raises a taught
error naming the offending terms when a dissipative closure reaches a
backward leg unfiltered. Sign-reversed ramped dissipation is out of
scope (§7).

**AR-D7 — ETDRK4 + time-dependent linear terms: taught error.**
`ETDRK4` freezes `L` in an eigenbasis; a ramped parameter inside a
`linear=True` term would silently integrate a stale operator. Assembly
raises a taught error pointing at the AB fallback and at the measured
split option (stiff time-independent part in the eigenbasis,
time-dependent part in the tendency —
`../research/exponential_stepper.md` §5). Same taught-error treatment
extends the shipped precedent `EnergyMetric`/eigenmodes already use
(`at_time=` freezing, `src/fridom/model/energy.py:261`).

**AR-D8 — `AdiabaticProjection` composes ramped *linear* legs around a
reference-end projector; forward-in-time legs.** Paper appendix B: to
project where no spectral decomposition exists, ramp the linearized
system to the reference configuration, project there, ramp back —
`P_adiab = up_leg @ P_ref @ down_leg` on
`model.variant(term_filter=fr.terms.linear)`. Linear legs need no
backward integration (the leakage bound is time-symmetric), so both
legs run `dt > 0` with mirrored ramps. `P_adiab` is only
*approximately* idempotent (exponentially small in `tau`); the
idempotency tolerance is part of its documented contract, and it is
`traceable=False` like every Propagator-bearing transform.

**AR-D9 — Refactoring `OptimalBalance` must not perturb shipped
behaviour.** Base-point exchange, cost accounting, divergence policy,
`stopped_by` reporting stay bit-for-bit;
`tests/model/transforms/test_optimal_balance.py` (18 tests) is the
gate and must pass unmodified.

## 2. What exists / what is missing (probed 2026-07-16 on `dev`)

| Exists | Evidence |
|---|---|
| Continuous stage-time ramp, curves `linear/cosine/exp` (exp = the paper's Gevrey-2 form) or callable; `.reversed()`; affine composition; zero-recompile endpoint sweeps | `src/fridom/model/time_dependent.py:213,281` |
| Propagator legs: `reset; set_state; advance(N)` with Ramp-valued parameter updates; `backward=True` flips `TIME_STEP` | `src/fridom/model/transforms/propagator.py:70,84,150` |
| Backward-in-time integration first-class (dt-free AB tables, sign-symmetric warm-up; re-warm on dt flip) | `src/fridom/model/time_steppers/adam_bashforth.py:157`, `model.py:1632` |
| One-sided term weighting at stage time (`scaling.rossby` multiplies only nonlinear terms — shared `linear=True` terms already computed once) | `src/fridom/shallowwater2/modules/sadourny.py:603`, `src/fridom/model/terms.py:139` |
| Term-predicate operator selection: `fr.terms.linear`, `linearize(model)`, `require_linear_operator` | `src/fridom/model/term_predicates.py:404,446` |
| Working `OptimalBalance(StateTransform)`: ramp cycle, base-point exchange via `Shift @ (Identity - P)`, `FixedPoint`, divergence policy | `src/fridom/model/transforms/optimal_balance.py:46,187-206` |
| Reference-end projectors from the *discretized* operator, incl. the paper's channel classification (Kelvin separatrix; Rossby–Yanai fast iff `beta >= 2 k^2`) | `src/fridom/shallowwater2/channel_eigenmodes.py:132,245`, `src/fridom/model/eigen_channel.py:253` |
| Energy metric (the inner product under which `L` is skew-adjoint), Ramp-frozen at `at_time` | `src/fridom/model/energy.py:244,261` |

Missing (the actual new work):

- **Time-dependent scalar parameters** (roadmap level 1):
  `FPlaneCoriolis(f0=Ramp(...))` raises `TypeError` from `jnp.full`
  (`coriolis.py:386`); declared defaults must route through
  `resolve_at`, with provides-constancy bookkeeping (`coriolis.f0`
  claims constancy, `coriolis.py:331`) and the AR-D7 taught error.
- **Blended Coriolis field** (AR-D2): the in-term
  `f0(t) + beta(t)*y` path for both `BetaPlaneCoriolis`
  (`coriolis.py:403`) and shallowwater2's conserving rotation
  (`src/fridom/shallowwater2/modules/coriolis.py`).
- **`AdiabaticRamping`** itself — nothing by that name exists; OB's
  leg construction (`optimal_balance.py:107-152`) is the code to
  re-home. OB also hard-rejects a time-dependent nominal Rossby
  (`optimal_balance.py:116-124`) — subsumed by the new surface.
- **`AdiabaticProjection`** and a **relative-imbalance helper**
  `eta(z) = ||(I-P) z|| / ||z||` (paper eq. 5.2) under `EnergyMetric`.
- **Backward-leg dissipation guard** (AR-D6).
- **Example + docs**: no example or docs page exercises balancing at
  all today; the double-ramp protocol is greenfield (the trigger the
  idea record was waiting for).

## 3. The design

Paper ↔ fridom vocabulary:

| paper | fridom |
|---|---|
| reference system `L(0)` | internal `model.variant(updates=endpoint params at lambda=0)` |
| target system `L(1)` | the user's assembled model |
| ramp `rho`, Gevrey class 2 | `Ramp(..., curve="exp")` (same closed form) |
| deformation / homotopy | `AdiabaticRamping(model, ramps={param: (ref, target)}, ...)` |
| four propagator legs | `.reversed` (endpoint swap) x `.backward` (dt sign) |
| staggered double ramp | composition `@` of legs (AR-D5) |
| relative imbalance `eta` | `relative_imbalance(z, projection, metric)` |
| adiabatic projector (app. B) | `AdiabaticProjection(ramping, reference_projection)` |
| fast–slow mode classification | shipped `ChannelEigenmodes` labeling |

Surface sketch (final signatures settled at stub time, R3):

```python
import fridom as fr

sw = ...  # target model: beta-plane channel, nonlinear (shallowwater2)
lin = sw.variant(term_filter=fr.terms.linear)

# Coriolis ramp on the linear system: f(y,t) = f0 + rho(t/tau_f)*beta*y
lin_up = fr.transforms.AdiabaticRamping(
    lin, ramps={"coriolis.beta": (0.0, beta)},
    ramp_period=tau_f, curve="exp")
lin_down = lin_up.reversed          # target -> reference, dt > 0

# nonlinear ramp at fixed beta — the optimal-balance special case
nl_up = fr.transforms.AdiabaticRamping(
    sw, ramps={fr.params.SCALING_ROSSBY: (0.0, ro)},
    ramp_period=tau_n, curve="exp")

# staggered double ramp (paper fig. 4); rightmost applies first
free = fr.Propagator(sw, runlen=t_diag)
double_ramp = lin_down @ nl_up.reversed @ free @ nl_up @ lin_up

# reference-end slow projector (vortical + stationary Kelvin) and eta
P_slow = sw.transforms.projection(...)   # from labeled eigenmodes
eta = fr.transforms.relative_imbalance(double_ramp(z0), P_slow, metric)

# optimal balance with an adiabatically obtained projector (app. B)
P_adiab = fr.transforms.AdiabaticProjection(lin_up, P_slow)
ob = fr.OptimalBalance(sw, base_projection=P_adiab, ramp_period=tau_n)
```

`OptimalBalance(AdiabaticRamping)` keeps its constructor and behaviour;
its up-leg is `AdiabaticRamping(model,
ramps={SCALING_ROSSBY: (0.0, nominal)}, ...)`, its down-leg
`up.reversed.backward`, and it contributes the base-point exchange,
`FixedPoint`, and divergence policy only.

New modules and mirrored tests:

| new module | tests |
|---|---|
| `src/fridom/model/transforms/adiabatic_ramping.py` | `tests/model/transforms/test_adiabatic_ramping.py` |
| `src/fridom/model/transforms/adiabatic_projection.py` | `tests/model/transforms/test_adiabatic_projection.py` |
| `relative_imbalance` in `src/fridom/model/transforms/norms.py` | `tests/model/transforms/test_norms.py` |
| touched: `parameters`/assembly (R1), `modules/coriolis.py` + sw2 rotation (R2), `optimal_balance.py` (R4) | their mirrored test files |

## 4. Stages

| Stage | Work | Effort | Gate |
|---|---|---|---|
| **R0** | Sign off AR-D1..D9; extend spec 08 §10 with the deformation contract (smooth path, affinity ⇒ exact convex combination, four legs, blend-cost table); this record flips `idea → active`. | S (0.5 d) | Owner approval of the decision block. |
| **R1** | Time-dependent scalar parameters: stubs first (declaration-path signatures + taught-error skeletons + tests), then route declared scalar defaults through `resolve_at`; provides-constancy bookkeeping for `coriolis.f0`; AR-D7 taught error in ETDRK4 assembly. | M (2–3 d) | `FPlaneCoriolis(f0=Ramp(...))` advances under AB and matches a hand-stepped oracle; ETDRK4 raises the taught error; static-path assembly bit-identical; mirrored tests + one model smoke file; ruff clean. |
| **R2** | Blended Coriolis field (AR-D2): in-term `f0(t) + beta(t)*y` path in `BetaPlaneCoriolis` and sw2's conserving rotation; static `y` ingredient profile; static-parameter fast path untouched. | M (2–3 d) | Static params: bit-identical tendencies vs `dev`. Ramped beta on the linear channel: leakage `eta` decays ~exponentially in `tau` on a small grid (quantitative tolerance, both ramp directions); forced-4 multi-device pass. |
| **R3** | `AdiabaticRamping`: stubs (class skeleton, docstrings, lazypimp exports, `test_init` rows) → implementation: ramps dict → Ramp-valued `updates`, step snapping, `.reversed`/`.backward`, cost/info reporting, AR-D6 guard. | M (2–3 d) | Endpoint exactness (params at leg ends equal declared endpoints); 4-leg matrix unit-tested (lambda path x dt sign); linear up-then-down round trip ≈ identity within stated tolerance; dissipative-term guard raises; OB tests still green (pre-refactor). |
| **R4** | `OptimalBalance(AdiabaticRamping)` refactor: legs re-homed, policy retained (AR-D9). | S–M (1–2 d) | `tests/model/transforms/test_optimal_balance.py` passes **unmodified**; cost accounting unchanged; `bench_balance.py` numbers move only within noise. |
| **R5** | `AdiabaticProjection` + `relative_imbalance`; appendix-B protocol wiring. | M (2–3 d) | On the beta-channel: `P_adiab` matches the direct labeled-eigenmode projection (in-tree oracle) with error decreasing in `tau`; approximate idempotency within documented tolerance; OB with `base_projection=P_adiab` converges on a midlatitude case. |
| **R6** | Double-ramp example (equatorial beta-plane, paper fig. 4 protocol) + docs page for the ramping family. | M (2–3 d) | Owner-reviewed privately per AGENTS.md docs flow (local `docs/<topic>` branch, projected working-tree review, zero `REVIEW:` markers + explicit approval); example fits the sphinx-gallery time budget. |
| **R7** | Hygiene: this plan → `plans/done/` with outcome-vs-gates; roadmap 3.8 entry moved to `done.md`; time-dependent-fields entry trimmed to what stays open. | S (0.5 d) | `open.md` holds only open work. |

Every stage lands on its own `<type>/<topic>` branch with mirrored
tests (95% branch coverage) and `ruff` clean, per AGENTS.md — suggested
names: `feat/time-dependent-scalar-params` (R1),
`feat/blended-coriolis` (R2), `feat/adiabatic-ramping` (R3),
`refactor/optimal-balance-subclass` (R4),
`feat/adiabatic-projection` (R5), `docs/adiabatic-ramping` (R6).
Dependencies: R1 → {R2, R3}; R4 needs R3; R5 needs R2 + R3; R6 needs
R5; R2 ∥ R3 may proceed in parallel (separate agents, separate
worktrees).

## 5. Risks / open verification items

- **Leg-boundary stepper restarts.** Every leg re-warms the multistep
  ring from first order (sign-symmetric bootstrap). The literature does
  exactly this (Euler/AB2/AB3 bootstrap, §6) and reports it benign, but
  Chouksey et al. found C-grid diagnosed imbalance noisy at small Ro
  until dt was cut 10x — the R5 gate should include one dt-halving
  check to confirm leakage floors are ramp-limited, not stepper-limited.
- **Quasi-convergence is inherent.** The nudging iteration stalls at an
  exponentially small residual set by fast-phase uncertainty; it never
  reaches machine zero. `on_divergence="stop_best"` and `tol` semantics
  already encode this; tests must assert plateaus, not convergence to 0.
- **Classification edge near `beta = 2 k^2`.** The Rossby–Yanai /
  Kelvin identity exchange (paper §4.3) means enhanced leakage is
  *physical* near the crossing; R2/R5 tolerances must be set away from
  it (the shipped labeling already implements the prescription).
- **provides-constancy fallout (R1).** Ramping `f0` invalidates the
  `coriolis.f0` constancy claim; consumers (eigenmodes, EnergyMetric,
  IMEX-by-linearity) each need either `at_time=` freezing or a taught
  error. Enumerate consumers during R1 stubs via
  `require_linear_operator` call sites.
- **Multi-device.** The in-term profile blend is pointwise
  (halo-neutral), but R2 must verify under
  `FRIDOM_TEST_FORCED_DEVICES=4` since `f` feeds every rotation stencil.
- **Scope creep toward general time-dependent fields.** AR-D2
  deliberately stops at affine blends of assembly-materialized
  profiles. Anything needing `f(y, t)` beyond an affine path in
  declared scalars stays on the open roadmap entry.

## 6. Numerics from the literature (survey 2026-07-16)

| choice | recommendation | source |
|---|---|---|
| ramp shape | exponential `exp(-1/s)/(exp(-1/s)+exp(-1/(1-s)))`, Gevrey-2, asymptotically optimal; linear is O(eps), cosine O(eps^2) | Masur & Oliver 2020 §3.4 |
| ramp period | non-monotone optimum ~ slow eddy turnover; C-grid models liked T≈2–4 (spectral 0.5–2); `5/Ro` in OBTA | Chouksey et al. 2023 §5.1; Rosenau et al. 2026 |
| nudging iterations | ~5 suffice, cap ~10; stop on relative base-point change ≤ 1e-4 | Masur, Mohamad & Oliver 2023 §1, eq. 32 |
| base point | PV strongly preferred over height (loss-of-derivatives divergence); oblique (linear-PV-preserving) projector, never orthogonal | Masur & Oliver 2020 §3.5–3.6 |
| projector provenance | must come from the *discretized* operator; analytic eigenvectors give O(1) balance error on a C-grid | Chouksey et al. 2023 §5.3 |
| backward legs | negate dt; inviscid ramped dynamics is reversible; disable dissipation during ramping (or ramp-scale + sign-reverse it) | Rosenau et al. 2026; Masur & Oliver 2020 §4.1 |
| time step | resolve the *slow* scale only (nudging solver); accuracy floors at small Ro may need smaller dt | Masur, Mohamad & Oliver 2023 §6; Chouksey et al. 2023 |
| convergence character | quasi-convergence: residual plateau exponentially small, not zero; damped update `p + alpha*(p_new - p)` enlarges the basin (optional) | Masur, Mohamad & Oliver 2023 §6 |
| endpoint smoothness | vanishing endpoint derivatives buy the order — same theorem family as quantum "boundary cancellation" and MD adiabatic switching | Ge/Molnár/Cirac 2016; Albash & Lidar 2018 |

fridom already conforms on: exp ramp (`curve="exp"` is the same closed
form), discrete-operator projectors (channel eigenmodes), dt-negation
backward legs, divergence-guarded fixed point. The plan adds the
missing conformance point: the AR-D6 dissipation guard.

## 7. Out of scope (designed-for, not precluded)

- **General time-dependent fields** (`f(y,t)` beyond affine paths,
  evolving topography): stays on the roadmap's time-dependent-fields
  entry; the AR-D2 contract is forward-compatible (ingredient profiles
  + stage-time scalars generalize to more ingredients).
- **Ramp-scaled, sign-reversed dissipation on backward legs** (Masur &
  Oliver 2020 §4.1): the AR-D6 guard leaves the door open; add if a
  consumer needs viscous ramping.
- **Damped nudging update / ramp-period annealing** (quasi-convergence
  paper §6): optional OB robustness knobs; add behind the existing
  `FixedPoint` policy surface if needed.
- **Interleaved (non-staggered) multi-parameter ramps**: possible today
  via `updates` windows; not documented surface (AR-D5).
- **Stratification / topography deformations and 3-D vertical-mode
  ramps** (paper §6 outlook): same affine-blend contract; schedule when
  a consumer appears.
- **Time-averaged reference projectors (OBTA)** as `P_ref` inside
  `AdiabaticProjection`: composes naturally with the shipped
  `TimeAverage`; verify, don't build.
