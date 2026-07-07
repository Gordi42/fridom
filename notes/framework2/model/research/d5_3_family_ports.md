# D5.3 — The ported projection family, walked end-to-end

Research report (see [`README.md`](README.md) for status). Note:
sketches here predate two d5_1 rulings (info via `call_with_info`,
not a `trace` attribute; the `on_divergence` spelling) — the
consolidated design is authoritative.

## 1. Spectral projections → Vortical/Wave/DivergenceProjection (Tier 1)

Archaeology: `GeostrophicSpectral` = fft → `q·(ẑ·p)` → ifft with
`q,p = grid.vec_q/vec_p(s)`; `WaveSpectral` is literally
`P(+1)+P(−1)`; only mset reads are f0/n2/dsqr (confirming D2.4).
**Nyquist**: `set_nyquist_to_zero` on q/p — the ± modes degenerate
at Nyquist and the normalization divides by ≈0.

Port: thin Tier-1 wrappers over the eigenmode seam; **dual source
constructors** — `VorticalProjection(model, *, discrete=True,
at_time=None)` (via `from_model`, inheriting all D2.4 validation
incl. the Ramp `at_time=` rule) or `VorticalProjection(em)` (the
explicit-params form). They never need the model — the true
dependency is `(grid, f0, n2, dsqr, discrete)`; the model form adds
consistency + FieldTable binding. **Nyquist lands inside
`em.q/p` materialization** (grid-level spectral mask), never in
transforms/user code; documented caveat: the completeness identity
`P_vort + P_wave + P_div = I` holds on the Nyquist-free subspace.

**Extra-component ruling needed** (tracer `dye` in the input): the
eigenspace has no tracer direction — default **`rest="zero"`**
(components outside the family map to zero; preserves completeness
restricted to the family; `state − P(state)` carries the full
tracer), `rest="pass"` as explicit opt-out ("geostrophic state keeps
its dye"). Old code never faced this (fixed-composition State).
**Signature ≠ treedef**: signatures compare
`(grid identity, mapped (name, space) subset)` — required so
explicit-params transforms and accumulator-augmented variants
interoperate.

## 2. GeostrophicTimeAverage → nh.transforms.TimeAverage (Tier 2)

Archaeology, precisely: **flat mean, endpoints inclusive**
(`(Σ_{k=0..n} z_k)/(n+1)`, n = ceil(period/dt), sampled every step);
`n_ave` (default 4) **nested** passes; `equidistant_chunks=True` →
periods `linspace(T/2, T, n_ave+1)[1:][::-1]` (descending — staggers
the sinc-filter zeros, widening the stopband); `backward_forward`
runs a mirrored dt<0 pass after each forward pass (symmetrizes the
filter). Twin: deepcopy, **advection-off only** (Smagorinsky kept
running — see d5_2's parity note), writers off. Two old defects:
`max_period=None → inertial period` promised but crashes
(unimplemented); deepcopy grid sharing. Both dissolve.

Port:

```python
nh.transforms.TimeAverage(model, *, period=None, n_ave=2,
    equidistant=True, backward_forward=False, filter=fr.terms.linear)
```

- Owns `model.variant(term_filter=filter)` (law 3); the user's
  inviscid requirement = `filter=fr.terms.linear &
  ~fr.terms.owned_by(fr.closures.ClosureBase)`.
- `period=None` reads `coriolis.f0` via `model.parameters` (hinted
  error if absent) — implements the old promise.
- Passes: `update_parameters({TIME_STEP: ±dt})` (auto-rewarm covers
  the flip) → `reset()` → `set_state` → accumulate → normalize.
- **Accumulation it-1: host-side `advance(1)` loop** (identical to
  old cost; zero machinery; `chunk(1)` is lazily compiled anyway).
  Upgrade path in order: (i) a DIAGNOSTIC/AUX **accumulator module**
  in the variant (self_update maintains a carry-resident running
  sum; one `advance(n)` per pass, fully fused — this is why
  signatures must not be treedefs); (ii) the 2.6 capture stream.
  Constructor-agnostic.

## 3. OptimalBalance → nh.transforms.OptimalBalance (Tier 2)

Archaeology: four ramp maps ({fwd,bwd} × {to-linear,to-nonlinear});
shapes exp/pow/cos/lin evaluated **piecewise-constant at θ=n/N**
(never reaching Ro in-leg — `fr.Ramp`'s continuous stage-time
evaluation is an accepted, *better* numerical difference; cutover by
tolerance); convergence `2‖a−b‖/(‖a‖+‖b‖)`, stop on `err<tol` or
`err>prev` — **but z_res was already updated: old code returns the
worse iterate**; base point refreshed per iteration under
`update_base_point` (OBTA); `return_details` tuple return.

Port:

```python
nh.transforms.OptimalBalance(model, *, ramp_period, base=None,
    ramp="exp", max_it=3, tol=1e-9, update_base_point=True,
    filter=None, backward_filter=None)
```

- Two owned variants (fwd/bwd) with Ramp-valued `scaling.rossby`
  (t=(0,T) up / t=(−T,0) down) and `TIME_STEP: -dt` on bwd;
  **`filter=` threads to both** (the user's requirement);
  `backward_filter=` is the mset_backwards successor (recommended
  content: `~owned_by(ClosureBase) & ~implicit`).
- **Public pieces**: `ob.base`, `ob.backward_to_linear`,
  `ob.forward_to_nonlinear` (Propagators),
  `ob.ramp_cycle = forward @ base @ backward` — the reusable
  composed transform. Secondary maps (to-linear/to-nonlinear
  reversed ramps) retarget the same two models' Ramp endpoint leaves
  — sequential use, reset-prefixed; a documented law-3 refinement
  (exclusive ownership is per *preset*; exposed sub-transforms
  sharing the preset's models must not be driven concurrently).
- The exchange is affine FixedPoint policy:
  `Shift(z_base) @ (Identity − P) @ ramp_cycle` with per-iteration
  base refresh via the FixedPoint **factory form**.
- Divergence: `on_divergence="stop_best"` — a deliberate behavior
  change from old (documented; the old divergence test's expectation
  changes).
- Details via `call_with_info` (d5_1 ruling; the report's `ob.trace`
  attribute sketch is superseded).

## 4. NNMD → nh.transforms.NNMD (Tier 2, host driver)

Archaeology: needs all three eigenpairs + **the pointwise ω array**
(`1/ω` with zero guard — not the Symbol); `N(z)` = full-minus-linear
tendency computed by *toggling advection* between two
`tendencies.update` calls (i.e. it evaluates the composed tendency
on a state without stepping); the bilinear form
`S(z1,z2) = ½(N(z1+z2)−N(z1)−N(z2))` with an identity shortcut (old:
per-field allclose — port uses object identity); optional
`use_model` slow-derivative branch (full tendency + Euler step +
recursive sub-NNMD); `enable_dealiasing` declared, never implemented
— **dropped**.

Port: `NNMD(model, order=3, *, discrete=True, use_model=True,
time_step_factor=0.01, nonlinear=~fr.terms.linear, at_time=0.0)`;
`N(z) = model.variant(term_filter=nonlinear).tendency(z)` — one
evaluation replaces toggle-and-diff. **Math caveat to document
(+ possible lint)**: NNMD assumes N is quadratic; the default
`~fr.terms.linear` is only valid when all nonlinear terms are —
restrict via `nonlinear=owned_by(AdvectionBase)` otherwise.
Ramp-valued parameters: evaluate `tendency` at fixed `at_time`.
The recurrence/memo table ports mechanically.

## 5. Surface additions D5 asks of D4/2.7 (argued)

| # | Addition | Consumers |
|---|---|---|
| S1 | **`model.tendency(state, *, t=None, filter=None, constraints=True) -> State`** — host-callable jitted read-only wrapper over the composed tendency (implicit terms via forward apply; CONSTRAINT stages applied when asked; never advances the carry) | NNMD (N(z), use_model); **per-term budget diagnostics** (`filter=named(...)`); term unit tests vs analytic solutions; linear-stability matvecs; the future TangentPropagator (jvp of exactly this) |
| S2 | **`model.blank_state()`** + `model.state_space(name)` | every IC recipe (D1.1 never named the State factory the recipes build on), transforms constructing outputs |
| S3 | **`em.omega_field(s) -> Field`** (extends the flagged `omega_at`) | NNMD |
| S4 | `variant(updates=)` may change value **specs** (scalar→Ramp, dt sign) — assembly-time, unlike post-assembly update_parameters | OB |
| S5 | **Variant-verify lemma**: filters only remove terms ⇒ demands ⊆ parent ⇒ verify path passes by construction (variants never GridFrozenError) | all Tier-2 |
| S6 | **Signature ≠ treedef**: `(grid identity, PROGNOSTIC names+spaces subset)` | all compose checking |

Explicitly not needed: trajectory access as D4 surface (host loop /
accumulator / capture, all designed); `time_discretization_effect`
(no transform needs stepper-discretized ω — that stays recipe-side
per D2.4).

## 6. The workflow walk (acceptance test) — verdict: composes

`VorticalProjection(model)` ✓ (from_model validation; Ro-Ramp fine —
eigenmodes don't read Ro); `TimeAverage(model, filter=...)` ✓ (the
S5 lemma; gaps: variant memory, period=None needs f0, it-1 per-step
sync cost — all documented); `nh.initial_conditions.jet(model)` —
**gap: needs S2** (`model.blank_state()`; the old Jet's internal
GeostrophicSpectral becomes VorticalProjection); the residual ✓
(with the rest="zero" ruling the residual carries the full tracer);
`model.set_state(...)` ✓ (transform outputs PROGNOSTIC-subset by
law; the user's model untouched until this line).

## 7. Risks

The `rest` ruling needs sign-off; the OB divergence-return behavior
change (documented, test expectation changes); Ramp-vs-stepwise
ramping numerics (tolerance-based cutover, expected slight
improvement); backward-leg dissipation default (old kept closures
active backward — consider a warning when closures are active on a
backward variant); OB memory (2 carries + user's 1 — the
shared-internal-model alternative halves it but conflicts with
exposing independent leg Propagators; revisit if it bites); NNMD
quadraticity; Nyquist ownership confirmation (Phase-2.7, per D2.4
risk 4).
