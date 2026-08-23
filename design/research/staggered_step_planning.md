---
status: frozen
date: 2026-08-23
---

# The staggered (phased) baroclinic step — planning record

Research report (see [`README.md`](README.md) for status). Question
(owner, 2026-08-23): plan option (b) of
[`../plans/active/flow_following_coordinates_plan.md`](../plans/active/flow_following_coordinates_plan.md)
§5.1 — MITgcm's staggered order (momentum + barotropic solve first,
then diagnose the realized thickness change and the relative vertical
velocity from the post-solve transports, then advance tracers) — and
implement it if the planners converge. Two AI planners worked the
question in parallel from disjoint lenses, read-only: an
architecture/schedule lens and a numerics/discrete-consistency lens.
Both converged on the architecture; the numerics lens corrected the
premise. Their findings are condensed here; the decisions are in the
plan. FRIDOM file:line references are on `dev` at `48d36f7f`.

## Verdict

- **Architecture (converged).** Express the staggered step as a
  **phase axis on the existing schedule**, not a new stage kind, not a
  stepper wrapper, not a nested model: the canonical substage chain
  `S1 SELF_UPDATE → S1' DIAGNOSE → S2 terms → S3 advance → S3'
  ADVANCE → S4 CONSTRAINT` runs once per variable group inside the
  multistep steppers (`AdamBashforth`, `IMEXMultistep`), group 0 =
  momentum + the free-surface fields (`u, v, ps[, U, V]`), group 1 =
  tracers. Because S1/S1' re-run per substage, MITgcm's
  `INTEGR_CONTINUITY`/`CALC_R_STAR` between `SOLVE_FOR_PRESSURE` and
  `THERMODYNAMICS` falls out as "re-run SELF_UPDATE + DIAGNOSE after
  the solve". The partition is composer-derived from the dry-run
  write sets and the `advances=` claims; `d3_3_stage_schedule.md` §7
  item 6 reserved exactly this ("per-variable-group substages").
- **Numerics (premise corrected).** The *space* GCL is already exact
  in the tree (measured: `D_b(ż) − η̇ = 7.8e-16` uniform / `1.1e-16`
  stretched; flux-form constancy residual exactly `0.0`; the diagnosed
  column flux closes at the surface to `4.4e-16` and at the bottom
  exactly). The implicit/split-explicit defect is O(Δt) at tendency
  level → **O(Δt²) per step in the content — the same order** as the
  residual the intensive multistep formulation carries anyway (the
  ring mismatch `R1`, the Campin cross term `R2`, the product rule
  `R3`). Measured: 3.9e-6 for explicit and implicit alike. **Option
  (b) alone changes a constant, not an order.** Exactness needs (b)
  plus a formulation change; exactly two regimes deliver both
  constancy and conservation to round-off: **(E)** explicit free
  surface + extensive tracer (`J b`) + relative vertical flux, any
  stepper (AB3 accuracy retained); **(S)** implicit / split-explicit
  free surface + extensive tracer + a **single-level** tracer phase
  using the post-solve transports (MITgcm — transport first-order in
  time). No production model achieves "exact conservation AND
  multistep accuracy AND an implicit surface".
- **An O(1) hazard for FV + z\*** (new): under z\* the surface is
  permeable to the *absolute* contravariant flux (`Jω_top = η̇`), but
  the mapped FV advection closes the column with the Inner
  `flux_diff` (zero top flux, `advection.py:3789-3799`) while the
  ALE flux route uses the Outer closure reading the true wall mesh
  flux (`moving_geometry.py:768-779`). Consistent for a moving
  *rigid* wall (nonhydro2's morph); for a *material* surface the net
  is a spurious source `≈ η̇ b_top` — constancy breaks in the surface
  cell at O(η̇ b), not O(Δt). The nodal gates cannot see it
  (advective form). Two consistent closures: **REL/Inner**
  (recommended: the vertical advective flux is the relative one
  `w − ż`, structurally zero at both walls; the ALE flux route
  collapses to the pointwise `−(b/J) D_b(ż)`) or ABS/Outer (advection
  reads the true top flux with a one-sided reconstruction identical
  to the ALE term's — two large fluxes cancelling, fragile under
  biased schemes). "Inner on both" is **not** an option: the surface
  cell's snapshot continuity then misses the converging inflow and
  constancy breaks (`(b/J) η̇/Δz`).
- **Two exact realized-Δη identities** make (b) cheap: the implicit
  backward-Euler solve satisfies `ε (ps^{n+1} − ps^n) = −Δt g
  T*(u^{n+1})` exactly (same face depth, same C-grid difference in
  the Helmholtz operator and the correction — MITgcm's `exactConserv`
  for free); the split-explicit subcycle satisfies `Δps = −Δt g ∇·Ū`
  exactly with the *secondary* weights `Ū = (2/N) Σ_l β_l T_l`,
  `β_l = Σ_{m>l} ω_m` (ROMS's two-way averaging derived for FRIDOM's
  own `lax.scan`; one extra accumulator), the tracer phase then
  needing MPAS's `u^corr = (Ū − U)/H` on the transport only.
- **Side finding**: `ExplicitRungeKutta.step` calls
  `stages.constrain(stage_state, ctx_i)` without assigning the result
  (`runge_kutta.py:301`; `LowStorageRK3` does assign), so per-stage
  constraints have no effect on that stepper's stage states. Possibly
  deliberate; flagged for a separate look.

## 1. Architecture — the recommended design (lens A)

Three declarations, one loop, no new stage kind:

1. **`fr.model.Phases`** (new `model/phases.py`): a frozen hashable
   record. `Phases.staggered()` — role-derived: group 0 = the Velocity
   role ∪ every PROGNOSTIC claimed by an ADVANCE/CONSTRAINT stage's
   `advances=` (picks up `ps`, `U`, `V` without a name list); group 1
   = the rest. `Phases(("u", "v", "ps"), ("b",))` the explicit escape
   hatch. Resolved at assembly against the field table; stored on
   `Schedule.phases`; absent ⇒ one total phase.
2. **`Stage.phase: int | None`**: `None` = the kind's default —
   SELF_UPDATE / DIAGNOSE every phase (the per-substage refresh that
   is `CALC_R_STAR`); ADVANCE / CONSTRAINT with `advances=` the phase
   owning those names; an unclaimed CONSTRAINT (projection,
   `MaskState`, clamps) every phase its write set intersects
   (idempotent by construction); DIAGNOSTIC unaffected. The pin is
   load-bearing: the split-explicit `_snapshot_barotropic` must be
   `phase=0`.
3. **`TendencyTerm.per_phase: bool`**: a term whose write set
   straddles groups is an assembly error unless `per_phase=True`
   ("evaluate once per phase I touch, keep that phase's keys") —
   `Advection` (writes `{u, v, b}` in one term), `MeshVelocityCorrection`,
   `ThermalWind`. `StepContext` gains one static `phase: PhaseView |
   None` (`.index`, `.fields`), `None` on the unphased path so every
   existing trace is byte-identical; `Advection._advect` loops over
   `self._advected ∩ ctx.phase.fields`.

The loop, in `AdamBashforth.step` / `IMEXMultistep.step`
(statically unrolled over the static phases; `BoundSchedule.prepare /
tendency / advance_stages / constrain` take `phase=`):

```
for p in phases:
    ctx = stages.context(clock, dt, stage_dt, phase=p)      # pre-tick time for every phase
    st  = stages.prepare(st, ctx, phase=p)                  # S1 + S1'
    s_p = stages.tendency(st, ctx, phase=p)                 # S2, this group's terms
    inc = combine(s_p, ring, weights)                       # same row, same counter
    st  = st.add(**inc.select(*p.fields).components)        # S3, this group
    st  = stages.advance_stages(st, ctx.with_sums(s_p), phase=p)   # S3'
    st  = stages.constrain(st, ctx.with_sums(s_p), phase=p)        # S4 (barotropic solve, p = 0)
clock = clock.tick(dt)
ring  = merge(levels)[:-1]           # one full-width ring; the b slot holds the post-solve tendency
```

One saturating warm-up counter, one ring. IMEX: the implicit merge
groups (`friction` on `u, v`; `mixing` on `b`) land in different
phases; an operator straddling groups is refused ("coupled implicit
blocks are atomic under by-variable splitting", spec §5.1). RK and
the exponential family are refused via `supports_phases`
(`LowStorageRK3`/ETD: a phase loop inside each substage triples the
barotropic solve; designed-for). Rejected alternatives: a CONSTRAINT
stage integrating `b` itself (must re-implement the combine and
filter `Advection`'s joint term); a stepper wrapper (duplicates the
`dt` leaf → `update_parameters(TIME_STEP)` desync); `phases=` on the
stepper (the partition is a property of the field table).

Realized Δη for the geometry: `ZStarGeometry` gains one AUX field
`eta_prev` and two phase-pinned SELF_UPDATE stages — `phase=0`:
`eta_prev ← eta; eta = ps/g; eta_dot = −T*` (today's body);
`phase=1`: `eta = ps^{n+1}/g; eta_dot = (eta − eta_prev)/ctx.dt` —
MITgcm's `rStarDhCDt = (rStarFacC − rStarFacNm1C)/deltaTFreeSurf`
verbatim. `MeshVelocityCorrection` needs only `per_phase=True`.

Public API: `hy.Model(..., phases=fr.model.Phases.staggered())`, a
Model kwarg beside `term_filter=` / `allow_unadvanced=`. Taught
errors: phases under a stepper with `supports_phases=False`; a
straddling term without `per_phase`; a straddling implicit merge
group; a straddling `advances=` claim; an empty phase; per-phase
coverage and overlap lints. Backward compatibility: `phases=None`
reduces the loop to the literal current sequence (bitwise); the new
static fields join the schedule token and the restart fingerprint.
Effort ≈ 900 + 600 LoC framework, 200 + 400 hydrostatic; 4–6 focused
agent-days, risk concentrated in the two steppers' ring merge and the
parity proof.

## 2. Numerics — the exactness theorem (lens B)

With `h_k = J Δz_k`, `J = H + η` column-uniform for z\* (measured
variation `0.0`) and the map linear in η, one AB step of the
intensive scheme gives per cell

```
h^{n+1} b^{n+1} = h^n b^n + Δ(b δ)_k + Δt Σ_j c_j D^{n−j}_k
                + Σ_j c_j [D^{n−j}(h^n/h^{n−j} − 1)] Δt     (R1: ring mismatch)
                + [surface Δ(b δ) mismatch]                 (R2: Campin cross term)
                + Δh_k (b^{n+1}_k − b^n_k)                  (R3: product rule)
```

`R3` is removed by dividing the increment by `h^{n+1}` (MITgcm's
`FREESURF_RESCALE_G`); `R1`, `R2` survive while the ring holds
intensive tendencies and the mesh flux is single-level. Exactness
(closed domain) ⟺ every `D^{n−j}` telescopes with a zero surface
flux (the *relative* flux), **and** `Δh_k = Δt Σ_j c_j ḣ^{n−j}_k`
(the realized thickness change is the same weighted combination of
the per-level continuity residuals), **and** the prognostic is
extensive (or the increment is `h^{n+1}`-divided). Regime (E) meets
all three because `ps` advances with the same weights on the same
`T*`; regime (S) because a single post-solve level makes the weighted
combination trivial. The plain staggered intensive scheme — AB-
weighted transport with a constraint-realized `Δh` — keeps constancy
exact and conservation at O(Δt²)/step with a smaller constant.

What the tracer phase must use: the post-solve transports
`u^{n+1}, v^{n+1}` weighted by `J^n`; `Δη = (ps^{n+1} − ps^n)/g`
from a `ps^n` snapshot; the relative flux built bottom-up from those
two (surface value a structural zero); the increment divided by
`h^{n+1}`. The ALE transport for the realized `Δη` must **not** ride
the AB weights (the weighted combination of realized increments is
not the realized `Δh`); the correct FRIDOM spelling of a step-boundary
operation on the state is a stage (`CONSTRAINT`/`ADVANCE` claiming
the tracers) — which the phase axis makes expressible.

Expected fully-discrete residuals:

| scheme | constancy | conservation |
|---|---|---|
| today (nodal ABS, any FS) | 0 | O(Δx²) + O(Δt²), measured 3.9e-6 |
| FV ABS with mismatched wall closures | **O(η̇ b)** | **O(η̇ b)** |
| FV REL, intensive, any FS | 0 | O(Δt²)/step |
| FV REL, intensive + (b) + `1/h^{n+1}` | 0 | O(Δt²)/step, smaller constant |
| FV REL, **extensive**, explicit FS, AB3 | **0** | **0** |
| FV REL, **extensive**, implicit/split FS, single-level tracer phase | **0** | **0** |

Manufactured tests that pin the claims: `max|D_b(ż) − η̇| < 1e-15`
on uniform and stretched columns; `w_rel_top == w_rel_bot == 0.0`
exactly under REL; the FV wall-pairing gate (uniform `b`:
`Σ_k J(advection + ALE)_k Δz == 0.0` in every cell including the
surface cell — fails today); `|Σ J b(t_N) − Σ J b(0)| / |Σ J b(0)| <
1e-13` over ≥ 20 steps at 40 % displacement in the exact regimes
(1e-13, not a truncation bound); for intensive schemes the
order regression (halve Δt ⇒ drift ÷ ≥ 3.5); the split-explicit
secondary-weight identity `|Δps + Δt g ∇·Ū| < 1e-14·scale` and the
implicit `|ε Δps + Δt g T*(u^{n+1})| < 1e-13·scale`, both exact
algebraic identities testable without a run; the autodiff regression
for every new stage (`1/h^{n+1}` needs the `η > −H` validity note).

## 2b. Measured correction (FV family landed, 2026-08-23)

The FV hydrostatic core measured the §2 hazard: under z\* on FV the
surface-cell constancy residual is **exactly `0.0`** (the advection's
H7 surface closure `−q·A(1)` is constancy-preserving by construction
and the ALE bracket vanishes for uniform `f`), so the Inner/Outer
mismatch is **not** an O(1) constancy break. The two closures
disagree only in the *reconstructed* surface value, a conservation
residual `(b_face(0) − b_cell(0))·η̇` of O(Δz): the FV `∫J b` drift
is 6.5e-6 against 3.9e-6 nodal. P1 stands as the prerequisite for
the FV budget gates, downgraded from "correctness" to "consistency".
With `surface_flux=False` the predicted `η̇ b/(JΔz)` signature does
appear (2.4e-3 vs 5.6e-4 with the closure on).

## 3. Reconciliation

The lenses agree on the architecture (the phase axis) and on where
the realized `Δη` comes from. Lens B adds two prerequisites lens A
could not see: the FV wall closure must become consistent (REL/Inner)
before any FV z\* budget gate means anything, and the exactness goal
the owner stated for option (b) requires the extensive tracer (or the
single-level tracer phase) on top of the phase axis — a formulation
change across every `b`-writing module (`ale_on_fv.md` option D, "a
horizon" when written). The plan records these as staged work with
the owner calls they entail.
