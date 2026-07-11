---
status: normative
date: 2026-07-07
---

# Model layer redesign — Paper validation

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map.

Status: **complete and signed off (2026-07-08)** — four adversarial
paper-validation walks over the resolved design (D1–D5), briefed to
hunt gaps rather than confirm. Result: **no
design-cannot-express blockers**; 40+ findings triaged below. All
eight §8.6 decisions were **signed off as recommended** (with one
sign-off note: `eps` is order-2-only — order ≥ 3 warm-up uses
textbook AB2, a deliberate startup delta added to §8.8), and the
§8.6 + §8.7 amendments were **folded into the normative files**
(01_concepts D1.4/D2.4, 02_rules, 03 §5.2–5.5, 04 §6.1–6.6,
08 §10.3/10.5, classes/grid.md). Known residuals from
[`07_open_threads.md`](07_open_threads.md) were excluded by brief.

## 8.1 Nonhydro (the first port)

Walked: the full 3D assembly (core + Coriolis + stratification +
advection + Smagorinsky + tracer) through all nine steps; the 2D
(x,z) degenerate case; the linear-model/dispersion workflow; all 11
old examples; the parity audit.

- Assembly composes cleanly; the old `_set_module_list` ordering
  hack is the kind-ordering theorem in action; Smagorinsky's role
  targeting reproduces the old ENABLE_* sets exactly.
- **V-N1 (decision)**: no rule defines a `Velocity(label)` whose
  label is not a grid coordinate — the 2D slice's collocated `v`
  must NOT enter `div` while an A-grid collocated `u` must, so the
  space signal is provably insufficient; the d1_4 label-validation
  rule even contradicts D1.2's blessed 2D case. → the
  **transverse-component rule** (§8.6-2b).
- **V-N2 (decision)**: `Module.extra_halo` is unusable as specified
  — the assembly dry run traces every term and `HaloTracer.data`
  raises, so a `.data`-using module cannot be validated at all.
  → §8.6-4.
- **V-N3 (decision)**: the nh preset's default stepper is unpinned
  (old default AB3+eps vs D3's RK3 flagship) — an invisible parity
  delta in every ported example. → §8.6-6.
- Verified non-deltas: **eps-in-warmup is exact parity** (the old
  `update_coeff_AB` level-2 row IS the eps'd AB2 row);
  project-state ≡ project-tendency on divergence-free eigenmode ICs
  (P linear + idempotent), so discrete eigenmodes remain
  step-eigenfunctions of the linearized variant — keep the old
  "wave propagates at discrete ω" run as the 2.7 regression.
- Example sweep: all 11 expressible. Notables: `advection.disable()`
  examples → module omission in the preset; `wave_package`'s
  cross-grid IC → the documented `set_fields(u=field.data)` array
  hop; `PolarizedWaveMaker`'s eigenvector pattern → explicit-params
  duplication (the one place `update_parameters` staleness can bite
  silently — documented); Rayleigh–Bénard = `ConstantStratification
  (n2=0)` (keeps b→w, kills −N²w). Doc items: the solver-dsqr
  wording (bind the Symbol *structure*, read the live leaf in-step
  — D2.4's "(grid, self.dsqr) at assembly" phrasing corrected); the
  IC-recipe return contract (bare State; wave metadata via
  `em.omega_at` + `time_discretization_effect` as a documented
  two-liner); dimension-generic eigenmodes as a 2.7 obligation;
  Smagorinsky ν-sharing across its two terms (XLA CSE + a cost
  regression test).

## 8.2 Shallowwater + backward runs

Walked: the sw assembly (u, v, p; Sadourny; variable csqr); the
equatorial-waves example (beta plane, hand-built Hermite ICs, no
eigenmode seam); OB-on-sw; the backward-time sweep.

- Assembly and 2D pass design-wide (no hidden 3D assumptions;
  `Profile("y")` retires the old topo TODO). The equatorial example
  walks via `blank_state` + `create_field`; `from_model` correctly
  refuses the beta plane — with the surfaced naming rule that
  **providing a scalar implies constancy** (BetaPlaneCoriolis holds
  an `f0` leaf but must not provide `coriolis.f0`).
- **V-S1 (decision cluster — backward time)**: the run reduction,
  trigger lowering, and writer-resume truncation were all written
  with implicit dt>0. Fixes (applied, §8.7): steps =
  `ceil((end−t0)/dt − eps)` with precondition `(end−t0)·dt > 0`;
  trigger lowering in *step space* (`ceil((t−t0)/dt)` —
  sign-agnostic); `truncate_after` keys on the **iteration**
  coordinate. **dt-in-manifest** (§8.6-5): dt is in neither the
  snapshot (stepper ∉ carry) nor the fingerprint (leaf) — a
  backward run's restart silently restores −dt-built buffers into
  script dt; the manifest records provided-parameter values (at
  least `TIME_STEP`), `load_snapshot` errors on sign mismatch.
- **V-S2**: the backward Ramp must span `[−T, 0]` (reset → clock 0,
  elapsed decreases); naive endpoint reversal over `[0, T]` clips
  to a *constant*. `Ramp.reversed()` is defined as reflecting the
  time domain (applied).
- **V-S3**: Sadourny splits into a nonlinear self-advection term +
  a `linear=True` background-advection term (else `fr.linearize`
  drops the latter) — exposing the missing lint sentence: the
  coverage lint counts transports **per field across all terms**
  (incl. same-module), and a scheme's role-selected transport set
  excludes its name-coupled components (applied to D1.4).
- **V-S4 (decision)**: the Tier-2 transform presets are homed
  `nh.transforms.*` only; sw is a primary OB consumer →
  **`fr.transforms` homing** with a core-module default-projector
  hook, packages shipping thin aliases (§8.6 — applied as the
  recommendation, flagged for the class specs).
- Old-bug found: old Sadourny uses the *scalar* csqr in `h_full`
  while everything else uses the field — the always-field rule
  silently fixes variable-depth runs (pinned delta, §8.8). Lint
  noise: a linear sw run (no advection) warns 3× on untransported
  ADVECTED fields → one aggregated lint line (applied).

## 8.3 Hydrostatic (the stress test)

Walked: the composed CNAB2 step re-derived substage by substage;
advecting-flow selection; implicit mixing × the barotropic split;
warm-up/restart with the subcycle; z*-geometry ownership.

- The step composes; the depth-mean correction provably preserves
  the implicitly-mixed shear (depth-uniform shift); omitting `g∇η`
  from the slow sum is provably harmless (depth-uniform, lives in
  the overwritten depth mean); the tridiagonal z-LOCAL layout walk
  closes against the decomposition notes; the schedule is static
  through warm-up; mid-warm-up restart is bitwise (filter weights
  are static structure).
- **V-H1 — the diagnosed-w ruling (the parked D1 residual,
  resolved)**: `w` is **DIAGNOSTIC**, written by a core-owned
  **DIAGNOSE stage** (`w = −∫ᶻ∇·u dz`) — p_hyd's exact twin.
  Placement at S1′ is load-bearing: the first substage after any
  `set_state`/restart recomputes w before any term reads it.
  Documented rider: IO sees w(tⁿ) vs u,v(tⁿ⁺¹); the fix is a
  derived writer expression, never an epilogue refresh.
- **V-H2 (decision — amends resolved D1.4)**: with roles
  PROGNOSTIC-only, `table.velocity()` returns (u,v) and every
  generic scheme silently advects with a 2-component flow. →
  **Velocity-on-DIAGNOSTIC** (§8.6-2a).
- **V-H3 (decision)**: §5.5's write-gate table has no ADVANCE-stage
  row → gate = declared advanced PROGNOSTIC subset ∪ own AUX (the
  declared home of stage-owned cross-step integrator state).
- **V-H4 (decision)**: the barotropic slow forcing gets the
  **increment-form default** `G = ∫(X* − Xⁿ)dz/dt` (MOM6/ROMS
  practice) — automatically CNAB2-weighted and warm-up-consistent,
  and it includes the implicit increment by construction; the raw
  ctx-sums variant becomes the constructor knob.
- **V-H5 (decision)**: the self_update scheduling trigger gains a
  `reads=("eta",)` slot for **state-derived** AUX (the
  time-dependent-input rule was Ramp-shaped and would never
  schedule geometry-follows-eta).
- **V-H6 (gap dissolved)**: z*-geometry ownership is *sanctioned as
  written* — the owner-leaves-only rule governs `default=` closures
  only and itself names self_update as the cross-module route; the
  recommended shape is a separate `hs.ZStarCoordinates` module
  (geometry not welded to one barotropic treatment).
- Doc items (applied): `model.tendency()` runs SELF_UPDATE +
  DIAGNOSE at `t` before evaluating terms; module-owned
  ADVANCE-stage integrator statics (substeps, filter spec) join the
  restart fingerprint.

## 8.4 Coupled models + multi-device + lifecycle

Walked: a two-model coupled run; 4↔1-device restart portability;
device-id permutation; adversarial sweeps (idiom B with Ramps;
module swaps on a frozen grid); transform × lifecycle interleaving;
the io=/outputs= seam.

- **V-C1 (decision — the one true blocker found)**: §6.6's
  "exchanged data enters as coupler-owned AUX" has **no write
  path** — set_fields is PROGNOSTIC-only, set_state ignores AUX,
  self_update can't see the other model. → **`model.set_aux(...)`**
  (§8.6-1): host-side, chunk-boundary-only, legal iff the owning
  module declared consent on the FieldDeclaration
  (`host_writable=True`) — the declaration is the owner's sanction,
  parallel to `default=` closures; **no rewarm by default**
  (exchange data is forcing; re-ramping every coupling window would
  degrade coupled AB3 permanently).
- **V-C2 (decision)**: the FPlane→BetaPlane swap on a frozen grid
  raises `GridFrozenError` (`Profile("y")` ∉ the recorded set), and
  module-swap demand sets are not nested. → document module-type
  sweeps as fresh-grid-per-composition (free at the step-cache
  level — different assembly records anyway) + optionally relax
  verify to *satisfiability* (adopt zero-halo-demand
  ConstantSpace-family spaces into the record) (§8.6-3).
- **V-C3 (applied)**: the fingerprint's "spaces" is pinned to
  **declared patterns / bare spaces** — never `Layout`, negotiation
  fingerprints, or device topology (else 4→1-device restore
  silently breaks). Device-id permutation restores correctly *by
  design* (no device identity anywhere in the source record) — CI
  cases: 4→1, 1→4, permuted device_ids.
- Regrid ownership noted for 3.2: grid-pair-bound → a free-standing
  bound operator (the Fourier pattern), built after both freezes,
  held by the host coupler; outputs re-homed like set_fields inputs.
- Interleaving walks clean (variant timing vs caches; two
  transforms per model; accumulator-twin set_state acceptance via
  signature ≠ treedef) — with doc items applied: the accumulator
  upgrade is a *fresh assembly*, not a variant (law-3 rewording +
  the reuse-existing-spaces rule); io binding split (name
  resolution at assembly, store creation at run start); stream
  dedupe by resolved path (`IOCollisionError`); `Snapshots` is
  run-config only; canonical idiom-B order
  (`update_parameters → reset → set_fields`); cross-transform
  frozen-config inconsistency; coupled-clock ulp drift (derive one
  dt exactly from the other); set_state partial-input behavior
  (missing components = leave untouched, documented).

## 8.5 Time-dependent geometry

Covered inside the hydrostatic walk (V-H5/V-H6): the ramped/
eta-following metric-field pattern is sanctioned by the existing
rules once the `reads=` trigger lands; the aux metric fields enter
halo negotiation through the traced tendency (confirming the
designed-for walk's demand); the fast substeps' 2D thickness stays
subcycle-internal (3.1 numerics, d3_4 risk 5).

## 8.6 Sign-off decisions (consolidated)

1. **`model.set_aux(**{name: value})`** — the coupler write path
   (V-C1): host-side, boundary-only, declaration-consented
   (`host_writable=True`), re-homed like set_fields inputs,
   **rewarm=False default** for exchange-class writes.
2. **Two amendments to D1.4's Velocity role**: (a) `Velocity` (and
   only Velocity) may be declared on DIAGNOSTIC fields — role-driven
   *reads* (advecting flow, CFL) span both lifecycles, role-driven
   *write-targeting* (friction) intersects PROGNOSTIC (physically
   right: diagnosed w has no momentum equation); ADVECTED/TRACER
   stay strictly PROGNOSTIC; U,V stay role-free. (b) **the
   transverse-component rule**: on tensor grids, a Velocity label
   absent from `grid.names` marks a transverse (slaved) component —
   excluded from divergence/gradient/advective-flux *directions*
   (its ∂ ≡ 0), full member of the family for friction/CFL/energy
   and as an advected quantity; assembly validation warns (not
   errors) on label-vs-staggering contradictions.
3. **The frozen-grid verify relaxation** (V-C2): document
   fresh-grid-per-composition for module-type sweeps; optionally
   adopt satisfiability-verify (zero-new-demand spaces adopted into
   the record) to dissolve the R2-family trap.
4. **`extra_halo` mechanics** (V-N2): the declaring module's terms
   are exempted from the *halo* trace (replaced by the declared
   spec) while key/write-gate validation runs on a second dry-run
   mode over real zero-fields; the "custom pointwise Operator
   (halo 0)" pattern is named in 02_rules as the closure-author
   escape, with the `Where`-condition construction specified.
5. **dt in the snapshot manifest** (V-S1): provided-parameter
   values (at least `fr.params.TIME_STEP`) recorded;
   `load_snapshot` errors on sign mismatch, warns on magnitude.
6. **The nh preset's default stepper** (V-N3): pin
   `AdamBashforth(order=3)` at cutover for parity — note the
   sign-off ruling that **eps is order-2-only** (order ≥ 3 warm-up
   uses textbook AB2) — with low-storage RK3 as the documented
   recommendation. *(Signed off.)*
7. **The hydrostatic package** (V-H1..H5, one bundle): diagnosed-w
   as DIAGNOSTIC + DIAGNOSE stage; the ADVANCE-stage write gate; the
   increment-form slow forcing default; the self_update `reads=`
   trigger. (All greenfield — no old behavior at stake.)
8. **`fr.transforms` homing** (V-S4): Tier-2 presets live in
   `fr.transforms` with a core-module default-projector hook;
   packages ship aliases.

## 8.7 Doc-tier amendments (accepted and **applied** with the
§8.6 outcomes, 2026-07-08 — see the amended passages in
01_concepts, 02_rules, 03, 04, 08, and classes/grid.md)

Backward-time trio (run reduction precondition; step-space trigger
lowering; iteration-keyed `truncate_after`); `Ramp.reversed()` =
time-domain reflection, signed endpoints; the per-field coverage
lint sentence + name-coupled exclusion; provides-implies-constancy;
the solver-dsqr wording fix; `model.tendency` runs
SELF_UPDATE/DIAGNOSE first; stage-integrator statics in the
fingerprint; fingerprint "spaces" = bare/declared; io
binding/collision/Snapshots-run-only rules; accumulator-twin
law-3 rewording + space-reuse rule; canonical idiom-B order;
cross-transform staleness note; coupled-clock dt derivation;
set_state partial-input = leave untouched; the aggregated
linear-run lint line; the AUX-user-content constructor pattern
(values/callables consumed by the declaration default); the
eigenmode-seam generalization paragraph (per-family parameter sets
+ degeneracy predicates); the IC-recipe return contract
(bare State + the omega/period two-liner); the two-grid IC idiom.

## 8.8 The consolidated cutover-parity list (2.7)

All parity claims below sit under the bitwise-equality umbrella
rule ([`02_rules.md`](02_rules.md)): bitwise means
identically-compiled paths; anything across compilations is
tolerance-based.

Project-the-state (exact-equivalent for explicit schemes);
`p = φ/stage_dt`; continuous Ramp vs piecewise-constant θ=n/N;
`stop_best` (old returned the diverged iterate); TimeAverage twin
drops Smagorinsky (old kept it); Sadourny reads the csqr *field*
(old used the scalar in h_full — old bug); `rest="zero"`;
empty-implicit CNAB2 = textbook AB2 (no eps);
`epot` N²=0 → hinted error (old silent formula switch); Nyquist
zeroing relocated into em.q/p; seeded-random ICs not bitwise
(per-DOF fold_in); float64 clock (forcing phases vs old float32);
netcdf→zarr + trigger snap-up + `every()` includes step 0; NaN
abort at chunk boundary; backward runs via signed dt (no
`run_backward`); **order ≥ 3 AB warm-up uses textbook AB2** (no
eps — sign-off ruling; one-step startup delta, tolerance-based);
**verified parity, not deltas**: discrete-eigenmode dispersion
(regression: one-period wave propagation at discrete ω); the AB2
eps'd row itself (order=2 runs are bitwise on the eager path;
jitted, tolerance-based — umbrella rule,
[`02_rules.md`](02_rules.md)).
