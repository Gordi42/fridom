# Model layer redesign — Open threads

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map.

Status: **live list.** Threads move out as decisions land in the
numbered sections; resolved threads keep a one-line pointer, the
grid-notes convention.

## 9. Open threads

1. **D1 — field registration**: **resolved 2026-07-07**
   ([`01_concepts.md`](01_concepts.md) D1; research reports in
   [`research/`](research/README.md)). Residual sub-points, to be
   picked up where noted:
   - unstructured-factor `SpacePattern` tags (`EDGE_NORMAL` — a
     multi-name mesh factor breaks the name-keyed premise;
     designed-for, `SpaceRule` covers it meanwhile);
   - typo'd coordinate names in patterns (silent non-match is the
     dimension-generality feature; mitigation: assembly logs the
     `name → pattern → resolved space` table, optional
     `require=("x",)` on patterns — decide with D4 error
     reporting);
   - `state_type`-on-Module mechanics under jaxify (fallback: a
     `Model(state_type=...)` kwarg);
   - `DIAGNOSTIC` vs module-private storage rule (does `div` get
     declared or stay private to the projection?);
   - multi-velocity futures (`table.velocity()` ambiguity under a
     split-explicit barotropic pair / coupling; possible `group=`
     qualifier — designed-for, don't build);
   - `set_fields` mutating vs functional spelling (owned by D4's
     Model-pytree decision);
   - 02_rules entries owed: defaults evaluated at allocation with
     assembly-time parameters; AUXILIARY fields excluded from
     tendency allocation but included in halo negotiation;
     `SpaceRule` purity contract; declared-space resolver overrides
     are grid-level only.
2. **D2 — parameter ownership**: **resolved 2026-07-07**
   ([`01_concepts.md`](01_concepts.md) D2; research reports in
   [`research/`](research/README.md)). All D1
   fed-forward requirements are satisfied by the resolution
   (ParameterReference mirrors FieldReference; auxiliary-field
   parameters; host-reachable `model.parameters`; Ramp composes
   with resolution and stage times). Residual sub-points after
   sign-off:
   - the explicit-wins mechanism for constructor-set values
     suppressing a declared reference (`USE_PROVIDED` sentinel
     proposal — API-sketch item);
   - the `model.update_parameters(...)` re-materialization
     lifecycle hook (owned by D4);
   - restart-fingerprint treatment of parameter specs (Ramp shape =
     structure, endpoints = leaves — amend the D1 restart rule);
   - `em.omega_at(k, s)` scalar accessor, `em.projector`
     composition semantics, non-Fourier eigenmode families
     (Phase 2.7 surface);
   - 02_rules entries owed: dotted-name lint; provided-parameters-
     must-be-dynamic; in-trace-field-authoritative vs host-side-
     scalar rule; diagnostics `with_metadata` rule; default-value
     policy (physically-identity values only); framework-level
     consumers consume models, not parameter names.
3. **D3 — step abstraction**: **resolved 2026-07-07** (full design
   in [`03_time_stepping.md`](03_time_stepping.md) §5.1–5.9; summary
   in [`01_concepts.md`](01_concepts.md) D3; research reports
   d3_1–d3_4 in [`research/`](research/README.md)). All seven D1/D2
   fed-forward requirements discharged (scorecard in d3_3); the
   D1 "(p, div)" amendment and the consolidated restart-fingerprint,
   float64-clock, lagged-coefficient, and diagnostics-metadata rules
   were **applied on sign-off** (01_concepts, 02_rules). Residual
   sub-points, picked up where noted: explicit-`order=` vs
   topological sort for DIAGNOSTIC chains (revisit if real chains
   appear); `add_prognostic` key-aligned add (fields.md follow-up,
   with `VectorField.add`); backward-run sign conventions through
   `stage_dt` (test at 2.7); optional initial projection of
   non-divergence-free ICs (D4/IC); stale multistep buffers under
   `update_parameters` (D4); chunked-scan buffer donation
   (implementation).
4. **D4 — composition/lifecycle**: **resolved 2026-07-08** (full
   design in [`04_run_loop_io.md`](04_run_loop_io.md) §6.1–6.9;
   summary in [`01_concepts.md`](01_concepts.md) D4; research
   reports d4_1–d4_4 in [`research/`](research/README.md)); the
   NaN-check cadence is default-per-step *pending benchmark* (2.4
   item). The queued amendments were **applied on sign-off**: the
   grid-notes merge-call-site + frozen-grid verify path
   (`../classes/grid.md`, `../02_rules.md` §3.4 — the last
   grid-note debt is now paid), the D1.1 softening (AUX default
   closures retained; unbound-method defaults), and the three
   02_rules entries. **All six D3 fed-forward requirements are
   discharged**
   (chunked scan confirmed with trigger-derived boundaries; the NaN
   mechanism resolved as flag + boundary abort, cond wrapper
   rejected; assembly composes the scan body with the dry run at
   step 6; `update_parameters` re-ramps by default; `io=` replaces
   `diagnostics=`; `apply_constraints()` is the initial-projection
   opt-in). Residual sub-points after sign-off (§6.9):
   `state_type`-under-jaxify; donation vs live `model.state` views
   (copy-on-read benchmark); the GPU-conditional and S5-fusion cost
   claims (benchmark in 2.4); the shared-jitted-runner discipline
   (assembly lint + compilation-count regression test);
   post-assembly writer attach + capture streams (2.6);
   multi-process walltime/interrupt consensus (3.2/3.3);
   `truncate_after` for CSV. **Amendments owed on sign-off**: the
   grid-notes merge-call-site amendment + frozen-grid
   fingerprint/verify path (`../classes/grid.md`, `../02_rules.md`
   §3.4 — the last grid-note debt); D1.1's "declarations discarded"
   softened (AUX default closures retained in the assembly record;
   `default=` accepts unbound owner methods); 02_rules entries
   (owner-leaves-only defaults + same-path rule; "no pickled
   models"; fingerprint-ignores-IC-leaves).
5. **Eigenmode objects**: the `omega`/`vec_q`/`vec_p` successor —
   State-valued, parameter-consuming assembly built from `Symbol`s;
   depends on D2 for its parameter feed. *Open (deferred by the
   grid notes to this design).*
6. **Initial conditions**: user code on the assembled state vs.
   `init=` in declarations vs. IC modules; interaction with
   restart. *Open.*
8. **D5 — the state-transform algebra**: **resolved 2026-07-08,
   NNMD descoped** (its future rewrite is a separate design exercise
   and will not contain a model propagator; the d5_3 archaeology is
   retained for it). Full design in
   [`08_state_transforms.md`](08_state_transforms.md) §10.1–10.8;
   research reports d5_1–d5_3. The behavior deltas
   (`rest="zero"`, `on_divergence="stop_best"`, the TimeAverage
   Smagorinsky parity delta) are signed off; the amendments were
   **applied on sign-off**: `model.tendency` /
   `model.blank_state()` / `state_space()` / `variant()` added to
   the D4 Model surface (04_run_loop_io §6.1), the verify-⊆ wording
   in the grid notes, and the SELF_UPDATE-before-consumers
   invariant in 02_rules. `em.omega_field` deferred with the NNMD
   rewrite. Residuals (§10.8): backward-dissipation warning (decide
   at the OB port), `Ramp.reversed()` spelling (API sketch), JVP
   linear-tag lint priority (2.5), OB memory, and the designed-fors.
9. **Validation-walk findings: resolved and applied (2026-07-08)**:
   four adversarial walks, consolidated in
   [`06_validation.md`](06_validation.md); all eight §8.6 decisions
   **signed off as recommended** (`set_aux`; the two Velocity-role
   amendments; the satisfiability verify relaxation; `extra_halo`
   mechanics; dt/provided-params in the snapshot manifest; nh
   preset = `AdamBashforth(order=3)` at cutover — **eps ruled
   order-2-only** at sign-off, order ≥ 3 warm-up is textbook AB2;
   the hydrostatic bundle; `fr.transforms` homing), and together
   with the §8.7 doc amendments **folded into the normative files**
   (01_concepts, 02_rules, 03, 04, 08, classes/grid.md). The
   §8.8 cutover-parity list feeds the 2.7 test plan. Remaining
   from the walks: only items already parked with owners
   (backward-dissipation warning → OB port; dimension-generic
   eigenmodes + `em.omega_at` → 2.7 surface; the named regression
   tests → 2.7).
10. **Coupling design-for: fully resolved 2026-07-08** —
   [`09_coupling_designfor.md`](09_coupling_designfor.md) (research
   c1–c3). All three decisions signed and **applied**: the
   windowed-accumulation bundle (lifecycle-polymorphic
   `host_writable` + the S6 accumulation idiom + D2.3 re-aimed —
   02_rules, 01_concepts, 03 §5.5), the re-materialization
   exemption + three-operation matrix (02_rules, 04 §6.5), and the
   **hybrid `advance`/Session surface** (3a typed
   AdvanceResult/PanicError + 3b `fr.ops.Session` over normative
   protocols, `run()` reimplemented on top — 04 §6.3). The
   CS-1..18 constraint list is final and carried by the class-spec
   briefs; §11.5's non-promises stand.
11. **Phase-1 reconciliation: resolved 2026-07-08** — the
   grid-implementation audit against these notes, with Silvano's
   rulings applied:
   - **Per-step sync amplification**: the model half is
     **discharged** — literal "tendencies are operators" is rejected
     (the signed D3 term surface; and the landed `OperatorSum` would
     not fuse syncs anyway — it syncs per term plus per pairwise
     addition); the `(self, state, ctx) -> dict` signature is
     sync-policy-neutral, so `../classes/decomposition.md`'s "decide
     before the update signature is fixed" deadline is met. The
     grid half is now also **decided** (owner sign-off 2026-07-08):
     the sync strategy is redone to consumption-side sync with
     trace-time halo-validity tracking — ROADMAP task 1.8; decision
     record in
     [`../classes/decomposition.md`](../classes/decomposition.md)
     open questions, work item 8 in
     [`../phase2_grid_followups.md`](../phase2_grid_followups.md).
     Results-neutral swap; nothing model-side waits on it.
     Note: `VectorField.add` will be the hottest sync site (one
     exchange per term per component) — flagged for the
     [`../classes/fields.md`](../classes/fields.md) follow-up.
   - **CS-17 precision**: resolved **global-precision-only** —
     no per-space width axis; accumulator roundoff covered by the
     S6 chunk-cadence host-side float64 accumulation
     ([`classes/declarations.md`](classes/declarations.md) open
     question 3; ruling 2026-07-08).
   - **Metadata-in-treedef**: resolved — the annotation-exempt
     equality amendment in
     [`../classes/fields.md`](../classes/fields.md); model.md's
     FieldTable-aux proposal rejected in favor of
     `FieldTable.subset` (see [`classes/model.md`](classes/model.md)).
   - **Bitwise umbrella rule adopted**
     ([`02_rules.md`](02_rules.md)): bitwise claims compare
     identically-compiled paths only; the cutover "order=2 bitwise"
     claim downgraded to eager-path-only (06 §8.8).
7. **Grid-notes amendments owed**: the dispatch-merge call site
   (D4 step 3); the `State` "model settings/parameter objects"
   phrasing in [`../classes/fields.md`](../classes/fields.md)
   (after D2); and — from D1, apply on sign-off —
   `VectorField.add(**contributions)` joins the it-1 surface,
   fields.md open question 3 closes (fully functional ports;
   raising teaching-shims only), and `ScalarField.data` gains an
   explicit raising setter with guidance — **the three D1 items were
   applied to `../classes/fields.md` on 2026-07-07**, and the
   "model settings/parameter objects" phrasing was **amended with
   D2's sign-off** (module-owned parameters; State holds none).
   *Remaining: the dispatch-merge call site (D4).*
