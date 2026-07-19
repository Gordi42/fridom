---
status: normative
date: 2026-07-13
---

# Model layer redesign — Open threads

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map.

Status: **live list.** All ten design threads are resolved, and the
model layer is implemented (`fridom.model`, ROADMAP 2.2–2.8). This
page is the map from each former thread to the section that now owns
its decision, followed by the residuals that are *still* open — the
only part of this page that is work.

## 9. Open threads (resolved — where the decisions live)

| # | Thread | Resolution lives in |
|---|--------|---------------------|
| 1 | **D1 — field registration** | [`01_concepts.md`](01_concepts.md) D1; classes in [`classes/declarations.md`](classes/declarations.md). Landed: `FieldDeclaration`, `Lifecycle`, `FieldReference`, `SpacePattern`/`SpaceRule`, roles. |
| 2 | **D2 — parameter ownership** | [`01_concepts.md`](01_concepts.md) D2; [`classes/declarations.md`](classes/declarations.md). Landed: `ParameterDeclaration`/`ParameterReference`/`Param`/`USE_PROVIDED`, `fr.params` names, `Ramp`/`TimeDependent`, `model.parameters`. |
| 3 | **D3 — step abstraction** | [`03_time_stepping.md`](03_time_stepping.md) §5.1–5.9; [`classes/time_steppers.md`](classes/time_steppers.md). Landed: terms + treatments, the five-kind stage schedule, `StepContext`, AB/RK/IMEX, `Clock`. |
| 4 | **D4 — composition / lifecycle** | [`04_run_loop_io.md`](04_run_loop_io.md) §6.1–6.9; [`classes/model.md`](classes/model.md), [`classes/io_ops.md`](classes/io_ops.md). Landed: the assembly pipeline, `advance`/`run`/`fr.ops.Session`, the chunked scan, IO and snapshots. |
| 5 | **Eigenmode objects** | **Closed by 2.7.** The `omega`/`vec_q`/`vec_p` successors ship as a two-tier, State-valued, parameter-consuming surface: `fr.model.eigenbasis(model)` dispatching the analytic tier and the numeric tiers (`numeric_eigenpairs`, `channel_eigenpairs`), with `projector(sel)`, `function(f, sel)` and `mode(...)` returning `StateTransform`s / states, and `EnergyMetric.from_model` supplying the metric. They consume `(model, parameters)` exactly as D2 foresaw. |
| 6 | **Initial conditions** | **Closed by 2.7.** ICs are *post-assembly user code on the assembled state* — `grid.create_field(space, init=...)` composed into a `VectorField`, with per-package recipe libraries (`nh.initial_conditions`, `sw.initial_conditions`) and the eigenmode surface supplying spaces and mode states. `init=` in declarations was **rejected**: `FieldDeclaration.default` exists only for module-owned AUXILIARY/DIAGNOSTIC allocation, never for prognostic ICs. Restart overwrites ICs (the fingerprint deliberately ignores IC leaves, [`02_rules.md`](02_rules.md)). |
| 7 | **Grid-notes amendments owed** | All paid: the `State` parameter phrasing and the three D1 field items (`VectorField.add`, functional-only ports, the raising `data` setter) in [`../grid/classes/fields.md`](../grid/classes/fields.md); the dispatch-merge call site in [`../grid/classes/grid.md`](../grid/classes/grid.md) (D4 assembly step 3). |
| 8 | **D5 — the state-transform algebra** | [`08_state_transforms.md`](08_state_transforms.md) §10.1–10.8; [`classes/transforms.md`](classes/transforms.md). Landed: `StateTransform` + algebra, `model.variant`, the projection family, `Propagator`, `TimeAverage`, `OptimalBalance`. The NNMD descoping held only until its own rewrite: `fr.transforms.BalanceExpansion` now ships ([`../nnmd/nnmd_design_note.md`](../nnmd/nnmd_design_note.md)) — as a transform over the eigenbasis, still with **no model propagator** inside it. |
| 9 | **Validation-walk findings** | [`06_validation.md`](06_validation.md) §8.6–8.8; all eight decisions signed and folded into the normative files. |
| 10 | **Coupling design-for** | [`09_coupling_designfor.md`](09_coupling_designfor.md); CS-1..18 final and carried by the class specs. |
| 11 | **Phase-1 reconciliation** | Per-step sync amplification: **both halves closed** — the model half by the sync-policy-neutral D3 term surface, the grid half by the consumption-side sync strategy (ROADMAP 1.8, shipped; contract in [`../grid/classes/decomposition.md`](../grid/classes/decomposition.md)). CS-17 precision: global-precision-only. Metadata-in-treedef: the annotation-exempt equality amendment plus `FieldTable.subset`. Bitwise umbrella rule: [`02_rules.md`](02_rules.md). |

Threads 1–4 and 8 were each resolved with a research report set
(d1_*–d5_*, [`research/`](../../research/README.md)); the reports are
frozen inputs, not normative text, and are the place to look before
re-arguing a decision.

## 9.1 Still open

Genuinely unresolved, in rough order of how much they bite:

1. **The State factory — RESOLVED (2026-07-19): BUILT.**
   [`04_run_loop_io.md`](04_run_loop_io.md) §6.1 and
   [`08_state_transforms.md`](08_state_transforms.md) S2 promised
   `model.blank_state()` and `model.state_space(name)`; both now ship
   on the Model surface. `blank_state()` returns the PROGNOSTIC subset
   at declared defaults (born sharded); `state_space(name)` returns the
   named component's function space. Each is one line of sugar over the
   field table (`model.field_table[name].space` and
   `grid.create_field`), which stays the equivalent low-level spelling.
2. **Accumulation — RESOLVED (2026-07-19): the idiom stands, the
   preset is DROPPED.** The S6 DIAGNOSTIC accumulation idiom is the
   sanctioned, normative spelling ([`02_rules.md`](02_rules.md)). The
   `fr.modules.WindowAccumulator` preset that several notes once named
   as shipping is not promised; a windowed-accumulation preset can be
   introduced with coupling if wanted (far future,
   [`09_coupling_designfor.md`](09_coupling_designfor.md)). The idiom
   needs no preset to be complete.
3. **`add_prognostic` — RESOLVED (2026-07-19): STRUCK.** The
   key-aligned add of a PROGNOSTIC-only vector onto a full state is
   served by `state.add(**components)`, which every stepper family's
   combine step already uses; that is the final spelling. No separate
   `add_prognostic` method is added.
4. **`em.omega_at(k, s)` / `em.omega_field(s)`** — scalar and
   field-valued frequency accessors. The eigenbasis exposes the batched
   `omega` array and `mode(...)`'s scalar; neither accessor is built.
   `omega_field` waits on a consumer (`function(f, sel)` makes it a
   two-liner).
5. **Backward-run sign conventions through `stage_dt`** — RK stage
   shifts, warm-up rows, the `p = φ/stage_dt` projection normalization
   and increment-form forcing must all read the signed dt coherently on
   a backward leg. The named regression test is still owed.
6. **Backward-dissipation warning** — should assembly warn when
   closures survive onto a sign-flipped (`TIME_STEP < 0`) variant? The
   old code kept them active. Decide at the OB port review.
7. **DIAGNOSTIC-chain ordering** — explicit `Stage.order=` ships;
   topological sort by declared reads/writes is the recorded upgrade,
   to be revisited only if real dependency chains appear.
8. **NaN-check cadence** — per-step S5 reduction is the shipped
   default *pending benchmark*; a cadence knob (every-k, or
   chunk-boundary-only) is sanctioned if profiling demands it.
9. **Shared-jitted-runner discipline** — the assembly lint for
   unhashable statics and the compilation-count regression test are
   still implementation obligations.
10. **Multi-process walltime / interrupt consensus** — a rank-0
    consensus broadcast at chunk boundaries (walltime stop, Ctrl-C,
    resubmit-once); deferred to 3.2/3.3 with the multi-host snapshot
    follow-up.
11. **Post-assembly writer attach + capture streams** — both are
    designed-for behind the `OutputStream` protocol; neither contract
    changes, only the spelling is owed.
12. **JVP linear-tag lint** — `fr.linearize` consumes the `linear`
    tag; the lint that keeps the tag honest is still unbuilt.
13. **`fr.at(times=)` under a later dt change** lands on different
    realized steps across resubmits; plan-time logging plus the
    realized times on the axis mitigate. A tolerance warning is a
    possible nicety.

Parked designed-fors (listed so the surface does not preclude them, no
consumer yet): unstructured-factor `SpacePattern` tags (`EDGE_NORMAL`);
the multi-velocity `group=` qualifier; `fr.terms.where(fn, token=)`,
glob support in `named`, stage filtering in variants; `OnComponents`,
`Transform.resync()`, `T.with_parameters(...)`, batched Tier-2
ensembles, a traced `lax.while_loop` `FixedPoint`, field-valued algebra
coefficients; adaptive-stepper controller state; `SpectralDiagonal`
implicit operators.
