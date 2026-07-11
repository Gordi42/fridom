---
status: frozen
date: 2026-07-07
---

# D3.3 — The stage schedule: kinds, ordering, traced signatures

Research report (see [`README.md`](README.md) for status).

## 1. Step anatomy; project-the-tendency vs project-the-state

**What the old code does, precisely**: the pressure trio lives
*inside* `MainTendency` — `TendencyDivergence` computes
`div(dz.velocity)`, the solver inverts the dsqr-scaled Laplacian,
`PressureGradientTendency` subtracts `grad p` from `dz`. The old
scheme **projects the tendency**: every tendency the stepper sees is
already divergence-free. Physically exact continuous-in-time
(`∇²p = ∇·F` gives the instantaneous physical pressure). It ran once
per AB step and once per RK internal stage (both steppers call the
full MainTendency per tendency evaluation).

**Decision: project-the-state, once after every state-producing
advance** (each RK internal stage and the final combination):

- **Equivalence** for the old use cases: with P the discrete
  projector (idempotent — the solver's k² is built from the same
  discrete operators as div/grad), explicit-RK stage states and the
  AB update coincide exactly between the two schemes when the state
  starts divergence-free (linearity of P). Cutover parity is safe.
- **Self-correction**: project-the-tendency *preserves* IC/restart/
  roundoff divergence forever; project-the-state removes it within
  one substage — retiring a real footgun under D1.1's free-form
  `set_fields`.
- **IMEX composes** (load-bearing): an implicit mixing solve does
  not preserve divergence-freeness; the constraint must act on the
  state after the solve — the standard fractional-step ordering.
  Project-the-tendency has no coherent IMEX story.
- D1.5's resolved text already committed to the state-acting form.

**Pressure normalization**: solve `∇²_dsqr φ = ∇·u*`, correct
`u ← u* − ∇φ` (w-component /dsqr), write the diagnostic
`p := φ / stage_dt` — recovers the physical pressure to O(dt).
**Precedent verified from Oceananigans source**
(`pressure_correction.jl`, `runge_kutta_3.jl`): correction inside
every RK3 substep on predictor velocities, `pNHS` normalized by Δt
so the stored diagnostic is the physical pressure — exactly this
design. (Dedalus: constraint-in-solver family; MITgcm: Chorin-style
project-state.)

**Canonical step**:

```
for each stepper substage i (incl. separate final combination):
  P0  ctx_i: eval_params(modules, t_i); clock at t_i; dt, stage_dt_i
  S1  SELF_UPDATE stages          (scheduled modules; replace own AUX)
  S2  tendency terms (EXPLICIT)   (contribution dicts -> composer -> add)
  S3  stepper advance math        + IMPLICIT solves per the scheme
  S4  CONSTRAINT stages           (replace velocities; write own DIAGNOSTIC p)
per-step epilogue:
  S5  NaN check -> carried panicked flag       (seam to D4)
  S6  DIAGNOSTIC stages           (accumulators, cfl-type writes)
  clock tick; host boundary (IO, progress, restart) is D4's
```

Old `SyncModule` dies (operator contract), `ResetTendency` dies
(contribution dicts), the trio collapses into one CONSTRAINT stage.

## 2. Stage kinds, ordering, determinism, user extension

Kind vocabulary (closed for it-1):

| kind | slot | cadence | write gate |
|---|---|---|---|
| `SELF_UPDATE` | S1, per substage | only if an input is time-dependent | own AUXILIARY, `replace` |
| `CONSTRAINT` | S4, per substage | every state-producing advance | role-selected PROGNOSTIC + own DIAGNOSTIC, `replace` |
| `DIAGNOSTIC` | S6, once per step | every step | own DIAGNOSTIC, `replace` |

(Tendency contributions are **terms, not stages**.)

**Ordering rule — discharges D1.3 commitment 5**: schedule position
is a pure function of declared **kind**; within a kind, order is
`(explicit order: int = 0, module tuple index, intra-module
declaration index)`. Correctness/determinism distinction:

- **Correctness never depends on list position** — an assembly lint
  errors on two same-kind stages with overlapping write (or
  write-read) sets and equal `order=`, demanding an explicit order.
- **Bitwise determinism may use list position** for tie-breaking and
  accumulation order — permuting the module tuple already changes
  the state treedef (D1.4 component order), so no new coupling.

**User extension**: the old "insert before the pressure trio" hack
becomes a theorem (all terms precede all constraints by kind) — a
user physics module contributes terms and lands correctly with zero
positioning API. Users own stages by declaring a kind (positivity
clamps → CONSTRAINT; step-frequency accumulators → DIAGNOSTIC;
time-dependent geometry → SELF_UPDATE). Deliberately no "insert
before X" API and no open kind set in it-1.

## 3. DIAGNOSTIC read semantics; the `div` ruling

**The rule**: a read of any component in any hook sees the value of
the **nearest preceding write in the schedule, crossing substage and
step boundaries**. A term reading `state["p"]` at substage i sees
substage i−1's projection (substage 1: the previous step's — the
warm-start semantics D1.5 promised); a DIAGNOSTIC stage sees this
step's final projection; IO sees everything post-S6.

**The `div` ruling** (thread-1 residual): **`p` is declared
DIAGNOSTIC; `div` is not declared at all.** General rule: *declare
DIAGNOSTIC iff the value is read outside the producing stage* (later
stage, step-cadence IO, or across steps/warm start). `p` qualifies
twice; `div` fails every clause — inside the new stage it is a plain
local (`div = grid.div(vel)` feeding the solve); stages compose
locally instead of communicating through ModelState as the old trio
did. Incompressibility monitoring is better served by a pure
function `nh.diagnostics.divergence(state)` (measures the *actual*
state divergence, near machine zero when healthy). **Amendment owed
to D1.3 commitment 1 / D1.5 stage sentence: "(p, div)" → p only.**

## 4. Signature packaging — the context object

`hook(self, state, ctx) -> dict`, with

```python
@fr.utils.jaxify                 # frozen bundle, all-scalar leaves
class StepContext:
    params:   Mapping[str, Array]   # eval_params(modules, stage_time)
    clock:    Clock                 # traced; .time == stage time
    dt:       Array                 # full step size
    stage_dt: Array                 # increment of the current advance
```

Why ctx beats positional growth: the list is already four and
growing (`stage_dt` for pressure normalization; `dt` for cfl);
HaloTracer is indifferent (the tracer wraps *state*; ctx is scalars
— zero mimicry machinery); ergonomics stay flat (Coriolis ignores
ctx; dsqr consumers read `ctx.params`; owners resolve their own
Ramps via `resolve_at(self.x, ctx.clock.time)`).

**Applied uniformly — every in-trace hook returns a dict applied per
the kind's write gate**: `tendency` → add (accumulated); implicit
`solve` → stepper (γΔt supplied by the scheme); `self_update`,
constraint stages, diagnostic stages → `replace`. This extends
D1.5's contribution-dict idiom to all hooks — one validation path,
one halo-trace path, and no hook ever mutates or returns self
(step-evolving module data that isn't a parameter leaf is a declared
DIAGNOSTIC/AUXILIARY field).

**D2.3 reconciliation**: pure diagnostics keep explicit kwargs —
"kwargs at the notebook boundary, ctx inside the trace"; the binding
layer converts.

## 5. self_update slot: per substage, at substage time

Not once per step: (i) it falls out of the single micro-schedule for
free; (ii) `eval_params` already runs per substage, so scalar and
field-valued time-dependent parameters stay stage-consistent (a
ramped scalar and a ramped N² profile see the same time); (iii)
once-per-step gives every stage N²(t) instead of N²(t+cᵢdt) — a
leading error −dt²/2·(dN²/dt)·w per step. Cost is a non-issue
(scheduled only when time-dependent; typical body is a broadcast
1-DOF/profile rewrite). A `cadence=STEP` opt-in for genuinely
expensive updates (moving geometry) is recorded, not built.

## 6. NaN/panic seam

Position: **S5 — after the last CONSTRAINT, before DIAGNOSTIC
stages, once per step** (NaNs propagate; per-substage checks are
wasted). Reduces `isnan` over PROGNOSTIC leaves into the carried
`panicked` flag; step body wrapped in `lax.cond(panicked, no-op,
step)`; host inspects at chunk boundary. Post-constraint = the
committed state is checked; pre-S6 keeps garbage out of DIAGNOSTIC
accumulators. Same observable behavior as the old step order, now
trace-compatible. Mechanism owned by D4.

## 7. Risks and open questions

1. General Butcher tableaus cost one projection per stage state;
   low-storage RK3 (Oceananigans-style) should be the flagship
   default. AB unaffected.
2. `p = φ/stage_dt` is an O(dt) pressure diagnostic — fine for IO;
   document for pressure-budget users.
3. Non-divergence-free ICs self-correct in the first substage but
   spike the first written `p`; consider an optional host-side
   initial projection in the IC step — D4/IC thread.
4. Iterative solvers (curvilinear): per-substage solves dominate;
   carried-`p` warm start is the mitigation; tolerance/elision knobs
   recorded, not designed.
5. Multiple CONSTRAINTs don't commute; the lint forces explicit
   `order=`; semantics (a clamp destroying divergence-freeness) are
   user responsibility — document.
6. By-variable splits: the micro-schedule generalizes to
   per-variable-group substages; the kind vocabulary must not
   hard-code "one advance per substage" — owed to 03_time_stepping
   (reconciled there with D3.4's ADVANCE kind).
7. dt read surface settled (`ctx.dt`/`ctx.stage_dt`); storage stays
   with D3.2.
8. DIAGNOSTIC-stage dependency chains: explicit `order=`;
   topological sort by declared reads/writes is the recorded upgrade.
9. Backward runs: sign conventions thread through `stage_dt` —
   untested corner, record.

## 8. Sketch — the composed nonhydro step

```python
class DynamicalCore(fr.Module):
    # declares u,v,w (roles); p DIAGNOSTIC; owns dsqr, rossby, the solver
    stages = (fr.Stage(kind=fr.StageKind.CONSTRAINT, func="project"),)
    def project(self, state, ctx):
        vel  = self._vel                              # bound names from bind(table)
        div  = grid.div(tuple(state[n] for n in vel)) # local temp — never declared
        phi  = self._solver(div)                      # ∇²_dsqr φ = div
        g    = grid.grad(phi)
        dsqr = ctx.params["nonhydro.dsqr"]
        return {"u": state["u"] - g[0], "v": state["v"] - g[1],
                "w": state["w"] - g[2] / dsqr,
                "p": phi / ctx.stage_dt}              # physical-pressure diagnostic

def step(carry):                                      # composed at assembly
    state, stepper_state, clock, panicked = carry
    for i in stepper.substages:                       # unrolled at trace time
        ctx = StepContext(params=eval_params(modules, t_i), clock=clock.at(t_i),
                          dt=stepper.dt, stage_dt=stepper.stage_dt[i])
        for m in schedule.self_updates:                       # S1
            state = state.replace(**m.self_update(state, ctx))
        tend = accumulate_terms(state, ctx)                   # S2 (named errors)
        state, stepper_state = stepper.advance(i, state, tend, stepper_state)  # S3
        for stg in schedule.constraints:                      # S4
            state = state.replace(**stg(state, ctx))
    panicked |= any_nan(state.prognostic)                     # S5
    for stg in schedule.diagnostics:                          # S6
        state = state.replace(**stg(state, ctx))
    return state, stepper_state, clock.tick(stepper.dt), panicked
```

The old eight-entry MainTendency maps: Sync → operator contract;
ResetTendency → contribution dicts; linear/advection/user → S2 terms
(order-free by summation); the trio → one CONSTRAINT; `add_module`
positioning → the kind-ordering theorem.
