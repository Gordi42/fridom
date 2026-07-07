# D3.2 — The stepper as a pure scan body: stepper state, dt, Clock

Research report (see [`README.md`](README.md) for status).

## 1. Stepper protocol and pytree layout

The stepper is *not* a Module but *is* a jaxified pytree whose only
dynamic leaf is `dt`; its evolving numeric state lives in a separate
`StepperState` pytree in the carry. The carry is
`(state, modules, stepper_state, clock)`; the stepper object is a
loop-invariant traced **input** to the jitted chunk, not carry.

```python
class TimeStepper:                      # jaxified; structure static, dt dynamic
    def init(self, tendency_template: State) -> StepperState: ...
    def step(self, stepper_state, state, stages, clock
             ) -> tuple[StepperState, State, Clock]: ...
    def time_discretization_effect(self, omega, *, dt=None): ...   # host-side
```

The stepper owns clock advancement (`clock.tick(dt)`) — required for
future adaptive dt.

| Thing | Where | Static/dynamic |
|---|---|---|
| order, eps, tableau choice | stepper | **static** |
| coefficient/warm-up tables | stepper | **static** nested tuples (constant-folded) |
| `dt` | stepper | **dynamic leaf** (§5) |
| AB ring buffer (tuple of States) | StepperState | **dynamic carry** |
| warm-up counter (int32) | StepperState | **dynamic carry** |
| RK stage values `k_i` | nowhere | **locals** (recomputed per step) |
| `start_date: datetime64` | Clock aux | **static/host** |
| `start`, `elapsed`, `it` | Clock | **dynamic** (float64/int64) |

Precedent: diffrax (`solver.init/step` with threaded solver_state);
Oceananigans (QAB2 holds Gⁿ/G⁻; RK3 holds nothing between steps).

## 2. Warm-up under scan

Old mechanism: Python `it_count` + `update_coeff_AB` writing a
**dense length-order coefficient vector, zero-padded beyond the
current level** — already branchless in the padded sense.

- (a) `lax.switch` per level: works, more jaxpr surface than needed.
- **(b) Dense zero-padded (order × order) table, row-indexed by a
  carried saturating counter — CHOSEN.** `weights =
  table[min(counter, order-1)] * dt`; branch-free single gather; the
  direct traced transcription of the old scheme.
- (c) First K steps outside the scan: **rejected** — makes the first
  chunk a different trace, breaks `run(steps=1)`-repeated equivalence,
  and violates D1's bitwise mid-warm-up restart (warm-up state must
  be ordinary carry).

Counter is **stepper-local, not `clock.it`** (old semantics:
`run(start_step=1000)` on a fresh model still warms up; `reset()`
re-warms — OptimalBalance depends on per-leg re-warm via
`model.reset()`). Ring buffer: **tuple of States shifted
structurally** (`(tend, *history[:-1])`) — pure dataflow renaming
under trace, preserves newest-first order; not a stacked array +
roll.

## 3. AB parity notes (ROADMAP 2.7 cutover)

- Tables: AB1 `[1]`; **AB2 `[3/2+eps, -1/2-eps]`** (eps default
  0.01, the quasi-AB2 stabilizing shift — Oceananigans' χ); AB3
  `[23/12, -4/3, 5/12]`; AB4 `[55/24, -59/24, 37/24, -3/8]`.
  **eps appears only in the AB2 row — but that row is also the
  warm-up row**: an order-3/4 run's *second step* uses the
  eps-corrected coefficients. The table builder must reproduce this.
- Warm-up (order 3): `(1,0,0)` → `(3/2+eps, −1/2−eps, 0)` → AB3.
  (Oceananigans instead warm-starts G⁻=Gⁿ — equivalent at order 2;
  fridom's explicit ramp generalizes to orders 3–4; keep the ramp.)
- **Bitwise parity rules**: compute `weights = row * dt` (scalar
  premult) first, then `Σ weights[j]·history[j]` ascending j
  (newest first), then add to state — never `dt·(c·h)` (different
  rounding). Coefficients follow `dtype_real()`.
- Evaluation time: AB evaluates the tendency at the **pre-tick
  time** (old `_compute_tendency` runs tendency then tick) — keep.

## 4. RK stage handling

Explicit fixed-step RK: `StepperState = ()` (the unit pytree —
documentation-by-construction that only multistep memory earns
carry). Stage loop unrolls (static order); stage clocks via
`clock.shifted(c_i·dt)` (functional; supersedes the old deep-copied
clock); `eval_params` at stage times is exactly D2's design. Old
`dz_list` stage buffers were mutation-idiom artifacts — contribution
dicts kill them. **Adaptive RK (embedded pairs) excluded from it-1**
(the old adaptive path was host-side `while` + float() coercions;
designed-for: dt into StepperState, bounded while_loop, diffrax as
reference; keep the `b_error` tableau data in the port).

## 5. dt ownership — dynamic scalar leaf on the stepper

Not static (dt is the sweep parameter par excellence — D2's own
"provided parameters must be dynamic" rule; backward runs flip its
sign), not carry (buys nothing at fixed step; adaptive later moves
it into StepperState). As a leaf on a jit input it is a
loop-invariant scalar the compiler hoists; the only loss vs static
is constant-folding `c·dt` — negligible. Boundary: constructor
accepts `float | np.timedelta64`, converted once
(`value / np.timedelta64(1,"s")` → dtype_real → asarray). No
dt-change hook survives (nothing is pre-baked).
**cfl access**: host/bound diagnostics read the live leaf
(optionally registered as `fr.params.TIME_STEP`); in-trace consumers
get `ctx.dt` (D3.3's packaging).

## 6. Clock (incl. backward runs)

```python
@partial(fr.utils.jaxify, dynamic=("start", "elapsed", "it"))
class Clock:
    # static aux: start_date: np.datetime64 | None (host-only calendar anchor)
    def tick(self, dt):     ...   # elapsed += dt, it += 1
    def shifted(self, tau): ...   # elapsed += tau, it unchanged (stage clocks)
    @property
    def time(self):         ...   # start + elapsed — the traced axis, f64 seconds
```

- Calendar strictly host-side (`start_date + timedelta64(elapsed)`
  for progress/writer timestamps/time-target parsing); no datetime64
  in a trace.
- `elapsed` accumulates by repeated `+= dt` (parity; also survives
  adaptive dt, unlike `it*dt`).
- **Backward runs: keep the capability, drop the method.** Verified:
  OptimalBalance never calls `run_backward` — it sets
  `dt = -abs(dt)` and drives steps in its own host loop; D2.4
  already re-plans it as a host driver. The primitive is **signed
  dt**, free with the §5 leaf: tick adds negative dt, `it` still
  increments, AB warm-up is sign-symmetric, clip-based Ramps
  evaluate fine at decreasing t. `Model.run_backward` is dropped.

## 7. `time_discretization_effect` successor

Verified: builds the AB stability polynomial with the **full-order
row (incl. eps at order 2)**, root-finds with `np.roots` per grid
point on CPU — inherently host-side analysis. Stays a stepper
method, never traced; gains an explicit `dt=` override (D2.4's
`at_time=`-style explicitness) defaulting to the live leaf;
`omega` is plain scalar/array — Symbol callers materialize first
(the stepper stays grid-free). Application sites are recipes
(`single_wave`), per D2.4. RK implementation deferred (parity: old
RK had none).

## 8. Run-loop shape (assumption handed to D4)

**Chunked scan** (presupposed by D1's restart decision): host loop
over chunks, each one jitted `lax.scan`; chunk boundary = host sync
(writer flush, snapshot — the carry IS the checkpoint, NaN/panic
check, progress, KeyboardInterrupt). Steps-count runs primary
(fixed chunk_len + remainder chunk — at most two compiled lengths);
time-target runs reduce to steps at fixed dt (reproducing the old
overshoot semantics); no traced while_loop in it-1. With the carried
warm-up counter, **the first chunk compiles to the same trace as
every other**, and a mid-warm-up crash restores bitwise.

## 9. Risks / open questions

1. **x64 rule needed**: Clock.elapsed/start must be float64 even in
   float32-field runs (float32 seconds lose sub-dt resolution within
   hours — Ramp/forcing phases silently wrong). 02_rules entry.
2. Per-stage aux staleness if self_update ran once per step —
   superseded by D3.3's per-substage ruling.
3. Tendency-vector alignment: history/k are PROGNOSTIC-only vectors
   added to a full state — needs a key-aligned `add_prognostic`
   (small fields.md follow-up).
4. Carry memory: AB(order) holds order PROGNOSTIC copies (same as
   old dz_list); check chunked-scan buffer donation.
5. Adaptive path debt recorded (embedded pairs, dt-in-carry,
   while_loop, controller state).
6. **Restart fingerprint must cover stepper statics**
   (order/eps/tableau — land in treedef aux via jaxify; verify the
   fingerprint hashes aux data). Ties to D2's Ramp-fingerprint item.
7. Cutover-parity tests in both float32 and float64.

## 10. Sketch — AdamBashforth(order=3) as a scan body

```python
@partial(fr.utils.jaxify, dynamic=("history", "warmup"))
class ABState:
    history: tuple[State, ...]   # length=order, newest first
    warmup:  jnp.int32           # saturates at order-1

@partial(fr.utils.jaxify, dynamic=("dt",))
class AdamBashforth:
    def __init__(self, dt=1.0, order=3, eps=0.01):
        self.order, self.eps = order, float(eps)
        rows = [(1.0,), (1.5 + eps, -0.5 - eps),
                (23/12, -4/3, 5/12), (55/24, -59/24, 37/24, -3/8)][:order]
        self.table = tuple(tuple(r) + (0.0,) * (order - len(r)) for r in rows)
        self.dt = fr.utils.to_seconds(dt)            # timedelta64 dies here

    def init(self, tendency_template):
        z = tendency_template.zeros()
        return ABState(history=(z,) * self.order, warmup=jnp.int32(0))

    def step(self, sst, state, stages, clock):
        tend    = stages.tendency(state, clock)      # one stage, c=(0,)
        history = (tend, *sst.history[:-1])          # structural shift
        table   = jnp.asarray(self.table, dtype=fr.utils.dtype_real())
        weights = table[sst.warmup] * self.dt        # premult first = old rounding
        incr = weights[0] * history[0]
        for j in range(1, self.order):               # static unroll, ascending j
            incr = incr + weights[j] * history[j]
        new_state = state.add_prognostic(incr)
        new_clock = clock.tick(self.dt)
        new_state = stages.constrain(new_state, new_clock)
        return ABState(history, jnp.minimum(sst.warmup + 1, self.order - 1)), \
               new_state, new_clock
```

Warm-up walk (order 3): step 1 forward Euler; step 2 eps-corrected
AB2; step 3+ AB3 — zero-padded weights mask the zero-initialized
history rows (no NaN hazard, identical to old). Restart restores
`(history, warmup)` bitwise; `model.reset()` (OptimalBalance per
ramp leg) calls `stepper.init` for a deliberate re-warm.
