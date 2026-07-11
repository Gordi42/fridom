# Model layer redesign — Class designs: time steppers

Part of the model-layer class designs; see [`README.md`](README.md)
for the document map, the shared template, and the cross-cluster seam
anchors. Normative design: [`../03_time_stepping.md`](../03_time_stepping.md)
(§5.1–5.9, primary), the D3 rules in [`../02_rules.md`](../02_rules.md)
(float64 clock, eps order-2-only, restart fingerprint, Ramp signs),
the run-loop demands in [`../04_run_loop_io.md`](../04_run_loop_io.md)
§6.3 (chunk body, AOT compile, donation), the D3 summary in
[`../01_concepts.md`](../01_concepts.md), and the master-clock
constraint CS-9 ([`../09_coupling_designfor.md`](../09_coupling_designfor.md)
§11.3). Research archive: [d3_2](../research/d3_2_stepper_core.md)
(stepper core, the AdamBashforth sketch),
[d3_4](../research/d3_4_imex_splitting.md) (IMEX families, buffers).

**Nothing here re-decides.** Where a spec detail is not fixed by the
notes, or where packaging refines a signed spelling, it is flagged
inline as **Deviation (D-n)** and listed at the end.

---

## Module placement

```
fridom/framework2/model/
    clock.py                 # Clock                        (fr.Clock)
    schedule.py              # Schedule, BoundSchedule, TendencySums
    time_steppers/
        __init__.py          # the fr.time_steppers namespace
        base.py              # TimeStepper, StepperState conventions
        adam_bashforth.py    # AdamBashforth, ABState
        runge_kutta.py       # ButcherTableau, tableaus,
                             #   ExplicitRungeKutta, LowStorageRK3
        imex.py              # IMEXMultistep, IMEXState, CNAB2, SBDF2
```

- The public namespace is **`fr.time_steppers`** — the acceptance
  spelling (`../05_api_sketches.md` 7.1/7.4/7.11:
  `fr.time_steppers.AdamBashforth(dt=60.0, order=3)`,
  `fr.time_steppers.CNAB2(dt=600.0)`,
  `fr.time_steppers.LowStorageRK3(dt=60.0)`). The `fr.steppers.`
  spelling in the d3_4 sketches is superseded by the sketch file.
- `Clock` is re-exported as **`fr.Clock`**: users meet it as
  `ctx.clock` and through `model`; assembly step 8 constructs it.
- `schedule.py` is **internal** — built by the `TendencyComposer`
  (cluster [`model.md`](model.md)) at assembly step 5, consumed by
  steppers; it has no `fr.`-level export. This file owns its class
  surface because the stepper protocol is meaningless without it.
- Tests mirror the package as
  `tests/framework2/model/time_steppers/**`.

## Cluster-wide rules

Restating the signed invariants every class below obeys (§5.3,
02_rules, README seam anchors):

- **The stepper is not a Module but is a pytree** whose only dynamic
  leaf is `dt`. Everything else on it (order, eps, tables, tableaus,
  coefficient levels) is **static**, lands in the treedef aux via
  jaxify, and is hashed by the **restart fingerprint** ("stepper
  statics" — 02_rules). The stepper is a **loop-invariant traced
  input** to the jitted chunk, never carry; the run loop passes it
  non-donated while the carry is donated (04 §6.3).
- **Evolving numeric state lives in `StepperState`**, an entry of the
  carry `(state, modules, stepper_state, clock, panicked)`. Nothing
  stepper-related exists outside the carry and the static stepper
  object (CS-11: snapshots never embed run-loop/driver state).
- **`dt` is a dynamic scalar leaf** — not static (the sweep parameter
  par excellence; sign flips give backward runs), not carry
  (adaptive-only, designed-for). `float | np.timedelta64` converts
  once at the constructor boundary (`fr.utils.to_seconds`;
  `dtype_real()`). The stepper joins the assembly binding table as
  **provider of `fr.params.TIME_STEP`** (its dt leaf) — the read
  surface for `cfl`-style host diagnostics and the write surface for
  `update_parameters` (including the sign flip that replaces
  `run_backward`).
- **Backward runs are a capability, not a method**: the primitive is
  signed dt. `tick` adds negative dt, `it` still increments, warm-up
  is sign-symmetric, clip-based Ramps evaluate at decreasing t
  (`Ramp.reversed()` reflects the time domain — 02_rules). The
  snapshot manifest records the dt value; `load_snapshot` errors on a
  sign mismatch and warns on magnitude.
- **Clock precision**: `start`/`elapsed` are float64 even in
  float32-field runs; `it` is int64. Calendar is strictly host-side.
- **Warm-up under scan**: dense zero-padded coefficient tables
  row-indexed by a **carried saturating int32 counter** that is
  stepper-local, never `clock.it` (`reset()` re-warms — OptimalBalance
  drives ramp legs through it). Ring buffers are **tuples of States
  shifted structurally** (dataflow renaming, newest first). With the
  counter in the carry, the first chunk compiles to the same trace as
  every other, and a mid-warm-up crash restores bitwise.
- **`supported_treatments` is an assembly check** (§5.1): an IMPLICIT
  term under a purely explicit stepper is an assembly error, never a
  silent demotion.
- **Projection is project-the-state** (§5.6): CONSTRAINT stages run
  after every state-producing advance; multistep families project
  once per step, IMEX-RK per stage; history buffers store
  **unprojected** explicit tendencies.
- **The stepper owns `clock.tick(dt)`** (§5.3) — required for the
  adaptive designed-for. S5 (NaN seam) and S6 (DIAGNOSTIC stages) are
  the **per-step epilogue owned by D4's chunk body**, outside
  `TimeStepper.step`.

---

### TimeStepper

The scan-body stepper protocol: `init`/`step` plus host-side
analysis; the base of every family below (§5.3).

| Aspect | Value |
|--------|-------|
| Kind | ABC |
| Pytree | jaxified per concrete class, `dynamic=("dt",)`; all other attributes static treedef aux (fingerprint-hashed); **not a Module, host-constructed** |
| Task | 2.4 (protocol, dt leaf, TIME_STEP provider role) |
| Design refs | §5.3; 02_rules (fingerprint, clock); 04 §6.2 step 2, §6.3; d3_2 §1, §5, §7 |

```python
"""Scan-body time steppers: the stepper core (design 03, §5.3)."""
from __future__ import annotations

import abc

import fridom.framework2 as fr


class TimeStepper(abc.ABC):
    """ABC of scan-body time steppers; a pytree, not a Module."""

    supported_treatments: ClassVar[frozenset[Treatment]]       # 2.4
    """Term treatments this stepper integrates (§5.1). Assembly
    errors on an unsupported treatment — never silent demotion."""

    def __init__(self, dt: float | np.timedelta64) -> None:    # 2.4
        """Convert dt once (fr.utils.to_seconds -> dtype_real ->
        asarray) onto the dynamic leaf; timedelta64 dies here."""
        ...

    @property
    def dt(self) -> jax.Array:                                 # 2.4
        """The dynamic scalar dt leaf (signed; dtype_real). The
        live host-side read surface; in-trace consumers use
        ctx.dt/ctx.stage_dt."""
        ...

    @property
    def provided_parameters(self) -> Mapping[str, str]:        # 2.4
        """Provider rows for assembly step 2:
        {fr.params.TIME_STEP: "dt"} — the dt leaf joins the
        parameter binding table (exact row shape: model.md)."""
        ...

    @abc.abstractmethod
    def init(self, tendency_template: State) -> StepperState:  # 2.4
        """Fresh StepperState: zeroed rings shaped like the
        PROGNOSTIC tendency template, warm-up counter 0. Called
        at assembly step 8 and by model.reset() (deliberate
        re-warm — the OptimalBalance ramp-leg contract)."""
        ...

    @abc.abstractmethod
    def step(                                                  # 2.4
        self,
        stepper_state: StepperState,
        state: State,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[StepperState, State, Clock]:
        """One canonical step (§5.2): per substage P0 ctx ->
        S1/S1' -> S2 terms -> S3 advance -> S3' ADVANCE stages ->
        S4 CONSTRAINT; owns clock.tick(dt). S5/S6 are the chunk
        body's epilogue, not the stepper's."""
        ...

    def time_discretization_effect(                            # 2.5
        self,
        omega: np.ndarray,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """Host-side discrete-dispersion analysis of a frequency
        array; never traced. dt= overrides the live leaf. omega is
        a plain array — Symbol callers materialize first (the
        stepper stays grid-free). Base raises NotImplementedError;
        applied by recipes (single_wave), per D2.4."""
        ...
```

Semantics, invariants, error behavior:

- **`step` receives a `BoundSchedule`** — the static assembly
  `Schedule` closed over the carry's *current* module pytree by the
  chunk body (`schedule.bind(carry.modules)`, pure dataflow inside
  the trace). **Deviation (D-1)**: §5.3 spells the argument
  `schedule`; binding it before the call is packaging, not a
  semantic change — it preserves the signed arity and is what keeps
  term `fn`s **unbound** until they meet live module leaves (the D2
  aliasing rule; a schedule capturing assembly-time modules would be
  the aliasing trap by construction).
- **The chunk-body contract (04 §6.3)**: the run loop jits one
  framework-level `step_chunk(assembly_record, carry, n)` with the
  carry donated and the stepper/schedule as loop-invariant,
  non-donated arguments; identical re-assemblies share the jit cache
  because the stepper's statics are hashable treedef aux. AOT
  `lower().compile()` reports compile time and peak memory. The
  stepper contributes nothing host-side to a step — no callbacks, no
  Python state.
- **`init` ≡ re-warm**: there is no `reset`/`_on_setup` successor;
  `model.reset()` swaps in `stepper.init(template)` output, and
  `update_parameters(..., rewarm=True)` (the default) zeroes the
  warm-up counter — buffers need no zeroing because warm-up rows
  never weight entries that have not been written since the re-warm
  (04 §6.5).
- **`time_discretization_effect`** is implemented by
  `AdamBashforth` in iteration 1 (full-order row **including eps at
  order 2**, `np.roots` per point on CPU); the RK and IMEX
  implementations are deferred with parity (the old RK had none) —
  the base default raises `NotImplementedError` with that pointer.
- **Statics and the fingerprint**: everything except `dt` must be
  reachable from the treedef aux — the restart fingerprint hashes
  stepper statics (order/eps/tableau/scheme), and a snapshot-manifest
  mismatch *diffs* ("stepper statics differ: cnab2 -> sbdf2"), never
  silently reuses history (02_rules; d3_2 risk 6).

---

### StepperState (conventions)

Per-family carried numeric state; a convention plus per-family
frozen pytrees, not a base class with behavior.

| Aspect | Value |
|--------|-------|
| Kind | type alias (`StepperState: TypeAlias = Any` jaxified pytree) + per-family concrete frozen classes |
| Pytree | fully dynamic carry (rings + counter); flattens with the carry, donated with it |
| Task | 2.4 (conventions, ABState); 2.5 (IMEXState) |
| Design refs | §5.3, §5.4 (buffer partitioning/ownership); d3_2 §1–2, §10; d3_4 §2 |

The normative conventions:

- **Rings are tuples of States, newest first**, shifted structurally:
  `(newest, *old[:-1])` — pure dataflow renaming under trace; never a
  stacked array + `roll`. Ring entries are **PROGNOSTIC-only
  tendency/state vectors** shaped like `init`'s template (their
  key-aligned application to a full state is
  `State.add_prognostic` — fields.md follow-up, open question 2).
- **The warm-up counter** is a saturating scalar
  (`jnp.minimum(counter + 1, levels - 1)`, int32), stepper-local.
  It selects a row of a dense zero-padded static table — one
  branch-free gather per step.
- **Buffers partition by treatment** (§5.4): the explicit ring
  stores the **summed** explicit contribution (per-term history is
  never needed); the implicit side buffers nothing, ever (buffering
  `L·Xⁿ` across the projection boundary is the staleness bug of
  d3_4 §1); SBDF adds the one new buffer class — past states.
  History stores **unprojected** explicit tendencies (§5.6).
- **Ownership is per ADVANCE stage**: `StepperState` belongs to the
  **primary** ADVANCE (the stepper). Module-owned ADVANCE stages
  (barotropic subcycle) persist their cross-step integrator state in
  **own-AUX fields** (V-H3), with their integrator *statics* joining
  the fingerprint — they never reach into `StepperState`.
- **RK's unit pytree**: explicit fixed-step RK carries
  `StepperState = ()` — documentation-by-construction that only
  multistep memory earns carry; stage values are locals.
- Restart restores every StepperState leaf bitwise (the carry is the
  checkpoint); `update_parameters` leaves up to s−1 slightly-stale
  buffered tendencies, which the default `rewarm=True` re-ramp
  resolves (04 §6.5).

Per-family shapes (skeletons live with their families below):

| Family | StepperState | Contents |
|---|---|---|
| `AdamBashforth` | `ABState` | `history` (order tendencies), `warmup` |
| `ExplicitRungeKutta` / `LowStorageRK3` | `()` | — |
| `IMEXMultistep` | `IMEXState` | `f_history`, `x_history` (SBDF only), `warmup` |
| adaptive RK (designed-for) | own class | `dt` moves in, controller state |

---

### Clock

The traced model clock: float64 `start`/`elapsed`, int64 `it`;
calendar strictly host-side (§5.3; d3_2 §6).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final |
| Pytree | `jaxify, dynamic=("start", "elapsed", "it")`; `start_date` static host-only aux |
| Task | 2.4 |
| Design refs | §5.3; 02_rules (clock precision); CS-9/CS-10; d3_2 §6 |

```python
"""The traced model clock (fr.Clock)."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@partial(fr.utils.jaxify, dynamic=("start", "elapsed", "it"))
class Clock:
    """Traced start/elapsed/it; host-side calendar anchor."""

    def __init__(                                              # 2.4
        self,
        start: float = 0.0,
        *,
        start_date: np.datetime64 | None = None,
    ) -> None:
        """float64 start (seconds), elapsed=0.0, it=0; start_date
        is a static, host-only calendar anchor."""
        ...

    @property
    def start(self) -> jax.Array:                              # 2.4
        """Run-start model time, float64 seconds (traced)."""
        ...

    @property
    def elapsed(self) -> jax.Array:                            # 2.4
        """Accumulated signed model time since start, float64
        (traced); accumulates by repeated += dt (parity; survives
        adaptive dt, unlike it*dt)."""
        ...

    @property
    def it(self) -> jax.Array:                                 # 2.4
        """Iteration counter, int64 (traced); increments on every
        tick, forward or backward."""
        ...

    @property
    def time(self) -> jax.Array:                               # 2.4
        """start + elapsed — the traced time axis, float64
        seconds. Equals the stage time on shifted clocks."""
        ...

    @property
    def start_date(self) -> np.datetime64 | None:              # 2.4
        """Static host-side calendar anchor; never traced."""
        ...

    @property
    def date(self) -> np.datetime64:                           # 2.4
        """Host-side calendar timestamp: start_date +
        timedelta64(elapsed). Raises if start_date is None; used
        by progress/writers/time-target parsing only."""
        ...

    def tick(self, dt: jax.Array) -> Clock:                    # 2.4
        """Functional step advance: elapsed += dt (signed),
        it += 1."""
        ...

    def shifted(self, tau: jax.Array) -> Clock:                # 2.4
        """Stage clock: elapsed += tau, it unchanged (supersedes
        the old deep-copied clock). tau carries dt's sign."""
        ...

    def __repr__(self) -> str: ...                             # 2.4
```

Semantics, invariants, error behavior:

- **Precision is a rule, not a default** *(amended 2026-07-08,
  global-precision reconciliation)*: "float64" in the traced-leaf
  docstrings means **global width** — float64 under the default
  x64-on run, where the original rule holds verbatim. In an x64-off
  (float32) run a float64 traced leaf is impossible (JAX downcasts);
  there the host-side float64 clock stays authoritative and
  re-anchors the traced `elapsed` at every chunk boundary, `it`
  (integer, exact) keys triggers/schedules, and the residual float32
  quantization of absolute stage time is an accepted property of
  float32 runs (02_rules, Clock precision). No per-clock dtype knob
  exists either way.
- **Signed-dt semantics**: `tick(-|dt|)` is the backward-run
  primitive; `it` still increments (it counts steps, not direction).
  Nothing on the Clock is direction-aware; Ramp reversal is a Ramp
  spec concern (02_rules).
- **Calendar host-side only**: no `datetime64` in a trace, ever.
  `date` is the single conversion point, consumed by D4's
  progress/writer machinery. `start_date` is static aux — changing
  it is a re-assembly-grade structure change (it does not affect
  numerics; it is not fingerprinted as a leaf).
- **`reset()` builds a fresh Clock** (start preserved, elapsed/it
  zeroed) — which is what restarts a Ramp leg (04 §6.5); no mutating
  method exists on the class.
- **Coupling (CS-9, design-for)**: coupled runs designate one master
  dt; every other model's dt derives by **exact division**, windows
  are exact step counts in every participant — per-model float64
  clocks are never compared by float equality; the cross-model
  consistency assertion reads the snapshot manifest header
  (clock time, it, dt — CS-10). Nothing on this class changes for
  Phase 3.

---

### AdamBashforth

Adams–Bashforth orders 1–4 at cutover parity; the nh preset default
at cutover (`order=3`, V-N3) (§5.3; d3_2 §2–3, §10).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final |
| Pytree | `jaxify, dynamic=("dt",)`; `order`/`eps`/`table` static (fingerprinted) |
| Task | 2.5 (ship); 2.7 (cutover-parity tests, both dtypes) |
| Design refs | §5.3 (warm-up, parity, eps amendment), §5.7; 02_rules (eps order-2-only); d3_2 §2–3, §7, §10 |

```python
"""Adams-Bashforth multistep steppers (orders 1-4)."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@partial(fr.utils.jaxify, dynamic=("history", "warmup"))
class ABState:
    """AdamBashforth carry: tendency ring + warm-up counter."""

    history: tuple[State, ...]   # length=order, newest first     # 2.5
    warmup: jnp.int32            # saturates at order-1           # 2.5


@partial(fr.utils.jaxify, dynamic=("dt",))
class AdamBashforth(TimeStepper):
    """Explicit AB1-4; quasi-AB2 eps damper at order 2 only."""

    supported_treatments = frozenset({fr.EXPLICIT})              # 2.5

    def __init__(                                                # 2.5
        self,
        dt: float | np.timedelta64,
        order: int = 3,
        eps: float | None = None,
    ) -> None:
        """order in 1..4. eps is legal at order=2 only (default
        0.01 there — the quasi-AB2 computational-mode damper);
        any other order rejects a non-None eps with ValueError."""
        ...

    @property
    def order(self) -> int:                                     # 2.5
        """Static AB order (1-4); fingerprinted."""
        ...

    @property
    def eps(self) -> float | None:                              # 2.5
        """Static order-2 damper (None at every other order);
        fingerprinted."""
        ...

    @property
    def table(self) -> tuple[tuple[float, ...], ...]:           # 2.5
        """Dense zero-padded (order x order) warm-up coefficient
        table, row = warm-up level; static, constant-folded."""
        ...

    def init(self, tendency_template: State) -> ABState:        # 2.5
        """Zeroed order-length ring, warmup=0."""
        ...

    def step(                                                   # 2.5
        self,
        stepper_state: ABState,
        state: State,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[ABState, State, Clock]:
        """One AB step: single substage + final combination; see
        the normative body below."""
        ...

    def time_discretization_effect(                             # 2.5
        self,
        omega: np.ndarray,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """Discrete AB dispersion via the stability polynomial of
        the FULL-order row (incl. eps at order 2); np.roots per
        point, host/CPU only; dt= overrides the live leaf."""
        ...
```

**The step body is normative** (the one skeleton in this cluster
shown with a body — it *is* the bitwise-parity algorithm, the direct
traced transcription of the old `update_coeff_AB` scheme; d3_2 §10
adapted to the `BoundSchedule` seam):

```python
    def step(self, sst, state, stages, clock):
        ctx     = stages.context(clock, dt=self.dt, stage_dt=self.dt)
        state   = stages.prepare(state, ctx)                # S1 + S1'
        sums    = stages.tendency(state, ctx)               # S2, pre-tick time
        history = (sums.explicit, *sst.history[:-1])        # structural shift
        table   = jnp.asarray(self.table, dtype=fr.utils.dtype_real())
        weights = table[sst.warmup] * self.dt               # premultiply first
        incr = weights[0] * history[0]
        for j in range(1, self.order):                      # static unroll, ascending j
            incr = incr + weights[j] * history[j]
        state = state.add_prognostic(incr)                  # S3
        clock = clock.tick(self.dt)
        ctx   = stages.context(clock, dt=self.dt, stage_dt=self.dt,
                               sums=sums)
        state = stages.advance_stages(state, ctx)           # S3'
        state = stages.constrain(state, ctx)                # S4
        return (ABState(history, jnp.minimum(sst.warmup + 1,
                                             self.order - 1)),
                state, clock)
```

Semantics, invariants, error behavior:

- **Coefficient rows** (dtype follows `fr.utils.dtype_real()`):
  AB1 `[1]`; AB2 `[3/2+eps, -1/2-eps]` **at order=2 only**;
  AB3 `[23/12, -4/3, 5/12]`; AB4 `[55/24, -59/24, 37/24, -3/8]`.
- **The eps ruling (signed amendment, 02_rules)**: `eps` is an
  **order-2-only** parameter. `AdamBashforth(order=2, eps=0.01)`
  accepts it (and `order=2` defaults to 0.01 for parity with the old
  AB2); any other order rejects a non-None eps with a `ValueError`
  naming the rule. **Order ≥ 3 warm-up uses textbook AB2
  `[3/2, -1/2]`** — a deliberate startup-only delta vs the old code
  (which applied its eps'd AB2 row as every order's warm-up row):
  one step, O(eps·dt), covered by a tolerance-based cutover test
  (§8.8 parity list). *Deviation (D-2)*: the `eps=None` sentinel
  with per-order resolution is spelling, not decision — it is the
  only signature that expresses "default 0.01 at order 2, illegal
  elsewhere" without a per-order default table.
- **Warm-up table**: dense zero-padded `(order × order)`, row-indexed
  by the saturating counter — e.g. order 3:
  `(1, 0, 0)` → `(3/2, -1/2, 0)` → AB3. Zero-padded weights mask the
  zero-initialized ring rows (no NaN hazard, identical to old).
  Unroll-first-K-steps is **rejected** (special first chunk, breaks
  bitwise mid-warm-up restart); `lax.switch` per level is rejected as
  needless jaxpr surface (d3_2 §2).
- **Bitwise parity rules** (2.7 cutover, tested in float32 and
  float64): `weights = row * dt` **premultiplied first** (never
  `dt·(c·h)` — different rounding); accumulation in **ascending j**
  over the newest-first ring; the tendency is evaluated at the
  **pre-tick time** (the old `_compute_tendency` order); the ring
  shift is structural. *Scope amendment (2026-07-08)*: these rules'
  bitwiseness is a claim about the **op sequence**, verified bitwise
  **eager-vs-eager only**; jitted cutover runs are compared
  tolerance-based (≤ a few ulp per step, accumulation-aware) because
  the old framework and framework2 are differently-compiled programs
  (phase-1 finding 1; the 02_rules.md bitwise-equality umbrella).
- The **increment is PROGNOSTIC-only** and lands via
  `state.add_prognostic(incr)` — the key-aligned add (fields.md
  follow-up; open question 2).
- **`time_discretization_effect`** uses the full-order row including
  eps at order 2 — warm-up rows never enter the asymptotic analysis.
- Backward runs need nothing: the table is dt-free, `weights`
  inherits dt's sign, warm-up is sign-symmetric.

---

### The explicit RK family

Fixed-step explicit Runge–Kutta at parity, plus the flagship
low-storage RK3 (§5.3, §5.7; d3_2 §4).

#### ButcherTableau

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen dataclass |
| Pytree | static (hashable; enters the fingerprint via its stepper's treedef aux) |
| Task | 2.5 (fixed-step data); embedded `b_error` retained for the adaptive designed-for |
| Design refs | §5.3 ("keep the b_error tableau data"); d3_2 §4 |

```python
"""Explicit Runge-Kutta steppers and tableau data."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import fridom.framework2 as fr


@dataclass(frozen=True)
class ButcherTableau:
    """Static explicit-RK tableau; tuples for hashability."""

    a: tuple[tuple[float, ...], ...]                           # 2.5
    b: tuple[float, ...]                                       # 2.5
    c: tuple[float, ...]                                       # 2.5
    b_error: tuple[float, ...] | None = None                   # designed-for
    """Embedded-error weights — data retained for the adaptive
    designed-for; unused by fixed-step stepping."""

    @property
    def stages(self) -> int:                                   # 2.5
        """Number of stages, len(b)."""
        ...
```

Module-level tableau presets in `runge_kutta.py` (the old `RKMethods`
enum contents survive as frozen constants; parity data):

```python
tableaus.EULER              # 2.5
tableaus.RK2                # 2.5
tableaus.RK3                # 2.5
tableaus.RK4                # 2.5
tableaus.RK4_38             # 2.5
tableaus.HEUN_EULER         # designed-for (b_error carrier)
tableaus.BOGACKI_SHAMPINE   # designed-for (b_error carrier)
tableaus.RKF45              # designed-for (b_error carrier)
```

#### ExplicitRungeKutta

| Aspect | Value |
|--------|-------|
| Kind | concrete |
| Pytree | `jaxify, dynamic=("dt",)`; tableau static (fingerprinted) |
| Task | 2.5 (parity); 2.7 (cutover tests) |
| Design refs | §5.2 (per-substage schedule), §5.3, §5.6; d3_2 §4 |

```python
@partial(fr.utils.jaxify, dynamic=("dt",))
class ExplicitRungeKutta(TimeStepper):
    """Fixed-step explicit RK over a Butcher tableau."""

    supported_treatments = frozenset({fr.EXPLICIT})            # 2.5

    def __init__(                                              # 2.5
        self,
        dt: float | np.timedelta64,
        tableau: ButcherTableau = tableaus.RK4,
    ) -> None:
        """Fixed-step driver; embedded tableaus (b_error set) are
        rejected here — adaptive stepping is the designed-for
        AdaptiveRungeKutta, not a flag."""
        ...

    @property
    def tableau(self) -> ButcherTableau:                       # 2.5
        """Static tableau; fingerprinted."""
        ...

    def init(self, tendency_template: State) -> tuple[()]:     # 2.5
        """The unit pytree () — RK carries nothing between steps;
        only multistep memory earns carry."""
        ...

    def step(                                                  # 2.5
        self,
        stepper_state: tuple[()],
        state: State,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[tuple[()], State, Clock]:
        """Statically unrolled stage loop + a separate final
        combination substage; stage clocks via
        clock.shifted(c_i * dt)."""
        ...

    def time_discretization_effect(self, omega, *, dt=None):   # designed-for
        """Deferred with parity (the old RK had none); raises
        NotImplementedError."""
        ...
```

Step algorithm (prose-normative; stage values `k_i` are **locals**,
recomputed per step — the old `dz_list` stage buffers were
mutation-idiom artifacts killed by contribution dicts):

1. For each stage `i` (static unroll): stage clock
   `clock_i = clock.shifted(c_i * dt)`; `ctx_i = stages.context(
   clock_i, dt=dt, stage_dt=<the stage's increment>)` — **eval_params
   at the stage time** (P0; exactly D2's per-stage obligation, and
   why a Ramp forcing is correct under RK with zero extra
   machinery); `stages.prepare` (S1/S1'); build the stage state from
   `state` plus the `a[i][j]`-weighted `k_j` via `add_prognostic`;
   `k_i = stages.tendency(stage_state, ctx_i).explicit`;
   `stages.constrain` applies to the stage state (project-the-state
   holds per produced state, §5.2's per-substage S4).
2. The **final combination is a separate substage** (§5.2): combine
   `b`-weighted `k_j` (premultiplied `b_j * dt`, ascending j — the
   same parity arithmetic as AB), tick, run S3' ADVANCE stages and
   S4 CONSTRAINT at the ticked clock.
3. Return `((), state, clock)`.

#### LowStorageRK3

| Aspect | Value |
|--------|-------|
| Kind | concrete, final |
| Pytree | `jaxify, dynamic=("dt",)`; coefficient set static |
| Task | 2.5 — **the flagship default** (documented recommendation; the nh preset itself pins `AdamBashforth(order=3)` at cutover, V-N3) |
| Design refs | §5.7; 04 §6.1 (`fr.time_steppers.LowStorageRK3(dt=60.0)`) |

```python
@partial(fr.utils.jaxify, dynamic=("dt",))
class LowStorageRK3(TimeStepper):
    """Three-stage low-storage (2N-register) explicit RK3; the
    documented recommendation for new configurations."""

    supported_treatments = frozenset({fr.EXPLICIT})            # 2.5

    def __init__(self, dt: float | np.timedelta64) -> None:    # 2.5
        """Fixed coefficient set; no tableau argument."""
        ...

    @property
    def coefficients(                                          # 2.5
        self,
    ) -> tuple[tuple[float, float, float], ...]:
        """Static (gamma_i, zeta_i, c_i) triples per stage;
        fingerprinted."""
        ...

    def init(self, tendency_template: State) -> tuple[()]: ... # 2.5

    def step(self, stepper_state, state, stages, clock): ...   # 2.5
```

Notes:

- The 2N-register recurrence
  (`q = gamma_i*q_prev + tendency; state += zeta_i*dt*q` in
  functional form) is why this is its own final class and not a
  `ButcherTableau` consumer — the storage shape is the point. Stage
  registers are still locals; `StepperState = ()`.
- Stage clocks/ctx/constraints follow the `ExplicitRungeKutta`
  algorithm verbatim (per-stage eval_params, per-substage S4).
- **Deviation (D-6)**: the notes sign "low-storage RK3 as the
  flagship default" but pin no coefficient set; the implementation
  pins it at 2.5 against the Oceananigans RK3 reference (the
  design's verification anchor) and records the choice in the
  fingerprint via `coefficients`.

---

### IMEXMultistep

The one generic IMEX multistep driver, shipping with the CNAB2 and
SBDF2 coefficient sets (§5.4, §5.7; d3_4).

| Aspect | Value |
|--------|-------|
| Kind | concrete (one driver; schemes are static coefficient sets) |
| Pytree | `jaxify, dynamic=("dt",)`; scheme + level tuples static (fingerprinted) |
| Task | 2.5 (with the reference vertical-diffusion consumer: exact 1D decay + stiff-kappa column tests); SBDF3 / IMEX-RK designed-for |
| Design refs | §5.1 (implicit surface), §5.4, §5.6, §5.7; d3_4 §1–2, §5, §8 |

```python
"""The generic IMEX multistep driver (CNAB2, SBDF2)."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@partial(fr.utils.jaxify, dynamic=("f_history", "x_history", "warmup"))
class IMEXState:
    """IMEXMultistep carry: explicit-F ring, SBDF state ring,
    warm-up counter."""

    f_history: tuple[State, ...]   # summed EXPLICIT sums, newest first  # 2.5
    x_history: tuple[State, ...]   # past states (SBDF only; () CNAB2)   # 2.5
    warmup: jnp.int32                                                    # 2.5


@partial(fr.utils.jaxify, dynamic=("dt",))
class IMEXMultistep(TimeStepper):
    """IMEX-by-treatment multistep: explicit history combine +
    per-field implicit solves, warm-up by whole-tuple switching."""

    supported_treatments = frozenset({fr.EXPLICIT, fr.IMPLICIT})  # 2.5

    def __init__(                                              # 2.5
        self,
        dt: float | np.timedelta64,
        scheme: Literal["cnab2", "sbdf2"] = "cnab2",
    ) -> None:
        """Select the static coefficient-level tuple set."""
        ...

    @property
    def scheme(self) -> str:                                   # 2.5
        """Static scheme name; fingerprinted."""
        ...

    @property
    def levels(self) -> tuple[IMEXLevel, ...]:                 # 2.5
        """Static warm-up levels, index = counter value; each an
        (explicit_weights, state_weights, apply_weight, gamma)
        tuple, zero-padded to the scheme's history depth."""
        ...

    def init(self, tendency_template: State) -> IMEXState:     # 2.5
        """Zeroed rings at the scheme's depths, warmup=0."""
        ...

    def step(                                                  # 2.5
        self,
        stepper_state: IMEXState,
        state: State,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[IMEXState, State, Clock]:
        """One IMEX step; algorithm below."""
        ...

    def time_discretization_effect(self, omega, *, dt=None):   # designed-for
        """Deferred (no parity predecessor); raises
        NotImplementedError."""
        ...


def CNAB2(dt: float | np.timedelta64) -> IMEXMultistep:        # 2.5
    """fr.time_steppers.CNAB2(dt=...) — Crank-Nicolson /
    Adams-Bashforth-2 coefficient set."""
    ...


def SBDF2(dt: float | np.timedelta64) -> IMEXMultistep:        # 2.5
    """fr.time_steppers.SBDF2(dt=...) — semi-implicit BDF2
    coefficient set."""
    ...
```

**Coefficient levels** (solve-normalized form — the `solve` protocol
is `(1 − dt_gamma·L)⁻¹ rhs`, so every level is stated with the
implicit mass normalized to 1; **Deviation (D-4)**: d3_4 writes
SBDF2 unnormalized, `(3/2 − dt·L)Xⁿ⁺¹ = 2Xⁿ − ½Xⁿ⁻¹ + dt(2Fⁿ −
Fⁿ⁻¹)`; dividing by 3/2 is arithmetic restatement, γ = 2/3 as the
§5.4 table states):

| Scheme | Level 0 (first step) | Level 1 (settled) |
|---|---|---|
| `cnab2` | FB Euler: explicit `(1, 0)`, state `(1,)`, apply `0`, γ = 1 | CNAB2: explicit `(3/2, −1/2)`, state `(1,)`, apply `1/2`, γ = 1/2 |
| `sbdf2` | SBDF1: explicit `(1, 0)`, state `(1, 0)`, apply `0`, γ = 1 | SBDF2: explicit `(4/3, −2/3)`, state `(4/3, −1/3)`, apply `0`, γ = 2/3 |

Step algorithm (prose-normative):

1. `ctx = stages.context(clock, dt=dt, stage_dt=dt)` at the pre-tick
   time; `stages.prepare` (S1/S1').
2. `sums = stages.tendency(state, ctx)` — the summed EXPLICIT
   contributions (S2); shift the F-ring structurally:
   `f_history = (sums.explicit, *f_history[:-1])`; for SBDF, shift
   the state ring with the substage-start state.
3. Gather the level by the saturating counter — **warm-up switches
   whole `(explicit_weights, state_weights, apply_weight, gamma)`
   tuples** (γ changes across levels: SBDF1 γ=1 → SBDF2 γ=2/3, FB
   Euler γ=1 → CNAB2 γ=1/2), which is exactly why **`dt_gamma =
   gamma · dt` is necessarily traced**. *Deviation (D-3)*: §5.4
   spells the switched tuple "(weights, γ)"; the `apply_weight`
   member is the CNAB forward-apply coefficient that d3_4's worked
   form contains (`+ dt/2·L·Xⁿ`) — carrying it in the same switched
   tuple is transcription of the worked form, not a new decision.
4. Build the rhs with the AB parity arithmetic (premultiplied
   weights, ascending j): `rhs = Σ state_weights[j]·x_hist[j] +
   Σ (explicit_weights[j]·dt)·f_history[j] + (apply_weight·dt)·(Σ
   implicit forward applies)` — the forward apply is computed
   **fresh each step** through the term's derived explicit path
   (`op.apply` via the §5.1 write-once rule; never buffered — the
   CN solve-only trick reads the pre-projection state, an O(dt)
   error every step).
5. For each merged implicit operator in `stages.implicit`:
   `partition = op.solve(rhs restricted to op.fields, dt_gamma,
   ctx)` — γ-agnostic solves, coupled blocks atomic; components
   under no implicit operator take the explicit combine only
   (via `add_prognostic`).
6. Tick; rebuild ctx with `sums` attached; `stages.advance_stages`
   (S3' — barotropic subcycle et al.); `stages.constrain` (S4 —
   **project-the-state once per step**, after the solves: the only
   coherent IMEX arrangement).
7. Return `(IMEXState(f_history, x_history, warmup+1 sat), state,
   clock)`.

Semantics, invariants, error behavior:

- **`CNAB2`/`SBDF2` are thin factories** returning configured
  `IMEXMultistep` instances — the acceptance spellings
  (`fr.time_steppers.CNAB2(dt=600.0)`) stay expressible; the
  fingerprint hashes `scheme` + `levels`, so factory-vs-subclass is
  invisible to restarts. *Deviation (D-5): packaging choice; the
  notes fix only the spelling.*
- **Buffer discipline** (§5.4): F-ring stores the **summed**
  explicit contribution; the implicit side buffers nothing;
  `x_history` exists only for SBDF (empty tuple for CNAB2 — a
  treedef difference between schemes, consistent with
  scheme-as-static). Buffered `Fʲ` carry their own step-time
  eval_params — Ramp-correct automatically; `update_parameters`
  staleness is handled by the default re-ramp (04 §6.5).
- **Empty-implicit degeneracy**: a composition with **no IMPLICIT
  terms** under `IMEXMultistep` is legal — the solve set is empty
  and the scheme degrades gracefully to its explicit member (CNAB2
  → textbook AB2, level-0 forward Euler; SBDF2 → its explicit
  two-step extrapolated-BDF member). Only the reverse direction (an
  IMPLICIT term under an explicit-only stepper) is the assembly
  error. This is the cheap A/B path for treatment flips
  (`treatment=fr.EXPLICIT` on the mixing module).
- **The per-treatment sums and the increment-form forcing** (§5.4,
  V-H4): post-TENDENCY stages read the per-treatment sums from ctx;
  the split-explicit barotropic stage's **default slow forcing is
  the increment form** `G = ∫(X* − Xⁿ) dz / dt`, computed **by the
  stage** from the substage-start state (buffered via the ADVANCE
  own-AUX gate) — automatically consistent with the outer scheme's
  weights and warm-up row, and including the implicit-mixing
  increment by construction; `forcing="tendency_sums"` is the
  module-constructor knob for the raw sums variant. The stepper's
  only obligations here are: run S3' after the primary advance, and
  attach `sums` to the post-advance ctx.
- **Composition limit** (assembly-checked, §5.4): IMEX-RK ×
  split-explicit is an assembly error (multistep outer drivers
  only); this driver is the sanctioned outer scheme for the 3.1
  hydrostatic step.
- 2.5 ships this against the **reference vertical-diffusion
  consumer** (exact 1D decay + stiff-κ column) — the implicit
  surface is debugged on a toy before the hydrostatic port depends
  on it.

---

### Schedule composition (internal)

What assembly hands the stepper: the kind-ordered stage groups as
callables, the per-treatment tendency accumulation, and the
ctx-construction obligation (§5.2, §5.5; assembly step 5 in 04
§6.2). Owned here because the stepper protocol consumes it; the
`TendencyComposer` that *builds* it is cluster
[`model.md`](model.md)'s.

| Aspect | Value |
|--------|-------|
| Kind | `Schedule` concrete static; `BoundSchedule` ephemeral in-trace view; `TendencySums` frozen jaxified |
| Pytree | `Schedule`: static assembly artifact (part of the hashable assembly record); `BoundSchedule`: not a pytree — a per-step closure over the carry's modules; `TendencySums`: dynamic (PROGNOSTIC-only vectors) |
| Task | 2.4 (context/prepare/tendency/constrain — the AB/RK path); 2.5 (treatment partition, implicit ops, advance stages) |
| Design refs | §5.2, §5.5; 04 §6.2 step 5, §6.3; d3_1 via §5.1 (composer attribution) |

```python
"""Schedule composition: the stepper-facing stage surface."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


class Schedule:
    """Static kind-ordered schedule; built once at assembly."""

    def bind(self, modules: tuple[Module, ...]) -> BoundSchedule:  # 2.4
        """Close the stage/term callables over the carry's current
        module pytree (pure dataflow; called per step by the chunk
        body). Unbound fns meet live leaves here and only here."""
        ...

    def describe(self) -> str:                                 # 2.4
        """The kind-ordered schedule listing for model.report
        (incl. why each self_update is scheduled and which
        implicit merges happened)."""
        ...


class BoundSchedule:
    """Per-step view: stage groups as (state, ctx) callables."""

    def context(                                               # 2.4
        self,
        clock: Clock,
        *,
        dt: jax.Array,
        stage_dt: jax.Array,
        sums: TendencySums | None = None,
    ) -> StepContext:
        """P0: build the frozen StepContext for a (sub)stage —
        eval_params(modules, clock.time) [D2], the stage clock,
        dt, stage_dt; sums attaches the per-treatment tendency
        sums for post-TENDENCY stages (§5.5)."""
        ...

    def prepare(self, state: State, ctx: StepContext) -> State:  # 2.4
        """S1 SELF_UPDATE + S1' DIAGNOSE stages, kind-ordered.
        S1'-placement is load-bearing: the first substage after
        set_state/restart recomputes diagnosed fields before any
        term reads."""
        ...

    def tendency(                                              # 2.4
        self, state: State, ctx: StepContext,
    ) -> TendencySums:
        """S2: accumulate EXPLICIT term contributions
        (deterministic order: module order, declaration order;
        VectorField.add; TermEvaluationError attribution)."""
        ...

    @property
    def implicit(self) -> tuple[BoundImplicitOperator, ...]:   # 2.5
        """The merged per-field implicit operators (kappa-summed
        framework families + at most one non-mergeable custom per
        field), bound to their module slots: each exposes
        .fields, .apply(state, ctx), .solve(rhs, dt_gamma, ctx)."""
        ...

    def advance_stages(                                        # 2.5
        self, state: State, ctx: StepContext,
    ) -> State:
        """S3': module-owned ADVANCE stages in kind order
        (barotropic subcycle); each writes its declared advanced
        subset plus own AUX; reads see the latest state
        (Gauss-Seidel by the §5.2 read rule)."""
        ...

    def constrain(self, state: State, ctx: StepContext) -> State:  # 2.4
        """S4 CONSTRAINT stages (projection replaces velocities,
        writes its own DIAGNOSTIC p = phi / ctx.stage_dt)."""
        ...

    def diagnostics(self, state: State, ctx: StepContext) -> State:  # 2.4
        """S6 DIAGNOSTIC stages (step-cadence accumulators; may
        read their own previous value — the accumulation idiom).
        Consumed by D4's chunk body in the per-step epilogue,
        after the S5 NaN reduction — never by the stepper."""
        ...


@fr.utils.jaxify                     # frozen; PROGNOSTIC-only vectors
class TendencySums:
    """Per-treatment tendency sums (the contribution-dict
    partition, consumed at accumulation time)."""

    @property
    def explicit(self) -> State:                               # 2.4
        """The summed EXPLICIT contribution (unprojected)."""
        ...

    def __getitem__(self, treatment: Treatment) -> State:      # 2.5
        """Sum by treatment; the IMPLICIT entry exists iff the
        driving scheme computed the forward applies this step
        (CNAB2), else KeyError — the increment-form default makes
        the barotropic stage independent of it (V-H4)."""
        ...
```

Semantics, invariants, error behavior:

- **The ctx-construction obligation**: every stage group call takes
  a ctx built by `context()` **at that (sub)stage's time** — the
  stepper must present the stage clock (`clock.shifted(c_i·dt)` for
  RK stages, the pre-tick clock for multistep tendency evaluation,
  the ticked clock for S3'/S4). `eval_params` at stage time is D2's
  signed design; a ramped scalar and a ramped N² profile see the
  same time as the stage that consumes them. `StepContext` itself
  (frozen, all-scalar leaves: `params`, `clock`, `dt`, `stage_dt`,
  the per-treatment sums for post-TENDENCY stages) is owned by
  [`module.md`](module.md); this cluster owns its construction and
  consumption sites.
- **`stage_dt` semantics**: the increment of the current advance —
  the full dt for multistep substages, the stage's effective
  increment for RK stages, the subcycle's own for module-owned
  ADVANCE stages. It is **signed** (it inherits dt's sign); the
  `p = φ/stage_dt` normalization and backward-run coherence are the
  2.7 test item (open question 1). It is *not* the solve's
  `dt_gamma`, which stays a separate positional (§5.8
  reconciliation 3).
- **Projection placement per family** (§5.6), as consumed through
  `constrain`: multistep (AB, IMEXMultistep) call it **once per
  step**, post-advance, post-S3'; explicit RK calls it per stage
  state and once for the final combination (§5.2's per-substage S4);
  the designed-for IMEX-RK projects per stage with implicit stage
  derivatives recorded pre-constraint (the sound internal
  optimization of d3_4 §1). History buffers always hold unprojected
  tendencies.
- **The epilogue split**: `step` never calls `diagnostics` — the
  chunk body runs the per-step epilogue (S5 `isfinite` reduction
  into the carried `panicked` flag, then S6) after `step` returns;
  this keeps the NaN mechanism (04 §6.3: flag + chunk-boundary
  abort, no `lax.cond` wrapper) out of every stepper.
- **One validation path**: the composer's assembly dry run and the
  halo trace wrap *state*; ctx is scalars — zero mimicry cost
  (§5.5). Attribution (module+term names on write-gate/key/space
  errors) is composer-owned and reaches steppers only as
  already-validated callables.

---

### Designed-for stubs

Specified so iteration 1 does not preclude them; **none are built**.

- **`AdaptiveRungeKutta`** (embedded pairs): `dt` moves from the
  stepper leaf **into its StepperState** (plus controller state);
  the chunk body's fixed-length scan gains a bounded
  `lax.while_loop` inner accept/reject loop; diffrax is the
  reference implementation. The `b_error` rows on the retained
  embedded tableaus (`HEUN_EULER`, `BOGACKI_SHAMPINE`, `RKF45`) are
  the data this stub keeps alive. Time-target runs stop reducing to
  static step counts under adaptive dt — a D4 interaction recorded,
  not designed. (§5.3, §5.7; d3_2 §4, risk 5.)
- **IMEX-RK tableau slots** (ARS(2,2,2)/ARS(4,4,3), build 2.7/3.1):
  self-starting (no warm-up, no rings — `StepperState = ()`),
  per-stage eval_params at `t + c_i·dt`, diagonal γ solves,
  per-stage projection, stage derivatives recorded post-solve
  pre-constraint; **assembly error when composed with a
  split-explicit ADVANCE stage** (multistep outer drivers only).
  Slots into `imex.py` beside `IMEXMultistep`, not into it.
- **SBDF3**: a third `IMEXMultistep` level tuple (γ = 6/11, depth-3
  rings) — pure data once SBDF2 ships.
- **Traceable `FixedPoint`** (owned by
  [`transforms.md`](transforms.md)): the OptimalBalance-style
  iteration drives this cluster only through already-signed
  surfaces — `model.reset()` → `stepper.init` re-warm per ramp leg,
  the signed dt leaf via `update_parameters` (rewarm required on
  sign flips), and the float64 clock. Nothing stepper-side is
  reserved beyond those; the pointer is recorded here so no future
  stepper change assumes forward-only, single-leg runs.

---

## Open questions

Genuinely unresolved residuals relevant to this cluster (carried in
[`../07_open_threads.md`](../07_open_threads.md) §9.3); decided
questions are not reopened.

1. **Backward-run sign conventions through `stage_dt`** (test at
   2.7): verify RK stage shifts `c_i·dt`, warm-up rows, the
   `p = φ/stage_dt` projection normalization, and the increment-form
   barotropic forcing all read the signed dt coherently on a
   backward leg. The named test lands with the cutover-parity suite.
2. **`add_prognostic`** — the key-aligned add of PROGNOSTIC-only
   vectors onto a full state (fields.md follow-up, alongside
   `VectorField.add`): every family's combine step in this file
   consumes it; owned by the grid-cluster fields spec.
3. **Chunked-scan buffer donation** (implementation): AB(order) /
   IMEX rings hold order PROGNOSTIC copies (same carry cost as the
   old `dz_list`); confirm donation covers the rings across chunk
   boundaries.
4. **Adaptive designed-fors** (parked until a user demands them):
   controller-state contents, the dt-in-StepperState packaging, and
   the D4 time-target interaction for `AdaptiveRungeKutta`; the
   RK/IMEX `time_discretization_effect` successors.

## Deviations (called out per the README rule)

- **D-1** (`TimeStepper.step`): the `schedule` argument arrives as a
  `BoundSchedule` — the static schedule closed over the carry's
  current modules by the chunk body. Packaging of §5.3's signature;
  preserves arity and the unbound-`fn` aliasing rule.
- **D-2** (`AdamBashforth.eps`): spelled `eps: float | None = None`,
  resolving to 0.01 at `order=2` and raising on non-None elsewhere —
  the signature form of the signed order-2-only rule plus the old
  AB2 parity default.
- **D-3** (`IMEXMultistep.levels`): the switched warm-up tuple
  carries a fourth member, the CNAB forward-`apply` weight, present
  in d3_4's worked form but not in §5.4's "(weights, γ)" wording.
- **D-4** (SBDF2 coefficients): stated in solve-normalized form
  (γ = 2/3, state weights (4/3, −1/3), explicit (4/3, −2/3)) —
  arithmetic restatement of d3_4's unnormalized worked form to match
  the `(1 − dt_gamma·L)⁻¹` solve protocol.
- **D-5** (`CNAB2`/`SBDF2`): thin factory functions returning
  configured `IMEXMultistep` — the notes fix only the
  `fr.time_steppers.CNAB2(dt=...)` spelling.
- **D-6** (`LowStorageRK3`): the notes sign the class and its
  flagship role, not a coefficient set; pinned at 2.5 against the
  Oceananigans RK3 reference and fingerprinted via `coefficients`.
