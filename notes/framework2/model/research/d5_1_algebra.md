# D5.1 — StateTransform: base surface, algebra semantics, laws

Research report (see [`README.md`](README.md) for status).

## 1. The base surface

```python
class StateTransform:
    domain:   StateSignature      # (grid identity, ordered PROGNOSTIC (name, space) table)
    codomain: StateSignature
    traceable: bool               # Tier 1 iff True; ANDs under composition
    idempotent: bool = False      # declared, never detected

    def __call__(self, state) -> State
    def call_with_info(self, state) -> tuple[State, TransformInfo]
    def cost(self) -> TransformCost          # model_steps; compositions sum
    def __repr__(self) -> str                # composition tree w/ tier + cost
    @property
    def complement(self) -> StateTransform   # Identity() - self; idempotent only
```

- Tier-1 transforms are **jaxify-registered frozen pytrees**
  (structure static; numeric captures — eigenvector data, Shift's
  state0 — dynamic leaves): the `fr.Ramp` pattern, which is what
  makes them jit/vmap-passable. Tier-2 transforms are host objects,
  never pytrees (they own a Model).
- **Info: `call_with_info`, `last_info` rejected** — a mutating
  attribute is silently wrong under jit and makes consecutive calls
  order-dependent (the D1.5-shim argument); callbacks stay as
  progress plumbing only (may observe, never influence). Law:
  `T(s) == call_with_info(s)[0]` bitwise; default implementation
  returns `EMPTY`. Info **composes structurally** (a tree mirroring
  the composition tree — OB's convergence record addressable by
  path). Fields: iterations, errors, model_steps,
  elapsed_model_time, extra. The old `return_details=True` dies.
- `cost()` counts **internal model steps** (Tier 1: 0; FixedPoint:
  upper bound flagged); `on_progress` hooks on Tier-2 constructors,
  path-prefixed through composites; the repr is the composition
  tree with tier/cost annotations (the cost-opacity answer).

## 2. The algebra, pinned

All signatures are concrete at construction → **every check is
eager** (+ a call-time recheck; for Tier-1 under jit it runs at
trace time — zero steady cost).

| Spelling | Typing rule |
|---|---|
| `A @ B` — `(A@B)(s) = A(B(s))` | `B.codomain == A.domain` |
| `A + B`, `A - B` — pointwise on outputs | domains equal AND codomains equal |
| `a * A`, `-A` | plain scalar only |
| `A ** n` | `n==0` → Identity(A.domain); `n>=2` needs endo; `n<0` raises |

- `Identity()` signature-polymorphic until composed/called; elided
  in chains. **No `Zero`** (no block layouts at state level).
- **Normalization is structural only** (flatten Compose-of-Compose /
  Sum-of-Sum, elide Identity) — no rewriting, no idempotent folding.
- `Shift(state0)`: `s ↦ s + state0`; Tier 1; the affine OB piece.
- **Scalar × Ramp: no** — transforms are autonomous maps, no clock
  in the surface; `TypeError` with "evaluate the ramp explicitly".
  Field-valued coefficients: designed-for.

### FixedPoint

```python
fr.transforms.FixedPoint(T,            # transform OR factory State -> StateTransform
    tol=1e-9, max_it=3,
    norm=fr.transforms.relative_l2,    # Callable[[State, State], float]
    on_divergence="stop_best",         # | "raise" | "ignore"
    on_iteration=None)                 # host progress hook
```

- `max_it==0` → input unchanged; `tol==0` → run max_it.
- **Divergence: keep the old rule, fix the old bug** — old OB
  stopped on `err > prev` *after* assigning the worse iterate and
  returned it; `"stop_best"` returns the argmin-error iterate; info
  records errors + stopped_by + returned_iteration.
- **The factory form absorbs `update_base_point`**: a callable
  `State -> StateTransform` evaluated on the current iterate per
  iteration (pure given frozen captures).
- Host-level (`traceable=False`) unconditionally in it-1; a
  `lax.while_loop` variant is designed-for, no consumer.

### The norm — resolved

Old stopping criterion: `norm_of_diff = 2‖a−b‖₂/(‖a‖₂+‖b‖₂)`
(volume-weighted, all components — verified `vector_field.py:470`).
D1 evicted norms from fields, so: **`fr.transforms.relative_l2`** —
a plain diagnostic function implementing that formula over
PROGNOSTIC components, shipped as the **explicit-kwarg default**
(relative + dimensionless → the mixed-units objection is largely
moot for a stopping criterion; reproduces old behavior;
parameter-free — which is exactly why the energy norm *cannot* be
default). `norm=nh.diagnostics.energy_norm(model)` one kwarg away.
A model-supplied default norm rejected (re-couples the combinator to
the model; invisible criterion).

## 3. The laws, hardened

**Law 1 — determinism.** Tier-2 `__call__` is normatively
`m.reset(); m.set_state(s); m.advance(N); return m.state.prognostic`.
Pinned: **PROGNOSTIC-only read-back** (AUX is the twin's
parameterization, DIAG is scratch); **the final clock is not state**
— it goes to info; frozen config = parameter leaves snapshotted at
construction (later parent updates don't propagate;
`T.with_parameters` designed-for). Proof chain: §6.5's
`reset(); set_state(z) ≡ fresh assembly + set_state(z)` + advance
purity + projection read-back. **New load-bearing invariant**:
`reset()` leaves AUX untouched, so a time-dependent AUX field holds
the previous call's end-time value across calls — determinism
survives *only because* SELF_UPDATE runs first in every substage
(the owner recomputes from the reset clock before any consumer
reads). Promote to a 02_rules invariant + regression test
(two consecutive transform calls, bitwise equal).

**Law 2 — signatures.** Compared: grid **object identity** + the
ordered PROGNOSTIC `(name, space)` table (names by equality, spaces
by identity, order included — strict now, relaxable later). Checked
eagerly at compose time AND at call time.
`SignatureMismatchError` carries the composition-tree path, a
componentwise diff table, and the grid-identity verdict with hints.
*(Reconciled with d5_3: family-built Tier-1 transforms declare a
mapped subset + a `rest` policy — see the consolidated design.)*

**Law 3 — isolation by construction.** A Tier-2 transform **never
stores the model you pass** — the constructor treats it as an
assembly spec and builds its own via `model.variant(...)`; there is
no injection path, so two transforms never share a model (no flag,
no lint). Old `deepcopy(mset)` maps to variant re-assembly (same
frozen grid, shared jit cache, snapshot semantics). Internal models:
`io=()`, named for log attribution, and a **trace guard** — Tier-2
`__call__` raises a taught `TraceError` on tracer-valued inputs.

**Idempotency**: `idempotent: bool`, declared never detected;
exactly three consumers — `complement` sugar (gated), a lint-level
`P @ P` hint (never rewriting), and a generic
`assert_idempotent` validation check. A `linear` flag deferred (no
consumer).

## 4. Traceability mechanics

- AND-rule per combinator; Identity/Shift `True`; FixedPoint
  `False` (it-1); Tier-2 `False`.
- **Stages vs Tier-1 transforms — ruled a use relation, not
  competition**: the pressure projection **stays a stage** (not
  autonomous — consumes `stage_dt`); any stage body **may call a
  Tier-1 transform** under three conditions: constructed at
  assembly/bind time as static structure; signature matches the
  owning model; `traceable=True`. Sanctioned example: a relaxation
  CONSTRAINT stage nudging only the balanced part via
  VorticalProjection.
- **vmap over Tier 1 works**: frozen pytrees + leading batch axis
  on State leaves + identity-hashed statics; sharding composes;
  ensembles through Tier-2 are a Python loop (batched internal
  models designed-for).

## 5. Precedents and risks

Precedents: optax `GradientTransformation` (pure, everything
returned — `call_with_info` support), equinox (frozen callable
pytrees = Tier-1 registration), PETSc `PCCOMPOSITE`
(multiplicative/additive composition as declared types; property
flags exploited never rewritten-by = the idempotent pattern),
deal.II `LinearOperator` (eager checks, identity elision),
scikit-learn Pipeline (inspectable steps transfer; fit/transform
split does not). No one ships a state algebra — novelty confirmed;
the small pinned surface is the over-engineering guard.

Risks: strict signature equality bites at tracer-augmented models
(→ the d5_3 `rest` ruling / `OnComponents` designed-for); composed
Tier-2 memory (one carry each — repr/cost mitigation; pooling
rejected); frozen-config staleness under sweeps
(`with_parameters` designed-for); default-norm unit mixing
(documented; energy norm one kwarg away); traceable FixedPoint
(no consumer); the SELF_UPDATE-first invariant (02_rules + test);
batched Tier-2 ensembles (designed-for).

## 6. Sketches

(User workflow + OB-in-the-algebra + the vmap IC-ensemble — see the
consolidated design; key lines:)

```python
P_vort   = nh.transforms.VorticalProjection(model)     # Tier 1
wave     = P_vort.complement                            # Identity() - P_vort
balanced, info = average.call_with_info(state0)

def ob_map(z_base):
    exchange = fr.transforms.Shift(z_base) @ (fr.transforms.Identity() - P_vort)
    return exchange @ fwd @ P_vort @ bwd
OB = fr.transforms.FixedPoint(lambda s: ob_map(P_vort(s)), tol=1e-9, max_it=3)

balanced_batch = jax.jit(jax.vmap(P_vort))(batch_of_states)   # 64 ICs, one call
```
