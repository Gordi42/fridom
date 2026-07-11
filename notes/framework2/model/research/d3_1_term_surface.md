# D3.1 — The TendencyTerm surface and treatment declarations

Research report (see [`README.md`](README.md) for status).

## 1. The term surface

**Recommendation: declared frozen data objects (`TendencyTerm`), with
a class-level decorator as the standard spelling and an overridable
`tendency_terms()` hook for bind-dependent cases** — the
FieldDeclaration pattern applied to behavior: the *declaration* is
transient assembly data; the *behavior* is an unbound method
referenced by it.

```python
@dataclass(frozen=True)
class TendencyTerm:                    # transient assembly input
    name: str                          # "Module.name/term.name" is the attribution key
    fn: Callable | None                # UNBOUND (module, state, ctx) -> {field: ScalarField}
                                       # None allowed iff implicit set (explicit path derived)
    treatment: Treatment = EXPLICIT
    advances: tuple[str, ...] | None = None   # None -> derived at dry run; declared -> checked
    transports: tuple[str, ...] = ()          # coverage-lint intent
    implicit: ImplicitOperator | None = None  # required iff treatment == IMPLICIT
    linear: bool = False               # strict: linear in state at fixed params/aux
```

`Module.tendency_terms()` defaults to scanning for `@fr.term`-stamped
methods in definition order (trivial modules: zero ceremony);
modules with constructed terms override it. Collection runs after
`bind(table)`, so `advances` may come from role selections.

Why declared objects:

- **Introspectability**: the composer needs
  `advances`/`transports`/`treatment` per term *before* evaluating —
  for the coverage lint, the treatment partition, and D3.4's
  by-variable table. A single returned dict hides all of it.
- **The jax-aliasing point (load-bearing)**: `fn` is stored
  **unbound**, never a bound method — a bound method captured at
  assembly closes over the *assembly-time* module instance while
  live parameters ride the carry's unflattened copy (the D2 trap
  again). The composer records `(module_slot, term)` pairs and calls
  `term.fn(carry.modules[slot], state, ctx)` inside the trace.
- **Halo tracing per term**: terms are separately callable
  plain-Python-over-fields functions — `trace_halo` runs each on the
  tracer state individually, un-jitted, giving a per-term HaloSpec.
- **Terms are not pytrees** — consumed at assembly, discarded; the
  composed step closes over slot indices and unbound functions.

**Signature packaging**: a frozen `StepContext` (`ctx.params`,
`ctx.clock`, later `ctx.dt`) as the uniform third argument;
`dt_gamma` stays a separate positional on solves (per-solve, not
per-stage). Confirmed by D3.3.

**Boundary restated**: terms only ever **add**; anything that
overwrites (the projection trio) is a **stage**. No write-anything
escape for terms.

## 2. Treatment vocabulary and the override story

**Exactly `EXPLICIT | IMPLICIT` in it-1** (closed enum; an
`EXPONENTIAL` treatment for exponential integrators is the known
future candidate — recorded). Constraints are stages, not a
treatment; subcycling is a stage concern.

**Treatment is author-declared, with the user override on the module
constructor — not the Model**: `VerticalMixing(kv=1e-4,
treatment=fr.IMPLICIT)` — the Oceananigans
`time_discretization=` precedent. The author mediates because only
the author knows whether an implicit path exists. A Model-level
override dict is rejected (string-keyed action at a distance that
can demand a solve the author never wrote); if ever wanted it may
only toggle terms whose declaration already carries `implicit` —
designed-for.

Validation: `IMPLICIT` without `implicit=` → assembly error;
steppers declare `supported_treatments` — an implicit term under
AB3 is an **assembly error, never silent demotion** (silently
integrating a stiff term explicitly changes the stability region).

**Write-once ergonomic**: a term with `implicit=op` and
`treatment=EXPLICIT` may omit `fn` — the composer derives the
explicit contribution from `op.apply`. The author writes L once;
flipping treatment cannot desynchronize the paths (Dedalus's
LHS-declared linear part, mechanically).

## 3. The implicit-solve interface

```python
class ImplicitOperator(Protocol):
    fields: tuple[str, ...]        # PROGNOSTIC components the implicit part advances (mandatory)

    def apply(self, module, state, ctx) -> dict[str, ScalarField]:
        """L·state — the explicit/forward evaluation (CNAB rhs, derived-fn path)."""

    def solve(self, module, rhs: dict[str, ScalarField],
              dt_gamma, ctx) -> dict[str, ScalarField]:
        """(1 - dt_gamma·L)^{-1} rhs; keys exactly `fields`."""
```

- **Block-structured**: common case length-1 (vertical tracer
  diffusion); a coupled block (semi-implicit Coriolis (u,v)) is one
  operator with two fields — atomic under by-variable splitting
  (feeds D3.4's group rules).
- `dt_gamma` is a **traced scalar supplied by the stepper** (CN:
  dt/2; SBDF-k: γₖ·dt; IMEX-RK: aᵢᵢ·dt) — warm-up γ-switching and
  future adaptive dt never retrace. Coefficients read from
  `ctx.params` at stage time (Ramp-correct).
- State-dependent coefficients (CATKE-like): evaluated on the passed
  state (lagged/predictor); the solve stays linear — 02_rules entry.

**Mergeable families (the composition problem).** Two closures
contributing implicit vertical diffusion on the same `b` cannot be
sequentially solved (that is Lie splitting inside one IMEX stage —
order-degrading). Oceananigans' answer adopted: closures expose
*coefficients*, the framework sums them into **one** solve:

- `fr.implicit.VerticalDiffusion(axis, fields, kappa=fn)` — the
  framework-owned tridiagonal family; same-axis operators touching a
  field merge by summing κ (valid: `(1 − dtγ(L₁+L₂))` *is* the
  combined operator); one Thomas solve per field; boundary rows from
  the field's declared space BCs; flux BCs enter as explicit forcing.
- `fr.implicit.SpectralDiagonal(symbol=...)` — diagonal in a
  coefficient space; merges by summing eigenvalues (iteration-2
  unless the sw semi-implicit case pulls it in).
- Custom protocol implementations are the escape hatch, with the
  rule: **at most one non-mergeable implicit operator per field**
  (`ImplicitCollisionError` naming both terms).

**Halo/shard story**: the tridiagonal solve is a grid-bound
registry-dispatched **Operator** declaring
`layout local along the solve axis` — negotiation keeps z LOCAL (the
decomposition notes' expected outcome) or plans pencils +
redistribute; HaloTracer intercepts it generically; no `.data`
bypasses, no `Module.extra_halo` for the standard families. This
retires the `RFFTPressureSolver` hand-built-sharding pattern. No
assembly-time factorization caching in it-1 (dt_gamma and κ traced);
noted as a constant-coefficient optimization.

## 4. `advances`: declared-optional, dry-run-verified

Under jit, contribution-dict keys are necessarily static — dry-run
derivation is fully reliable — so: `advances` optional-declared,
always cross-checked (mismatch = assembly error; catches
wrong-but-valid-component bugs); `transports` declared-only (intent,
not derivable: diffusion and advection both write `b`, only one
transports it — feeds the D1.4 coverage lint); implicit `fields`
mandatory (they shape the solve dict). The assembly dry run
evaluates **each term separately** on tracers, validates write-gates
and keys, backfills/cross-checks `advances`, runs the coverage lint,
and retains the per-term table as D3.4's by-variable input.

## 5. Error attribution — the TendencyComposer

All attribution lives in the composer (assembly-step-4 object);
neither `VectorField.add` nor `Module` grows any of it:

1. **Assembly dry run** (on tracers, before allocation): key/
   write-gate validation, advances cross-check, implicit-collision
   and treatment/stepper compatibility, and space validation —
   `SpaceMismatchError` wrapped with module/term names, raised
   before allocation.
2. **In-trace accumulation**: per-term try/except re-raising as
   `TermEvaluationError(module=..., term=..., cause=...)` — executes
   at trace time only, zero runtime cost.
3. **Deterministic order**: (module order, declaration order) —
   float summation is not associativity-stable; stable order keeps
   jaxprs and restart fingerprints reproducible.

## 6. Linear tag: adopt now

`linear: bool = False`, strict semantics ("linear in the state at
fixed params/aux; state-independent forcing is NOT linear").
Justification: `IMPLICIT` already presupposes a linear L; the
consumers are committed by D2.4 (linearized-model presets, eigenmode
seams, OptimalBalance's model pair); it is one field of frozen
assembly data, and **verifiable** (debug lint: evaluate at s and 2s,
compare). Not built: richer vocabulary, automatic linearization
(jax `jvp` covers mechanical needs; the tag records exactness).

## 7. Risks and open questions

- `StepContext` packaging confirmed D3-wide by D3.3.
- Merging limits: two *custom* implicit operators on one field stay
  an error (composite solves are research-grade).
- **Restart fingerprint must include per-term treatments** (a
  restart under a flipped treatment would reuse incompatible
  stepper history) — amend the D1 restart rule with the D2 Ramp item.
- Coupled implicit blocks are atomic under by-variable splitting —
  constrains D3.4's grouping.
- `extra_halo` stays per-module; per-term is a trivial extension if
  needed.
- Terms cannot write DIAGNOSTIC intermediates (write-gate); a
  closure exposing κ for IO owns it as AUXILIARY via self_update —
  verify ergonomics when porting the first real closure.

## 8. Sketches

```python
@partial(fr.utils.jaxify, dynamic=("kh", "kv"))
class HarmonicMixing(fr.Module):
    def __init__(self, kh=0.0, kv=0.0, fields=None,
                 vertical_treatment=fr.IMPLICIT):        # user's run-level choice
        self.kh, self.kv = kh, kv
        self._explicit_fields = fields
        self._vt = vertical_treatment

    def bind(self, table):
        self._targets = self._explicit_fields or table.select(fr.roles.TRACER)

    def tendency_terms(self):
        return (
            fr.TendencyTerm(name="horizontal", fn=type(self).horizontal,
                            advances=self._targets, linear=True),
            fr.TendencyTerm(name="vertical", treatment=self._vt,
                            advances=self._targets, linear=True,
                            implicit=fr.implicit.VerticalDiffusion(
                                axis="z", fields=self._targets,
                                kappa=type(self).vertical_kappa)),   # fn=None: derived
        )

    def horizontal(self, state, ctx):
        kh = fr.resolve_at(self.kh, ctx.clock.time)
        return {n: kh * (state[n].diff("x", 2) + state[n].diff("y", 2))
                for n in self._targets}

    def vertical_kappa(self, state, ctx, field):
        return fr.resolve_at(self.kv, ctx.clock.time)    # coefficients, not a solver


@partial(fr.utils.jaxify, dynamic=("n2_input",))
class ConstantStratification(fr.Module):
    @fr.term(advances=("w",), linear=True)
    def buoyancy_force(self, state, ctx):
        return {"w": state["b"].to(state["w"].space) / ctx.params["nonhydro.dsqr"]}

    @fr.term(advances=("b",), linear=True)
    def restoring(self, state, ctx):
        return {"b": -(state["w"] * state["n2"].to(state["w"].space))
                     .to(state["b"].space)}
    # two separate terms deliberately: per-field granularity for by-variable
    # splitting; a semi-implicit gravity-wave variant is ONE term with a
    # coupled ImplicitOperator(fields=("w","b")) — atomic.


@fr.utils.jaxify
class FPlaneCoriolis(fr.Module):
    @fr.term(advances=("u", "v"), linear=True)
    def coriolis(self, state, ctx):
        f, u, v = state["f_coriolis"], state["u"], state["v"]
        return {"u":  (v * f.to(v.space)).to(u.space),
                "v": -(u * f.to(u.space)).to(v.space)}
```

Sources: Oceananigans closures docs + scalar_diffusivity.jl
(`time_discretization=` constructor precedent; coefficient-merging
into one batched tridiagonal); Dedalus (IMEX LHS/RHS split; the
declared linear part); SciML SplitODEProblem types.
