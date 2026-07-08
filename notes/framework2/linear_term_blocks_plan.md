# Linear-term block signatures (the H1 pivot) — sub-plan

**Status: design sketch, 2026-07-08.** Scopes the term-system change
that [`blocksymbol_l_assembly.md`](blocksymbol_l_assembly.md) §3 flags
as the pivot for symbolic `L`-assembly (roadmap phase H1). Goal: let a
**linear** tendency term expose its `(out, in) → Operator` block
structure so `L` can be assembled from the operator algebra — with the
numeric tendency derived from the *same* blocks (single source of
truth). Backed by an audit of `terms.py`, the four linear terms, and
the `schedule.py` invocation path.

## 1. The shape today vs the target

A linear term is an **opaque closure** `fn(module, state, ctx) → dict`
(the Coriolis `du = f.to(u) * v.to(u)` etc.). It declares `advances`
(outputs) but **no inputs and no block structure**; `linear=True` is
stored and read by nobody (`fr.terms`/`fr.linearize` are stubs). Every
block coupling is buried in field arithmetic.

**Target.** Each linear term declares its blocks:

```
block = LinearBlock(out, src, op, coeff)
    out, src : prognostic field names (row, column of L)
    op       : an operator SPEC (grid-free) resolving to the retained
               fr.Operator that maps coeff-space(src) → space(out)
    coeff    : a Coeff spec — runtime source + symbolic constant + sign
```

From one `blocks=(...)` declaration, two consumers:
- **numeric** (the running model): `increment[out] += coeff · op(state[src])`;
- **symbolic** (`from_model`): `L[out,src] = coeff_const · op.eigenvalues(grid, src_space)` → a scalar `Symbol` placed in a `BlockMatrix`; terms sum.

Because `.diff()`/`.to()` resolve to the **interned singleton**
operators, `op(field)` is the identical object the closures build
transiently — the numeric path is bit-reproducible (see §3).

## 2. The four linear terms, re-authored

`Interp()` resolves to the `src→out` `LinearInterp`; `Diff(axis)` to the
`FiniteDifference`; `A @ B` is operator composition; `Scale(aux)`
multiplies by a constant AUX field (flux form).

```python
# FPlaneCoriolis  (coeff = the constant f0 via the f_coriolis aux)
blocks = (
    LinearBlock("u", "v", Interp(), Coeff(aux="f_coriolis", const=CORIOLIS_F0)),
    LinearBlock("v", "u", Interp(), Coeff(aux="f_coriolis", const=CORIOLIS_F0, sign=-1)),
)
# ConstantStratification
blocks = (
    LinearBlock("w", "b", Interp(), Coeff(param=DSQR, invert=True)),   # +b/δ²
    LinearBlock("b", "w", Interp(), Coeff(param=STRATIFICATION_N2, sign=-1)),  # −N²w
)
# sw DynamicalCore.gravity
blocks = (
    LinearBlock("u", "p", Diff("x"), Coeff(const=-1)),
    LinearBlock("v", "p", Diff("y"), Coeff(const=-1)),
    LinearBlock("p", "u", Diff("x") @ Scale("csqr"), Coeff(const=-1)),  # −∂ₓ(c²u)
    LinearBlock("p", "v", Diff("y") @ Scale("csqr"), Coeff(const=-1)),
)
```

The staggering retag is carried by each `op`'s `codomain()` (interp
Center↔face, FD Center↔Right) — no new bookkeeping. Note gravity's
`p←u` puts the `c²` **inside** the derivative (`Diff @ Scale`) to match
the current `diff(c·u)` ordering bit-for-bit (§3).

## 3. Numeric derivation & bit-identity

The derived `fn` lands the same increment dict at the one call site
(`schedule.py:261` → `sums.add(**result)`), so nothing downstream
changes. Bit-identity holds **iff** the block `op` composes the
coefficient in the same order as today:
- Coriolis / buoyancy / restoring: coeff is a pure multiply *outside*
  the operator — identical.
- gravity `p←u`: today is `diff(c²·u)`, so the block must be
  `Diff @ Scale("csqr")` (scale-then-diff), **not** `coeff=c², op=Diff`
  (which is `c²·diff(u)` — equal in exact arithmetic, off in the last
  ULP). Symbol is the same either way (`c²·k̂`), so this is purely a
  numeric-ordering choice.

**Rollout safety.** Land `blocks` first as the *symbolic* source only,
keeping the hand `fn`, plus a test asserting the block-derived increment
equals the hand `fn` on random state (guards against drift). Then flip
`fn` to derived and delete the hand body — the write-once pattern the
codebase already uses for `fn=None ⇒ derived from implicit.apply`.

## 4. Symbolic extraction & the constant-coefficient gate

`from_model` selects the linear terms (`model.variant(term_filter=
fr.terms.linear)`), reads each term's `blocks`, and for each block:
- resolve `op` to the retained operator, take `op.eigenvalues(grid,
  src_space)` → scalar `Symbol` (this is where phase C's per-operator
  eigenvalues are consumed);
- resolve `coeff` to a **constant scalar**: a `param=` reads
  `model.parameters[name]` (frozen `at_time` if `Ramp`); an `aux=`
  reads its declared `const=` param — **declining with
  `EigenbasisError` if the model does not provide it** (a
  `BetaPlaneCoriolis` provides no `coriolis.f0`; a variable-depth core
  no `csqr`). This is exactly today's `from_model` provides-implies-
  constancy gate, now expressed through the coeff spec.

Place each `coeff·Symbol` into `BlockMatrix[out,src]`; sum blocks across
terms (the dormant `base.py` product/sum/scale algebra, once
`BlockSymbol` implements matrix `@`/`+`/`*`). Result: `L(k)`.

**The pressure constraint stays out-of-band.** Terms expose only the
*raw* linear blocks (Coriolis + buoyancy + stratification + gravity).
For nonhydro, `from_model` composes the Leray `Symbol.inverse`
(`grad·(∇²)⁻¹·div`, phase D) onto the assembled raw `L` — so term
re-authoring is **model-agnostic and identical in difficulty** for SW
and nonhydro; only the `from_model` post-step differs.

## 5. API surface to build

1. **`LinearBlock` + `Coeff` + operator specs** (`Interp`, `Diff`,
   `Scale`, `@`) — grid-free records resolved at assembly when spaces
   are known (like `field_declarations`). Small new vocabulary module.
2. **`TendencyTerm.blocks` field + `@fr.term(blocks=...)`**
   (`terms.py:133-139`, `:189-198`) — the input/block slot terms lack
   today. Assembly validates each block's `(out∈advances, src∈state)`.
3. **Derived-`fn` builder** — turns `blocks` into the numeric closure
   (§3); the equivalence test; then retire hand bodies.
4. **`fr.terms.linear` + `model.variant`/`fr.linearize`**
   (`term_predicates.py`, currently stubs) — the selection surface
   `from_model` calls. This is wave-7-A work already on the plan.
5. **`BlockMatrix.eigenvalues()` + `BlockSymbol`** (phase H1) — consume
   the per-term block Symbols; assemble/sum into `L(k)`.
6. **`Coeff` constancy resolution** — the constant-scalar gate reusing
   provides-implies-constancy; a `ConstantSpace`/param check.

## 6. Staging & rollout

- **T1 — vocabulary.** `LinearBlock`/`Coeff`/op-specs + `TendencyTerm.
  blocks` + assembly validation. No behavior change (blocks unused).
- **T2 — SW first.** Add `blocks` to `gravity` + `FPlaneCoriolis`;
  equivalence test; SW `L` is a **pure** `BlockSymbol` (no inverse), so
  once phase C + `BlockMatrix.eigenvalues` exist, `sw.eigenmodes.
  from_model` assembles `L` symbolically and `eigh(iML,M)` must
  reproduce the analytic Tier-0 modes.
- **T3 — nonhydro.** Add `blocks` to `ConstantStratification`
  (+ shared Coriolis, done in T2). Needs phase D (`Symbol.inverse`) for
  the Leray post-step in `from_model`.
- **T4 — derive `fn` from `blocks`** across the linear terms; delete the
  hand bodies; the equivalence test becomes the regression.
- **Rollout reach.** Only linear terms need `blocks`: today the four
  above (Coriolis shared, stratification ×2, gravity). Nonlinear terms
  (Sadourny, advection) are untouched — they stay opaque `fn` and are
  simply not `linear`, so `fr.terms.linear` skips them.

## 7. Dependencies & what it unblocks

- **Depends on.** Phase C (per-operator `eigenvalues`) for the symbol of
  each block; the wave-7-A selection surface (`fr.terms.linear`,
  `variant`); phase D only for the nonhydro Leray post-step.
- **Unblocks.** Phase H1 (symbolic `BlockSymbol` `L`), and — as a
  bonus beyond eigenmodes — a real backing for `linear`/IMEX
  partitioning and general linear-stability tooling, since every linear
  term now carries its operator, not just a closure.
- **Interface stability.** `Eigenmodes.from_operator(L, M, grid)` is
  unchanged; T2–T4 only swap how `L` is produced. The H0 probe
  bootstrap remains available until T2–T3 land.
