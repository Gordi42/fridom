---
status: superseded
date: 2026-07-09
superseded_by: commits d06441d1 + d3309640 (2026-07-09)
---

# Readable linear-term DSL — proposal (HELD, not implemented)

> **Superseded on its premise (2026-07-09, recorded 2026-07-13).** The
> `LinearBlock`/`Coeff`/`Interp`/`Diff`/`Scale` IR this DSL lowers to
> was deleted the same day the proposal was held (`d06441d1` +
> `d3309640`); the four linear terms are now plain numeric
> `@fr.term(linear=True)` methods (see
> [`../plans/active/linear_term_blocks_plan.md`](linear_term_blocks_plan.md)).
> The line-number references below point at files that no longer
> exist. Kept for the front-end design and the flux-form-vs-scalar
> coefficient argument, which would still apply to any future linear
> DSL — but such a DSL would need a target IR built first.

Status: **proposal only** (Silvano chose to hold, 2026-07-09). The current
`LinearBlock` form stays. This note preserves the design + feasibility
finding for later.

## Goal

Let a module author write a linear term the way the equation reads —
`∂ₜp = -∂ₓ(c²·u)` — instead of the matrix-entry form:

```python
# today (linear_blocks.py IR):
LinearBlock("p", "u", Diff("x") @ Scale("csqr", CSQR), Coeff(const=-1))
LinearBlock("w", "b", Interp(), Coeff(param=DSQR, invert=True))
```

## The hard constraint (why the RHS can't be a lambda)

Each block is consumed by **two** paths, both of which a front-end must
reproduce losslessly:

- **numeric** — `apply_linear_blocks` (`linear_blocks.py:414-449`): folds
  `increment[out] += coeff.apply_numeric(op.apply_numeric(state[src], out_space, …))`.
- **symbolic** — `linear_blocks(model)` → `_assemble_raw`
  (`symbolic_eigen.py:241-250`): `coeff * op.eigenvalues(grid, src_space)`
  scattered into `L[out, src]` (the per-mode eigenvalue matrix).

So the RHS must be an **introspectable expression tree** that lowers to
`(out, src, op, coeff)`. Everything it needs is syntactic: the `(out, src)`
names, the ordered operator chain, and — critically — **where each
coefficient sits relative to the derivatives**:

- coefficient *inside* a derivative → flux form `d/dx(c²u)` → `Scale` (folded
  into the operator symbol as `gate(const)·Identity`, `linear_blocks.py:207`);
- coefficient *outside* → `c²·d/dx(u)` → `Coeff` (a scalar).

Same symbol, different last-ULP numerics; the shallowwater core relies on the
flux form (`core.py:44-48`). The DSL makes this a *visible* source
distinction rather than a hidden `Scale`-vs-`Coeff` choice.

## Recommended design (feasibility-verified, all 5 existing blocks lower exactly)

Two-layer: a new `linear_dsl.py` front-end (~250 lines) that **lowers to the
existing `LinearBlock` IR and calls `fr.linear_term`**. Zero changes to
`linear_blocks.py`, `symbolic_eigen.py`, `eigen.py` — the eigenvalue path is
untouched. Effort **M**, additive, per-call-site reversible migration.

Handles are grid-free (built at class-body time, like the blocks):

- `eq.fields("u","v","p")` → field handles;
- `eq.aux(name, const=ParamName)` → constant AUX-field coefficient (carries
  runtime field name + symbolic `ParamName`);
- `eq.param(ParamName)` → traced-scalar coefficient;
- `eq.d(axis)` → derivative operator (`eq.d("x")(expr)`);
- `eq.interp(expr)` → explicit interp escape hatch.

Arithmetic dunders build the tree; `eq.ddt(field) << rhs` binds a tendency
(`<<`, because `==` must return bool). `eq.linear_term(name, *equations,
advances=…)` lowers and delegates to the existing block-based `fr.linear_term`.

### Side-by-side

```python
# shallowwater gravity
u, v, p = eq.fields("u", "v", "p"); csqr = eq.aux("csqr", const=CSQR)
dx, dy = eq.d("x"), eq.d("y")
gravity = eq.linear_term("gravity",
    eq.ddt(u) << -dx(p),
    eq.ddt(v) << -dy(p),
    eq.ddt(p) << -dx(csqr*u) - dy(csqr*v),   # c² inside the divergence
)

# nonhydro buoyancy + restoring
w, b = eq.fields("w", "b")
buoyancy_force = eq.linear_term("buoyancy_force", eq.ddt(w) << b / eq.param(DSQR))
restoring      = eq.linear_term("restoring",      eq.ddt(b) << -eq.param(N2) * w)

# coriolis
u, v = eq.fields("u", "v"); f = eq.aux("f_coriolis", const=CORIOLIS_F0)
coriolis = eq.linear_term("coriolis", eq.ddt(u) << f*v, eq.ddt(v) << -f*u)
```

Lowering: `-dx(p)` → `Coeff(const=-1)`; `-dx(csqr*u)` → `Diff("x") @
Scale("csqr", CSQR)` with `Coeff(const=-1)`; `b/param(DSQR)` →
`Coeff(param=DSQR, invert=True)`; `-param(N2)*w` → `Coeff(param=N2,
sign=-1)`; a pure coupling with no derivative inserts `Interp()` (which
resolves to `Identity` when spaces already match, `linear_blocks.py:253-262`).
Python precedence groups `-a - b` before `<<`, so no parentheses are needed.

### Known limits (pre-existing IR limits, surface as lowering-time errors)

- param coefficient *inside* a derivative (`d("x", dsqr*u)`) — `Scale` only
  supports an aux field, not a traced param. No current term needs it.
- inversion by an aux/literal (`u/csqr`) — `Coeff.invert` is param-only.
- two source fields in one additive term — a real nonlinearity, correctly
  rejected (stays an opaque `@fr.term` closure).

Key files: IR `linear_blocks.py:370-519`; symbolic consumer
`symbolic_eigen.py:207-250` + `linear_blocks.py:549-619`; term stamping
`terms.py:66-201`; call sites `shallowwater2/modules/core.py:49-56`,
`nonhydro2/modules/stratification.py:32-38`,
`framework2/modules/coriolis.py:81-86`.
