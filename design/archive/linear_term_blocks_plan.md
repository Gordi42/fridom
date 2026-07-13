---
status: superseded
date: 2026-07-13
supersedes: none
superseded_by: commits d06441d1 + d3309640 (2026-07-09)
---

# Linear-term block signatures (the H1 pivot) — sub-plan

**Status: superseded, 2026-07-09 (record rewritten 2026-07-13).** This
plan scoped the term-system change that
[`../../decisions/blocksymbol_l_assembly.md`](../decisions/blocksymbol_l_assembly.md)
§3 flagged as the pivot for symbolic `L`-assembly (roadmap phase H1):
let a linear tendency term declare its `(out, src) → Operator` blocks so
`L` could be assembled from the operator algebra, with the numeric
tendency derived from the same blocks.

**It was built in full, then deliberately removed.** The block IR
(`LinearBlock`/`Coeff`/`Interp`/`Diff`/`Scale`), `TendencyTerm.blocks`,
the derived-`fn` fold, the R13 halo derivation and the analytic
`BlockSymbol` `L(k)` all landed; nothing consumed them. Both models kept
their hand-written `eigenmodes.py`, the production pressure projection
stayed a hand-composed `Div @ Lap⁻¹ @ Grad`, and the machinery was dead
weight, so it was deleted and the four linear terms were rewritten as
plain numeric `@fr.term(linear=True)` methods. The plan has no remaining
work. It is kept as the record of why the block route is not the shape
FRIDOM uses.

## Landed, then reverted

Built:

- `85e865f2` — round 4: linear-term block signatures + `BlockSymbol`
  (`model/linear_blocks.py`, `grid/operators/block_symbol.py`).
- `06111093` — round 4 completion: nonhydro/SW linear-term block
  re-authoring (the four terms of §2).
- `496c230b` — R13: auto-derive a linear term's halo from its block
  operators.
- `e650fe91` — `linear_term` factory, `fr.leaf`, bound-method hooks.
- `026f4c62` — round 6 (H1): symbolic `BlockSymbol` `L`-assembly + `eigh`
  (`model/symbolic_eigen.py`, Leray as a `BlockSymbol`).
- `114c5d72` — a readable linear-term DSL on top of the IR; held, never
  merged into the term surface (now
  [`../../archive/linear_dsl_proposal.md`](linear_dsl_proposal.md)).

Reverted, 2026-07-09:

- `d06441d1` — the four linear coupling terms (SW gravity, nonhydro
  buoyancy + restoring, the shared Coriolis rotation) become plain
  numeric `@fr.term(linear=True)` methods using `.to`/`.diff`,
  bit-faithful to the `apply_linear_blocks` fold.
- `d3309640` — delete `model/linear_blocks.py`, `model/symbolic_eigen.py`,
  `grid/operators/block_symbol.py` + `BlockMatrix.eigenvalues`,
  `TendencyTerm.blocks` and the R13 halo hook, and their tests. Reason:
  no consumer. Linear terms trace halos normally as pure field
  arithmetic.

## What the term system carries today instead

- **`linear=True` is the only linear metadata.** Terms declare it, and
  nothing else about their block structure: `model/terms.py`,
  `model/modules/coriolis.py`, `nonhydro2/modules/stratification.py`,
  `shallowwater2/modules/core.py` (gravity), `shallowwater2/modules/
  sadourny.py`, `nonhydro2/modules/advection.py` (background advection).
- **The selection surface shipped** (the plan's §5.4):
  `fr.model.terms.linear`, `model.variant(term_filter=...)`,
  `fr.model.linearize` — `model/term_predicates.py`.
- **`L` is honesty-gated, not symbolically assembled** (`f3ca888c`):
  `Module.linear_operator_gap` + `require_linear_operator` +
  `LinearOperatorGapError` refuse to hand an incomplete `L` to
  `linearize`, the eigenmode/projection/balance constructors — the job
  the plan's §4 constant-coefficient gate was meant to do, done through
  a module declaration instead of a `Coeff` spec.
- **`L(k)` comes from the H0 probe or from hand-written analytics.**
  `fr.model.numeric_eigenpairs` (`model/eigen.py`) probes
  `fr.linearize(model)` with unit impulses; `nh.eigenmodes.from_model`
  and `sw.eigenmodes.from_model` read `model.parameters` and build the
  analytic `A(k)` directly. Scalar `Symbol` survives for the pressure
  solve.

## Remaining

None in this plan. Two threads inherit from it and live elsewhere:

- Phase H1 in
  [`projection_eigenmode_roadmap.md`](../plans/active/projection_eigenmode_roadmap.md)
  §H (and the decision record §4) still names the symbolic `BlockSymbol`
  target and points here. That target is withdrawn — H1 should be closed
  against the probe + analytic path, or reopened with a named consumer.
- The IMEX-by-linearity / general linear-stability tooling the block
  signatures would have backed (§7) is unbuilt and would have to be
  re-motivated on its own before any block IR returns.
