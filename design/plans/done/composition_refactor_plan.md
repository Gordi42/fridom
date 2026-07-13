---
status: done
date: 2026-07-13
---

# Composition refactor — the realized-map category & one composition core

**Shipped 2026-07-08/09.** Extended
[`../../decisions/symbol_stack_design.md`](../../decisions/symbol_stack_design.md)
(the realized-map algebra) with the implementation decision: build the
realized-map category together with a consolidation of the composition
machinery that was duplicated across the operator, symbol, block-symbol
and state-transform algebras.

The composition core is in the tree. What the plan called S3–S4 was never
composition work — it is the `Banded` / mixed-representation capability
tier, which belongs to (and is tracked by) Phase I of
[`projection_eigenmode_roadmap.md`](../active/projection_eigenmode_roadmap.md).
This plan is closed on its own scope.

## Landed

- **S0a — the tag validator.** `compose_spaces` / `union_spaces` in
  `spatial/spaces/composition.py`; `Symbol.__matmul__` and
  `Symbol._elementwise` route through it, the two local helpers deleted.
  Behaviour-neutral, interned results unchanged. `04f59868`
- **S1 — the `RealizedMap` category.** `spatial/operators/realized.py`:
  `RealizedMap` (runtime-checkable Protocol: fixed `domain`/`codomain`,
  `__call__`, `@` `+` `inverse` `conj`), `RealizedComposite` (lazy,
  right-to-left) + `RealizedSum` + `compose_realized` (flatten,
  eager-fuse adjacent `Symbol`s, typecheck each adjacency through S0a),
  and the `BoundTransform` adapter. `Symbol` conforms and stays `@final`.
  Existing paths bitwise-unchanged. `efe9c312`
- **S1 — explicit materialization (decision A).** `_guard_operator`: a
  bare `symbol @ recipe` raises, pointing at `recipe.eigenvalues(grid,
  space)`. `efe9c312`
- **S2 — `SpectralSolve` is a composition.** It now builds `backward @
  symbol.inverse() @ forward` and delegates to the composite; the
  nonhydro pressure solver rides the same chain (`Symbol × field` for
  the `1/dsqr` weight). Bitwise-identical (`maxdiff == 0.0`). `0ce2663f`
- **S0b — `normalize_chain`/`normalize_sum`. Dropped (2026-07-09).** The
  three flatten implementations share only "flatten nested same-kind +
  collapse singleton"; the variations dominate (operator: `Zero` +
  interning + `Identity`; `RealizedComposite`: symbol-fusion +
  adjacency typecheck, no `Zero`; `StateTransform`: `Identity`, no
  `Zero`). A shared helper needs ~5 hooks over ~5 lines each — the
  over-abstraction the "small pinned surface" guard rejects. The
  load-bearing shared piece was the tag validator (S0a).

Tests: `tests/spatial/operators/test_realized.py`,
`tests/spatial/operators/test_spectral_solve.py`,
`tests/spatial/spaces/test_composition.py`.

## What the plan assumed and reality overruled

- **`BlockSymbol` is gone.** `d3309640` deleted the linear-block IR and
  the analytic `BlockSymbol` spectral path (nothing consumed it: both
  models keep hand-written `eigenmodes.py`, and the production pressure
  projection is a hand-composed `Div @ Lap⁻¹ @ Grad`). So S1's
  "`BlockSymbol` adopts the protocol / gains `__call__`+`inverse`" is
  moot, as is S4's "`BlockSymbol`-of-`Banded` nesting". Any future block
  algebra starts from the `RealizedMap` protocol, not from the deleted
  class.
- **The operator layer kept its own tag checks.** S0a deliberately left
  `OperatorSum.codomain` (strict all-identical) alone: routing it
  through the `Constant`-wildcard union would change behaviour. The
  operator/realized duplication that remains is therefore intentional,
  not debt.

## Not done, and not this plan's business

- **`Banded` as a `RealizedMap` (was S3).** `spatial/operators/banded.py`
  is still free functions (`second_difference_matrix`,
  `apply_along_axis`, `solve_along_axis`), consumed by
  `model/implicit.py`. Promoting it to a type is the substrate for the
  mixed `Fourier(x,y) ⊗ Chebyshev/Nodal(z)` block-diagonal solve (the
  per-mode banded z-solve `SpectralSolve` documents as deferred).
- **Variable-coefficient / wall-bounded eigenmodes (was S4).** Phase I
  of the eigenmode roadmap. Its prerequisites from this plan — the
  realized-map category, mixed transforms, `Symbol × field` — are all
  in. The design sketch in §4 of the pre-rewrite version of this file
  (`git show e1e8e537`) is the reference for the `Banded` type.
  **Both are tracked by Phase I of
  [`projection_eigenmode_roadmap.md`](../active/projection_eigenmode_roadmap.md)**;
  that roadmap's pointer at "S3–S4 of this plan" should be re-read as
  "on the realized-map layer this plan shipped". Neither is scheduled —
  they wait on an actual demand for boundary-trapped / vertical-
  structure modes.
- **S5 — `StateTransform` adopts Layer 0.** Mostly moot: the
  `normalize_*` half died with S0b. What survives is routing
  `model/transforms/algebra.py`'s signature check through a State-level
  `compose_spaces` — a small, optional tidy with no consumer asking for
  it. Not worth opening a branch for on its own.
