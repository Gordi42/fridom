---
status: active
date: 2026-07-13
---

# Boundary-closure plan — what is left

Most of this plan shipped on 2026-07-11 (merge `9a95202a`). What
remains is **2e** (the Robin dynamic-data path) and **2f** (an
optional halo-claim refinement). The governing principle and the
signed-off decisions are kept below because 2e is built on them.

## The principle (governs the rest)

**The space key carries only the boundary's *structure* — the
per-side closure kind that determines DOF counts, fill stencils, and
row legality. Every boundary *value* is dynamic data. The storage
layer never invents values: each closure is (a) structural (mirror
fills), (b) data-parameterized (declared boundary data through a
dispatch row), or (c) an explicitly chosen one-sided scheme. Nothing
else exists.**

A BC tag is a **wall-value claim, not a parity/extension claim**
(owner, walled-Sadourny work). That is why dev's kept-tag products
and `Inner` codomains are load-bearing for the machine-precision
walled conservation — and why the abandoned `framework2-boundaries`
stage 2b (kind-flipped `Outer` codomains, product BC-drop) was
**superseded and will not land**.

## Landed (merge 9a95202a, 2026-07-11; full suite 7225 green)

- **2a** `ea0b7bf7` — `BC.ROBIN` added structurally; per-side shape
  rule (Robin keeps its boundary DOF, like Neumann); mixed-kind
  tuples through `space_key`; per-side fill dispatch verified.
- **2c'** `0730f1a4` — **R1**: the BC-free bounded extrapolation
  fill is gone (`decomposition/tensor.py`); exterior reads on
  BC-free bounded axes raise `DispatchError` with BC-structured
  hints. BC-structured operands keep dev's tag-governed ghost-fill
  behavior unchanged.
- **2d** `ac8d98dd` — **R2**: `boundary="one_sided"` opt-in rows for
  FD and interp (exact moment-solved weights,
  `OperatorRequirements(layout="local")`, reset validity claims).
- **2b** — superseded, will not land (see the principle above). The
  BC-row coverage it was meant to provide exists on dev's semantics;
  `phase1_findings`' BC-row API gap is closed by 2a/2c'.

Truth now lives in the code: `spatial/bc.py`,
`spatial/decomposition/tensor.py` (`_boundary_ghosts`),
`spatial/operators/{finite_difference,interp,staggering}.py`, and
`tests/spatial/test_boundary_closures.py`. Rationale for R1 (the
`d²` counterexample, rejected alternatives):
[`../done/bc_free_boundaries.md`](../done/bc_free_boundaries.md);
grounding analysis:
[`../../research/boundary_design_explainer.md`](../../research/boundary_design_explainer.md).

Honest consequence carried by the landed design: a tagged field's
derivative output is BC-free (no tag can claim its wall value), so
chained exterior-needing ops require an explicit
`boundary="one_sided"` opt-in.

## 2e — dynamic boundary data (Robin α, inhomogeneous g)

**Open. No longer blocked**: it was gated on ROADMAP 2.2/2.3
(module-owned parameters, grid in the traced state), both
implemented 2026-07-08.

Today `BC.ROBIN` is structure-only: `_boundary_ghosts`
(`tensor.py:966`) raises `NotImplementedError` pointing here, and
`grid.sync(boundary_data=...)` (`grid.py:584`) raises the same way.
2e materializes that path.

### R3 (signed 2026-07-07/11) — kinds in the key, values dynamic

- The key holds per-side **kinds only** — never α, never g. A float
  in the interning key would recompile per value under the single
  jit (ROADMAP 2.4) and foreclose autodiff through BC parameters.
  Both constraints become tests (below).
- The homogeneous Robin fill is affine in the near-boundary DOF with
  coefficients from (α, dx); with α a *dynamic scalar* it is
  jit-stable across α values and differentiable.

### The data path — decided (b)

| option | verdict |
|---|---|
| (a) BC data as extra field pytree leaves | rejected: fattens every field, and arithmetic must define how BC data combines (fine for linear ops, undefined for products) — a correctness trap in the core type |
| (b) `("ghost_fill", space)` rows holding dynamic leaves | **adopted**: a grid-bound `GhostFill` operator per BC-structured space, registered at assembly, carrying (α, g) as dynamic jaxified leaves (the `ScaledOperator` precedent). The consumption-side sync dispatches it at the designed-for seam; modules own and update the values. `grid.sync(boundary_data=...)` becomes the explicit override of the same row |
| (c) homogeneous-only grid + lifting (`u = u_h + u_lift`) | kept as a *documented pattern* (it composes with (b), needs no mechanism), rejected as the only path: it cannot express Robin's α and pushes boilerplate onto every user |

### Robin derivatives: flux-form, not ghost chains

The derivative of a Robin field has no local mirror structure, so its
codomain is BC-free and — under R1 — exterior-needing follow-ups are
correctly blocked. The supported route for Robin diffusion is **flux
form**: the wall flux follows from the Robin data
(`u' = (g − u)/α`), so the flux-difference row closes the boundary
exactly — the `FluxDifference`-INNER precedent generalized to
data-parameterized closures.

### Work

- `GhostFill` operator + `("ghost_fill", space)` registry kind;
  seeding at grid assembly (coordinate the seam with the model
  layer's assembly).
- Homogeneous Robin fill in `_boundary_ghosts`' place, reached
  through the row rather than as a hardcoded mirror.
- Flux-form Robin closure row.
- `grid.sync(boundary_data=...)` resolves through the row (drop the
  `NotImplementedError`).
- Codomain kind mapping for the data-bearing rows (diff maps
  Dirichlet→Neumann-structured and Neumann→Dirichlet-structured per
  side; interp preserves the kind).

### Gates

- Robin diffusion (flux-form) against the analytic solution;
  inhomogeneous Dirichlet solved both ways (lifting vs the
  `ghost_fill` row) agreeing to rounding.
- jit stability: sweeping α over values compiles **once**
  (compile-counter test) and `jax.grad` through α runs — the two
  recorded Robin constraints, as tests.
- Full suite + forced-4.

### Out of scope (designed-for)

Robin coefficient-space pairs (no classical fast transform; later via
Chebyshev tau/Galerkin); mixed-kind transform pairs (quarter-wave
DST/DCT variants); Robin/mixed on 2-D unstructured boundaries
(blocked on the spaces.md boundary-mesh question).

## 2f — mirror-claim extension (perf, optional)

The halo-validity claim gate is still `mesh.periodic`
(`decomposition/halo.py:487`): a homogeneous Dirichlet/Neumann mirror
fill *does* commute with the stencil families, so bounded chains
currently re-sync where they need not. Extend the gate from
"periodic" to "fill commutes", plus side-aware validity, **only if
profiling of bounded chains warrants it**. Exchange-count gates would
extend to bounded chains.

Validity claims per closure kind (task-1.8 interaction):

| closure on the applied axis | claim | why |
|---|---|---|
| periodic wrap | consume (shipped) | copies commute bitwise |
| homogeneous Dirichlet/Neumann mirror | reset today; **consume is the 2f refinement** | mirror symmetry commutes with the stencil families |
| Robin / inhomogeneous (data-bearing, 2e) | reset | affine-in-data fills don't commute across space changes |
| one-sided rows (shipped) | reset | no fill to commute with |
| BC-free | n/a | exterior-needing rows don't exist (R1) |

## Ordering

2e is unblocked and next; 2f last or deferred indefinitely.
