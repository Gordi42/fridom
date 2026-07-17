---
status: frozen
date: 2026-07-17
---

# A convergence stopping criterion for the CG pressure solve

Research report (see [`README.md`](README.md) for status). Question:
the mapped / immersed pressure solve
([`krylov.py`](../../src/fridom/spatial/operators/krylov.py)) runs a
**fixed** iteration budget (CS-D2, no tolerance break) so it
jit-compiles once and stays reverse-mode differentiable. On a
gently-mapped grid most of the default 30 iterations do nothing. Can we
stop early **without** losing the exact `jax.grad` invariant
(AGENTS.md, "Differentiability policy") or the O(1) trace/compile? Method:
profile the real mapped solve's contraction, prototype two early-stop
mechanisms, and gradient-vs-central-finite-difference them (cpu, x64).
Experiment scripts were throwaway; every load-bearing number is inlined
here.

## Verdict

**Yes — a masked `lax.scan` (each step wrapped in
`lax.cond(converged, no-op, real-step)`) stops early, keeps the fixed
scan structure, differentiates the *actual truncated algorithm* to
FD precision, and retains O(1) compile. Shipped opt-in
(`tolerance=None` default). The rejected `while_loop` +
`custom_linear_solve` alternative changes the gradient to
implicit-function-theorem semantics and carries a silent-wrong-grad
trap.**

## The solve wastes most of its budget

Measure-weighted relative residual `sqrt(<r,r>) / sqrt(<b,b>)` vs
iteration `k`, real mapped pressure operator, 16³, x64:

| terrain `H(x)`            | 1e-2 | 1e-4 | 1e-6 | 1e-8 | floor           |
|---------------------------|------|------|------|------|-----------------|
| gentle `1 + 0.2·sin x`    | k=3  | k=5  | k=7  | k=9  | ~3e-17 by k≈16  |
| strong `1 + 0.6·sin x`    | k=6  | k=10 | k=15 | k=19–20 | ~4.5e-14 at k=30 (still contracting) |

Contraction is **monotone** — no stagnation, no plateau — so a residual
threshold is a clean stopping rule. The counts are independent of the
time step and of the nodal/FV discretization family. On the gentle grid
the default 30 iterations means ~21 exact no-ops per solve; even the
strong grid finishes near k≈20. The immersed operator contracts more
slowly on genuine partial cells (8³, `min_fraction = 0.1`): 1e-4 / 1e-6
/ 1e-8 at k = 9 / 14 / 19.

## Two mechanisms, prototyped and differentiated

### Prototype B (chosen) — masked scan

Keep the fixed `lax.scan` of length `max_iter`; each body step is
`lax.cond(rr <= tol²·bb, identity, real_step)`. Once the residual clears
the threshold, every remaining step is a no-op.

- **`lax.cond` lowers to a real `stablehlo.case`** (a runtime
  conditional), *not* a compute-both `select`. So the skip is real:
  converge-at-5-of-30, 48³ stencil, cpu — **8.2 ms** (masked, early
  exit) vs **24.6 ms** (all 30 real steps), a ~3× forward speedup
  scaling with the fraction of steps skipped.
- **`jax.grad` is exact** — `scan` + `cond` differentiate the truncated
  algorithm. grad-vs-FD w.r.t. the RHS and w.r.t. an operator parameter
  agree to ~1e-9, with **no transpose / adjoint machinery**. Works
  sharded on forced-4 devices; an outer `lax.scan` (model-run shape)
  around the solve is grad-exact too. Forward-mode `jax.jvp` agrees
  with reverse mode (both AD modes stay alive).
- **Bitwise / perf neutral off.** With `tolerance=None` the code path is
  the pre-existing fixed-iteration recurrence, byte for byte (no extra
  dot, no cond). A tolerance so tiny it never fires runs the identical
  real-step arithmetic; measured **maxdiff 0.0** vs `tolerance=None`
  (preconditioned and unpreconditioned) — the extra `<r,r>` dot and the
  step counter do not feed the iterates, so neutrality is exact, not
  approximate.

### Prototype A (rejected) — `while_loop` + `custom_linear_solve`

A `lax.while_loop` with a dynamic trip count, gradient supplied by
`lax.custom_linear_solve`. Rejected on two grounds:

- **Wrong gradient semantics.** `custom_linear_solve` returns the
  *implicit-function-theorem* gradient of the exact solution, whose
  error is **proportional to the tolerance** — it does not
  differentiate the truncated iterate the forward pass returns. That
  breaks the repo invariant (grad exact to FD precision at the *actual*
  output).
- **A silent-wrong-reverse-gradient trap.** `custom_linear_solve(...,
  symmetric=True)` assumes Euclidean self-adjointness. The mapped /
  immersed operators are self-adjoint only in the **measure-weighted**
  product; `symmetric=True` then produces a **~70 %-wrong `grad_b`** on
  any such operator. It is correct *today* only by accident: the current
  computational measure is uniform (a constant XLA folds), so
  measure-weighted and Euclidean self-adjointness coincide — see the
  symmetry note below. `symmetric=False` avoids the wrong grad but not
  the IFT-error issue.

The `while_loop` also buys little raw speed at equal iteration count:
`while_loop / fixed-scan` runtime ratio **0.896** at 30 iterations
(48³), so the shorter trip count is the only win it offers, and the
masked scan captures that with the runtime-skip conditional.

## The symmetry finding (why the trap is dormant, and contingent)

The mapped and immersed flux-form Laplacians are **Euclidean-symmetric
to roundoff** in the current code — `<A p, q> == <p, A q>` in the plain
dot product, tolerance ~1e-12 relative — *because the computational
measure `grid.measure` is uniform* (a single constant that cancels in
`<A p, q>` vs `<p, A q>`). This is contingent on the mesh: a genuinely
stretched computational grid (non-uniform `measure`) makes the
operators self-adjoint only in the measure-weighted product, and any
Euclidean-symmetry assumption (`custom_linear_solve(symmetric=True)`,
or a Euclidean CG) goes silently wrong. The masked scan needs no
symmetry assumption at all — it runs the same measure-weighted
recurrence the fixed solve already uses.

## The T4 floor trap (a NaN, and how the tolerance removes it)

The fixed recurrence's `_guarded_ratio` guards only an **exact** zero
denominator. If a tolerance sits **below** the achievable residual
floor (~1e-14 conservative for a preconditioned f64 solve), the cond
never fires: the solve runs its full budget and the post-floor
iterations divide a tiny residual by a tiny residual — forward value
finite, **reverse gradient NaN**. This is a *pre-existing*
fixed-iteration hazard (the fixed 30-iteration solve NaNs the reverse
gradient the same way once it over-converges past the floor), not a new
one. A tolerance that actually **fires** stops before the floor and so
*removes* the hazard for gradient users. The docstring states the rule:
keep the tolerance above ~1e-14.

Corollary, discovered while wiring the model-level autodiff regression:
the **mapped** nonhydro2 model is *not* reverse-mode differentiable in
this geometry at all — a short mapped run NaNs the reverse gradient
w.r.t. an initial field or `dsqr` **identically for `tolerance=None`
and for a firing tolerance** (forward-mode `jvp` and FD are finite; the
NaN is a masked metric singularity in the mapped step path, unrelated
to this feature). The **immersed** CG consumer *is* reverse-safe, so it
carries the end-to-end tolerance autodiff regression
([`tests/nonhydro2/test_immersed_model_autodiff.py`](../../tests/nonhydro2/test_immersed_model_autodiff.py)):
a firing `pressure_tolerance = 1e-6` (fires ~k=14 inside a 20-step
budget) gives a finite FD-matched reverse gradient. The mapped
reverse-mode singularity is logged here as a separate, pre-existing
issue.

## Decision and why default-off

Ship the **masked scan**, opt-in, **`tolerance=None` the default**:

- **Results change at the tolerance level in existing configs.** Any
  currently-tuned run that relies on the exact fixed-iteration output
  would shift by ~tolerance if a default tolerance were introduced —
  an unrequested physics change.
- **A safe default tolerance is problem-dependent.** The floor trap
  makes "above the floor" mesh- and precondition-dependent; there is no
  single value safe for every mapped/immersed/precision combination.
- **House preference for explicit knobs over silent physics changes**
  (memory: *prefer explicit over risky auto-magic*).
- **The flat periodic default never needs it** — that path uses the
  exact spectral solve, not CG.

The knob is a keyword-only `tolerance: float | None` on
`ConjugateGradient`, threaded as `tolerance=` /
`pressure_tolerance=` through the mapped and immersed pressure solvers,
`DynamicalCore`, the `nh.Model` factory, and
`hy.ImplicitFreeSurface`. Semantics: stop once
`sqrt(<r,r>) <= tolerance · sqrt(<b,b>)` (`b` the projected RHS),
compared squared with the threshold formed once outside the loop; zero
RHS converges immediately. `solve()`'s `info["iterations"]` becomes a
**traced 0-d array** on the tolerance path (data-dependent) vs the
Python int on the fixed path. **Do not `vmap` the tolerance solve** —
under `vmap` `lax.cond` degrades to a compute-both `select`, forfeiting
the runtime skip (T5).

## Composition and follow-ons

- **Multigrid** ([`../plans/active/multigrid_pathway_plan.md`](../plans/active/multigrid_pathway_plan.md)):
  MG will be a *fixed linear preconditioner* inside this same CG. A
  strong MG preconditioner contracts the outer residual faster, so an
  outer-CG tolerance becomes **more** valuable (fewer outer steps to the
  threshold). A tolerance in the outer CG composes cleanly; tolerance-
  like adaptivity in the inner MG components stays forbidden (it would
  break the fixed inner trace).
- **Carried-`p` warm start** (recorded, undesigned): reusing the search
  direction / iterate across solves of a slowly-varying operator would
  cut the iteration count further — complementary to the tolerance, a
  separate follow-on.
- **GPU re-measure** (open): the 3× win is CPU micro-timing; XLA:GPU
  buffer assignment inside the real chunked model step can reverse
  standalone loop wins (as the padded-carry experiment did, krylov
  docstring). Re-measure on A100 before claiming the step-level win;
  entry in [`../roadmap/open.md`](../roadmap/open.md).
