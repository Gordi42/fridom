---
status: frozen
date: 2026-07-17
---

# jax.grad through a model run — status, blockers, fixes

Research report (see [`README.md`](README.md) for status). Question:
does reverse-mode autodiff work through a full new-stack model run —
e.g. `d loss / d nu` for a viscous shallow-water or nonhydrostatic
integration? Method: gradient-vs-central-finite-difference experiments
(cpu, x64) against a worktree pinned at `d9a3a7c7`, plus a hazard
audit of the step path. Experiment scripts were throwaway; every
load-bearing spelling and number is inlined here.

## Verdict

**The differentiable kernel exists and is exact; the public driver is
not differentiable; three closure/advection formulas poisoned
reverse mode and are fixed on branches.**

- The per-chunk step `_chunk_body(record, n, carry, stepper)`
  ([`model.py`](../../src/fridom/model/model.py) ~644–711) is a pure
  `lax.scan` over the `ModelState` pytree. Coefficients (`friction.nu`,
  `mixing.kappa`, `nu4`, Smagorinsky constants, stepper `dt`) are
  dynamic pytree leaves (`leaf()`), read live from `ctx.params`, and
  field arithmetic deliberately accepts traced 0-d scalars — so the
  math is differentiable by construction.
- **nonhydro2**: `jax.grad` through a 25-step run — 30-iteration CG
  pressure solve included — matches central FD at machine precision
  for `nu`, `kappa`, `dt`, and the initial buoyancy field, linear and
  nonlinear alike (rel. err 1.5e-11…3.9e-10). The scan-based CG
  (krylov_scan_plan) differentiates transparently; no custom_vjp
  needed at current iteration budgets.
- **shallowwater2**: reverse mode was NaN for every parameter with a
  data path while forward values and forward-mode `jax.jvp` were fine
  (jvp–FD agreement ~1e-9). Cause: the Sadourny PV division
  `q = zeta / p_full.to(zeta)` divides by exact zeros in never-valid
  padding cells; the masked `0/0` is invisible forward but its VJP
  (`-a/b^2`, `b=0`) poisons every cotangent. Fixed (below); post-fix
  grad–FD agreement 5.8e-11 (`nu`) / 6.0e-11 (initial `p`),
  advection on.
- Two sqrt-at-zero hazards of the same masked-singular-VJP class:
  biharmonic coefficient split `kh ** 0.5`
  ([`diffusion.py`](../../src/fridom/model/closures/diffusion.py)
  354–355; NaN grad exactly at `nu4 = 0`, knife-edge — finite already
  at 1e-30) and the Smagorinsky buoyancy clip
  `_positive_part(...) ** 0.5`
  ([`smagorinsky_lilly.py`](../../src/fridom/nonhydro2/modules/smagorinsky_lilly.py)
  326–327; **any** clipped cell NaNs the whole gradient, which default
  settings hit routinely — with the clip inactive the full
  state-dependent eddy-viscosity path matches FD to 1.5e-10). Fixed
  (below); post-fix, clip active (79/512 cells), grad–FD 2.3e-10 and
  reverse/forward agreement 4.1e-16.

## Why the public path cannot be differentiated

`jax.grad` through `Model.advance` (or a model constructed inside the
loss) fails before any physics runs; the kernel bypass is required.

1. Closure constructors coerce eagerly: `HarmonicFriction(nu=tracer)`
   raises `TypeError` at `diffusion.py:121` (`float()` on the tracer).
   Post-assembly, parameters are written by name
   (`update_parameters`), so this is by design — but it rules out
   "rebuild the model inside the loss".
2. `step_chunk`'s AOT cache key reads `leaf.sharding`
   (`model.py:725`, `_leaf_signature`) — `AttributeError` on tracers.
3. Even past that: the chunk executable donates the carry
   (`donate_argnums=(2,)`, `model.py:765`) and `advance()` host-syncs
   `bool(carry.panic.flag)` per chunk (`model.py:1821`).
4. `Model` is deliberately a host object, never a pytree.

## The working recipe (current spelling, private surface)

```python
from fridom.model.model import _chunk_body

model = sw.Model(..., modules_extra=(
    fr.model.closures.HarmonicFriction(nu=0.02),))
model.set_fields(p=...)
record, carry, stepper = (model._artifacts.record, model._carry,
                          model._stepper)
nu0 = next(m.nu for m in carry.modules
           if type(m).__name__ == "HarmonicFriction")
leaves, treedef = jax.tree_util.tree_flatten(carry)
idx = next(i for i, l in enumerate(leaves) if l is nu0)

def loss(nu):
    new = list(leaves)
    new[idx] = nu
    out = _chunk_body(record, 100,
                      jax.tree_util.tree_unflatten(treedef, new),
                      stepper)
    return sum(jnp.sum(f.data ** 2) for f in out.state)

g = jax.grad(loss)(nu0)        # also: value_and_grad, jit(grad(...))
```

Notes: initial conditions substitute `carry.state[name].storage` (the
padded leaf; halo cells correctly receive ~0 gradient); `dt` is a leaf
on `stepper` (4th argument, loop-invariant); chunk composition is
exact (`loss(1×100) == loss(2×50)` bitwise); `jit(grad)` warm calls
are ~6–7 ms after a ~2 s cold compile on the toy sizes.

## Measured results (grad vs central FD, f64, cpu)

nonhydro2, 8³, dt 0.02, 25 steps (pressure solve in the loop):

| target | rel. err |
|---|---|
| `friction.nu` (linear) | 1.5e-11 |
| `mixing.kappa` (linear) | 2.8e-10 |
| `friction.nu` (advection on) | 4.2e-11 |
| biharmonic `nu4` at 1e-4 | 3.9e-10 |
| stepper `dt` | 3.3e-11 |
| initial `b` (dot test) | 2.1e-10 |
| Smagorinsky `Cs` (clip inactive) | 1.5e-10 |
| Smagorinsky `Cs` (clip active, post-fix) | 2.3e-10 |

shallowwater2, 16², dt 5e-3, 100 steps, advection on, post-fix:

| target | rel. err |
|---|---|
| `friction.nu` | 5.8e-11 |
| initial `p` (dot test) | 6.0e-11 |

Caveats found while validating: a buoyancy-only IC on the f-plane is
a motionless steady state, so `d/d nu` is a true ~0 (5e-21) — seed a
velocity before concluding "gradient broken"; the preset scalar
`csqr` is inert (the dynamics read the `csqr` *field*), so its grad
is a legitimate 0.

## Fixes (all merged to dev, 2026-07-17)

Landed as merges `a7de0701` (1), `8d6eb07b` (2), and `f2e3ca56` (3).
(3) also added a fourth guard — the immersed PV division
(`_advect_immersed`, which landed with I4 after this investigation
and reintroduced the flat-path hazard; pre-guard, reverse mode NaN'd
144/144 entries of the IC gradient; post-guard both immersed models
FD-match to 1e-10..1e-12) — plus the systematic autodiff regression
shards (`tests/model/test_model_autodiff.py`,
`tests/nonhydro2/test_nonhydro2_autodiff.py`, ~29 s cold / ~12 s warm
total) and the binding **Differentiability policy** in `AGENTS.md`
(one small grad-vs-FD test per new step-path feature).

1. `fix/sadourny-grad-safe-pv-division` @ `4f6e86b6`.
   `_potential_vorticity(zeta, thickness)` helper guards the
   denominator (`jnp.where(storage == 0, 1, storage)`) preserving
   `halo_valid`, applied to both the flat and chart PV divisions.
   Interior forward values bitwise identical (verified against a
   non-rest state); 316 shallowwater2 tests + ruff green; new
   regression shard `test_sadourny_autodiff.py` (grad finite +
   FD-matched). Perf: official `benchmarks/model` step guard,
   `sw_flat`/`sw_sphere` deltas within the base-vs-base noise band
   (`compare --fail-on-regression` exit 0).
2. `fix/sqrt-zero-grad-guards` @ `686cc016`.
   Biharmonic: `_biharmonic_root` double-where guard around the
   `kh/kv ** 0.5` split (plain `** 0.5` kept for the halo trace's
   demoted Python scalars). Convention: the gradient at exactly
   `nu4 = 0` becomes 0 (the true one-sided derivative is finite
   nonzero; NaN → 0 at a measure-zero point — an optimization started
   at exactly 0 stalls there; start at any positive value). Grad–FD at
   `nu4 = 1e-4`: 2.0e-12; perf flat (0-d coefficient).
   Smagorinsky: a first double-where guard on the strain field cost
   **+13% step time** (measured, non-overlapping bands) — reworked as
   a `custom_jvp` `_sqrt_clipped`: the forward primal stays a plain
   `** 0.5` (bitwise identical to base; interleaved A/B confirms the
   regression is gone) and the guard lives only in the tangent rule,
   which jax transposes for reverse mode (`custom_vjp` would have
   broken forward-mode `jvp`). In clipped cells 0 is the correct
   subgradient. Grad–FD 2.3e-10; jvp–grad agreement 4.1e-16.
   Gates: 50 closure/Smagorinsky tests + ruff green; regression tests
   cover grad-finite-at-zero, FD-match, and forward-mode survival.

   **Reconciliation note:** dev's merge `b447b8e5` accidentally
   absorbed a byte-identical copy of the `_biharmonic_root` hunk from
   a dirty working tree (evil merge — neither parent contains it), so
   the biharmonic half of branch 2 is already on dev *without its
   tests*; the Smagorinsky `custom_jvp` half is only on the branch.
   Merging branch 2 legitimizes the accidental hunk with tests (the
   identical hunk merges as a no-op).

## Remaining hazards (audited, unfixed)

- [`coriolis.py`](../../src/fridom/model/modules/coriolis.py) 185
  (`/ w.to(v)`, thickness-weighted rotation) and 243–244 (`/ w_1`,
  `/ w_2`, chart/metric rotation): same interpolated-thickness-zero-
  in-padding pattern as Sadourny; medium-high confidence by
  inspection, bites when `metric_weight` is set. Same guard applies.
- [`mapped_pressure.py`](../../src/fridom/nonhydro2/modules/mapped_pressure.py)
  748 (`/ jacobian`): static geometry, likely nonzero in padding;
  flag only.
- `metric ** 0.5` in shallowwater2 `state.py`/`diagnostics.py`:
  static positive metric, outside the tendency path; safe.

## Structural limits (by design, document rather than fix)

- **Memory**: reverse mode stores the whole scan tape — O(n_steps)
  (measured ~1.4 MB/step at 8³; 515 → 621 MB RSS for 25 → 100 steps).
  Production-resolution adjoints need `jax.checkpoint` on the scan
  body (a `propagator(remat=...)` knob, below).
- **ETDRK4** freezes `exp(L dt)` at assembly
  (`freezes_linear_operator=True`): gradients w.r.t. an L-parameter
  under it see a stale operator by design.
- Forward-mode `jax.jvp` works through everything tested, including
  pre-fix Sadourny (tangents follow the primal data path; only
  reverse mode crosses the sealing mask backwards).

## Recommendation: a public differentiable surface

The kernel is proven; what is missing is a supported spelling. A
`model.propagator(*, wrt=("friction.nu",), steps=..., remat=...)`
returning a pure `(params, state) -> state` function would need no
new machinery: `update_parameters` (`model.py:1628`) already resolves
dotted names through the binding table to `(slot, attr)` and writes
leaves functionally (`_replace_leaf`); the propagator is that
resolution + `_chunk_body` without donation. The forward-mode
counterpart (`TangentPropagator` over `model.tendency`) is already
anticipated in
[`04_run_loop_io.md`](../specs/model/04_run_loop_io.md) §D5. The
`custom_vjp`-via-IFT route for the CG solve
([`krylov_scan_plan.md`](../plans/done/krylov_scan_plan.md)
follow-ons) stays unnecessary until iteration budgets grow — the
scanned solve differentiates exactly today. An autodiff regression
gate (the new `test_sadourny_autodiff.py` pattern: grad finite +
FD-matched on a tiny run) per model package would keep this surface
from silently rotting.
