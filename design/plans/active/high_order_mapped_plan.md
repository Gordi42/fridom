---
status: draft
date: 2026-07-12
---

# High-order stencils on mapped grids

Not started. The obstacle, the options, and the staged route for
lifting the mapped-mesh refusals recorded in
[`../../specs/grid/classes/operators_stencils.md`](../../specs/grid/classes/operators_stencils.md)
("Amendment (2026-07-12, stages C0–C4): mapped-mesh grounding"):
`FiniteDifference` order > 2, the one-sided FD closure, the biased
reconstructions (`WenoReconstruction`, `Fallback`, the nonhydro2
`UpwindAdvection`/`WENOAdvection`). Opened by the coordinate-systems
work ([`../done/coordinate_systems_plan.md`](../done/coordinate_systems_plan.md)
§8).

## 1. The obstacle

The biased schemes are Shu–Osher finite-difference **flux** schemes.
Two separate assumptions bind them to a uniform mesh:

1. the reconstruction weights (the Shu rows, the Jiang–Shu smoothness
   indicators) are derived for **uniform node offsets**;
2. the flux difference is divided by a **constant h**.

Stage C0 replaced (2) with the grid's measure field, which makes the
scheme *consistent* on a stretched mesh — and that is precisely the
trap. The measure field is a **two-point** difference of node
positions, i.e. a discrete Jacobian of order 2. Dividing a wide
uniform-offset stencil by it caps the composite at 2nd order: the
metric, not the stencil, sets the order. Measured (tanh stretching):
upwind-5 and weno-5 both 5.0 -> 2.0, upwind-3 3.0 -> ~2.6. Nothing
raised; the numbers just quietly stopped being what the scheme
promised.

The condition the curvilinear-FD literature imposes is the **discrete
metric identity** (free-stream preservation): the Jacobian must be
computed with the *same wide operator* that differentiates the flux,
so that a constant physical state has an exactly zero discrete
divergence. "Divide by the measure" satisfies it only at order 2,
which is why order 2 is the exact boundary of what C0 landed.

## 2. Options

| # | Route | Reuse | Risk |
|---|---|---|---|
| **(i)** | **Reconstruct in computational space**, then divide by an **order-matched discrete Jacobian** (the same wide row applied to the node coordinates). | High — the existing uniform-offset kernels are already the computational-space rows; only the divisor changes. | Low. Also retires the `FiniteDifference` order > 2 and one-sided deferrals in one move (same divisor). |
| (ii) | **Genuinely nonuniform WENO**: position-dependent Shu rows (per-cell weights from the actual node positions) plus rescaled Jiang–Shu smoothness indicators. | Low — new tables, new per-cell coefficient fields. | **High, and quiet**: a wrong beta scaling degrades shock capturing while smooth-order convergence tests still pass. The failure mode is invisible to the gates we would naturally write. |
| (iii) | **Chain rule**: differentiate in computational space, multiply by an analytic/derived metric factor. | — | For a *derivative* this is (i) with a different divisor. For a *reconstruction* (`derivative=0`, the biased face values) there is no metric factor to multiply by, so it collapses into (i) anyway. Not a distinct route. |

**Recommendation: (i).** It is the smallest change that makes the
existing kernels honest, it fixes the FD deferrals as a side effect,
and it is the route the curvilinear-FD literature takes. (ii) stays
available if a real shock problem on a strongly stretched mesh shows
that computational-space reconstruction damages the ENO property; it
is a follow-up, not a prerequisite.

## 3. Open spike (blocking — do this before scoping the stages)

**Which Jacobian?** Two candidates:

- the **wide staggered difference of the node coordinates** (the same
  row as the flux operator: metric identity by construction), or
- the **analytic `grid.metric` Jacobian** (exact, but *not* the
  discrete inverse of the wide row).

They differ at O(h^p), and the choice decides whether **free-stream
preservation** survives — a constant state must produce exactly zero
tendency, and only the first candidate guarantees it discretely. The
analytic Jacobian is the more obvious API (it is already there) and is
the one that will silently break the invariant.

This is unsettled and it determines the shape of everything below, so
it needs a **1D spike** first: a stretched 1D advection of a constant
state and of a smooth wave, both divisors, measuring (a) the
free-stream residual and (b) the convergence order at orders 3 and 5.
No stage below is scoped in earnest until the spike answers it.

## 4. Prerequisite ordering (hard)

**The walled closure comes first.** Terrain-following and
boundary-fitted grids — the whole reason mapped high-order matters —
**always have a bounded column** (the vertical, or the fitted
horizontal boundary). The biased schemes are currently *also*
periodic-only (`_supports_walled = False`): their order-wide windows
reach across the wall. So a mapped-capable biased scheme that still
cannot handle a wall is useless for every target that motivated the
work.

Walled closure is therefore a **prerequisite**, not a parallel
workstream. The graded `Fallback` ladder
([`fallback_operator_plan.md`](fallback_operator_plan.md)) is the
existing answer on uniform meshes; it must reach the advection modules
before, or together with, the mapped divisor.

Order: **walled biased advection -> the spike (§3) -> mapped divisor
(option (i)) -> retire the FD deferrals -> (ii) only if a shock case
demands it.**

## 5. Scope notes

- The `Fallback` ladder's reduced wall rungs are Shu rows too, so they
  inherit the mapped refusal — one divisor change covers interior and
  wall rungs alike.
- The `UpwindOne` rung and `LinearInterp(boundary="one_sided")` are
  already grounded on mapped meshes and must stay untouched (audited;
  see the spec amendment). They are the sanity anchors for any new
  divisor: neither may change.
- Clenshaw–Curtis measures on `ChebyshevMesh` (the other C0 deferral)
  are a separate measure-materializer gap, not part of this plan.
