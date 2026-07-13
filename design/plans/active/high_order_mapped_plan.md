---
status: draft
date: 2026-07-13
---

# High-order stencils on mapped grids

Not started; **unblocked**. The obstacle, the options, and the route for
lifting the mapped-mesh refusals recorded in
[`../../specs/grid/classes/operators_stencils.md`](../../specs/grid/classes/operators_stencils.md)
("Amendment (2026-07-12, stages C0–C4): mapped-mesh grounding"). Opened
by the coordinate-systems work
([`../done/coordinate_systems_plan.md`](../done/coordinate_systems_plan.md) §8).

The refusals still standing in the tree today:

- `weno.require_uniform_mesh` (`spatial/operators/weno.py`) — the biased
  reconstructions (`WenoReconstruction`, the `Fallback` rungs of order
  >= 3);
- the mapped guard in `nonhydro2/modules/advection.py` (both
  `UpwindAdvection` and `WENOAdvection`, mapped mesh *and* mapped
  column);
- `FiniteDifference`: order > 2 on a mapped factor
  (`_MEASURE_ORDER = 2`), and `boundary="one_sided"` on a mapped factor.

All four raise a taught error pointing at
`staggering.mapped_order_hint`. Nothing here has landed and nothing has
become obsolete; one prerequisite (§4) has been paid.

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
metric, not the stencil, sets the order. Measured (tanh stretching,
constant advecting velocity — the only regime in which the design order
is observable at all, see §5): upwind-5 and weno-5 both 5.0 -> 2.0,
upwind-3 3.0 -> ~2.6. Nothing raised; the numbers just quietly stopped
being what the scheme promised.

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

## 3. Next step: the Jacobian spike (still open, blocking)

**Which Jacobian?** Two candidates:

- the **wide staggered difference of the node coordinates** (the same
  row as the flux operator: metric identity by construction), or
- the **analytic `grid.metric` Jacobian** (exact, but *not* the
  discrete inverse of the wide row; `spatial/coordinate_mapping.py`
  derives it by autodiff and it is already reachable from every
  operator).

They differ at O(h^p), and the choice decides whether **free-stream
preservation** survives — a constant state must produce exactly zero
tendency, and only the first candidate guarantees it discretely. The
analytic Jacobian is the more obvious API and is the one that will
silently break the invariant.

**Do this next, before scoping any stage:** a **1D spike** — stretched
1D advection of (a) a constant state and (b) a smooth wave, run with
both divisors, measuring the free-stream residual and the convergence
order at orders 3 and 5. Nothing below is scoped in earnest until the
spike answers it.

Then: mapped divisor (option (i)) for the biased reconstructions ->
retire the `FiniteDifference` order > 2 and one-sided deferrals with
the same divisor -> (ii) only if a shock case demands it.

## 4. Prerequisite: walled closure — **paid**

Terrain-following and boundary-fitted grids always have a bounded
column, so a mapped-capable biased scheme that could not handle a wall
would be useless for every target that motivated this work. That
prerequisite is now satisfied: the graded near-wall closure shipped on
both routes (`spatial/operators/graded.py`, the FV `Fallback` ladder
and the nodal C-grid advection: 4eac7ccf, 5005f985, 719ff4cd), with a
configurable bottom rung (773af55b, 0526d06b); see
[`fallback_operator_plan.md`](../done/fallback_operator_plan.md) (status
`done`). The biased schemes accept walled grids today via
`boundary="graded"`; only the mapped refusal remains.

## 5. Scope notes

- **What the payoff actually is.** The composite tendency of the
  nodal C-grid biased advection is formally 2nd order whenever the
  advecting velocity varies along the flux axis — on *any* mesh,
  uniform included (04804c45, pinned in `tests/nonhydro2/test_advection.py`;
  the FV benefit ledger was corrected accordingly, 058c4432). So the
  mapped divisor does not buy back asymptotic order for
  `UpwindAdvection`/`WENOAdvection` in the general case; what it buys
  is the ENO property and dispersion behaviour those schemes exist for,
  restored on a stretched mesh, plus honest design order for the pieces
  that genuinely have it: the `WenoReconstruction` operator itself, the
  constant-velocity regime, and `FiniteDifference` at order > 2.
- The `Fallback` ladder's reduced wall rungs are Shu rows too, so they
  inherit the mapped refusal — one divisor change covers interior and
  wall rungs alike.
- The `UpwindOne` rung and `LinearInterp(boundary="one_sided")` are
  already grounded on mapped meshes and must stay untouched (audited;
  see the spec amendment). They are the sanity anchors for any new
  divisor: neither may change.
- Clenshaw–Curtis measures on `ChebyshevMesh` (the other C0 deferral)
  are a separate measure-materializer gap, not part of this plan.
