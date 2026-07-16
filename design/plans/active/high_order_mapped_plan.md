---
status: draft
date: 2026-07-13
---

# High-order stencils on mapped grids

> **Sized 2026-07-13. The SPIKE ran 2026-07-16 and is answered — the
> divisor is the same-row discrete Jacobian (§3); the FULL LIFT is
> deferred (medium, 1-2 weeks) — held back on payoff, not
> difficulty.** The corrected payoff is
> ENO/dispersion quality on stretched meshes plus honest order for
> standalone `WenoReconstruction` and `FiniteDifference` order > 2 — *not*
> asymptotic order for the advection modules, whose C-grid tendency is 2nd
> order on any mesh once the advecting velocity varies.
>
> De-risking argument found while sizing: the refusals key on
> `MappedIntervalMesh`, a per-axis monotone self-map, so the Jacobian is
> **diagonal and separable** — no cross-derivative metric terms, which is
> where multi-D curvilinear free-stream preservation actually bites. The
> identity reduces to the 1D case the spike tests, so option (ii) is a
> follow-up if a shock case shows ENO damage, not a fallback if the spike
> fails.

Spike done, full lift not started. The obstacle, the options, and the
route for lifting the mapped-mesh refusals recorded in
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

## 3. The Jacobian spike — **answered 2026-07-16**

Full numbers and the inlined spike constructions:
[`../../research/mapped_jacobian_spike.md`](../../research/mapped_jacobian_spike.md).

**The divisor is the same-row discrete Jacobian** — the wide linear
row applied to the (seam-unwrapped) node coordinates, staggered
difference for the flux form, collocated row for `FiniteDifference`.
What the spike measured (wavy-stretched mesh, constant velocity,
semi-discrete tendency error):

- Both candidates restore design order — upwind-3 3.00, upwind-5
  4.99, weno-5 (masked) 5.00, FD-4/6 3.99/5.97 — and are numerically
  indistinguishable at truncation level; the current measure divisor
  reproduces the trap (order 2 across the board). Convergence alone
  cannot choose.
- The **metric identity chooses**: the same-row divisor satisfies the
  discrete linear-preservation identity *exactly* (residual 0.0 at
  every order and resolution, by construction); the analytic Jacobian
  misses it at O(h^p) — the silent invariant break §2 predicted. The
  1D free-stream residual itself is trivially zero for every divisor
  (constant reconstructions are exact), confirming the de-risking
  argument that only multi-D can show that failure.
- Two structural findings for the lift: the widths are **static**
  (a weno-5 flux over a *linear-row* width still converges at 5 with
  free-stream exact, so no nonlinear coupling — one data-independent
  field per (space, order, bias), materialized like the measure
  fields); and `grid.metric` is not even reachable from a stretched
  `MappedIntervalMesh` (it needs a `CoordinateMapping`), so the
  analytic route would also have needed new plumbing.
- Upwind selection detail for the lift: with sign-varying velocity the
  width must be **branch-consistent** — reconstruct the coordinate
  with both biases (two static face fields) and `Where`-select with
  the same predicate as the flux before differencing; the identity
  then holds per cell with purely static inputs.

Route unchanged: mapped same-row divisor (option (i)) for the biased
reconstructions -> retire the `FiniteDifference` order > 2 and
one-sided deferrals with the same divisor -> (ii) only if a shock case
demands it.

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
