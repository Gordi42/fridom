---
status: draft
date: 2026-07-13
---

# High-order stencils on mapped grids

> **Status 2026-08-23 — the reconstruction half is SHIPPED, by route
> (ii).** The survey
> [`../../research/nonuniform_weno_survey.md`](../../research/nonuniform_weno_survey.md)
> (Oceananigans, JAX-Fluids, a coefficient spike, and the 3-D constancy
> argument of its §5) overturned the §2 recommendation for the biased
> *reconstruction* rows: `WenoReconstruction`, the graded `Fallback`,
> and the `UpwindAdvection` / `WENOAdvection` face kernels (nodal and
> FV families, both biases, graded walls) now derive per-face tables
> from the factor's cell widths and accept a stretched
> `MappedIntervalMesh`; the static uniform path is bitwise untouched
> (entry in [`../../roadmap/done.md`](../../roadmap/done.md)). Route (i)
> — the same-row discrete Jacobian divisor, spike answered in §3 —
> stands for the **collocated** rows only.
>
> **What remains open here:**
>
> 1. `FiniteDifference(order > 2)` and `boundary="one_sided"` on a
>    mapped factor — the route-(i) lift (`_MEASURE_ORDER = 2`), the
>    spike's DA divisor; a collocated derivative has no projection to
>    agree with, so the constancy objection of the survey does not
>    apply to it.
> 2. Owner call — the **nodal** C-grid family on a stretched axis is
>    2nd order by construction whatever rows it uses (its point-value
>    flux difference is high-order only through the uniform-lattice
>    Shu–Osher identity; measured upwind-5 nodal 2.0 vs FV 5.0). As
>    shipped it uses the width-aware FV rows (one code path with the FV
>    family); the static rows measured a ~3x smaller O(h²) constant on
>    the wavy map (1.1e-4 vs 3.3e-4 at n = 128). Switching is one line
>    in `advection._face_widths`; pinned in
>    `tests/model/modules/test_advection_stretched.py`.
> 3. Stretched factor on an **immersed** grid: taught refusal
>    (`_supports_stretched_immersed`; the mask-keyed ladder has no
>    co-window seam yet). Biased schemes on a mapped **column**
>    (terrain following): taught refusal kept (`_supports_mapped_column`);
>    the survey's §7 argues it composes in base coordinates.
> 4. Perf (owner-triggered guard): the one-pass WENO kernel on a
>    stretched axis carries ~21 small `where` selects of the two bias
>    table sets (the generator itself folds to trace-time constants on
>    a device-local axis; on a sharded mapped axis it stays staged,
>    ~5x the kernel size, correct).

History below is kept as written; §2's recommendation is superseded
for the reconstruction rows.

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

**Recommendation: (i)** *(superseded 2026-08-23 for the reconstruction rows — status block above; stands for the `FiniteDifference` rows)*. It is the smallest change that makes the
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
