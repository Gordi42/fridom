---
status: done
date: 2026-07-15
---

# Chart / sphere setup ergonomics

> **Done 2026-07-15.** All five items have landed; E2 (the last) closed
> below. Kept as a record; superseded content stays for context.

Gaps found while building minimal spherical and torus shallow-water
examples on the stage-C2 chart machinery
([`../done/coordinate_systems_plan.md`](../done/coordinate_systems_plan.md)),
each confirmed by ablation. The physics is right; the *setup surface*
is not.

## Landed

- **E2 — the diagonal index-move overrides were undiscoverable.**
  Closed by `feat/chart-orthogonal` (2026-07-15). *The failure:* on a
  chart with a **bounded** axis a model would not assemble without the
  5-line incantation `grid.merge_overrides({"raise_index":
  RaiseIndex(coords, diagonal=True), "lower_index": ...})` — the grid
  seeded `RaiseIndex(chart)` / `LowerIndex(chart)` with
  `diagonal=False`, so the expansion emitted the cross-term
  interpolation chain, and across a wall there is no legal
  interpolation row (`DispatchError: no operator registered for kind
  'interpolate' on Inner(lat)`, which never named the fix). Worse, the
  recipe was silently *optional* on a fully periodic chart (the cross
  terms interpolate fine, just multiplied by ~zero metric), so the same
  chart code worked or failed on the boundary condition of an axis.

  *The preferred auto-detect was found unachievable.* The plan proposed
  seeding `diagonal=True` automatically when the off-diagonal metric is
  "structurally zero." But the induced metric is computed by **numeric
  autodiff** (`jax.jvp` of the chart callable, then dot the tangents),
  not symbolically: the sphere and torus off-diagonal `g_uv` evaluate to
  **~1e-16 roundoff, not exact zero** (measured), while a genuinely
  non-orthogonal chart reads O(1). So "structurally zero" collapses to a
  *tolerance* decision, and a tolerance risks a false positive silently
  dropping real cross terms → wrong physics — the exact bug-class E1
  ("was silently wrong") exists to kill, and *strictly worse than the
  status quo* for a periodic non-orthogonal chart (correct today, would
  become silently wrong).

  *Landed instead (owner decision):* an explicit `orthogonal=True` on
  `CoordinateMapping(chart=..., orthogonal=True)` — one honest flag at
  the natural place. The grid reads `mapping.orthogonal` at seed time
  and seeds `RaiseIndex`/`LowerIndex` with `diagonal=True`; the default
  keeps the full expansion (correct-or-loud on a non-orthogonal chart).
  A **taught-error safety net** (`_bounded_cross_term_error`,
  `spatial/operators/composed.py`) catches the missing interpolation row
  in the index-move expansion and re-raises naming both
  `orthogonal=True` and the low-level `RaiseIndex(..., diagonal=True)`.
  The recipe was dropped from the `sw.Model` docstring and every chart
  test; `merge_overrides` for **wall-closure** rows (one-sided
  interpolate) is unrelated and stays.
- **E1 — `FPlaneCoriolis` on a chart grid was silently wrong.** Closed
  by 169e00ee (chart rejection at `bind`) and 715a0b0f: rotation is now
  opt-in (`coriolis=None` installs no rotation term at all, rather than
  a flat `f0=1.0`), and `FPlaneCoriolis` / `BetaPlaneCoriolis` reject an
  embedding-chart grid at `bind` (`_reject_chart_grid`,
  `model/modules/coriolis.py`).
- **E3 — `SphericalCoriolis` was misnamed.** Closed by 169e00ee /
  715a0b0f, and better than proposed: `SphericalCoriolis` is gone,
  subsumed by the chart-generic `RotationCoriolis(omega=...)`, which
  *derives* `f = 2 Omega . n_hat` from the chart normal instead of
  taking a sphere formula. The f-plane and `2 Omega sin(lat)` are
  special cases.
- **E4 — the conserved energy invariant was not a diagnostic.** Closed
  by daa75518: `sw.diagnostics.ekin_full` / `epot_full` / `etot_full`
  expose the thickness-weighted invariant the Sadourny scheme conserves;
  the validated test helpers now call the public diagnostic. (f70fad97
  then made the Coriolis term conserve it exactly, both routes.)
- **E5 — no scalar accessor on a field.** Closed by daa75518:
  `ScalarField.item()` (`spatial/fields/scalar_field.py:586`).
