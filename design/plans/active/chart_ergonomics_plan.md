---
status: active
date: 2026-07-13
---

# Chart / sphere setup ergonomics

> **Roadmap 2026-07-13: a next step.** E2 is the last item, and it is the
> one standing between a user and a working bounded chart.

Gaps found while building minimal spherical and torus shallow-water
examples on the stage-C2 chart machinery
([`../done/coordinate_systems_plan.md`](../done/coordinate_systems_plan.md)),
each confirmed by ablation. The physics is right; the *setup surface*
is not.

Four of the five original items landed on 2026-07-12/13 (see Landed).
One remains: the diagonal index-move overrides. It is the last thing
standing between a user and a working bounded chart.

## E2 — the diagonal index-move overrides are undiscoverable

On a chart with a **bounded** axis, a model does not assemble without

```python
grid.merge_overrides({
    "raise_index": RaiseIndex(("lon", "lat"), diagonal=True),
    "lower_index": LowerIndex(("lon", "lat"), diagonal=True)})
```

The grid seeds `RaiseIndex(chart)` / `LowerIndex(chart)` with
`diagonal=False` (`spatial/grid.py:1855`), so the expansion emits the
cross-term interpolation chain — and across a wall there is no legal
interpolation row. The failure is

```
DispatchError: no operator registered for kind 'interpolate' on Inner(lat)
```

(`spatial/operators/registry.py`), which never names the fix.

Worse, whether the recipe is *required* depends on a property the user
is never told about: on a fully periodic chart (torus) the same
overrides are **silently optional** — the cross terms interpolate
fine, they are just multiplied by structurally zero metric entries.
So the same chart code works or fails depending on the boundary
condition of an axis.

The recipe is currently propagated by documentation only: the
`sw.Model` docstring spells it out, and every chart test
(`tests/shallowwater2/test_spherical.py`,
`tests/validation/test_spherical_shallowwater.py`,
`tests/model/modules/test_coriolis.py`, ...) repeats it verbatim.

**Fix (preferred):** seed the index moves with `diagonal=True`
automatically when the derived off-diagonal metric is **structurally
zero** — the grid derives the metric from the chart and can check this
at seed time; both target charts (lat-lon sphere, torus) are
orthogonal. Users with a genuinely non-orthogonal chart get the full
expansion, and on a bounded axis that expansion legitimately has no
row.

**Fix (fallback, if the structural check is not cheap):** catch the
missing interpolation row inside the index-move expansion and re-raise
a taught error naming `RaiseIndex(..., diagonal=True)`.

Once the seeding is automatic, drop the recipe from the `sw.Model`
docstring and from the chart tests.

## Landed

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
