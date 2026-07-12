---
status: draft
date: 2026-07-12
---

# Chart / sphere setup ergonomics

Not started. Gaps found while building minimal spherical and torus
shallow-water examples on the stage-C2 chart machinery
([`../done/coordinate_systems_plan.md`](../done/coordinate_systems_plan.md)),
each confirmed by ablation. The physics is right; the *setup surface*
is not. Items are severity-ordered: E1 is silent wrong physics, the
rest are papercuts.

## E1 — `FPlaneCoriolis` on a chart grid is silently wrong

`sw.Model(coriolis=None)` installs `FPlaneCoriolis(f0=1.0,
metric_weight="csqr")` (`shallowwater2/model.py:143`), the **flat**
rotation term: a constant `f` on a `Profile()` and the Cartesian
`du/dt = f v` pairing, with no metric factors anywhere.

`FPlaneCoriolis` has **no `bind` override**
(`model/modules/coriolis.py:160`), so nothing rejects it on a chart
grid. `SphericalCoriolis` does check (it requires a chart grid whose
coordinates match its `coords`), but the check only runs if you
already knew to reach for it. A spherical run with a forgotten
`coriolis=` argument therefore assembles, compiles, runs, conserves
mass — and integrates the wrong rotation. There is no error, no
warning, and no diagnostic that goes obviously wrong.

**Fix:** `FPlaneCoriolis` (and `BetaPlaneCoriolis`) reject a grid with
`grid.chart_coords is not None` at `bind`, with a taught error naming
`SphericalCoriolis`. A user who genuinely wants a flat `f` on a chart
grid opts in explicitly.

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

(`spatial/operators/registry.py:354`), which never names the fix.

Worse, whether the recipe is *required* depends on a property the user
is never told about: on a fully periodic chart (torus) the same
overrides are **silently optional** — the cross terms interpolate
fine, they are just multiplied by structurally zero metric entries.
So the same chart code works or fails depending on the boundary
condition of an axis.

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

## E3 — `SphericalCoriolis` is misnamed

The **rotation term** is chart-generic: the metric-aware M-skew form
(`model/modules/coriolis.py:323`) is correct on *any* chart and runs
unmodified on a torus. Only two things are spherical:

- the name and the `coords=("lon", "lat")` default, and
- the `f_coriolis` **default field**, `f = 2 Omega sin(lat)`
  (`coriolis.py:_f_default`) — the one genuinely sphere-specific bit.

**Fix:** split the two. A chart-generic metric-aware rotation module
(name TBD — `ChartCoriolis`?) taking an `f` provider, with
`SphericalCoriolis` as the thin preset supplying `2 Omega sin(lat)`
and the lat-lon defaults. Purely a naming/factoring change; the
numerics stay.

## E4 — the conserved energy invariant is not a diagnostic

`sw.diagnostics.ekin`/`epot` are the **linearized quadratics**
consistent with the energy metric `M = diag(1, 1, 1/c^2)`
(`shallowwater2/diagnostics.py`). They are chart-aware (`ekin` carries
`g_11`/`g_22`), but they are *not* the invariant the Sadourny scheme
actually conserves: the metric, thickness-weighted total energy.

So the natural check on a spherical run — "is energy conserved?" —
measures the wrong quantity and drifts, and the quantity that does not
drift has to be assembled by hand.

**Fix:** add a chart-aware `diagnostics.energy()` returning the
thickness-weighted invariant the scheme conserves (the C2 semi-discrete
energy rate, measured 2.6e-17, is the gate that already knows the
right expression — reuse it).

## E5 — no scalar accessor on a field

`ScalarField.integrate()` returns a `ScalarField`
(`spatial/fields/scalar_field.py:512`); there is no way to get a
python float out except

```python
float(field.integrate().data.ravel()[0])
```

Every diagnostic print in every example does this.

**Fix:** a scalar accessor on `ScalarField` (`.item()`, or
`float(field)` via `__float__`) that asserts the field is
zero-dimensional / constant-space and gathers.

## Ordering

E1 first (it is the only correctness item). E2 next (it blocks every
bounded chart at the front door). E3/E4/E5 are independent and can
land in any order; E3 touches the same module as E1, so folding them
into one change is natural.
