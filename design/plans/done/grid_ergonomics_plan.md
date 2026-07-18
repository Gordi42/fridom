---
status: done
date: 2026-07-15
---

# Grid ergonomics — a fast assemble for spherical (and cartesian) grids

> **Landed 2026-07-15 on `feat/spherical-grid`.** Shipped exactly the
> layered shape below: the `fr.spatial.charts.lonlat_sphere` chart
> primitive, the `fr.spatial.spherical.Grid` convenience (required
> `lat_extent`, optional `lon_extent` → closed zonal walls, radians,
> `lon`/`lat` names, a loud pole guard), and the sibling
> `fr.spatial.cartesian.Grid` (filled from its spec). All five sub-
> decisions landed as recommended (D1 required `lat_extent`, D5 both
> conveniences). Dogfooded: the five `sphere_grid` test helpers and the
> `sw.Model` spherical docstring now use `spherical.Grid`, with the
> spherical-shallow-water assertions as the (green) parity gate. Closes
> open item #1 of [`phase2_grid_followups.md`](../done/phase2_grid_followups.md).

> **Sized 2026-07-15: SMALL (~1-2 days, ~200-350 LOC incl. tests).**
> Ergonomics, not blocking. Sibling to the just-landed chart-ergonomics
> work ([`chart_ergonomics_plan.md`](chart_ergonomics_plan.md),
> E2 `orthogonal=True`) and to open item #1 of
> [`phase2_grid_followups.md`](../done/phase2_grid_followups.md) (the unbuilt
> `cartesian.Grid` convenience). This plan builds the spherical
> convenience the owner asked for and settles the cartesian one at the
> same time so the two stay consistent.

## Motivation

Assembling a lat-lon sphere grid today is a ~9-line incantation the user
must copy from a docstring, and three of those lines are error-prone
domain knowledge, not user intent:

```python
mlon = fr.spatial.meshes.IntervalMesh(nlon, (0.0, 2*np.pi), name="lon")
mlat = fr.spatial.meshes.IntervalMesh(nlat, (-lat_max, lat_max),
                                      periodic=False, name="lat")
mapping = fr.spatial.CoordinateMapping(chart={"X": lambda lon, lat: (
    a*jnp.cos(lat)*jnp.cos(lon), a*jnp.cos(lat)*jnp.sin(lon),
    a*jnp.sin(lat))}, orthogonal=True)         # <- embedding lambda +
grid = fr.spatial.Grid((mlon, mlat), mapping=mapping)   #    orthogonality
```

A user should not have to hand-write the embedding lambda, know that the
lat-lon chart is orthogonal (`orthogonal=True`), or remember the
periodic-lon / bounded-lat / `(0, 2π)` conventions.

`SphereMesh` is **not** the route — the design already rejected it
([`../../specs/grid/classes/meshes.md`](../../specs/grid/classes/meshes.md)
§SphereMesh: *"otherwise the chart is the answer"*). The verbose part is
the **chart**, so the fast assemble is a chart preset plus a thin grid
convenience over it, not a bespoke 2D mesh.

## Owner requirements (2026-07-15)

- A **chart preset** is the right primitive (approved).
- **Latitude extent must be configurable** so an asymmetric band — a
  northern-hemisphere `(0, lat_max)` — is possible, not only the
  symmetric `(-lat_max, lat_max)`.
- **Longitude extent is an optional parameter.** Omitted → the default
  full periodic circle `(0, 2π)`. Given → the zonal axis becomes
  **bounded (closed E/W boundaries)** — a longitude sector.

Because the extents and periodicity are *mesh* properties, the preset
must build the meshes too. That is the layered shape below.

## Chosen shape — layered (chart primitive + convenience grid)

The chart embedding is **extent-independent**: the map
`(lon, lat) -> (a cos lat cos lon, a cos lat sin lon, a sin lat)` is the
same whether latitude runs `(0, π/2)` or `(-π/2, π/2)` and whether
longitude is periodic or walled. So the extent-free mapping stays a
reusable primitive, and the extents live in a convenience grid on top.

### 1. Chart primitive — `fr.spatial.charts.lonlat_sphere`

New module `src/fridom/spatial/charts.py`, re-exported as
`fr.spatial.charts` (a namespace future charts — cylinder, torus,
terrain — can join).

```python
def lonlat_sphere(radius: float = 1.0) -> CoordinateMapping:
    r"""The lat-lon sphere embedding chart (orthogonal, radius a)."""
    return CoordinateMapping(
        chart={"X": lambda lon, lat: (
            radius * jnp.cos(lat) * jnp.cos(lon),
            radius * jnp.cos(lat) * jnp.sin(lon),
            radius * jnp.sin(lat))},
        orthogonal=True)
```

- Extent-free: composes with any lon/lat mesh layout (uniform,
  stretched `MappedIntervalMesh`, immersed polar caps, any device set).
- Binds to coordinate names **`lon`, `lat`** (the chart callable's
  parameter names *are* the coord names — `CoordinateMapping`
  introspects the signature). Custom names are out of scope for v1 (a
  lambda cannot carry dynamic parameter names; see open sub-decision D3).
- `orthogonal=True` is **provably correct for every lon/lat extent**:
  `g_lon,lat = ∂_lon X · ∂_lat X ≡ 0` analytically (verified numerically:
  ~1e-8 autodiff roundoff across the domain; the E2 finding that the
  metric is numeric, not symbolic, is why we assert rather than
  auto-detect). This is what lets the diagonal index moves assemble
  across *both* walls in the sector case below.

### 2. Convenience grid — `fr.spatial.spherical.Grid`

New subpackage `src/fridom/spatial/spherical/` (`grid.py` + lazy
`__init__.py`), mirroring `src/fridom/spatial/cartesian/`. Subclass of
the base `Grid`, exactly as `cartesian.Grid` is spec'd to be
([`../../specs/grid/classes/grid.md`](../../specs/grid/classes/grid.md)
§cartesian.Grid) — builds meshes and delegates, adds no new
methods/properties.

```python
class Grid(fr.spatial.Grid):
    """Lat-lon sphere convenience grid: two IntervalMesh factors under
    the orthogonal sphere chart."""

    def __init__(
        self,
        shape: tuple[int, int],                 # (nlon, nlat)
        radius: float = 1.0,
        *,
        lat_extent: tuple[float, float],        # radians; required (D1)
        lon_extent: tuple[float, float] | None = None,
        defaults: Mapping[DispatchKey, Operator] | None = None,
        immersed: ImmersedDomain | None = None,
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        ...
```

Body:

- **lon mesh.** `lon_extent is None` → `IntervalMesh(nlon, (0, 2π),
  periodic=True, name="lon")` (full circle). Otherwise →
  `IntervalMesh(nlon, lon_extent, periodic=False, name="lon")` (closed
  zonal boundaries — the sector case).
- **lat mesh.** Always bounded:
  `IntervalMesh(nlat, lat_extent, periodic=False, name="lat")`.
- **mapping.** `charts.lonlat_sphere(radius)`.
- Delegate: `super().__init__((mlon, mlat), mapping=mapping,
  defaults=defaults, immersed=immersed, device_ids=device_ids)`.

Both the periodic-lon (one wall, at the lat caps) and the sector
(walls on all four edges) cases assemble because the chart is orthogonal
→ E2 seeds `RaiseIndex`/`LowerIndex(diagonal=True)` → the off-diagonal
cross terms a wall cannot interpolate are dropped.

Result — the call site the user gets:

```python
# full sphere, capped at ±80°:
grid = fr.spatial.spherical.Grid((128, 64), radius=a,
                                 lat_extent=(-lat_max, lat_max))
# northern hemisphere:
grid = fr.spatial.spherical.Grid((128, 64), lat_extent=(0.0, lat_max))
# a longitude sector with closed E/W walls:
grid = fr.spatial.spherical.Grid((64, 64), lat_extent=(0.0, lat_max),
                                 lon_extent=(0.0, np.pi/2))
```

## Sub-decisions

Settled with the owner (2026-07-15):

- **D1 — `lat_extent` is required (no default). DECIDED.** The poles are
  metric-singular (`sqrt_g = a cos lat -> 0` at `lat = ±π/2`), so any
  default is either a magic cap (the old `±80°`) or an unsafe `±π/2`.
  Correct-or-loud house style
  ([[prefer-explicit-over-risky-automagic]]): the user states the band.
  A northern-hemisphere run passes `(0, lat_max)` explicitly anyway.
- **D5 — build `cartesian.Grid` in the same pass. DECIDED.** It is a
  9-line stub with a finished spec (grid.md §cartesian.Grid) and is open
  item #1 of `phase2_grid_followups.md`; building `spherical.Grid`
  without it leaves two sibling conveniences half-done and risks
  divergent conventions. Both land as `shape=`-first, keyword-only,
  subclass-not-factory.

Going with the recommendation unless the owner objects:

- **D2 — units: radians.** What the mesh and chart already use; a
  `degrees: bool = False` convenience is deferred to a follow-up if
  asked. One honest unit in v1 (silent unit-mixing is the classic
  footgun).
- **D3 — coordinate names fixed to `lon`/`lat` in v1.** The chart lambda
  binds by parameter name, so custom names need a dynamically-named
  function (ugly). Out of scope; noted.
- **D4 — loud pole guard.** Raise if `lat_extent` reaches or crosses
  `±π/2` (a cell face would sit on the singular pole): turns a silent
  `inv_g -> inf` into a taught error naming the cap. Sub-π/2 is the
  user's responsibility; we reject only the provably-singular endpoint.

## Files

New:

- `src/fridom/spatial/charts.py` — `lonlat_sphere` (+ module docstring).
- `src/fridom/spatial/spherical/__init__.py` — lazypimp, mirroring
  `cartesian/__init__.py`.
- `src/fridom/spatial/spherical/grid.py` — the convenience `Grid`.
- (D5) fill `src/fridom/spatial/cartesian/grid.py` from the grid.md spec.
- `tests/spatial/test_charts.py`, `tests/spatial/spherical/test_grid.py`
  (+ `test_init.py`), and — for D5 —
  `tests/spatial/cartesian/test_grid.py`.

Edited:

- `src/fridom/spatial/__init__.py` — re-export `charts` and `spherical`
  (and confirm `cartesian` is exported) at the `fr.spatial.*` level;
  update its `test_init.py`.
- Dogfood (proves parity, shrinks the docs): rewrite the `sphere_grid`
  helpers in `tests/validation/test_spherical_shallowwater.py`,
  `tests/shallowwater2/test_coriolis.py`, `test_diagnostics.py`,
  `test_spherical.py`, and `tests/model/modules/test_coriolis.py` to
  `fr.spatial.spherical.Grid(...)`, and collapse the `sw.Model`
  spherical-recipe docstring block
  (`src/fridom/shallowwater2/model.py`) to the one-line grid call.
- Resolve open item #1 in `phase2_grid_followups.md` (link here); on
  completion move this plan to `design/plans/done/` and record it in
  `design/roadmap/done.md` + `design/README.md`.

## Tests

- `lonlat_sphere`: returns a `CoordinateMapping`, `orthogonal is True`,
  `chart_coords == ("lon", "lat")`; on a built grid the induced metric
  matches the analytic sphere (`g_lonlon ≈ a²cos²lat`, `g_latlat ≈ a²`,
  `g_lonlat ≈ 0`) and `radius` scales it.
- `spherical.Grid` default (periodic lon): `mlon` periodic, `mlat`
  bounded, mapping attached, `raise_index`/`lower_index` resolve with
  `diagonal is True`, and a covariant vector's index raise applies.
- `spherical.Grid` sector (`lon_extent` given): `mlon` bounded
  (`periodic is False`); index moves still assemble across the extra
  walls; a field builds and an operator applies.
- Northern hemisphere `lat_extent=(0, lat_max)`: extents wired through.
- D4 pole guard: `lat_extent` touching `±π/2` raises a taught error.
- Re-export/init tests for `charts`, `spherical` (and `cartesian`).
- Parity: the rewritten validation `sphere_grid` reproduces the prior
  results (the existing spherical-shallow-water assertions are the
  regression gate — no numeric change expected, same meshes+mapping).

## Merge gate & workflow

Feature branch `feat/spherical-grid` + worktree (AGENTS.md). Gate:
the mirrored tests for every edited source file + the `nonhydro2`
chartless model smoke (framework-core-adjacent: `__init__` re-exports)
+ `ruff check src tests`. Land `--no-ff` onto `dev`, delete branch and
worktree same session. Base the branch on current `dev` (it carries the
E2 `orthogonal=True` seeding this depends on — the perf working tree
does not).
