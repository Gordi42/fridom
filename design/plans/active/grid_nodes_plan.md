---
status: draft
date: 2026-08-23
---

# Grid nodes as xarray — plotting the mesh

> **Status 2026-08-23.** Owner asked for a built-in way to look at a
> grid: "convert the node points to an xarray Dataset / DataArray and
> use its plotting functions", with four requirements — moving
> geometry (z*, isopycnal), several function spaces on one plot (faces
> and centres), a single factor of a tensor mesh (only the `z` mesh of
> a 3D grid), immersed grids. **§3 is the user-level API awaiting the
> owner's approval; nothing is implemented.** The facts in §2 come
> from a code survey of 2026-08-23 (file references inline).

## 1. Motivation

`examples/hydrostatic/coastal_upwelling.py` draws the stretched
column by recomputing the node positions by hand
(`y_nodes = (np.arange(ny) + 0.5) * LY / ny`,
`z_nodes = column((np.arange(nz) + 0.5) / nz)`) and scattering them
with matplotlib. Every user who wants to see a stretched mesh, the
C-grid staggering, a terrain-following column or the wet cells of an
immersed domain has to do the same, and the hand-rolled version is
already wrong on a mapped column (it ignores `maps=`) and blind to
immersed masks. The grid owns the node positions; it should hand them
out in the container people plot from.

## 2. What exists (survey 2026-08-23)

- **Node positions live per function space, never per mesh.** A
  `Mesh` carries no arrays (`meshes/mesh.py`, `structured_1d.py`); the
  placement is decided by the node set of the factor space
  (`center`/`left`/`right`/`outer`/`inner`, `cell_avg`/`face_avg`).
  The single source is `Grid._node_vector(factor)` (`grid.py:2040`),
  a 1D jax vector per factor, exposed as
  `grid.evaluation_nodes(space, name) -> ScalarField` (`grid.py:1149`)
  and forwarded by `ScalarField.nodes(name)` (`scalar_field.py:1012`).
  The N-D meshgrid is never stored (spec 01_concepts.md §"meshgrid");
  `evaluation_nodes` returns the 1D vector broadcast-tagged with the
  other factors replaced by `mesh.constant`. Factor access by name is
  `grid.factor(name) -> Mesh` (`grid.py:372`).
- **The xarray export already computes the coordinate skeleton.**
  `spatial/export.py` splits a values-free `ExportLayout` (dims,
  1D coords, attrs `c_grid_axis_shift`, `representation`, `units`
  from `grid.coordinate_units`) from the gather. Dim naming is the
  spec'd xgcm style (`design/specs/grid/classes/grid.md` §xarray):
  plain names for `ScalarField.xr`, position suffixes (`x`, `x_right`,
  `x_outer`, `x_inner`, `x_left`) wherever two positions of one axis
  must coexist (`VectorField.xr`, the io Writer). Coordinate label
  vectors go through `_host_labels` (`export.py:178`), which is a
  **multi-process collective** (`process_allgather`) on non-addressable
  arrays — any grid-level export must be called on every rank.
- **Mapped geometry never materialises positions.**
  `CoordinateMapping` (`coordinate_mapping.py:746`) holds the map
  callables (`maps={"zp": lambda z, H: z * H}`) and derives only
  metrics (`dzp_dz`, …) by `jax.jvp` through `grid.metric(space, name,
  params=)`. The mapped name (`zp`) is by construction **not** a grid
  coordinate (`_bind`, `:912`). Time dependence rides the state as
  AUXILIARY parameter fields named after the map parameters (`H`,
  `eta`, with `<p>_dot`), recovered by `mapping_params(state, grid)`
  (`model/modules/moving_geometry.py:66`). z* shipped 2026-08-23 on
  this seam (`hy.zstar_mapping`, `hy.ZStarGeometry`, map key `"zp"`,
  parameter `"eta"`). The only evaluation of a map's *value* is the
  private static-parameter quadrature seam
  `CoordinateMapping._column_correction` (`:1301`).
  **Consequence today:** `field.nodes("z")` on a terrain or z* column
  is the computational coordinate, and `b_total` (hy/nh diagnostics)
  adds `N^2 z` with that `z` — correct on a flat column, wrong under
  `maps=`. A public physical-position accessor fixes both the plot
  and this diagnostic.
- **Chart grids keep 1D `lon`/`lat` nodes in radians**
  (`spherical/grid.py`, `charts.lonlat_sphere`); the embedding `X` is
  not exposed, only the induced metric and `normal_x|y|z`.
- **Immersed domains** expose `grid.immersed.mask(space)` (bool
  `ScalarField`, wet = True) and `.fraction(space)`
  (`immersed_domain.py:264, :317`), derived per space (faces combine
  neighbours by the slip rule), **only for a space that resolves every
  grid coordinate**. Fields and `.xr` are mask-unaware; dry DOFs hold
  zeros.
- **No grid-level or space-level xarray export exists**, and no
  plotting code anywhere in `src/fridom/spatial`. xarray is a lazy
  optional import in `export.py`.
- xarray 2026.4 plots all three candidate layouts (checked
  2026-08-23): structured 1D coords on distinct dims
  (`ds.plot.scatter(x="y", y="z")`), a 2D physical coordinate on
  computational dims, and a flat point cloud with `hue="space"`.

## 3. Proposed user-level API (for approval)

### 3.1 `grid.nodes(...)` — the node cloud of one or several spaces

```python
grid.nodes(*what, params=None) -> xr.Dataset
```

`what` is any number of

| argument | meaning | `space` label |
|---|---|---|
| a `SpaceLike` | the nodes of that space | the suffixed dims, `"x_right*y*z"` |
| a `ScalarField` | the nodes of the field's space | the field's name, `"u"` |
| a `VectorField` (a model state) | one entry per component | the component names |
| a coordinate name `"z"` | the centres **and** outer faces of that factor alone (1D) | `"z"` / `"z_outer"` |
| nothing | the all-centre product space | `"x*y*z"` |

The result is **one layout for every call**: a point cloud with the
single dimension `node`, in C order of each space's index space, the
spaces concatenated in argument order.

```
<xarray.Dataset>
Dimensions:   (node: 9312)
Coordinates:
    space     (node) <U9   'u' 'u' ... 'w' 'w'        # hue label
    i_y, i_z  (node) int64                             # index per axis
Data variables:
    y, z      (node) float64 [units]                   # grid coordinates (nodes of the mesh placement)
    zp        (node) float64 [units]                   # physical coordinate(s) of maps=, if any
    wet       (node) bool                              # immersed wet mask, if any
Attributes:
    grid: <repr>, spaces: {label: repr}
```

- **Grid coordinates**: one variable per grid coordinate of the
  space, values from `evaluation_nodes` (so a `MappedIntervalMesh`
  stretch and a chart's radians appear as they are). Coordinate units
  come from `grid.coordinate_units` when declared (a chart); otherwise
  none (the grid is scaling-agnostic; see 3.3 for the model-side
  stamp).
- **Physical coordinates** (`maps=`): one variable per mapped name
  (`zp`), the map **value** at the nodes. `params=` is a mapping of
  parameter fields or a state (`VectorField`) from which the mapping's
  `param_names` are looked up, exactly like `mapping_params`; `None`
  means the mapping's static defaults (the reference geometry).
  `grid.nodes(state["b"], params=model.state)` is the z* column at
  the current free surface. A factor-only query (`"z"`) on a mapped
  axis carries no `zp` (the map needs the other coordinates).
- **Immersed**: `wet` from `grid.immersed.mask(space)` for every
  space that resolves all coordinates (a factor-only query carries no
  mask). Dry nodes are kept; `ds.where(ds.wet, drop=True)` drops
  them, `hue="wet"` colours them.
- **Index variables** `i_<axis>` let a user recover the structure
  when lines rather than dots are wanted:
  `ds.set_index(node=("i_y", "i_z")).unstack("node")` gives 2D
  `y`, `zp` to draw the z* layers as lines.
- Host-side, not jittable, collective-safe: built on `ExportLayout`
  and `_host_labels`, gathered once per space.

The four requests, as one-liners:

```python
# the stretched column of the coastal page, centres and faces
grid.nodes("z").plot.scatter(x="z", y="space")

# the C-grid staggering of a model, one colour per field
grid.nodes(model.state).plot.scatter(x="x", y="y", hue="space", s=4)

# faces and centres of the section on one plot
grid.nodes(model.state["b"], model.state["w"]).plot.scatter(
    x="y", y="z", hue="space")

# a z* column at the current free surface, physical height on the axis
grid.nodes(model.state["b"], params=model.state).plot.scatter(
    x="y", y="zp")

# an immersed domain, wet cells only
cells = grid.nodes()
cells.where(cells.wet, drop=True).plot.scatter(x="x", y="y", s=2)
```

### 3.2 Physical positions at the field level (the primitive)

```python
grid.evaluation_nodes(space, name, *, params=None) -> ScalarField   # name may be a mapped name
field.nodes(name, *, params=None) -> ScalarField
```

`evaluation_nodes` accepts, besides the grid coordinates it takes
today, the mapped names of `grid.mapping` and returns the map value at
the nodes of `space`, parameters resolved through the same `params=`
overload `grid.metric` uses (static defaults when `None`). This is
the traced, jit-safe primitive that 3.1 gathers from, and it is what
`b_total` should read on a mapped column (`b + N^2 zp`). The
`CoordinateMapping` side is one new public method
(`positions(space, name, params=)`) next to `metric`, built on the
existing `_jvp`/`_param_at_nodes` machinery but evaluating the primal
with live parameters. A chart's embedding is **not** in this step
(`lon`/`lat` are the sphere's own coordinates); ambient `x, y, z`
would be a later `ambient=True` on 3.1.

### 3.3 Sugar (phase 2, optional)

`model.nodes(*what)` = `grid.nodes(*what, params=model.state)` with
the model's unit rows stamped onto the coordinate variables
(`model.units`, the same rows the Writer stamps), so the axes of
`plot.scatter` read `y [m]` and a nondimensional run reads `1`. The
grid-level call stays unit-less because the grid does not know the
scaling.

## 4. Alternatives considered

1. **Structured layout** (dims = the space's coordinates, 1D coordinate
   variables, `zp` as a 2D/3D variable on them). Plots just as well
   for one space, and is the natural home of `zp` as a CF auxiliary
   coordinate. It cannot hold several spaces with a `hue` label, it
   collides on dim names once two spaces share a position, and it has
   no future on an unstructured mesh. It is also already available
   per axis: `grid.evaluation_nodes(space, "z").xr` is the 1D
   `DataArray` with dim `z`. Hence one point layout plus index
   variables, rather than a layout that switches with the number of
   arguments.
2. **Method name.** `grid.nodes` keeps the codebase's meaning of
   "nodes" (positions; `evaluation_nodes`, `field.nodes`) and differs
   only in container. Rejected: `grid.xr` (a property cannot take
   spaces), `grid.to_xarray` (reads as exporting the grid's data),
   `grid.coords` ("coords" means coordinate *names* throughout the
   spatial layer: `chart_coords`, `param_coords`).
3. **A `wet=True` drop keyword.** xarray's `where(..., drop=True)`
   already says it; the mask as a variable also serves `hue`.
4. **Returning a `DataArray`.** Several coordinates plus labels need a
   `Dataset`; `ds["zp"]` is the DataArray when one is wanted.

## 5. Decision points for the owner

1. Name and home: `grid.nodes(*what, params=None)` on `Grid`
   (recommended), or a free function `fr.spatial.nodes(grid, ...)`.
2. One point layout always (recommended), or structured for a single
   space and points for several.
3. Factor-only query spelled by coordinate name (`grid.nodes("z")`,
   recommended) or by mesh (`grid.nodes(grid.factor("z"))`); and
   whether it returns centres + outer faces (recommended) or centres
   only.
4. `params=` accepting a state (`VectorField`) by name lookup
   (recommended, mirrors `mapping_params`) or only an explicit mapping.
5. Extend `evaluation_nodes`/`field.nodes` with mapped names and
   `params=` (recommended, and it fixes `b_total` under `maps=`), or
   add a separate `grid.positions`.
6. Index variable naming: `i_y` (recommended) or `y_index`.
7. Phase 2 `model.nodes` with units: in scope now or later.

## 6. Implementation sketch (after approval)

- `spatial/coordinate_mapping.py`: `CoordinateMapping.positions(space,
  name, *, params=None) -> ScalarField` (primal of the map at the
  nodes; static defaults or live fields via the existing `params=`
  overload). `grid.evaluation_nodes` routes mapped names to it;
  `ScalarField.nodes` gains `params=`.
- `spatial/export.py`: `nodes_dataset(grid, *what, params=None)`;
  `Grid.nodes` forwards. Per space: `ExportLayout` for the dims and
  the 1D coordinate vectors, `jnp.meshgrid`-free broadcast to the
  point cloud, `_host_labels` for the gather, mask and `zp` gathered
  through `decomposition.gather` like field values.
- `hydrostatic/diagnostics.py`, `nonhydro2/diagnostics.py`: `b_total`
  reads the physical column on a mapped grid (own test on a terrain
  column and on z*).
- Tests (mirrored): `tests/spatial/test_export.py` shards for the
  node cloud (stretched mesh, staggered spaces, factor-only, chart
  units, immersed mask, terrain `zp` with static and live params,
  forced-4 device invariance); `tests/spatial/test_coordinate_mapping*`
  for `positions`; the `b_total` gates.
- Docs: the coastal page replaces its hand-rolled scatter; a short
  "Looking at the grid" section in the guide; an immersed and a z*
  example pick it up when they land.

## 7. Non-goals

Drawing mesh lines or cell polygons (the index variables make it
possible by hand), a plotting module of FRIDOM's own, the chart
embedding and cubed-sphere/unstructured meshes (no node path exists
for them yet), CF auxiliary-coordinate promotion of `zp` in the io
Writer (its `coordinates` attribute is hardcoded to `iteration`).
