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

## 3. Proposed user-level API (for approval, revised 2026-08-23)

Owner constraints (2026-08-23, second round): the **function space is
the only input** that decides what is plotted; **no `x_right`-style
names**; the coordinates of a node are addressed by the plain names
`x`, `y`, `z`; a dimension subset such as `("x", "y")` must be
selectable; xarray's own plotting must work, including nodes drawn
over a heatmap. The first draft's point cloud with multi-space labels
and suffixed names is withdrawn.

### 3.1 `grid.nodes(space)` — the nodes of one function space

```python
grid.nodes(space, *, params=None) -> xr.Dataset
```

One space in, one `Dataset` out, laid out **exactly like
`ScalarField.xr`** (spec `grid.md` §xarray, single-field rule): the
dims are the space's coordinate names, plain, and each dimension
coordinate is the 1D node vector of that factor **at the factor's own
position** — centres for `center`/`cell_avg`, faces for
`right`/`outer`/`inner`/`left`/`face_avg`. The position survives only
as the `c_grid_axis_shift` attribute, as it does on `field.xr`.
Constant factors are squeezed, as everywhere in the export.

```
<xarray.Dataset>                                   # grid.nodes(model.state["w"].function_space)
Dimensions:  (x: 1, y: 96, z: 49)
Coordinates:
  * x        (x) float64                            # centres
  * y        (y) float64                            # centres
  * z        (z) float64  c_grid_axis_shift: 0.5    # the outer faces, where w lives
Data variables:
    zp       (x, y, z) float64                      # physical coordinate of maps=, when a mapping is attached
    wet      (x, y, z) bool                         # immersed wet mask, when a domain is attached
```

- **Tensor coordinates** stay 1D (the meshgrid is never stored, spec
  01_concepts); xarray broadcasts them when a plot asks for two of
  them. A `MappedIntervalMesh` stretch and a chart's radians appear as
  the node values; `units` is set when the grid declares it
  (`grid.coordinate_units`, a chart), otherwise absent, the grid
  being scaling-agnostic.
- **Physical coordinates of `maps=`** are full N-D data variables
  named by the map key (`zp`), the map **value** at the nodes.
  `params=` is a mapping of parameter fields or a state (a
  `VectorField`) looked up by the mapping's `param_names`, the way
  `mapping_params` does; `None` is the mapping's static default (the
  reference geometry). `grid.nodes(space, params=model.state)` is the
  z* column at the current free surface.
- **Immersed**: `wet` from `grid.immersed.mask(space)` when the space
  resolves every coordinate (the mask is defined for nothing less);
  dry nodes are kept and filtered with `where`.
- Host-side, not jittable, collective-safe (built on `ExportLayout`
  and `_host_labels`, gathered once).

### 3.2 The four requests, in xarray's own vocabulary

```python
b = model.state["b"]; w = model.state["w"]
nodes_b = grid.nodes(b.function_space)
nodes_w = grid.nodes(w.function_space)

# nodes over a heatmap of the field, two spaces on the same axes
ax = b.xr.isel(x=0).plot(x="y", y="z").axes
nodes_b.isel(x=0).plot.scatter(x="y", y="z", ax=ax, color="k", s=4)
nodes_w.isel(x=0).plot.scatter(x="y", y="z", ax=ax, color="w", marker="_")

# only some dimensions: the tensor coordinates of (y, z), a 2D section
nodes_b[["y", "z"]].plot.scatter(x="y", y="z")

# only the z mesh: the 1D coordinate itself
nodes_b.z                      # DataArray (z: 48); nodes_w.z are the 49 faces
nodes_b.z.diff("z").plot()     # the spacing of the stretched column

# moving geometry: physical height at the current state
grid.nodes(b.function_space, params=model.state).isel(x=0).plot.scatter(x="y", y="zp")

# immersed: wet cells only
cells = grid.nodes(grid.factor("x").center * grid.factor("y").center)
cells.where(cells.wet, drop=True).plot.scatter(x="x", y="y", s=2)

# the list of points, when one is wanted
nodes_b.stack(node=("x", "y", "z"))      # (node: nx*ny*nz), or .to_dataframe() for pandas
```

All of this is checked against xarray 2026.4 (scatter broadcasts 1D
dimension coordinates; `ax=` overlays on a `pcolormesh`; `isel` keeps
`zp` consistent on a mapped grid; `ds[["y", "z"]]` is the dimension
subset; `stack` gives the point list). Hence **no `dimension=`
keyword**: `ds[["y", "z"]]` selects tensor coordinates, and `isel` is
the right spelling when a mapped coordinate depends on the dropped
axis (a subset cannot decide which slab). Several spaces on one plot
are several calls on one `ax`, matplotlib's own idiom; a `hue` label
would need names for spaces, which is what the owner does not want.

### 3.3 Physical positions at the field level (the primitive)

```python
grid.evaluation_nodes(space, name, *, params=None) -> ScalarField   # name may be a mapped name
field.nodes(name, *, params=None) -> ScalarField
```

`evaluation_nodes` accepts, besides the grid coordinates it takes
today, the mapped names of `grid.mapping` and returns the map value at
the nodes of `space`, parameters resolved through the same `params=`
overload `grid.metric` uses (static defaults when `None`). This is
the traced, jit-safe primitive 3.1 gathers `zp` from, and it is what
`b_total` should read on a mapped column (`b + N^2 zp`). On the
`CoordinateMapping` side it is one new public method
(`positions(space, name, params=)`) next to `metric`, built on the
existing `_jvp`/`_param_at_nodes` machinery but evaluating the primal
with live parameters. A chart's embedding is not in this step
(`lon`/`lat` are the sphere's own coordinates); ambient `x, y, z`
would be a later `ambient=True`.

### 3.4 Sugar (phase 2, optional)

`model.nodes(space)` = `grid.nodes(space, params=model.state)` with
the model's unit rows stamped on the coordinates (`model.units`, the
rows the Writer stamps), so the axes of a plot read `y [m]`.

## 4. Alternatives considered

1. **A flat point cloud** (`node` dimension, variables `x, y, z`, a
   `space` label) — the first draft. Matches "a list of points", and
   `hue="space"` draws several spaces in one call, but it needs a
   name per space (the suffixed spellings the owner rejects or field
   names that a bare space does not have), a `dimension=` keyword to
   avoid duplicate points, and it cannot take `isel`; on a mapped grid
   a dimension subset has no single answer. The structured layout
   reaches the point list in one `stack`, so nothing is lost.
2. **`dimension=` keyword.** `ds[["y", "z"]]` is the same thing in
   xarray's words, and `isel` is the correct one under `maps=`.
3. **Method name.** `grid.nodes(space)` keeps the codebase's meaning
   of "nodes" (positions; `evaluation_nodes`, `field.nodes`) and
   differs only in container; rejected `grid.xr` (a property cannot
   take a space), `grid.to_xarray` (reads as exporting data),
   `grid.coords` ("coords" means coordinate *names* in the spatial
   layer).
4. **pandas or an own plotting module.** Not needed: xarray draws the
   overlays, and `to_dataframe()` is one call away.

## 5. Decision points for the owner

1. `grid.nodes(space, *, params=None)` returning the structured
   `Dataset` of 3.1 (recommended), i.e. the `ScalarField.xr` layout
   with plain names.
2. Extend `evaluation_nodes`/`field.nodes` with mapped names and
   `params=` (recommended; fixes `b_total` under `maps=`) rather than
   a separate `grid.positions`.
3. `params=` accepting a state by name lookup (recommended).
4. Phase 2 `model.nodes` with units: now or later.

## 6. Implementation sketch (after approval)

- `spatial/coordinate_mapping.py`: `CoordinateMapping.positions(space,
  name, *, params=None) -> ScalarField` (primal of the map at the
  nodes; static defaults or live fields via the existing `params=`
  overload). `grid.evaluation_nodes` routes mapped names to it;
  `ScalarField.nodes` gains `params=`.
- `spatial/export.py`: `nodes_dataset(grid, space, params=None)`;
  `Grid.nodes` forwards. `ExportLayout` (plain names) gives the dims
  and the 1D coordinate vectors through `_host_labels`; `zp` and
  `wet` are gathered through `decomposition.gather` like field
  values.
- `hydrostatic/diagnostics.py`, `nonhydro2/diagnostics.py`: `b_total`
  reads the physical column on a mapped grid (own test on a terrain
  column and on z*).
- Tests (mirrored): `tests/spatial/test_export.py` shards for the
  node dataset (stretched mesh, staggered spaces, lone factor, chart
  units, immersed mask, terrain `zp` with static and live params,
  forced-4 device invariance); `tests/spatial/test_coordinate_mapping*`
  for `positions`; the `b_total` gates.
- Docs: the coastal page replaces its hand-rolled scatter; a short
  "Looking at the grid" section in the guide; an immersed and a z*
  example pick it up when they land.

## 7. Non-goals

Drawing mesh lines or cell polygons (the N-D `zp` variable makes
lines possible by hand), a plotting module of FRIDOM's own, the chart
embedding and cubed-sphere/unstructured meshes (no node path exists
for them yet), CF auxiliary-coordinate promotion of `zp` in the io
Writer (its `coordinates` attribute is hardcoded to `iteration`).
