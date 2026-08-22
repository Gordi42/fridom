---
status: complete
date: 2026-08-22
---

# Agent brief: CDFViewer quiver scaling on anisotropic sections, plus two smaller defects

A self-contained task description for an agent working in
`~/Projects/CDFViewer.jl`. Found while authoring the FRIDOM gallery page
`examples/hydrostatic/coastal_upwelling.py` (branch `docs/coastal-upwelling`)
against CDFViewer **v2026.8.3** (`master` at `2829706`, tag `v2026.8.3`).
Copy everything below the rule into a CDFViewer.jl session.

---

## Context

CDFViewer's consumer here is FRIDOM's documentation gallery. An example
records through the python package (`python/src/cdfviewer/_run.py::record`,
which builds a CLI line in `_command.py` and formats values with
`_format.py::format_value`), so every fix has to work from the command
line with `--kwargs`, without a window and without per-plot hand tuning.
(On the command line the base plot type `-p` has to be given before an
`--over` layer, or the overlay is refused with "Select a plot type
before overlaying a second field"; the python wrapper passes it.)
The call that exposed the defects is

```python
cv.record("coastal_upwelling.zarr", var="b_total", x="y", y="z", dims={"x": 0},
          plot_type="heatmap", ani_dim="time",
          over=["v,w", "b_total"], over_plot=["quiver", "contour"],
          kwargs={"color": "black", "arrows": (60, 20), "aspect": 3.0,
                  "figsize": (1200, 430), "xunit": "km", "cbarlabel": "auto",
                  "over2.levels": 15, "over2.color": "white", ...},
          filename="coastal_upwelling.mp4", framerate=16)
```

on a store whose `y` spans 0 to 45000 m and `z` spans -150 to 0 m, with
`|v|` up to 0.1 m/s and `|w|` up to 4e-4 m/s. Three things went wrong,
in order of importance. Work through them as three commits with their
own tests; release as a `v2026.8.x` tag when done (FRIDOM then bumps its
pinned package version).

## Standalone reproduction (no FRIDOM needed)

```python
import numpy as np, xarray as xr
y = np.linspace(234.375, 44765.625, 96)                 # m, cell centres
z = -150.0 + (np.arange(48) + 0.5) * 150.0 / 48          # m
time = np.arange(3.0)
shape = (time.size, y.size, z.size)
b = np.broadcast_to(1e-4 * z, shape).copy()
v = np.broadcast_to(0.1 * np.exp(z / 10.0), shape).copy()  # surface layer flowing toward +y
w = np.zeros(shape)
ds = xr.Dataset(
    {"b": (("time", "y", "z"), b, {"long_name": "Total buoyancy", "units": "m/s^2"}),
     "v": (("time", "y", "z"), v, {"long_name": "Offshore velocity", "units": "m/s"}),
     "w": (("time", "y", "z"), w, {"long_name": "Vertical velocity", "units": "m/s"})},
    coords={"time": time, "y": ("y", y, {"units": "m"}), "z": ("z", z, {"units": "m"})})
ds.to_netcdf("section.nc")
```

```bash
cdfviewer section.nc -v b -x y -y z -p heatmap --over v,w --over-plot quiver \
  --kwargs='color=:black, arrows=(60, 20), aspect=3.0, figsize=(1200, 430), xunit="km"' \
  --savefig -s 'filename="section.png"'
```

Expected: a field of black arrows pointing right, longest in the top
ten metres, each about as long as the gap to its neighbour. Actual: a
lattice of dots. The same happens with `--record`, and with or without
`xunit`.

## 1. Quiver arrows vanish on an anisotropic section (the main task)

**Cause.** `decimate_vector_field` (`src/Plotting.jl`, the function
above `vector_field_observable`, about line 4230) derives one arrow
length for the whole field, in data units:

```julia
cellx = length(dx) > 1 ? abs(dx[2] - dx[1]) : 1.0
celly = length(dy) > 1 ? abs(dy[2] - dy[1]) : 1.0
lengthscale = Constants.VECTOR_ARROW_FILL * min(cellx, celly) / reference
```

with `VECTOR_ARROW_FILL = 0.9` and the reference the 0.98 quantile of the
speed (`src/Constants.jl:196`). `quiver_plot!` hands that single number
to `arrows2d!` as `lengthscale`. On the section, `cellx` is 750 m and
`celly` a few metres, so an arrow of the reference speed is a few metres
long along an axis that spans 45 km, which is well under a pixel. On a
square domain both spacings agree and the rule is right; on a section it
takes the spacing of the axis the arrows do not point along. The drawn
*direction* is wrong on any anisotropic axis too, since a data-space
vector is stretched by the axis aspect on screen.

**What the fix should do.** Size and orient the arrows in screen space.
The cleanest version: take the axis' data-to-pixel factors (the
`finallimits` span against the scene viewport size), convert the
decimated components to pixels per unit speed, pick the length scale so
that the reference speed spans `VECTOR_ARROW_FILL` of the *pixel* gap
between neighbouring arrows, and convert back to the data-unit
components `arrows2d!` wants. The direction on screen is then the
direction of `(u, v)`, and the length is independent of the data aspect.
The field has to re-lay when the viewport changes (a resize, or the axis
rebuild a display unit triggers), the way it already re-lays on the
arrow-count and cutoff settings (`refresh_vector_density!`). A simpler
variant that is exact only when the arrow grid is isotropic in pixels is
to scale each component by its own spacing (`du * cellx / reference`,
`dv * celly / reference`, `lengthscale = VECTOR_ARROW_FILL`). The
geographic branch is the precedent for a per-component correction
before the call (the `1/cos(latitude)` loop right below the line above).

**Keep working.** `xunit`/`yunit` (the axis rebuild replays the kwargs
through `apply_property_mappings!`, about line 3170; `color=:black` has
to survive the rebuild, which the replay comment there says it now
does, so add a test for that while you are in there), `minspeed`,
`every`, the map projection corrections, and the empty-field guard.

**Tests** go next to the existing ones in `test/test_plotting.jl`
(`"Robust length scale"`, `"Decimated field"`, `"Geographic
corrections"`, from about line 2640). The first of those pins
`field.lengthscale ≈ 0.9 * 1.0 / 1.0` on a 100x10 unit grid; adapt it
rather than delete it. Add: a 96x48 grid with `x` in 0..4.5e4 and `y` in
-150..0, `u = 0.1`, `v = 0`, `arrows=(60, 20)`, asserting that the drawn
arrow length in pixels is a sizeable fraction of the horizontal arrow
spacing in pixels (whatever the new struct exposes for that), and that
an isotropic square grid draws exactly what it drew before.

**Docs.** `docs/src/usage/plot_types.md`, the quiver table: say that
arrows are sized on screen, so a section with axes in different units
draws them at the same length as a map does.

## 2. `over.lengthscale` is accepted and ignored

Passing `over.lengthscale=1.5e4` (or the unprefixed form) through
`--kwargs` records without any message and changes nothing.
`resolve_kwarg` (about line 3050) finds no owner in the flat namespace,
splits the prefix, finds no layer *setting* of that name, and lands on
the `lengthscale` attribute of the layer's `arrows2d` plot. That
attribute is the lifted observable `@lift($field.lengthscale)` from
`quiver_plot!`, so the next field notification, which every recorded
frame produces, writes the automatic value back over it.

Once task 1 is in, an explicit override is still worth having as an
escape hatch. Make `lengthscale` a layer setting like `arrows`, `every`
and `minspeed` (`LayerSettings`, about line 900; `set_layer_setting!`,
about line 3128; the per-layer readers `layer_arrows` and friends), read
by `decimate_vector_field` as "use this instead of the automatic
scale", in the same units the automatic scale has after task 1. Then
`over.lengthscale` and `lengthscale` mean the same thing on every
replay, and `del lengthscale` restores the automatic one. Document it in
the quiver table.

## 3. An explicit `levels` vector on a contour overlay fails to render

```bash
cdfviewer section.nc -v b -x y -y z -p heatmap --over b --over-plot contour \
  --kwargs='over.levels=[-0.015, -0.014, -0.013, -0.012, -0.011, -0.01, -0.009, -0.008, -0.007, -0.006, -0.005, -0.004, -0.003, -0.002, -0.001], over.color=:white' \
  --record -s 'filename="section.mp4", framerate=16'
```

fails with

```
Failed to update renderobject - skipping update
  exception =
   Failed to resolve gl_renderobject:
   [ComputeEdge] gl_renderobject = #register_robj!##0((space, scaled_color, ...
     @ GLMakie/src/plot-primitives.jl:238
```

while `over.levels=15` works. The python package writes the list as a
Julia vector literal, so the value arrives as a vector, which
`user_levels`/`pin_levels!` (about line 2572) explicitly allow ("an
explicit vector always wins over the pin"; the pin is skipped for
vectors). The comment in `pin_levels!` about Makie's compute graph typing
the levels edge from its first render, and range versus vector not
converting into a frozen edge, is the likely neighbourhood: check what
the first render was built with and what type the parsed vector has
(`Vector{Any}` from the parser would be a candidate). Reproduce with
`--savefig` first, which may already fail, then `--record`. Add a test
that a contour layer given an explicit vector draws those lines, in the
REPL path and the kwargs path.

## Related, for awareness

- `design/research/artifacts/cdfviewer_aspect_header_prompt.md` in the
  FRIDOM repo (2026-08-13) reported two rendering defects from the same
  gallery work; check whether they are closed before releasing.
- The FRIDOM style guide records that `color=:black` on a quiver and
  `xunit="km"` were mutually exclusive in the 2026.7 line (a unit
  conversion regenerated the magnitude colours). The replay logic in
  `apply_property_mappings!` reads as the fix; confirm it with a test
  under task 1 so the FRIDOM note can be retired.

## Verification from the consumer side

After the tag, in `~/Projects/fridom`: bump the `cdfviewer` pin, then run
`examples/hydrostatic/coastal_upwelling.py` (about a minute; it writes
`coastal_upwelling.mp4` next to itself) and look for right-pointing black
arrows in the top ten metres at the left coast, left-pointing ones
below, and fifteen white isopycnals; `examples/hydrostatic/geostrophic_adjustment.py`
is the isotropic control and must look as it does today.
