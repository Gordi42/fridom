---
status: complete
date: 2026-08-22
---

# Hand-off prompt: CDFViewer quiver arrows vanish on anisotropic sections

Copy everything below the rule into a CDFViewer.jl session. Found while
rendering the coastal upwelling gallery animation
(`examples/hydrostatic/coastal_upwelling.py`) against the `cdfviewer`
python package pinned in `uv.lock` (`~/Projects/CDFViewer.jl`).

---

A `quiver` overlay draws as a field of dots on a vertical section whose
two axes differ by orders of magnitude in data units. The consumer is
FRIDOM's documentation gallery, where an example records

```python
cv.record("coastal_upwelling.zarr", var="b_total", x="y", y="z",
          dims={"x": 0}, plot_type="heatmap", ani_dim="time",
          over=["v,w"], over_plot=["quiver"],
          kwargs={"color": "black", "arrows": (60, 20), "aspect": 3.0,
                  "figsize": (1200, 430), "xunit": "km", ...})
```

on a store whose `y` spans 0 to 45000 m and `z` spans -150 to 0 m, with
`|v|` up to 0.1 m/s and `|w|` up to 4e-4 m/s.

Cause, from `src/Plotting.jl` (`decimate_vector_field`):

```julia
lengthscale = Constants.VECTOR_ARROW_FILL * min(cellx, celly) / reference
```

One scalar in data units, taken from the *smaller* sample spacing. Here
that is the vertical one (about 7.5 m between arrow rows), so an arrow
of the reference speed is 7.5 m long along an axis that spans 45 km,
which is well under a pixel at any figure size. On a square domain the
two spacings agree and the rule is right; on a section it picks the
axis the arrows are not pointing along.

What the fix should do: size the arrows in screen space rather than in
data units, so that the reference speed spans a fixed fraction of the
arrow spacing *in pixels* whatever the data aspect. Equivalently, scale
each component by its own axis (`u` by `cellx`, `v` by `celly`, or by
the axis' data-to-pixel factor), so that the drawn direction is the
direction on screen. Makie's `arrows2d!` takes a single `lengthscale`,
so the component scaling has to happen in `du`/`dv` before the call,
the way the `1/cos(latitude)` correction already does for maps. The
`xunit`/`yunit` axis conversion has to keep applying after that, and
`color=:black` has to survive it (the same interaction noted in the
geostrophic adjustment hand-off).

A regression to add: a 2-D field on a grid with `x` in 0..4.5e4 and
`y` in -150..0, `u = 0.1`, `v = 0`, `quiver` with `arrows=(60, 20)`,
and an assertion that the drawn arrow length in pixels is a sizeable
fraction of the horizontal arrow spacing in pixels.
