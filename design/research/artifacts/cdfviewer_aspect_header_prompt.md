---
status: complete
date: 2026-08-13
---

# Hand-off prompt: CDFViewer aspect and header defects

Copy everything below the rule into a CDFViewer.jl session. Found while
rendering FRIDOM gallery animations at v2026.7.2
(`~/Projects/CDFViewer.jl` @ `8b933f6`).

---

Two rendering defects in CDFViewer, both hit while recording animations
of ocean model output. Repo is `~/Projects/CDFViewer.jl`, currently at
`8b933f6` (v2026.7.2). The consumer is FRIDOM's documentation gallery:
example scripts call `cdfviewer <store> ... --record` as a visible line
in the published page, so any fix has to work from the CLI without
per-plot hand-tuning.

## Defect 1 — the data aspect is discarded for elongated domains

`compute_aspect` (`src/Plotting.jl:3152`) computes the data ratio and
then keeps it only inside a hard window:

```julia
x_ext = maximum(x) - minimum(x)
y_ext = maximum(y) - minimum(y)
ratio = x_ext / y_ext
ratio > 0.25 && ratio < 5 && return ratio
# compute default aspect from figure size
figwidths[1] / figwidths[2]
```

**Why this is wrong for the main use case.** Ocean and atmosphere
cross-sections are routinely far more elongated than 5:1 — a domain
2400 m wide by 120 m deep is 20:1, and that is an ordinary shelf slice,
not a pathological input. Such a plot silently loses its true aspect
and is drawn at the *figure's* aspect instead, so the axis letterboxes
into a near-square box floating in the canvas, with the physical
geometry of the slope distorted. Since the aspect is what carries the
physics in these plots (a 1-in-19 bottom slope has to look like one),
the fallback defeats the plot.

The current workaround at the call site is to pass `aspect=6.0`
explicitly on every such plot, which means the caller has to know the
threshold exists.

**Related inconsistency.** The 3-D overload (`src/Plotting.jl:3182`)
applies the same `0.25 / 5` clamp but substitutes `1` rather than the
figure aspect:

```julia
ratio = [r > 5 || r < 0.25 ? 1 : r for (i, r) in enumerate(ratio)]
```

So the two paths disagree about what an out-of-range aspect should
become. Whatever the fix, the two should agree.

**What to work out.** The clamp presumably exists so an extreme ratio
cannot produce an unusable sliver of an axis. That is a real concern
and the fix should not simply delete the bound. Worth considering:
honour the data aspect over a much wider range and clamp only at the
genuinely degenerate end; or keep the data aspect but bound the
*rendered* axis extent rather than the ratio; or let the aspect through
and let the figure size absorb it. Please look at what Makie actually
does with a large `aspect` on a `DataAspect`-style axis before picking,
and check the interaction with `rebuild_header!`, whose docstring
already accounts for "an aspect-letterboxed axis floats centred in its
cell".

**Repro.** Any 2-D heatmap whose x extent over y extent is outside
`[0.25, 5]`, for instance a store on a 2400 x 120 m (x, z) grid, with
no explicit `aspect` in `--kwargs`.

## Defect 2 — title and animation label collide

At the default `animlabelpos=:title` (`Constants.ANIMLABEL_POSITION`),
a long `title=` together with a long animation label overlap on the
header line.

`rebuild_header!` (`src/Plotting.jl:1511` onward) does intend to share
the line: `header_label_width` reserves the label's width and
`fit_title_size` shrinks the title to give way. But that shrink floors
at `Constants.TITLESIZE_MIN`:

```julia
max(Float64(size) * Float64(available) / width,
    Float64(Constants.TITLESIZE_MIN))
```

so once the label claims enough of the line that the title would need
to go below the floor, the two are drawn over each other instead of
either being truncated, wrapped, or moved.

**What to work out.** The floor is reasonable (an unreadably small
title is not a fix either), so the question is what should give when
both cannot fit: elide the title, move the label to `:overlay`
automatically, wrap to a second header line, or shrink the label as
well as the title. Whichever it is, the failure mode should not be
silent overlap.

The current workaround at the call site is
`animlabelpos=:overlay, animlabelcorner=:rt`.

**Repro.** Record any animation with both a `title=` of a dozen or so
characters and a wide `animlabel=` (for instance
`animlabel="t = {rawvalue} s", animlabelnumfmt="%.1f"`) at the default
`animlabelpos`.

## Notes

- Please add regression coverage for both, in whatever form the repo's
  test suite supports for rendering behaviour — at minimum, a unit test
  on `compute_aspect` pinning what a 20:1 ratio returns, and one on
  `header_label_width` / `fit_title_size` pinning the both-too-wide
  case.
- If the aspect fix changes default rendering for existing plots, say
  so explicitly: FRIDOM example pages pass explicit `aspect=` in a few
  places as a workaround for defect 1, and those call sites should be
  cleaned up once the default is right.
- Release is a `v*` tag push. FRIDOM's docs CI pins the release binary,
  so a fix needs a tagged release before the gallery can drop its
  workarounds.
