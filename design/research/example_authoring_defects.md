---
status: complete
date: 2026-08-13
---

# Defects surfaced while authoring the new gallery examples

Findings from the 2026-08-13 gallery expansion: four capability audits
of the new stack plus the authoring of new example pages, each of which
assembled and ran real models. Companion to
[`gallery_expansion_survey.md`](gallery_expansion_survey.md) (which
examples to add and what they cost).

**Nothing here has been acted on.** Every item is a `src/` (or docs, or
CDFViewer) change outside the scope of the example work. Entries are
marked **reproduced** (an executed probe hit it) or **read** (found by
code audit, not executed). This file is a frozen input; whichever items
the owner promotes belong in
[`../roadmap/open.md`](../roadmap/open.md), which this change does not
touch.

## A. Framework bugs

### A0. Advection is unconditionally unstable at a sloping or cut boundary — **reproduced, minimal repro in framework units**

**The most serious finding in this file.** With advection enabled, a
model on a terrain-following (mapped) grid or an immersed cut-cell grid
blows up within tens of steps, at every amplitude, slope, resolution and
time step tried. The exact solution in the repro is a steady flow over a
bump.

Minimal reproduction (`clean_repro.py`, artifact below) uses
`tests/model/modules/test_advection_mapped.py`'s own grid — `zp = z*H`,
`H = 1 + 0.2 sin x` on a 2-pi box — with uniform `u = 0.4`, `n2=0`, no
friction:

```
n=32 dt=0.02  (u dt/dx = 0.041)   step 25: |u|max=1.06e3 at iz=31; step 50 non-finite
n=32 dt=0.005 (u dt/dx = 0.010)   step 100 non-finite
n=64 dt=0.01                      step 25: |u|max=1.9e6  at iz=63; step 50 non-finite
```

The identical run on a **flat** grid holds `|u| = 0.05000` to all digits
indefinitely.

Established by bisection:

- The growing mode sits in the **boundary-adjacent cell row** (`iz=0` or
  `iz=nz-1`, whichever carries the sloping wall) and is smooth in x
  (Nyquist content 1e-16 of the peak) — not a checkerboard.
- Growth rate is **independent of `dt`** (dt/4 gives the same growth per
  unit time) and scales like `U/dz`: a spatial-discretization
  instability, not a CFL violation.
- Hits **both** geometries: the mapped grid at slopes down to 1.7%, the
  immersed grid at ~34% (immersed survives gentle ~4% slopes).
- Hits **both** advection families: `WENOAdvection(order=5)` on the
  immersed grid fails the same way, so it is not centered-vs-biased.
- `HarmonicFriction` does not rescue it (worse at `nu=5` / `nu_v=0.2`).
- **Not** a divergence-operator mismatch: constancy preservation was
  checked explicitly and flux-form advection of a constant tracer leaves
  a residual of 3.6e-11 (mapped) / 4.2e-10 (immersed), i.e. at the
  pressure-solver tolerance.

Adjacent observation from the same probe, and a candidate cause for the
immersed half: projecting a uniform `u = 1` on the immersed grid produces
`|u|max = 5.70` in the cut cells near the crest, against 1.64 on the
mapped grid (which is the physical continuity speed-up). A 5.7x
overshoot in sliver cells has the shape of an unguarded small-cell
problem.

**Why the suite does not catch it.**
`tests/model/modules/test_advection_mapped.py` only ever evaluates a
**single tendency** (flat identity, 2nd-order convergence, taught
errors). Nothing in the suite integrates a mapped or cut-cell grid
forward in time with advection and a moving flow.

*Cost so far:* `examples/nonhydro/internal_tide_over_a_ridge.py` ships
both legs with `advection=False`. The physics justifies that
independently (tidal excursion 142 m against a 1000 m ridge, so the
response is linear), but **a nonlinear version of that example cannot
currently be made to run.** Any future example with flow over
topography is blocked on this.

Repro scripts preserved at
[`artifacts/advection_slope_instability/`](artifacts/advection_slope_instability/).

### A1. Stretched vertical mesh is unusable with `nh.Model` — **reproduced**

A bare `MappedIntervalMesh` in `z` fails at the pressure projection:

```
TermEvaluationError: Core/projection: evaluation failed with DispatchError:
  "no operator registered for kind 'transform' on CellAvg(z, bc=(NEUMANN, NEUMANN))"
```

`nonhydro2/modules/core.py:776-781` routes the projection on
`grid.mapping.column_corrections` only. A stretched **mesh factor**
never sets that, so the grid falls through to
`SpectralPressureSolver`, which needs a spectral transform on the z
factor — and `MappedIntervalMesh.cosine()` deliberately raises.
Reproduced on both families (`nodal` gives the same error on
`Center(z, ...)`) and with `pressure_preconditioner="multigrid"`. The
failure is a bare `DispatchError`, not a taught error.

**The capability exists.** `MappedIntervalMesh` plus a dummy
`CoordinateMapping(maps={"zp": lambda z: z})` works once
`pressure_preconditioner="none"` is set, so the mapped PCG route
handles a stretched column correctly; only the routing predicate
misses it.

*Cost so far:* the Ekman spiral example
(`examples/nonhydro/ekman_spiral.py`) was to demonstrate a stretched
mesh resolving the surface layer. It ships a uniform resolution
convergence study instead.

### A2. Uncaught `StopIteration` on the chart vertical spelling — **reproduced**

`nonhydro2/modules/core.py:584` —
`next(vs.factor(axis) for vs in vel if ...)` finds no staggered face on
the vertical axis and raises a bare `StopIteration`. Triggered by
`Grid((mx, my, IntervalMesh(nz, (0,1), name="s")), mapping=...)` with
`nh.Core(vertical="s", coords=("x","y","s"))`; same for a map output
named `"zp"`.

### A3. `pressure_tolerance` early exit appears not to fire — **measured, not root-caused**

On the masked / spectral-preconditioned immersed route, dropping
`nh.Core(pressure_iterations=...)` from the default 30 to 12 roughly
doubled throughput (11.1 -> 23.4 steps/s) while producing **state
identical to all printed digits after 400 steps**. That suggests the
default `pressure_tolerance=1e-8` early exit is not firing before the
fixed iteration budget is spent, so iterations beyond roughly 12 are
computed and discarded.

If it generalizes, this is an across-the-board speedup on every
immersed run, and the highest-value item in this file. Not
investigated further; `src/` untouched.

### A3b. Advection `background=` is silently a no-op on an immersed grid — **reproduced**

`nh.WENOAdvection(order=5, background={"u": U})` on a grid carrying an
`ImmersedDomain` assembles, runs, and **simulates nothing**: the
immersed body is completely transparent to the background flow.

The Doppler split is exact for a constant `U` only through the
*equations*; the boundary condition it needs at the body is
inhomogeneous, `u'.n = -U.n`. `ImmersedPressureSolver` + `MaskState`
enforce the homogeneous `u'.n = 0` and zero `u'` in dry cells, so
**`u' == 0` is an exact fixed point.** Measured: a 256x128 grid with a
disc, seeded with 1e-3 noise, gave `max|u'| = 1.1e-3` after 350 steps
with no wake structure whatsoever.

This looks like a **missing bind-time check** rather than design
intent. `model/modules/advection.py` already validates that the
wall-normal background component vanishes on a *walled* mesh
(`_check_background`, bind validation near line 221); there is no
equivalent for immersed grids —
`grep background src/fridom/nonhydro2/modules/immersed_pressure.py`
is empty. A taught `AssemblyError` there would turn a silent
wrong-physics run into an instant diagnosis.

**Severity note:** unlike A0, this fails *quietly*. A user gets a
clean run, a plausible-looking flat field, and no indication that the
obstacle did nothing.

*Cost so far:* `examples/nonhydro/island_wake.py` abandoned the
`background=` route entirely and uses the total velocity as the
prognostic with a `Relaxation` inflow fringe.

### A4. `Model.advance` dispatches every remainder step individually — **mechanism confirmed, magnitude unresolved**

`_chunk_plan` (`model/model.py:2430-2436`) compiles chunk lengths
`{C, 1}` only:

```python
full, tail = divmod(steps, self._chunk_size)
for _ in range(full):
    yield self._chunk_size
for _ in range(tail):
    yield 1
```

So any `advance(N)` where `N` is not a multiple of `chunk_size`
(default 256) runs the remainder **one step per dispatch**. The
gallery's normal pattern — a writer trigger subdividing a run into
frames of a few tens of steps — therefore dispatches *every step*
singly, since `N < C` gives `full = 0`.

**Magnitude is unresolved and should not be quoted until measured on a
quiet machine.** One authoring agent measured 107 -> 496 steps/s
(4.6x) at 96x48 by passing `chunk_size=427` for a 427-step call. An
independent probe under heavy contention could not reproduce that
scale: 1.28x on the long-call case, no penalty at all on the
short-segment case, and an internally inconsistent ordering
(`chunk_size=8` fastest) that indicates the measurement was dominated
by load rather than by the effect. Both measurements were contended.

If the effect is real at scale, the fix direction is for `_chunk_plan`
to emit a second compiled length for the tail rather than falling to 1.
Compile cost is reportedly insensitive to chunk length (AB3's
`scan_unroll` is 3, so the scan body is 3 steps regardless).

### A5. `ScalarField + numpy.ndarray` silently degrades to `ndarray` — **reproduced**

```
type(b)        -> ScalarField
b + ndarray    -> ndarray                    # silent, no error
b + jax.Array  -> TypeError                  # correct
```

The jax case raises properly; the numpy case does not, and the loss of
type only surfaces much later inside `VectorField.replace` as
`AttributeError: 'numpy.ndarray' object has no attribute 'metadata'`
(`spatial/fields/vector_field.py:657`). Setting
`__array_ufunc__ = None` on `ScalarField` would make numpy defer and
raise the same `TypeError` at the point of the mistake.

## Ax. Performance and stability constraints found while sizing

Not bugs, but each shaped an example and none is written down anywhere
a user would look.

- **IMEX steppers have no damping escape hatch.** `CNAB2` on the
  hydrostatic model needs the internal-wave Courant number well under
  one: at `omega*dt ~ 1` it panics with non-finite state around
  iteration 211, identically with `advection=False` (so not the
  nonlinearity). `SBDF2` is worse (iteration 127). There is no `eps`
  knob like `AdamBashforth(order=2, eps=0.1)`. Any hydrostatic example
  on an IMEX stepper must size `dt` from the internal wave, not the
  diffusion.
- **Immersed geometry flips the hydrostatic barotropic solve** from
  spectral to a 30-iteration CG: 14.7 steps/s against ~140 flat-bottom
  on the same grid. A shelf in a hydrostatic example needs its own
  budget.
- **Closures on chart grids act along-coordinate.** `diff` divides by
  the coordinate measure in radians and the chart factor never enters,
  so `nu` would carry rad^4/s and the meridional grid scale damps ~80x
  faster than the zonal at a 2:1 aspect. Expressible via `nu_v` +
  `vertical="lat"`, but not a defensible default.
- **Output firings cost 0.25-1.0 s each** against ~10 ms per model
  step. One authored page spent ~60 s of output overhead on a ~45 s
  run before this was noticed.
- **A persistent JAX compilation cache is enabled at import**
  (`~/.cache/fridom/jax`), so the second run of any script is warm —
  worth ~30 s on one measured example. Any budget claim meant to model
  CI must be measured against a cold cache.
- **`metric_weight="csqr"` is a no-op for a constant-depth core** and
  costs measurable step time. Required only for a variable-depth core.

## B. Untested claims and latent hazards — all **read**

- **`SmagorinskyLilly`'s terrain rejection does not exist.** The
  docstring claims a terrain guard; no such guard is in the file. A
  terrain nh2 model would likely bind and run along-sigma silently.
  Untested either way.
- **Flux-form advection is metric-blind on charts** and the intended
  fence is not in the tree. Currently safe only because both 3-D
  models refuse charts first, which makes A2's bare `KeyError`
  load-bearing.
- **Spherical is advertised for `nh` and `hy`**
  (`nonhydro2/model.py:134`, `hydrostatic/__init__.py:78`) with zero
  tests. `nh.Model` on a chart grid raises a bare `KeyError`, not a
  taught error.
- **No test drives `run()` on a sphere.** Coverage is tendency-level
  plus a 6-step `_chunk_body`; the survey's probe was the first
  end-to-end evidence.
- **`TimeAverage` is tested only on a 1-D 8-point periodic toy** — no
  sw2/nh2/hy/channel test exists.
- **`Writer(chunks=)` has zero test coverage.**
- **`BC.ROBIN` is structure-only**: `grid.sync` raises
  `NotImplementedError("ghost_fill")`.

## C. Import-surface drift — **reproduced** (verified absent by import)

Source docstrings advertise spellings that do not exist. An example
copied from any of these fails at import.

| Docstring says | Site | Real spelling |
|---|---|---|
| `fr.BC` | `spatial/bc.py:23` | `fr.spatial.BC` |
| `fr.Real`, `fr.Complex` | `spatial/scalars.py:9,52` | `fr.spatial.*` |
| `fr.Profile` | `model/modules/coriolis.py:35-36,586,721` | `fr.spatial.Profile` |
| `fr.every`, `fr.at` | `io/triggers.py:6` **and its own error messages** | `fr.io.every`, `fr.io.at` |
| `fr.transforms.*` | `model/transforms/__init__.py:6` | `fr.model.transforms.*` |
| `fr.modules.*`, `fr.closures.*`, `fr.time_steppers.*`, `fr.terms`, `fr.params` | various | `fr.model.*` |
| `nh.Relaxation` | — | `fr.model.modules.Relaxation` |

The `io/triggers.py` case is the worst of these: the module's own
**error messages** name a spelling that does not exist, so a user
following the error lands on an `AttributeError`.

Also **absent although a user would reach for them**:

- `fr.spatial.IntervalMesh` — `fr.spatial.Grid`, `CoordinateMapping`,
  `ImmersedDomain` and `BC` all resolve, but meshes live one level
  down (`from fridom.spatial.meshes import IntervalMesh`). Every
  example that builds a non-uniform grid trips on this.
- `nh.geostrophic_energy_spectrum` — `nonhydro2/__init__.py` re-exports
  the random-IC factories but not the spectrum function that is their
  natural companion, so it is only reachable as
  `nh.initial_conditions.geostrophic_energy_spectrum`.
- `fr.spatial.export` and `fr.spatial.symbols` are not reachable as
  submodule attributes (absent from `all_modules_by_origin`), and
  `MissingComponentError` has no public re-export.

`BalanceExpansion` is not lifted to `fr.model.*` although its
siblings are.

## D. Docs falsehood — **read**

`docs/source/tutorials/using_models/parallelization.rst` states in
full that FRIDOM "does not yet support parallelization. We plan to
parallelize the framework using jaxDecomp." That describes the old
stack and is false for the new one. Highest-value single docs fix
surfaced by the audit.

## E. CDFViewer (separate repo, `~/Projects/CDFViewer.jl`) — **reproduced**

Both hit while rendering example animations at v2026.7.2. A ready-to-
hand-off prompt for these lives at
`design/research/artifacts/cdfviewer_aspect_header_prompt.md`.

- **Aspect fallback discards the data aspect for elongated domains.**
  `compute_aspect` (`src/Plotting.jl:3152`) returns the data ratio
  only when `0.25 < ratio < 5`, else falls back to the figure aspect.
  An ocean x-z slice is routinely 20:1, so it letterboxes into a
  near-square axis. The 3-D overload (`:3182`) applies the same clamp
  but substitutes `1`, so the two disagree about the fallback.
- **Title and animation label collide** at the default
  `animlabelpos=:title`. `rebuild_header!` does share the line via
  `fit_title_size`, but that shrink floors at
  `Constants.TITLESIZE_MIN`, below which the two overlap.

Further viewer findings from this round:

- **The recorded kwarg-ordering rule is incomplete.** `roadmap/open.md`
  §5 says `animlabel=` must precede `colorrange=`. With
  `animlabel, colormap, colorrange` and no `animlabelnumfmt`, the
  colorrange was still silently discarded; inserting
  `animlabelnumfmt=` between them fixed it. So the trigger is not
  simply the relative order of those two keys.
- **An existing mp4 is silently renamed, not overwritten** — a
  re-record lands in `name(1).mp4`. Harmless under sphinx-gallery,
  but it corrupted one re-measurement run.
- Two undocumented kwargs that materially improved pages:
  `xunit="km"` / `yunit="km"`, and `animunit="d"` with
  `animlabel="t = {value}"` (renders "t = 7.5 d" rather than raw
  seconds).

Pre-existing viewer items already recorded in `roadmap/open.md` §5 and
not re-verified here: `colorscale=Makie.Symlog10(...)` being passed as
an unevaluated string, and the `-a time` fallback warning.

## F. Minor API surprises — **reproduced**

Each cost an authoring probe cycle. None is a bug, but each is a place
the surface reads differently than it behaves.

- `model.state.fields` does not exist (it is `_fields`).
- `grid.factor("x").centers` does not exist (it is `.center`).
- Field arithmetic drops metadata: a computed field's `.xr` carries
  `long_name="Unnamed"`, `units="unknown"`, so derived plots need
  explicit labels.
- `model.run(runlen=...)` rounds to whole steps, so a sampling window
  that is not an integer multiple of `dt` de-syncs. This cost the Ekman
  example 3-6% on the transport diagnostic before it was found; the
  shipped version divides the sample interval exactly.
- **`fr.io.TimeSeries` has no `mode="w"`.** An existing CSV with a
  matching header is appended to (the resume path), so re-running an
  example forks its own time axis. Two pages had to call
  `Path(...).unlink(missing_ok=True)` and explain why. A `mode=`
  matching `Writer`'s would remove the wart. (An uncommitted
  `src/fridom/io/series.py` adding an in-memory `fr.io.Series` was
  observed in the main checkout and may supersede this.)
- **`OptimalBalance` returns a bare `VectorField`, not the model
  package's `State`.** `balanced.w` and `balanced.rel_vort_z` raise
  `AttributeError`, so a page must write `balanced["w"]` and cannot
  reach the derived diagnostics at all. `BalanceExpansion` and the
  projections both preserve `nh.State` via `type(state)(...)`, so this
  reads as an oversight in `OptimalBalance._evaluate` rather than a
  design choice.
- **`nh.random_vortical` costs ~13 s** on a 32x32x8 grid (spectral
  synthesis over the full lattice), and its default spectrum has no
  vertical decay, so the field is grid-rough in `z` and its peak
  vorticity is ~10x its peak velocity. Both are reasons a budgeted
  page should prefer `nh.coherent_eddy`.
- **`fr.model.modules.Source` with a non-`Harmonic` law** evaluates as
  `law(t) * Q(x)` with no amplitude parameter, so the amplitude must
  live inside the pattern. Not documented beyond "escape hatch".
- **`advection=False` on a 2-D slice fails the coverage lint**:
  `AssemblyError: PROGNOSTIC fields ('v',) are advanced by no term`.
  The workaround is `nh.FPlaneCoriolis(f0=0.0)`, which declares the
  terms with a zero leaf. A non-rotating linear model is otherwise
  unassemblable in a 2-D slice.
- **`BoundaryFlux` sign convention**: a positive flux at the top wall
  *drains* the surface cell, so a wind stress is `flux = -tau_x`.
  `nonhydro2` has a `WindStress` wrapper owning the oceanographic
  sign; `hydrostatic` does not, so every hydrostatic wind-forcing
  example must carry the negation itself.

## G. Stale records and housekeeping — **read**

- `roadmap/open.md` §4b P4 claims `internal_wave_maker` /
  `multiple_wave_makers` / `wave_package` still use deleted APIs; all
  three were ported 2026-08-11.
- `roadmap/open.md` §2e describes `dancing_eddies.py` prose that the
  uncommitted port already removed.
- `FRIDOM_EXAMPLES_FAST` survives in three example files
  (`shallowwater/barotropic_instability.py:34-36`,
  `shallowwater/equatorial_waves.py:49-51`,
  `hydrostatic/comparison_baseline.py:42-43`) despite its retirement
  on 2026-08-12.
- `nonhydro2/modules/core.py:298` still claims mapped + immersed is
  rejected; `_require_fv_capable` lifted that.
- Three shipped plans remain in `plans/active/`:
  `mapped_immersed_composition_plan.md`,
  `immersed_graded_advection_plan.md`,
  `partial_bottom_phyd_plan.md`.
- `ruff check examples/` fails on pre-existing old-stack scripts, so a
  new example author cannot lint the tree as a whole and must lint
  their own file by path. Reported independently by four authors.
- **`examples/**/*.mp4` is not gitignored** although
  `examples/**/*.zarr` is. Every example that records an animation
  leaves an untracked artefact its author must remove by hand; two did
  so, and one added `examples/**/*.csv` for the same reason.
