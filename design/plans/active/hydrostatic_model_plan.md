---
status: active
date: 2026-07-16
---

# Hydrostatic model — implementation plan (ROADMAP 3.1)

**Goal (owner, 2026-07-16):** a hydrostatic primitive-equation model
on the new stack, built like `nonhydro2`, whose end state is a
**matched-numerics comparison setup** against pyOM3, Veros, and
Oceananigans.jl on a doubly-periodic 3D Cartesian grid. The
barotropic (free-surface) solver is the one genuinely new design;
everything else should generalize from existing machinery so that
mapped / time-dependent grids and higher-order advection come out of
the box.

Companion records: the frozen split time-stepping design
[`../../specs/model/03_time_stepping.md`](../../specs/model/03_time_stepping.md)
(§5.2 stage schedule with the hydrostatic DIAGNOSE slots, §5.4 the
split-explicit free surface, "build at 3.1"); the mapped-solve
pattern [`../done/coordinate_systems_plan.md`](../done/coordinate_systems_plan.md)
(CS-D2); the shallow-water barotropic discretization
[`../done/coriolis_energy_correction.md`](../done/coriolis_energy_correction.md);
external-model numerics survey: §6 below.

## 1. Decisions (all seven signed off by owner, 2026-07-16 — the
recommended option in each case)

- **HY-D1 — package name `fridom.hydrostatic`.** The old
  `hydrostatic` package was removed 2026-07-11 (roadmap 3.1 note), so
  the name is free: build un-suffixed from day one, alias
  `import fridom.hydrostatic as hy`. Nothing to rename at cutover.
- **HY-D2 — formulation.** Prognostic `u, v` (C-grid faces), buoyancy
  `b` (linear EOS as a buoyancy tracer, the Oceananigans
  `BuoyancyTracer` / linearized-Vallis common ground), and surface
  pressure `ps = g*eta` (2D, `Profile("x", "y")`, constant along z —
  the `shallowwater2` `p = g*eta` convention, so `csqr = g*H` stays
  the single barotropic parameter). Diagnosed: `w` from continuity
  and `p_hyd` from hydrostatic balance, both recomputed in S1'
  DIAGNOSE stages exactly as the spec schedules them. The model is
  the `dsqr -> 0` limit of `nonhydro2`: `dsqr * dw/dt` drops,
  `d p/dz = b` splits `p` into `p_hyd + ps`.
- **HY-D3 — one free-surface module family, three variants.**
  `ExplicitFreeSurface` (a plain tendency term — the correctness
  oracle and the Oceananigans `ExplicitFreeSurface` analogue),
  `ImplicitFreeSurface` (backward-Euler 2D Helmholtz solve, with a
  pyOM-style `epsilon` knob whose `epsilon=0` limit is the rigid lid)
  and `SplitExplicitFreeSurface` (the §5.4 frozen design). The
  variant is a constructor argument of the model factory; each
  variant module owns the `ps` declaration (PROGNOSTIC for
  free-surface variants, DIAGNOSTIC for the rigid lid).
- **HY-D4 — the implicit solve is a CONSTRAINT-stage projection, not
  a stepper-side `ImplicitOperator`.** The spec names "grad eta as an
  IMPLICIT term" as the designed alternative, but the shipped IMEX
  driver solves merged operators independently from the *raw* rhs
  combine and `updates.update(...)`s the results
  ([`imex.py:349-353`](../../../src/fridom/model/time_steppers/imex.py))
  — two operators sharing a field (a free-surface block on
  `(u, v, ps)` plus `VerticalDiffusion` on `(u, v)`) would pass
  assembly (the collision lint only rejects two *custom* operators
  per field, [`composer.py:736-751`](../../../src/fridom/model/composer.py))
  and then silently drop one solve. A CONSTRAINT stage after S3
  reads the post-mixing state by construction — which is exactly
  pyOM's operator ordering (AB2 slow terms, implicit vertical
  mixing, then the surface solve) and exactly `nonhydro2`'s
  project-the-state pattern (§5.6: "solve, then project"). It also
  composes unchanged with **any** stepper — `AdamBashforth(order=2,
  eps=0.1)` alone reproduces pyOM's quasi-AB2 + implicit surface
  without IMEX. Cost of the route: the barotropic gravity coupling
  leaves the declared linear terms, so the implicit variant declares
  `linear_operator_gap` (the mechanism built for exactly this case);
  see HY-D7.
- **HY-D5 — rehome the advection modules to `fr.model.modules`.**
  `CenteredAdvection` / `UpwindAdvection` / `WENOAdvection`
  ([`nonhydro2/modules/advection.py`](../../../src/fridom/nonhydro2/modules/advection.py))
  transport whatever is role-tagged ADVECTED with whatever carries a
  Velocity role — nothing in them is nonhydro-specific. Move them to
  the shared model layer (the Coriolis precedent:
  `fr.model.modules.coriolis`), keep `nonhydro2` re-exports so no
  user code moves, gate on bitwise parity. The hydrostatic model
  then gets centered/upwind/WENO — and the FV family — for free
  instead of by cross-package import.
- **HY-D6 — the comparison protocol is convergence + physics, with
  bit-level agreement only in the shared limit.** See §6. The
  common-denominator configuration: C-grid, `(P, P, bounded-z)`,
  uniform flat bottom, centered 2nd-order flux-form advection for
  momentum *and* tracers, quasi-AB2 (`eps=0.1` exposed),
  backward-Euler implicit linear free surface (FFT-solved), explicit
  f-plane Coriolis, linear EOS. This matches Oceananigans'
  `ImplicitFreeSurface` FFT default *exactly in formulation* and
  pyOM's `enable_free_surface` (`eps=1`); the `epsilon=0` rigid lid
  covers the Veros streamfunction physics on a doubly-periodic
  domain.
- **HY-D7 — eigenmodes are built against the full linear physics;
  the implicit variant declares the gap.** The hydrostatic
  eigenmode/transform stack (the "eigenvectors" deliverable of 3.1)
  uses the complete linear operator including the barotropic mode.
  Dispersion oracles run on the explicit variant; the implicit
  variant's `linear_operator_gap` keeps `require_linear_operator`
  consumers honest.

## 2. What exists / what is missing (probed 2026-07-16 on `dev`)

Exists and carries directly:

| Capability | Evidence |
|---|---|
| Depth integral to `ConstantSpace`, cross-shard `collective=True`, `jacobian=` weighting | [`integrate.py:60-`](../../../src/fridom/spatial/operators/integrate.py); `ScalarField.integrate/.mean` |
| Transforms drop `ConstantSpace` factors -> a 2D horizontal spectral solve on `ps` is natural, slab-distributable | [`transform.py:1023`](../../../src/fridom/spatial/operators/transform.py), [`distributed_solve.py`](../../../src/fridom/spatial/operators/distributed_solve.py) |
| Spectral elliptic inversion with null-space gauge (`where_zero`) | [`spectral_solve.py`](../../../src/fridom/spatial/operators/spectral_solve.py) |
| Matrix-free PCG with `project_mean=` (singular periodic Poisson) | [`krylov.py`](../../../src/fridom/spatial/operators/krylov.py) |
| The elliptic-solver template incl. walled Neumann-sibling retag | [`nonhydro2/modules/pressure.py`](../../../src/fridom/nonhydro2/modules/pressure.py) |
| The barotropic *equations* + energy-conserving Sadourny/Coriolis discretizations | [`shallowwater2/modules/core.py`](../../../src/fridom/shallowwater2/modules/core.py), [`sadourny.py`](../../../src/fridom/shallowwater2/modules/sadourny.py) |
| Quasi-AB2 with pyOM's damper (`eps` is the order-2-only parameter) | [`adam_bashforth.py`](../../../src/fridom/model/time_steppers/adam_bashforth.py) |
| IMEX driver (CNAB2/SBDF2) + mergeable `VerticalDiffusion` tridiagonal | [`imex.py`](../../../src/fridom/model/time_steppers/imex.py), [`implicit.py`](../../../src/fridom/model/implicit.py) |
| S1' DIAGNOSE slots scheduled for `p_hyd = ∫b dz` and `w = −∫∇·u dz` | spec [`03_time_stepping.md`](../../specs/model/03_time_stepping.md) §5.2 |
| Diagnosed-`w`-with-Velocity-role validated for (amendment V-H2) | [`declarations.py:526-530`](../../../src/fridom/model/declarations.py) |
| Constant-along-z 2D prognostics (`Profile`, `Dof.CONSTANT`) | [`space_patterns.py:418-`](../../../src/fridom/spatial/space_patterns.py); spec §5.4 ("eta, U, V — 2D, constant-along-z") |
| Metric-aware `grad/div/laplacian`, J-weighted SPD CG mapped solve | CS-D2, [`mapped_pressure.py`](../../../src/fridom/nonhydro2/modules/mapped_pressure.py) |

Missing (the actual new work):

- **No cumulative (partial) integral operator.** `Integral` only
  reduces an axis; `w(z) = −∫_{−H}^{z} ∇_h·u_h dz'` and
  `p_hyd(z) = −∫_z^0 b dz'` need a staggered running sum
  (centers -> faces and back), z-local layout — the same
  local-along-the-solve-axis contract `VerticalDiffusion` already
  declares. The only cumsums in the tree are Chebyshev-internal
  ([`spectral.py:486-505`](../../../src/fridom/spatial/operators/spectral.py)).
- **No free-surface / barotropic module of any kind** (shallowwater2
  steps its gravity waves fully explicitly; nonhydro2 has no 2D
  mode). All three HY-D3 variants are new.
- **`SpectralDiagonal` is designed-for, unbuilt** — under HY-D4 it is
  *not needed* (the Helmholtz inversion lives in `SpectralSolve`
  inside the stage, not in the stepper). Note kept for the record.
- **Latent IMEX footgun** (found while probing HY-D4): a custom
  implicit operator sharing a field with a mergeable family passes
  assembly and silently clobbers in the solve loop. Not exercised by
  any shipped model; a taught assembly error is cheap insurance
  (rides H3).
- Biased/WENO advection rejects mapped meshes at bind — pre-existing
  roadmap line ("High-order mapped stencils"), not this plan's
  problem; centered advection is mapped-capable today.

## 3. The model

Package layout mirrors `nonhydro2`: `model.py` (thin factory),
`state.py`, `params.py`, `modules/{core, free_surface,
stratification}.py`, `initial_conditions.py`, `diagnostics.py`,
`energy.py`, `eigenmodes.py`, `transforms.py`. Advection and
Coriolis come from the shared `fr.model.modules` (HY-D5).

Equations (nondimensional, `Ro = scaling.rossby`, `csqr = g*H`):

```
du/dt = -Ro (v·∇)u + f v − ∂x(p_hyd + ps)     [+ vertical mixing]
dv/dt = -Ro (v·∇)v − f u − ∂y(p_hyd + ps)     [+ vertical mixing]
∂z p_hyd = b,  p_hyd|_{z=0} = 0     ->  DIAGNOSE (top-down cumsum)
∂z w = −∇h·uh,  w|_{z=−H} = 0       ->  DIAGNOSE (bottom-up cumsum)
db/dt = -Ro v·∇b − N² w
d ps/dt = −csqr ∇h·ū,  ū = depth-mean(u, v)   [variant-owned]
```

`HydrostaticCore` declares `u, v` (`.velocity`, ADVECTED), `w`
(Staggered z, DIAGNOSTIC lifecycle + `Velocity("z")` role, *not*
ADVECTED — the V-H2 case), `p_hyd` (Collocated, DIAGNOSTIC), owns
the two DIAGNOSE stages and the linear pressure-gradient term, and
publishes `hydro.csqr` and `scaling.rossby`. `ConstantStratification`
declares `b` and contributes the `−N² w` restoring (nonhydro2
pattern minus the `buoyancy_force` term, which hydrostatic balance
replaces). S1' placement guarantees `w`/`p_hyd` are fresh before any
term (and after restart/`set_state`) — the spec ordered the schedule
around exactly this.

The free-surface variants (HY-D3):

- **Explicit**: one linear term, `d ps/dt = −csqr ∇h·ū` — the
  shallowwater2 gravity term acting on the depth mean. CFL-limited
  by `sqrt(csqr)`; exists as oracle and for wave-resolving runs.
- **Implicit** (the workhorse): CONSTRAINT stage after the advance.
  With predictor depth-mean `ū*` and `dt' = ctx.stage_dt`:

  ```
  (ε − dt'² ∇h·(csqr ∇h)) ps^{n+1} = ε ps^n − dt' csqr ∇h·ū*
  u^{n+1}(z) = u*(z) − dt' ∇h ps^{n+1}        (z-uniform correction)
  ```

  `ε=1` is the linear implicit free surface (backward Euler — the
  pyOM/Oceananigans choice; symbol `1 + csqr dt'² k²` on the
  periodic flat grid, non-singular, `SpectralSolve` fast path,
  slab-distributable). `ε=0` is the rigid lid: the singular Poisson
  with the `where_zero` mean gauge, enforcing a divergence-free
  depth mean — pyOM's option 1, "identical to MITgcm". Variable
  `csqr(x, y)` (topography) or mapped grids flip the same solve to
  the CS-D2 route: J-weighted SPD operator, `ConjugateGradient`,
  flat spectral inverse as preconditioner. Under IMEX the stage
  runs after the mixing solve (S3 -> S4), reproducing pyOM's
  splitting; under RK it runs per stage like the nonhydro
  projection.
- **Split-explicit** (build after implicit): the §5.4 frozen design
  verbatim — module-owned `ADVANCE({eta, U, V})`, `lax.scan` over N
  static substeps, forward-backward substep, Shchepetkin–McWilliams
  (2005) averaging kernel (`p=2, q=4, r=0.18927` — Oceananigans'
  defaults, for comparability), increment-form slow forcing
  `G = ∫(X* − Xⁿ)dz / dt` (V-H4), `CONSTRAINT({u, v})` depth-mean
  correction. `U, V` carry no Velocity role.

Factory sketch:

```python
hy.Model(grid=grid, dt=dt,
         free_surface=hy.ImplicitFreeSurface(),   # | Explicit... |
                                                  # SplitExplicit...(substeps=N)
         csqr=..., coriolis=hy.FPlaneCoriolis(f0=...),
         stratification=hy.ConstantStratification(n2=...),
         advection=fr.model.modules.CenteredAdvection(),
         time_stepper=fr.model.AdamBashforth(dt, order=2, eps=0.1))
```

Generalization comes from the substrate, by construction: mapped /
stretched vertical via `CumulativeIntegral(jacobian=...)` +
metric-aware operators + the CG solve route; time-dependent
mappings via `MovingGeometry` (CS-D4) — the V-H5 amendment already
names "z*-geometry following eta" as the designed follow-on;
higher-order advection via HY-D5. None of these are iteration-1
gates (§7).

## 4. Stages

| Stage | Work | Effort | Gate |
|---|---|---|---|
| **H0** | Rehome advection to `fr.model.modules.advection` (+ FV/WENO/walls variants), `nonhydro2` re-exports kept | S (1–2 d) | bitwise parity on the full `test_advection*` shards + nonhydro smoke; ruff |
| **H1** | `CumulativeIntegral` operator: nodal + FV rows, centers<->faces staggering, top-down/bottom-up, `jacobian=` weighting, z-local layout negotiation | M (2–4 d) | manufactured exactness both directions; discrete telescoping (sum of increments == `Integral`); forced-4 multi-device |
| **H2** | Package skeleton + kinematics + explicit free surface: state, `HydrostaticCore` (DIAGNOSE `w`/`p_hyd`, pressure term), stratification, params, factory, `ExplicitFreeSurface`, shared advection/Coriolis wired | M (4–6 d) | `w`/`p_hyd` manufactured solutions; barotropic Poincaré + hydrostatic internal-wave dispersion vs analytic; geostrophic steady state to machine precision; energy conservation (inviscid, centered); `test_init` |
| **H3** | `ImplicitFreeSurface`: CONSTRAINT solve (SpectralSolve path + `epsilon` knob incl. rigid-lid gauge), `linear_operator_gap`, taught IMEX same-field assembly error | M (4–6 d) | small-`dt` convergence to the H2 oracle; stable at `sqrt(csqr)`-CFL >> 1; implicit dispersion matches the backward-Euler factor analytically; rigid-lid depth-mean divergence at machine zero |
| **H4** | Vertical mixing wiring (shared `VerticalDiffusion` on `u, v, b` under `CNAB2`/`SBDF2`) + eigenmodes / energy weights / transforms (HY-D7) | M (3–5 d) | analytic column-decay; biorthogonality + projection round-trip (nonhydro2 test pattern); `require_linear_operator` honesty |
| **H5** | Comparison protocol: the HY-D6 config as a preset + example script; extend the out-of-tree `benchmarks/comparison` harness (Oceananigans local; Veros; pyOM3 source: `github.com/ceden/pyOM3`) | M (1 wk) | matched-protocol physics: geostrophic adjustment, dispersion, Eady-type growth rates, spin-down energy budgets; grid-refinement convergence |
| **H6** | `SplitExplicitFreeSurface` per the frozen §5.4 design | M/L (1–2 wk) | vs the implicit reference at matched physics; volume/tracer conservation through the filter; own-AUX restart fingerprint; forced-4 multi-device |

Every stage lands on its own `<type>/<topic>` branch with mirrored
tests (95% branch coverage) and ruff clean, per AGENTS.md. H0 and H1
are independent of each other; H2 needs both; H3–H6 are sequential
on H2. H5 can start (protocol + harness) once H3 exists.

## 5. Risks / open verification items

- **`w` advecting-velocity plumbing (H2, first item):** the shared
  advection modules must pick up the velocity trio when one member
  is DIAGNOSTIC-lifecycle. V-H2 says the declaration layer supports
  it; the advection module's role query is the part to verify.
- **`ps` in the carry:** a `Profile("x", "y")` PROGNOSTIC rides
  every stepper's ring buffers; ConstantSpace factors in stepper
  state are designed-for (§5.4) but nothing exercises them yet.
- **Cumulative integral under decomposition:** z-local layout
  negotiation cost on 4 GPUs — measure in H1, `VerticalDiffusion`
  is the precedent that it is affordable.
- **Eigenmode completeness under HY-D4/HY-D7:** the implicit
  variant's declared-linear gap must not silently degrade the
  projections — the gap declaration plus explicit-variant oracles
  is the mitigation; revisit if a consumer needs implicit-variant
  eigenmodes directly.
- **pyOM3 access:** resolved 2026-07-19 — owner supplied the source:
  `https://github.com/ceden/pyOM3.git` (verified reachable). The
  pyOM2 doc remains the authoritative discretization description;
  pick the reference config from the repo when the H5 leg runs.

## 6. External-model numerics (survey summary, 2026-07-16)

| | pyOM2/3 | Veros | Oceananigans `HydrostaticFreeSurfaceModel` |
|---|---|---|---|
| Grid | C-grid, z-level | C-grid (pyOM2.1 port) | C-grid finite volume, `RectilinearGrid` |
| Prognostic | u, v, T, S | u, v, T, S | u, v, eta, tracers |
| w, p_hyd | diagnosed | diagnosed | diagnosed |
| Barotropic mode | rigid-lid `ps` Poisson (CG, "identical to MITgcm"); implicit free surface (`enable_free_surface`, Helmholtz, `eps`); streamfunction+islands | rigid-lid streamfunction; pluggable Poisson backends (scipy / pyAMG / PETSc) | `ImplicitFreeSurface` (**FFT default** on regular grids; PCG otherwise); `SplitExplicitFreeSurface` (SM2005 filter, forward-backward); `ExplicitFreeSurface` |
| Time stepping | quasi-AB2 (`eps=0.1`) + backward-Euler vertical mixing | same | `QuasiAdamsBashforth2` default; `SplitRungeKutta3` |
| Momentum advection | centered-2 flux form, always | same | configurable; default vector-invariant — **force `Centered(order=2)` flux form for comparison** |
| Tracer advection | centered-2 (+ superbee option) | same | configurable; default centered-2 |
| EOS | TEOS-10 or Vallis model EOS (linear reduction) | same | linear (`alpha, beta`) / `BuoyancyTracer` / TEOS-10 |

Sources: Eden, *pyOM2.0 documentation* (2014) and
`github.com/ceden/pyOM2`; Häfner et al. 2018 (GMD 11, 3299) and
`veros.readthedocs.io`; Oceananigans docs
(`clima.github.io/OceananigansDocumentation`) and Silvestri et al.
(arXiv:2502.14148). Bit-level agreement is only meaningful in the
shared limit (centered-2, quasi-AB2, implicit free surface, linear
EOS); everything else compares via convergence and physical
diagnostics (HY-D6).

## 7. Out of scope (designed-for, not precluded)

T/S with a nonlinear EOS (buoyancy tracer first); topography /
immersed boundaries (variable-`csqr` solve route is specified, not
built in it-1); z* / ALE moving vertical coordinate (V-H5 names it;
first consumer of `MovingGeometry` + free surface); the spherical
chart (3.7 names the hydrostatic model as its likely first
consumer); IMEX-RK x split-explicit (assembly error per spec §5.4);
a Veros-style superbee limiter in the shared advection family;
`SpectralDiagonal` (unneeded under HY-D4).

## 8. Implementation record

**Update 2026-07-17: every stage has shipped and merged** — H0, H1,
H2 (+H2b), H3, H4, H5 (+H5b), H6; entries below in stage order. What
remains open (tracked in the roadmap 3.1 entry): the cross-model
*execution* legs of §6 (the out-of-tree `benchmarks/comparison`
harness is not on this machine; pyOM3 source access pending owner),
and the owner review of `examples/hydrostatic/comparison_baseline.py`
(local branch `docs/hydrostatic-example`, never merged, per the
AGENTS.md docs workflow). The §7 designed-fors are untouched **except
terrain, now built** (see below).

### Terrain-following (sigma-coordinate) core (branch `feat/hydrostatic-terrain`)

Builds the ratified terrain items H0/H1/H2/H4 + the explicit/split
depth fix of `stretched_terrain_combined.md` §6 (§7 addendum ruling 2).
New seam `hydrostatic/modules/terrain.py` (`discover_column`): the
single-base sigma column `(mapped, base)` on the vertical mesh axis, or
`None` off a mapped grid (flat / stretched-only — byte-identical).

- **H1 (`p_hyd`)** — the top-down center `CumulativeIntegral` carries
  `jacobian=(mapped,)` (the wired seam), i.e. `-∫ b J dz`. Converges at
  second order on uniform- **and** stretched-sigma columns.
- **H2a (`w`)** — the diagnosed `w` is the **contravariant vertical
  volume flux `Jω`** (not the Cartesian `w_phys`), from the flux-form
  horizontal divergence `-∫[∂ₓ(Ju)+∂_y(Jv)]dz` (`J` on the u/v faces).
  This keeps both flat invariants **exactly**: the machine-exact FTC
  `∂_z(Jω) == -[∂ₓ(Ju)+∂_y(Jv)]` and the **exact** bottom seed `Jω = 0`
  (zero normal flow on the sigma bottom — `w_phys` is nonzero over a
  slope; `Jω=0` is the natural prognostic-free choice). Flat `J=1`,
  `Z=0` collapses byte-for-byte to the Cartesian form.
- **H2b (baroclinic pressure gradient)** — the horizontal force is the
  gradient at constant physical height `-∂ₓ p|_zp = -(∂ₓ p|_z -
  (Zₓ/J)∂_z p)`, assembled by hand (`HydrostaticCore._slope_gradient`,
  slope coefficient on the *column-face* space — mirrors the mapped
  advection's nodal divergence) **not** the `physical_diff` verb, whose
  composite reciprocal-Jacobian seals a never-valid-padding singularity
  that NaNs the reverse pass (differentiability policy). The rest state
  over a seamount converges at second order (linear and nonlinear); the
  slope coefficient's `Z/J` division carries a double-`where` guard.
- **H4 (energy)** — with `_slope_gradient` on the column-face space the
  baroclinic pressure gradient is the **exact discrete adjoint** of the
  flux-form continuity that diagnoses `w`, so the KE↔PE conversion +
  surface cancellation conserve energy to roundoff on a **resolved**
  state (grid-scale noise breaks the interpolation-transpose pairing —
  smooth is the terrain analog of the flat random-field test). The
  **barotropic** pair conserves energy to roundoff (any state) under
  the physical-volume metric.
- **depth fix** — the physical column depth `H(x,y)=∫J dz` (in-trace,
  double-`where` guarded reciprocal) replaces the computational extent
  in `_depth_mean_div` (flux-form transport divergence `∫[∂ₓ(Ju)+
  ∂_y(Jv)]dz / H`) and the ps energy weight.

**Two sigma tensions, recorded (not bugs — inherent to a C-grid sigma
model with a single `c²`).** (1) *Pressure-gradient vs energy*: the
slope-corrected gradient cannot be simultaneously machine-exact for
rest-state balance **and** machine-energy-conserving; `_slope_gradient`
takes exact energy (adjoint) + second-order rest state (`physical_diff`
takes the reverse and is grad-unsafe besides). (2) *Barotropic volume
vs energy*: with constant `c²` over a variable physical depth, the
physical-depth `H(x,y)` depth-mean conserves **energy** to roundoff but
drifts `∫ps` by O(slope); the reference-depth form would conserve
volume, not energy. Per the §6 depth-fix instruction (use `H(x,y)`) the
build takes energy; exact volume needs the variable-`c²(x,y)` solve
(deferred with the implicit variant).

**Deferred behind taught errors (H0 gates).** `ImplicitFreeSurface` on
a chart grid (variable-coefficient barotropic Helmholtz — the
`mapped_pressure.py` analogue, unbuilt); `SplitExplicitFreeSurface` on
a chart grid (transport-depth-consistent subcycle, unbuilt); an
embedding `chart=` (spherical / curvilinear); a vertical axis that is
not the base of a single-base analytic column; immersed **and** terrain
together. [Update: the `ImplicitFreeSurface` chart taught error was
retired 2026-07-18 (multigrid_generalization phase B, GM-D1 volume-exact
solve) and the `SplitExplicitFreeSurface` one 2026-07-19 (the same
GM-D1 option-1 volume-exact terrain subcycle, `H_a = ∫J dz`, constant
`c²/H_ref` gravity, no `1/H` division); immersed **and** terrain together
is now supported for the explicit / implicit variants (M5) and stays a
**narrowed** taught error for the split-explicit variant only. See
done.md.] **Known gaps (reported, not gated):** nonlinear advection on
a terrain grid is forward-finite but **not** reverse-safe (the shared
nodal mapped divergence divides `Z/J` unguarded — a shared-advection
fix). [Update 2026-07-18: sealed via the shared `_safe_ratio`
double-where (grad=FD at 9e-10); the hydrostatic terrain autodiff
gate now runs `advection=True`. See done.md.] And
`EnergyMetric`/`eigenmodes` on a terrain grid use the plain
(extent) ps weight and so are physically inconsistent (energy
diagnostics off a chart, unfixed here).

Tests: `tests/hydrostatic/test_terrain.py`, `test_core_terrain.py`,
`test_free_surface_terrain.py` (the H1/H2/H4 + depth + rest-state +
taught-error + autodiff gates); the full flat `tests/hydrostatic`
suite unchanged.

### H0 — advection rehomed (2026-07-16)

`nonhydro2/modules/advection.py` -> `fr.model.modules.advection`
(pure move; one docstring line changed), re-exported into `nonhydro2`
through the shared-modules origin (the Coriolis pattern); the six
`test_advection*` shards moved to `tests/model/modules/`. Gates: 494
shard tests + nonhydro smoke + ruff, `git diff --find-renames` clean.

### H1 — `CumulativeIntegral` (2026-07-16)

`spatial/operators/cumulative.py`: the staggered running integral on
one bounded axis (`direction="up"/"down"`, `target="face"/"center"`,
`jacobian=`), verb `cumint`, nodal + FV rows. The face form carries
the machine-exact discrete fundamental theorem and telescopes exactly
to `Integral`; the center form is the pyOM half-cell midpoint (its
exact identity: `diff == interpolated integrand`). Periodic axes are
a taught error; a decomposed axis reshards through the negotiated
layout (bitwise vs single-device). Gates: 39+2 single-device, 41
forced-4, 100% coverage.

### H2 — package + kinematics + explicit free surface (2026-07-16)

Shipped `fridom.hydrostatic` (`import fridom.hydrostatic as hy`):
`params`, `state` (`u,v,w,b,ps` accessors + `rel_vort_z` /
`hor_divergence`), `energy`, `diagnostics` (`ekin`/`epot`),
`initial_conditions` (`single_wave`, `jet`), the `Model` factory,
lazypimp `__init__`s, root export; and modules `HydrostaticCore`
(declares `u,v,w,p_hyd`, owns the two DIAGNOSE stages + the linear
pressure-gradient term, publishes `hydrostatic.csqr` / `scaling.rossby`),
`ConstantStratification` (`b` + the single `-N^2 w` restoring), and
`ExplicitFreeSurface` (`ps` on `Profile("x","y")` + `-c^2 div(u_bar)`).

**Discrete DIAGNOSE choices.** `w = -CumulativeIntegral("up","face")`
of `(d_x u + d_y v)` — the bottom-up FACE form, so
`d_z w == -(d_x u + d_y v)` machine-exactly (measured FT residual
1e-14). `p_hyd = -CumulativeIntegral("down","center")` of `b` — the
top-down half-cell CENTER form; the top-cell value is exactly
`-(dz/2) b_top`, which is what cancels the surface boundary term and
makes the KE<->PE conversion exactly energy-conserving.

**Verified gates.** Linear energy skew `<X, M dX/dt>` over the full
system (baroclinic + barotropic + f-plane Coriolis) = 4e-16 with
`M=diag(1,1,1/N^2,1/c^2)` and **ps integrated over the 3D volume**
(the depth factor `H` that makes `-grad ps` and `-c^2 div(u_bar)` an
exact adjoint pair). Manufactured `w`/`p_hyd` exact. Geostrophic
null-eigenvector state steady to 1e-15/step. Barotropic Poincaré
dispersion vs `omega^2=f^2+c^2 k_disc^2` (discrete C-grid symbol) to
8e-5 via the reduced linear operator; hydrostatic internal-wave
`m_disc^2 = N^2 kh_disc^2/(omega^2-f^2)` identical across two `kh`
(the discrete `omega^2=f^2+N^2 kh^2/m^2`) and within 0.6% of the
continuous `(pi/H)^2`.

**Deviations from §3 (two).**
1. **`w` lives on `Outer(z)` (a `SpaceRule`), not `Staggered("z")`.**
   `Staggered` resolves to `Inner` (interior faces) on a bounded axis,
   but the face-form running integral lands on `Outer` (both boundary
   faces), and the surface DOF `w(0)` — the barotropic column
   divergence, non-zero under a free surface — is load-bearing: with
   `w` on `Inner` the machine-exact FT and the machine-exact linear
   energy conservation both break (measured skew 0.17 vs 6e-17). `w`
   keeps `Lifecycle.DIAGNOSTIC` + `Velocity("z")` (V-H2) but is
   declared through the `SpaceRule` escape hatch (no `Dof` tag
   resolves to `Outer`).
2. **Nonlinear advection is not yet wired; the factory default is a
   linear model (`advection=False`), and a truthy `advection` raises a
   taught error.** The shared flux-form advection transports a
   cell-centred tracer through the *interior* vertical faces (`Inner`)
   and interpolates the advecting velocity there (`w.to(Inner)`); with
   `w` on `Outer` there is no registered vertical interpolation between
   `Outer` and `Inner` (only `Center->Inner` and `Outer->Center`
   exist), so the vertical leg cannot consume the diagnosed `w`
   (`table.velocity()` DOES pick `w` up — the V-H2 role query works;
   the gap is purely spatial-layer interpolation). Unblocking needs a
   vertical `Center<->Outer` / `Outer->Inner` interpolation (or a
   shared-advection enhancement restricting an `Outer` velocity to the
   interior flux faces) — a small follow-up. All H2 dispersion /
   geostrophic / energy gates are on the linear model and are
   unaffected.

**Gates (all green).** `tests/hydrostatic/` = **73 passed** (serial);
`ruff check src tests` clean; **100% coverage** of the hydrostatic
source (231 stmts / 22 branches); the `tests/nonhydro/test_linear_model.py`
framework smoke unaffected by the root `__init__` export.

**Other notes.** The barotropic divergence uses `Integral()["z"]`
applied to the *collocated* `d_x u + d_y v` (the halo tracer has no
`.mean`/`.integrate` sugar, and `Integral` on the staggered `u` hit a
measure-halo mismatch; reducing the collocated divergence — which the
z-reduction and horizontal derivative commute through — sidesteps
both and keeps the adjoint pairing exact). `ExplicitFreeSurface.bind`
reads the depth `H` from the mesh **extent** (not a materialized
`grid.measure(...).data`), because the latter re-fetches a
halo-shaped array against the frozen decomposition when a second model
is assembled on the same grid — the shared-grid reuse the canonical D4
preset test (identical full-`_carry` treedef) depends on; extent-`H`
is telescoping-exact, so energy conservation stays at `4e-16`. Package
param name is `hydrostatic.csqr` (matching the `shallowwater.csqr` /
`nonhydro.dsqr` package-namespace convention).

### H2b — nonlinear advection wired (2026-07-17)

Resolves H2 deviation #2. New spatial row `fr.operators.Restriction`
(kind `"restrict"`, `restrict.py`): the exact `Outer -> Inner`
face-set restriction — `Outer ⊃ Inner`, so it drops the two boundary
faces as a size-1 identity staggering (`Inner[m] == Outer[m+1]`),
halo-0, metric-free (exact on stretched meshes), complex preserved.
Seeded on every bounded factor's `Outer` (periodic has none; a
`ChebyshevMesh` `Outer` un-seeds — no `Inner`). `ScalarField.to` routes
`Outer -> Inner` to the new kind (`_conversion_kind`), leaving the
distinct `Outer -> Center` interpolate (`w.to(b)`, stratification)
untouched. The shared advection needed one additive dispatch-level
touch (`advection.py`, fires only for an `Outer` velocity, so existing
models are bitwise unchanged): `_flux_space` tags the tracer flux
homogeneous-Dirichlet `Inner` on the restriction axis (the zero-wall-
flux claim its divergence closes on, the wall-normal-velocity
substitution's `Outer` twin), and the biased `_velocity_face` uses the
exact `.to` restriction rather than the order-coupled interpolation on
that axis. `hy.Model` default is now `CenteredAdvection()`;
`advection=False` keeps the linear model; `UpwindAdvection` /
`WENOAdvection` accepted. **Closure & conservation.** Dropping `w(0)`
is the fixed-domain linear-free-surface treatment — **zero advective
flux through the boundary faces**: tracer mass conserved to roundoff
(measured `~1e-14`, all three schemes). The advection is energy-
orthogonal in the M metric (`<q, M A(q)>` at machine zero, vertical leg
active) so the semi-discrete H2 energy skew is unchanged by advection;
the one exception is exactly localized — `A(b=const)` is machine-zero in
every interior cell and non-zero only in the surface cell, the dropped-
`w(0)` term the `ps` equation (not advection) carries. **Note:** a
time-integrated inviscid run on a coarse grid is nonlinearly unstable
(a resolution property of centered advection, not the scheme), so the
energy gate is the semi-discrete skew, not a time-integrated dt-slope.
**Gates.** `tests/hydrostatic/` 91 passed; new
`tests/hydrostatic/test_advection.py`, `tests/spatial/operators/
test_restrict.py`, extended `test_grid.py` / `test_scalar_field.py`;
advection regression `test_advection*.py` 200 passed (unchanged);
`ruff` clean.

**Amended by H7 (2026-07-17):** the surface-cell constancy exception
recorded above turned out to destabilize the implicit free surface
under advection (and to corrupt amplitudes even where it stays
finite). The constancy-preserving surface closure is now the
**default**; the fixed-domain dropped-`w(0)` closure survives as the
explicit `surface_flux=False` opt-out. See H7.
### H3 — implicit free surface + taught IMEX assembly error (2026-07-17)

**Coupling-ownership refactor (load-bearing).** Each free-surface
variant now owns BOTH sides of its barotropic coupling: the core's
linear term reads `p_hyd` only, and `ExplicitFreeSurface` gained the
`-\nabla_h ps` momentum term (the adjoint of its `-c^2\nabla_h·ū`
term; `ps.diff().to(u).retag(u)` is bitwise-identical to the old
`ps.to(p_hyd).diff()`). The H2 energy skew stays `<1e-12` (structural,
independent of grouping) and all H2 gates remain green (`tests/hydrostatic`
= 103 passed).

**`ImplicitFreeSurface(epsilon=1.0)`** (`modules/free_surface.py`): a
CONSTRAINT stage (write set `{u, v, ps}`) solving
`(ε − dt'² Div∘Diag(c^2)∘Grad) ps = ε ps_old − dt' c^2 ∇h·ū*` then
`u ← u − dt' ∇h ps`. The operator is the honest C-grid `Div∘Grad`
(built via `Gradient`/`Divergence`/`Diag`, `-dt'^2 c^2` folded into the
diagonal so its own eigenvalue symbol is `dt'^2 c^2 k_disc^2`), plus the
static `ε` identity, inverted by `SpectralSolve` on the 2D `Profile`
space (the transform drops the ConstantSpace-z factor — a natural 2D
solve, verified). `ε>0`: `ps` PROGNOSTIC, symbol `1+c^2 dt'^2 k_disc^2`
non-singular; `ε=0`: `ps` DIAGNOSTIC (the rigid lid), pure Poisson with
the `where_zero` mean gauge. `epsilon` is a **static** constructor arg
(it selects the lifecycle) — not sweepable. Declares `linear_operator_gap`
(HY-D7) and `extra_halo` (exempts the spectral solve, V-N2 precedent).

**Model-layer additions (two, additive).** (1) The D1.4 coverage lint
now counts a CONSTRAINT-stage **`advances=` claim** as covering a
PROGNOSTIC field — `ps` is advanced only by the projection, genuinely
integrated forward, and the stage claims exactly it. *(Corrected
2026-07-17: the initial H3 form credited a CONSTRAINT stage's whole
write set, which silenced the lint for the nonhydro2 velocity
projection and broke three of its tests — full-suite catch; the claim
form restores them and matches the spec §5.4 amendment pattern.)*
(2) `composer._implicit_groups`: a non-mergeable custom implicit operator
overlapping a mergeable family's fields is a taught `ImplicitCollisionError`
(the §2 latent footgun — the driver's independent solves clobber), both
declaration orders, naming both parties. No shipped model exercises the
overlap, so behavior is unchanged elsewhere.

**Gates (all green, CPU).** Backward-Euler factor `|G|=1/√(1+ω_disc²dt'²)`
and phase `arctan(ω_disc dt')` (ω_disc from the discrete symbol) matched
to 5e-16 via the constraint's restricted eigenmap. Small-dt convergence
implicit-vs-explicit-oracle first-order (slopes 0.91→0.98). Rigid lid:
post-constraint depth-mean divergence 4.7e-15 (rel to |u|), reducing a
pre-divergence of 62 to machine zero; `ps` matches an independent numpy
C-grid FFT Poisson solve to 4e-16. Stable at `√(c^2)dt/dx=50`, 200 steps
(energy bounded/decaying, finite). Geostrophic null-eigenvector steady to
1e-15 under both variants. `require_linear_operator` refuses the implicit
(and rigid-lid) config, passes the explicit. Stepper smokes: `AB2(eps=0.1)`
(pyOM), `AB3`, `LowStorageRK3`; and CNAB2 + an in-test `VerticalDiffusion`
consumer (the 2.5 reference consumer is test-only) + `ImplicitFreeSurface`
(mixing solve then surface constraint) — treedef stable. `ruff` clean;
100% coverage of the hydrostatic source, new composer branches covered.

### H4 — vertical mixing + eigenmodes / transforms (HY-D7) (2026-07-17)

**(a) Implicit vertical mixing.** New shared model-layer module
`fr.closures.VerticalMixing(kv=..., kb=..., treatment=fr.model.IMPLICIT)`
(re-exported `hy.modules.VerticalMixing`): one mergeable
`VerticalDiffusion(axis="z")` term per leg (velocity viscosity `kv` /
buoyancy diffusivity `kb`, PROGNOSTIC role targets resolved at bind;
the diagnosed `w` is not a target), the two legs merging into one
tridiagonal solve set. Treatment is author-declared (§5.1); `EXPLICIT`
is the write-once `op.apply` path and declares an (empty) `extra_halo`
so the raw-`.data` column solve is halo-trace exempt (V-N2), validated
in the real-field dry run. Gates: CNAB2/SBDF2 1D column decay vs the
*discrete* eigen-decay 1.2e-5 / 7.3e-5 (the tridiagonal is exact for the
discrete operator — residual is pure time-discretization); stiff
`kappa dt/dz^2 ~ 640` bounded/decaying; EXPLICIT tendency `== L@b` to
3.5e-16; CNAB2 + `ImplicitFreeSurface` composes (mixing S3, surface S4,
treedef stable); friction+mixing legs merge to one implicit operator on
`(u,v,b)`.

**(b) Eigenmodes / energy / transforms.** Design decision (documented
in `eigenmodes.py`): the free surface couples the depth-mean divergence
to `ps` through a rank-1 barotropic term, and although the staggered
cumint pair is an exact transpose pair (`C_p = M^T`, so the vertical
operator `V = N^2 M^T M + c^2 Pi` is symmetric and the modes separate
cleanly with real dispersion), the constant vector is NOT an eigenvector
of `M^T M` — the barotropic mode is z-constant only to `O(N^2/c^2)`, so
there is no exact analytic z-constant eigenvector (the exactly-z-constant
Poincaré triplet with `omega^2=f^2+c^2 k_disc^2` is the barotropic
*restriction*, the H2 oracle to ~1e-3). So `hy.eigenmodes` /
`hy.transforms` reuse the shared dense-column engine
(`fr.model.eigen_channel.channel_eigenpairs`, bounded axis `z`): the
**exact numeric eigenbasis** of the assembled linear operator under the
hydrostatic energy metric `diag(1,1,1/N^2,1/c^2)` (`ps` depth-weighted
`H/c^2` — the `EnergyMetric.from_model` hydrostatic branch + the
`_bounded_measure` ConstantSpace fix, both model-layer). Because the
engine `linearize`s first, the implicit/rigid-lid variants (which
declare `linear_operator_gap`) are refused by `require_linear_operator`.
`HydrostaticEigenmodes` labels the `3nz+1` columns/plane into six
families (barotropic/baroclinic × geostrophic/wave±): the geostrophic
zero space split by a depth-mean/`ps` barotropic-overlap rotation (one
barotropic column/plane), the waves by surface-pressure energy.
`hy.transforms.{Vortical,Wave,Barotropic,Baroclinic}Projection`
(`ProjectionFactory`, dual-source). Gates (all machine precision):
biorthogonality (hermiticity 1.7e-16, M-orthonormality 3e-15); round-trip
`Vortical+Wave == Barotropic+Baroclinic == identity` 1.5e-15; idempotency
1.3e-15; orthogonality 8.7e-16; baroclinic `m_disc^2` identical across
`kx` 5e-15; barotropic dispersion vs `f^2+c^2 k_disc^2` 1.6e-3 (the
z-constant tolerance); implicit/rigid-lid refused. Model-layer touches
(`model:`): `EnergyMetric.from_model` hydrostatic branch,
`eigen_channel._bounded_measure`, `closures.VerticalMixing` — sibling
channel-eigenmode suites unaffected (118 passed).
### H5 — comparison preset + physics-validation suite (2026-07-17)

**Delivered (in-tree).** The HY-D6 common-denominator config as a
documented factory `hy.comparison_model(grid, dt, ...)`
(`comparison.py`, own module; lazypimp-exported): pins
`AdamBashforth(order=2, eps=0.1)` (pyOM quasi-AB2),
`ImplicitFreeSurface(epsilon=1)` (backward-Euler linear free surface),
`CenteredAdvection()` (centered-2 flux form on momentum *and* the `b`
tracer), `FPlaneCoriolis`, `ConstantStratification`, explicit `csqr`.
Only the physical parameters (`csqr`, `coriolis_f0`, `n2`,
`rossby_number`) and the two comparison knobs (`epsilon`, `eps`) are
exposed; the docstring carries the reference-model map (pyOM `AB_eps` /
`enable_free_surface`; Oceananigans `QuasiAdamsBashforth2` /
`ImplicitFreeSurface`-FFT / `Centered(order=2)`; the Veros rigid-lid
axis via `epsilon=0`). Physics-validation suite
`tests/hydrostatic/test_comparison.py`; a runnable baseline
`examples/hydrostatic/comparison_baseline.py` (the geostrophic-
adjustment problem; owner-reviewed on `docs/hydrostatic-example`).

**Geostrophic (Rossby) adjustment (validated).** A released single-mode
`ps` step, `n2=0` (barotropic subsystem, `epsilon=1`): the implicit
free surface damps the inertia-gravity waves while the zero-frequency
geostrophic mode is steady, so the state settles to the geostrophically
adjusted mode. The retained pressure amplitude AND energy fraction both
equal the **discrete-deformation-radius** prediction

    ps_g/ps_i = E_g/E_i = gamma^2 / (gamma^2 + kd^2 Ld^2),
    Ld^2 = c^2/f^2,  gamma = prod_a cos(k_a dx_a/2),

where `gamma` is the energy-conserving C-grid Coriolis interpolation
symbol (`v.to(u)`) — i.e. the discrete deformation radius is
`Ld_disc = Ld/gamma`. Measured on the preset (nonlinear advection at
amplitude 1e-3, negligible): retained = 0.15962 (nx=16) / 0.16626
(nx=32), matching the `gamma` formula to **3e-5** (energy and amplitude
alike); the balanced field is the input mode (correlation > 0.9999) and
steady (drift < 1e-4/50 steps). The discrete fraction converges to the
continuous Rossby fraction `1/(1+k^2 Ld^2)` at second order (error
8.9e-3 -> 2.2e-3 across nx=16->32; the nx=64 slope check rides behind
`FRIDOM_TEST_COMPARISON_SLOW`).

**Wave-packet dispersion (validated).** A localized internal-wave packet
(gravest baroclinic vertical mode; `epsilon=0` rigid lid so there is no
barotropic branch to contaminate it) launched from the linear
operator's exact discrete eigenvector propagates at the discrete
**group** velocity `c_g = (omega^2-f^2)/(kh omega) cos(kh dx/2)`,
distinct from the phase speed. Measured envelope-centroid speed 0.137 vs
`c_g` 0.143 (4.4%), and `|meas - c_g|` 6.3e-3 << `|c_p - c_g|` 5.9e-2 —
unambiguously the group velocity, in a genuinely dispersive regime
(`c_g/c_p = 0.71`). The dispersion is a linear-core property, so the
operator is probed on an `advection=False` twin while the packet runs on
the preset.

**Eady baroclinic instability — DEFERRED (STOP), by design.** The Eady
problem needs a thermal-wind-balanced mean state: a vertical shear
`U(z)=Lambda z` AND a mean meridional buoyancy gradient
`d_y B = -f Lambda`, whose conversion term `v' d_y B` in the buoyancy
equation is the baroclinic energy source. The shared advection
`background=` **does** compose with the hydrostatic model (verified: it
assembles, runs, adds `background_u` as an AUXILIARY field, and
contributes the mean-flow Doppler advection `U d_x(.)` — a `b`-mode gets
`db/dt = 1.089`), but it keys **only** velocity components (`u/v/w`), so
it supplies no mean *buoyancy* gradient; and the doubly-periodic
y-domain cannot carry the non-periodic `d_y B = -f Lambda` as a state
field. There is no module for a background horizontal buoyancy gradient
(`MeridionalStratification` supplies only `N^2(y)`, the vertical
restoring). Consequently the config has no `v' d_y B` term at all: a
v-only, divergence-free state (w exactly zero) gives `db/dt == 0.0`
(pinned in `test_eady_baroclinic_conversion_term_is_absent`), so the
configuration cannot sustain an Eady instability and a growth-rate
measurement is not achievable here. Unblocking needs a new
background-buoyancy-gradient module (a `-M^2 v` restoring on `b`, the
thermal-wind twin of `ConstantStratification`) plus the momentum-tilting
`w' d_z U` term — a follow-up (candidate: a `ThermalWindShear` module
supplying both, sampled as AUXILIARY profiles like the background flow).

**External legs — PENDING (out of tree, this machine).** The
out-of-tree `benchmarks/comparison` harness named in the §4 H5 gate is
**not present** on this machine, and pyOM3 source access is pending owner
input (§5). The cross-model *execution* legs are therefore recorded as
pending, to be run against this exact preset once the harness and source
land:
- **Oceananigans** (`HydrostaticFreeSurfaceModel`, local): match with
  `ImplicitFreeSurface()` (FFT), `momentum_advection = Centered(order=2)`,
  `tracer_advection = Centered(order=2)`, `QuasiAdamsBashforth2`
  (default), `FPlane(f=coriolis_f0)`, `BuoyancyTracer()` with `N^2 = n2`.
- **Veros**: rigid-lid streamfunction on the doubly-periodic box =
  `comparison_model(..., epsilon=0)`; centered-2 tracer/momentum;
  quasi-AB2.
- **pyOM2/3**: `enable_free_surface` (backward-Euler, `eps=1`) for
  `epsilon=1` / the rigid-lid Poisson for `epsilon=0`; `AB_eps=0.1`;
  centered-2 flux form (always).
  The comparison metric is HY-D6: bit-level only in the shared limit,
  otherwise convergence + the physical diagnostics above.

**Gates (all green, CPU).** `tests/hydrostatic/` = **139 passed, 1
skipped** (the slow refinement); `ruff check src tests` clean;
`comparison.py` is branchless (a single `return Model(...)`) and covered
by construction across the pin tests (`epsilon` 1/0, custom `eps`/`csqr`/
`f0`/`n2`/`rossby`/`name`) — the local `--cov` run aborts silently (known
issue). The example runs end-to-end under a non-interactive backend.

### H5b — thermal-wind background: the Eady gate unblocked (2026-07-17)

Resolves the H5 STOP. New module
`hy.ThermalWindBackground(shear=Λ)` (`modules/thermal_wind.py`,
lazypimp-exported at `hy.` and `hy.modules.`) carrying the two
linearized mean-flow interaction terms of a zonal thermal-wind state
`U(z)=Λ(z−z₀)`, `B(y)=−f₀Λy` on the doubly-periodic f-plane that the
shared advection `background=` does **not** supply. New package param
`hydrostatic.shear`.

**Probe (why a separate module).** The shared advection's
`background=` `background_advection` term
([`advection.py:2389-2424`](../../../src/fridom/model/modules/advection.py),
`_linear_transport`) loops **only over the sampled velocity's own
axis** — with `background={"u": U(z)}` that is `x` alone — so it is
exactly the Doppler transport `−U∂ₓq'` of every advected `q'∈{u',v',b'}`
(`U` independent of `x`), and nothing else. It carries **neither**
mean-flow interaction whose carrier is a mean *gradient* rather than an
advecting velocity: the buoyancy source `−v'∂_yB` (no mean-buoyancy
carrier; the periodic y cannot hold `B(y)`) and the momentum tilting
`−w'∂_zU` (the mean momentum `U` is only an advecting velocity in
`background=`, never an advected quantity). Verified directly: a v-only
divergence-free mode (`w≡0`) gives `db/dt==0` with `background=` alone —
the H5 pin — and `≠0` with this module.

**Term set and signs (thermal-wind-derived, verified exactly).** Both
`linear=True`, no Rossby factor (O(1) mean-flow interactions, like
Coriolis/stratification/Doppler):
- tilting: `du/dt += −Λ·w'` = `−(shear·w.to(u))` (matches the
  independently reconstructed diagnosed `w` to 3.5e-15);
- conversion: `db/dt += −v'∂_yB = +f₀Λ·v'` = `(f0·shear)·v.to(b)`
  (exact); `dv/dt` carries no thermal-wind term.
Signs derived in the module docstring from `∂_yB=−f₀∂_zU` (geostrophy +
hydrostasy of the mean state, the package's `+fv`/`−fu` and
`∂_z p_hyd=b` conventions) — both locked to the single `Λ` so their
relative sign, the one the instability depends on, cannot drift. `f₀`
is read via a `parameter_reference` to `coriolis.f0` (only
`FPlaneCoriolis` **provides** it), so a non-f-plane config is a taught
`MissingParameterError`. Convenience wiring `tw.background_velocity()`
returns the matching `U(z)` callable for
`CenteredAdvection(background=…)`.

**Energy honesty (conversion identity, no conserved norm).** Under
`M=diag(1,1,1/N²,1/c²)` (`hy.energy`) the thermal-wind pair is the
**sole** non-conservative source: `⟨X,M dX/dt⟩` over every *other*
linear term (Doppler, Coriolis, pressure, stratification, free surface)
is machine-zero skew (measured 1.5e-16·scale), and with `Λ=0` the full
system recovers the machine-exact H2 skew. So **no sign-definite
perturbation quadratic form is conserved** — the defining feature of the
instability: the mean flow is a genuine energy source
`dE/dt = −Λ⟨u',w'⟩ + (f₀Λ/N²)⟨v',b'⟩ > 0` for the growing mode. The
internal KE↔PE `⟨w'b'⟩` conversion (the H2 skew) is unchanged. Gate is
the conversion-term identity (the module's contribution equals exactly
those two quadratic forms) plus the eigenvalue growth, not a norm.

**Growth validation.** The discrete linear operator is assembled on the
`(u,v,b)` reduced mode space at fixed `kₓ` (ky=0), rigid-lid CONSTRAINT
projection applied (`model.tendency(filter=linear, constraints=True)`,
the H5 wave-packet probing pattern). Its most-unstable eigenvalue is the
target: seeding the eigenvector and time-integrating, the perturbation
energy grows exponentially at **exactly** the operator rate — measured
`σ=0.11303` vs operator `0.11300`, **rel err 2e-4** (residual is the AB2
`O(σ²dt²)`), a clean `1.070×`/window envelope. `σ∝f₀Λ/N` confirmed
across 5 parameter combos.

**PE vs QG (the honest refinement axis).** The operator eigenvalue sits
**below** the QG continuous `σ≈0.31·f₀Λ/N` (peak at `kN H/f≈1.606`) by
the **primitive-equation (Stone 1966) non-geostrophic correction** at
finite Richardson number `Ri=N²/Λ²`. This is **not** a discretization
error — it is nx- and nz-converged (σ flat to <1% for nz≥12) — so the
gap does **not** close under mesh refinement but along the physical `Ri`
axis: measured `σ/σ_QG` = 0.90 (Ri=1) → 0.945 (Ri=1.78) → 0.98 (Ri=4)
→ →1 as Ri→∞. The "trends toward continuous" gate is therefore the `Ri`
trend (documented), and the QG 0.31 is context, not the target.

**Flipped pins.** `test_comparison.py`:
`test_eady_baroclinic_conversion_term_is_absent` →
`…is_present` (v-only mode now drives `db/dt≠0`); the
composition / Doppler pins and the "deferred STOP" narrative updated to
"unblocked (H5b)". Full growth/energy/sign validation in the new
mirrored shard `tests/hydrostatic/test_thermal_wind.py`.

**Gates (all green, CPU).** `tests/hydrostatic/` = **153 passed, 1
skipped**; `ruff check src tests` clean; `thermal_wind.py` covered by
construction (every method exercised — the local `--cov` run aborts
silently, known issue). No framework-core files touched (module +
package param + exports only), so no cross-model smoke needed.
### H6 — split-explicit free surface (2026-07-17)

**`SplitExplicitFreeSurface(substeps=N, filter=(2,4,0.18927),
forcing="increment"|"tendency_sums")`** (`modules/free_surface.py`). Declares
`ps` (`Profile`) and the barotropic transports `U, V` (staggered-face,
constant-along-z; **no** Velocity role — `table.velocity()` stays `u,v,w`),
plus own-AUX depth-mean buffers `ubar_prev, vbar_prev`. Three halo-trace-exempt
stages: a SELF_UPDATE snapshot of the substage-start depth mean (V-H4
reference), an ADVANCE `lax.scan` over N forward-backward substeps
(`dtau=2dt/N`, the exact linearization of `ExplicitFreeSurface`'s two terms +
the slow forcing G) committing the SM2005-averaged `ps, U, V`, and a CONSTRAINT
replacing depth-mean `u,v` with `U/H, V/H`. Slow forcing G is the increment
`(ū*−ū_start)/dt` (default, scheme-consistent) or the depth-mean of `ctx`'s
per-treatment sums (`forcing="tendency_sums"`, EXPLICIT + IMPLICIT when present).
SM2005 weights are host-computed at construction (Oceananigans normalization,
Σ=1 exact, discrete first moment ~1). Declares `linear_operator_gap` (HY-D7).

**Model-layer touch (additive, for the taught error).** A `supports_split_advance`
ClassVar on `TimeStepper` (default False; True on `AdamBashforth` and
`IMEXMultistep`) plus a `time_stepper` handle on the bind table; the module's
`bind` refuses a non-multistep outer driver (RK / IMEX-RK) with a taught
`AssemblyError` (§5.4 "multistep outer drivers only"). Integrator statics
(N, filter, forcing) ride the ADVANCE stage's attribution name into the restart
fingerprint (V-H3), no further model touch.

**Gates (all green, CPU).** Geostrophic null-eigenvector steady to du=7e-16,
dp=3e-15 (the correction preserves balance). Convergence vs the explicit oracle
on the SLOW mode: buoyancy b converges near second order (rates ~2.3, ~2.9),
the velocity decreases toward the fast-mode-filtering floor (the barotropic
gravity mode is FILTERED, not resolved — captured to ~1% vs a fine explicit
reference). Barotropic stability: `sqrt(csqr)dt/dx=8` with N=32
(substep-CFL 0.5) stable over 100 steps; N=4 (substep-CFL 4) caught by the NaN
seam (PanicError). Conservation: ps-mean 1.7e-16, tracer mass 1.8e-15 over 30
steps. Restart: N/filter/forcing each change the fingerprint; snapshot
round-trip bitwise (own-AUX carried); a different N refuses resume
(`SnapshotMismatchError`). Forced-4 multi-device bitwise-identical to
single-device. `tests/hydrostatic/test_free_surface_split.py` 39 tests + 1
forced-device; `tests/hydrostatic` 163 passed; steppers/assembly/end-to-end 188
passed; nonhydro smoke 8. `ruff` clean; coverage by construction (the local
`--cov` SIGABRT is environmental — all new branches exercised: degenerate
filter, every validation, both guard outcomes, all three forcing paths).

### H7 — constancy-preserving surface advective flux, default on (2026-07-17)

**Symptom (found by the out-of-tree Oceananigans comparison bench,
2026-07-17).** `hy.comparison_model` (implicit + centered, HY-D6) goes
non-finite at 512²×32 (~iter 700 at dt=1e-3) and every larger rung.
Autopsy (bench `autopsy/*.json`): onset at fixed **physical** time
0.65–0.8 for dt ∈ {1e-3, 5e-4, 2.5e-4} and unchanged under forward
Euler / AB2 `eps=0.5` — a semi-discrete defect, not temporal; the
grown mode is an **interior, large-scale (kx=ky=1–2)** w–b overturning
descending from the upper interior; 256²×32 never panics but is
**corrupted** — max|w| saturates 5–8 and b grows 0.01→0.8 while the
quiet yardsticks (implicit with `advection=False`: u~0.49, w~0.05;
the Oceananigans implicit twin: u~0.46, η~4.8e-3) stay small.

**Root cause: the H2b constancy exception.** Dropping `w(0)` leaves
`A(q=const) ∝ q·w(0)/dz` in the surface cell for every ADVECTED field
(1/dz = 32 amplifies). Under the divergent comparison IC, `w(0)` is
the large-scale ∂η/∂t of the barotropic adjustment (O(0.3–1) early),
so the top layer is pumped (~2× hot top-cell u in the parity twin —
exactly the recorded H5 parity anomaly), the stratification
overturns, and the O(1)-amplitude inviscid centered state blows up at
fine dx (the H2b note's "nonlinearly unstable" resolution property).
Refuted along the way: AB2 over-extrapolation of the implicit term
(the HY-D4 projection stays out of the ring, in fridom AND in
Oceananigans — verified in both sources); any dt/stepper mechanism.
The split-explicit variant escapes only because SM2005 averaging
damps the (purely barotropic — the comparison IC is z-uniform)
adjustment to u~5e-3 before the pump acts; Oceananigans is
structurally immune (it advects **through** the top face with the
continuity `w`, constancy-preserving in every cell).

**Fix + ruling (owner-ratified in chat, 2026-07-17).** The
constancy-preserving surface closure is the **default**:
`A_new(q) = A_old(q) − q·A(1)`, with the lean `A(1)` = the flux
divergence of the (immersed-weighted) interpolated face velocities
themselves — exact because every reconstruction preserves constants,
so `face(1) ≡ 1` — interleaved into the per-axis flux loop
(`v_face`/`flux_space` reuse is XLA-CSE'd; the separate-pass form is
bitwise-identical). Equivalent to advecting through the boundary
faces with the one-sided face value: the Oceananigans linear-free-
surface treatment. Surface: `surface_flux: bool | None = None` on the
flux-form base (`CenteredAdvection`/`UpwindAdvection`/
`WENOAdvection`), tri-state — None = **auto**, on iff an advecting
velocity's own-axis factor sits on the both-boundary `Outer` node set
(the exact `_outer_to_inner` seam predicate), resolved at bind. So
hydrostatic advection is corrected for every scheme and construction
path, while nonhydro2/shallowwater2 resolve off and stay **bitwise**
unchanged (tested). `hy.Model` / `hy.comparison_model` forward
`surface_advective_flux: bool | None = None`; `False` restores the
H2b fixed-domain closure (tracer content conserved to roundoff — and
with it the instability).

**Consequences.** Tracer content is now exchanged with the moving
surface (the accumulated `−∮ b_top·w(0) dA`): oscillating, rel
~1e-5..1e-3 on the parity run, not secular. The H2b conservation /
surface-localization tests are pinned to `surface_flux=False` as
legacy characterization (5 tests). Energy-orthogonality is
**unchanged** (⟨q, M A(q)⟩ = −1.7e-18 vs 6.2e-17 legacy): the
corrected operator is the advective form with the divergence-free
continuity velocity. Background-split terms are NOT covered by the
correction (a background `w` with nonzero surface value would still
be dropped; no such background exists today).

**Validation (causal, single A100; bench `autopsy/fix/*.json`).**
512²×32 implicit+centered: stable 3000 steps (was panic @700),
settles u~0.47/w~0.03/ps~1, tracking the advection-free reference
through the (physical) w≈0.71 adjustment transient. 256²×32: quiet
through 8000 steps (was w~5–8, b~0.8). Parity twin (64²×8): top-cell
u 0.264→0.1376 = Oceananigans to 0.001% (n2=0) / 0.06% (n2=1), flat
profile. Discriminators on the UNFIXED code: divergence-free IC →
quiet at 512² (the pump is `w(0)` of the divergent adjustment);
split-explicit → quiet (filter, see above).

**Cost + follow-up.** +1.37 ms/step at 512²×32 implicit+centered
(2.98 → 4.35, +46%): one extra flux divergence per advected field per
axis, memory-bound, irreducible by face-velocity reuse (XLA CSE
already dedups — the interleaved and separate-pass forms measure the
same). Real-suite resweep (2026-07-17, post-merge): `se_centered`
+18/+30/+36/+45/+49% across the ladder (101.8 → 151.4 ms/step at
2048²×64), `se_weno5` +11–36%, `*_linear` unchanged; the centered
hydro oc/fridom ratio drops from ~break-even to 0.79–0.88, and all
five `im_centered` rungs now complete stably (rf23–28 were
non-finite). [Update: the 2D slice-only `A(1)` evaluation executed
2026-07-18 — the H7 `A(1)` term lowers on a `TraceSpace` 2-D slice
where valid (flat + collocated tracers; the embed route stays the
default elsewhere). See done.md, boundary trace machinery.]

**Gates.** `tests/hydrostatic` + `tests/model/modules` 759 passed /
4 skipped; `ruff` clean; autodiff regression (grad through
`_chunk_body`, default-on) FD-matched. Merge `8bd91dfe` (branch
`fix/hydro-surface-advective-flux`, 5 commits).
