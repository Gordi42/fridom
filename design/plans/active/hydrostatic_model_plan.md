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
| **H5** | Comparison protocol: the HY-D6 config as a preset + example script; extend the out-of-tree `benchmarks/comparison` harness (Oceananigans local; Veros; pyOM3 pending source access — owner input) | M (1 wk) | matched-protocol physics: geostrophic adjustment, dispersion, Eady-type growth rates, spin-down energy budgets; grid-refinement convergence |
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
- **pyOM3 access:** publicly undocumented; the pyOM2 doc (its
  discretization core) is authoritative meanwhile. Owner to supply
  pyOM3 source/config for the H5 delta check.

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

*(H0/H1 entries land above this line when they ship; H2 below.)*

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
now counts a CONSTRAINT-stage write as covering a PROGNOSTIC field —
`ps` is advanced only by the projection, genuinely integrated forward.
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
