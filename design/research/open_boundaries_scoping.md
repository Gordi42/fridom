---
status: frozen
date: 2026-07-17
---

# Open boundaries (inflow/outflow) — scoping and sizing

**Question (Silvano, 2026-07-17):** how much work is it to implement
open boundaries (prescribed inflow, outflow, radiation) in fridom?

**Method:** probed `dev` at `5cfd3668` (two independent code sweeps:
boundary/spatial layer, solver/model machinery), plus an external
survey of MITgcm `obcs`/`rbcs`, Oceananigans.jl `OpenBoundaryCondition`,
ROMS/CROCO, NEMO `BDY`, and the FFT-pressure-solver strategies
(sources in §6). Everything below is scoping — no owner decisions are
made here.

## 1. Where the new stack stands

Open boundaries are **designed-for but nowhere implemented**. The
grid spec already contains the architecture
(`specs/grid/02_rules.md` §3.6): boundary *structure* lives in the
space, boundary *data* are dynamic module-owned trace fields, and one
`OpenBoundary` module dispatches per-discretization enforcement
(nodal → `("ghost_fill", space)` rows, FV → boundary-face DOFs on the
`Outer` flux space, Galerkin → boundary modes). The pieces on `dev`:

| Piece | State |
|---|---|
| Per-side BC kinds in the space key, per-side fills, Dirichlet-drops-the-DOF shape rule | shipped (`spatial/bc.py`, merge `9a95202a`) |
| Homogeneous ghost fills (Dirichlet odd / Neumann even); **inhomogeneous fill rejected**: "designed-for; iteration 1 is homogeneous only" | shipped / gap (`decomposition/tensor.py:984`) |
| Dynamic boundary-data path (`("ghost_fill", space)` rows) | **stage 2e, deferred**, sized 1–2 wk with two recorded design questions ([`../plans/active/boundary_plan.md`](../plans/active/boundary_plan.md)) |
| One-sided operator rows (motivated by "open boundaries, diagnostics") | shipped, `boundary="one_sided"`, `layout="local"` (axis must be undistributed) |
| Wall-normal velocity | structurally impermeable: declared Dirichlet on `Inner` — the boundary face is **not a DOF** (`grid.py:2335`, `declarations.py:447`) |
| Advection at walls | structural exact-zero wall flux + interior-only graded stencils; **never reads an inflow value** (`model/modules/advection.py:163–206`) |
| Flux-type boundary forcing | shipped as tendency injection (`BoundaryFlux`); wall-normal velocity **refused by taught error**: "prescribing wall-normal flow is an open-boundary condition, out of scope" (`boundary_flux.py:243–248`) |
| Relaxation/nudging with spatial mask + swept rate | shipped (`model/modules/relaxation.py`) — the sponge primitive |
| Scalar time curves (`Ramp`/`TimeFunction`/`TimeSeries` via `ctx.params`) | shipped; general time-dependent *fields* are an open roadmap item |
| Trace meshes (`mesh.boundary` → `PointMesh`) | shipped (`meshes/structured_1d.py:168`) |
| State-overwrite hook for boundary updates | CONSTRAINT stages replace prognostic state (the projection itself is one, `nonhydro2/modules/core.py:501`) |
| Pressure solve | spectral, structural per axis: periodic → Fourier, bounded → Neumann/DCT-II sibling (`nonhydro2/modules/pressure.py:60`); wall-face gradient structurally zero (`_dirichlet_mid`, `pressure.py:104`); CG + spectral preconditioner exists for mapped grids (`mapped_pressure.py`, `spatial/operators/krylov.py`) |

## 2. What other models do (survey, condensed)

- **Sponge/relaxation zones** (NEMO FRS; MITgcm `rbcs`; Lavelle &
  Thacker 2008) are additive nudging tendencies — no solver contact.
  The robust workhorse for idealized setups.
- **Prescribed/clamped inflow + radiation outflow** (MITgcm `obcs`
  Orlanski/Stevens; ROMS Flather/Chapman/radiation-nudging;
  Oceananigans `Open` BCs with `PerturbationAdvection` /
  `FlatExtrapolation` matching schemes).
- **The FFT-solver question** (fridom-relevant): Oceananigans keeps
  its fast transform solver by filling the open boundary **in the
  predictor only** and leaving it untouched by the correction — the
  solver still sees homogeneous Neumann; their docs call open BCs
  "typically already unphysical". The non-negotiable companion is the
  **solvability condition**: with Neumann/periodic on every face,
  `∇²p = ∇·u*/Δt` is solvable only if net volume flux through open
  faces vanishes — MITgcm enforces it explicitly
  (`useOBCSbalance`: adjust all normal boundary velocities to zero
  net inflow; one reduction + uniform correction). Honest
  inhomogeneous-Neumann alternatives if predictor-only proves too
  crude: per-mode tridiagonal solve on the open axis (FFT elsewhere;
  Costa 2018), capacitance-matrix correction, or full CG.
- **Well-posedness**: one BC per incoming characteristic — inflow
  needs nearly all fields prescribed, outflow wants
  radiation/extrapolation; over-specified outflow reflects.

## 3. The tiers, sized

### Tier 0 — possible today (zero code)

A sponge-emulated open boundary next to an ordinary wall:
`Relaxation` with a coordinate-dependent mask
(`grid.create_field(init=...)` / `fr.Profile`) ramping over the last
N cells, nudging `u, v, w, b` toward a prescribed exterior state,
`TimeDependent` rate. Absorbs outgoing waves and imposes far-field
state. **No net through-flow** (the wall stays impermeable), and the
target profile is static (time-varying targets = the open
time-dependent-fields roadmap item; affine blends of profiles are
covered by the shipped R1/R2 ramping machinery).

### Tier 1 — packaged sponge module: SMALL (days)

`fr.modules.SpongeLayer(fields, coord, side, width, tau, target=...)`
— a per-side graded-mask builder (tanh/linear over `width`, measure-
aware on stretched meshes) wrapping the `Relaxation` term for several
fields; taught errors per the `BoundaryFlux` template. ~200–400 LOC
source + mirrored tests. No solver contact, no new machinery —
`Relaxation` + `build_wall_weight` precedents make this mechanical.
Covers the majority of idealized-model "open boundary" needs
(wave-absorbing edges, prescribed far fields).

### Tier 2 — genuine through-flow: LARGE (3–6 weeks), four shippable stages

The expensive part is **not the pressure solver** — it is making the
boundary-normal velocity's wall face a DOF at all, and feeding data
to it. Stages:

- **2-i. Open-side structure (~1 wk + design record).** A per-side
  "open" declaration under which the wall-normal velocity keeps its
  boundary-face DOF on that side (per-side mixed kinds already flow
  through `space_key`; the candidate spelling is an `Outer`-family
  space with one Dirichlet side dropped, or a new `BC.OPEN` kind —
  owner call). Wide blast radius: the declared-space resolver
  (`grid.py:2335` maps STAGGERED→`Inner` on bounded), the advection
  structural-zero wall fluxes and graded closures, the projection
  retags, the `BoundaryFlux`/velocity taught errors — every
  wall-aware consumer must decide what an open side means. This is
  the reason Tier 2 is weeks, not days.
- **2-ii. Boundary data + enforcement (~1–2 wk).** Prescribed
  inflow values as module-owned AUXILIARY tangential-profile fields
  (the `BoundaryFlux` wall-weight precedent shards them for free).
  Enforcement per spec §3.6: on the **FV path (now the default
  family)** the prescribed advective flux occupies the `Outer` flux
  space's wall slot that today carries the structural zero — the
  cleanest route, no ghost fills. The **nodal path** is exactly
  boundary-closure **stage 2e** (`("ghost_fill", space)` dynamic
  rows), already sized 1–2 wk on its own; its two recorded design
  questions (traced-not-frozen data; `shard_map` `in_specs` for
  transverse-sharded boundary faces) apply verbatim. Time-dependent
  inflow: `TimeDependent` scale × static profile works today; fully
  general `g(y, z, t)` joins the open time-dependent-fields design.
- **2-iii. Flux balance + projection integration (~1 wk incl.
  gates).** Keep the spectral solver untouched: homogeneous-Neumann
  pressure at open sides, prescribed normal velocity applied
  pre-projection and left uncorrected — structurally already true
  (`_dirichlet_mid` pins the wall-face gradient to zero). Add the
  **mandatory balance pass** (MITgcm `OBCS_balance` pattern: one
  reduction over open faces + uniform correction) — without it the
  `k = 0` pseudo-inverse gauge silently swallows the residual and
  leaves a uniform spurious divergence. Gates: machine-zero global
  divergence from deliberately unbalanced prescribed inflow;
  volume/tracer budgets `d/dt ∫ = Σ boundary fluxes` to rounding.
- **2-iv. Outflow/radiation schemes (~1–2 wk).** Orlanski /
  perturbation-advection boundary updates as CONSTRAINT-stage
  overwrites of the boundary-face DOFs and near-boundary tracers
  (precedent: the projection, `MaskState`); branch-free `where`
  math is jit-pure. One-sided phase-speed estimates ride the shipped
  `boundary="one_sided"` rows — whose `layout="local"` constraint
  means **iteration 1 requires the open axis undistributed** (the
  shard-masked `patch_physical_ends` pattern is the later lift).

**shallowwater2 Flather is the cheap pilot (~1 wk).** `p` is
prognostic and there is no elliptic solve, so Flather/FRS are pure
boundary-update + relaxation modules — it exercises the open-side
structure (2-i) without the projection questions, and the
hydrostatic model would inherit it.

**Honest-pressure upgrade (only if needed, +1–2 wk).** If
predictor-only inflow proves too crude, the inhomogeneous-Neumann
solve on one open axis = per-mode tridiagonal (FFT on the other
axes) or CG on the true BC — `ConjugateGradient` + the
spectral-preconditioner pattern already exist (mapped solve; the
immersed plan's masked Poisson), so this is an extension, not new
machinery.

## 4. Design questions needing owner calls (before Tier 2 starts)

1. **Which consumer?** Per the roadmap's who-wants-it rule: sponge
   (Tier 1) vs genuine through-flow (Tier 2) serve different studies.
   Nothing currently on the roadmap needs Tier 2.
2. **Open-side spelling**: `BC.OPEN` kind vs per-side-mixed `Outer`
   space; what every wall-aware module does with an open side.
3. **Enforcement route**: FV wall-slot flux as the primary path
   (recommended by the FV-default flip) vs pulling stage 2e forward
   for the nodal path.
4. **Well-posedness policy as taught errors**: which field sets must
   be prescribed at inflow / radiated at outflow; which combinations
   are refused.
5. **Distribution**: accept `layout="local"` (open axis undistributed)
   for iteration 1?

## 5. Interactions

- **Boundary-closure 2e**: Tier 2's nodal data path *is* 2e — doing
  Tier 2 finally gives 2e its consumer.
- **Immersed partial cells plan**: shares the "solver beyond pure
  spectral" machinery (masked/inhomogeneous CG) with the
  honest-pressure upgrade.
- **Time-dependent fields (roadmap)**: general `g(y, z, t)` inflow
  is a consumer of that design.
- **ETDRK4**: boundary updates are stage-based state overwrites, not
  `L`-resident terms — no frozen-`L` interaction.

## 6. External sources

MITgcm obcs docs (mitgcm.readthedocs.io/en/latest/phys_pkgs/obcs.html;
`useOBCSbalance`); Oceananigans numerical-implementation BC docs
(github.com/CliMA/Oceananigans.jl, docs/src/numerical_implementation/
boundary_conditions.md — predictor-only open fill, homogeneous-Neumann
retention) and arXiv 2502.14148 (immersed FFT/CG fallback); WikiROMS
Boundary Conditions (Chapman/Flather/RadNud); NEMO BDY manual (FRS +
Flather); Orlanski 1976; Flather 1976; Lavelle & Thacker 2008 ("a
pretty good sponge"); Costa 2018 arXiv 1802.10323 (FFT + tridiagonal
solvers); Neumann-Poisson compatibility (J. Comput. Phys.,
0021999187900088).
