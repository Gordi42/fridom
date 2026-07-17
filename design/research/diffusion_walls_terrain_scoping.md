---
status: frozen
date: 2026-07-17
---

# Diffusion/friction closures at walls and on terrain — scoping

**Question (Silvano, 2026-07-17):** can `HarmonicFriction` run with
no-slip / free-slip boundary conditions, and on terrain-following
grids? Today: no on all counts — the whole family
(`Harmonic|Biharmonic` × `Diffusion|Friction`) rejects any bounded
grid with a taught error (`model/closures/diffusion.py:307-317`),
which incidentally also catches every terrain grid (bounded column),
and `VerticalMixing` is Neumann-rows-only. This record designs the
lift: the machinery inventory, external practice, per-case design,
staged sizing, and the owner calls.

**Method:** probed `dev` at `c05f68c3` (two independent code sweeps —
wall/BC machinery, mapped machinery — plus a design-space analysis),
and an external survey of MITgcm, NEMO, MOM6, Oceananigans.jl and the
terrain-following mixing literature (sources §8). One agent
disagreement (stretched-column correctness of `VerticalDiffusion`)
was resolved against the primary source (§3.7). No owner decisions
are made here; §6 lists them.

## 1. Where the stack stands

**Why the chain fails at walls (the load-bearing mechanism).** The
family is a two-pass flux chain `(q.diff(a) * kh).diff(a)`
(`diffusion.py:165-182`). Three shipped rulings make the second pass
impossible on a bounded BC-free axis, by design:

- A derivative's codomain is always the **BC-free sibling** — "a BC
  tag is a wall-value claim, not a parity/extension claim"
  (`boundary_plan.md:39-44`); the operand's tag governs only the
  ghost fill (`operators/finite_difference.py:147-176`).
- The bounded `diff` signatures are `Center→Inner` and
  `Inner|Outer→Center` only — there is **no `Center→Outer`**, so the
  chain never emits a wall-face value.
- **R1**: exterior reads on BC-free bounded axes raise; the storage
  layer never invents values (`plans/done/bc_free_boundaries.md`).

So pass 1 lands the flux on BC-free `Inner` (interior faces only)
and pass 2 (`Inner→Center`) needs the wall-face flux it cannot have.
The rejection is honest; the fix is to close the wall flux
explicitly, per case (§3).

| Piece | State |
|---|---|
| BC kinds (structure only, values dynamic) + per-side `BCStructure` in the space key | shipped (`spatial/bc.py:19-137`; merge `9a95202a`) |
| Homogeneous ghost fills: Dirichlet = odd mirror, Neumann = even mirror; single source of truth, runs inside `shard_map`; BC-free sides return nothing (R1) | shipped (`decomposition/tensor.py:1461-1534`) |
| Shape rule: Dirichlet on `Center`/`Inner`/`CellAvg` drops **no** DOF (boundary not a member); Neumann never drops shape — wall tags on closure targets are shape-stable | shipped (`spaces/nodal.py:58-107`) |
| BC-sibling retag: pure tag swap, no data move | shipped (`fields/scalar_field.py:1228-1260`, `.retag`) |
| Wall-normal velocity: auto-derived `BC.DIRICHLET` on `Inner` — impermeability structural, wall face not a DOF | shipped (`model/declarations.py:454-478`, C8) |
| **Tangential** velocity factors: **no wall tag today** (`BC.NONE`) — the open slot the slip choice fills | gap (`nonhydro2/modules/core.py:446-472`) |
| Advection wall precedent: flux space adopts the wall-normal Dirichlet tag (BC-sibling substitution) → structural exact-zero wall flux; graded interior-only ladders for wide stencils | shipped (`model/modules/advection.py:163-215`, `operators/graded.py`) |
| Walled FV dispatch rows: `flux_diff` registered on BC-free `Inner` **and** `Inner[Dirichlet]`; `face_diff` on `CellAvg[Neumann]` (F4) — the closing rows §3.1 needs | shipped (`nonhydro2/modules/core.py:246-266`) |
| Wall-row tendency precedent: `BoundaryFlux` injects `s·F·W`, `W = 1/Δn` in the wall cell; reads per-side BC kinds | shipped (`model/modules/boundary_flux.py`) |
| Inhomogeneous / Robin boundary data (`("ghost_fill", space)` dynamic rows) | stage 2e, deferred (`boundary_plan.md:77-146`) |
| Physical (constant-z) derivative on mapped grids: `"physical_diff"` dispatch kind, chain-rule `∂ₓ|_z = ∂ₓ|_σ − (Z/J)∂_σ`, params-threaded | shipped (`spatial/operators/mapped.py:226-487`) |
| Mapped elliptic template: `∂ᵢ(Kⁱʲ∂ⱼp)`, `K = J·Bᵀ W B`, corner cross hops with exact-transpose pairing (SPD) | shipped (`nonhydro2/modules/mapped_pressure.py`) |
| Conservative mapped flux divergence + unified convention (stretching from `grid.measure`, terrain from `grid.metric`, J-weighting module-side, matched pairs) | shipped (`advection.py:2408-2491`; `stretched_terrain_combined.md` §6) |
| `sqrt_g` for analytic `maps=` grids (§7-addendum ruling 4) | ruled, unimplemented |
| `VerticalDiffusion` column matrix: **uniform spacing inferred from the first two nodes**, Neumann corners hardcoded | shipped-with-hazard (`operators/banded.py:49-81`; §3.7) |

## 2. What other models do (condensed)

| Model | Free-slip | No-slip | Default | Biharmonic BC | Terrain mixing |
|---|---|---|---|---|---|
| MITgcm | mask wall stress to zero | explicit side-drag body force in wall cells, `sideDragFactor=2` (half-cell ⇒ factor 2), thin-wall aware | **no-slip** | same drag on `u` and `∇²u` | z-coord; rotated tensor via GMREDI |
| NEMO | coastal f-point shear zeroed (`shlat=0`) | ghost `u=−u_int` over Δ/2 (`shlat=2`); **continuous partial slip 0–2** | user choice | same masking, both passes | per-config orientation |
| MOM6 | corner shear masked | corner shear doubled (`2−mask`); **Laplacian only** | **free-slip** | iterated-√B, thickness on 2nd pass only; **biharmonic is free-slip-only** | isopycnal (sidesteps σ) |
| Oceananigans | zero-flux halo (default) | `Value(0)` on tangential velocity → spacing-aware ghost | **free-slip** | halo fills per pass | stretched-z, immersed |

Convergent facts that matter here: (a) the universal discrete
**no-slip** is the factor-of-2 ghost (`u_ghost = −u_int` across the
half-cell) — exactly fridom's shipped Dirichlet odd mirror; the
universal **free-slip** is zero tangential wall stress — exactly
fridom's structural zero wall flux. (b) Griffies & Hallberg (2000,
read in full): per-pass `√B`, the **same slip condition reused on
both Laplacian passes**, thickness only on the second pass — fridom's
iterated-sqrt form is already the canonical one. (c) There is **no
consensus default** (MITgcm no-slip vs MOM6/Oceananigans free-slip).
(d) Terrain: along-coordinate **viscosity** is accepted practice
(ROMS default `MIX_S_UV`); along-σ **tracer** diffusion is the
spurious-diapycnal-mixing hazard, and Marchesiello et al. (2009)
show the implicit diffusion of upwind advection dominates it —
rotating only the explicit operator is insufficient anyway. (e)
C-grid no-slip at coarse resolution acts as an uncontrolled
Δ-dependent side drag — why NEMO exposes continuous `shlat`.

## 3. The design, case by case

### 3.1 No-flux tracers and free-slip tangential — structural, nearly free

The wall flux is exactly zero in both cases (tracer no-flux;
free-slip = zero tangential stress). Mechanism (the advection
precedent verbatim): form the interior flux `q.diff(a) * kh` on
BC-free `Inner`, **retag it to `Inner[Dirichlet]`** — a true
wall-value claim, the flux *is* zero there — and let
`flux_diff` (`Inner[Dirichlet]→Center|CellAvg`) telescope with the
structural exact-zero wall flux. The closing dispatch rows already
exist on walled meshes (`core.py:246-266`). Per-axis; periodic axes
bit-identical to today. Homogeneous-Dirichlet *tracer* walls (fixed
wall value 0) are also expressible via the operand's odd-mirror fill;
inhomogeneous wall values (heated plate) are 2e consumers, out of
scope here.

### 3.2 Wall-normal component — derivable, no choice, free

For the component staggered along the walled axis (`w` at a lid):
its own space is already `Inner[Dirichlet]`, so pass 1
(`Inner[Dirichlet]→Center`) closes on the structural wall value 0,
and pass 2 (`Center→Inner`) is interior-only — no wall flux is ever
needed. Only a final `retag` onto the component's tagged space is
added (the `core.py:604-609` projection pattern). Impermeability and
the normal condition coincide for both slip flavors — this case has
no knob.

### 3.3 No-slip tangential — the one genuinely new stencil

No-slip has a **nonzero** wall-face flux `≈ −2ν u₁/Δn` (the
factor-of-2 ghost against the wall value 0 across the half-cell).
The chain structurally cannot emit it (§1). Two spellings:

- **(a) Wall-row correction (recommended).** Keep the §3.1 chain
  (which alone would be free-slip) and add the wall-adjacent cell
  correction `−flux_wall/Δn` with `flux_wall = −2ν u₁/Δn` — a
  static, homogeneous-linear-in-`u₁` row: differentiable, no masked
  singularity, no dynamic-data machinery. This is MITgcm's side-drag
  form and the `BoundaryFlux` `W = 1/Δn` idiom; module-side, per
  the F5 "capability in the consuming module" precedent.
- **(b) `Center[Dirichlet]→Outer` diff row.** Register the missing
  wall-face-emitting derivative (the odd-mirror ghost gives the wall
  face `2u₁/Δn` automatically), multiply by ν on `Outer`, close with
  the existing `Outer→Center` row. One mechanism for both slips,
  stays entirely in halo-traced field arithmetic — but it is a
  spatial-operator/registry change with wider blast radius, and the
  new row needs its own graded/validity account.

Recommendation: (a) first — smallest blast radius, precedented both
in-repo and externally; note (b) as the clean long-term unification
if a second consumer of wall-face fluxes appears (open-boundary
Tier 2 would be one).

### 3.4 Biharmonic at walls

Apply the **same wall treatment on both passes** (Griffies &
Hallberg's construction; MOM6/NEMO practice). The intermediate
`H(q)` lands on `Center` BC-free and pass 2 simply re-runs the same
§3.1/§3.3 closure — structurally natural, nothing new. The no-slip
biharmonic *pair* (`u=0 & H(u)=0` vs `u=0 & ∂ₙH(u)=0`) is genuinely
ambiguous in the literature; the iterated form fixes "same treatment
both passes". MOM6 ships its biharmonic free-slip-only. Pragmatic
option: biharmonic walls ship free-slip-only first (owner call §6).

### 3.5 Implicit vertical: Dirichlet rows

`VerticalDiffusion` hardcodes Neumann corners
(`banded.py:77-78`); no-slip needs the Dirichlet corner (odd-mirror
⇒ `−3` main-diagonal corner). Change: per-side
`bc=("neumann"|"dirichlet")` on `second_difference_matrix` +
`_diffusion_operator`, surfaced as `VerticalMixing(bottom=...,
top=...)`. **The merge key must carry the BC structure**
(`implicit.py:381-391`: today `(type, axis)` — kappa-summing a
Dirichlet-row and a Neumann-row operator would silently mix
different operators). Implicit no-slip vertical friction is the
higher-value first vertical deliverable: the bottom-boundary-layer
regime is exactly the stiff `κΔt/Δz² ≫ 1` case explicit stencils
cannot afford, and it is currently unavailable anywhere in the
stack. The flagged variable-coefficient column
(`implicit.py:483-487`) stacks orthogonally.

### 3.6 Terrain-following (mapped) grids

Options for `div(ν grad q)` on a `maps=` grid:

- **(A) Full metric-tensor flux form** — the mapped-pressure
  structure `K = J·Bᵀ W B` with `W = diag(ν)`, corner cross hops,
  or its conservative FV twin (`_mapped_fv_divergence`'s J-weighted
  telescoping). Metrically correct including cross terms; a
  medium-large lift per velocity component; the `physical_diff`
  primitive and `grid.metric(..., params=...)` supply everything.
- **(B) Along-coordinate (along-σ) diffusion** — nearly free:
  `q.diff(σ)` already divides by the codomain measure, so the
  existing order-2 chain on a terrain column is the honest
  along-coordinate operator (the same-row Jacobian argument that
  makes order-2 combined advection exact,
  `stretched_terrain_combined.md` §5). "Horizontal" mixing then
  tilts with terrain — the documented ROMS-default behavior for
  viscosity, a physics error on steep slopes for tracers.
- **(C) Rotated/geopotential tensor** — correct orientation, but
  slope caps + implicit-vertical coupling for stability (Lemarié et
  al.); the largest lift, and per Marchesiello the explicit operator
  is not the dominant spurious-mixing source anyway.

Recommendation: **B first, with honest naming and a documented tilt
caveat** (friction is the main consumer and along-coordinate
viscosity is accepted practice), then A as the "geopotential-correct"
follow-up reusing the mapped-pressure machinery, C deferred until a
tracer-fidelity consumer appears. **Anisotropy caveat that must ship
with B**: the family splits axes by *name* (`diffusion.py:331-334`)
— on a terrain grid `nu_v` acts along σ (tilted), not along physical
z; the docstring must say so. Note: a fully-periodic mapped grid
binds today with no gate and silently computes coordinate diffusion
missing the cross terms — stage 0 adds the explicit mapped gate.

### 3.7 `VerticalMixing` on stretched/terrain columns — live silent wrongness

Confirmed against `banded.py:73-81`: `second_difference_matrix`
infers **one uniform `dz` from the first two nodes**. On a stretched
column the matrix silently applies the first interval everywhere
(wrong `d²/dz²`); on a terrain column the solve axis is base-σ and
the operator is `d²/dσ²` with no `H(x)` (`evaluation_nodes` reads
only the mesh's own coordinate map — the chart never enters,
`grid.py:1821-1874`). `VerticalMixing.bind` gates immersed grids
only (`vertical_mixing.py:225-233`) — **both cases bind and run
silently wrong today**. This is the exact class the taught-error
doctrine targets (the hydrostatic silent-27%-`p_hyd` sibling,
`stretched_terrain_combined.md` §4). Fix now: taught
`NotImplementedError` on a non-uniform solve-axis column or a
mapped column (two-predicate check, the F5 gate precedent). Real
fix later: a measure-aware banded column (true non-uniform spacing
from `grid.measure` + J-weighting for terrain) — the same
face-averaged conservative column as the variable-kappa follow-up.

## 4. API spelling

`slip="free" | "no"` kwarg on the **friction** closures only
(mixing closures have no slip choice — tracers are no-flux), with
the existing `per_field_options` machinery accepting a per-field /
per-wall mapping refinement. The wall-normal component ignores it
(derived, §3.2). Deriving slip from field-declared BC tags or a
grid-level wall declaration was considered and deferred: the core
declares no tangential tags today, and coupling a closure knob to
field declaration is a wider-blast-radius decision that should wait
for a second consumer (§6). Partial slip (NEMO `shlat`) is exactly a
Robin wall closure — the 2e `("ghost_fill", space)` data path is its
natural future home; the flux-form design here leaves that door open
(`boundary_plan.md:106-116` names flux form as the Robin diffusion
route).

## 5. Stages, sized

Calibration (real diffstats): walled centered advection `4b4bc85c`
+270 src/+210 test; graded ladder `4eac7ccf` +513/+230;
`VerticalMixing` landing `b467f7b6` +683 total; boundary stage
`9a95202a` 21 files/+1054.

| Stage | Content | Size (src/test) |
|---|---|---|
| **0 — taught gates (ship now)** | `VerticalMixing`/`VerticalDiffusion` reject non-uniform or mapped solve columns (§3.7); explicit family adds the mapped gate for the periodic-mapped edge (§3.6) | ~30-60 / ~40-80; `pytest.raises` tests only |
| **1 — free-slip walls, explicit family** | §3.1 + §3.2: flux retag to `Inner[Dirichlet]`, wall-normal via its own tag; drop the walled rejection | ~150-250 / ~180-260 |
| **2 — no-slip + `slip=` API** | §3.3 spelling (a) wall rows + kwarg + per-field/per-wall plumbing | ~150-300 / ~180-280 |
| **3 — implicit Dirichlet rows** | §3.5: banded corners, `merge_key` carries BC, `VerticalMixing(bottom/top=...)` | ~80-160 / ~150-250 |
| **4 — mapped along-σ (honest)** | §3.6-B: enable order-2 chain on mapped columns, tilt + `nu_v`-along-σ docs | ~100-200 / ~150-250 |
| **5 — full metric / rotated** | §3.6-A (then C) | deferred, separate plan |

Ordering: 0 unconditionally now; 1→2 (free-slip is the substrate
no-slip corrects); 3 independent, high value (bottom drag); 4 after
1 (terrain columns are walled); 5 deferred. Per-stage gates: the
mirrored `test_diffusion*.py` / `test_vertical_mixing.py` shards;
one discrete-symbol convergence test per slip flavor (cosine modes
for Neumann/free-slip, sine for Dirichlet/no-slip — the
`test_diffusion.py:85-92` pattern); a manufactured-solution order-2
test on a combined stretched+terrain column for stage 4; one
autodiff regression per feature (`_chunk_body`, ≤8³, ≤10 steps,
central FD rtol 1e-4 — the no-slip row is linear in `u₁`, no
singularity guard expected); ruff clean; patch coverage ≥95%.

## 6. Owner calls

1. **Default slip.** Recommend `slip="free"` as a documented
   convention (MOM6/Oceananigans default; both slips are valid
   physics — this is not the silent-wrong-physics class the
   explicit-over-automagic ruling targets). Alternative: require an
   explicit choice on walled grids (MITgcm defaults the other way).
2. **Biharmonic no-slip pair.** Accept "same treatment both passes"
   (the iterated-form natural choice), or ship biharmonic walls
   free-slip-only first (MOM6 precedent)?
3. **Terrain first deliverable.** Is along-coordinate (tilted)
   friction acceptable as stage 4 (accepted practice, honest docs),
   or must the first terrain closure be geopotential-correct
   (promotes stage 5)?
4. **Slip ownership long-term.** Keep slip a closure kwarg, or
   promote it to a per-wall grid/field declaration once a second
   wall-stress consumer exists (one source of truth across
   advection/closures/open boundaries)?
5. **Stage-0 scope.** Gate-only now (recommended), or fold the
   measure-aware implicit column (real fix) into the same change?

## 7. Interactions

- **Boundary plan 2e**: partial slip (`shlat`-style) and
  inhomogeneous wall values (heated plate, moving lid) are 2e
  consumers; this design adds none of that machinery but its flux
  form is 2e-compatible by construction.
- **Immersed partial cells**: masked closures remain a separate
  residual (fraction-weighted viscous stencils, IP-D8); the
  `ClosureBase` immersed gate stays. The immersed spec's ruling
  "structure (no-slip vs free-slip) is the mask combination rule"
  (`specs/grid/02_rules.md:418-420`) should adopt the same `slip=`
  vocabulary when that work starts.
- **Open boundaries Tier 2**: spelling (b) of §3.3 (wall-face flux
  DOFs) is the shared mechanism if both land; neither blocks the
  other.
- **Hydrostatic terrain build (H0–H4)**: stage 4's honest-naming
  caveat applies verbatim to the hydrostatic vertical closure once
  terrain lands there; the `sqrt_g`-for-`maps=` ruling (§7-addendum
  4) is an upstream dependency for any integral-form mapped closure.
- **Stretched-column preconditioners (N1–N3)**: the measure-aware
  banded column (stage-0 real fix) is the same object the
  stretched-column multigrid smoother needs — build once.

## 8. External sources

MITgcm algorithm docs (mitgcm.readthedocs.io — side drag,
`no_slip_sides`, `sideDragFactor`; defaults from `PARAMS.h` /
`set_defaults.F`); NEMO manual dynldf + `rn_shlat` (nemo-ocean.eu);
MOM6 `mom_hor_visc` API docs (ncar.github.io/MOM6 — `NOSLIP`,
corner-shear masking, biharmonic thickness rule); Oceananigans
numerical-implementation BC docs (github.com/CliMA/Oceananigans.jl —
Value/Gradient/Flux halo semantics); Griffies & Hallberg 2000, MWR
128:2935 (gfdl.noaa.gov — read in full: √B per pass, same-BC both
passes, stability ceiling `B < Δ²·A_max/8`); ROMS WikiROMS
Horizontal Mixing / `UV_VIS2` (`MIX_S_UV`/`MIX_GEO_*`); Marchesiello
et al. 2009, Ocean Modelling 26 (spurious diapycnal mixing; rotated
advection-diffusion); Lemarié et al. 2012 (hal-00665826, rotated
harmonic/biharmonic stability); "no-slip at coarse resolution",
Ocean Modelling 2011 (S1463500311000874, abstract-level).
Unverified externals are flagged in place (MITgcm/Oceananigans
biharmonic intermediate-BC details).
