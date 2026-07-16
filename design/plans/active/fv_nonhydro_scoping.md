---
status: active
date: 2026-07-13
---

# FV nonhydro — scoping study

**Question (owner, 2026-07-12):** make the nonhydro model a
**finite-volume model by default** — prognostic variables on the
average family (`CellAvg`/`FaceAvg`) rather than the nodal family
(`Center`/`Right`/`Outer`/`Inner`). *How much would we have to
change?*

This is a scoping study, not an implementation plan: it reports the
current-state seams with evidence, records the decisions, and stages
the work. The staged plan of §6 is the entry point; ROADMAP 3.5
points here. **Update 2026-07-16: stages F0–F3 are implemented and
merged — the periodic nonhydro model is FV by default; see §10 for
the implementation record and the corrections it surfaced. Later the
same day F4 (walls, FV-D4 — §11) and F6 (hygiene G7/G8/G9 — §12)
shipped: walled grids serve explicit `family="fv"` at parity with the
nodal model, and (owner ruling, same day) **walled grids are FV by
default** — auto = FV iff unmapped and unimmersed.** F5 (mapped/chart
FV) remains open as staged; the walled-FV *analytic vertical
eigenmode* stack is a recorded taught gap (§11 addendum).

## 1. Headline

Two facts dominate the estimate.

**(1) The model layer is already family-agnostic.** `nonhydro2` names
a concrete function space almost nowhere. Field declarations go
through the grid-free `SpacePattern` vocabulary
(`Collocated()` / `Staggered("x")` / `Profile("z")`,
[`core.py:95-108`](../../../src/fridom/nonhydro2/modules/core.py)),
whose `Dof` tags (`COLLOCATED` / `STAGGERED` / `CONSTANT`,
[`space_patterns.py:41-58`](../../../src/fridom/spatial/space_patterns.py))
carry **no family**. Which family a tag lands in is decided by a
single per-mesh registry row, `("declared_space", mesh)`
([`grid.py:1915-1975`](../../../src/fridom/spatial/grid.py)), and those
rows are grid-level seedable. Swapping `COLLOCATED -> CellAvg` is
**one resolver row**, and every field declaration in the package
follows for free.

**(2) At 2nd order, the FV and nodal C-grid stencils are the same
numbers.** Probed on a periodic 16-cell mesh (re-verified
2026-07-13): the divergence leg (`FluxDifference: Right -> CellAvg`)
and the nodal `FiniteDifference: Right -> Center` agree to **0.0**;
the gradient leg (`FaceDifference: CellAvg -> Right`) and the nodal
`FiniteDifference: Center -> Right` agree to **0.0**. Same stencil,
different codomain type. So "switch the nonhydro to FV at 2nd order"
is, numerically, a **retag** — and bitwise parity against the
validated nodal model is the right acceptance gate, not an error-norm
comparison.

**Consequence.** The switch is *cheap where it is uninteresting*
(2nd-order periodic: a registry profile) and *expensive exactly where
the payoff would be* (bounded domains, mapped grids). The long pole is
not the model: it is **four missing `eigenvalues` methods** on the FV
operators, which the spectral pressure solve, the eigenmode stack, and
the state transforms all sit on top of.

## 2. What the FV machinery already provides

The average family is complete at the geometry layer and the flux
layer.

| Capability | Evidence |
|---|---|
| `CellAvg` / `FaceAvg` spaces, shapes | [`spaces/average.py`](../../../src/fridom/spatial/spaces/average.py) |
| `flux_diff` (exact Gauss): `Right`/`Outer`/`Inner` -> `CellAvg` | [`flux_diff.py:133-267`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `DualFluxDifference`: `Center`/`CellAvg` -> `FaceAvg` | [`flux_diff.py:269-360`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `FaceDifference` (the FV pressure gradient): `CellAvg` -> `Right`\|`Inner` | [`flux_diff.py:362-441`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `("diff", CellAvg)` = `FVDerivative` = `flux_diff @ reconstruct` | [`flux_diff.py:443-474`](../../../src/fridom/spatial/operators/flux_diff.py), [`grid.py:1828`](../../../src/fridom/spatial/grid.py) |
| `reconstruct` rows: `CellAvg` -> `Right`/`Inner`; `FaceAvg` -> `Center`; `Right`/`Outer`/`Inner` -> `CellAvg`; `Center` -> `FaceAvg` | [`reconstruct.py:342-360`](../../../src/fridom/spatial/operators/reconstruct.py) |
| WENO + graded `Fallback` on the FV family, bounded-capable | [`operators/weno.py`](../../../src/fridom/spatial/operators/weno.py), [`operators/fallback.py`](../../../src/fridom/spatial/operators/fallback.py) |
| Measures on `CellAvg`/`FaceAvg`, incl. **mapped** meshes; `flux_diff` divides by the codomain measure | [`grid.py:1582-1641`](../../../src/fridom/spatial/grid.py), [`flux_diff.py:118-127`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `integrate` exact on `CellAvg` | [`integrate.py:132`](../../../src/fridom/spatial/operators/integrate.py), `tests/.../test_integrate.py:151` |
| Average-origin coefficient spaces (`Fourier(x, origin=CellAvg)`) and the `sinc(k dx/2)` inter-origin factor (`SincShift`, seeded as `("interpolate", coeff)`) | [`spectral.py:615-757`](../../../src/fridom/spatial/operators/spectral.py), [`grid.py:2154-2157`](../../../src/fridom/spatial/grid.py) |
| FFT on average origins (origin-agnostic) | [`fourier.py:14-15`](../../../src/fridom/spatial/operators/fourier.py) |
| Export: averages carry `representation="cell_mean"`, labels not positions | [`export.py:181`](../../../src/fridom/spatial/export.py) |
| `ImmersedDomain` cut-cell fractions on `Center`/`CellAvg` | [`immersed_domain.py:348-375`](../../../src/fridom/spatial/immersed_domain.py) |
| `ConjugateGradient` (PCG) — space-agnostic, measure-weighted | [`krylov.py`](../../../src/fridom/spatial/operators/krylov.py) |

## 3. The nine gaps, ranked

All nine were re-probed on 2026-07-13 against `dev`; **all nine are
still open**.

| # | Gap | Kind | Blocks |
|---|---|---|---|
| **G1** | No `eigenvalues` on any FV operator | fill-in (long pole) | pressure solve, eigenmodes, transforms |
| **G2** | `FaceAvg` cannot be differentiated | new rows | FV-D2 option B only |
| **G3** | No same-location `CellAvg -> Center` deconvolution | new row | `diagnostics.py` |
| **G4** | No `("interpolate", CellAvg/FaceAvg)` rows | new rows | mapped FV (C1–C3) |
| **G5** | `grad`/`div`/`laplacian` resolve `("diff", CellAvg)` -> collocated | registry profile | the FV C-grid |
| **G6** | Walled FV has no BC story | **open design** | walls on FV |
| **G7** | Dealiased (padded) transforms reject average origins | fill-in | the 2/3-rule path |
| **G8** | `discretize` on averages is collocation, not quadrature | fill-in | high-order initialization |
| **G9** | `reconstruct: CellAvg -> Outer` deliberately ungrounded | designed-for | one-sided wall faces |

**G1 (the long pole). No `eigenvalues` on any FV operator.**
`FluxDifference`, `DualFluxDifference`, `FaceDifference` and
`LinearReconstruction` all inherit the raising base
([`base.py:225-254`](../../../src/fridom/spatial/operators/base.py));
their class docstrings still say `eigenvalues` is "designed-for"
([`flux_diff.py:148,281,373`](../../../src/fridom/spatial/operators/flux_diff.py)),
and [`symbols.py`](../../../src/fridom/spatial/symbols.py) carries no
average-family row. Probed: `EigenbasisError: FluxDifference has no
eigenvalues on Fourier(x, origin=CellAvg)`. The FV `Laplacian`
therefore cannot even form its symbol, so `SpectralSolve` cannot run
([`spectral_solve.py:102-114`](../../../src/fridom/spatial/operators/spectral_solve.py)).
*Mitigating:* the four rows have closed forms and every ingredient
exists (origin-agnostic `fourier_wavenumbers`, `diagonal_symbol`,
average-origin Fourier spaces, `SincShift`). This is fill-in, not
research:

| Operator | Symbol | Signature |
|---|---|---|
| `FluxDifference` | `i k sinc(k dx/2)` | `Fourier(Right) -> Fourier(CellAvg)` |
| `DualFluxDifference` | `i k sinc(k w/2)` (dual width `w`) | `Fourier(Center) -> Fourier(FaceAvg)` |
| `FaceDifference` | `2i sin(k dx/2)/dx · sinc`/phase | `Fourier(CellAvg) -> Fourier(face)` |
| `LinearReconstruction` | sinc-corrected averaging | `Fourier(CellAvg) -> Fourier(face)` |

**G2. `FaceAvg` is a dead-end space.** No `("diff", FaceAvg)` and no
`("flux_diff", FaceAvg)` row exists (probed: `DispatchError`). You can
*produce* a `FaceAvg` field and reconstruct it to `Center`, but you
cannot differentiate it. This kills "velocities on `FaceAvg`" (FV-D2
option B) unless new rows are written; option A never needs them.

**G3. No same-location deconvolution `CellAvg -> Center`.** The
registered `reconstruct` row on `CellAvg` lands on `Right` (a *shift*;
probed), so `p.to(center)` fails with `SpaceMismatchError`. This
breaks [`diagnostics.py`](../../../src/fridom/nonhydro2/diagnostics.py),
which interpolates every staggered quantity onto the pressure cell.

**G4. No `("interpolate", CellAvg/FaceAvg)` rows** on physical spaces
(probed: `DispatchError`; [`grid.py:1808`](../../../src/fridom/spatial/grid.py)
seeds `interpolate` on nodal + tagged spaces only). This blocks the
C1 `physical_diff` correction chain, the C2 metric cross-terms /
`RaiseIndex` / `LowerIndex`
([`composed.py`](../../../src/fridom/spatial/operators/composed.py)),
and the C3 corner rows.

**G5. The vector-calculus builders resolve `("diff", factor)`**
([`composed.py:435-444`](../../../src/fridom/spatial/operators/composed.py)).
On `CellAvg` that resolves to the **collocated** `FVDerivative`
(`CellAvg -> CellAvg`, probed), not a staggered face gradient — so
`grad`/`div`/`laplacian` silently produce a collocated FV operator and
the C-grid pressure chain `Div @ Diag @ Grad` collapses. An FV C-grid
needs the `diff` rows re-pointed (FV-D3).

**G6. Walled FV has no BC story — the one genuinely open design
problem.** Average spaces are always BC-free (they carry no BC
structure at all), while the walled spectral solve keys its trig
transforms on **BC-tagged nodal origins**: `_neumann_sibling` and
`_dirichlet_mid` both gate on `isinstance(factor, NodalSpace)`
([`pressure.py:59-132`](../../../src/fridom/nonhydro2/modules/pressure.py))
and would silently no-op on an average space. Mathematically a DCT on
cell averages is fine (cell averages of an even function are even),
but there are no BC-tagged average origins and no trig rows keyed on
them. This is FV-D4.

**G7. Dealiased (padded) transforms reject average origins** —
`NotImplementedError`, "average origins change their sinc factor under
refinement"
([`transform.py:813-830`](../../../src/fridom/spatial/operators/transform.py)).
The 2/3-rule path would regress.

**G8. `discretize` on average spaces is collocation, not quadrature.**
`Grid._discretize` has no branch on `AverageSpace`; it samples at the
evaluation nodes, which for `CellAvg` are the cell midpoints
([`grid.py:1069-1105,1464-1466`](../../../src/fridom/spatial/grid.py)).
The result is the midpoint rule **by accident** — 2nd-order correct,
but the spec's per-cell quadrature does not exist, there is no
higher-order rule, and no test pins the semantics.

**G9. `reconstruct: CellAvg -> Outer` on a bounded mesh is
deliberately ungrounded** (R1: wall faces need exterior values a
BC-free space does not define,
[`reconstruct.py:322-336`](../../../src/fridom/spatial/operators/reconstruct.py)).
A one-sided variant is designed-for.

## 4. What `nonhydro2` assumes

8071 LOC, grouped by kind of change.

**Declaration-only — zero code change under a resolver-row swap.**
Every field declaration in the package (`core.py` u/v/w/p,
`stratification.py` b/n2, the wave makers, `advection.py`,
`eigenmodes.py`) goes through `Collocated()` / `Staggered(...)` /
`Profile(...)`. A *mixed* model (some fields nodal, some average)
needs the pattern vocabulary extended — FV-D1.

**Family-agnostic algebra — works once the missing rows land.**
`state.py` (`rel_vort_z`), `smagorinsky_lilly.py`, `core.py`'s
`_project` scaffolding, `energy.py`, the `composed.py` builders, the
Krylov solver. All written in `.diff()` / `.to()` / `.retag()`; they
run on FV once **G3/G4/G5** are closed.

**Genuinely different numerics — the real work.**

| File | LOC | What breaks |
|---|---|---|
| [`modules/advection.py`](../../../src/fridom/nonhydro2/modules/advection.py) | 2295 | The private biased/WENO reconstruction rows are **nodal-only**: the biased codomain accepts `Center -> Right` / `Right -> Center` and nothing else. An FV advection module *shrinks*: it reuses `spatial/operators/weno.py` + `fallback.py` on `CellAvg -> face`. |
| [`modules/pressure.py`](../../../src/fridom/nonhydro2/modules/pressure.py) | 212 | `Div @ Diag @ Grad` collapses on `CellAvg` (G5); `_neumann_sibling`/`_dirichlet_mid` are nodal-gated (G6); the solve needs G1. |
| [`modules/mapped_pressure.py`](../../../src/fridom/nonhydro2/modules/mapped_pressure.py) | 820 | **Hard-wired nodal.** The flux-row resolution assumes the `diff` codomain is a staggered face space; the corner rows resolve four `("interpolate", ...)` rows that do not exist on averages (G4). The exact-symmetry property the CG relies on is built from those rows. |
| [`eigenmodes.py`](../../../src/fridom/nonhydro2/eigenmodes.py) + [`channel_eigenmodes.py`](../../../src/fridom/nonhydro2/channel_eigenmodes.py) + [`transforms.py`](../../../src/fridom/nonhydro2/transforms.py) + [`initial_conditions.py`](../../../src/fridom/nonhydro2/initial_conditions.py) | ~2700 | All built on the symbol kit (`kit.diff(n, on=...)`). Blocked entirely on **G1**; mechanical once symbols land. |
| [`diagnostics.py`](../../../src/fridom/nonhydro2/diagnostics.py) | 136 | `.to(center)` fails on `CellAvg` (**G3**). |

Three modules with genuinely nodal numerics, four files blocked on
symbols, one file blocked on one missing row.

## 5. Decisions

### FV-D1 — how does a declaration say "average"? *(recommendation)*

`Dof` is a closed vocabulary (`COLLOCATED`/`STAGGERED`/`CONSTANT`)
carrying no family. Two routes: **(a)** a resolver-row swap — the
family is a *grid* property, every declaration follows, but it is
all-or-nothing per grid (no `CellAvg` tracer next to a `Center` one);
**(b)** extend `SpacePattern` with a `family=` field — value-hashable,
still a valid dispatch-merge key, permits mixed models.

**Recommendation: (b), with (a) as the default.** Add `family=` to
`SpacePattern` plus an FV resolver row; a grid-level default sets the
family, a per-field `family=` overrides it. Do **not** use `SpaceRule`
(the escape hatch, [`space_patterns.py:417`](../../../src/fridom/spatial/space_patterns.py))
— it is identity-hashed and is explicitly never a dispatch key.

### FV-D2 — the staggering *(DECIDED 2026-07-12, owner: option A)*

**Scalars on `CellAvg^3`; velocities as face-normal values**
(`Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)`). The DOF is a point value in
the normal direction and a cell average transversely — i.e. exactly
the **face-area average of the normal velocity**, which is what the
divergence theorem consumes, so continuity telescopes exactly. It
closes with existing rows (probed: all four legs resolve), it is what
the operator signatures were evidently built for (`FaceDifference:
CellAvg -> Right|Inner` *is* the FV pressure gradient), and it is the
standard FV-ocean / MITgcm choice. It gives exact **mass**
conservation — the conservation that matters for an incompressible
model.

The rejected alternative (option B) puts velocities on `FaceAvg`
(dual-cell volume averages, the momentum control volume), which would
make momentum advection exactly conservative via
`DualFluxDifference`. It does **not** close today: G2 — no `diff`, no
`flux_diff` on `FaceAvg`, and no `FaceAvg -> face point value`
reconstruction, so continuity cannot even be formed. Recorded as
designed-for, not built.

**On `FaceAvg` (owner's follow-up: is it now redundant?).** No.
`Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)` is a point value in x averaged
over y and z (a *face-area* average); `FaceAvg(x) ⊗ ...` additionally
averages **across** x over the dual cell `x_i -> x_{i+1}` (a *volume*
average over the shifted momentum CV). `FaceAvg` is the dual-cell
sibling of `CellAvg` exactly as `Right` is of `Center`, and it is
where the dual measure lives. **Resolution: keep the space, invest
nothing in it.** Option A never instantiates it; it stays a dead-end
(G2) with no rows. Revisit as a deliberate deletion later if still
unused, not as a redundancy cleanup now.

### FV-D3 — how `grad`/`div`/`laplacian` stagger on FV *(recommendation)*

The builders resolve `("diff", factor)` (G5). For an FV C-grid they
must resolve `face_diff` on `CellAvg` and `flux_diff` on the face
family.

**Recommendation: an "FV C-grid registry profile"** — override
`("diff", CellAvg) -> FaceDifference()` and
`("diff", Right/Inner) -> FluxDifference()`. This is exactly parallel
to the nodal C-grid (`("diff", Center) -> Right`,
`("diff", Right) -> Center`), it is **two rows**, and because the
stencils are bitwise identical (§1) the whole existing model —
pressure chain included — then works unchanged. Keep the collocated
`FVDerivative` as the default for a *collocated* FV grid.

### FV-D4 — walls *(DECIDED 2026-07-16; design and record in §11)*

BC tags extend to the average family as **wall-value claims** (the
boundary principle unchanged); the walled FV pressure solve runs on
BC-tagged average origins (Neumann `CellAvg` → DCT-II) while the face
legs stay nodal-tagged (FV-D2 option A). Alternatives, the point-by-
point design, gates, and implementation corrections: §11.

## 6. Stages and gates

| Stage | Work | Effort | Gate |
|---|---|---|---|
| **F0 — FV symbols** (the long pole) | the four `eigenvalues` methods of §3 G1 (`FluxDifference`, `DualFluxDifference`, `FaceDifference`, `LinearReconstruction`) | **M** (~2-4 d) | `SpectralSolve` on a `CellAvg` Laplacian drives the discrete divergence to machine zero on a periodic box; each symbol matches its composed operator numerically |
| **F1 — conversion rows** | `reconstruct: CellAvg <-> Center` (same-location deconvolution, G3); `("interpolate", CellAvg/FaceAvg)` (G4) | **S-M** (~2-3 d) | `p.to(center)` works; `diagnostics.py` runs unchanged on an FV state |
| **F2 — FV tracer slice** | `SpacePattern.family=` (FV-D1b); `b` on `CellAvg`; FV flux-form advection reusing `spatial/weno.py` + `fallback.py` | **M** (~1 wk) | Exact tracer-mass conservation to machine zero over a run; FV WENO advection on periodic **and** walled axes at parity with the nodal graded path |
| **F3 — FV C-grid profile** (the actual "default") | the FV `("declared_space", mesh)` resolver + the two diff overrides (FV-D3); FV pressure chain | **S-M** (~3-5 d) | **Bitwise parity** of a linear and a nonlinear periodic nonhydro run against the nodal model |
| **F4 — walls on FV** (FV-D4, open design) | BC structure / BC-tagged origins for average spaces; walled FV pressure solve | **L** (~2-4 wk, design first) | Walled-channel pressure solve; the C3 wall closure |
| **F5 — mapped/chart FV** | average-family rows for the C1 `physical_diff` chain, the C2 metric `grad`/`div`, and `MappedPressureSolver` (820 LOC, hard-wired nodal) | **L** (~3-4 wk) | Terrain-following nonhydro on FV |
| **F6 — hygiene** | per-cell quadrature `discretize` (G8); dealiased transforms on average origins (G7); one-sided `CellAvg -> Outer` (G9) | **M** | quadrature convergence test; a 2/3-rule run on FV |

**Totals, honestly.** Periodic FV nonhydro at 2nd order, with an FV
tracer: **F0-F3 ≈ 3 weeks.** Feature parity with today's nodal model
(walls + mapped grids): **+F4+F5 ≈ 5-8 more weeks**, and F4 contains
an unresolved design question.

## 7. The benefit ledger

**What FV buys.**
- **Type-level conservation.** Flux telescoping is exact *by
  construction*; the codomain of `flux_diff` is the proof. For a
  tracer this is exact mass conservation to machine zero.
- **Cut cells.** `ImmersedDomain` already carries `CellAvg` fractions;
  cut-cell fractions as weights in the flux operators is the natural
  next rung, and it is FV-only.
- Non-oscillatory front capturing (the ENO property).

**What FV does *not* buy — the two claims to keep withdrawn.**

1. **Higher asymptotic order.** The composite tendency of a flux-form
   C-grid scheme is formally **2nd order whenever the advecting
   velocity varies along the flux axis**, and FV does not repair it:
   the reconstruction row is a deconvolution, so a two-point
   difference of face values is high-order only if the face value is
   the deconvolved *flux* `R(u q)`; the scheme forms `u_face · R(q)`,
   and the mismatch is the cross term `~ (h²/24)·2u'q'`. Measured (1D,
   periodic, uniform, exact face velocities, no velocity
   interpolation): rate 5.00 with constant `u`, **2.00** with
   `u = 1 + 0.5 sin x`. A consistent cell-average reading restores
   5.00 **in 1D only**; in genuine 2D the transverse covariance term
   `(h²/12)·∂_y u ∂_y q` caps it at ~2.0 again. Only the FD
   flux-reconstruction route (Shu-Osher; Mishra, Pares-Pulido &
   Pressel, arXiv:1905.13665, Algorithm 5) survives multi-D (measured
   4.78-4.98 in 2D), and it is **not** an FV-vs-nodal question.
   Precedent: Oceananigans' WENO is algebraically the same scheme and
   its lead developer records it as "effectively second order" on a
   staggered grid (CliMA/Oceananigans.jl#1705, closed as not worth
   pursuing); MITgcm/MOM6/ROMS share the flux form.
2. **Bounded-domain WENO.** This *was* the headline win of the
   original draft; it is no longer FV-only. The graded near-wall
   closure was factored into `spatial/operators/graded.py` and wired
   into the nodal biased schemes (merged 2026-07-12,
   `719ff4cd`), so `UpwindAdvection`/`WENOAdvection` already run on
   walled axes **without** FV.

At 2nd order on a periodic box, FV is numerically identical to what
ships today (probed: 0.0 difference). Anyone expecting better
conservation *of the discrete fields* or higher order from the switch
alone will measure neither; what changes is that conservation becomes
structural rather than incidental, and that cut cells become
reachable.

The route that would restore design-order tendencies (reconstruct the
flux `u q` rather than `q`) is orthogonal to this plan and carries its
own price: exact-zero wall flux becomes truncation-level, and
constancy preservation (`q = const` -> `q·div(u) = 0` exactly) fails
unless the **pressure projection** is changed to enforce the same wide
reconstructed divergence — a different Poisson operator. That is why
staggered ocean models do not do it.

**What is lost / made harder.**
- **Walls and mapped grids regress** until F4/F5. Both work today.
- **Dealiasing regresses** (G7).
- The eigenmode / state-transform stack (~2700 LOC) is dark until F0.
- **Spectral exactness survives**, which is the good news: the FV
  Laplacian's symbol is diagonal (`i k sinc(k dx/2)`), so
  `SpectralSolve` still applies and the pressure solve does *not* go
  iterative — once F0 lands.

## 8. Recommendation

Make the family a **model-assembly choice**
(`nh.DynamicalCore(..., family="fv"|"nodal")`, selecting the resolver
row and the diff overrides) rather than a wholesale flip. The
coexistence cost is low precisely because the model is already
family-agnostic (§4): only the three numerics modules need an FV
sibling or a family branch, and parity tests are cheap because the
2nd-order stencils are bitwise identical.

1. **F0** — land the four FV symbol rows. They are specified, they
   unblock four files, and they are the only true long pole.
2. **F1 + F2** — the FV tracer slice: a `CellAvg` tracer advected by a
   face-normal velocity (which *is* FV-D2 option A, restricted to one
   field). Small, self-contained, and it delivers exact tracer-mass
   conservation while the pressure solve — which never touches `b` —
   stays untouched.
3. **F3** — flip the default for periodic grids, gated on bitwise
   parity.
4. Re-decide F4/F5 with the tracer slice in hand.

Do **not** flip the default wholesale before F4/F5: it would trade a
validated walled + mapped model for a type-level guarantee that, at
2nd order, changes no number.

## 9. Out of scope (designed-for, not precluded)

- **Momentum-conservative FV** (velocities on `FaceAvg`, FV-D2 option
  B) — needs the `FaceAvg` differentiation rows (G2).
- Cut-cell flux weighting.
- FV shallow water (`shallowwater2`) — the same machinery applies;
  Sadourny's energy-conserving forms would need their own FV story.
- The old stack (`framework/`, `nonhydro/`) — untouched.

## 10. Implementation record — F0–F3 shipped (2026-07-16)

Four merges on `dev` (each gated on mirrored tests + ruff, reviewed
by the orchestrating session): `feat/fv-symbols` (F0),
`feat/fv-conversion-rows` (F1), `feat/fv-tracer` (F2),
`feat/fv-cgrid-default` (F3), plus `perf/fv-step-parity` (the FV/FD
step-parity guard). Gates as staged: symbols match their composed
operators to ~1e-14 and `SpectralSolve` drives the FV divergence to
machine zero (F0); `p.to(center)` works and the diagnostics land
**bitwise** on nodal values under the C-grid overrides (F1); exact
tracer-mass conservation to machine zero on periodic *and* walled
axes, FV advection tendencies bitwise nodal on periodic (F2); linear
and nonlinear periodic runs **bit-identical** to the nodal model over
12 steps, `tests/nonhydro2` green under the new default,
`shallowwater2` untouched (F3). Perf: step-suite clean vs the
pre-FV baseline (flipped cases −1.2%..+0.1%), FV/FD = 0.997..1.003
at 32³..512³ (1× A100; see `benchmarks/RESULTS.md`).

**Gap ledger now (post F4/F6, 2026-07-16):** G1, G3, G4, G5 closed
here; G6 closed by F4 (§11); G7, G8, G9 closed by F6 (§12). G2 stays
a dead-end by decision (FV-D2) — the only open gap, by choice.

**Corrections to this study, found during implementation:**

1. **§3 G1 symbol table (F0).** The table lists magnitudes only; the
   exact symbols carry the inter-origin half-cell phase
   `e^{i k δ dx}` (δ = first-node-offset difference), and at 2nd
   order `LinearReconstruction` is the plain two-point mean
   (`cos(k dx/2)`) — **no** `1/sinc` deconvolution factor (that is a
   higher-order member). The FV Laplacian's phases cancel to the
   real `−k̂²`.
2. **§3 G3 / §4 (F1).** `diagnostics.py` was *not* blocked only on
   G3: `ekin`/`epot` already worked through pre-existing rows, and
   `linear_pot_vort` needs the **G5** staggered-diff overrides (F3),
   not the deconvolution. The same-location conversion also needed
   its own dispatch kind (`"deconvolve"`) — `"reconstruct"` on
   `CellAvg` is load-bearing for the staggering hop (FVDerivative,
   C-grid `u.to(v)`), and a registry kind resolves one codomain.
3. **§5 FV-D3 "two rows" is incomplete (F3).** The pressure chain
   needs only the two `diff` overrides, but the **symbol kit** also
   needs the face→cell staggering interpolation on average-family
   fields. That cannot be a global `("interpolate", Right)` override
   (it would break nodal `.to` on mixed grids); it is inferred
   per-field in `GridSymbols._axis_symbol` (an average-family field
   routes a nodal-face `interpolate` through the `"average"` kind).
4. **§8 flip mechanism (F3).** "Selecting the resolver row" at grid
   build does not fit a passed-in bare grid: the flip is a
   **pre-freeze grid-default mutation** (`Grid.set_default_family`)
   applied by the `nh.Model` factory after `resolve_model_family`
   (auto = FV iff fully periodic, unmapped, unimmersed), and the
   C-grid diff overrides ride a new assembly hook
   (`module.grid_dispatch_overrides(grid)`). A second model on a
   grid frozen by a different-family model is a taught
   `AssemblyError` (latent re-assembly gap, closed in F3).
5. **New F4 input (F2).** Walled FV *stratified* coupling has a
   concrete blocker beyond FV-D4: `w.to(b)` from a BC-tagged
   `Inner(DIRICHLET)` face onto `CellAvg` does not resolve —
   `("average", tagged-face)` is unseeded and `LinearReconstruction`
   rejects BC-tagged domains (pinned deliberately). Walled tracer
   *advection* works (adopt-then-strip of the velocity's tag around
   the BC-free `flux_diff`); the walled mixed stratified model does
   not assemble until F4 seeds the tagged rows.
6. **Multi-device.** All F0–F3 gates ran on cpu and 1 GPU; the
   distributed solve on average origins and the 4-GPU step baseline
   are still to be validated (next 4-GPU campaign).

## 11. FV-D4 — walls on FV (decided + shipped 2026-07-16; F4 record)

**Decision: BC tags extend to the average family as wall-value
claims; the walled FV solve runs on BC-tagged average origins with
trig rows keyed on them — the nodal tagged-origin closure (the C3
seeding block, `grid.py`) reproduced for averages.** The rejected
alternative — a solve-internal family-retag of the `CellAvg`
divergence onto the nodal `Center(NEUMANN)` sibling — is
bitwise-equivalent at 2nd order but bypasses the type discipline
exactly where the codebase invests in it, leaves the walled
eigenmode/transform stack dark on FV, and hands F5 nothing.

The design, point by point:

1. **What a tag means on an average space.** The same thing it means
   on a nodal space: a wall-value claim about the *represented
   function* (boundary principle, `boundary_plan.md`). It was never a
   claim about a stored DOF — Neumann on `Center` drops no DOF
   either. `CellAvg` keeps shape `(n,)` under every tag (the nodal
   shape law degenerates: no boundary DOF is in the set); the tag's
   operational content is (i) selecting trig-transform origins and
   (ii) licensing claim-consuming rows. Average spaces stay BC-free
   by *default*; the walled machinery mints tagged siblings where it
   needs them (`mesh.average(kind, bc=...)`, mirroring `nodal()`).
   `FaceAvg` stays untaggable (dead-end by FV-D2): taught error.
2. **Trig origins.** `Cosine`/`Sine` accept tagged `CellAvg` origins;
   cell-average samples live on the cell-midpoint grid, so `CellAvg`
   maps to the **type-II kernels** exactly as `Center` does (Neumann
   `CellAvg` → DCT-II — the walled FV pressure basis; Dirichlet
   `CellAvg` → DST-II, mechanical, no consumer, unseeded). The
   coefficient layer needed nothing: `CoefficientSpace` inherits
   `origin.bc` and interns per origin identity.
3. **Symbols.** `FaceDifference`/`FluxDifference` carry the trig
   branch of the `FiniteDifference` precedent. The partners are
   **inter-family** (the face side stays nodal under FV-D2 option A,
   `fv_trig_diff_codomain`): grad leg `Cosine(CellAvg(N)) →
   Sine(Inner(D))`, symbol `−2 sin(k dz/2)/dz`; div leg the reverse,
   `+2 sin(k dz/2)/dz`; composed Laplacian the real `−k̂_z²`. **No
   sinc at 2nd order** (correction-1 pattern: the FV stencils are
   bitwise the nodal ones and the trig basis diagonalizes them
   exactly). Symbols match the analytic values to 0.0 and the
   composed field operator to <1e-12.
4. **The solve.** `_neumann_sibling` maps bounded factors of *both*
   families to their `BC.NEUMANN` siblings; `_dirichlet_mid`
   unchanged (the FV mid legs carry a nodal face factor);
   `_bc_siblings` extended to average siblings so the retag-in/out
   seam works across the family.
5. **The stratified blocker (§10 correction 5).**
   `("average", Inner(DIRICHLET))` seeded as a claim-consuming row:
   the Dirichlet tag claims the wall-face value (homogeneous: 0), the
   face→cell Gauss average consumes it at wall cells (zero-pad + the
   two-point mean), interior cells **bitwise** the BC-free row.
   Neumann-tagged face domains stay unseeded with a taught hint (a
   Neumann tag claims no wall value).
6. **Model gates.** Explicit `family="fv"` served on walled unmapped,
   unimmersed grids; mapped/immersed stay a taught error (F5). The
   walled auto-default initially stayed nodal pending the owner's
   call; **ruled the same day — see the addendum below: auto = FV iff
   unmapped and unimmersed.**
7. **The F4→F5 seam.** Walls enter FV chains only through (i) tagged
   average origins for transforms and (ii) nodal face factors
   carrying claims plus structural Inner-codomain Gauss closures. The
   mapped solver consumes (ii) unchanged.

**Shipped** (merge `7449c15f`; spatial/nonhydro2/tests commits
`2ad2c7b4`, `af8cfd8a`, `f75cc0a4`; import follow-up `ab90e4bb` after
the advection rehome landed mid-flight). Gates: walled manufactured
Poisson <1e-12; walled-z/x/y FV projection drives the discrete
divergence to machine zero; the solver runs on the Neumann *average*
sibling and the periodic path never retags; stratified walled FV
assembles (`w.to(b)` resolves) and conserves total buoyancy ~1e-18;
forced-4 FV siblings of the distributed walled-z/walled-x fast paths
assert plan resolution, wall-axis locality, and device-count
invariance; the nodal walled suite untouched (805 passed on the
branch, re-verified on `dev` post-merge).

**Corrections found during implementation:**

1. **Bitwise parity is exact eagerly, not under jit.** The
   unconstrained tendency and `constrain` are **exactly**
   bit-identical to nodal under `jax.disable_jit()` (0.0 — the
   stencil-identity claim, pinned as a test). The jitted 12-step
   walled run differs by ≤1.2e-14: XLA fuses the mixed
   `Fourier ⊗ Cosine` solve with a different float ordering than the
   nodal HLO. Periodic stays bit-identical (all-Fourier → identical
   HLO). The jitted gate is therefore a tight tolerance (<1e-12),
   not equality.
2. **`FluxDifference.codomain` must accept `Inner(DIRICHLET)`**: the
   *physical* divergence applies it to the walled velocity directly
   (not only through seeded rows); the homogeneous no-normal-flow
   claim is exactly the zero wall flux the Inner branch imposes.
   Neumann-tagged: taught error.
3. **`fv_cgrid_overrides` must key the tagged origins too** —
   `CellAvg(NEUMANN) → FaceDifference` and `Inner(DIRICHLET) →
   FluxDifference` — else the base seeding leaves the tagged
   `CellAvg` on the collocated `FVDerivative`.
4. `fv_trig_diff_codomain` raises `EigenbasisError` (not
   `SpaceMismatchError`) on non-FV trig factors, keeping the
   `eigenvalues` contract.
5. Only Neumann `CellAvg` is seeded (per the design); the DST-II
   Dirichlet variant exists mechanically behind the shared kernel
   path.

**F4 → F5 handoff notes:** the metric-aware rows on averages (C1/C2
chain) are still absent; `_reconstruct_walled_face` uses a uniform
zero-pad + two-point mean — the mapped variant needs measure
weighting (mirror `FluxDifference`'s mapped Inner branch); the
`_require_fv_capable` mapped/immersed taught error is the single
remaining capability gate.

## 12. F6 — hygiene record (shipped 2026-07-16)

- **G7 — dealiased padded transforms on average origins** (merge
  `d778e7c8`). The padded Fourier kernels bracket the embed/trim with
  the cell-averaging `sinc(k dx/2)` diagonal: deconvolve on the
  source spectrum, refine (zero-embed with the inter-origin phase),
  reconvolve on the target spectrum — `sinc ≥ 2/π` on the stored
  band, so every division is exact; `_sibling_origin` names the
  refined average sibling. `FaceAvg` falls out free (Fourier is
  periodic-only, dual width = dx). Round-trips ~1e-16, analytic
  cell-average refinement ~1e-15, alias-free products exact. Note:
  **no model-level 2/3-rule consumer exists today** (no spectral
  advection scheme instantiates a padded transform) — G7 is the
  operator-level capability, smoke-tested on an FV-default grid.
- **G8 — per-cell quadrature `discretize`** (merge `d9fd2d58`).
  `create_field(order=)` sets the per-cell Gauss–Legendre point count
  on average factors; `None`/`1` remain the midpoint shortcut
  **bitwise** (midpoint *is* 1-point Gauss, so the F2/F3 parity gates
  are untouched); `order >= 2` is genuine per-cell quadrature — exact
  to degree `2·order − 1`, per-cell edges honor `coordinate_map`
  (stretched axes quadrature each cell on its own width), mixed
  spaces average only their average axes. Semantics on chart-mapped
  grids are the **chart**-cell average (Jacobian-weighted physical
  volumes are F5 territory, documented); `FaceAvg` higher-order is
  guarded (the dual-cell edge seam does not exist).
- **G9 — one-sided `CellAvg → Outer` wall reconstruction** (merge
  `5687d5c3`). `LinearReconstruction(target=NodeSet.OUTER,
  boundary="one_sided")` — a per-instance opt-in on the nodal
  `LinearInterp` one-sided precedent, never a default row (the
  `("reconstruct", CellAvg)` kind keeps its staggering codomain).
  Interior faces stay bitwise the `CellAvg → Inner` mean; wall faces
  get geometry-derived one-sided weights (exact-rational
  `(3c₀ − c₁)/2` on uniform axes, value-moment solve on mapped ones);
  `layout="local"` required on the applied axis; R1 remains for the
  closed default. Subtlety recorded for F5: the interior symmetric
  mean is computational-uniform (degree-0-exact on stretched axes)
  while the patched walls are degree-1-exact — both 2nd-order
  convergent; making the interior geometry-aware would be a separate
  change to the symmetric kernel.

### §11 addendum — the walled auto-default flip (owner ruling 2026-07-16)

Ruled by Silvano the same day F4 shipped: **the auto family default is
FV on every grid that can carry it — periodic and walled; only mapped
and immersed grids stay nodal** (`_fv_capable` = unmapped ∧
unimmersed, exactly the complement of `_require_fv_capable`'s
rejection). Shipped as the `feat/fv-walled-default` merge: the gate,
the docstring sweep (`core.py`, `model.py`, `eigenmodes.py`), the
walled test sweep (nodal coverage pinned `family="nodal"` where the
nodal path is the test's purpose — the F3-flip precedent), and
explicit nodal siblings for the walled step-benchmark cases
(`nh_flat_walled_nodal` / `nh_flat_walled_x_nodal`; committed
baselines untouched — re-record on the next GPU campaign).

**Discovery pinned during the flip — the walled-FV analytic
eigenmode gap.** Under a walled-FV default, the analytic
walled-vertical eigenmode kit (`Eigenmodes` / `from_model` with a
bounded vertical) fails at kit construction: it declares BC-tagged
collocated analysis spaces, and the FV declared-space resolver
rejects a BC on a collocated coordinate (`ValueError`, "average
factors carry no boundary structure, C8" — fires naturally, pinned in
`test_fv_default.py::test_walled_fv_eigenmodes_are_a_taught_gap`).
Probing showed the gap is the whole walled-FV transform stack, not a
row: missing are (a) a declared-space story for BC-tagged average
analysis spaces, (b) the DST-II `Dirichlet CellAvg` transform row
(unseeded by design, §11.2), and (c) the reconstruction/interp trig
eigenvalues on tagged average origins (the kit's `a`/`ab` symbol
families). F5-adjacent; the numeric channel eigenbasis
(`ChannelEigenmodes`) is family-agnostic and works on FV. A bespoke
`from_model` guard naming the deferral would improve the UX
(follow-up). Also flagged: `MeridionalStratification`'s `n2` Profile
carries no `family=` and follows the grid default onto `CellAvg` on
walled-FV models (assembles and passes; whether `n2` should pin nodal
independent of family is an open owner flag).
