---
status: draft
date: 2026-07-12
---

# FV nonhydro — scoping study

**Question (owner, 2026-07-12):** make the nonhydro model a
**finite-volume model by default** — prognostic variables on the
average family (`CellAvg`/`FaceAvg`) rather than the nodal family
(`Center`/`Right`/`Outer`/`Inner`). *How much would we have to
change?*

This is a scoping study, not an implementation plan: it reports the
current-state seams with evidence, names the decisions the owner must
make, and stages the work. No code was changed.

## 1. Headline finding

**Two facts dominate the estimate.**

**(1) The model layer is already family-agnostic.** `nonhydro2`
names a concrete function space almost nowhere. Field declarations go
through the grid-free `SpacePattern` vocabulary
(`Collocated()` / `Staggered("x")` / `Profile("z")`,
[`core.py:97-107`](../../../src/fridom/nonhydro2/modules/core.py)),
whose `Dof` tags (`COLLOCATED` / `STAGGERED` / `CONSTANT`,
[`space_patterns.py:41-58`](../../../src/fridom/spatial/space_patterns.py))
carry **no family**. Which family a tag lands in is decided by a
single per-mesh registry row, `("declared_space", mesh)`
([`grid.py:1915-1937`](../../../src/fridom/spatial/grid.py)), and
those rows are grid-level seedable
([`space_patterns.py:640-650`](../../../src/fridom/spatial/space_patterns.py)).
Swapping `COLLOCATED -> CellAvg` is **one resolver row**, and every
field declaration in the package follows for free.

**(2) At 2nd order, the FV and nodal C-grid stencils are the same
numbers.** Probed directly on a periodic 16-cell mesh: the divergence
leg (`FluxDifference: Right -> CellAvg`) and the nodal
`FiniteDifference: Right -> Center` agree to **0.0**; the gradient leg
(`FaceDifference: CellAvg -> Right`) and the nodal
`FiniteDifference: Center -> Right` agree to **0.0**. Same stencil,
different codomain type. So "switch the nonhydro to FV at 2nd order"
is, numerically, a **retag** — and bitwise parity with the validated
nodal model is the right acceptance gate, not an error-norm
comparison.

**The honest consequence:** the switch is *cheap where it is
uninteresting* (2nd-order periodic: a registry profile) and
*expensive exactly where the payoff is* (high-order on bounded
domains, walls, mapped grids). The long pole is not the model, it is
**four missing `eigenvalues` methods** on the FV operators, which the
spectral pressure solve, the eigenmode stack, and the state transforms
all sit on top of.

## 2. Current state — what the FV machinery already provides

Substantially more than expected. The average family is **complete at
the geometry layer and the flux layer**.

| Capability | Status | Evidence |
|---|---|---|
| `CellAvg` / `FaceAvg` spaces, shapes | done | [`spaces/average.py:32,42`](../../../src/fridom/spatial/spaces/average.py) |
| `flux_diff` (exact Gauss): `Outer`/`Inner` -> `CellAvg`, `Right` -> `CellAvg` (periodic) | done | [`flux_diff.py:158-198`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `DualFluxDifference`: `Center`/`CellAvg` -> `FaceAvg` | done | [`flux_diff.py:291-320`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `FaceDifference` (FV pressure gradient): `CellAvg` -> `Right`\|`Inner` | done | [`flux_diff.py:383-403`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `("diff", CellAvg)` = `FVDerivative` = `flux_diff @ reconstruct` | done | [`grid.py:1827-1828`](../../../src/fridom/spatial/grid.py), [`flux_diff.py:443-474`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `reconstruct` rows: `CellAvg` -> `Right`/`Inner`; `FaceAvg` -> `Center`; `Right`/`Outer`/`Inner` -> `CellAvg`; `Center` -> `FaceAvg` | done | [`reconstruct.py:342-350`](../../../src/fridom/spatial/operators/reconstruct.py) |
| WENO + graded `Fallback` on the FV family, **bounded-capable** | done | `operators/weno.py`, `operators/fallback.py` |
| Measures on `CellAvg`/`FaceAvg`, incl. **mapped** meshes; `flux_diff` divides by the codomain measure | done (C0) | [`grid.py:1582-1586,1626-1641`](../../../src/fridom/spatial/grid.py), [`flux_diff.py:118-127,252-262`](../../../src/fridom/spatial/operators/flux_diff.py) |
| `integrate` **exact** on `CellAvg` (§3.13) | done | [`integrate.py:132`](../../../src/fridom/spatial/operators/integrate.py), test `test_integrate.py:151` |
| Average-origin coefficient spaces + the §3.2 `sinc(k dx/2)` factor (`SincShift`) | done | [`spectral.py:611-757`](../../../src/fridom/spatial/operators/spectral.py) (`jnp.sinc` at `:721`), [`structured_1d.py:508-519`](../../../src/fridom/spatial/meshes/structured_1d.py) |
| FFT on average origins | done | [`fourier.py:14-15,107-124`](../../../src/fridom/spatial/operators/fourier.py) |
| Export: averages carry `representation="cell_mean"`, labels not positions | done | [`export.py:108-111,180-181`](../../../src/fridom/spatial/export.py) |
| `ImmersedDomain` cut-cell fractions on `Center`/`CellAvg` | done | [`immersed_domain.py:369-375,555-560`](../../../src/fridom/spatial/immersed_domain.py) |
| `grid.random.normal` on average spaces | done | [`random_fields.py:62-103`](../../../src/fridom/spatial/random_fields.py) |
| `ConjugateGradient` (C3 PCG) — space-agnostic, measure-weighted | done | [`krylov.py:124-150,224-247`](../../../src/fridom/spatial/operators/krylov.py) |

### 2.1 The gaps — nine of them, ranked

**G1 (long pole). No `eigenvalues` on any FV operator.**
`FluxDifference`, `DualFluxDifference`, `FaceDifference`, and
`LinearReconstruction` all inherit the raising base
([`base.py:225-254`](../../../src/fridom/spatial/operators/base.py));
probed: `EigenbasisError: FluxDifference has no eigenvalues on
Fourier(x, origin=CellAvg)`. The FV `Laplacian` therefore **cannot
even form its symbol**, so `SpectralSolve` cannot run
([`spectral_solve.py:102-114`](../../../src/fridom/spatial/operators/spectral_solve.py)).
*Mitigating:* the four rows are already **specified with closed
forms** in the active
[`operator_symbols_plan.md:124-127`](operator_symbols_plan.md)
(`i k sinc(k dx/2)` and friends), and every ingredient exists
(origin-agnostic `fourier_wavenumbers`, `diagonal_symbol`,
average-origin Fourier spaces). This is fill-in, not research.

**G2. `FaceAvg` is a dead-end space.** No `("diff", FaceAvg)` and no
`("flux_diff", FaceAvg)` row exists (probed: `DispatchError`). You can
*produce* a `FaceAvg` field and reconstruct it to `Center`, but you
cannot differentiate it. This kills the "velocities on `FaceAvg`"
staggering (FV-D2 below) unless new rows are written.

**G3. No same-location deconvolution row: `CellAvg -> Center`.** The
registered `reconstruct` row on `CellAvg` lands on `Right` (a *shift*),
so `p.to(center)` **fails** (probed: `SpaceMismatchError`). This breaks
`diagnostics.py` ([`:83-85,103,126-128`](../../../src/fridom/nonhydro2/diagnostics.py)),
which interpolates every staggered quantity onto the pressure cell.

**G4. No `("interpolate", CellAvg/FaceAvg)` rows** on physical spaces
([`grid.py:1808`](../../../src/fridom/spatial/grid.py) seeds
`interpolate` on `nodal + tagged` only). This blocks C1's
`physical_diff` correction chain, C2's metric cross-terms /
`RaiseIndex` / `LowerIndex`
([`composed.py:1039,1063,1077,1110`](../../../src/fridom/spatial/operators/composed.py)),
and C3's corner rows.

**G5. The vector-calculus builders resolve `("diff", factor)`**
([`composed.py:443`](../../../src/fridom/spatial/operators/composed.py)).
On `CellAvg` that resolves to the **collocated** `FVDerivative`
(`CellAvg -> CellAvg`), not a staggered face gradient — so
`grad`/`div`/`laplacian` silently produce a *collocated* FV operator,
and the C-grid pressure chain `Div @ Diag @ Grad` collapses. An FV
C-grid needs the `diff` rows re-pointed (see FV-D3).

**G6. Walled FV has no BC story.** Average spaces are **always
BC-free** (they carry no BC structure at all;
[`immersed_domain.py:553`](../../../src/fridom/spatial/immersed_domain.py)),
while the walled spectral solve keys its trig transforms on **BC-tagged
nodal origins** — `_neumann_sibling` / `_dirichlet_mid` both gate on
`isinstance(factor, NodalSpace)`
([`pressure.py:60-98,126-132`](../../../src/fridom/nonhydro2/modules/pressure.py))
and would silently no-op on an average space. This is the one
genuinely **open design problem**, not a fill-in.

**G7. Dealiased (padded) transforms reject average origins** —
`NotImplementedError`, "average origins change their sinc factor under
refinement"
([`transform.py:813-830`](../../../src/fridom/spatial/operators/transform.py)).
The 2/3-rule path would regress.

**G8. `discretize` on average spaces is collocation, not quadrature.**
`Grid._discretize` ([`grid.py:1069-1105`](../../../src/fridom/spatial/grid.py))
has **no branch on `AverageSpace`**; it samples at the evaluation
nodes, which for `CellAvg` are the cell midpoints
([`grid.py:1464-1466`](../../../src/fridom/spatial/grid.py)). The
result is the midpoint rule **by accident** — 2nd-order correct, but
spec §3.10's per-cell quadrature does not exist, there is no
higher-order rule, and no test pins the semantics (the exactness test
routes through `data=`, `test_integrate.py:151`).

**G9. `reconstruct: CellAvg -> Outer` on a bounded mesh is
deliberately ungrounded** (R1: wall faces need exterior values a
BC-free space does not define,
[`reconstruct.py:322-336`](../../../src/fridom/spatial/operators/reconstruct.py)).
A one-sided variant is designed-for.

## 3. Current state — what `nonhydro2` assumes

7380 LOC. Grouped by **kind of change**:

### 3.1 Declaration-only (zero code change if the resolver row swaps)

Every field declaration in the package: `core.py:97-107` (u, v, w, p),
`stratification.py:72,154,157` (b, n2),
`polarized_wave_maker.py:65-68`, `gaussian_wave_maker.py:125-126`,
`advection.py:873`, `eigenmodes.py:216-234`. All go through
`Collocated()` / `Staggered(...)` / `Profile(...)`. **These do not have
to change at all** under a resolver-row swap — though a *mixed* model
(some fields nodal, some average) needs the pattern vocabulary
extended (FV-D1).

### 3.2 Family-agnostic algebra (works, given the missing rows)

`state.py:83-85` (`rel_vort_z`), `smagorinsky_lilly.py:298-365`,
`core.py:_project` scaffolding, `energy.py`, the `composed.py`
builders, `krylov.py`. These are written in `.diff()` / `.to()` /
`.retag()` and will run on FV **once G3/G4/G5 are closed**.

### 3.3 Genuinely different numerics (the real work)

| File | LOC | What breaks |
|---|---|---|
| [`modules/advection.py`](../../../src/fridom/nonhydro2/modules/advection.py) | 1788 | The private biased/WENO reconstruction rows are **nodal-only**: `_biased_codomain` accepts `Center -> Right` / `Right -> Center` and nothing else (`:474-521`), `_CenteredFaceInterpolation` (`:526`), upwind/WENO (`:629-780`). They raise taught errors on walled *and* mapped grids (`:923,959,1008`). An FV advection module **shrinks**: it reuses `spatial/operators/weno.py` + `fallback.py` on `CellAvg -> face`, which already handle walls. |
| [`modules/pressure.py`](../../../src/fridom/nonhydro2/modules/pressure.py) | 212 | `Div @ Diag @ Grad` collapses on `CellAvg` (G5); `_neumann_sibling`/`_dirichlet_mid` are nodal-gated (G6); the solve needs G1. |
| [`modules/mapped_pressure.py`](../../../src/fridom/nonhydro2/modules/mapped_pressure.py) | 660 | **Hard-wired nodal.** `_resolve_flux_rows` (`:263-270`) assumes the `diff` codomain is a staggered face space; `_resolve_corner_rows` (`:272-308`) resolves four `("interpolate", ...)` rows that do not exist on averages (G4). The exact-symmetry property the CG relies on is built from those rows. |
| [`eigenmodes.py`](../../../src/fridom/nonhydro2/eigenmodes.py) + [`channel_eigenmodes.py`](../../../src/fridom/nonhydro2/channel_eigenmodes.py) + [`transforms.py`](../../../src/fridom/nonhydro2/transforms.py) + [`initial_conditions.py`](../../../src/fridom/nonhydro2/initial_conditions.py) | 2695 | All built on the symbol kit (`kit.diff(n, on=...)`, `eigenmodes.py:241-243`). Blocked entirely on **G1**; mechanical once symbols land. |
| [`diagnostics.py`](../../../src/fridom/nonhydro2/diagnostics.py) | 136 | `.to(center)` fails on `CellAvg` (**G3**). |

**Count:** 3 modules with genuinely nodal numerics (advection,
pressure, mapped_pressure), 4 files blocked on symbols, 1 file blocked
on one missing row.

## 4. Decisions needed

### FV-D1 — how does a declaration say "average"?

`Dof` is a closed vocabulary (`COLLOCATED`/`STAGGERED`/`CONSTANT`) and
carries no family. Two routes:

- **(a) resolver-row swap** — the family is a *grid* property; every
  declaration follows. Cheapest, but **all-or-nothing per grid**: you
  cannot have a `CellAvg` tracer next to a `Center` one.
- **(b) extend `SpacePattern` with a `family=` field** — value-hashable,
  stays a valid dispatch-merge key, permits mixed models.

**Recommendation: (b), with (a) as the default.** Add `family=` to
`SpacePattern` and an FV resolver row; a grid-level default sets the
family, a per-field `family=` overrides it. Do **not** use `SpaceRule`
(the escape hatch, `space_patterns.py:417`) — it is identity-hashed and
is explicitly never a dispatch key.

### FV-D2 — the staggering: where do velocities live? *(the key decision)*

> **DECIDED 2026-07-12 (owner): option A.** Scalars on `CellAvg^3`,
> velocities as face-normal values (`Right(x) ⊗ CellAvg(y) ⊗
> CellAvg(z)`). The move to FV is committed (ROADMAP 3.5); the
> sequencing recommendation of §5 stands (symbol rows → conversion
> rows → the FV **tracer slice** → the C-grid profile behind a
> bitwise-parity gate; no wholesale default flip while walls and
> mapped grids would regress).
>
> **On dropping `FaceAvg` (owner's question).** `Right(x) ⊗
> CellAvg(y) ⊗ CellAvg(z)` is **not** the same object as
> `FaceAvg(x) ⊗ CellAvg(y) ⊗ CellAvg(z)`, so `FaceAvg` is not
> redundant with option A:
>
> - `Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)` is a **point value in x**
>   (at the face plane) **averaged over y and z** — the *face-area*
>   average `(1/ΔyΔz) ∫∫ u dy dz`. This is what the divergence
>   theorem consumes, which is why continuity telescopes exactly.
> - `FaceAvg(x) ⊗ ...` additionally averages **across x**, over the
>   dual cell `x_i → x_{i+1}` — a *volume* average over the shifted
>   (momentum) control volume. It is what a fully-FV **momentum**
>   equation on the staggered CV would want.
>
> The difference is whether the face-normal direction is averaged
> over. `FaceAvg` is the dual-cell sibling of `CellAvg` exactly as
> `Right` is the dual-cell sibling of `Center` (§3.9's family
> symmetry), and it is also where the **dual measure** lives
> (§2.7). **Resolution: keep the space, invest nothing in it.**
> Option A never instantiates it; it stays a dead-end (G2) with no
> rows. Deleting it would break the family symmetry and the measure
> story for a space that costs nothing to leave unused. Revisit as a
> deliberate deletion later if it is still unused, not as a
> redundancy cleanup now.

- **Option A — face-normal point values.** Scalars on
  `CellAvg ⊗ CellAvg ⊗ CellAvg`; `u` on
  `Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)`. The DOF is a point value in the
  normal direction and a cell average transversely — i.e. **exactly the
  face-area average of the normal velocity**, which is what the
  divergence theorem wants. **Closes with existing rows** (probed: all
  four legs OK). This is the standard FV-ocean / MITgcm choice.
- **Option B — velocities on `FaceAvg`** (dual-cell averages, the
  momentum control volume). Momentum advection becomes exactly
  conservative (`DualFluxDifference`). **Does not close today**: G2 —
  no `diff`, no `flux_diff` on `FaceAvg`, and no `FaceAvg -> face
  point value` reconstruction, so **continuity cannot be formed**.
  Needs ≥3 new operator rows plus a deconvolution family.

**Recommendation: Option A.** It is the coherent, closed choice, it is
what the operator signatures were evidently built for
(`FaceDifference: CellAvg -> Right|Inner` is *the FV pressure gradient*,
`flux_diff.py:362-375`), and it gives exact **mass** conservation —
which is the conservation that matters for an incompressible model.
Option B buys exact *momentum* conservation and should be recorded as
designed-for, not built now. Note that under Option A, `FaceAvg` is
**unused by the nonhydro model**.

### FV-D3 — how do `grad`/`div`/`laplacian` stagger on FV?

The builders resolve `("diff", factor)` (G5). For an FV C-grid they must
resolve `face_diff` on `CellAvg` and `flux_diff` on the face family.

**Recommendation: an "FV C-grid registry profile"** — override
`("diff", CellAvg) -> FaceDifference()` and
`("diff", Right/Inner) -> FluxDifference()`. This is precisely parallel
to the nodal C-grid (`("diff", Center) -> Right`,
`("diff", Right) -> Center`), it is **two rows**, and because the
stencils are bitwise identical (§1), the whole existing model — pressure
chain included — then works unchanged. Keep the collocated
`FVDerivative` as the default for a *collocated* FV grid.

### FV-D4 — walls (G6)

Average spaces carry no BC structure; the walled spectral solve needs
BC-tagged origins. Mathematically a DCT on cell averages is fine (cell
averages of an even function are even), but the machinery has no
BC-tagged average origins and no trig rows keyed on them.

**Recommendation: defer, and scope it separately.** Do not let the FV
switch regress the working walled nodal model. Until FV-D4 is answered,
**FV is periodic-only**.

## 5. Migration options

**(a) Wholesale switch.** Cheap in code (FV-D1a + FV-D3 = one resolver
row + two overrides) but **forfeits walls (C3) and mapped grids
(C0–C4)** until G6 and G4 land. The nodal model has both working today.
**Not recommended as a first move** — it trades a validated capability
for a type-level one.

**(c) FV tracers only.** As stated ("keep velocities nodal") this looks
like a compromise, but **it is not a separate strategy**: a `CellAvg`
tracer advected by a face-normal velocity **is** Option A. And at 2nd
order the nodal `Center` tracer and the `CellAvg` tracer differ by a
retag. So (c) is not a third option — it is **the cheapest coherent
slice of (b)**, and a genuinely attractive first stage: retag `b` (and
only `b`) to `CellAvg`, gaining exact tracer conservation and the
**bounded-domain WENO path** (which the nodal biased family cannot do
at all, `advection.py:923`), while the pressure solve — which never
touches `b` — is untouched.

**(b) Family as a model-assembly choice (RECOMMENDED).**
`nh.DynamicalCore(..., family="fv"|"nodal")` selecting the resolver row
and the diff overrides. The coexistence cost is low **because the model
is already family-agnostic** (§3.1/3.2): only the 3 numerics modules
need an FV sibling or a family branch. Tests parametrize over the
family for those modules; parity tests are cheap because 2nd-order
stencils are bitwise identical.

## 6. Stages and gates

| Stage | Work | Effort | Gate |
|---|---|---|---|
| **F0 — FV symbols** (the long pole) | 4 `eigenvalues` methods (`FluxDifference`, `DualFluxDifference`, `FaceDifference`, `LinearReconstruction`); closed forms already in [`operator_symbols_plan.md:124-127`](operator_symbols_plan.md) | **M** (~2-4 d) | `SpectralSolve` on a `CellAvg` Laplacian drives the discrete divergence to machine zero on a periodic box; the symbol matches the composed `flux_diff @ reconstruct` numerically |
| **F1 — conversion rows** | `reconstruct: CellAvg <-> Center` (same-location deconvolution, G3); `("interpolate", CellAvg/FaceAvg)` (G4) | **S-M** (~2-3 d) | `p.to(center)` works; `diagnostics.py` runs unchanged on an FV state |
| **F2 — FV tracer slice** (option (c)) | `SpacePattern.family=` (FV-D1b); `b` on `CellAvg`; FV flux-form advection reusing `spatial/weno.py` + `fallback.py` | **M** (~1 wk) | Exact tracer-mass conservation to machine zero over a run; **WENO advection on a walled domain** — a case the nodal family cannot run |
| **F3 — FV C-grid profile** (the actual "default") | FV `("declared_space", mesh)` resolver + the 2 diff overrides (FV-D3); FV pressure chain | **S-M** (~3-5 d) | **Bitwise parity** of a linear + a nonlinear periodic nonhydro run against the nodal model |
| **F4 — walls on FV** (FV-D4, open design) | BC structure / BC-tagged origins for average spaces; walled FV pressure solve | **L** (~2-4 wk, design first) | Walled-channel pressure solve; the C3 wall closure |
| **F5 — mapped/chart FV** | Average-family rows for C1 `physical_diff`, C2 metric `grad`/`div`, C3 `MappedPressureSolver` (660 LOC, currently hard-wired nodal) | **L** (~3-4 wk) | Terrain-following nonhydro on FV |
| **F6 — hygiene** | Per-cell quadrature `discretize` (G8); dealiased transforms on average origins (G7); one-sided `CellAvg -> Outer` (G9) | **M** | §3.10 quadrature convergence test; 2/3-rule run on FV |

**Totals, honestly:**
- **Periodic FV nonhydro, 2nd order + bounded WENO tracers: F0-F3 ≈ 3 weeks.**
- **Feature parity with today's nodal model (walls + mapped): +F4+F5 ≈ 5-8 more weeks**, and F4 contains an unresolved design question.

## 7. Benefits — and what is lost

**Real benefits.**
- **Type-level conservation** (§3.9): flux telescoping is exact *by
  construction*; the codomain of `flux_diff` is the proof.
- **High order on bounded domains.** The FV WENO/`Fallback` path
  already handles walls; the nodal biased family raises a taught error
  on walled grids (`advection.py:923`) and on mapped grids (`:959`).
  **This is the single largest concrete win** and it is available at
  stage F2.
- **Cut cells** (§3.7): `ImmersedDomain` already carries `CellAvg`
  fractions (`immersed_domain.py:369-375`); cut-cell fractions as
  weights in flux operators is the natural next rung, and it is FV-only.
- Shock/front capturing follows from the above.

**What is *not* a benefit.** At 2nd order on a periodic box, FV is
**numerically identical** to what ships today (probed: 0.0 difference).
Anyone expecting better conservation from the switch alone will not
measure any.

**What is lost / made harder.**
- **Spectral exactness survives** — this is the good news. The FV
  Laplacian's symbol is diagonal (`i k sinc(k dx/2)`), so
  `SpectralSolve` still applies and the pressure solve does **not** go
  iterative. But it is blocked on F0.
- **Walls and mapped grids regress** until F4/F5. Today they work.
- **Dealiasing regresses** (G7).
- The eigenmode / state-transform stack (2695 LOC) is dark until F0.

## 8. Recommendation

Do **(b)**, staged, starting with the tracer slice:

1. **F0** — land the four FV symbol rows. They are already specified,
   they unblock four files, and they are the only true long pole.
2. **F1 + F2** — the FV tracer slice. Small, self-contained, and it
   delivers the one thing the nodal stack genuinely cannot do:
   high-order bounded-domain advection.
3. **F3** — flip the default for periodic grids, gated on bitwise
   parity.
4. Re-decide F4/F5 with the tracer slice in hand.

Do **not** flip the default wholesale before F4/F5: it would trade a
validated walled + mapped model for a type-level guarantee that, at
2nd order, changes no number.

## 9. Out of scope (designed-for, not precluded)

- **Momentum-conservative FV** (velocities on `FaceAvg`, FV-D2 option
  B) — needs the `FaceAvg` differentiation rows (G2).
- Cut-cell flux weighting (§3.7 point 2).
- FV shallow water (`shallowwater2`) — the same machinery applies;
  Sadourny's energy-conserving forms would need their own FV story.
- The old stack (`framework/`, `nonhydro/`) — untouched.
