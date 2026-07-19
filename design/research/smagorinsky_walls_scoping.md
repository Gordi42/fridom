---
status: complete
date: 2026-07-19
---

# Walled Smagorinsky–Lilly — deep design research

Owner-requested research (2026-07-19): walled Smagorinsky is
important ("most setups will have vertical boundaries"); how should
it be done? Survey of reference models + literature, mapped onto the
landed walled diffusion/friction machinery.

**Bottom line.** The walled Smagorinsky lift maps almost entirely
onto the *already-landed* walled diffusion/friction machinery. The
one genuinely new object versus linear friction is the
**off-diagonal shear strain at the wall-adjacent edge/corner** — and
every ocean C-grid code, plus Griffies–Hallberg's theory, confirms
this is the *sole* place the slip choice enters (the diagonal
"tension" strains are structural). Free-slip walled Smagorinsky is
the `Inner[Dirichlet]` **retag** of the diffusion campaign's §3.1,
applied to the strain instead of the flux and reused in both `|Σ|`
and the stress — no new spatial operator. No-slip is the same §3.3
wall-row story, with one Smagorinsky-specific twist. Van Driest
damping is **omitted**. The uniform-mesh rejection is orthogonal to
walls and falls in its own stage (W3).

**Owner rulings (2026-07-19), recorded up front:**

- Default `slip="free"` (inherits the friction closures' ratified
  convention).
- Full scope W1 → W2 → W3 lands pre-docs.
- No-slip spelling: **(a) wall-row correction** (parity with the
  diffusion campaign); the `Center[Dirichlet]→Outer` diff row is
  recorded as the refactor to take if a third wall-stress consumer
  appears.
- Van Driest wall damping: **omit**. Scotti anisotropy factor on Δ:
  **omit** (bare local mean, as Oceananigans/MOM6).
- Long-term slip ownership (per-closure kwarg vs per-wall grid
  declaration): **parked** (deferred.md) until a third wall-stress
  consumer exists.
- Immersed and terrain/mapped Smagorinsky stay deferred (§vi).

## (i) Survey: what each reference model does at a wall

| Model | Near-wall strain | Free-slip | No-slip | Default | Filter width Δ | Wall damping |
|---|---|---|---|---|---|---|
| **Oceananigans.jl** | Plain staggered `δ` operators over **BC-filled halos**; no one-sided stencils | tangential `NoFlux` = even mirror → shear strain 0 at wall | opt-in `Value(0)` = odd mirror → one-sided gradient `2(cᵇ−cⁱ)/Δ` | **free-slip** | `(ΔxΔyΔz)^{1/3}` local; **no** wall-cell correction | **none** (dynamic coefficient is their near-wall substitute) |
| **PALM / ABL-LES** (Moeng, Sullivan et al.) | Wall strain **not resolved** — Monin–Obukhov surface-stress matching | zero stress | MOST wall-stress BC | MOST stress | Deardorff cbrt, capped `min(Δ,1.8z)`/`min(l,κz)` near wall | length-cap "rarely invoked"; van Driest essentially never |
| **MITgcm** | Tension `D_T` at tracer pt; shear `D_S` at **vorticity (corner) pt** | **mask** `τ₁₂` corner stress to 0 | interior stencil clean, **wall drag injected as body force** (`sideDragFactor=2`) | **no-slip** | `L²=(viscC2Smag/π)²·area` | none |
| **MOM6** | `sh_xx` at h-pts, `sh_xy` at **q pts**, "computed separately to permit distinct BC application" | land mask zeros `sh_xy` at wall q-pt | `NOSLIP`: `sh_xy` one-sided | **free-slip** | `Δ²=2Δx²Δy²/(Δx²+Δy²)` | none |
| **NEMO** | tension@T (`tmask`), shear@F (`fmask`) | `rn_shlat=0` → coastal shear zeroed | `rn_shlat=2` → reflected ghost; continuous partial slip 0–2 | user choice | `(rn_csmc/π)²·area` | none |
| **Griffies–Hallberg 2000** (theory) | C-grid: `D_S` at vorticity pt, `D_T` at thickness pt | "**`D_S` vanishes on all boundaries**"; boundary term in the energy identity dropped legitimately *because* masking zeroes it | B-grid no-slip | free-slip (C-grid) | — | — |

**Convergent facts that decide the design.** (a) The wall BC is
entirely a statement about the off-diagonal shear strain — universal
across all C-grid codes and stated verbatim by Griffies–Hallberg.
(b) Free-slip = zero the corner shear (structural); no-slip =
one-sided/reflected shear or injected wall drag — **exactly**
FRIDOM's shipped Dirichlet-retag and factor-2-ghost wall correction.
(c) The C-grid free-slip default is universal. (d) Filter width is
the local cell-volume geometric (or harmonic) mean with no wall-cell
modification. (e) No production ocean/idealized-LES code applies van
Driest.

## (ii) Recommended design, mapped onto named in-repo mechanisms

Current state: the closure rejects walls at
`smagorinsky_lilly.py:275-280`, transverse velocities at `:282-287`,
non-uniform meshes at `:304-311`; the strain is the plain centered
`_strain` (`:341-349`) that reads across a wall; `_eddy_viscosity`
(`:351-377`) interpolates the off-diagonal squares to centers via
`.to(anchor)` (`:373-374`); `_stress` (`:382-399`) diffs the stress
flux on the same axes.

### Stage W1 — free-slip walled Smagorinsky (uniform mesh)

The only wall-choice-dependent strain is the shear
`Σ_{a,t} = ½(∂uₜ/∂a + ∂u_a/∂t)` on the wall-normal (`Inner`-along-`a`)
edge. At a free-slip wall it is **structurally 0** (impermeability
makes `u_a = 0` along the wall so `∂u_a/∂t = 0`; free-slip sets
`∂uₜ/∂a = 0`). Both consumers — `|Σ|²` at the wall-adjacent center
and the stress divergence — get that wall-face value from the *same*
retag:

- **Strain (`_strain`).** For each off-diagonal component on a
  walled axis, form the interior part exactly as today
  (`Center→Inner`, interior faces only), then **retag the
  `Inner`-along-`a` factor to `Inner[Dirichlet]`** — a
  `_dirichlet_edge` helper mirroring diffusion's `_dirichlet_face`
  (`diffusion.py:470-478`). This is a true wall-value claim: the
  shear **is** zero there. Diagonal `Σ_aa` (wall-normal velocity,
  already `Inner[Dirichlet]` via `model/declarations.py:454-478`)
  closes on its own tag; tangential diagonals are interior.
- **`|Σ|²` (`_eddy_viscosity`).** `(s*s).to(anchor)` for the
  retagged off-diagonal becomes `Inner[Dirichlet]→Center`
  interpolation, which telescopes with the structural 0.
  *Feasibility verified:* `interp.py:176-200` registers
  `Inner→Center`, and `require_grounded_bounded_sides`
  (`staggering.py:348-409`) grounds it because the Dirichlet tag
  carries BC structure on the needy side.
- **Stress (`_stress`).** `(s * nu_t.to(s))` is formed on the
  interior `Inner` edge, then retagged to `Inner[Dirichlet]` before
  `.diff(ax_j)` — verbatim the tangential branch of `_harmonic`
  (`diffusion.py:556-558`). Wall-normal stress components retag back
  onto the component's own tag (`_harmonic`'s `_WALL_NORMAL` branch,
  `:554-555`).
- **Bind.** Classify each velocity axis with the diffusion
  `_wall_treatment` vocabulary (`diffusion.py:300-338`); drop the
  walled rejection; **keep** the transverse and uniform-mesh
  rejections. Default `slip="free"`.

**Ghost fill over one-sided stencils, decisively:** the
`boundary="one_sided"` patch (`interp.py:285-290`,
`finite_difference.py:357-360`, `patch_one_sided_edges`
`staggering.py:412-439`) declares `layout="local"` — static
physical-edge indices require the walled axis un-sharded. The
`Inner[Dirichlet]` retag runs the homogeneous mirror ghost fill
*inside* `shard_map` (`decomposition/tensor.py:1461-1534`) and is
distribution-transparent. FRIDOM decomposes across walled axes on
multi-GPU; the retag composes, the one-sided patch does not. (Also
exactly what Oceananigans does.) At two-wall **corners** both
factors are Dirichlet-tagged (free-slip: both structural 0) — the
retag handles corners by composition; no-slip corners are genuinely
subtle, a reason no-slip stages second.

### Stage W2 — no-slip + `slip=` API (owner spelling: (a))

No-slip's wall shear is `Σ_{a,t}^wall = uₜ₁/Δn ≠ 0` (odd mirror).
Spelling **(a) wall-row correction** (ruled): keep the free-slip
retag and inject the wall shear. **Smagorinsky-specific twist:**
unlike linear friction (one flux correction, `_wall_correction`
`diffusion.py:481-518`), the injection lands in *two* consumers —
the wall-adjacent center's `|Σ|²` **and** the stress flux — and must
be kept consistent. Reuse the folded-static-weight discipline
(`1/Δn²` folded into a true-shape weight array, never divided as a
field) that `_wall_correction`'s docstring documents, to keep the
VJP finite (ghost-slot `0/0` hazard). Recorded alternative (b): a
`Center[Dirichlet]→Outer` `FiniteDifference` target row (currently
absent; `LinearInterp` has `target=OUTER`, `interp.py:205-227`) —
one mechanism for both slips, shard-map-safe, but a registry change
with wider blast radius; take it only if a third wall-stress
consumer appears.

### Stage W3 — non-uniform / stretched Δ

On a uniform mesh Δ is constant near walls, so W1/W2 need not touch
it — the `:304-311` rejection stays through W1/W2. W3 replaces the
scalar `self._filter_width` (`:215, 312-313`) with a per-cell
`Δ = (∏ᵢ Δxᵢ)^{1/n}` field from `grid.measure` at centers (ν is
already built at centers). This unlocks stretched meshes (periodic
or walled) and is the prerequisite for any terrain filter width.
Scotti–Meneveau–Lilly anisotropy factor: omitted (ruled).

## (iii) Answers to the six design questions

1. **Wall strain: ghost fill vs one-sided.** BC-parity ghost fill
   (the `Inner[Dirichlet]` retag), decisively — shard_map-transparent
   where one-sided patches force `layout="local"`; composes with the
   landed `_harmonic`/`_dirichlet_face` machinery unchanged. The
   choice changes the answer **only** for the off-diagonal shear.
2. **Filter width.** Near a wall on a uniform mesh Δ is unchanged →
   the uniform-mesh rejection stays for W1/W2; non-uniform Δ is the
   orthogonal W3 (local cell-metric mean, no wall correction —
   matches Oceananigans/MOM6).
3. **Van Driest.** Omit. Oceananigans omits it; ABL LES "rarely
   invokes" even the `κz` cap and never at a free-slip wall (no
   viscous sublayer — Sullivan–McWilliams–Moeng, Pope §13, Sagaut);
   van Driest exists only for resolved no-slip walls. If ever
   needed it folds in as a multiplicative `Γ_wall` beside the
   existing Ri damping — needs `u_τ` + wall distance (wide blast
   radius). Not a stage item.
4. **`|Σ|` and autodiff.** The existing `_guarded_sqrt` custom_jvp
   (`:97-142`) already seals `|Σ| = 0`; the wall terms add no new
   singularity — free-slip wall shear is a structural constant 0,
   no-slip wall shear is `uₜ₁/Δn` (smooth, linear). Caveat: the
   no-slip `1/Δn²` must be folded into a static weight array (never
   divided as a field) — the `_wall_correction` discipline.
5. **Dissipation guarantee.** For frozen
   `ν = (CsΔ)²|Σ| + ν_bg ≥ 0`, `⟨u, ∇·τ⟩ = −⟨ν, |Σ|²⟩ ≤ 0` — the
   variable-coefficient generalization of the linear friction
   identity (Griffies–Hallberg's construction: prove the sign for
   frozen ν, bolt the nonlinearity onto the dissipative linear
   skeleton). Three discrete requirements, all structural in FRIDOM:
   ν ≥ 0 by construction; ν interpolated to the strain location with
   nonnegative weights; strain (velocity→edge) and stress divergence
   (edge→velocity) exact transposes on the C-grid matched pairs. At
   the wall both slips keep it: free-slip → wall shear 0 makes the
   boundary term vanish (G–H's "neglect surface terms" is legitimate
   *because* of the masking); no-slip → the discrete wall-drag row
   is `−2ν uₜ₁²/Δn² ≤ 0` (MITgcm side-drag sign). The G–H
   `√B`-intermediate subtlety bites only a future biharmonic
   Smagorinsky — deferred.
6. **Transverse-velocity rejection.** Independent gap, not removed
   by the walled lift: `transverse` (`field_table.py:194, 211-221`)
   is a slaved/reduced-dimension component with no directional
   derivative for the strain tensor. Stays.

## (iv) Sizing against the landed walled diffusion/friction work

| Stage | Content | Rough size (src/test) | Reuses |
|---|---|---|---|
| **W1 free-slip walls** | `_dirichlet_edge` retag in `_strain`/`_eddy_viscosity`/`_stress`; bind classifier; drop walled rejection | ~120–200 / ~180–260 | `_wall_treatment`, `_dirichlet_face`, `Inner[Dirichlet]→Center` (all shipped) |
| **W2 no-slip + `slip=`** | spelling (a): dual wall injection (`|Σ|` + stress) + `slip=` API | ~150–300 / ~200–320 | `slip=` vocab `diffusion.py:257-287`; `_wall_correction` folded-weight discipline |
| **W3 non-uniform Δ** | per-cell `grid.measure` Δ field | ~80–160 / ~120–200 | measure machinery (shipped) |

## (v) Gates (mirroring `test_diffusion_walls.py`)

- **Constant-coefficient reduction (decisive):** `Cs=0` walled
  Smagorinsky stress == `HarmonicFriction(ν_bg, slip="free")`
  walled, **bit-for-bit** — extends the periodic
  `test_cs_zero_stress_is_background_friction` (`:85`) to
  `periodic=False`.
- **Free-decay energy monotone** on a walled grid, both slips
  (guards the strain↔stress transpose pairing at the wall).
- **Doubled-domain mirror symmetry** — free-slip walled `[0, L]` ==
  even restriction of periodic `[0, 2L]` with mirror-symmetric IC.
- **Free-slip uniform wall-parallel flow → zero drag; no-slip →
  drags only wall cells** (the `test_diffusion_walls.py:170,178`
  patterns).
- **Wall-normal component slip-independent**; **two walled axes
  compose** per-axis.
- **Autodiff FD-match** through a walled run, both slips (step-path
  policy; the no-slip row is linear in `uₜ₁`).
- **Sharded walled == single-device** (the `forced_devices` shard
  pattern, `test_diffusion_walls.py:412`) — proves the ghost-fill
  route survives decomposition of the walled axis.

## (vi) What stays deferred, and why the layering is safe

- **Immersed Smagorinsky** — reuse the diffusion immersed pattern
  (mask velocities + fraction-weighted fluxes + sealed wet-fraction
  divergence, `diffusion.py:390-464`). Safe: the wall retag acts on
  domain walls, immersed masking on cut faces — different faces,
  proven to compose for linear friction (`diffusion.py:116-125`).
  The `ClosureBase` immersed gate stays.
- **Terrain / mapped Smagorinsky** — needs the metric-tensor strain
  (cross terms + J-weighting) via `physical_diff` and `grid.metric`,
  and depends on W3 (J-weighted Δ). Along-σ first (ratified
  diffusion §6.3), full-metric later.

The layering is safe because each stage reduces, in the `Cs=0`
limit, to the corresponding already-ratified walled friction
closure — walls to `HarmonicFriction` walls, immersed to immersed
friction, terrain to along-σ friction. The only irreducibly new
ingredient is the off-diagonal shear retag, contained in W1/W2.

## (vii) Landing addendum (2026-07-19)

Shipped same day as designed: merge `c028a84d` (stage commits
`4d27e12a` W1, `40040763` W2, `8c7982fd` W3). All §(v) gates green,
including the two decisive bit-for-bit oracles: `Cs=0` ≡ walled
`HarmonicFriction`, and no-slip ≡ the odd-mirrored doubled-periodic
run at maxdiff 0.0 (pinning the two-consumer consistency); forced-4
sharded-walled ≡ single-device at 1e-12 for both slips; all periodic
tests bit-identical. Implementation deviations, all recorded in the
tests/docstrings:

1. **The ½ convention**: the module's `τ = νΣ` (no factor 2) makes a
   wall-parallel shear's stress half the Laplacian rate, so the
   `Cs=0` gate holds against `HarmonicFriction(ν_bg/2)` for shear
   components (full rate on the wall-normal diagonal) — the §(v)
   wording omitted the ½ the convention forces.
2. **N² buoyancy-gradient retag**: `b.diff(z).to(anchor)` cannot
   ground at a walled vertical; it takes the same `Inner[Dirichlet]`
   retag (∂b/∂z = 0 at a no-flux wall).
3. **Walled FV taught error (new)**: dropping the blanket reject
   would have exposed walled `CellAvg` grids via the FV promotion;
   W1 is nodal-scoped, so walled FV is refused rather than run
   unvalidated (follow-up on the deferred shelf).
4. **W3 tested at the closure level**: the full nonhydro2 model
   cannot assemble on a raw `MappedIntervalMesh` (pre-existing core
   limitation — no spectral transform), so the per-cell-Δ stage is
   exercised on a bound field table.
5. **|Σ|² injection form**: `interp_t((u_t1/Δn)²)` at wall cells,
   factor 1.0 (the off-diagonal ×2 and interp ×½ cancel) —
   calibrated against, and bit-for-bit consistent with, the
   odd-mirror oracle.

**Key files:** design substrate
[`diffusion_walls_terrain_scoping.md`](diffusion_walls_terrain_scoping.md);
template `src/fridom/model/closures/diffusion.py:257-566`; target
`src/fridom/nonhydro2/modules/smagorinsky_lilly.py:275-419`;
operator feasibility `src/fridom/spatial/operators/interp.py:132-227`,
`finite_difference.py:148-218`, `staggering.py:348-439`; velocity BC
`src/fridom/nonhydro2/modules/core.py:446-472`,
`src/fridom/model/declarations.py:454-478`; gates
`tests/nonhydro2/test_smagorinsky_lilly.py`,
`tests/model/closures/test_diffusion_walls.py`.
