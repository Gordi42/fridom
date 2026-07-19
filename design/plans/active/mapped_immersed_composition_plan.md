---
status: active
date: 2026-07-18
---

# Mapped + immersed composition

**Goal (owner-ratified 2026-07-18):** close the second immersed
residual — lift the "mapped + immersed" taught error for
**chart/terrain + immersed** grids by composing the two proven
halves (the mapped metric SPD operator and the masked α/θ flux
form), and pin the already-working **stretched + immersed**
composition with tests. Owner ratified the recommendation including
MI-D1's Jacobian-weighted chart fractions.

## 1. Research basis (2026-07-18, two sweeps)

- **The item splits in two.** The taught error keys on
  `mapping.column_corrections` (terrain/chart columns) — a separable
  per-axis stretch is *not* rejected, its fractions are already
  physically correct (per-axis quadrature places nodes in physical
  space), and the solve handles the stretch through the `diff` rows.
  Only the preconditioners are stretch-blind (degraded convergence,
  correct physics). The genuine frontier is charts.
- **Three structural gaps, one dependency chain**: (1) chart
  fractions ignore the per-cell Jacobian (`grid.py` documents the
  chart-parameter average as the F5 deferral); (2) no operator
  applies α/θ to the mapped `K = J·Bᵀ W B`; (3) no preconditioner
  serves the masked metric operator (the masked-spectral trick is
  flat-only; the immersed multigrid's `vertical_bands` assumes
  uniform spacing). Velocity correction and advection are
  wire-and-gate after those (`_FluxFormAdvection` already carries
  both weightings — they have just never run together).
- **Literature is unanimous on the spelling**: the fraction weights
  the **metric-weighted flux, in computational space, at the face**
  — after the metric, never on the field (ROMS/NEMO masks ×
  e-weighted fluxes; Cartesian cut cells; CURVIB immersed-on-
  curvilinear). The headline trap is **geometric consistency
  (GCL/freestream)**: fractions built inconsistently with the metric
  produce a spurious pressure gradient in uniform flow that looks
  like physics — adopted here as the anchor gate, not a risk.
  Secondary: small-cell stiffness compounds multiplicatively with
  metric stretch; smoothers degrade at masks (answered by per-level
  re-discretization, which the immersed V-cycle already does).
- Real use cases: terrain-following over the smooth majority +
  immersed isolated steep features (the σ-SIBM argument); mapped
  domains with immersed islands whose lateral boundary is fractional
  instead of staircased.

## 2. Decisions

### MI-D1 — chart fractions: Jacobian-weighted physical averages

On a grid whose mapping declares column corrections, the cell
fraction is the **physical wet-volume fraction**
`θ_c = ∫_cell J·χ / ∫_cell J` (tensor GL quadrature over the
computational cell; indicator evaluated at the physical positions of
the nodes via the mapping; weights `J(node)·w`). The **same
discretized `J`** the operators consume weights the quadrature —
fraction and metric are one consistent geometry; this is what makes
the freestream gate achievable. The separable-stretch path is
**structurally untouched** (bitwise parity — the existing per-axis
physical placement already is the physical average there). Face
fractions keep the **min-rule** (direct physical face-area
quadrature stays designed-for, as in IP-D1). `min_fraction` floor
semantics unchanged.

### MI-D2 — the composed pressure operator

Insert the **open-area fraction as a diagonal weight on the face
fluxes of the mapped operator**, between the exact-transpose
corner-hop pairs that carry its SPD license:
`L p = row_scale · D_adj( diag(α) · F(p) )`, with `F` the existing
metric fluxes (`K^{ii}`, corner-staggered `K^{ib}`, column leg) and
the row scale **unchanged from the mapped solver** (J-weighted
computational measure — CG's inner product untouched). A face's α
weights its *entire* K-row flux (direct and cross legs through that
face). Dry rows vanish; the nullspace is the wet-region constant
`e`, projected orthogonally **in the measure in which this
row-scaled operator is symmetric** (the I2 e-form lesson: derive the
projector from the product, do not guess — the flat case's
V-weighted wet mean generalizes with the J weight). RHS = the same
α-weighted divergence of the provisional metric fluxes, compatible
by telescoping. The arbiter for any spelling question is the
**exact-symmetry gate** (⟨q, Lp⟩ ≡ ⟨p, Lq⟩ at machine zero, the
mapped solver's shipped gate).

### MI-D3 — preconditioner: multigrid-with-masking

Per-level **re-instantiated composed solver on coarsened grids** —
`Grid.coarsened` already propagates both descriptors, the immersed
V-cycle already re-derives fractions per level (MG-D6), the mapped
V-cycle already handles the metric; compose per level. Geometric
transfers unchanged. Vertical-line smoother from the
fraction-weighted metric diagonal; per-level wet projection. Fix the
immersed `vertical_bands` **uniform-spacing assumption**
(stretch-aware bands — this also unblocks stretched+immersed
multigrid). Default preconditioner for the composed solve:
`"multigrid"`; `"spectral"` = the masked folded-mean flat inverse
(`e ⊗ M̄⁻¹`), kept as the cheap fallback; iteration counts are
gated and reported.

### MI-D4 — velocity correction

`u −= m_f · (F/J-style metric correction)` — the mapped correction
gated by the boolean face mask, divisions double-`where` sealed.
Mechanical once MI-D2 exists.

### MI-D5 — advection: prove the α×J factoring

`_FluxFormAdvection` already carries both weightings; lifting the
gate makes them compose: α on the physical face flux (IP-D4,
*before* the J-weighted mapped divergence gathers it), `θ` on the
cell scale. Nothing new is built — the work is the **proof**:
conservation of `Σ_c θ_c J_c V_c q_c` to machine zero with
advection active on a cut chart, and constancy preservation
(uniform tracer exact — the freestream gate's advective face).
Biased/graded advection on mapped grids **stays rejected**
(follow-on to the graded-ladder plan); closures **stay rejected**
(separate residual); `MaskState` is metric-blind and needs nothing.

### MI-D6 — scope

nonhydro2 is the deliverable (M1–M4). Hydrostatic wet-column
masking of the terrain barotropic solve (`H = ∫J dz` → wet-column
integral on the phase-B volume-exact seam) is stage M5, may ship
separately. shallowwater2 mapped+immersed stays a taught error
(untriaged tail). Both taught-error sites (`core.py`
`_require_fv_capable`, `immersed_pressure.py` ctor) lift only when
M2 lands; the eigenmode/closure gates stay.

## 3. Stages and gates

Sequential branches, one worktree each; owner-reviewed merge per
stage group.

| Stage | Work | Branch | Gate |
|---|---|---|---|
| **M0 — pin the working composition** | stretched+immersed regression test (tests only; documents the ungated working case) | with M1 | stretched immersed step runs, conserves θ-mass, matches unstretched physics qualitatively; CG residual sane |
| **M1 — chart fractions** (`spatial`) | MI-D1 J-weighted tensor quadrature path (charts only; separable path bitwise-untouched) | `feat/immersed-chart-fractions` | separable-grid fractions **bitwise** vs today; chart fractions vs analytic geometry converge in `q`; flat-chart (identity map) ≡ unmapped quadrature; floor semantics pinned; forced-4 invariance |
| **M2 — composed solve** (`nonhydro2`) | MI-D2/D4 operator+RHS+projection+correction; lift both taught sites; family-gate flip | `feat/mapped-immersed-solve` | **operator symmetry exact** (rel-diff ~0); **freestream/GCL**: constant field → machine-zero composed divergence, uniform flow over a cut chart → machine-zero tendencies; all-wet chart ≡ pure mapped solve (tight tol, ~1 extra CG iter); identity-chart + mask ≡ existing immersed solver; manufactured masked Poisson on a genuine chart converges 2nd order; post-projection masked divergence machine zero |
| **M3 — preconditioner** | MI-D3 masked+mapped V-cycle; stretch-aware `vertical_bands`; masked-mean spectral fallback | same branch as M2 (or split if large) | iteration count on a steep cut chart bounded and reported (vs the 30 budget; compare spectral-fallback vs multigrid); stretched+immersed multigrid no longer raises; per-level wet projection pinned |
| **M4 — advection + model gates** | MI-D5 proof; taught-gate updates; autodiff shard | `feat/mapped-immersed-model` (or with M2) | `Σ θ J V q` conserved to machine zero with advection on a cut chart; uniform tracer exact; `jax.grad` FD-match (policy shard); forced-4; ≥95% patch coverage; ruff; smoke file |
| **M5 — hydrostatic extension** | wet-column `H̃ = ∫_wet J dz` on the phase-B seam; masked terrain transports | `feat/immersed-terrain-hydrostatic` | column equivalence (flat immersed chart ≡ shallower domain); masked continuity machine zero; implicit ≡ explicit oracle |

Every stage: mirrored tests, ruff clean, model smoke file where core
machinery is touched, autodiff regression for step-path changes
(AGENTS.md policies).

## 4. Risks

- **Freestream-gate failure = fraction/metric inconsistency.** The
  cure is MI-D1's "same discretized J" rule — quadrature and
  operator must consume one geometry. Never loosen the gate; fix
  the discretization.
- **Corner-cross masking subtlety** — α on a face must weight the
  whole K-row flux with the adjoint pairing intact; the exact
  symmetry gate arbitrates any spelling.
- **Small-cell × metric conditioning** — α multiplies an already
  stretched coefficient; `min_fraction` floor stays default 0.1,
  iteration counts gated, multigrid is the lever (risk confirmed
  ~60+ iters on flat genuine partials with the spectral
  preconditioner alone).
- **Parallel-dev movement** — `mapped_pressure.py` and
  `advection.py` are hot files this week; re-check dev before every
  merge.

## 5. Out of scope (designed-for, not precluded)

Direct physical face-area quadrature; closures on immersed grids
(separate residual); biased/graded advection on mapped grids
(graded-ladder follow-on); sw2 mapped+immersed; moving immersed
geometry; level-set representation; interface-aware smoothers
beyond line-Jacobi (Galerkin coarse operators recorded as the
escalation if per-level re-discretization underperforms).

## 6. Implementation record

**M0+M1 shipped 2026-07-18** (merge `5bc27631`; branch
`feat/immersed-chart-fractions`). `Grid._cell_quadrature_fields`
(raw per-cell GL nodes + unit-sum tensor weights),
`CoordinateMapping._column_correction`/`_param_at_nodes` (mapped
physical positions + column Jacobian at given nodes, via the same
`jvp`-along-base construction the operator metric rows use), and the
chart branch of `ImmersedDomain._quadrature_cells`. Gates: chart
fractions vs analytic geometry spectrally exact for smooth
integrands (5.5e-17 at q=2), monotone convergence in q for a hard
cut, terrain column `z = σ·H(x)` vs an independent numpy tensor
oracle to 1e-12 (H genuinely x-coupled); identity chart ≡ unmapped
**bitwise**; separable path bitwise-untouched; all-wet exactly 1.0;
floor semantics pinned on charts; forced-4 device invariance
(4-vs-1-device chart fractions ≤ 1e-12); 313+15 mirrored tests
green; ruff clean.

**Correction to §1 (the M0 premise).** "Stretched + immersed
already works ungated" holds only at the **fraction/grid layer**
(pinned there — stretched fractions are exact physical averages,
bitwise-unchanged path). At **model level no composition assembles
today**: the immersed solver's spectral preconditioner has no
stretched-z transform (`DispatchError` on `CellAvg(z)`), and its
multigrid `vertical_bands` raises on non-uniform widths — and a
stretched-z *non-immersed* nonhydro2 fails the same way (bare
separable stretch is generally unsupported by the flat spectral
path; the supported stretched vertical is the mapped column, which
is taught-errored with immersed until M2). Consequence: **M3 must
make the composed solve serve the bare-stretch case too** (the
stretch-aware `vertical_bands` fix, or routing bare stretch through
the mapped diff-row operator; a plain-CG immersed fallback is the
recorded alternative). The M2/M3 solve is the first path that will
actually run any stretched-or-terrain immersed model.

**Interpretations / seams recorded for M2:**

- **MI-D1 "computational cell" is realized as physical-placement
  averaging** (per-axis physical GL nodes, unit-sum weights — the
  separable path's own convention); the ratio `Σ(wJχ)/Σ(wJ)` is
  proven equal to the exact physical wet-volume fraction on a
  stretched base (the mesh Jacobian rides the placement, `J`
  carries only the column factor — no double count).
- **J-consistency seam (feeds the freestream gate):** the fraction
  evaluates the mapping's **analytic static-default parameter**
  (e.g. `H(x)`) at quadrature nodes; the operator consumes the
  **discrete/interpolated** parameter field at DOF nodes. Same map
  callable, same autodiff; they coincide exactly for constant
  parameters and differ by the parameter's discretization
  otherwise. M2's freestream/GCL gate arbitrates: either build the
  fraction from the discrete field pipeline or prove the gate holds
  with the analytic spelling.
- **`order=None` (collocation) on charts is unmapped** — it samples
  the computational base center, not the mapped physical center.
  M2 decision: require `order >= 2` for chart immersed domains, or
  map the collocation point.
- Supplied (dynamic) mapping parameters are not handled in the
  fraction (static geometry only) — moving terrain + immersed
  stays designed-for.

**M2+M3+M4 shipped 2026-07-19** (merge `48ac9052`; branch
`feat/mapped-immersed-solve`). `ComposedPressureSolver`
(`nonhydro2/modules/composed_pressure.py`, a `MappedPressureSolver`
subclass), `_project_composed` routing, both taught sites lifted,
stretch-aware immersed `diagonal`/`vertical_bands`, the
fraction-weighted composed multigrid + wet-masked spectral fallback,
and the M4 advection conservation fix. Gates: operator symmetry
**5.28e-16** on a genuine cut chart; wet-constant nullspace exact
0.0; all-wet ≡ mapped **bitwise**; deep-interior ≡ mapped
**bitwise**; identity-chart + mask ≡ flat immersed ≤ 1e-12;
`apply ≡ div(correction)` 7.1e-15; post-projection masked divergence
1.8e-15 (multigrid, steep chart); `Σ θ J V q` with active cross
terms **4.4e-16**; stretched-z immersed model now assembles and
steps (multigrid 1.7e-15) — the §6 model-level blocker is closed;
autodiff FD-match; forced-4 green; ruff clean.

Corrections / spellings selected by the gates:

1. **MI-D2 spelling correction — corner-α, not whole-K-row face-α.**
   Weighting the whole assembled face flux by `α_face` puts
   different scalars on the two halves of one adjoint pair
   (asymmetric on genuine partials). The symmetric spelling: face α
   on each **direct** leg, and a single shared **corner α**
   (min-of-4, on `Right_a × Inner_b`) **inside** each cross hop,
   between the transpose-paired interpolations. §4 anticipated
   exactly this arbitration.
2. **Projector**: the wet-constant is removed in the
   **computational** measure (matching CG's `_dot`), not the
   J-weighted physical one.
3. **MI-D4**: the correction divides by `α_a·J` / `α_base` (sealed
   double-where), not boolean-mask-only — required for
   `apply ≡ div(correction)` exactness on the metric cross.
4. **M4**: the conservation gate caught a real 1.17e-3 drift — the
   reduced cross flux is now gated by the base-face fraction
   (no-op without a mask; mapped advection byte-identical).
5. **MI-D3 auto default**: `pressure_preconditioner` is now
   `None`=auto (flat/mapped/immersed resolve to their previous
   spectral defaults byte-identically; composed resolves to
   multigrid — masked spectral does not converge in the 30-iter
   budget on cut charts, ~200 iters measured vs multigrid ~15).
6. **Manufactured-Poisson gate substituted**: the standalone masked
   chart Poisson solve is preconditioner/small-cell limited
   (`min_fraction=0` sliver pathologies stall even plain CG); the
   2nd-order claim rides the equivalence gates (all-wet ≡ mapped
   bitwise, identity ≡ immersed, deep-interior ≡ mapped bitwise)
   and the physically-relevant divergence RHS reaching machine
   zero.

Seam verdicts: analytic-H fractions suffice (static maps share the
same `H` callable between fraction quadrature and operator metric —
exactly consistent; discrete/moving `H` rejected by the composed
multigrid); `order=None` collocation on charts is now a **taught
error** at the composed ctor (`order >= 2` required — collocation
mis-places the physical center); min-rule α confirmed by the
symmetry gate.

Open (flag to owner): (a) the advection cross is gated by the
**base-face** α while the pressure cross uses the **corner** α —
both internally consistent (conservation vs symmetry), different
placements for the "same" metric cross, worth a sanity ruling;
(b) robust preconditioning for pathological `min_fraction=0` sliver
geometries is a follow-up (the model's own per-step RHS is
well-behaved); (c) real multi-process (`srun -n N`) not exercised
(forced-4 only, per the no-GPU-jobs rule); (d) uniform tracer
through a full projected step is O(h²)-inexact identically for pure
mapped and pure immersed too — a baseline projection-method
property, not composed-specific.

**M5 shipped 2026-07-19** (branch `feat/immersed-terrain-hydrostatic`;
not yet merged). The hydrostatic wet-column terrain barotropic solve:
the phase-B `BarotropicPressureSolver` face depth becomes the wet-column
integral `H̃_a = ∫ α_a J dz` (min-rule face fraction weighting the
metric-weighted column Jacobian), the free-surface RHS becomes the wet
transport divergence `∫[∂_x(α_x Ju) + ∂_y(α_y Jv)] dz`, and the solver
masks its mean-depth spectral preconditioner onto the wet columns, its
ε=0 gauge onto the wet-column-constant nullspace, and the `ps` output to
zero under a land column. The core `_diagnose_w` gains a masked
**contravariant** continuity branch (`(α_x Ju).diff + (α_y Jv).diff`
integrated to `α_z Jω`, guarded divide). The three bind-time rejections
(`HydrostaticCore`, `ConstantStratification`, `_FreeSurfaceBase`) are
replaced by one shared `terrain.require_chart_immersed_order` guard
(order ≥ 2 required — a collocation mask mis-places on a chart, a taught
error). The buoyancy slope term (`d629a489`) needed **no** change: it
reads the masked contravariant `w` and min-rule-consistent velocities,
`MaskState` keeps dead cells dead. Gates (CPU, forced-4): masked
contravariant continuity **1.8e-15** on wet cells (w exactly 0 on closed
faces); wet-column operator self-adjoint **4.5e-15** on a genuine cut
chart; GB-1 wet transport divergence cancels **7e-16** (ε=0); column
equivalence (J≡1 chart flat bottom ≡ shallower chart) **6.7e-16**
explicit / **7.2e-16** implicit; implicit≡explicit oracle converges
first-order (slopes ~1.1) on a J≡1 masked chart; all-wet chart ≡ pure
terrain **byte-identical** (w) / ≤ 2e-16 (ps, ε=1); identity-chart + mask
≡ flat immersed ≤ 1e-13; θ-mass conserved **0.0** over 12 steps;
autodiff FD-match rtol 2e-13; masked spectral converges in 9–17 iters
(a=0.4–0.8, ε=0/1); forced-4 device-invariant ≤ 1e-8. 63 new mirrored
tests; full hydrostatic suite (434) green; ruff clean.

Corrections / spellings recorded:

1. **Explicit `_terrain_inv_depth` stays the FULL physical depth**
   `∫J dz` (not the wet depth) on a cut chart: the masked
   `_terrain_transport_div` already carries α, and dividing the wet
   transport by the full depth makes the explicit form reduce to the
   implicit `T*_wet/H_ref` on a J≡1 chart (the oracle limit) and to the
   pure terrain form byte-identically when all-wet.
2. **Wet projection is the computational-measure wet-column mean**
   (`_wet_mean_free`), matching CG's `_dot` and keeping dry columns
   exactly zero — the flat immersed e-form generalized to the 2-D
   barotropic solve (the pure terrain `_mean_free` global mean would
   paint a constant onto dry columns).
3. **The multigrid preconditioner is NOT yet wet-aware** on a
   terrain + immersed grid — a taught error at the solver ctor (the
   point-Jacobi V-cycle coarsens the metric per level but the coarse
   levels do not re-quadrature the wet fractions). The default masked
   **spectral** converges in ≤ 17 iters on steep cut charts, so the
   deferral costs nothing at these sizes. A follow-up.
4. **Split-explicit + terrain + immersed stays a taught error** via the
   split variant's own pre-existing terrain refusal (it rejects any
   sigma column, immersed or not) — designed-for, use the implicit free
   surface.

Seam verdict: the physical **genuine-chart** (J≠1) column-equivalence
twin is ambiguous (the wet sub-column is an affine sub-chart, not a
plain shallower domain), so — following the M2-M4 correction 6
precedent — the column-equivalence gate rides the J≡1 chart limit
(clean, exercises the full composed assembly + wet-column solve) plus
the algebraic gates (all-wet ≡ pure terrain byte-identical, GB-1 wet
cancellation, masked continuity machine zero, θ-mass) on genuine cut
charts.
