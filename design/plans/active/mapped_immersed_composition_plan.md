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
