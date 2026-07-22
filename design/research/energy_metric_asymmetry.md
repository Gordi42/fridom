---
status: implemented (option 1, analytic spelling) — see §4 addendum
date: 2026-07-18
---

# The terrain energy "asymmetry": probes, mechanism, and the missing term

**Status: research record, 2026-07-18.** Investigates the
baroclinic-vs-barotropic energy-metric asymmetry flagged by the
physical-integral-default work
([`../decisions/physical_integral_default.md`](../decisions/physical_integral_default.md)).
Outcome: the asymmetry is the *symptom* of a missing O(slope) term in
the terrain hydrostatic buoyancy equation; with that term supplied the
physical metric is energy-consistent at second order, and a discrete
adjoint spelling can plausibly make it exact. Probe scripts lived in
the session scratchpad (`probe1..6`, single-session artifacts); every
number below is reproducible from the recipes given.

## 1. What the probes established

Setup: the `tests/hydrostatic/test_free_surface_terrain.py` builders
(`zp = z·H`, `H = 1 + a·sin(2πx)cos(2πy)`), linear model, f = 0,
n = 16 unless stated. "Physical" inner product: the (now seeded)
J-weighted `integrate` with `ps` lifted and `H/c²`-paired; the
"computational" product uses raw `Integral()`.

1. **Barotropic pair** (`−∇ps` ↔ flux-form depth-mean divergence):
   exactly skew under the **physical** product — machine-zero at every
   amplitude, confirmed *structurally* by the bilinear test
   `⟨X, MLY⟩ + ⟨Y, MLX⟩` on independent random states (≤ 3.4e-16).
   Under the computational product it leaks ∝ a.
2. **Baroclinic pair** (slope gradient ↔ w-diagnosis ↔ buoyancy):
   exactly skew **only on the flat grid**. On terrain it leaks O(a)
   under **both** plain products (bilinear-structural; e.g. at a = 0.2
   comp −8.7e-3, phys −1.17e-2; the physical leak is *linear* in a,
   |skew|/a ≈ 0.044 flat across a). The prior belief that the pair is
   computational-exact on terrain is **refuted**; the existing smooth
   gate (`test_baroclinic_energy_conversion_is_conserved_to_roundoff`)
   passes by *state selection accident*: single-mode / mode-gap states
   sit in the leak's null set (probes: single horizontal modes at any
   wavenumber including near-Nyquist are machine-zero; broadband
   states leak; **vertically** rough states dominate the leak —
   smooth-xy/random-z leaks 1.3e-1 vs random-xy/smooth-z 1.8e-3).
3. **Spectrum**: the dense linear operator (materialized by
   `jax.jacfwd` of the tendency — valid for the hydrostatic explicit
   path; NOT for nonhydro2, whose CG projection jacfwd silently
   drops) is **purely imaginary** at every amplitude (a ≤ 0.4), with
   and without rotation, at D = 800 and D = 3136: max|Re|/max|Im|
   ~ 5e-16, zero eigenvalues above 1e-10. **No spurious growing
   modes**; energy in a 400-step nonhydro2 terrain run stays bounded
   (≤ E0, mild downward drift). The scheme is stable; only the
   *representation* of its invariant is at issue.
4. **The decisive scaling**: on a *fixed resolved broadband* state the
   physical-metric baroclinic skew does **not converge** with
   resolution (0.095 / 0.104 / 0.106 at n = 16/32/64) — the coded
   linear system converges to a continuum system that is *not* the
   Boussinesq hydrostatic equations over terrain.
5. **The missing term**: linearizing adiabatic buoyancy about a
   physically resting stratified state over terrain gives, in sigma
   coordinates,
   `∂b/∂t = −N²·(Jω + u·Zₓ + v·Z_y)` — the physical vertical velocity
   `w_true = Jω + uZₓ + vZ_y`, not the contravariant flux alone. The
   coded `stratification.restoring` is `−N²·w` with terrain
   `w = Jω` (`core.py` diagnoses the contravariant volume flux): the
   slope-advection half is **absent**, while the u-equation carries
   the *exact* constant-physical-height gradient (H2b) — the
   inconsistent half-pair. Adding `−N²(u·Zₓ + v·Z_y)` analytically in
   the probe (metric fields `dzp_dx`/`dzp_dy` sampled at the b cell)
   collapses the physical skew to **second-order convergence**:
   2.0e-2 → 5.3e-3 → 1.3e-3 at n = 16/32/64. The rulings record
   (`stretched_terrain_combined.md`) never discusses the term — a
   silent omission, not a documented convention.
6. **nonhydro2 mapped** carries a *smaller* cousin leak (bare
   operator, divergence-free state: −6.5e-3 at a = 0.2,
   CG-iteration-independent, nodal ≡ fv bitwise); its energy is
   bounded in time integration. **Resolved 2026-07-19: §5** — the
   n-scaling probe shows first-order convergence (truncation, not a
   convention error), and the buoyancy pair carries exactly zero
   leak (the residual is projection-borne).

## 2. Interpretation

The terrain hydrostatic linear physics is O(a)-wrong in the buoyancy
equation: internal-wave dynamics over slopes (dispersion, wave-terrain
interaction) deviate at first order in the slope. Every energy-metric
awkwardness follows: with half the pair physically exact (u-equation)
and half flux-form (b-equation), no plain diagonal quadratic is
conserved; the operator is still skew under *some* hidden SPD metric
(hence the imaginary spectrum and bounded energy), but that metric
represents the wrong physics. Fixing the metric without the term
would polish the diagnostic of an inconsistent operator.

Note the user-facing corollary: on terrain the model's diagnosed "w"
is the contravariant flux `Jω`, not the physical vertical velocity —
also a labeling/documentation issue for output.

## 3. Options (presented to the owner)

1. **Fix the physics (recommended): couple `b` to the physical
   vertical velocity on terrain.** Add the slope-advection term to
   `stratification.restoring`'s terrain branch — ideally spelled as
   the discrete **adjoint** of the H2b slope-gradient interpolation
   pattern (same `Z_i` metric rows, transposed), which would make the
   baroclinic pair *exactly* physical-skew like the barotropic one:
   one metric, exactly conserved, asymmetry gone. Fallback if the
   exact-adjoint spelling fights the staggering: the plain analytic
   spelling is already probe-proven O(h²)-consistent. Gates: bilinear
   random-state physical-skew test (replacing the accidental smooth
   gate), a terrain wave oracle (convergence against a refined
   reference), rest-state regression (term vanishes at u = v = 0),
   autodiff, H7 interplay. Effort S-M.
2. **The metric-surface mechanics (do regardless; the roadmap
   remainder):** `EnergyMetric` hydrostatic `ps` weight →
   `H(x,y)/c²` field (also fixing the flat depth ≠ 1 missing-H bug),
   eigen-channel `_bounded_measure` J-weighting for stretched-z maps,
   terrain taught error replacing the misleading Hermiticity-residual
   message.
3. **Metric-only accommodations (parked):** hybrid or
   operator-valued metrics for the *current* operator, or numerically
   deriving its hidden invariant — moot once option 1 lands, and
   before that they dignify inconsistent physics.

Follow-ups: n-scaling probe of the nonhydro2 mapped leak (same recipe
as §1.4) and an audit of its mapped w/buoyancy convention; the "w is
Jω" output-labeling question.

## 4. Outcome (2026-07-18, `fix/terrain-buoyancy-slope-term`)

**Implemented: option 1, the analytic spelling.**
`ConstantStratification.restoring` now couples `b` to the physical
vertical velocity `w_true = Jω + u·Zₓ + v·Z_y` on a terrain column —
the added half is `−N²(u·Zₓ + v·Z_y)` with `Zᵢ = d<mapped>_d<axis>`
sampled at the `b` cell and `u`/`v` interpolated onto it (`u.to(b)`).
Finite metric multiply (no `1/J`, no reverse-mode singularity to
seal); vanishes at rest; byte-identical on flat grids
(`self._column is None`). The module discovers the column itself
(`discover_column`, mirroring the core) and declares its own
`extra_halo` (`x:1, y:1, z:0` — the horizontal `u.to(b)`/`v.to(b)`
interps; the `w.to(b)` Outer→Center interp is vertically reach-0)
because the term reads `grid.metric` fields the halo tracer cannot
follow.

**Why analytic and not the preferred exact adjoint.** The exact
discrete adjoint *was* attempted and *does* work: a probe built the
metric-adjoint of the slope-correction operator `corr` via
`jax.linear_transpose` sandwiched with the physical quadrature weights
(`S = −(N²/W_b)·corrᵀ(W_u·u, W_v·v)`) and drove the **bilinear
physical-metric skew to machine-zero (~2e-16) for arbitrary random
states** — proving that the plain-z-gradient ↔ `Jω`-restoring pair
(couplings 1↔2) is *already* exactly skew on terrain (the J-weighted
`p_hyd` center-cumint `C_J(b) = C(J⊙b)` and the flux-form `Jω`
telescope exactly under the physical measure), so only the slope
correction `corr` (coupling 3) lacked its partner, and `corr`'s exact
adjoint (coupling 4) closes the pair. But that spelling **fights the
staggering** and was not shipped:

- it bakes the grid **quadrature weights** `W_u, W_b` into the buoyancy
  tendency — a buoyancy RHS that depends on the integration measure is
  a transpose-designed artifact, not a physical discretization, and it
  couples this fix to the still-in-flux physical-measure definition (the
  `EnergyMetric` `ps` weight is being fixed on a parallel branch);
- it requires transposing the C-grid interpolation chain and the FTC
  diff — either `jax.linear_transpose` of the pressure-gradient
  machinery **every step** (≈2× the PG cost, fragile, couples the
  stratification module to the core's private internals), or
  hand-assembled adjoint-interpolation rows that only reduce to
  reverse-`.to()` on uniform-periodic-horizontal / uniform-z columns and
  carry boundary/measure-ratio corrections on walled/stretched columns;
- it is against the differentiability policy's spirit (keep the step
  path clean primal operations, no explicit transpose machinery).

The analytic form is the actual physics (`db/dt = −N² w_true`), local,
cheap, trivially forward+reverse differentiable, and delivers the full
**correctness** fix (right continuum limit). It conserves the physical
energy to O(h²) rather than to roundoff, but the scheme is already
stable (§1.3, imaginary spectrum), so the machine-zero property is a
diagnostic nicety, not a correctness need. (If the exact machine-zero
invariant is later wanted, the recipe above is proven — it is an
engineering/robustness call, recorded here so it need not be
re-derived.)

**Gates** (`tests/hydrostatic/test_core_terrain.py`). The accidental
smooth single-mode gate `test_baroclinic_energy_conversion_is_conserved
_to_roundoff` (computational metric) is replaced by
`test_baroclinic_energy_conversion_collapses_under_physical_metric`:
the **bilinear** skew `⟨X, L Y⟩_M + ⟨Y, L X⟩_M` on independent
broadband random states X, Y (all components, fixed seeds) under the
hand-built physical metric M (u/v/b legs J-weighted, `ps` leg lifted to
3D so the volume integral supplies `H/c²`). Pre-fix skew is
resolution-independent (~0.76/0.83/0.85 at n = 16/32/64); with the
term it collapses at ~2nd order — **1.8e-1 / 2.6e-2 / 5.7e-3**
(orders 2.7 / 2.2, asserted `> 1.7`). Plus: rest-state regression (the
term vanishes at u = v = 0), flat byte-identity, and a
`Model.propagator` autodiff gate (grad wrt initial `u`, which the new
term feeds into `db/dt`; finite + central-FD-matched to rtol 1e-4).

**Gate-metric addendum (2026-07-22).** The volume-exact explicit
terrain free surface (`961c2046`, owner ruling 2026-07-19:
`d_t ps = -g T*`, no `1/H(x, y)` division) flipped the barotropic
pair's skew-making surface weight from the depth-mean spelling's
`H(x, y)/c²` to the constant `1/g` **area** weight — which is also the
true physical surface PE `ps²/(2g)` per area. That commit updated
`test_free_surface_terrain.py` but not this gate's hand-built metric
(`_phys_skew` still lifted `ps` to 3D and J-integrated), so the gate
sat red from `961c2046` with a resolution-independent ps-leg leak
(−6.5e-3; total skew ~0.75 flat in n) that mimicked the pre-fix
missing-term signature. A per-leg decomposition confirmed the physics
intact (buoyancy `−N²·w_phys` to 1e-14; baroclinic u+b residual
2nd-order; ps leg machine-zero skew under `1/g`). Fixed 2026-07-22:
the ps leg is now the 2D `1/g` area integral; gate values
1.9e-1 / 2.8e-2 / 6.1e-3 (orders 2.80 / 2.21), matching the §4
post-fix record.

Corrects the stretched+terrain done-entry over-claim ("baroclinic
energy legs machine-precision"): that held only for single-mode states.
Both remainders are now closed: the "w is Jω" output labeling by the
physical-`w` storage flip (`physical_state_components.md` ruling (b),
merge `28a5ff3d`) and the nonhydro2 mapped cousin leak by the §5 probe.

## 5. Cousin-leak probe outcome (2026-07-19)

The §1.6 n-scaling probe ran on dev `8ed5be2a` (recipe: the §1.4
fixed-resolved-broadband methodology; band-limited modes kx, kz <= 4,
states made discretely divergence-free by the model's own Leray
projection, tendencies evaluated through the model machinery — never
`jacfwd`, which silently drops through the CG projection; probe script
preserved in `artifacts/nh_energy_scaling/nh_energy_scaling.py`).
Probe validity gates all passed: flat a = 0 is machine-zero
(bilinear 9.7e-17 — metric + projection + tendency plumbing validated
end to end), the recorded −6.5e-3 order of magnitude reproduces
(quad −1.2e-3 at n = 16, −7.4e-3 at a = 0.4/n = 32), post-projection
divergence <= 8e-10 at every n, the leak is bitwise CG-independent
across a 6-order divergence range, and nodal ≡ fv bitwise (both
matching the §1.6 signature).

**Verdict: CONVERGING — ordinary discretization truncation; the
convention audit is closed.** Physical-metric skew at a = 0.2,
n = 16/32/64/128: bilinear 3.81e-3 / 2.86e-3 / 1.78e-3 / 1.00e-3
(log2 orders 0.41 / 0.69 / 0.83, rising monotonically toward 1 — a
first-order asymptote with a pre-asymptotic transient; the walled-z
zero-flux divergence closure is 1st-order at the terrain rows).
Contrast the hydrostatic pre-fix missing-term signature: resolution-
independent (orders ~0). a-scaling at n = 32: 3.7e-4 / 2.9e-3 /
1.9e-2 at a = 0.1/0.2/0.4 (~a^2.7-3, superlinear — the hydrostatic
missing term was linear in a).

**Mechanism (sharpens the §4 prediction).** The prediction — nonhydro2
stores physical `w`, so `−N²·w` is the correct coupling — is confirmed
and strengthened: the w↔b buoyancy pair is **exactly** skew on terrain
(bilinear raw ≡ 0 to machine precision at every n and a, because the
column Jacobian `J = H(x, y)` is z-independent, so the vertical
staggering commutes with the J-weight bit-exactly). The *entire* leak
lives in the **pressure projection**: the residual equals the
physical-metric pressure-gradient pairing exactly, i.e. the Leray
projector is orthogonal under the solver's SPD product, not under the
physical (J-weighted) energy metric, leaving a first-order-converging
residual skew on the velocity legs. No fix is warranted: the residual
is a genuine discretization truncation that vanishes with resolution
and (superlinearly) with slope.
