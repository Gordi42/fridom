---
status: research complete, options presented
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
   CG-iteration-independent, nodal ≡ fv bitwise). Whether it is the
   same class (its mapped buoyancy/w convention) or ordinary
   interpolation-transpose truncation is **untested** (no n-scaling
   run); its energy is bounded in time integration.

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
