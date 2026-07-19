# Immersed closures + fraction-weighted Sadourny plan

Status: **approved 2026-07-19** (owner, in chat). Fourth immersed
residual (roadmap "Immersed partial cells — residuals"), split into
two independent implementation stages plus recorded deferrals:

- **Stage A (closures):** masked harmonic diffusion + free-slip
  friction on immersed grids. Branch: `feat/immersed-closures`.
- **Stage B (Sadourny):** fraction-weighted Sadourny momentum in
  sw2. Branch: `feat/immersed-sadourny`.
- **Deferrals (§5):** no-slip immersed drag, Smagorinsky,
  VerticalMixing — taught errors naming this plan.

Stages A and B touch disjoint files and run in parallel.

## 1. Problem

Two immersed self-rejection surfaces remain:

1. Every closure refuses immersed grids outright (blanket reject in
   `model/closures/base.py`, ~178–186). For harmonic diffusion and
   friction the fraction spelling is mechanical (the IP-D4 pattern
   already shipped for advection), so the blanket reject
   over-refuses.
2. sw2's Sadourny scheme runs on immersed grids
   (`shallowwater2/modules/sadourny.py`, `_advect_immersed`,
   ~722–799) with the thickness transport fraction-weighted (mass
   exact) but the **corner mass fluxes `fu`, `fv` and the kinetic
   energy unweighted** — which breaks the energy-exchange
   antisymmetry exactly at cut cells ("interior-exact, boundary
   approximate").

## 2. Stage A decisions (closures)

- **CL-D1 (per-closure capability).** Replace the blanket reject
  with a per-closure `_supports_immersed` capability (mirroring the
  advection idiom). Harmonic diffusion + friction flip to True;
  Smagorinsky, VerticalMixing, and any no-slip immersed request
  keep taught errors that name this plan and the §5 deferral
  design.
- **CL-D2 (fraction spelling).** The IP-D4 pattern, verbatim in
  spirit: face-α weights the interface stress flux
  (`weight_flux`), the divergence is scaled by the guarded wet
  volume (`scale_divergence`, sealed /θ). The min-rule face α is
  exactly zero across wet/dry faces, so the telescoping wet-region
  conservation is exact by construction.
- **CL-D3 (boundary condition).** Zeroed stress at cut faces
  **is free-slip**, and free-slip is THE immersed boundary
  condition for stage A (mainstream mask practice; matches the
  advective corner convention). It composes with the existing
  wall-retag closure on domain walls (`model/closures/diffusion.py`
  wall-retag precedent, ~281–414): walls keep their machinery,
  immersed faces get α — the two act on different faces.
- **CL-D4 (conservation).** Diffusion conserves wet tracer content
  exactly; friction conserves wet momentum exactly (telescoping).
  First-order stress accuracy at cut faces is the accepted trade
  (MITgcm/NEMO precedent), conditioned by the `min_fraction` floor.
- **CL-D5 (differentiability).** Sealed divides only
  (double-`where` on every /θ, /α); autodiff shard per closure.

### Stage A gates

- **A-G1 (keystone): immersed staircase diffusion ≡ walled
  diffusion at machine zero** (the graded-advection precedent shows
  machine precision is achievable). Same for friction.
- **A-G2 all-wet ≡ unimmersed, bitwise.**
- **A-G3 conservation:** wet tracer content (diffusion) / wet
  momentum (friction) drift machine-zero over a short run with
  genuine partials.
- **A-G4 taught errors tested** (`pytest.raises(..., match=...)`)
  for Smagorinsky + VerticalMixing + no-slip on immersed grids.
- **A-G5 autodiff shards; A-G6 forced-4 invariance.** Mirrored
  tests, ruff zero, coverage by construction.

## 3. Stage B decisions (fraction Sadourny)

- **SA-D1 (corner mass fluxes).** The corner mass fluxes `fu`, `fv`
  carry the **same face α** the thickness transport carries. This
  restores the skew energy antisymmetry between thickness transport
  and momentum exchange — the whole point of the stage.
- **SA-D2 (corner thickness).** The corner thickness in
  `q = ζ/h_corner` is the **wet-count-weighted average** of the
  wet neighbor thicknesses (the NEMO `nn_een_e3f=1` precedent) —
  never a plain 4-average that dilutes with dry zeros.
- **SA-D3 (masking order).** Relative vorticity is masked **before**
  the thickness divide; planetary vorticity stays **outside** the
  mask (NEMO shipped the opposite order for years — bug #773,
  silently corrupted coastal PV; the ordering is load-bearing).
- **SA-D4 (seal).** The `q` divide is double-`where` sealed: a
  floored corner thickness beside a thin cell with ζ ≠ 0 has a
  finite forward value and a singular VJP without the seal.
- **SA-D5 (conservation target).** Wet-weighted **energy**
  conservation at machine zero is the target the spelling must
  meet. Exact **enstrophy** conservation provably does not survive
  fractional corner thickness — documented approximate (the NEMO
  EET precedent; Ketefian & Jacobson 2009 recorded as the
  escalation if exact boundary enstrophy is ever wanted).
- **SA-D6 (corner ζ).** The existing corner-ζ zeroing (free-slip)
  is unchanged — it is what the conservation structure is
  compatible with (and matches CL-D3).

### Stage B gates

- **B-G1 (keystone, new): wet-weighted energy conservation at
  machine zero** on genuine partials (short inviscid run).
- **B-G2 mass conservation unregressed** (existing gate).
- **B-G3 staircase ≡ walled unregressed; B-G4 all-wet ≡
  unimmersed, bitwise.**
- **B-G5 autodiff shard** (the SA-D4 seal is what it certifies);
  **B-G6 forced-4 invariance.** Mirrored tests, ruff zero,
  coverage by construction.

Scope note: sw2 **mapped**+immersed stays a taught error (separate
follow-up); stage B is flat immersed sw2.

## 4. Parallelism & merge

Stage A: `model/closures/*` + mirrored tests. Stage B:
`shallowwater2/modules/sadourny.py` + mirrored tests. Disjoint;
independent branches, merged `--no-ff` sequentially as each passes
its gates (dev races reconciled at merge, the standing pattern).

## 5. Deferrals (recorded designs, taught errors)

- **No-slip immersed drag:** an added mask-keyed side-drag term on
  tangential velocity next to dry cells (the MITgcm `(1 - h_ζ)`
  pattern; the `apply_graded_mask` selector idiom serves the
  neighborhood keying) — never an interior-stress edit. Deferred
  until a use case demands it.
- **Smagorinsky immersed:** the nonlinear coefficient reads shear
  across the mask (the documented coastal-viscosity bias); needs
  wet-only strain rates, and **walled** Smagorinsky must land
  first. Deferred.
- **VerticalMixing immersed:** the implicit column solve assumes
  uniform dz (`model/closures/vertical_mixing.py` ~272–279,
  `model/implicit.py` ~300–311); partial bottom cells need a
  wet-aware variable-dz tridiagonal — the intersection of two
  generalizations the implicit machinery has already deferred.
  Highest real-world value of the three, and its own roadmap item
  when picked up. Deferred.

## 6. Implementation record

### Stage B (fraction Sadourny) — landed on `feat/immersed-sadourny`

`shallowwater2/modules/sadourny.py`, `_advect_immersed` rewritten to
fraction-weight the momentum so the semi-discrete **wet-weighted
energy** conserves at cut cells, not only in the fully-wet interior.

- **SA-D1** the corner mass fluxes are the α-weighted mass fluxes the
  thickness divergence carries (`weight_flux(immersed, u * p_full.to(u))
  .to(zeta)`), interpolated to the corner. This is the whole fix for
  the vorticity-flux exchange: with `fu = Fu_face.to(zeta)` the
  transpose of the `.to(u)` average lands exactly on `fu`, so the
  `+f_v q` / `-f_u q` coupling cancels pointwise (`fu fv q - fv fu q`).
- **SA-D5** the kinetic energy is fraction-weighted,
  `ekin = 0.5(mean(α_u u²) + mean(α_v v²)) / θ` (`_wet_kinetic_energy`,
  sealed): this is the placement that makes the KE-gradient / mass-flux
  pair telescope (`θ ekin = 0.5(mean(α_u u²)+mean(α_v v²))` exactly,
  wet and dry), so the `II_mom + H` half of the budget closes. The
  gravity-momentum / pressure-energy half (`I_mom + P`) closes for any
  KE because `θ dp = -(D_x Fu_face + D_y Fv_face)` with
  `Fu_face = grav_flux + Ro·thick_flux = α_u h.to(u) u`.
- **SA-D2** `_wet_corner_thickness`: the corner `h` is the wet-count
  average `(m h).to(corner) / m.to(corner)` (m the cell wet mask), so a
  partly-dry corner is never diluted by the dry cells' background `c²`;
  collapses to `h.to(corner)` bitwise when all-wet. Orthogonal to
  energy (the exchange cancels for any q), a PV-quality choice.
- **SA-D3/SA-D6** the relative vorticity is masked **before** the
  divide (`zeta = mask_field(immersed, zeta)` then
  `_potential_vorticity(zeta, h_corner)`); the corner-ζ free-slip
  zeroing is unchanged. Masking the numerator (not the finished PV)
  keeps the sealed divide's reverse mode finite and leaves any future
  planetary vorticity outside the mask.
- **SA-D4** every masked divide routes through the shared double-`where`
  seal `_sealed_divide` (the existing `_potential_vorticity` /
  `_sealed_metric_divide` refactored to delegate to it; the corner
  thickness and KE divides added).
- **SA-D5 enstrophy** exact potential enstrophy provably does not
  survive the fractional corner thickness (documented approximate in
  the module + method docstrings, NEMO EET / Ketefian & Jacobson 2009
  as the escalation).

**Measured gates** (`tests/shallowwater2/test_sadourny_immersed.py`,
CPU; forced-4 for B-G6):

- **B-G1** wet-weighted energy rate `|dE/dt|/scale`: **before the α fix
  ≈ 2.4e-3** (periodic-x) / 1.1e-3 (walled-x) — the O(1) antisymmetry
  break; **after ≈ 1.5e-16 / 2.8e-16** (≤ 2e-16 across seeds 3/7/11/20).
- **B-G2** wet mass rate ≈ 1.6e-17 (unregressed).
- **B-G3** staircase tendency vs walled model ≤ 5.6e-17.
- **B-G4** all-wet tendency vs unimmersed. Bitwise recovery was
  attempted seriously and the site is **irreducible**. Diagnosis
  (2026-07-19): an *eager* reconstruction of every Sadourny intermediate
  — `fu`/`fv` (α-weighted), `_wet_kinetic_energy`, `_wet_corner_thickness`,
  `q`, the assembled `du` incl. the final `mask_field` — is **exactly
  bitwise** across the immersed and unimmersed grids (the α/θ folds are
  a true mathematical no-op at α=1/θ=1; the halos are identical (2,2),
  so it is not a halo-width effect). The residual is **JIT-only and
  ≤ 1 ulp** of the field scale (measured worst case **1.0 ulp** over
  seeds × periodic/walled): the immersed branch's HLO carries the extra
  (identity-at-α=1) momentum fraction ops — the `weight_flux` on
  `fu`/`fv` and the `/θ` divide in `_wet_kinetic_energy` — and XLA
  contracts their stencil FMAs differently from the flat branch (the
  documented FMA-contraction class, `core.py` gravity docstring). The
  `p` tendency stays **bitwise**; only the momentum shifts, and on
  walled it coincides in magnitude with the **pre-existing** (unchanged)
  `_gravity_immersed` ≤1-ulp artifact. The named site is the mandated
  SA-D1/SA-D5 weighting, so it cannot be removed without dropping the
  fix. A halo-preserving zeta-mask spelling was tried and gave an
  *identical* residual (it is not the lever), and reverted. **Pinned**
  in `test_all_wet_tendency_matches_unimmersed` to `≤ 32 ulp` of the
  field scale (NOT 1e-13). **Flag stands for owner ratification.**
- **B-G5** `jax.grad` through an 8-step immersed run (genuine
  partials, min_fraction=0) finite and FD-matched to rtol 1e-4 — the
  new corner-thickness and KE seals certified.
- **B-G6** forced-4 (`XLA_FLAGS=--xla_force_host_platform_device_count=4
  FRIDOM_TEST_FORCED_DEVICES=4`) many-vs-one advance ≤ 1e-11.

Ruff clean; the mirrored + immersed suite is 80 passed / 1 skipped (the
B-G6 multi-device test, verified green under forced-4).

**Chart+immersed taught error (closed a silent-wrong-physics gap).**
The stage-B scope note *assumed* `mapped`+immersed was already a taught
error; it was **not**. Discovery (2026-07-19): a chart-coupled (e.g.
spherical) sw2 grid carrying an `ImmersedDomain` built and computed a
tendency without error — the `advect` dispatch tests `chart_coords is
not None` *first*, so it took `_advect_chart` and **silently ignored**
the immersed mask (advecting across the wet-region boundary). This is
now the explicit refusal the scope note assumed: `SadournyAdvection.bind`
raises `NotImplementedError` when the grid carries **both** chart coords
and an immersed domain, naming the silent-wrong-physics reason and the
mapped+immersed composition follow-up. Pinned by
`test_chart_plus_immersed_is_a_taught_error` on the real bind path. (No
pre-existing test relied on the silent behaviour — none built such a
grid.)
