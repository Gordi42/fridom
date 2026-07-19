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

(appended as stages land)
