---
status: active
date: 2026-07-18
---

# Graded-mask biased advection on immersed grids

**Goal (owner-ratified 2026-07-18):** close the first immersed
residual (`roadmap/open.md`, IP-D8 taught gate) — make
`UpwindAdvection`/`WENOAdvection` (orders 3/5) run on immersed grids
via a **mask-keyed graded ladder**, generalizing the wall precedent
(`spatial/operators/graded.py`).

## 1. Research basis (2026-07-18, two sweeps)

- **Production practice is unanimous: degrade the order near the
  mask.** NEMO's UBS builds its curvature correction from
  `umask`-zeroed first differences, collapsing to centered-2 at the
  coast; MITgcm's DST3/limited schemes build slopes from
  `maskW`-zeroed differences (its manual flags the boundary treatment
  as "under investigation" and accepts it); MOM6 reconstructs a
  reduced-information monotone parabola at boundary cells. **PALM is
  the direct precedent for the mechanism**: WS5 → WS3 at the second
  point from topography → 1st order adjacent to it, implemented by
  computing *all three* flux orders everywhere and selecting per
  point/direction with precomputed static bit flags — exactly the
  static-select design a jax code wants.
- **Rejected alternatives.** Difference-zeroing (NEMO/MITgcm spelling)
  is a global respelling of perf-tuned kernels, covers no WENO, and
  its implicit degradation diverges from the wall ladder (staircase
  equivalence would fail); it is anyway the ladder's
  `wall="centered2"` special case. Substencil-masked WENO
  (renormalize survivor weights) is deferred — the ladder's weno3
  rung is that with a restricted survivor set. Ghost-cell fill / ILW
  (fidelity ladder point 3, `specs/grid/02_rules.md` §3.7) needs
  sub-cell boundary position the boolean+hFac mask does not carry,
  and naive ghost extrapolation under biased stencils is the branch
  with documented GKS instability. Cut-cell flux/state redistribution
  targets sliver cells the `min_fraction=0.1` floor already excludes.
- Stability note adopted: narrow-real-data-stencil fallback is the
  stable branch of the boundary-WENO literature; the bottom rung
  stays monotone (`upwind1`) by default because a pure centered
  fallback lets grid-scale noise pool at the boundary.

## 2. Decisions

### GA-D1 — keying: per-face distance-to-dry, wall-ladder arithmetic

The rung at each output face is chosen by the **same ladder
arithmetic as walls** (`biased_ladder`/`centered_ladder`, same
`shift` and wall-cell/structural-zero semantics), with the keying
distance `d` = index distance from the face to the nearest **dry
operand DOF** along the axis (in place of distance-to-wall).
Union-window (sign-independent) keying: an order-`p` biased pair's
union window fits in `d` wet cells iff `p ≤ 2d − 1` — exactly the
wall rung formula — so a face-aligned staircase reproduces the wall
ladder **by construction**. Selectors materialize per rung as boolean
fields (`d ≥ t` ≡ a shifted-AND window product of the operand-space
boolean mask); no integer distance field is needed. The `shift`
slots (the wall path's synthesized Dirichlet zeros, e.g. the
closed-face normal velocity under `NO_SLIP`) are **exempt from the
wetness demand** — they are structural zeros the window may read,
exactly as at walls; getting this exemption right is what makes the
`shift = 1` dual path staircase-exact.

### GA-D2 — mechanics: pre-masked operand, full-array rungs, static select

- The operand's storage is multiplied by its halo-synced,
  dry-exterior-filled **boolean mask** before windowing: dry and
  exterior slots become exact zeros. This (a) reproduces the wall
  path's synthesized zeros, (b) makes the interior pass NaN-safe
  regardless of ghost-slot content, (c) seals the VJP at dead slots
  (cotangent × 0), independent of `MaskState`.
- Every rung is evaluated **full-array** with the existing rung
  kernels (`_weighted_windows` static-row sums, `weno_reconstruct`);
  the face value is a nested `jnp.where` over the static selector
  fields, widest-first. The selectors are trace-time-constant
  arrays, so this is constant-folded structure, not data-dependent
  branching — the moral equivalent of graded.py's static index
  partition.
- Sign selection (the left/right `Where`) stays outermost, unchanged.
  The selected-input weno5 optimization keeps serving the widest
  (interior) rung — it already defers to the both-ladders spelling
  where fallbacks apply.

### GA-D3 — selectors are ImmersedDomain-materialized

`grid.immersed` grows a memoized derivation alongside
`fraction()`/`mask()` (keyed on laid-out space + window spec + slip +
negotiated halo) returning the per-rung selector fields. Static
concrete arrays on the ordinary store+sync halo path. Halo demand is
the biased kernels' declared `order//2 + 1`, routed through the
explicit `extra_halo` mechanism the immersed path already uses
(halo tracing is disabled on immersed grids).

### GA-D4 — walls are subsumed on immersed grids

The mask staggering already paints the **dry exterior** on bounded
factors, so the selectors grade toward domain walls automatically:
when `grid.immersed` is present, the mask-keyed kernels are installed
**instead of** the index-patch wall closure — one mechanism, not two.
Unimmersed grids keep the untouched wall path (structural parity, as
IP-D4). The `_check_walled_extent` precondition does not apply to the
mask path: any `α > 0` face has two wet neighbors (min rule), so the
bottom rung is always legal and narrow wet pockets self-serve. The
staircase gate (below) arbitrates the equivalence.

### GA-D5 — both kernels, both families, one knob

Mask keying covers:

- the nodal biased face reconstruction
  (`_BiasedFaceReconstruction`) **and** the centered velocity-face
  interpolation (`_CenteredFaceInterpolation`, `centered_ladder` —
  size 4 at order 5 reaches dry cells);
- the FV `Fallback` reconstruction
  (`spatial/operators/fallback.py`) — immersed nonhydro2 runs FV by
  default (IP-D7), so the FV route is the primary consumer; the
  hydrostatic model exercises the nodal route. Verify the actual
  family dispatch before wiring.

The bottom rung reuses the existing `wall=` knob (`WALL_RUNGS`;
default `"upwind1"`, `"centered2"` optional) — one ladder, one knob.

### GA-D6 — correctness scope

Reconstruction ignores fractions: partial cells enter only as the
IP-D4 `α`/`θ` flux weights (NEMO/MITgcm/MOM6 practice).
Centroid-shifted reconstruction on genuine partials is designed-for,
not built. `_supports_immersed` flips `True` on
`UpwindAdvection`/`WENOAdvection`; the IP-D8 taught error shrinks
accordingly (eigenmode / mapped+immersed / closure gates unchanged).

## 3. Stages and gates

One branch, `feat/immersed-graded-advection`, one worktree.

| Stage | Work | Gate |
|---|---|---|
| **G1 — selectors** (`spatial`) | `ImmersedDomain` selector derivation (GA-D1/D3) | mirrored tests (`tests/spatial/test_immersed_domain.py`): staircase mask reproduces the wall thresholds per rung/shift; slip rules; shift-slot exemption; memoization; forced-4 invariance |
| **G2 — graded core** (`spatial/operators`) | mask-keyed apply (a `graded.py` sibling of `apply_graded_walls`) + `Fallback` wiring (GA-D2/D5) | mirrored tests (`test_graded.py`, `test_fallback*.py`): rung values vs hand-built windows; pre-mask NaN-safety; all-wet ≡ interior kernel |
| **G3 — advection wiring** (`model`) | kernel install when immersed, pre-mask, velocity-interp ladder, `_supports_immersed` flip, halo declaration (GA-D4/D5/D6) | existing wall/immersed advection tests untouched-green |
| **G4 — gates** | new shard `tests/model/modules/test_advection_immersed_graded.py` + autodiff | see below |

G4 gates:

- **staircase equivalence** (the anchor): face-aligned immersed box ≡
  walled graded biased advection, up3/up5/weno5, over 12 jitted
  steps, to the I2 precedent tolerance (~1e-14 or better); covers
  the FV path (nonhydro2) and the nodal path (hydrostatic);
- all-wet immersed ≡ unimmersed (near-bitwise, I4-correction-3
  tolerance);
- uniform tracer preserved exactly on every rung;
- `θ`-weighted tracer conservation to machine zero on genuine
  partials with biased advection active;
- narrow wet pocket (1–2 cells wide) stable and conserving;
- taught-error removal pinned; the remaining IP-D8 gates still pinned;
- autodiff: `jax.grad` FD-match through a short immersed biased run
  (policy shard, rtol 1e-4, ≤ 8³, ≤ 10 steps);
- forced-4 device invariance; ≥ 95% patch coverage; ruff clean;
  smoke file (`tests/nonhydro/test_linear_model.py`).

## 4. Risks

- **`shift = 1` staircase drift** — the dual/velocity windows read
  the structural-zero slot; the ladder-arithmetic reuse (GA-D1) is
  designed to make this exact. If drift appears, fix the selector
  arithmetic against the wall path; never loosen the gate.
- **Cost** — immersed grids pay all rungs everywhere (est. ≤ 2× the
  advection term for weno5, less for upwind). Accepted for
  iteration 1; band-restricted evaluation is a designed-for
  optimization gated on an owner-run GPU A/B (agents never submit
  GPU jobs).
- **Family dispatch** — the FV vs nodal reconstruction routes must
  be verified in code before wiring; the staircase gate covers both.

## 5. Out of scope (designed-for, not precluded)

Substencil-masked WENO weight renormalization (best surviving order
at the mask); ghost-cell fill / ILW (fidelity ladder point 3);
centroid-shifted reconstruction on genuine partials; band-restricted
rung evaluation (perf); mapped + immersed composition (separate
residual).
