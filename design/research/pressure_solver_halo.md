---
status: frozen
date: 2026-07-18
---

# Pressure-solver halo demand — can the projection derive its own `extra_halo`?

Charge (owner, 2026-07-18): the nonhydro pressure projection declares a
hardcoded `extra_halo = 2`. For the default operators it probably needs
1 — and if the solver ever moved to higher-order operators, a
hardcoded declaration "would silently be wrong". Research how the
solver could understand its halo requirement automatically.

Answer up front: **the premise splits in two.** (i) The hardcoded 2 is
indeed over-provisioned — the true demand of every default
pressure/constraint path is **1** per side (bitwise-verified on the
spectral paths; the hydrostatic surface solve already declares 1).
(ii) The "silently wrong" fear does **not** hold: an under-provisioned
declaration fails **loudly** (the registry operators' consumption
guards raise a taught `ValueError`), and the spectral symbol is built
from the same registry rows the stage applies, so an operator swap
cannot desync stencil from eigenvalue. The stale-declaration failure
modes are wasted bytes or a loud error — never silent physics.
Auto-derivation is nonetheless the right fix (it keeps the value tight
under swaps and removes the literal), and it is feasible today:
`extra_halo` is read after `bind` and the dispatch merge, so the
declaration can resolve the very operator rows the stage applies and
compose their two-sided `reach`. Design in §5; follow-ups in §7.

## 1. Mechanism — what `extra_halo` actually is

- **The declaration does double duty** (`model/module.py:369`): any
  non-`None` spec — including an empty `HaloSpec({})` — *exempts* the
  module's terms/stages from the halo trace; the numeric value is only
  the substitute demand. The wave makers declare `HaloSpec({})` purely
  for the exemption. Assembly reads it twice: step 5 merges the demand
  (`assembly.py:1988`), step 7 builds the exempt set and passes the
  merged spec into negotiation (`assembly.py:2036-2045`), where it
  joins the traced demand by symmetric `merge_max`
  (`decomposition.py:757-761`, `grid.py:853-854`).
- **The exemption is structurally necessary for global solves.** The
  halo trace has no representation of "every output depends on every
  input": a `Transform` contributes `reach = (0, 0)` (its true
  coupling is global, not ghost-shaped), and `HaloTracer.data` raises
  by design (`halo.py:522-528`), so a raw-array spectral solve cannot
  be traced at all. This is why the projection is exempt — not
  conservatism. (The tracer's only "global" notion is the `Reshard`
  reset, `movement.py:123-140`, which *drops* accumulated depth; it
  demands nothing.)
- **The declared value has exactly one runtime effect: storage
  width.** It never supplies halo validity. The full state is sealed
  to full ghost validity at every step's carry boundary
  (`model.py:518-545`, called at `:687` and `:704`), and within a
  stage `_ensure_valid` (`operators/base.py:1622-1666`) re-syncs any
  operand whose validity does not cover the consuming stencil. The
  runtime never "trusts" the declaration.
- **What V-N2 verifies and what it does not.** The exempt module's
  keys/write-gates/result-spaces are validated in the zero-field
  dry run (`composer.py:283-316`); its *halo reach* is validated
  nowhere. The reach protection instead lives on the consumption
  side: `apply_staggered` / `apply_fv_staggered` raise
  "negotiated halo width ... too small for the S-point stencil"
  (`staggering.py:762-768`, `reconstruct.py:355-359`) for every
  registry-dispatched stencil, exempt stage or not.

## 2. The premise test — silent wrongness disproven

Chase the failure chain for an under-declared exempt stage: the stage
applies a registry operator whose `reach` exceeds valid ghosts →
`_ensure_valid` syncs (fills to storage width) → if storage itself is
too narrow, the consumption guard raises the taught error. The only
data-reading ops in the solver paths that bypass the guard are the
hand-rolled 2-point diagonal builders of the CG solvers
(`mapped_pressure.py:249,288`, `immersed_pressure.py:124` —
`jnp.roll`/concat spellings), and those cannot under-read in any
runnable model: the D1.4 coverage lint requires u,v-advancing physics,
and every such term (Coriolis interp, advection, buoyancy staggering)
already floors the negotiated width at 1. Confirmed empirically — the
forced-lowering matrix in §4 produced no NaN, no garbage, and no
crash anywhere; and the mapped CG family is *structurally* 2nd-order
only (`mapped_factor` refuses order > 2, `staggering.py:553-618`), so
the higher-order-swap scenario cannot even reach the hardcoded legs.

Swap-consistency beyond halo: for the spectral paths the discrete
eigenvalue is built from the same registry `("diff", ...)` rows the
stage applies (`build_flat_spectral_solve`, `pressure.py:187-197`;
`spectral_solve.py:322-327`). A higher-order override would move the
applied stencil *and* the symbol together; the only stale thing left
is the halo literal — and that fails loudly (order 6 → reach 3 > 2).
(Order 4 → reach 2 would be covered by the current literal by luck.)

## 3. True demand per path (audited + measured)

| path | ops (data-reading) | true reach/side | declared |
|---|---|---|---|
| nonhydro2 flat spectral `_project` (`nonhydro2/modules/core.py:570`) | div (2-pt), FFT/DCT (global, 0), grad (2-pt) | **1** | 2 |
| mapped/immersed CG (`mapped_pressure.py:929`, `immersed_pressure.py:382`) | registry grad/div legs + 2-pt corner/diagonal rolls | **1** | 2 |
| hydrostatic surface solve (`free_surface.py:782`) | depth-mean div, 2-D solve, grad correction | **1** | **1** (already tight) |
| hydrostatic core terrain DIAGNOSE (`hydrostatic/modules/core.py:319`) | 2-pt diffs/interps, column cumsum | **1** | 2 |
| sw2 gravity, orthogonal chart (`shallowwater2/modules/core.py:345`) | diff + diagonal raise_index | **1** | 2 |
| sw2 gravity, non-orthogonal chart | diff + cross-interp hop | 2 | 2 (tight) |
| sw2 Sadourny / Coriolis corrections (`sadourny.py:630`, `coriolis.py:544`) | vorticity corner chain (diff→interp→interp) | **2** | 2 (tight) |

> **Correction (2026-07-18, found at implementation).** Two cells
> above over-count. (i) sw2 non-orthogonal chart: the true reach is
> **1**, not 2 — the cross-interpolation (face→centre) is
> opposite-biased to the gradient difference (centre→face) it
> re-aligns, so the two-sided window telescopes
> `[0,+1] ⊕ [-1,0] = [-1,+1]`; the "2" was symmetrize-then-sum, the
> very over-counting §1 of `storage_halo_width.md` diagnosed.
> Bitwise-verified at width 1 on three sheared charts (width 0 fails
> the stencil guard). (ii) The hydrostatic *immersed* path (grouped
> under the terrain row) has **no vertical stencil** — the masked
> continuity's column integral is a reduction — so its demand is
> (1,1,**0**); width-0 vertical runs bitwise. The shipped derivation
> (`model/halo_demand.py`, entry in `../roadmap/done.md`) encodes
> both.

The key accounting subtlety, and why the projection's sum-of-parts
"1 + 1 = 2" intuition over-counts: div and grad sit on opposite sides
of a **global transform, which acts as a runtime validity barrier** —
its output carries no ghost validity, is synced once regardless of
width, and the trailing grad then needs only its own reach 1. Legs
separated by a barrier merge by max, not Minkowski sum. (Same shape
as the interval-accounting finding for sync-free chains,
[`storage_halo_width.md`](storage_halo_width.md) §1 — here the
barrier does the tightening instead of the window asymmetry.)

## 4. Empirical matrix (16³, 10 steps, f64, CPU)

`DynamicalCore.extra_halo` forced to 1 and to 0, vs the stock run:
triperiodic and walled-y spectral paths, linear/centered/upwind5 —
negotiated width drops 2 → 1 (upwind5 stays 3, chain floors), final
states **bitwise identical (0.0)** in every case. Mapped-terrain CG,
linear: width 2 → 1, maxdiff 1.1e-16 (~1 ulp, CG inner-product
reassociation under the shape change); mapped centered floors at 2 via
the advection chain, bitwise. Hydrostatic implicit free surface:
already 1, bitwise. Forcing to **0 changes nothing further** — the
physics floor (Coriolis/buoyancy interps) keeps width ≥ 1, which is
precisely why a too-small projection declaration cannot
under-provision a runnable model. Probe scripts:
[`pressure_solver_halo/`](pressure_solver_halo/).

## 5. Design — derive the declaration from the bound operators

Assembly ordering permits it: `extra_halo` is read at steps 5/7,
strictly after the dispatch merge (step 3) and `bind` (step 4), and
conditional declarants (`_charted`, `_immersed`, `_treatment`,
`_column`) already branch on bind-captured state. No declarant yet
computes reach from operators — but the machinery is the same engine
behind the traced width: `OperatorRequirements.reach`,
`staggering.window_reach` / `reach_or`, and the per-side Minkowski sum
of `_chain_requirements` (`base.py:1731-1748`).

Options considered:

1. **Derived declaration (recommended).** A small helper on the
   module (or in `model/module.py`) with the contract *"declare the
   structure, derive the numbers"*: the module states its stage's
   dataflow as legs and barriers — for `_project`:
   `barrier_chain(div_row, BARRIER, grad_row)` — resolving each leg's
   operators from the merged registry (the same rows
   `build_flat_spectral_solve` consumes), Minkowski-summing reaches
   within a leg, and taking the per-side max across barrier-separated
   legs. The projection's declaration then tracks any operator
   override automatically, in both value and symbol, from one source
   of truth. Costs one `bind` (or property-time grid capture) on
   `nonhydro2.DynamicalCore`, which today has none.
2. **Assembly-time lint instead of derivation** — keep literals,
   compare against the derived value, warn on mismatch. Strictly
   weaker: if the derived value is computable, declaring it is
   simpler than checking it. Useful only as a transition aid.
3. **Un-exempt the solver by teaching the trace a global barrier.**
   The reset machinery exists (`Reshard._trace_reset_names`), and
   V-N2's own "sanctioned escape" already points toward real
   operators over raw `.data`. But the projection's body is raw for
   good reasons (the kernel recipe, jax#39100-era shard_map spellings)
   and making it tracer-safe is a refactor far beyond the payoff;
   the sw2 precedent (chart gravity un-exemption, reverted then
   landed via a `_TracerGrid` answer,
   [`../plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md)
   §1.3) shows even small un-exemptions have sharp edges. Long-term
   direction, not this item.

Scope of option 1 when implemented: `nonhydro2.DynamicalCore` (2 → 1),
`HydrostaticCore` terrain/immersed (2 → 1), `sw2.DynamicalCore`
chart-gated gravity (2 → 1 on orthogonal charts, 2 stays on
non-orthogonal). Leave the Sadourny/Coriolis corner-chain 2s as
literals or derive them with the same helper (sequential legs, no
barrier — derivation reproduces 2); leave `VerticalMixing`
(column-dense, 0) and the mask/wave-maker empties untouched.

## 6. The gate no width change may skip

The GPU A/B of the biased-width narrowing
([`storage_halo_gpu_ab.md`](storage_halo_gpu_ab.md)) showed compiled
step time on the A100 is **shape-sensitive beyond byte volume** — the
old `n+8` shape is up to ~11-19% faster than `n+6` at some production
sizes despite more bytes. The centered `n+4 → n+2` saving this
research unlocks (and any derived-declaration width drop) therefore
ships only behind the same gate: a fresh-process GPU step A/B at
production sizes, both schemes, before a default flips.

## 7. Follow-ups

1. Implement option 1 (derived `extra_halo`, "structure declared,
   numbers derived") for the three over-declared cores; centered flat
   nonhydro then negotiates width 1 (storage `n+2`) — behind the §6
   GPU gate.
2. Consumption-side completeness (small, optional): the CG diagonal
   builders' hand-rolled 2-point spellings are the one guard bypass;
   a one-line width assert at solver build would make even the
   physics-floor argument unnecessary.
3. Not pursued: tracing the solver (option 3) — revisit only if the
   projection body is ever rewritten in traced operators.
