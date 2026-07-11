# C1 — Coupling precedent survey

Research report (see [`README.md`](README.md) for status); feeds
[`../09_coupling_designfor.md`](../09_coupling_designfor.md).

## Architecture comparison

| System | Executables | Seq/Conc | Regrid weights | Flux computation | Conservation |
|---|---|---|---|---|---|
| OASIS3-MCT | MPMD or single | both (LAG for conc.) | offline SCRIP / online-at-init | in components (typically atm) | 1st/2nd-order conservative + optional global CONSERV fixer |
| ESMF/NUOPC | single | both (petLists, run sequence) | online/offline ESMF | Mediator (app-written) | conservative available; budgets = author's job |
| CESM CPL7→CMEPS | single, hub-and-spoke | both/mixed | CPL7 offline; CMEPS online | **atm–ocn flux in the mediator** (CMEPS `aoflux_grid = ogrid\|agrid\|xgrid`) | conservative for fluxes, bilinear for states; budget diagnostics |
| GFDL FMS | single | both (serial default) | offline mosaic intersection | **on the exchange grid** | exact by construction |
| ICON + YAC | single or multi | concurrent | **online at init** (parallel spherical clipping) | atmosphere | 1st-order conservative |
| ECMWF IFS–NEMO | single (NEMO as library, 6-call API) | **sequential, shared tasks** | offline SCRIP | atmosphere | **none** — admitted gap, still unfixed |

Cross-cutting: (1) **sequential single-executable coupling is a
deliberate production choice** (ECMWF moved off OASIS concurrency to
NEMO-as-subroutine — lag-free at equal-or-better throughput; FMS
serial default; ClimaCoupler sequential-only); (2) real conservation
lives on **exact geometric intersections** (FMS exchange grid,
CMEPS xgrid) — by construction beats fixers beats hoping.

## Irreducible problems (any architecture)

1. **Windowed flux accumulation fast→slow** (OASIS LOCTRANS, TM673
   accumulate-and-normalize, CESM averaging, ClimaCoupler
   FluxAccumulator, SamudrACE's 20×6h→5d means).
2. **Declared per-field policy for state slow→fast**:
   instantaneous-and-held (HadGEM3/CESM) vs window-averaged
   (IPSL-CM6) — production genuinely differs.
3. **Lag/sequentiality/iteration**: concurrent = previous-window
   forcing, an O(Δt_cpl) splitting error, diurnal-locked (Marti et
   al. 2021: parallel 1-h coupling errors >100% near
   sunrise/sunset); sequential leading-component-first is the
   better one-iteration truncation; **Schwarz used nowhere in
   production** (reference-solution only; can fail on sea-ice
   albedo discontinuities).
4. **Coupler-held state is restart state** (OASIS restart files
   persist partial LOCTRANS sums; CESM rpointer.drv; FMS
   coupler.res).
5. **Clock divisibility + one calendar** (window = integer multiple
   of every dt; Gregorian/no-leap mismatch drifts ~1 d / 4 yr).
6. **Conservative remap for fluxes, bilinear fine for states**
   (Mahadevan et al. 2020) + the FRACAREA/DESTAREA mask dilemma.
7. **Initialization shock** (up to 2× day-1 RMSE, Mulholland et al.
   2015) — a data problem no architecture removes.

## What evaporates in single-program jax

MPI handshakes/put-get/namcouple → function calls over pytrees;
deadlocks → exceptions + the existing panic mechanism; the LAG
mechanism and its restart fragility → unnecessary under sequential;
PE-layout load balancing → gone; **window rollback becomes free**
(immutable carries — Schwarz's rewind machinery costs nothing
structurally, only the k× compute remains); offline weight files →
an operator built at assembly. **Remains**: problems 1–7 — they are
properties of the time axis, geometry, physics; not MPI. ML-era
systems (SamudrACE, DLESyM/Ola) dodge rather than solve (shared
grids at training time; state-only, no conservation).

## Recommendation sketch

Thin hub, sequential, per-window leading-component declared;
host-side `Coupler` driver owning a **declarative exchange table**
(CMEPS addmap/addmrg idea: `(source, transform, time policy, target
AUX)`) + the Regrid operators; receiver-side coupler Modules own the
AUX targets. **Flux computation in a fast-side module inside the
trace** (recomputed every fast step against held slow-state, the
production pattern) with the window-mean accumulated in carry-
resident state — which makes mid-window coupled restart free via
the existing snapshot mechanism. **The one new normative rule: no
coupling state may live as host-side Coupler attributes — anything
with time-integral semantics is a declared field in some model's
carry.** Direct first-order conservative Regrid (on integer-ratio
tensor grids ≈ the grid layer's cell-average restriction), not an
exchange grid — but keep the flux-host a switch (CPL7's decade-long
bake-in regret). Two-policy time vocabulary (ACCUMULATE/AVERAGE vs
INSTANT-hold). Schwarz: design-for via pure window functions, don't
build. Initialization shock → point at OptimalBalance/transforms.
Out of scope: fraction merging, area corrections, global fixers —
ship the budget *diagnostic*, not the fixer.

## Adopt / avoid (sourced in the full report)

Adopt: sequential single-program primary; declarative per-field
exchange table; per-fast-step flux recompute; conservative-for-
fluxes/bilinear-for-states; online weights at assembly;
carry-resident accumulators; component-as-library + dummy-partner
test driver (a scripted set_aux forcing rig); standing budget
diagnostics. Avoid: LAG choreography; hard-coded flux grids;
non-conservative flux remap "for now" (ECMWF's admitted regret);
global fixers; host-side mutable coupler state; building Schwarz;
state-only no-conservation exchange as the long-term story.
