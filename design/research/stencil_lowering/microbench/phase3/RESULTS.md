# Phase 3 — one-path upwind spellings (A100-SXM4-80GB, 1 GPU)

Prices two algebraic "one-path" reformulations of upwind-biased
advection, in the isolated micro tendency (M1) **and** the real
nonhydro2 step (M2, the verdict).  Methodology reuses the parent
`microbench/` harness (`common.py`, `e3.py`) unchanged; identical to
Phase 2 (block_until_ready, 5 warmups, median of 40, fori_loop T=50
loop probe, `memory_analysis`/`cost_analysis`/HLO capture).  Scripts:
`p3a_micro.py`, `p3b_model.py`.  Raw JSON in `results/`, per-variant
optimized chunk HLO in `results/hlo/`.  Correctness oracle (host CPU):
`../phase3_prep/validate_spellings.py`.

The two spellings (mirror-symmetry facts proved by the oracle):

- **selected-input** — `recon_right(U) == recon_left(reversed(U))`, so
  the per-face upwind select is pushed to the stencil *inputs* (one
  cheap `where` per tap on the order+1 union window) and a **single**
  left reconstruction runs, instead of computing both biased recons and
  selecting the outputs.  Applies to linear-upwind and WENO.
- **dissipation form** (linear upwind only) —
  `flux = v*(c_sym.U) - |v|*(c_diss.U)`, exact, `c_sym =
  _centered_row(order+1)`, `c_diss = (R_U - L_U)/2`.  Patched as the
  face value `c_sym.U - sign(v)*c_diss.U`.

## M1 — micro E3 composed tendency (256^3, f64 flux)

| variant | ms single | vs base | loop periter ms | f64 div | f32 div | flops G | temp MB | max_abs vs base |
|---|---|---|---|---|---|---|---|---|
| **upwind5** baseline (both+select) | 1.979 | — | 1.232 | 0 | 0 | 1.40 | 404 | — |
| upwind5 selected-input | 1.425 | **-28.0%** | 1.080 | 0 | 0 | 1.09 | 404 | 0 (bitwise) |
| upwind5 dissipation | 1.627 | -17.8% | 1.283 | 0 | 0 | 1.55 | 404 | 5.3e-15 |
| **weno5** baseline (both+select) | 3.106 | — | 4.121 | 24 | 0 | 6.57 | 404 | — |
| weno5 selected-input | 1.886 | **-39.3%** | 2.021 | 12 | 0 | 3.82 | 404 | 0 (bitwise) |
| weno5 selected + f32 weights | 1.191 | **-61.7%** | 0.872 | 0 | 12 | 4.48 | 404 | 2.5e-6 |

- Temp bytes **identical (404 MB)** for every f64 variant: the isolated
  tendency fuses fully regardless of spelling, so the micro is
  compute/divide-bound and every one-path spelling wins (same P2a
  pattern).  selected-input **halves** the WENO divides (24->12) and the
  weight subgraph; the linear-upwind spellings have no divides to save.
- **The micro is a poor oracle for the real step** (the P2 lesson): its
  biggest winner (`weno5 selected+f32w`, -62%) is *not* the real winner,
  and the linear-upwind micro wins (-28%/-18%) **reverse sign** in the
  real step (see M2).

## M2 — real nonhydro2 step (the verdict; fresh process per variant)

Matched Oceananigans-comparison config: periodic, 10000x10000x100,
f0=1e-4, N^2=(50 f0)^2, dt=20, AB3, smooth jet IC, chunk_size=50.
Harness validated: `centered` = 7.509 ms/step matches the repo baseline
(P2b 7.486); `w5_baseline` = 25.543 / 239.830 ms/step matches P2b
(25.640 / 239.62) and its 22450 MB temp exactly.

### ms/step (median ~= min, stable), step chunk divides, compiled temp

| variant | 256^3 ms/step | vs its baseline | 512^3 ms/step | vs baseline | temp 256 MB | temp 512 MB | f64 div | f32 div |
|---|---|---|---|---|---|---|---|---|
| centered (ref) | 7.509 | — | — | — | 2103 | — | 13 | 0 |
| **upwind5 baseline** | **13.699** | — | **127.174** | — | 3076 | 23562 | 13 | 0 |
| upwind5 selected-input | 14.496 | **+5.8% (SLOWER)** | — | — | 3075 | — | 13 | 0 |
| upwind5 dissipation | 14.275 | **+4.2% (SLOWER)** | — | — | 3075 | — | 13 | 0 |
| **weno5 baseline** | **25.543** | — | **239.830** | — | 2932 | 22450 | 493 | 0 |
| **weno5 selected-input** | **15.479** | **-39.4%** | **128.493** | **-46.4%** | 3075 | 24665 | 253 | 0 |
| weno5 selected + f32w | 16.084 | -37.0% | 134.263 | -44.0% | 3075 | 24665 | 13 | 240 |

- **Upwind5 baseline is a new number** (the repo suite lacked it):
  13.699 ms/step at 256^3, 127.174 at 512^3 (temp 23562 MB).  Adds **no
  divides** over centered — the +6.2 ms is pure wide-stencil bandwidth +
  the both-then-select duplication.
- **weno5 selected-input beats P2b's shipped f32w-alone** (19.742 /
  181.76 ms, -23% / -24%) by a wide margin: -39% / -46%, and it is
  **exact** (f32w drifts ~1e-8).  With selected-input, WENO at 512^3
  (128.5 ms) costs essentially the **same as plain linear upwind** (127.2
  ms) — the nonlinear-weight tax was being paid *twice* (once per bias)
  and is now paid once.
- **f32 weights on top of selected-input HURT** (16.084 > 15.479;
  134.263 > 128.493): once the divide count is already halved, the f32
  weight subcomputation's casts cost more than the divides they save.
  Ship plain selected-input, not the combination.

### Temp memory — the P2 disqualifier does NOT fire

selected-input keeps temp **essentially flat**: 2932 -> 3075 MB (+4.9%)
at 256^3, 22450 -> 24665 MB (+9.9%) at 512^3 — the order+1 union window
and 5 tap `where`s cost a little live memory but nowhere near a
blowup.  Contrast **P2 single-divide**, which *doubled* temp
(2932 -> 6553 MB) and **OOM'd at 512^3** (~50 GB): it *added* product
intermediates (s_m, n_m) that XLA materialized in the fused step.
selected-input instead *removes* a whole reconstruction pass, so the
memory/fusion-bound WENO step gets cheaper on exactly the axis that
matters.  This is why the micro win **translates** here and did not for
single-divide.

### Correctness (max_abs state diff after 20 steps, identical IC, 256^3)

| variant | u | v | w | b | finite |
|---|---|---|---|---|---|
| upwind5 selected-input | 2.5e-13 | 7.5e-13 | 4.0e-15 | 6.8e-16 | yes |
| upwind5 dissipation | 2.9e-13 | 7.5e-13 | 3.5e-15 | 7.1e-16 | yes |
| weno5 selected-input | 3.7e-13 | 7.5e-13 | 3.6e-15 | 7.4e-16 | yes |
| weno5 selected + f32w | 5.8e-9 | 1.3e-8 | 9.6e-10 | 1.1e-11 | yes |

(w starts at zero in the IC, so its absolute diff is tiny and its
relative diff meaningless.)  selected-input and dissipation are exact
algebra modulo FP reassociation (~1e-13); f32w matches P2b's ~1e-8
weight-precision drift.  Patch liveness confirmed by the trace-call
counter (24 calls) and the step divide-count delta (493 -> 253).

## Verdict (per scheme, against the P2 lesson)

- **WENO5 selected-input — SHIP.**  This is the clear winner and the
  spelling f32w was groping toward.  -39% at 256^3, -46% at 512^3,
  **exact** (1e-13), temp flat (+5% / +10%, not the doubling that
  disqualified P2 single-divide).  Roughly -10 ms/step at 256^3 and
  -111 ms/step at 512^3 — larger than P2b's f32w win, and without its
  numerical drift.  The micro (-39%) and the real step (-39%/-46%)
  **agree**, unlike single-divide (micro -38%, real +43%): removing a
  reconstruction pass reduces both work and bytes, whereas single-divide
  added live intermediates.  Do **not** stack f32 weights on top — the
  combination is slower.

- **UpwindAdvection5 selected-input / dissipation — DON'T SHIP.**  Both
  win in the micro (-28% / -18%) but **lose in the real step** (+5.8% /
  +4.2%), a P2-style reversal.  Linear upwind has **no divides**, so the
  one-path trick removes no expensive work — it only reshuffles cheap
  weighted sums, and the extra tap `where`s / dissipation ops cost a
  little more than the baseline's two fused rows.  The upwind step is
  bandwidth/fusion-bound and the both-then-select spelling already
  fuses well.

## Gotchas found

1. **Dual-staggering m0 shift** (the one that bit).  The union-window
   alignment must be `biased_offset(order,"left") + _wall_shift(domain)`.
   `_wall_shift` is **1** on the dual `Right->Center` direction (velocity
   self-advection) and 0 on `Center->Right` (tracer).  Omitting it left
   the tracer `b` exact (6e-16) but corrupted every velocity component's
   self-advection by ~1.6e-4 — a plausible-looking but wrong-physics
   error that only the machine-precision correctness gate exposes.
2. **`_face_value` cannot be patched via `weno_reconstruct` alone.**  The
   upwind select needs the face velocity `v`, which the array kernel
   never receives (the select is the `Where` in `_face_value`).  The
   monkeypatch therefore lives at the `_face_value` *method* level and
   builds the tap-selected union window through the real
   `apply_fv_staggered` plumbing, replicating the operator `__call__`
   template (`resolve_codomain` -> `_ensure_valid` halo sync ->
   `_finalize` layout re-attach).  Skipping `_finalize` raises a
   layout-mismatch; skipping the halo-tracer delegation raises
   `HaloTracer has no _data` during negotiation (the union reach equals
   the real reconstruction's, so delegating the trace to the original is
   faithful).
3. **One patch serves both schemes.**  `WENOAdvection` inherits
   `_face_value` from `UpwindAdvection`; the patched body reads
   `self._weighting` (`"linear"`/`"weno"`), so a single method swap
   prices both — but the chunk-executable cache is keyed on the static
   AssemblyRecord (identical across spellings), so **fresh process per
   variant** for timing is mandatory (verified via `fridom_file` and the
   trace counter).
4. **Harness scope.**  The tap-select slices the sign from
   `v_face._data` in the storage frame, which assumes `q` and `v_face`
   share storage length along the axis — true on the periodic
   uniform-halo benchmark (fail-loud shape error otherwise, never silent).
   A production implementation would thread the union window through the
   operator machinery generally (walled grids need the graded-ladder
   path too); the measurement here is faithful (exact vs baseline over
   20 steps).

## Bottom line

Ship **WENO selected-input** (single left reconstruction of the
tap-selected union window).  It supersedes P2b's f32-weights
recommendation: bigger (-39%/-46% vs -23%/-24%), exact instead of
~1e-8, and with flat temp.  It clears the P2 disqualifier that killed
single-divide.  Leave linear `UpwindAdvection` on the both-then-select
spelling (one-path buys nothing without divides to remove).
