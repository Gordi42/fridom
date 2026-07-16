---
status: frozen
date: 2026-07-16
---

# Stencil lowering on XLA:GPU — how stencils compile, where advection time goes

Research report answering two questions raised by the Oceananigans
comparison (fridom's edge shrinks as advection arithmetic grows:
linear 1.86x -> centered 1.38x -> upwind5 1.05x -> weno5 1.10x at
512^3 on one A100):

1. For a local stencil `f_i = sum_j a_j f_{i+j}`: what does XLA:GPU
   actually compile — one buffer per shift and a sum over buffers, or
   a single operation without extra buffers?
2. How do higher-order derivatives compile — one buffer per
   derivative stage, or a single fused operation?

Evidence: a pure-jax micro-benchmark study (no fridom) on the A100,
a code map of the fridom hot path, and an external survey of XLA:GPU
fusion internals. All scripts, result JSONs, and saved optimized-HLO
texts: [`stencil_lowering/microbench/`](stencil_lowering/microbench/)
(reproduce via `run_all.sh`; phase-2 A/B in `microbench/phase2/`).
Setup: A100-SXM4-80GB, jaxlib 0.10.2, float64, 256^3 primary /
512^3 spot-checks, both stencil axes 0 and 2. Measured roofline:
f64 triad 1272 GB/s (256^3), 1679 GB/s (512^3); dispatch floor
~67 us/call, so cross-spelling comparisons use a fori_loop T=50
per-iteration probe and 512^3 single calls.

## 1. Headline answers

**Q1 — no extra buffers.** The slice-window weighted sum (fridom's
spelling in `spatial/operators/stencil_kernels.py::apply_stencil`)
compiles to **one loop-emitter fusion kernel with zero temp bytes**
for every width k in {2,3,5,7}: the k static slices fold into the
consumer's index expression, the weighted sum happens in registers,
and the output is written once. `memory_analysis()` reports
`temp_size_in_bytes = 0`; the optimized HLO is a single `fusion`
whose body is k `slice` + k `multiply` + (k-1) `add`. Notably, even
the "bad" spellings avoid extra buffers *at this size*: a
`jnp.roll`-sum (each roll = concatenate of two slices) and a
stack-then-tensordot both still fuse into one kernel with temp = 0.
The real traps are different lowerings entirely:
`lax.conv_general_dilated` becomes a cuDNN custom call (no fusion,
2–4x slower) and a `fori_loop` over taps does not unroll (k
sequential dynamic-slice fusions, 4–5x slower).

**Q2 — XLA's cost model decides per case, and both outcomes occur.**
A composed second derivative (`diff∘diff`) fuses into ONE kernel
with zero temp — the inner diff is recomputed inline at the two
offsets the outer one reads (cheap producer duplication). The
composed biharmonic (`lap∘lap`) does NOT fuse: XLA materializes the
inner Laplacian (one full f64[258^3] = 137 MB temp buffer, 2
kernels) rather than duplicate a 7-point producer into 7 consumers.
The direct 13-point ∇⁴ stencil fuses to 1 kernel / 0 temp but costs
4.0x the flops — and ties the composed form on wall time. So "one
buffer per derivative" happens exactly when duplication would be
too expensive, and XLA's choice was not beatable by hand in either
direction here.

## 2. Micro-benchmark results

### E1 — `sum_j a_j f_{i+j}` spellings (k=5 shown)

Pure-kernel GB/s (256^3 loop probe; 512^3 single call):

| spelling             | fusions | temp | 256^3 ax2 | 256^3 ax0 | 512^3 ax2 | 512^3 ax0 |
|----------------------|--------:|-----:|----------:|----------:|----------:|----------:|
| valid slice-window   |       1 |    0 |       801 |       756 |  **1505** |   **871** |
| jnp.roll sum         |       1 |    0 |       763 |       737 |      1564 |      1311 |
| stack + tensordot    |       1 |    0 |       823 |       784 |         — |         — |
| conv (cuDNN)         |     0¹  |    0 |       484 |       313 |       680 |       388 |
| tap fori_loop        |       5 |    0 |       165 |       166 |         — |         — |

¹ one `__cudnn$convForward` custom call. Stencil width changes
flops, never fusion structure. **Axis effect:** every fused
slice-sum runs ~40% slower along axis 0 (large stride) than axis 2
(contiguous) at 512^3 — coalescing, not fusion, is the differentiator.

### E2 — derivative composition

- `diff∘diff` == direct 3-pt: one kernel, 0 temp, ~990 GB/s both.
- 3-D Laplacian: composed and hand-written 7-pt both 1 kernel /
  0 temp, but the **hand-written 7-point is +40% faster** (890 vs
  625 GB/s) — composed per-axis slicing makes a bulkier fused loop.
- Biharmonic: composed = 2 kernels + 137 MB temp @ 0.24 Gflop;
  direct 13-pt = 1 kernel + 0 temp @ 0.97 Gflop; **equal wall**
  (~0.57 ms). An `optimization_barrier` after the inner Laplacian
  changes nothing — it was already materialized.
- Forcing materialization where fusion was free is ruinous:
  2nd-deriv barrier +57% wall +135 MB; Laplacian barrier +140% wall
  +411 MB. Never break these chains by hand.

### E3 — C-grid flux-form tendency (the real hot pattern)

For every reconstruction (centered-2, upwind-3/5, WENO-5),
**composed per-op functions and a hand-fused per-point expression
produce byte-identical optimized HLO** — same 2 kernels, temp,
flops, wall. Within one jit, XLA erases the difference between
fridom's operator composition and an Oceananigans-style per-point
spelling. The 2-kernel lowering: kernel 1 is a multi-output fusion
emitting the three per-axis flux arrays (404 MB = 3 field-sizes of
temp at 256^3), kernel 2 does the three flux-differences + sum.

| recon     | spelling   |    ms | GB/s | fus | temp MB | Gflop | HLO ops | compile s |
|-----------|------------|------:|-----:|----:|--------:|------:|--------:|----------:|
| centered2 | composed   | 1.061 |  682 |   2 |     404 |  0.39 |      59 |      0.12 |
| centered2 | hand-fused | 1.064 |  679 |   2 |     404 |  0.39 |      59 |      0.08 |
| centered2 | +barriers  | 2.649 |  273 |  11 |     673 |  0.39 |      74 |      0.07 |
| upwind5   | composed   | 1.816 |  398 |   2 |     404 |  1.40 |     159 |      0.23 |
| weno5     | composed   | 2.977 |  243 |   2 |     404 |  6.57 |     500 |      0.61 |
| weno5     | +barriers  | 4.202 |  172 |  11 |     809 |  6.57 |     514 |      0.66 |

Breaking fusion at every op boundary costs +150% (centered) to +41%
(weno) wall and +270–400 MB — an upper bound on what cross-boundary
fusion is worth, and proof the advection gap is **not** composition
overhead. The production multi-GPU workaround
`--xla_disable_hlo_passes=multi_output_fusion` (jax#39100) splits
the flux fusion 2 -> 4 kernels but is free on wall (±10% noise) and
temp on a single GPU.

### E4 — WENO-5 is divide/instruction-bound

At 256^3 f64 the weno5 tendency runs at **21% of the 9.7 TFLOP/s
f64 roof and 17% of measured bandwidth** — neither roof binds. The
uncounted cost is f64 division (~10–40x an FMA): each reconstruction
pays r+1 = 4 divides (`alpha_i = d_i/(beta_i+eps)^2` ×3 plus the
final normalization). Full-f32 is **3.0x faster** at identical
nominal flops (a purely bandwidth-bound kernel would gain ~2x).
There is no cross-axis or cross-branch duplication to fix (per-axis
flops ×3 add up exactly; hoisting shared subexpressions behind a
barrier does not help — free fusion was already optimal).

### E5 — HLO volume

weno5 tendency = 500 optimized-HLO instructions / 592 jaxpr lines /
0.61 s compile vs centered2 = 59 / 115 / 0.12 s: ~8.5x HLO volume
and ~5x compile at the stencil level. This is the stencil-side
driver of the model-level weno-vs-centered compile gap (8.5–10 s vs
~2 s; see [`time_to_first_step.md`](time_to_first_step.md)).

## 3. How this maps onto fridom's hot path

From the code map (all anchors at dev 693258c8):

- Every kernel on the advection path is the slice-window sum — no
  roll/gather/concat/transpose anywhere
  (`stencil_kernels.py:178,226`, `weno.py:449`, `flux_diff.py:219`).
  Given E1, this spelling is structurally optimal; nothing to fix.
- The whole tendency is one XLA graph under the step trace; Field
  boundaries are Python-level only. The hard breaks per step are the
  storage re-pad at the tail of every staggered apply (~30–42x,
  `staggering.py:675`), the ghost-fill `jnp.take` on the ~12 fresh
  flux fields (`tensor.py:1727`), and one `optimization_barrier` per
  state field at the scan-carry seal (`model.py:457`) — all
  deliberate, previously measured choices.
- State fields are sealed once per step to the full negotiated halo
  width; wide stencils do NOT refill repeatedly (`_SYNC_CACHE`,
  `base.py:1609`). Ghost overhead at 256^3 order-5 is ~7% (262^3).
- Per step (advecting {u,v,w,b} on 3 axes = 12 flux terms): centered
  ~30 stencil applies; upwind5 ~42 + 12 `jnp.where` sign-splits;
  weno adds ~96 f64 divides/step. Both biased reconstructions are
  always computed then `where`-selected (`advection.py:2190`) — the
  branchless-upwind tax, 24 wide reconstructions where a per-point
  branching kernel computes ~12.
- The biharmonic closure is composed `diff∘(k·)∘diff` twice
  (`diffusion.py:170,356`) — per E2 this is the right call (¼ the
  flops of a direct 13-point, equal wall, smaller HLO); keep it.

## 4. External grounding (primary sources in agent survey)

- XLA:GPU's loop emitter does **no shared-memory tiling** for
  stencil fusions — neighbor reuse rides on L2 only; only the
  transpose/reduction emitters stage shared memory, and Triton
  codegen is reserved for matmul/softmax-class fusions
  (openxla.org/xla/emitters, /gpu_architecture). A hand kernel that
  stages the halo in shared memory is the one structural trick XLA
  cannot express — and at 256^3–512^3 our fused stencils already
  reach 60–90% of triad bandwidth, so the remaining headroom there
  is bounded.
- Priority fusion duplicates a producer into **all** consumers or
  materializes it (no partial fusion), with code-blowup guards
  (openxla RFC #6407) — observed directly in E2's biharmonic.
- `jnp.roll` = concatenate of two slices (jax 0.10.2
  `lax_numpy.py:8486`); fuses at these sizes but is periodic-only
  and no faster — no reason to prefer it over slices.
- Oceananigans fuses one prognostic variable's whole tendency into
  one KernelAbstractions kernel with ~2 full-size temporaries per
  step (arXiv:2309.06662). Given E3 (composition is free) and E4
  (weno is divide-bound), its remaining advantages over the XLA
  lowering are arithmetic structure and register-staged reuse — not
  kernel count.
- **Oceananigans does NOT branch per point either.** Its linear
  UpwindBiased reconstruction is `ifelse(bias == LeftBias, <full
  left sum>, <full right sum>)` — Julia `ifelse` is a function,
  both arguments always evaluate, lowering to a predicated select
  exactly like XLA's `where` (deliberate warp-divergence avoidance;
  `src/Advection/upwind_biased_reconstruction.jl`). So for linear
  upwind the compute-both tax is symmetric between the two codes.
  **For WENO it is not:** Oceananigans threads the bias into
  stencil-*index* selection (`ifelse` reorders already-loaded tuple
  entries, `weno_interpolants.jl`) and runs the smoothness
  indicators / nonlinear weights / reconstruction ONCE per point,
  where fridom computes both full WENO biases and selects — ~2x the
  nonlinear arithmetic that E4 identifies as the binding cost.
- Pallas on A100 = Triton backend only (Mosaic GPU is Hopper+),
  documented "best-effort, not recommended", with known f64 rough
  edges (jax#23179). An escape hatch of last resort, not a default.
- jax#39100 (the 4-GPU multi-output-fusion DUS-aliasing miscompile)
  is still open upstream; the narrow `multi_output_fusion` disable
  remains the workaround, and E3 re-confirms it costs nothing on
  one GPU. Adjacent open bug: openxla/xla#24186 (DUS + sharding +
  f64).

## 5. Phase 2 — divide-reduction A/B (micro + real model)

Scripts/results: `microbench/phase2/` (RESULTS.md, JSONs, HLO).
Divide cost reference (A100): a lone `x/y` is bandwidth-bound
(equal to multiply); compute-bound f64 divide = 2.85x an f64
multiply and 2.6x an f32 divide.

**The micro and the real model disagree — the real model is
decisive.** Micro E3 weno5 tendency (256^3): single-divide
product-form weights (`n_i = d_i * prod_{j!=i}(beta_j+eps)^2`, one
final divide) −38%; f32-weights hybrid (betas/alphas in f32, f64
flux) −48%; combined −45%; full-f32 −64%.

Real nonhydro2 step (matched Oceananigans config, 256^3;
`WENOAdvection(order=5)`; harness reproduces the repo centered
baseline 7.49 ms/step to 0.1%):

| variant          | ms/step 256^3 | step divides | chunk temp |
|------------------|--------------:|-------------:|-----------:|
| centered (ref)   |          7.49 |    13 f64    |    2.1 GB  |
| weno5 baseline   |         25.64 |   493 f64    |    2.9 GB  |
| weno5 singlediv  | 36.63 (+43%)  |   133 f64    |    6.6 GB  |
| weno5 f32weights | 19.74 (−23%)  | 13 f64+480 f32 |  3.9 GB  |

512^3: baseline 239.6 ms/step (22.5 GB temp); f32-weights 181.8
(−24%); **single-divide OOMs** (~50 GB temp). Correctness over 20
steps: single-divide ≤7e-13 (exact algebra); f32-weights ~1e-8
state drift.

**Why single-divide backfires:** in the full-step fusion context
XLA materializes the product intermediates (temp 2.9 -> 6.6 GB) —
the whole step is memory/fusion-bound even though the isolated
tendency is divide-bound. (At n=32 both variants are identical;
the blowup is an HBM-scale XLA scheduling decision.) Also
noteworthy: the baseline step executes 493 f64 divides where the
algebra needs ~100 — XLA duplicates divide-bearing subgraphs
across fusion boundaries in the real step.

**Standing lessons:** (1) never ship a stencil-spelling change on
micro evidence alone — the fusion context flips verdicts; (2)
watch `memory_analysis` temp bytes as the leading indicator; (3)
f32-overflow landmine: with the real `eps=1e-10` the product-form
weights overflow f32 (peaks ~1e40) — an f32-weights hybrid must
keep the standard per-candidate `alpha = d/(beta+eps)^2` spelling
(peaks ~1e19, in range).

**Verdict: f32-weights hybrid ships (−23..24%, opt-in pending
owner's precision call); single-divide is dead.**

## 6. One-path upwind spellings (derived and validated on CPU)

fridom's biased schemes compute BOTH the left- and right-biased face
reconstructions, then select on the face-velocity sign
(`advection.py:2190,2231`). XLA cannot branch per element, but two
algebraic reformulations execute only one path's arithmetic. CPU
oracle: `microbench/phase3_prep/validate_spellings.py` (all pass).

**Mirror symmetry (exact).** For linear order-3/5 and weno-3/5, the
right-biased reconstruction is exactly the left one on the reversed
window (weight rows reversed bitwise; WENO tables at `weno.py:306`
reverse coeffs and beta rows per candidate while `optimal` d_m stays
attached to its candidate). Window convention: with union window
`U = cells [F - order//2 .. F + order//2]` (order+1 cells around
face F), left reads `U[0..order-1]`, right reads `U[1..order]`.

**(A) Selected-input (linear + WENO):** select the taps, not the
results: `tap_i = where(u>0, U[i], U[order-i])`, then run the LEFT
reconstruction once on the selected taps. The k selects are cheap
fused elementwise ops; the expensive part (for WENO: betas, alphas,
divisions) runs once instead of twice. No new halo (union reach =
today's combined reach). This is exactly Oceananigans' WENO
structure translated to XLA. FP identity: bitwise on u>0 faces (and
on all of weno-3); ~1 ulp on u<=0 faces of linear-3/5 and weno-5
(reversed summation order).

**(B) Dissipation form (linear upwind only):** the exact identity
`flux = u*(c_sym . U) - |u|*(c_diss . U)` with
c_sym = (L+R)/2, c_diss = (R-L)/2 (Wicker & Skamarock 2002; MITgcm
spells upwind-3 this way). No selects at all. Exact rows:
- order 3: c_sym = [-1/12, 7/12, 7/12, -1/12],
  c_diss = [1/12, -1/4, 1/4, -1/12]
- order 5: c_sym = [1/60, -2/15, 37/60, 37/60, -2/15, 1/60],
  c_diss = [-1/60, 1/12, -1/6, 1/6, -1/12, 1/60]
`c_sym == _centered_row(order+1)` exactly — fridom's own cached
table (`advection.py:512`); only c_diss is a new row. `positive =
v + |v|` becomes unnecessary. ~1-2 ulp vs the selected scheme
(different algebra, never bitwise). Identity holds for the
UNLIMITED scheme only (fridom's current mode); a flux limiter would
break the clean split.

Implementation seams: replace the `Where(positive, left, right)` at
`advection.py:2190-2193` / `2231-2234`; (A) needs a
`_BiasedFaceReconstruction` variant taking q + the face-sign field
and doing the tap selects at array level before one left
`weno_reconstruct`; (B) is a centered-row + new-diss-row pair on
the existing `_centered_kernel` machinery. Same halo widths; the
`Where` halo-merge bookkeeping (`select.py:126`) disappears.

**Phase-3 GPU A/B (`microbench/phase3/`, fresh process per
variant, correctness gates at machine precision):**

Micro E3 tendency (256^3): upwind5 selected −28%, dissipation
−18%; weno5 selected −39%, selected+f32weights −62%. As in phase
2, the micro is a poor oracle — the real step flips two of these.

Real nonhydro2 step (matched config; upwind5 baseline newly
recorded):

| variant             | 256^3 ms/step | 512^3 ms/step | temp 256/512 |
|---------------------|--------------:|--------------:|-------------:|
| centered (ref)      |          7.51 |             — |     2.1 GB/— |
| upwind5 baseline    |         13.70 |        127.17 |     3.1/23.6 |
| upwind5 selected    | 14.50 (+5.8%) |             — |         3.1/ |
| upwind5 dissipation | 14.28 (+4.2%) |             — |         3.1/ |
| weno5 baseline      |         25.54 |        239.83 |     2.9/22.5 |
| **weno5 selected**  | **15.48 (−39.4%)** | **128.49 (−46.4%)** | 3.1/24.7 |
| weno5 sel+f32w      | 16.08 (−37%)  | 134.26 (−44%) |     3.1/24.7 |

- **WENO5 selected-input: SHIP.** Halves the step's WENO divides
  (493 -> 253) and the whole nonlinear-weight subgraph for 5 cheap
  tap selects; temp stays flat (+5..10%), so the phase-2
  disqualifier does not fire; results exact vs baseline (~1e-13,
  reversed-sum ulps only). At 512^3 weno5-selected costs the same
  as plain upwind5 (128.5 vs 127.2 ms/step) — the nonlinear-weight
  tax was being paid exactly twice. Supersedes the phase-2
  f32-weights recommendation (bigger win AND exact); stacking f32
  weights on top is a net loss (16.08 > 15.48) — once the divides
  are halved, the casts cost more than they save.
- **Linear upwind5 one-path (both variants): DON'T ship.** The
  micro win reverses to +4..6% in the real step — linear upwind
  has no expensive per-path work to save (no divides), the step is
  bandwidth/fusion-bound, and both-then-select already fuses well.
  Consistent with Oceananigans also computing both sides there.

Implementation cautions found by the harness: the union-window
alignment must include the dual-staggering shift
(`biased_offset(order,"left") + _wall_shift` — omitting
`_wall_shift` leaves the tracer exact but corrupts velocity
self-advection by ~1.6e-4: plausible-looking wrong physics, catch
it with a machine-precision gate); the upwind select lives in
`_face_value` (the array kernel never sees the face velocity), so
the production change is at that level and must preserve the
operator template (resolve_codomain / _ensure_valid / _finalize)
and the HaloTracer delegation.

## 7. Ranked levers

1. **WENO5 selected-input reconstruction: −39% (256^3) / −46%
   (512^3) on the real step, exact, temp flat — the headline
   result; implement it (§6).** Projection onto the Oceananigans
   comparison: fridom's weno5 edge should widen from ~1.10x toward
   the linear-model ~1.8x (projection — re-run the comparison
   suite after landing).
2. Superseded/dead WENO arithmetic levers, kept as negative
   results: f32-weights hybrid (−23/−24% alone, ~1e-8 drift) is
   beaten by selected-input and a net LOSS stacked on top of it;
   single-divide product form is dead (micro −38%, real +43%,
   512^3 OOM, §5). Linear-upwind one-path spellings are dead
   (micro win, real +4..6%, §6).
3. **Axis-0 stencils are ~40% slower than contiguous-axis** at
   512^3 (coalescing) — a layout/loop-order question (x is also the
   sharded axis). Wide blast radius; roadmap-level, not a quick fix.
4. **Do not** rewrite composition: composed == hand-fused
   byte-for-byte; keep the tendency in one jit; keep biharmonic
   composed; never insert barriers into stencil chains.
5. **Do not** use `lax.conv` (cuDNN, 2–4x slower) or tap loops
   (4–5x slower) for stencils; slice-window sums are the right
   primitive. `scan`-folding stencil structure trades runtime for
   compile time — reject for hot kernels.
6. **Compile time:** weno's ~8.5x HLO volume is intrinsic to its
   unrolled expression tree; the only spelling that shrinks both
   HLO and runtime is an opaque custom call (Pallas) — not worth
   the f64/Triton maintenance risk today.
7. **Memory:** the 3-field flux materialization (404 MB at 256^3,
   3.2 GB at 512^3) is XLA's chosen lowering for flux-form
   tendencies regardless of spelling. A single-pass fusion that
   recomputes fluxes at both faces would trade it for ~2x cheap
   reconstruction flops — unmeasured hypothesis; probe only if
   tendency temp memory becomes the binding constraint (it is part
   of the transient-allocation story behind the Oceananigans memory
   gap).

## 8. Open questions

- **Productionizing weno selected-input — DONE 2026-07-16 on
  `perf/weno-selected-input`.** Shipped as a module-private
  `_SelectedFaceReconstruction` operator
  (`nonhydro2/modules/advection.py`): it takes two operands (the sign
  carrier and `q`), does the tap `where`s on the order+1 union window
  and ONE left `weno_reconstruct`, and delegates the signature / the
  halo-negotiation trace / the codomain plumbing to the interned left
  `_BiasedFaceReconstruction` (the union frame *is* that
  reconstruction's, so the `order//2+1` halo is unchanged and the
  dual-staggering `_wall_shift` rides in for free). Walled axes keep
  the interior tap-select and restore the K wall faces per side from
  both graded ladders under the same sign select (byte-identical wall
  faces); periodic axes do no ladder work. `WENOAdvection._face_value`
  uses it; the linear `UpwindAdvection` path stays on both-then-select
  (byte-identical). Production A/B on the shipped code (A100, matched
  config): **weno5 −39.3% @256³ (25.65→15.57 ms/step), −45.9% @512³
  (239.80→129.78)** — reproduces this section's monkeypatch numbers.
  Parity over 20 steps ≤1.9e-13 (weno5) / bitwise (weno3); linear
  upwind5 chunk HLO byte-identical. Gates: the mirrored shard
  `tests/nonhydro2/test_advection_selected.py` (Where(left,right)
  parity on both C-grid directions, both orders, periodic + z-walled,
  the v=0 tie), the existing weno/advection suites, and the A/B record
  [`stencil_lowering/microbench/phase3/IMPLEMENTATION_AB.md`](stencil_lowering/microbench/phase3/IMPLEMENTATION_AB.md).
  Still open: multi-host (`srun -n P`) confirmation of the walled path,
  re-running the Oceananigans comparison, and the forced-4 knife-edge
  divergence test that the kernel-shape roundoff now also tips for
  `weno5` (report, do not retune — a pre-existing `upwind5` knife-edge
  fails identically on the parent).
- The single-pass flux-recompute trade (lever 7) — worth a probe
  when attacking transient memory.
- Whether a per-point sign-branched Pallas kernel could beat the
  selected-input XLA weno enough to justify Triton-f64 risk —
  re-evaluate only after lever 1 lands.
- Re-run the Oceananigans comparison suite after landing lever 1
  to confirm the projected weno5 edge.
