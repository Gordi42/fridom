---
title: Merging the geometry line with the performance line
status: active
created: 2026-07-13
owner: Silvano
---

# Merging the geometry line with the performance line

Two lines of work diverged at `c2f0644d` and must be reconciled:

| head | commits | content |
|---|---|---|
| `origin/dev` @ `295bce78` | 80 | mapped / stretched / spherical meshes, terrain-following coordinates, PCG pressure solve, graded near-wall advection, moving geometry / ALE, reworked Coriolis |
| local `dev` @ `7ce7b65d` | 37 | the performance wave: shard-local re-blocking, fused `rfftn`, distributed slab FFT, storage-frame arithmetic + ghost claims, scan unroll, lean carry, non-divisible sharding |
| `feat/distributed-transform-planner` @ `7b921b3b` | 5 (on top of local `dev`) | the transform-plan-driven distributed solve; **retires `slab_fft.py`**. Closes the push gate. |

The geometry line was developed against the *unoptimized* stack. The goal is
(1) merge without losing the performance gains, then (2) optimize the new
geometry code.

`origin/dev`'s own roadmap already assumes this order — `design/roadmap/open.md`
("Performance optimization (in progress, elsewhere)") says the geometry work
lands first and *"the multi-device cost below is expected to be largely absorbed
by"* the performance pass. The two lines are complementary, not competing.


## 1. What the investigation established

### 1.1 The collision surface is tiny

`origin/dev` **never touched** the files carrying the deepest optimizations:
`spatial/decomposition/*` (`tensor.py`, `decomposition.py`, `halo.py`),
`model/time_steppers/*`, `operators/transform.py`, `operators/spectral_solve.py`,
`operators/fourier.py`, `operators/base.py`.

Only **5 source files** are touched by both lines:

| file | theirs | ours | conflict |
|---|---|---|---|
| `nonhydro2/modules/core.py` | +75 | +13/-1 | 3 hunks — both added a ctor kwarg |
| `nonhydro2/model.py` | +25/-7 | +9/-1 | 1 hunk — same |
| `spatial/fields/scalar_field.py` | +67/-2 | +61/-13 | auto-merged |
| `model/model.py` | +12 | +229/-50 | auto-merged |
| `spatial/operators/__init__.py` | +31/-3 | +8 | auto-merged (lazy-import dicts) |

Plus `ROADMAP.md` (they deleted it — it became `design/roadmap/{open,done}.md`)
and `design/README.md` (they rewrote the plan index).

Every conflict is "both sides added a keyword argument next to each other".
`single_precision_solve` (ours, configures the **flat** spectral solve) and
`pressure_iterations` (theirs, configures the **mapped** PCG solve) are
orthogonal and route to different call sites — keep both.

### 1.2 A trial merge already passes

Resolved in a scratch worktree; result: **ruff clean, 8041 passed, 3 failed**.

- 2 failures (`test_single_precision_solve_model_still_projects`,
  `test_second_advance_with_both_options_compiles_nothing`) are **our test
  fixtures**, broken by their opt-in-rotation change: the tests build
  `nh.Model(..., advection=False)` with no Coriolis, so `u`/`v` are now advanced
  by no term and the assembly coverage lint (D1.4) rejects the model. One-line
  fixture fix: pass an explicit `coriolis=`.
- 1 failure is a **real merge-induced regression** — see §3.

### 1.3 Their new physics is gated; the flat path is structurally preserved

Every new capability is behind a **trace-time Python check on a static mesh
attribute** (`mapping is None`, `chart_coords is None`, `mesh.coordinate_map is
None`, `mesh.periodic`, `self._column is None`) — not a runtime `jnp.where`. A
plain periodic Cartesian run compiles the same graph as before. In particular
the mapped PCG pressure solve is reached only via
`nonhydro2/modules/core.py:177` (`if mapping is not None and
mapping.column_corrections`), and their own docstring states the flat grid
*"takes exactly the code path below (zero behavior change)"*.

**Two ungated exceptions**, both found:

1. **The Coriolis default flipped**: `coriolis=None` now installs *no* module
   (`NoCoriolis`/`SphericalCoriolis` were deleted). A default `nh.Model(grid)`
   loses a tendency term and an AUXILIARY carry field. **Any before/after
   benchmark across this merge must pass an explicit `coriolis=` or it will
   report a spurious speedup.**
2. **`sw.DynamicalCore.extra_halo` is unconditionally 2 per coordinate**
   (`shallowwater2/modules/core.py:140-142`) — a "chart worst case" that is not
   gated on `chart_coords`. Masked on default shallow water (Sadourny already
   needs 2), but **linear shallow water goes 1 → 2 halo cells per axis, ~2× the
   exchange volume**. The correctly-gated form appears to be next door in
   `nonhydro2/modules/advection.py:1347-1351`.

   **It is not a two-line fix — attempted and reverted in stage 1.** The
   `extra_halo` declaration *doubles as a halo-trace exemption*: a module that
   declares a width is skipped by the tracer. Returning `None` on flat grids
   un-exempts the module, and the tracer then runs the gravity term against
   `_TracerGrid` — a deliberately minimal stub exposing only `dispatch`
   (`spatial/decomposition/halo.py:274-288`) — which has no `chart_coords`, so
   the term raises. Fixing it properly means either teaching `_TracerGrid` to
   carry `chart_coords` (and deciding what it should report, given the tracer
   only reaches non-exempt modules) or making the term body tracer-safe.
   Deferred to stage 3; note this is **their** pre-existing cost, not
   merge-induced, so deferring it costs nothing against the status quo.

### 1.4 The performance wave has almost no automated guard

This is the biggest risk in the whole exercise, and it is *pre-existing*:

- `git log c2f0644d..dev -- benchmarks/` is **empty**. The perf wave added zero
  benchmarks. Every headline number (115 → 27.8 ms/step, 37.5 → 20.97,
  12.3 → 8.6) exists **only in commit messages**. `benchmarks/results/` is
  gitignored, so the baselines are not reproducible from a clone.
- The CI `benchmark-smoke` job says verbatim *"No timing assertions."*
- Correctness guards are strong (HLO goldens, collective-free assertions,
  compile counters). **Speed guards do not exist.** A merge that silently
  reverts an optimization leaves the suite green.
- The forced-4-device CI leg covers only `tests/spatial/decomposition`,
  `test_slab_fft.py`, `test_spectral_solve.py`. The compile-count guards for the
  chunk `out_shardings` fixed point and the cached re-block callables live in
  `tests/model/` and **run single-device only** — yet both bugs they guard were
  multi-device-only.

### 1.5 Pre-flight bug found

`feat/distributed-transform-planner` renames
`tests/spatial/operators/test_slab_fft.py` → `test_distributed_solve.py`, but
`.github/workflows/tests.yml:63` still hard-codes the old path. **The
multi-device CI leg breaks the moment that branch lands.** Fix in the same
commit.


## 2. Merge order

Land `feat/distributed-transform-planner` onto `dev` **first** (it is ours,
tested, and it closes the stated push gate: *"dev is not to be pushed until the
merged distributed solve is reconciled with the design"*), then merge
`origin/dev`. This resolves `spatial/operators/__init__.py` once, against the
final state, instead of twice.

Their `design/research/xla_spmd_fft_fault.md` independently rediscovered the XLA
SPMD FFT fault and names the transform planner's reshard stages *"the real fix
and… already the designed-for path"* — so the planner branch is what makes their
tall-column mapped solves work. Two of their validation gates are
`single_device`-marked because of that fault
(`tests/validation/test_moving_geometry.py:143,:181`); un-skipping them is a
merge deliverable, not a follow-up.


## 3. The one real regression: identity chart is no longer bitwise flat

`tests/validation/test_spherical_shallowwater.py::test_identity_chart_run_is_bitwise_flat`
asserts that on the identity chart `X = (x, y, 0)` (where `sqrt_g = g_xx = g_yy
= 1` exactly) the shallow-water tendency is **bitwise** equal to the chartless
Cartesian run. It **passes on `origin/dev` and fails on the merged tree.**

Measured: `u`, `v` differ by **1 ULP** (`3.6e-15` absolute, `1.3e-16`
relative); `p` is bitwise exact.

Investigated and **ruled out**:
- *Variance-tagged spaces falling off the storage-frame fast path.* Instrumented
  `_linear_combine`: the chart run takes the fast path **32/32** times, the flat
  run **30/30**. Neither falls back. (`Variance` is part of the intern key, but
  `join` evidently preserves it.)
- *Ghost-claim sync elision.* Forcing a zero claim on the storage-frame route
  does not restore bitwise equality.

Leading hypothesis (unconfirmed): the chart path runs **2 extra
mathematically-identity combines** (× 1.0 metric factors). Pre-merge those ran on
the unpadded true-shape array; post-merge every combine runs on the **padded
storage frame**, which changes the shape of the fused XLA graph — and XLA is free
to contract `mul + add` into an FMA differently between the two graphs. That
would produce exactly a 1-ULP, momentum-only delta.

Only `scalar_field.py` (storage-frame arithmetic) among our 37 commits can reach
a single-device shallow-water *tendency* — no FFT, no stepper, no decomposition
is involved — so the cause is almost certainly there.

**This needs a root-cause session, not a guess.** It is bounded, well-isolated,
and reproducible in 5 seconds. Deferred to the strong-model slot (§5).

### Root cause — confirmed (2026-07-13)

Decisive experiment: run the same comparison under `jax.disable_jit()`.

```
jit on              u: bitwise=False   maxdiff=3.55e-15
jax.disable_jit()   u: bitwise=True    maxdiff=0.00e+00
```

**With the compiler out of the way the two paths are bitwise identical.** The
Python arithmetic is exact — there is no algebraic difference, no stray term, no
bug in either code path. The entire delta is created by XLA at compile time.

Mechanism: the chart momentum tendency runs 2 extra field combines (metric terms
that are algebraically neutral when `g = I`). Field operations used to round-trip
through `unpad → op → pad`, and those round trips acted as **fusion barriers**.
The storage-frame arithmetic deleted them, so XLA now fuses long elementwise
chains and contracts multiply+add pairs into FMAs — and it makes those choices
differently for the 32-op chart chain than for the 30-op flat chain. A contracted
FMA rounds differently from a separate multiply-then-add. Hence one ULP.

Forcing the two graphs to round identically would require reinstating the fusion
barriers (= reverting the storage-frame optimization), or globally suppressing
FMA contraction (= slowing everything down), or making the chart graph
structurally identical to the flat one (impossible in general — the extra metric
terms only vanish in the degenerate identity case).

**Resolution taken (stage 1):** test the invariant where it is actually true.
The tendency comparison is asserted **bitwise under `jax.disable_jit()`** — which
pins the real invariant, that the chart path evaluates the same floating-point
expression as the flat path — and to a few ULP under jit. Demanding bitwise
agreement between two structurally different compiled programs would forbid the
compiler from fusing them differently, i.e. it would forbid the optimization
rather than test the physics.


## 3a. Where their code was written against our *old* seam

Three places where the merge is *correct* but the new geometry code does not get
the benefit of (or actively fights) the new seams. None of these blocks the
merge; all three are Stage-3 material.

### 3a.1 The CG loop tears its iterates down to the true shape every iteration

`operators/krylov.py`'s module docstring states the premise outright:

> *"every carried iterate is the output of field arithmetic, which **routes
> through the storage write path and hence claims zero ghost validity**. The
> rebuild is consequently **exchange-neutral** — it re-declares the state the
> iterates already had."*

Our storage-frame arithmetic invalidated that sentence. Probed on the merged
tree:

```
f.diff('x') claim    : {'x': 1, 'y': 2}
(d + d) claim        : {'x': 1, 'y': 2}   <- arithmetic now PROPAGATES the claim
d.with_data(d.data)  : {'x': 0, 'y': 0}   <- krylov's rebuild DESTROYS it
```

The result stays **correct** (`with_data` → `store` → `pad` at zero claims is
always sound), but the teardown/rebuild is now a net cost: **3 × unpad + 3 × pad
full-array round trips per CG iteration**, × 30 iterations. That is exactly the
per-op round trip the storage-frame combine was built to remove (~9.5 ms of the
62 ms 512³ linear step). Under the *old* semantics arithmetic paid it anyway, so
this is an opportunity missed rather than a regression — the optimization simply
does not reach inside the mapped pressure solve.

Fix: carry the storage-frame arrays (`x._data`) through the `lax.scan` carry and
rebuild at zero claims. The carry treedef stays trivially stable (raw arrays).
Update the stale docstring either way.

### 3a.2 `Grid.measure` is not memoized → one exchange per operator application

`operators/staggering.py:576-584` does `grid.sync(grid.measure(query, name=axis))`
on every mapped-axis application. `Grid.measure` has no cache, and the
`_SYNC_CACHE` in `operators/base.py:1568` is identity-keyed on the field object —
so a freshly built measure field misses it every time. Every `FiniteDifference` /
flux-diff application on a mapped axis therefore emits its own `Grid.sync`: the
"one exchange per operator application" pattern `tests/spatial/test_exchange_counts.py`
exists to forbid, multiplied by the CG iteration count.

Not yet proven to bite: the measure is a trace-time constant, so single-device
sync constant-folds and multi-device collectives may CSE. **The existing
exchange-count test does not cover it** — uniform `IntervalMesh` takes the
scalar-`dx` fast path. Needs a mapped-mesh case added to that test, then
measurement.

### 3a.3 `single_precision_solve` is silently ignored on mapped grids

`nonhydro2/modules/core.py` stores the flag and forwards it to
`SpectralPressureSolver` on the flat path only; `_project_mapped` builds
`MappedPressureSolver` with no precision option (it has none). So on a mapped
grid the flag runs full f64 while **still changing the module treedef** — minting
an extra compiled program with identical behavior. Either thread precision into
the `SpectralSolve` preconditioner or raise/warn when the flag is set on a mapped
grid.


## 3b. Confirmed non-issues

Checked and cleared, so no one re-litigates them:

- **Variance does not break the storage-frame fast path.** Their `join`
  re-applies the variance tag and `_intern` keys on it, so
  `a.function_space is joined` still holds for same-variance operands.
  Instrumented: chart run takes the fast path 32/32, flat 30/30. Only the
  untagged ⊕ tagged lift falls back — correct and rare.
- **Their operators declare halo requirements correctly.** The whole halo-validity
  machinery predates the fork; neither side touched `operators/base.py`,
  `products.py`, `reconstruct.py`. `graded`/`fallback` read only true DOFs, so
  they are immune to the computed-garbage ghosts storage-frame arithmetic leaves
  behind. `integrate.py` sums true-shape data.
- **The ghost-fill vocabulary is unchanged** (`spatial/bc.py` untouched; no
  inhomogeneous Dirichlet anywhere in their diff), so the linear-homogeneity
  premise licensing our `merge_min` ghost claims still holds.
- **`model/model.py`'s +12 lines are inert**: a host-side read-only `modules`
  property. It never enters the jitted carry and cannot perturb the
  `out_shardings` fixed point, the donated carry, the scan unroll, or the
  zero-template builder.
- **No three-way export collision** in `spatial/operators/__init__.py`; the
  planner branch's hunks land on lines their diff never touches.
- **Non-divisible sharding does not break their code**: no `% devices` /
  `// devices` assumptions anywhere in `mapped/graded/krylov/composed/
  coordinate_mapping/mapped_pressure/moving_geometry`, and `patch_physical_ends`
  is already padded-even aware. (Reasoned + single-device tested; **not yet
  exercised on 4 devices**.)


## 3c. The moving-geometry gates: the blocker moved

`tests/validation/test_moving_geometry.py::test_ale_keeps_the_physical_interpretation_in_place`
and `::test_without_ale_the_field_stays_frozen_at_the_nodes` are marked
`single_device` because the tall columns tripped *"a PRE-EXISTING XLA spmd fault
in the mapped projection … (fft_thunk layout RET_CHECK)"*. The expectation
recorded in §2 was that the transform planner's reshard stages would unblock
them.

**They do not — but the failure is no longer the XLA fault.** Un-skipped and run
on forced-4, both now fail with:

```
MeshVelocityCorrection/mesh_velocity: one-sided boundary variants patch
the physical edges at static indices, so 'z' must be undistributed
(layout='local'); reshard first
```

That is a *layout-negotiation* limitation, not a compiler bug: the one-sided
boundary variants need the `z` axis local, and negotiation sharded it. It is
tractable (declare the layout requirement so negotiation keeps `z` local, or
reshard at the module seam) where the `RET_CHECK` was not. The gates stay
`single_device` for now; re-point their skip comments at this reason when the
fix lands.

## 4. Stage 1 — land the merge (mechanical, low risk)

Everything here is understood and rehearsed. No open design questions.

| # | Step | Notes |
|---|---|---|
| 1.1 | Merge `feat/distributed-transform-planner` → `dev` (`--no-ff`) | Closes the push gate. **Fix `.github/workflows/tests.yml:63`** (`test_slab_fft.py` → `test_distributed_solve.py`) in the same commit. |
| 1.2 | Merge `origin/dev` → `dev` (`--no-ff`) | 4 conflicts, resolutions known: keep both kwargs in `nonhydro2/{model,modules/core}.py`; accept their `ROADMAP.md` deletion; take their `design/README.md` and re-add our plan rows. |
| 1.3 | Fix our 2 test fixtures | Pass an explicit `coriolis=` in the two `nonhydro2` tests. |
| 1.4 | Re-home the ROADMAP follow-up | The decomposed/gather-free TensorStore write (old ROADMAP 2.6) → `design/roadmap/open.md`. |
| 1.5 | ~~Gate `sw.DynamicalCore.extra_halo`~~ | **Attempted, reverted** — not mechanical (§1.3(2)). Moved to stage 3. |
| 1.6 | Widen the forced-4 CI leg | `test_exchange_counts`, `test_halo_validity`, `test_step_chunk`, `test_run`, `test_end_to_end`, `test_model` — all verified passing on 4 devices. Closes the gap where multi-device-only bugs were guarded by single-device tests. |
| 1.7 | Resolve the identity-chart failure | Root-caused (§3): bitwise under `jax.disable_jit()`, few-ULP under jit. |
| 1.8 | Un-skip the two `single_device` moving-geometry gates | **Tried; they stay skipped — but the blocker changed.** See §3c. |

**Status: LANDED 2026-07-13.** Gate met: `ruff check src tests` clean;
**8043 passed, 0 failed** on 1 device; **297 passed** on the forced-4
decomposition/transform leg; **138 passed** on the forced-4 model/field guard leg
(which also clears the open question of whether their new `Variance` intern key
breaks the re-block / transform-plan / solve caches — it does not).

Stage 1 was deliberately *only* mechanical work; the one item that turned out not
to be (1.5) was reverted rather than pushed through.


## 4a. Where we stand — measured (2026-07-13, 4x A100-80GB)

**The flat path is unchanged by the merge.** Measured against the numbers
recorded pre-merge, at the same configs:

| flat nonhydro, linear | recorded pre-merge | measured post-merge |
|---|---|---|
| 256^3, 1 GPU | 7.47 | **7.46** |
| 256^3, 4 GPU | 3.54 | **3.54** |
| 512^3, 1 GPU | 57.75 | **59.13** |
| 512^3, 4 GPU | 20.97 | **21.17** |

Within run-to-run noise on all four. The byte-for-byte no-op argument in §1.3
is now also an empirical result, not just a structural one.

**Mapped meshes.** 2D chart (lat-lon sphere) vs flat, and 3D
terrain-following vs flat — ms/step:

| | 1 GPU flat | 1 GPU mapped | 4 GPU flat | 4 GPU mapped |
|---|---|---|---|---|
| sw 1024^2 | 0.46 | 1.63 (3.6x) | 0.87 | 1.40 (1.6x) |
| sw 2048^2 | 1.56 | 5.66 (3.6x) | 1.26 | 2.31 (1.8x) |
| nh 128^3 | 1.33 | 28.6 (21.6x) | 1.45 | 37.6 (26.0x) |
| nh 256^3 | 8.72 | 216.2 (24.8x) | 6.40 | 177.1 (27.7x) |

(3D mapped at the shipped default `pressure_iterations=30`; at the 12 their
own tests use, 10.7x / 11.8x.)

Three results worth carrying forward:

1. **The 694x multi-device blow-up was a harness artifact.** On real GPUs the
   mapped solve *scales* — 4 GPUs are 1.22x faster than 1. The open question
   the roadmap filed is answered, and the answer is that there is no
   multi-device catastrophe. The mapped penalty is ~25x on **both** 1 and 4
   GPUs: it is a *per-iteration* cost, not a communication cost.
2. **One CG iteration costs roughly one entire flat model step**
   (6.85 ms vs 8.72 ms at 256^3, 1 GPU — differencing the 30- and
   12-iteration runs). Extrapolating to zero iterations leaves ~10.8 ms, so
   the metric/measure machinery is only ~25% over flat and **the CG loop is
   ~95% of the mapped step**. Every lever in §6 that touches the iteration
   body is therefore worth ~30x its per-iteration saving.
3. **2D chart cost parallelizes** (3.6x on 1 GPU -> 1.8x on 4): the metric
   operators are fine multi-device. The 3D problem is the solver, not the
   geometry.

**Side finding, not previously recorded: walled flat grids do not scale
multi-device.** `flat_walled` gets 1.36x from 4 GPUs where `flat_periodic`
gets 2.11x (256^3), because the distributed transform path bails on walled
(trig) grids and falls back to the replicated solve. This is a documented
fallback, but it was never priced. It also caps what the mapped
preconditioner can get from multi-GPU, so it is now the top multi-device
item after the CG loop itself.

Method: linear (advection off), AB3, f64, one chunk of 50 steps
(`_chunk_size = steps`), `block_until_ready`, best of 3. Multi-GPU runs
carry `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`
(jax-ml/jax#39100). Script: throwaway, not committed — building the
reproducible harness is still §5 item 2.1.


## 4b. Stages 2 and 3 — executed (2026-07-14, strong-model slot)

Stage 2 is closed; stage 3's cheap levers are resolved — three
landed, two closed as wontfix with the reason recorded, one
**reverted by measurement** — and the expensive lever is quantified
and scoped. Flat-path guard across all of it: every flat/sw suite
case within ±1.6% of baseline on both device configs.

### Stage 2 outcomes

| # | outcome |
|---|---|
| 2.1 | LANDED — `benchmarks/model/bench_step.py` (17 trap-aware step cases), checked-in A100 baselines (`benchmarks/baselines/step-gpu{1,4}.json`), compare workflow in `benchmarks/README.md`, CI smoke of all three suite dirs. Reproduces the §4a ad-hoc numbers to <1%. Bonus catch: the ad-hoc sphere config was latently unstable (fixed dt blows up at 2048² after ~300 steps, past the ad-hoc horizon); the finite guard flagged it, the suite dt now scales 1/n. |
| 2.2 | Closed by 2.1 — the baseline run IS the A/B; flat matches pre-merge. |
| 2.3 | Closed in stage 1 (§3). |
| 2.4 | LANDED — `tests/nonhydro2/test_distributed_projection.py` (forced-4 CI leg): the production projection must resolve the distributed solve on a periodic multi-device grid; the walled replicated fallback is asserted as documented-and-priced. The storage-frame arithmetic fast path was already guarded post-merge by the claim-propagation assertions in `tests/spatial/fields/test_halo_validity.py`. |
| 2.5, 2.6 | Closed in stage 1. |

### The CG iteration budget — measured (256³, 1 A100)

Standalone mapped solve, marginal cost 6.15 ms/iteration:

| piece | ms | share |
|---|---|---|
| preconditioner `M⁻¹ r` (mixed-transform SpectralSolve) | 3.59 | **58%** |
| operator `A p` (mapped flux form) | 1.94 | 32% |
| 3 weighted reductions | ~0.51 | 8% |

(Unpreconditioned marginal 2.78 ms/it — consistent.)

### Convergence vs iteration budget (the lever-6 data)

Relative residual after k iterations (random mean-free rhs, 64³):

| k | 20% slope | 4.5× depth ratio |
|---|---|---|
| 8 | 2.8e-08 | 3.7e-02 |
| 12 | 3.1e-12 | 3.9e-03 |
| 20 | 5.0e-18 | 5.7e-05 |
| 30 | (floor) | 2.6e-07 |

**Verdict: `pressure_iterations=30` stays.** It is not a trace-era
artifact — steep terrain genuinely needs it (the folded-mean
preconditioner degrades with slope). Mild-terrain users can halve
the cost with 12.

And the budget is **resolution-independent**, which makes it a sound
default rather than a tuned constant — iterations to `1e-10`:

| n | mild 1.5× | steep 4.5× |
|---|---|---|
| 32 | 11 | 44 |
| 64 | 11 | 45 |
| 128 | 11 | 45 |
| 192 | 11 | 45 |

The iteration count does not grow with the grid: at 512³ or 1024³ the
same budget holds and only the per-iteration cost scales.

### Stage 3 lever outcomes

| lever | outcome |
|---|---|
| 2 (storage-frame CG carry) | **REVERTED by measurement** — the suite's first real catch. Standalone solve improved, but the in-model mapped step regressed +9.5–23.6% (n-dependent) on ONE GPU; 4-GPU neutral; peak memory *lower* (-7% at 256³). Probe: restoring the true-shape carry on the same tree puts `nh_mapped[n=256,iters=30]` back on the 10.81 s baseline bit-for-bit. Hypothesis (recorded in the krylov docstring with the re-attempt gate): the zero-copy carry couples buffer lifetimes across the scan boundary that the pad/unpad copies decouple, and XLA:GPU single-device buffer assignment loses more than the copies cost. The `ScalarField.storage`/`with_storage` API stays. |
| 3 (Grid.measure memo) | LANDED. No step-time movement on uniform meshes (scalar-dx fast path; integrate's constants fold at trace time) — the win is stretched meshes and trace time, and the sync memo now hits on the stable field object. |
| 4 (batch the reductions) | WONTFIX — chained by data dependence; fusing them is pipelined/s-step CG, out of scope under CS-D2. Recorded in the krylov docstring. |
| 5 (MeshVelocityCorrection memo) | WONTFIX for nonhydro — u/v/w/b sit on distinct staggerings, so the per-field metric derivations do not repeat; only multi-tracer setups sharing a space would benefit. |
| 6 (re-tune iterations) | KEEP 30 (measured above). |
| 7 (precision into the preconditioner) | LANDED — `single_precision_solve` now selects a float32 preconditioner on mapped grids (mixed-precision PCG) instead of being silently ignored. Measured −10% on the 30-iteration solve (188.6 → 169.1 ms at 256³); the walled column re-widens the trig stages, so the gain is below the naive half-the-FFT estimate. Residual floor identical. |
| 8 (gate `sw.DynamicalCore.extra_halo`) | LANDED — the stage-1 blocker was exactly `_TracerGrid.chart_coords`; the stub now answers `None`, which is always correct: a charted module is exempt through its declaration, so only flat-gated bodies reach the tracer. Flat linear shallow water negotiates halo 1 (was 2 — half the exchange volume); chart and Sadourny keep their declared 2. |

### Lever 1 — the preconditioner is 58% of the iteration, and it stays

58% of the iteration is the preconditioner; at 30 iterations that is
~55% of the whole mapped step. It is nonetheless **the right
algorithm, and every cheaper alternative was measured and rejected**
(2026-07-14). Do not re-propose these without reading this section.

**It earns its cost by >25× in wall time, and the margin grows with
n.** Iterations to `1e-10`, mild terrain, vs unpreconditioned CG
(which costs 2.78 ms/it against the preconditioned 6.15 — a 2.2×
per-iteration discount, so it must save >2.2× in iterations to win):

| n | preconditioned | unpreconditioned |
|---|---|---|
| 32 | 11 | >600 (only reached 9.7e-06) |
| 64 | 11 | >600 (only reached 3.7e-03) |
| 128 | 11 | >600 (only reached 1.2e-02) |

The preconditioned system is **mesh-independent**; the unpreconditioned
one is O(h⁻²)-conditioned and gets *worse* with resolution. Dropping
the preconditioner is never right.

**Rejected — per-column tridiagonal, horizontally-averaged
coefficients.** The idea: keep the z-profile of `K^bb` (which the
scalar mean destroys) and solve a tridiagonal per horizontal
wavenumber. Measured: **zero change** (28/45 iterations either way).
Reason: `K^bb = (w_x z²H′² + w_m)/H` is dominated by the vertical
weight `w_m = 1/dsqr = 4`, while the `z²` slope term maxes at ~0.16 —
so the column profile varies by **1.4%** (13.33 → 13.53) and is
already effectively constant. The slope structure is real but
negligible against the weight.

**Rejected — per-column tridiagonal, LOCAL coefficients
(block-Jacobi over vertical lines).** The corrected idea: capture the
horizontal coefficient variation exactly, one tridiagonal per column,
no FFT. Measured at 64³, iterations to `1e-10`:

| terrain | spectral | line-Jacobi | additive (both) |
|---|---|---|---|
| mild 1.5× | 11 | 204 | 51 |
| steep 4.5× | 45 | 221 | 72 |

**4–20× worse**, and the additive combination is worse than the
spectral solve alone. What buys the mesh-independence is the *global
horizontal coupling* — inverting every horizontal mode, including the
smooth ill-conditioned ones. A per-column solve has none, so those
modes return unpreconditioned; local-coefficient accuracy is worth far
less than the coupling it gives up.

**Rejected on algebra — diagonal / Jacobi rescaling.** For any
pointwise `D`, the scaled entry is `A_ij / sqrt(D_i D_j)`: at a cell
the x- and z-couplings are divided by the same `D_i`, so their *ratio*
is unchanged. And the ratio is exactly the error — `K^xx : K^bb =
H : w_m/H = H²/w_m` varies ~20× for a 4.5× depth range. No diagonal
scaling and no constant-coefficient transform can represent a
horizontally-varying anisotropy.

**The one family that could beat it: multigrid with vertical line
smoothing** — coefficient-robust *and* mesh-independent. That is a
real project (restriction/prolongation on the staggered mapped grid,
smoothers, coarse solve, all under a static jit trace), justified only
if steep bathymetry at 45 iterations becomes a real workload.

**What survives is implementation, not algorithm.** The 3.59 ms is the
mixed trig/FFT transform pair on the walled column:

1. **Multi-device**: extend the distributed transform to mixed (trig)
   plans — the replicated fallback caps walled flat scaling at 1.36×
   and the mapped preconditioner inherits it
   (`test_distributed_projection.py` documents the fallback). This is
   the top remaining item.
2. Reduced precision on the transform pair — landed (lever 7, −10%).


## 5. Stage 2 — prove the performance survived (CLOSED 2026-07-14, see §4b)

Stage 1 proves the merged tree is *correct*. It proves nothing about *speed*,
because (§1.4) nothing in the repo does.

| # | Step | Why it is hard |
|---|---|---|
| 2.1 | **Build the perf guard first.** A committed, reproducible A/B harness for the flat-grid step: 256³/512³, 1 and 4 GPUs, with `benchmarks/results/` no longer gitignored (or a checked-in baseline JSON). | Everything else in Stage 2 and 3 depends on being able to *measure*. This is the single highest-value item in the whole plan. |
| 2.2 | A/B the merged tree against pre-merge `dev` on the flat path. | Must pass an **explicit `coriolis=`** (§1.3(1)) or the result is meaningless. |
| 2.3 | Root-cause the identity-chart 1-ULP delta (§3) and take the owner decision. | Subtle floating-point / XLA-fusion reasoning. |
| 2.4 | Close the "is the fast path actually taken?" gaps. | Nothing asserts the production nonhydro solve resolves a distributed solve, nor that field `+` field takes the storage frame. Every *fallback* is tested; the *fast path* is not. A regression that pushes the real solve off the distributed path leaves the suite green and 512³ back at 37.5 ms/step. |
| 2.5 | Re-verify space interning under the new `Variance` intern key, multi-device, with compile counters. | The re-block plan cache, the transform plan cache and the solve cache are all keyed on interned space identity. A cache miss here silently re-traces per call. |
| 2.6 | Widen the forced-4 CI leg to `tests/model/test_step_chunk.py`, `test_run.py`, `test_end_to_end.py`. | Their guards are multi-device bugs guarded by single-device tests. |


## 6. Stage 3 — optimize the new geometry code (cheap levers resolved 2026-07-14, see §4b; lever 1 scoped there)

The cost centre is unambiguous: **the mapped PCG pressure solve**.

Per CG iteration (`operators/krylov.py:303-341`):

| work | count |
|---|---|
| mapped flux-form Laplacian `A p` (+ its halo syncs) | 1 |
| **preconditioner `M⁻¹ r` = a full `SpectralSolve` (FFT + iFFT pair)** | 1 |
| global reductions (`_dot` ×2, mean projection ×1) → all-reduce per shard | 3 |
| axpy field ops | 3 |

Default `pressure_iterations = 30` ⇒ **per timestep: 30 mapped-Laplacian applies,
30 full spectral solves, ~90 global all-reduces.**

Their own measurement (roadmap "Multi-device compile and execution cost",
formerly 3.9, open/unscheduled):

| | 1 device | forced-4 |
|---|---|---|
| jit compile | 0.7 s | 9.1 s |
| jitted per call | 0.28 ms | **194 ms (694×)** |
| HLO lines | 4 750 | 52 200 |

They already did the trace-cost work (`lax.scan` CG: 245k → 468 HLO lines, flat
in iteration count) and a per-solve metric memo. They explicitly did **not**
touch runtime, and flagged the multi-device blow-up as unscheduled — expecting
this performance pass to absorb it.

Ordered levers (to be confirmed by measurement from 2.1):

1. **The preconditioner is a full FFT pair per CG iteration** — and it is *our*
   code, which the transform planner just rewired to distribute. On 4 devices
   this is where the 694× lives. Either make the preconditioner shard-local, or
   let the distributed transform plan carry it. Highest value, and the reason
   the two lines of work had to merge before either could finish.
2. **Carry the CG iterates on the storage frame** (§3a.1) — removes 6 full-array
   pad/unpad round trips per iteration, ×30.
3. **Memoize `Grid.measure`** (§3a.2) — potentially removes one halo exchange per
   mapped operator application, ×30 iterations. Measure before assuming.
4. **Batch the 3 global reductions per iteration** into one all-reduce.
5. **`MeshVelocityCorrection` has no metric memo** (unlike the pressure solver) —
   it re-derives `d{mapped}/d{p}` per corrected field.
6. Re-tune `pressure_iterations` (30 was chosen under the old unrolled trace
   cost, for trace reasons, not convergence reasons).
7. Thread precision into the mapped preconditioner, or reject the flag (§3a.3).
8. Gate `sw.DynamicalCore.extra_halo` on the chart (§1.3(2)) — needs
   `_TracerGrid` to carry `chart_coords`, or a tracer-safe term body.

Levers 2 and 3 are notable in that they are pure *reach* problems: our
optimizations already exist, they simply do not extend inside the mapped solve.


## 7. Sequencing against the strong-model slot

Stage 1 is mechanical and rehearsed → **do it now**. It unblocks everything and
carries no design risk.

Stages 2 and 3 are where the difficulty is concentrated: floating-point
root-causing (§3), building a measurement harness that does not yet exist
(§1.4), and a distributed-solve optimization that spans both lines of work
(§6.1). These are exactly the tasks that benefit from the stronger model, and
none of them is urgent. **Defer them deliberately** rather than half-doing them
today.

*Executed in the strong-model slot 2026-07-14 (§4b). The deferral paid
off concretely: the harness built first (2.1) caught the very next
optimization (lever 2) regressing the mapped step — measured, probed,
reverted same-day. Remaining open work: lever 1's two follow-ups
(§4b), and the moving-geometry layout gates (§3c).*


## 8. The halo-fill campaign (2026-07-14/15): spelling, then site

Two levers, found by the "does the sync copy the field?" question
(2026-07-14) and closed on `perf/halo-sync-fusion`.

### 8a. The spelling of the lazy (consumption-side) fill — the index map

`_fill_axis` is an **index map** (`_axis_map` + one `jnp.take`), commit
`d726f999`. The mechanism, measured on 1x A100 (flat nonhydro 256^3,
ms/step, against a no-fill floor of 5.86 linear / 8.18 advective):

| spelling | linear | advective | why |
|---|---|---|---|
| `concatenate` | 7.46 | 10.61 | fusion ROOT: every unabsorbed fill
  materializes a fresh O(field) buffer (~75x write amplification) |
| `dynamic_update_slice` | 6.04 | 11.03 | ABSORBED into consumers; the
  write chain is re-derived at every offset a wide stencil reads
  (two 2.7 ms projection fusions swallowed 42 DUS each) |
| index map (landed) | 6.03 | 9.36 | absorbed like the DUS (no HBM
  round-trip), O(1) to re-derive like the concatenate |

XLA never materializes a well-fused fill; the spelling decides what the
*absorption* costs. The map is the only spelling cheap in both regimes
— see the `_fill_axis` docstring, which is the contract.

### 8b. The site of the state fill — the carry seal

Even the absorbed map cost the advective step ~1.2 ms/step: the carry
entered every step with ZERO ghost claims (`_reset_ghost_claims`, the
scan-treedef fixed point), so every state field was re-filled at
consumption, inside the step's widest kernels. The fix
(2026-07-15): **seal the state vector at the carry boundary instead**
(`_seal_carry_ghosts` in `model.py`) — sync each rebuilt state field
where its buffer materializes anyway. The seal's fill is
`sync(materialize=True)`: in-place DUS ghost writes behind an
`optimization_barrier`. All three ingredients are load-bearing:

- **write spelling**: at a materialization point an index map becomes
  a fresh O(field) gather buffer (measured: sealing with the map cost
  +1.3 ms on BOTH cases);
- **barrier**: `scan_unroll` (the AB ring renaming) splices up to 3
  steps into one trace, so 2 of 3 seals have live wide-stencil
  consumers that would absorb a bare write chain (measured: +3.2 ms
  advective, the `8c940666` mechanism);
- **dead operand**: every reader (stencil AND elementwise, including
  the next step's update) reads the sealed field, so the pre-seal
  buffer's only user is the seal and XLA's in-place DUS emitter
  patches O(halo) bytes with zero copies. This is why every earlier
  barrier experiment at the CONSUMPTION site failed: there the
  unsynced field stays live (the update reads it) and the barrier
  forces ~18-21 full copies.

Result (1x A100, 256^3, ms/step): linear 6.03 -> **5.79**, advective
9.36 -> **8.66** — at/near the no-fill floor; the residual advective
~0.5 ms is the mid-step lazy fill of the unprojected velocities
(absorbed maps feeding the divergence — correctly spelled: their
operand stays live through the projection's elementwise read).
Chunked-vs-stepwise values are bitwise identical at unroll granularity
(chunk 1 and 3); a while-looped chunk drifts by ~1e-15/50 steps
(codegen-level FP contraction under changed kernel shapes — the
`test_scan_chunk_matches_repeated_chunk1` tolerance precedent).

Follow-up lever (open): seal the unprojected velocities too by making
the projection consume the synced objects, killing the last absorbed
fills of the advective step (~0.5 ms at 256^3).

Known cost: the seal adds ~40 O(halo) kernel launches per step
(~+77 us/step), a fixed latency invisible at production sizes but
+66% on the latency-bound 32^3 toy cases (5.83 -> 9.66 ms/50 steps;
the map commit had made them FASTER than the concat baseline's 7.01).
If tiny-case latency ever matters: seal only ghost-consumed fields
(drops ~1/6), or group the per-field barriers into one per step.

### 8c. Paired finding: the spectral symbol k^2 is rebuilt per step

Confirmed (2026-07-14): the solver symbol is recomputed every step
(XLA even sinks it into the mapped PCG loop on GPU); cost 0.7-1.4% of
the step. Cheapest fix when it is picked up: build the 1-D k leaves in
`spatial/operators/spectral.py` with numpy so they land as HLO
constants. Full precompute (8 N^3 bytes persistent) is why XLA
declines to hoist. Not part of this branch.


## Non-goals

- The 2-D device mesh (pencil decomposition) stays out of scope.
- No new physics.
- Do not relax `test_identity_chart_run_is_bitwise_flat` before §3 is
  root-caused.
