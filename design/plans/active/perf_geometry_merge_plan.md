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

**Status: LANDED 2026-07-13.** Gate met: `ruff check src tests` clean;
**8043 passed, 0 failed** on 1 device; **297 passed** on the forced-4
decomposition/transform leg; **138 passed** on the forced-4 model/field guard leg
(which also clears the open question of whether their new `Variance` intern key
breaks the re-block / transform-plan / solve caches — it does not).

Stage 1 was deliberately *only* mechanical work; the one item that turned out not
to be (1.5) was reverted rather than pushed through.


## 5. Stage 2 — prove the performance survived (needs care; do with the strong model)

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


## 6. Stage 3 — optimize the new geometry code (the actual second half)

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


## Non-goals

- The 2-D device mesh (pencil decomposition) stays out of scope.
- No new physics.
- Do not relax `test_identity_chart_run_is_bitwise_flat` before §3 is
  root-caused.
