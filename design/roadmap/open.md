---
status: active
date: 2026-07-13
---

# Roadmap — open work

The single open-work tracker for FRIDOM, ordered: **next steps** first,
then **long-term goals**. Shipped phases are recorded in
[`done.md`](done.md). Task numbers are historical and stable — other
records cite them ("ROADMAP 3.5") — so they do not run in order.

The four items that were unplaced were sized on 2026-07-13 (against real
diffstats of comparable landed work); each carries its estimate below.

## Guiding target (end state)

- **No ModelSettings**: a `Model` is assembled from a grid and modules;
  every physical parameter (`f0`, `n2`, `csqr`, ...) lives in a module.
- **Everything is one pytree**: modules can modify anything during
  `update`, and the whole run is a single `jax.jit` call (no Python time
  loop).
- **Grid = function spaces**: fields live on function spaces, operators
  map between spaces, mesh arrays are lazy, and new grid types
  (stretched, spherical) slot into the same abstraction.
- **Fields are ergonomic**: init from callables, dimension reduction
  (`g = f.sel(x=a)`).
- **Models**: nonhydro, shallowwater, and coupled multi-model runs
  (multi-device, later multi-host).

Phases 1 and 2 delivered the first four; the grid is `fridom.spatial`
and the model layer is `fridom.model`.

---

# Next steps

## Performance optimization (in progress, elsewhere)

Cross-cutting: single- and multi-device, compile and runtime, memory,
cpu and gpu. **Being worked on outside this checkout.** The plan is to
land the work in flight here first, then rebase the optimization pass on
top, so that everything written here is optimized in the same sweep.

Nothing else should start optimization work in parallel. **The
multi-device cost below is expected to be largely absorbed by this**,
which is why it sits under it.

## Multi-device compile and execution cost

The mapped pressure solve is correct under decomposition but
disproportionately expensive there. Measured 2026-07-13 (forced-4 host
devices on cpu, 16x16 mapped solve, 12 iterations):

| | 1 device | forced 4 |
|---|---|---|
| jit compile | 0.7 s | 9.1 s |
| jitted per call | 0.28 ms | 194 ms (**694x**) |
| HLO lines | 4 750 | 52 200 |

The `lax.scan` conversion (3.6) removed the O(iterations) blowup, so this
is *not* the unrolled-loop cost: it is what SPMD partitioning of the
sharded metric-scaled stencil chains and the CG's cross-shard
dot-product reductions cost per iteration.

**First task is to measure on real devices** (gpu / multi-host). Forced
host devices exaggerate it — four "devices" contending for one cpu, with
collectives over no interconnect — so it is not yet known how much is
harness artifact. Then: count the collectives per CG iteration (the halo
exchange plus two all-reduces), and see whether the reductions can be
batched or the preconditioner made shard-local. Numbers and method:
[`../plans/done/krylov_scan_plan.md`](../plans/done/krylov_scan_plan.md).
Gates the credibility of 3.3.

## Chart ergonomics (E2)

*Small.* On a chart with a **bounded** axis a model does not assemble at
all without an undiscoverable `merge_overrides` incantation for the
diagonal index moves — the grid seeds `RaiseIndex`/`LowerIndex` with
`diagonal=False`, so the expansion emits a cross-term interpolation chain
that has no legal stencil across a wall. Fix: auto-seed `diagonal=True`
when the derived off-diagonal metric is structurally zero (both target
charts — the lat-lon sphere and the torus — are orthogonal), with a
taught error as the fallback. Then drop the copied recipe from the
`sw.Model` docstring and five chart test files. E1/E3/E4/E5 landed.
[`../plans/active/chart_ergonomics_plan.md`](../plans/active/chart_ergonomics_plan.md)

## Grid follow-ups — the ergonomics half

*Small (~2-3 days, ~300-450 LOC).* **Land this before the docs rebuild
writes Getting Started**, which is what makes it a next step rather than
a nicety:

- **`cartesian.Grid(shape=, extent=, periodic=)`** — still a 9-line stub.
  The old stack had this convenience constructor and the *existing docs
  pages use it*; without a new-stack equivalent, Getting Started opens
  with two deep imports and a generator expression over `IntervalMesh`.
- `ImmutableStateError` on `.data` assignment (today: a bare
  `AttributeError`; the error class already exists in `model/errors.py`).
- Re-export the transform classes and `NodeSet` from `fr.spatial`; type
  `Grid.dispatch`.

**Deferred, not next:** coefficient-space product/power rows. That is not
a missing row but a semantics decision — elementwise multiplication of two
Fourier-coefficient fields is *not* the product of the represented
functions (it is a convolution), so registering it under the same
`("multiply", space)` kind invites silent nonsense. Needs an owner call
first; it blocks nothing.
[`../plans/active/phase2_grid_followups.md`](../plans/active/phase2_grid_followups.md)

## Finite-volume nonhydro

Move the nonhydro model to the average family (`CellAvg` scalars,
face-normal velocities — decision FV-D2 **option A**, owner 2026-07-12).
**No FV code is written yet**; all nine operator gaps are open. Staged:
the four FV symbol rows (the long pole — they block `SpectralSolve` and
hence the pressure solve), the missing conversion rows, an FV tracer slice
— which is where the payoff lands: **exact tracer-mass conservation and
the cut-cell path** — then the C-grid profile with **bitwise parity** as
the gate. Do **not** flip the default wholesale first: walls and mapped
grids work today and would regress.
[`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md)

## High-order stencils on mapped grids — **the spike only**

*Spike: small (~1 day, a throwaway script, no production edits).* The
plan's own stated blocker, and it collapses most of the uncertainty:
1D stretched advection, constant state plus a smooth wave, run with both
candidate divisors — the analytic `grid.metric` Jacobian vs a wide
discrete Jacobian — measuring the free-stream residual and the order at
3 and 5.

Sizing also turned up a **de-risking argument**: the refusals key on
`MappedIntervalMesh`, a per-axis monotone self-map, so the Jacobian is
diagonal and separable — there are no cross-derivative metric terms, which
is where multi-D curvilinear free-stream preservation actually bites. The
identity reduces to the 1D case the spike tests, so the wide discrete
Jacobian should preserve it nearly by construction.

**The full lift is medium (1-2 weeks) — decide it on the spike's
numbers**, and see the payoff caveat under "sized, deferred" below.
[`../plans/active/high_order_mapped_plan.md`](../plans/active/high_order_mapped_plan.md)

## Docs & examples rebuild

The CI skeleton and one executed pilot example landed. What remains is the
bulk: **12 example ports** (only `shallowwater/barotropic_instability.py`
is on the new stack) and the **entire prose page tree** — `docs/source/`
still holds the old-stack pages. Also retires the pre-rendered-media
machinery (`@skip_on_doc_build`, the git-LFS videos).

Two upstream gaps it needs: an `fr.io` root alias, and `pot_vort` on
shallowwater2 (documented, absent). Reader-facing content is
owner-reviewed privately before it reaches `dev` (see AGENTS.md).
[`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md)

## Cutover — retire the old stack

Gated on the docs rebuild (above) being far enough along not to break the
build, plus owner sign-off of the intentional-deltas table. **Physics
parity is closed**; what remains is mechanical. The old packages are still
on disk (136 modules, 107 test files) and still exported from
`src/fridom/__init__.py`.

- Rehome the two survivors (`framework/utils/`, `framework/logger.py`) —
  everything else depends on this decision (likely a top-level
  `fridom.utils`); then rewrite the 45 source + 16 test imports.
- Delete the old packages and their tests; rename `nonhydro2` /
  `shallowwater2`; fix the root exports, `tests/conftest.py`, the CI
  multi-device path, coverage config, benchmarks.

The old "rename `framework2` → `framework`" plan is obsolete: the split
already landed the new stack as `spatial` + `model`, so only the two model
packages still carry a `2`.
Records: [`../plans/active/cutover_parity_plan.md`](../plans/active/cutover_parity_plan.md)
(parity, sign-off) and
[`../plans/active/cutover_checklist.md`](../plans/active/cutover_checklist.md)
(the executable swap list).

## Generalized adiabatic ramping

An `AdiabaticRamping` base transform that ramps declared parameters from a
start to an end value over a ramp period (continuous stage-time `Ramp`
evaluation; curves `"linear"` / `"cosine"` / `"exp"` or a callable), with
**`OptimalBalance` as a subclass** contributing only the balancing policy.
Also buys adiabatic spin-up and parameter continuation. Its dependency
(2.8) has shipped, and `Propagator(updates={param: Ramp(...)})` already
covers much of the mechanism — so this is largely an ergonomics/factoring
task, not new capability.
[`../plans/active/adiabatic_ramping.md`](../plans/active/adiabatic_ramping.md)

---

# Sized, deferred — real work, but nothing is waiting on it

Both were sized on 2026-07-13. Neither is hard to justify *technically*;
both fail the "who wants it" test today. Promote either the moment a
consumer appears.

## Boundary closures, stage 2e — the Robin dynamic `(α, g)` path

*Medium (1-2 weeks; ~400-700 LOC source, ~300-400 test, against the
`9a95202a` analogue of 21 files / +1054).* **No consumer exists**: `BC.ROBIN`
appears only in `bc.py`, two `NotImplementedError` raises, and the tests
asserting those raises. Both models use only Dirichlet/Neumann walls, and
no bottom-drag or flux-BC module exists or is planned. Do it when a model
needs one — otherwise it ships untested-by-use.

Two genuine design questions, unspecified in the record, are why this is
not days: **(1)** "seeded at assembly" fights "compiles once across an α
sweep" — `Grid` is not a pytree and the registry carries no dynamic
fields, so a row seeded at assembly makes α a captured constant and a
swept α *does* retrace; α/g must arrive as traced module params read at
apply time. **(2)** Under sharding the fill runs *inside* `shard_map`, so
an inhomogeneous `g` on a boundary face is a transverse-sharded operand
needing its own `in_specs`/masking.

Stage 2f (the mirror/halo-claim widening) is small but pure perf, gated on
profiling nobody has done. Defer.
[`../plans/active/boundary_plan.md`](../plans/active/boundary_plan.md)

## Eigenmode phase I — `Banded` as a first-class realized map

*Medium (1-2 weeks; ~500-700 LOC source, 8-12 files) for the FD/nodal
vertical route — but **large** if the `Fourier ⊗ Chebyshev` headline is
taken literally.* **No consumer exists**: the walled nonhydro pressure
solve is already purely diagonal (`Fourier×Fourier×Cosine`);
variable-coefficient wall-bounded eigenmodes are served by the shipped
channel engine (`model/eigen_channel.py`); and the terrain-mapped pressure
has cross terms, so it is not block-diagonal and `Banded` would not serve
it either. The one plausible win — a stretch-only vertical map, where a
per-mode Thomas solve would replace fixed-iteration CG exactly — is
speculative and unrequested.

**Landmine:** `d/dz` in *Chebyshev coefficient* space is dense-triangular,
not banded. A real Fourier⊗Chebyshev solve needs the Shen/Galerkin BC
basis and Clenshaw–Curtis measures, neither of which is built. Only the
FD/nodal-vertical variant is actually in reach.
[`../plans/active/projection_eigenmode_roadmap.md`](../plans/active/projection_eigenmode_roadmap.md)

## High-order mapped stencils — the full lift (after the spike)

*Medium (1-2 weeks).* Held back from "next steps" on payoff, not
difficulty. The plan was corrected: the C-grid biased advection tendency
is only 2nd order on **any** mesh once the advecting velocity varies, so
the mapped divisor does **not** buy asymptotic order back for
`UpwindAdvection` / `WENOAdvection`. The real payoff is ENO/dispersion
behaviour restored on stretched meshes (the reason those schemes exist),
honest design order for standalone `WenoReconstruction` and
`FiniteDifference` order > 2, and mapped one-sided FD. That is quality on
stretched meshes, not a new capability headline — worth doing, not urgent.
Run the spike (next steps) and decide on its numbers.

---

# Long-term goals

| #   | Task | Notes |
|-----|------|-------|
| 3.1 | **Hydrostatic model** | Dropped from the cutover (2026-07-08, executed 2026-07-11): the old `hydrostatic` package is removed and this is a greenfield future feature, not a cutover gate. Scope when picked up: linear tendency, hydrostatic pressure solver, advection wiring, eigenvectors. Implicit vertical mixing (and an optional split-explicit free surface) build on 2.5. |
| 3.7 | **Spherical nonhydro** | The 3D spherical chart (`X(lon, lat, h)`, so the metric comes out diagonal and `w = dh/dt` is already physical) needs the C2 chart metrics and the C3 elliptic machinery to meet: the pressure operator becomes the Laplace–Beltrami on the chart — still SPD under the sqrt(g)-weighted product, so the PCG structure carries over, but the operator assembly must be written. Not the first 3D-spherical consumer: a hydrostatic model needs no pressure solve and is the likelier first use (3.1). |
| 3.2 | **Coupled models — design** | `jax.distributed`, field exchange between models on different meshes/devices/processes, a `Coupler` module plus regridding operators, a synchronization schedule. **Pre-designed** in [`../specs/model/09_coupling_designfor.md`](../specs/model/09_coupling_designfor.md) (precedent survey + adversarial walk + architecture; the class specs carry its CS-1..18 constraints, so 3.2 stays a pure addition). |
| 3.3 | **Coupled models — implementation** | Same-process multi-device, then multi-host. Depends on 3.2 — and on 3.9/3.10: decomposed runs must be affordable before coupling them is credible. |

---

## Known gaps in the shipped surface

Small, but they are promises the specs make that the code does not keep
(surfaced by the 2026-07-13 spec audit):

- `model.blank_state()` / `model.state_space(name)` — specified, never
  built. Either add the two one-liners or strike them from the spec.
- `fr.modules.WindowAccumulator` — named as shipping in four places;
  does not exist.
- `add_prognostic` — every stepper family's combine step wants it.

## Cross-cutting rules

- Mirrored tests (95% branch coverage gate), ruff-clean.
- Benchmarked with the 0.1 infrastructure (runtime, compile, memory).
- The old `framework` stays runnable until the cutover; new work does not
  go into it.
- Unstructured grids stay out of scope; the designed-for grid extensions
  must not be precluded by the iteration-1 core.
