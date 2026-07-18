# Boundary trace / scatter machinery (`TraceSpace`) — plan

Status: **phases 1–3 shipped** (dev merge `4adcc933`, 2026-07-18;
outcome record in §9). Remaining: the owner-gated GPU items in §9.
Owner picked Option B in session, 2026-07-18. Immediate consumer: the H7 hydrostatic surface-flux
correction's slice-only `A(1)`
([roadmap entry](../../roadmap/open.md), Oceananigans-gap list;
[`hydrostatic_model_plan.md`](hydrostatic_model_plan.md) §H7).
Design driver: generalized 3D<->2D boundary machinery usable by future
consumers (wind stress, surface buoyancy flux, bottom drag, SST-style
diagnostics), an explicit owner requirement.

## 1. Decision record (2026-07-18)

Four architectures were developed in depth by parallel design studies
against a shared verified framework map; all probe findings below were
reproduced live against the repo.

- **Option 0 — no machinery** (one bespoke fused H7 operator, ~80
  LOC): cheapest for H7 alone, but produces no 2D value at all — a
  boundary diagnostic is inexpressible, and every future consumer
  re-derives the four hard cross-cutting constraints (padded-storage
  boundary indexing, sharded collapse axis, mapped measure slice, AD
  seal) by hand, exactly the `_masked_w_faces`-style hack the operator
  layer exists to absorb. **Rejected: fails the generality
  requirement.**
- **Option A — reuse `ConstantSpace`/`Profile` as the trace codomain**
  (~580 LOC, zero new spaces): every mechanic verified working, and
  the 2D result interoperates with all existing Constant-z machinery
  for free. Fatal generality flaw, probed live: a traced boundary
  value combined with a 3D field via plain `+`/`*` **silently
  broadcasts into every interior level** (`join_factor` lifts
  Constant to full; `constant.py:36-39` itself warns a trace must not
  broadcast). The two most plausible next consumers hit this through
  the *obvious* code (`b_tend + Q_surf/dz` heats the whole column —
  no crash, no NaN). No automatable guard exists: the interop benefit
  and the landmine are the same `is_constant` property. **Rejected:
  silent-wrong-physics trap in exactly the consumers the machinery is
  for** (and contrary to the standing owner preference for explicit
  taught errors over risky auto-magic).
- **Option C — activate the `PointMesh`/`PointValues` stub** (trace
  lands on `mesh.boundary.points`): three intrinsic blockers
  reproduced live, all caused solely by the mesh crossing — the grid
  ownership gate (`_owns_mesh`) rejects boundary-mesh fields; the
  storage layer mis-pads a `PointValues` axis (true-(2,) boundary
  field stored (6,)); the registry never seeds pointwise kinds on
  boundary meshes, so even `q_top * A1_top` fails to dispatch.
  `TensorProductSpace.replace` requires same-mesh factors, so the
  separable-operator codomain path is unavailable. None of that cost
  serves a tensor-product consumer — even lateral wall traces need no
  located geometry (a wall's area is the perpendicular measures).
  **Rejected: strictly dominated by B.** Its one unique asset — a
  located boundary manifold with real coordinates — is graftable onto
  B later as an on-demand view, without rework.
- **Option B — a new same-mesh, non-broadcasting `TraceSpace` factor**:
  **accepted.** Probe-confirmed core: because `TraceSpace` is a
  distinct *non-constant* interned factor on the *parent* mesh, the
  existing strict space algebra auto-rejects Trace x full arithmetic
  (`join_factor` returns None for distinct non-constant same-mesh
  factors) — the wind-stress mistake becomes a loud
  `SpaceMismatchError` at trace time — while
  `TensorProductSpace.replace`, the `SeparableOperator` per-factor
  lift, and Profile-like sharding (sharded horizontals, replicated
  size-1 z) all keep working. No fatal flaw found.

## 2. Space design — `TraceSpace`

New `spaces/trace.py`: `TraceSpace(FunctionSpace)` — a standalone
subclass like `ConstantSpace`/`AverageSpace`, **not** a `NodalSpace`
subclass (it must not enter `isinstance(x, NodalSpace)` branches).

- `shape == (1,)`; `is_constant == False` (the point of the design);
  BC-free (`mesh._free_bc`); `_repr_label = "Trace"`.
- Static attrs, all in the intern key via `_variant_key`:
  `parent_node_set: NodeSet` (OUTER/CENTER/... the trace was taken
  from — the embed reads its target row and staggered offset off
  this), `side: Side` (LOW/HIGH), `depth: int = 0` (signed true-node
  index from that side; **only 0 is implemented** — the slot keeps
  interior fixed-depth slices open at zero present cost, but an
  interior plane sits on an arbitrary z-shard and would force the
  novel partial-gather, so `codomain` rejects `depth != 0` for now).
- Mesh factory `StructuredMesh1D.trace(node_set, side, depth=0)`
  (next to `nodal`/`constant`), interning via `mesh._intern`.
- Decomposition traits `(LOCAL,)` — replicated size-1 axis, no halo,
  same storage shape behavior as `ConstantSpace` (a
  `Center(x)⊗Center(y)⊗Trace(z)` field shards on x/y exactly like a
  `Profile`).

**Algebra** (per the join probe):
- Trace ⊔ identical Trace = Trace (interned identity). Trace ⊔ full
  and Trace ⊔ different-locator Trace reject automatically.
- Trace ⊔ Constant: add an explicit guard in `join_factor` (~4 lines)
  so it **rejects** rather than implicitly lifting the Constant to the
  boundary (implicit relocation = auto-magic; conversions below are
  the sanctioned path).
- Sanctioned conversions (pure retags on identical size-1 storage,
  exact, own VJPs): `as_profile` (Trace -> Constant, opt-in broadcast
  — the bridge into every existing Constant-z consumer: the 2D
  spectral solve, `measure`, export) and `adopt`
  (Constant -> Trace, e.g. wrapping a wind-stress input file as
  boundary-located).

**The type-core split.** The framework's `is_constant` /
`isinstance(ConstantSpace)` predicate currently bundles three roles;
`TraceSpace` needs "collapsed axis" semantics without the broadcast
sanction. Consolidate roles A+B behind a new predicate
`collapses_axis` (True for Constant and Trace), then split:
- Role A, storage/locality (**must add Trace**, hot path):
  `structured_1d.py:213` / `chebyshev.py:179` (LOCAL traits),
  `decomposition/tensor.py:299` (`_width` -> 0), `:322`
  (`_n_shards` -> 1), `:625` (blocking), `halo.py` fill-skips
  (~5 sites).
- Role B, dispatch/axis skipping: `registry.py:461`
  (`_resolve_product` skips Trace like Constant so horizontal ops
  bind), `base.py` `_required_halo` sites. The `SeparableOperator`
  ConstantSpace identity short-circuit (`base.py:715`) needs **no**
  Trace twin (trace/embed run their own `_apply_factor`; horizontal
  ops never bind the z factor).
- Role C, broadcast sanction (**must stay Constant-only** — the
  non-firing on Trace is the design win): `join_factor`,
  `_broadcast_factor`, symbol/composition sites.
- Role D, audits (~15 sites, ~8 need a Trace branch):
  `jacobian_weight.py:78`, `energy.py:364`, `grid.measure`
  (`grid.py:1190` — taught error pointing at trace-of-measure),
  `export.py:258` (squeeze size-1 z), wavenumber guards, io.

Roughly 24 edited sites, ~50 audited (the study's exhaustive list;
re-verify at implementation time). Known non-interchangeability to
document loudly: a Trace field is **not** a Profile —
Constant-assuming 2D machinery (`_flat_spectral`'s
`active_axis_names`, `_depth_mean_div`, `measure`, export) requires
an explicit `as_profile` first.

## 3. Operator set

All follow the `restrict.py` template: `@final @interned`
(`_intern_key` over static state), loud `SpaceMismatchError` on every
unsupported domain (the probe-driven seeding self-unseeds on the
raise), `requirements()` declared.

- **`BoundaryTrace(SeparableOperator)`**, kind `"trace"`,
  `_intern_key = (side, depth)`. `codomain(full factor) ->
  mesh.trace(node_set, side)`; rejects periodic (no boundary),
  coefficient/constant factors, `depth != 0`, and — side-aware — a
  requested side whose boundary DOF is Dirichlet-eliminated
  (`_BOUNDARY_MEMBERSHIP` + `bc.components`; strictly finer than
  `Restriction`'s both-sides guard: a field Dirichlet at one wall is
  still traceable at the other). `requirements: halo=0,
  layout="local"`. `_apply_factor` works in the **true frame** like
  `Integral` (slice `[idx]` with keepdims on `f.data`, then `store`)
  — this sidesteps padded-storage index math entirely; reshard-local
  when the axis is sharded (the proven `cumulative.py:337-349`
  pattern).
- **`BoundaryEmbed(SeparableOperator)`**, kind `"embed"`:
  `Trace -> parent node set`, sparse-3D materialize (value at the
  boundary row, zeros elsewhere; target row from `parent_node_set` +
  `side`). The mutual VJP of the trace.
- **`BoundaryScatterAdd(BinaryOperator)`**, kind `"scatter_add"`:
  `(full, trace) -> full`, row-scatter-ADD into the boundary row —
  bespoke codomain (must not route through the sanctioned lifts,
  which reject Trace x full). Operands share grid + horizontal layout
  (enforced by the binary base). A `set` variant (overwrite) is what
  absorbs `_masked_w_faces` (§6).
- **`AsProfile` / `Adopt`** conversion ops (kinds `"as_profile"` /
  `"adopt"`), pure retags.
- Exposure: **not `.to`** (`_conversion_kind` sees only the source and
  cannot carry the side; a dimension-collapsing trace is not a
  same-names conversion). Field verbs `f.trace(axis, side=...)`,
  `t.embed(axis)`, `scatter_add(f, t)` via `Dispatched` singletons in
  `verbs.py`; seeding rows for the probe machinery in
  `_seed_default_registry`; `operators/__init__.py` lazypimp (3 sites
  per module); pointwise kinds (`multiply` etc.) must resolve on
  Trace-carrying products (Role B skip covers this).

**Measure / metric / immersed: no new accessor.** The single new
primitive is the trace; everything rides it, read at application
(rules 2.3/3.8, uncached, params-aware):
`grid.measure(q.space, "z").trace("z", side=HIGH)` = top-cell
thickness; `grid.metric(..., params=...)` for mapped/moving;
`grid.immersed.fraction(space).trace(...)` for wet weights.

## 4. Distribution

Reshard-local first (proven; declared `layout="local"`): in the
hydrostatic step z is already local where H7 traces (the `cumint` w
diagnosis), so the trace is nearly free there. The trace keeps the
parent's horizontal sharding; the Trace z-factor is replicated —
identical layout shape to `Profile`. The efficient
partial-gather-one-plane pattern (keep horizontals sharded, no column
gather) has no precedent and is **deferred** — revisit only if a
consumer traces a sharded axis outside a local region.
Gate: real multi-host validation under `srun -n 4 --gpu-bind=none`
(manual submission only, per the standing no-auto-cluster-jobs
ruling); forced-4 alone is insufficient.

## 5. Autodiff

Slice and scatter are mutual native VJPs (FD-verified in the studies);
no `custom_vjp` anywhere (step path). The one poison is
`1/dz_top` with a dry/sealed top cell (masked singularity): seal in
the consumer's `_safe_ratio` with the double-`jnp.where` pattern —
mandatory even where the current graph happens to DCE the NaN ghosts.
Ship the standard regression tests: an operator-level grad micro-test
(trace -> scale -> scatter-add) and the `_chunk_body` FD-match through
a short hydrostatic run with the surface closure on (mapped grid,
<=8³, <=10 steps, rtol 1e-4).

## 6. H7 consumer rewrite (`model/modules/advection.py`)

Replace the per-axis full-3D `corr` accumulation
(`_transport`) and the full-3D `q * corr` subtraction (`_advect`)
with the direct boundary term — only the vertical axis contributes a
dropped face (the `_outer_to_inner` seam predicate); the slice form
replaces the 3-axis telescoping sum:

```python
w0     = v_face.trace(vz, side=HIGH)              # 2D Trace
a0     = grid.immersed.fraction(flux_space).trace(vz, HIGH)
dz_top = grid.metric(q.space, ..., params=params).trace(vz, HIGH)
A1_top = _safe_ratio(a0 * w0, dz_top)             # sealed divide
corr2d = q.trace(vz, HIGH) * A1_top               # Trace x Trace
tend   = scatter_add(tend, -(ro * scale2d(corr2d)))
```

Two lowerings exist — (i) sparse-3D `embed` + the existing full-3D
AXPY, (ii) the fully-2D scatter-add above; **measure both** (the
scatter's fusion behavior inside the tendency graph is empirical) and
keep the winner. Expected: correction traffic drops from ~5 full-3D
passes to O(N²) (~1/N_z), reclaiming most of the +46% (`im_centered`
512²x32) / +18–49% (`se_centered` ladder) overhead. **Not bitwise**
for hydrostatic defaults (interior telescoping noise ~1e-16 becomes
exact zeros); the H7 constancy gate (<=1e-13 every cell) only
improves; nonhydro2/shallowwater2 stay bitwise (closure resolves
off). Legacy `surface_flux=False` path untouched.

## 7. Phasing & gates

Branch `feat/boundary-trace` in its own worktree.

- **Phase 1 — machinery**: `TraceSpace` + type-core arm (§2), the five
  operators (§3), verbs/seeding/lazypimp. Gates: mirrored operator
  test files (`test_restrict.py` template: interning, kind,
  requirements, codomain + loud rejections, exactness on uniform AND
  stretched/mapped meshes), multi-device tests (backend-aware
  `invariant()`/`bitwise()`), operator-level autodiff micro-test,
  `test_init.py` updates, ruff zero, 95% patch coverage, framework
  smoke `tests/nonhydro/test_linear_model.py`.
- **Phase 2 — H7 consumer**: §6 rewrite + both-lowerings measurement +
  `_chunk_body` autodiff test. Gates: `tests/hydrostatic` +
  `tests/model/modules` green; H7 validation reruns (512²x32
  implicit+centered stability, constancy gate); **perf merge gate**
  (touches a tendency module): green manually-submitted
  `benchmarks/ci/step_guard.sbatch` on the DKRZ A100 node; the
  comparison-suite resweep number recorded here.
- **Phase 3 — absorb `_masked_w_faces`** (`hydrostatic/core.py:399-437`)
  via the scatter set-variant: removes the one raw `.data` boundary
  hack; bitwise-equivalence test against the old path.
- **Deferred** (recorded, not planned): `depth != 0` interior slices;
  variable-depth (per-column) bottom faces — an immersed-index
  problem no option solves with a static side, orthogonal to this
  plan; the partial-gather distributed pattern; C's located
  `PointMesh` geometry as an on-demand view on `TraceSpace`.

## 8. Success criteria

The roadmap item closes when: the H7 correction runs slice-only with
hydro step overhead vs `surface_flux=False` reduced from +18–49% to
single digits on the comparison ladder (target: centered-hydro
oc/fridom ratio back to ~break-even), all Phase-1/2 gates green, and
the machinery is documented as the sanctioned boundary path (module
docstrings pointing future boundary consumers at
trace/scatter-add instead of `.data` surgery).

## 9. Implementation outcome (2026-07-18)

Phases 1–3 shipped the same day the plan was accepted (dev merge
`4adcc933`; branch commits `99b585bd`/`286f24d3` Phase 1a,
`253b0918` Phase 1b, `947e53e5`+`322661ab` Phase 2, `a6d2fb14`
Phase 3). Combined gates green: ~5,600 tests across
`tests/spatial` + `tests/model` + `tests/hydrostatic` + smoke,
forced-4 multi-device, ruff zero — including two mid-flight `dev`
merges (interval halo accounting, Tier-1 transform guard; no
semantic conflicts, the `collapses_axis` skip excludes collapsed
factors before interval accounting runs).

**Findings that amend the design above:**

- **Slice-form scope (amends §6).** The slice form equals the old
  full-3D correction only where `A(1)` is genuinely boundary-only:
  flat (uniform/stretched-z) grids, and collocated tracers on
  immersed grids (~1e-16 agreement). Measured NOT boundary-only:
  terrain-mapped columns (interior `A(1)` mass ≈1.7 — the mapped
  physical divergence differs from the flux-form continuity the
  w-diagnosis enforces) and immersed staggered momentum (≈0.78).
  Those route per-component to an exact full-3D fallback
  (`_correction_full`, byte-identical to the old accumulation).
  The comparison-ladder configs (flat `se_/im_centered`) all take
  the slice path. The denominator on the slice path is the primal
  `grid.measure` cell width (what `flux_diff` divides by), not a
  mapped `1/J` metric — mapped never reaches the slice path.
- **Lowering switch:** `advection._SURFACE_FLUX_LOWERING`, default
  `"scatter"` (fully-2D, FV-native); `"embed"` is nodal-only (an FV
  `CellAvg` trace embeds onto nodal `Center`) and falls back to
  scatter for FV. GPU A/B pending (below).
- **Operator-set deltas (amend §3):** `BoundaryTrace` also traces FV
  `CellAvg` (recorded as `parent_node_set=CENTER`, so scatter into a
  `CellAvg` tendency works natively) and rejects `FaceAvg`
  (bounded dual cells never touch the wall). `Adopt` is a
  whole-space `UnaryOperator`, not separable (the separable base
  short-circuits to identity on Constant axes). `BoundaryEmbed`
  needs no Dirichlet guard (its codomain is the BC-free parent).
  Trace is seeded with a default `Side.HIGH` variant; specific
  sides are built directly by `f.trace(name, side)`. The halo
  tracer gained trace/embed/as_profile/adopt forwarders (the H7
  closure is halo-traced on the flat path).
- **Cross-node-set boundary writes** (Phase 3): moving a boundary
  value between node sets (Center cell → Outer face) must route
  trace → `as_profile` → `adopt` → `scatter_set` — sanctioned,
  exact (the `_masked_w_faces` rewrite is bitwise on all five
  immersed grid classes tested), but a three-verb chain; if a
  second cross-set consumer appears, add a convenience verb
  (`trace_as(node_set, side)`) — recorded as deferred.
- Local `--cov` SIGABRTs under jax on this cluster (C tracer);
  patch coverage was verified by construction per phase and rides
  the CI gate.

**GPU measurements (2026-07-18 arm sweep, owner-requested; single
A100 l50042, commit `c4497db5`, JSONs in the out-of-tree
`benchmarks/comparison/results/fridom-hydro-gpu1-slice-{off,scatter,embed}.json`):**
three arms (`surface_flux=False` / scatter / embed) over the
advected ladder configs.

- **Centered — §8 met.** Overhead vs off: `se_centered`
  +2.6/+4.4/−8.6/−15.6/−16.0% across the ladder (scatter);
  oc/fridom back to 0.92/0.98/1.04 at the top three rungs, and
  `im_centered` runs 0.99–1.12 (faster than oc at the top rungs,
  all five rungs stable; its off arm reproduces the documented
  pre-H7 instability at rf ≥ 23). Vs the pre-slice shipped state
  (`565eaa51`): −13..−17% se, −19..−23% im.
- **A/B verdict is per-scheme:** scatter wins centered everywhere;
  on the (pre-reroute, see below) all-slice weno5, embed won
  (+1.4..+5.8% vs off; scatter +8.7..+16.3%). Landed as a
  per-scheme default (`_surface_flux_lowering` ClassVar; module
  `_SURFACE_FLUX_LOWERING: str | None = None` is now a global
  override knob), merge `893e82a8`.
- **Slice-exactness audit (same day) — scope narrowed again.** The
  constancy oracle convicted the slice for order-5 biased
  **staggered momentum**: `_surface_boundary_term` relocates the
  traced surface `w` onto the momentum column with the two-point
  `.to`, while the biased schemes interpolate velocity faces with
  an `(order−1)`-point centered row — coincident only at order 3.
  Shipped impact (dev `4adcc933`..`81995781`): WENO5/Upwind5 u/v
  top-row tendency erred by ~15–17%, single top layer, interior
  bit-exact; buoyancy/tracers, order-3 biased and centered
  unaffected. Fixed in merge `81995781`
  (`_slice_relocation_exact` ClassVar): biased staggered momentum
  routes to the exact full-3D `_correction_full`, tracers keep the
  slice, centered bit-identical; constancy + autodiff regressions
  added. Consequence: the weno5 arm numbers above measured the
  pre-reroute all-slice path — post-reroute weno5 overhead and its
  lowering choice are unmeasured (the biased embed default is
  marked provisional in-code).
- **Flag:** the `surface_flux=False` opt-out's absolute cost at
  the ≥512²×64 rungs is +28–48% above its pre-H7 record (rf21
  matches exactly, so not the node); some 07-17→07-18 respelling
  hit the legacy opt-out path. Cosmetic for defaults (nobody runs
  off), but it inflates "vs off" denominators — the oc ratios
  above are the denominator-free evidence.

**Real multi-host validation (2026-07-18 evening, owner-requested)
— §4 gate MET.** Protocol: hydrostatic ladder physics
(split-explicit, surface flux on, purely baroclinic ICs) at
64×64×16, 20 steps; full gathered global true state (u, v, b, ps
via `export.gathered_values`) compared between a single-GPU
reference and a real `srun -n 4 --gpu-bind=none` +
`jax.distributed.initialize` run (one A100 per process; jobs
26351875/26352067, node l50024, commit `33707661`). Results:
`centered` (scatter lowering) **bitwise identical**; `weno5`
(embed tracer slice + full-3D biased momentum) max rel 9.2e-16
on u/v/b, 1.9e-14 on ps — tolerance 1e-11 beaten by three orders.
Findings from the campaign:

- **Default persistent compile cache deadlocked every real
  multi-process GPU run** (rank 0 at its first step collective,
  other ranks inside build compiles; XLA:GPU shard-autotuning
  cross-rank compile rendezvous never completes once per-rank
  cache state diverges). Fixed the same evening: `configure()`
  detects a real multi-process launch and leaves the cache off
  (merge `e4e54cfc`, reconciled onto dev in `94786a7c`; explicit
  `FRIDOM_JAX_CACHE_DIR` still honored with a documented
  `--xla_gpu_shard_autotuning=false` requirement; AGENTS.md
  recipe updated).
- **Split-explicit models do not carry the barotropic part of a
  velocity IC** — a z-independent `set_fields` velocity vanishes
  from the entire carry after one step (max|u| 0.98 → 2.6e-4 in
  5 steps; nothing in the FS subsystem holds it). Possibly the
  intended init semantics (barotropic mode starts at rest), but
  it means the se comparison-ladder rungs effectively ran
  near-zero-velocity flows while Oceananigans got the full IC —
  needs an owner ruling (roadmap notes it).
- weno5 hydro cannot run multi-device on shallow grids that
  shard z (nz=8: negotiated z-halo < the 6-point stencil; same
  under forced-4, pre-existing; fine at nz=16 with x/y sharded).

**Remaining (owner-gated GPU work; the roadmap entry tracks it):**
step-guard checkpoint whenever Silvano next batches one (his own
trigger, never agent-initiated); post-reroute weno5 ladder
re-measure (overhead vs off + embed-vs-scatter for the tracer
slice).
