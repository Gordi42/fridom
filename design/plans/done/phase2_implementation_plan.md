---
status: done
date: 2026-07-08
---

# Framework2 Phase-2 implementation plan

How ROADMAP Phase 2 (tasks 2.2–2.8) is executed with parallel
subagents on one machine — the Phase-2 analogue of
[`phase1_implementation_plan.md`](phase1_implementation_plan.md). The class
specs under [`classes/`](../../specs/model/classes/README.md) are the normative
contract; agents implement the iteration-1 surface and **report
deviations instead of redesigning**. The API sketches
([`../../specs/model/05_api_sketches.md`](../../specs/model/05_api_sketches.md)) are the acceptance
surface; the design notes (01–09) are consulted for semantics, never
reopened.

**Scope extension (2026-07-08, awaiting sign-off).** The energy-metric
projection design ([`../active/projection_eigenmode_plan.md`](../active/projection_eigenmode_plan.md),
build order in [`../active/projection_eigenmode_roadmap.md`](../active/projection_eigenmode_roadmap.md))
and the spectral/eigen substrate ([`../active/operator_symbols_plan.md`](../active/operator_symbols_plan.md),
[`../../decisions/blocksymbol_l_assembly.md`](../../decisions/blocksymbol_l_assembly.md),
[`../active/linear_term_blocks_plan.md`](../active/linear_term_blocks_plan.md)) refine
two things here: **wave 7's projections become energy-metric** (drop the
biorthogonal `vec_p`; `p = M q` under a first-class `EnergyMetric`), and
a new **Phase 2.9** (waves 9–11, below) adds the scalar-`Symbol` /
`SpectralSolve` substrate, the operator-algebra assembly of the
linearized `L` as a `BlockSymbol`, and the numeric / non-Fourier
eigenmode generalization. Waves 2.2–2.7 are unaffected.

## Ground rules

- Integration branch: `framework2-model-impl` (off `dev`; merged to
  `dev` at wave gates). Implementer agents work in isolated worktrees
  based on this branch and never commit to it — the orchestrator
  reviews each diff against the class spec, runs its tests, merges in
  dependency order, and commits.
- Conflict-free by construction: the Wave-0 scaffold pre-creates the
  `model/`, `transforms/`, `io/`, `ops/` package trees, `__init__`
  wiring (lazypimp), `model/errors.py`, and the
  `tests/framework2/model/` skeleton. Each agent owns an exclusive
  file list named in its prompt; shared files (`__init__.py`,
  `conftest.py`, `errors.py`) are orchestrator-only.
- Compute: at most **3 concurrent agents**. Agents run only their own
  test files, serially, on tiny grids (8–16 points); multi-device via
  `XLA_FLAGS=--xla_force_host_platform_device_count=4`. Full suite +
  ruff + coverage run at wave boundaries only, with no agents active.
  Benchmarks always run exclusively.
- **Naming protocol**: the specs flag confirm-at-first-use spellings
  (`RunTargetError`, `TIME_STEP = "stepper.dt"`, `@fr.self_update`,
  Writer `mode="w-"`, `lower_trigger`, the stepper deviations
  D-1..D-6, the `default=` owner-method disambiguation, ...). The
  agent implements the spec-proposed spelling and lists it in its
  report; the orchestrator batches them for Silvano's confirmation at
  the wave gate. A deviation forced by landed grid code is reported
  the same way — never silently designed around.

## Coordination with the 1.8 sync rework

A separate session owns ROADMAP 1.8 (consumption-side halo-validity
sync). Rules for this plan:

- No agent here touches sync-placement internals:
  `operators/base.py` (`_finalize`, `OperatorSum` sync calls),
  `fields/storage.py` (store-side sync), or the decomposition
  sync/shard_map machinery. Wave 1's grid follow-ups touch
  *neighboring* code (`decomposition.py` negotiate, `halo.py` trace
  entry) — keep those diffs surgical.
- Rebase the integration branch onto `dev` at every wave gate to
  absorb 1.8 when it merges. 1.8 is results-neutral by construction,
  so absorbing it must not change any test expectation.
- No test in this plan asserts exchange counts or sync placement.
  Multi-device performance work (the 2.7 benchmark table) assumes 1.8
  has landed; if it has not, the benchmarks run anyway and the table
  is annotated.

## Waves

| Wave | Content (roadmap task) | Parallel split |
|------|------------------------|----------------|
| 0 | Scaffold: `model/` + `transforms/` + `io/` + `ops/` trees, stubs, lazypimp wiring, `model/errors.py` (registry skeleton), `tests/framework2/model/` conftest (reuse the compile-counter fixture) | serial |
| 1 | Grid follow-ups ([`../active/phase2_grid_followups.md`](../active/phase2_grid_followups.md), the 2.2-blockers; item 8 excluded — 1.8 session) | A items 2+3+5: `merge_overrides` facade, `("declared_space", mesh)` resolver rows, freeze fingerprint/verify/`GridFrozenError` (registry + `grid.py`) · B items 1+4: negotiate `merge_max`, name-keyed `trace_halo` (decomposition + halo) · C items 6+7: `VectorField.add` + metadata re-attach, annotation-exempt metadata equality (fields + jaxify in `framework/utils/jax_utils.py` — scoped so old-framework pytrees are unaffected) |
| 2 | Declarations (2.2) | A field vocabulary: `Lifecycle`, `Role`/`Velocity`/`fr.roles`, `Dof`/`SpacePattern`/`Collocated`/`Staggered`/`Profile`/`SpaceRule`, `FieldDeclaration` (+ templates), `FieldReference` · B parameters + time: `ParameterDeclaration`/`ParameterReference`/`REQUIRED`, `fr.Param`/`USE_PROVIDED`, `ParamName` + `fr.params` registry, `TimeDependent`/`Ramp` (+ `reversed()`)/`resolve_at` · C terms + stages vocabulary: `Treatment`, `TendencyTerm`/`@fr.term`, `ImplicitOperator` protocol + `VerticalDiffusion`, `StageKind`/`Stage`/`fr.self_update`, `StepContext` |
| 3 | Assembly tables + Module (2.2/2.3) | A `FieldRecord`/`FieldTable` (+ `subset`)/`VelocitySelector`, `ParameterBinding(Table)`/`Params`, `RematerializationTable` · B `Module` base: declarations/references collection, `dispatch`, `bind` (no field materialization), `extra_halo`, `state_type` · C `TendencyComposer` (attribution, write-gate + dry-run validation, deterministic accumulation via `VectorField.add`) + `Schedule`/`BoundSchedule`/`TendencySums` |
| 4 | Model core + run loop (2.3/2.4) | A the nine-step assembly (`../../specs/model/classes/model.md` §6.2, `merge_max` halo union at step 7), `AssemblyRecord` (+ `step_fn` memoization), `AssemblyReport`, `Fingerprint` · B `ModelState` carry, `step_chunk` (donation, AOT lower/compile, chunk lengths {C,1}), `Clock`, `TimeStepper` base + `AdamBashforth`/`ABState` (eps order-2-only), `advance` + results types + `PanicError` + the S5 NaN seam · C io for 2.4: `Trigger` family + `fr.every`/`fr.at` + `lower_trigger` (sign-agnostic), snapshot store (`SnapshotManifest`, atomic write/rotate/find_latest), `fr.slurm` + `fr.io.resubmit`, `WalltimeGuard`/`ProgressReporter`/`ChunkStats` |
| — | **Gate: first end-to-end jitted run.** Module-only toy model (tracer advection–diffusion, no fake core; CS-13), single jit over chunked scan, 1 and 4 devices. Oracles: treedef stability, compile counter (one compile across a parameter re-assembly sweep), repeated-`advance` ≡ uninterrupted run, snapshot round-trip bitwise, NaN-abort at chunk boundary. Merge to `dev`. | |
| 5 | Ops + steppers + writers (2.4/2.5/2.6) | A `fr.ops.Session` + `run()` reimplemented over it, boundary sequence (sync → panic → flush → progress → walltime), interrupt handling, `debug_nan` replay · B RK family (`ButcherTableau`/`tableaus`, `ExplicitRungeKutta`, `LowStorageRK3` — pin coefficients vs Oceananigans) + treatment partition/`implicit`/`advance_stages` in the schedule + `IMEXMultistep`/`IMEXState` + `CNAB2`/`SBDF2` factories, with the `VerticalDiffusion` reference consumer (1D decay + stiff-κ) · C `fr.io.Writer` (zarr sink, xarray/xgcm-openable, `truncate_after` resume) + `TimeSeries` (CSV) + `IOCollisionError`/`SnapshotMismatchError` + restart-under-scan integration tests |
| 6 | Model ports (2.7) | A nonhydro: core module + `FPlaneCoriolis`/`BetaPlaneCoriolis`, `ConstantStratification` (registers `b`), advection, pressure projection as CONSTRAINT stage + solver operator, eigenmodes (`from_model`, `em.q`/`em.p`/`em.projector`), `State` vocabulary, preset factory · B shallowwater: same shape (`csqr` field, Sadourny, Rossby scaling), plus `WindowAccumulator` and the first closure port on `ClosureBase` · C the §8.8 cutover-parity suite ([`../../specs/model/06_validation.md`](../../specs/model/06_validation.md)) + examples/docs refresh + benchmark table vs old framework at identical sizes |
| 7 | State transforms (2.8) | A `StateSignature`/`TransformInfo`/errors, `StateTransform` base + algebra nodes, `Identity`/`Shift`/`FixedPoint`/`relative_l2`/`assert_idempotent`, `model.variant` + term predicates (`fr.terms`) + `fr.linearize` + `model.tendency` · B Tier-2 presets: `Propagator`, `TimeAverage`, `OptimalBalance` (+ SELF_UPDATE-first regression, info law, trace guard) · C **energy-metric** projections: `fr.EnergyMetric` + State inner-product/norm (+ SW `ekin`/`epot`, nh `epot` diagnostics), reformulate the wave-6 eigenmodes to derive `p = M q` (drop hand-written `vec_p`), then `VorticalProjection`/`WaveProjection`(`=P(+1)+P(−1)`)/`DivergenceProjection`(`=complement`) + `nh.transforms`/`sw.transforms` aliases + the D5 behavior-delta tests (`stop_best`, continuous Ramp, TimeAverage-drops-Smagorinsky) |
| 8 | Final gate: full suite + 95% coverage + ruff, benchmark table, §8.8 sign-off review, merge to `dev` | serial |

Wave 3's tracks interlock through the pre-scaffolded stubs (A's
`FieldTable` is consumed by B's `bind` and C's composer) — the class
specs fix the signatures, so the tracks build against the stubs and
the orchestrator merges A → B → C. Waves 6 A/B are independent by
package; C follows their merges.

## Phase 2.9 — energy-metric substrate & operator-algebra `L` (new; awaiting sign-off)

Extends Phase 2 with the spectral/eigen substrate the wave-6 ports
hand-rolled and the operator-algebra assembly of the linearized system
operator. Full build order + dependency graph:
[`../active/projection_eigenmode_roadmap.md`](../active/projection_eigenmode_roadmap.md).
**Wave 9 is grid-layer and independent of wave 7 — may run in
parallel.** Waves 10–11 depend on wave 7C's `EnergyMetric` and on wave
9. The `Eigenmodes.from_operator(L, M, grid)` seam is stable across
10–11; only the `L`-producer changes (probe → symbolic `BlockSymbol`).

| Wave | Content (roadmap phase) | Parallel split |
|------|-------------------------|----------------|
| 9 | Spectral-solve substrate (C/D; S1/S2) | A scalar `Symbol` (`@final`) in `operators/symbol.py` + per-operator `eigenvalues` (§4 table) — lights the dormant `base.py` product/sum/scale composition; delivers the pure-diagonal pressure solver · B `SpectralSolve` (diagonal/banded partition) + lift the tridiagonal primitive from `model/implicit.py` into `grid/operators/banded`; retire the hand-rolled `nonhydro2/pressure.py` inverse. Gates `../active/operator_symbols_plan.md` §7 A–E |
| 10 | Linear-term blocks + numeric eigenmodes (T1–T4, H0) | A `LinearBlock`/`Coeff`/op-specs + `TendencyTerm.blocks` + `@fr.term(blocks=)` + assembly validation (T1); re-author the four linear terms (shared Coriolis, stratification ×2, sw gravity) with the block-vs-hand equivalence test, then derive `fn` from `blocks` (T2–T4) · B the `from_operator` seam + H0 probe: `L(k)` by `jax.jvp` of `fr.linearize(model).tendency(·, constraints=True)`, `eigh(iML, M)`, validated against the analytic Tier-0 modes |
| 11 | Symbolic `BlockSymbol` + general eigenmodes (H1, I) | A `BlockSymbol` (separate type from scalar `Symbol`) + `BlockMatrix.eigenvalues()`; assemble `L(k)` from block Symbols (SW pure; nonhydro composes the Leray `Symbol.inverse` from wave 9) — swap behind the unchanged `from_operator` seam · B non-periodic / vertical general eigenmodes (sine/cosine structure functions; banded generalized `eigh`); walls-in-vertical additionally need Shen/Galerkin Chebyshev + Chebyshev `grid.measure` quadrature (own prerequisites) |
| 12 | 2.9 gate: full suite + 95% coverage + ruff, symbol/eigen validation review, merge to `dev` | serial |

Named oracles: **W9** — the [`../../specs/model/06_validation.md`](../../specs/model/06_validation.md)
eigenvalue identities (`i k sinc` = average-of-∂ₓ, `bwd @ fwd` real
Laplacian); `SpectralSolve` residual ≡ the hand-rolled pressure solver
on the shared path. **W10** — block-derived `fn` ≡ hand `fn` on random
state (the desync guard); H0 `eigh(iML, M)` modes ≡ analytic `em.q` /
`em.omega` to tolerance; `from_model` beta-plane / variable-coefficient
decline preserved. **W11** — symbolic `L(k)` ≡ the H0 probe `L(k)`;
assembled SW/nonhydro dispersion ≡ analytic; the 1×1 Laplacian block ≡
the scalar `kx² + ky²` symbol.

## Module/jaxify discipline (verbatim into agent prompts)

The Phase-2 analogue of the Phase-1 kernel rules — the static
discipline that keeps the single-jit run honest:

- every callable slot (`TendencyTerm.fn`, `default=`,
  `VerticalDiffusion.kappa`) is stored **unbound**; assembly lints
  reject bound methods (`__self__` present);
- dynamic leaves are coerced through `jnp.asarray`; provided
  parameters must name dynamic leaves (assembly-checked);
- statics must be cheap and structurally comparable: interned
  identity-hashed objects (grid, spaces); host-side observers listed
  in `_eq_ignored_attrs`; flatten order = declared
  `dynamic_jax_attrs` order, never reordered in a released module;
- `tree_unflatten` bypasses `__init__` — no constructor-established
  invariant may be assumed in-trace;
- **no `io_callback` in the traced step, ever** — host effects happen
  at chunk boundaries on the synced carry;
- `bind` precomputes operators/spaces/name-tuples only; a field
  materialized at bind is a bug (stranded by step-7 renegotiation);
- `StepContext` is frozen, all-scalar (per-treatment sums the one
  exception); `ctx.stage_dt` is never `dt_gamma` — γΔt is a separate
  stepper-supplied argument;
- bitwise claims follow the umbrella rule
  ([`../../specs/model/02_rules.md`](../../specs/model/02_rules.md)): identically-compiled paths only;
  cross-compilation comparisons are tolerance-based (≤ a few ulp per
  step, accumulation-aware).

## Do-not-build list (designed-for slots; keep the seam, skip the body)

`SpectralDiagonal`, `AdaptiveRungeKutta` (embedded-pair tableaus stay
as data), IMEX-RK tableau steppers, SBDF3, `PendingAdvance`
(signature reserved via `advance(steps, *, sync=True)` only),
`cadence=` on `self_update` (reserved, documented hazard), capture
streams + post-assembly writer attach, TensorStore backend +
async/split/`sel=` writer features, multi-model Session maturation
(3.2), NNMD (descoped — no model propagator), `Velocity` `group=`
qualifier, batched Tier-2 / `with_parameters` / `resync()` /
`OnComponents`, `em.omega_field`.

The scalar `Symbol`, `SpectralSolve` + `grid/operators/banded`,
`BlockSymbol`, `EnergyMetric`, and the operator-algebra `L` assembly —
formerly deferred "designed-for" slots — are **now scheduled in Phase
2.9** (waves 9–11), not do-not-build. Still deferred: `SpectralDiagonal`
(the implicit spectral operator), `em.omega_field`, and NNMD.

## Standing test gates

At every wave boundary: full suite green, ruff-clean, coverage on the
new cluster ≥ 95%, treedef-stability and compile-counter tests green.
Key named oracles by wave:

- **W3**: write-gate matrix per stage kind; same-kind overlap lint;
  coverage lint (ADVANCE `advances` claims count); `Ramp`
  static/dynamic recompile counts (endpoint sweep = 0 recompiles).
- **W4 gate**: see the table row above.
- **W5**: `run()` ≡ user-written Session loop, chunk for chunk
  (bitwise, shared jit-cache entry + compile counter); AB coefficient
  rows + eps order-2-only; IMEX level tables (CNAB2/SBDF2,
  solve-normalized); empty-implicit degeneration to textbook AB2
  (documented as not bitwise); warm-up mid-chunk restart bitwise;
  predictive walltime stop; first-Ctrl-C zero-loss return.
- **W6**: the §8.8 cutover-parity list verbatim — AB parity bitwise
  eager-vs-eager and tolerance-based jitted; discrete-eigenmode
  dispersion regression; S6 accumulation restart-exact;
  `preset ≡ explicit assembly` treedef identity for every shipped
  preset; device-count-changed resume tolerance cases (4→1, 1→4,
  permuted).
- **W7**: SELF_UPDATE-first bitwise twin-call regression (Ramp-valued
  AUX in the twin); the info law per transform; `assert_idempotent`
  on the projections; the three signed behavior deltas pinned.
