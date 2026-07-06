# Framework2 Phase-1 implementation plan

How ROADMAP Phase 1 (tasks 1.1–1.7) is executed with parallel
subagents on one machine (8 cores, ~15 GB free). The class docs under
[`classes/`](classes/README.md) are the normative contract; agents
implement the `# it-1` surface and **report deviations instead of
redesigning**.

## Ground rules

- Integration branch: `framework2-impl` (off `dev`; merged to `dev` at
  wave gates). Implementer agents work in isolated worktrees based on
  this branch and never commit to it — the orchestrator reviews each
  diff against the class doc, runs its tests, merges in dependency
  order, and commits.
- Conflict-free by construction: the Wave-0 scaffold pre-creates the
  whole package tree, `__init__` wiring, and test skeleton. Each agent
  owns an exclusive file list named in its prompt; shared files
  (`__init__.py`, `conftest.py`) are orchestrator-only.
- Compute: at most **3 concurrent agents**. Agents run only their own
  test files, serially (`uv run pytest tests/framework2/<cluster> -q`),
  on tiny grids (8–16 points); multi-device via
  `XLA_FLAGS=--xla_force_host_platform_device_count=4`. Full suite +
  ruff + coverage run at wave boundaries only, with no agents active.
  Benchmarks always run exclusively.

## Waves

| Wave | Content (roadmap task) | Parallel split |
|------|------------------------|----------------|
| 0 | Scaffold: package tree + stubs + lazypimp wiring, `errors.py`, static markers, interning helper, **jaxify flatten-order fix** (stated prerequisite, doc 02), `tests/framework2/` conftest incl. compile-counter fixture, benchmark-suite skeleton | serial |
| 1 | Static structure (1.1) | A meshes + traits seam · B spaces + interning + `TensorProductSpace`/strict algebra · C `Layout`/`HaloSpec` + trivial single-device `TensorDecomposition` |
| 2 | Field core (1.2) | A `ScalarField` + `create_field` · B `Operator` base + algebra + registry · C `FiniteDifference`/`LinearInterp`, correctness-grade with reference oracles |
| 3 | Fan-out (1.3 ∥ 1.4 ∥ 1.5) | A average family + FV + `integrate` + composed factories · B transforms + reshard planner + `Symbol` · C `negotiate` + `HaloTracer` + shard_map sync + `Reshard`/`Sync` + multi-device |
| 4 | Advection + perf (part of 1.3/1.7 hot paths, 1.6) | A `WenoReconstruction` + upwind pair (parity vs `weno_interpolation.py`) · B optimization pass over hot kernels · C immersed subset + `f.xr` export |
| 5 | Validation gate (1.7) | Hand-rolled advection/diffusion PDE single + multi device, `05_validation.md` checks, coverage/ruff gates, old-vs-new benchmark table, merge to `dev` |

Waves 3's tracks are sequential in the roadmap but genuinely
independent after 1.2 — this is where parallelism pays most.

## Kernel optimization rules (FD, interpolation, FV, WENO, transforms)

Correct, then fast; the naive reference implementation stays in the
tests as oracle (polynomial exactness up to order, measured
convergence rates, FV conservation identities).

Kernel rules (verbatim into perf-agent prompts):

- slice-based fused stencils — never `roll`/gather;
- coefficients as static Python floats baked into the jaxpr;
- one fused expression per kernel (XLA should fuse to a single loop);
- no data-dependent Python branching; static shapes, no host
  callbacks — shard_map/GPU-shaped even though tuning happens on CPU;
- WENO: fused smoothness indicators, precomputed linear weights,
  minimized divisions, selection via the `("select", …)` dispatch
  design.

Merge gate for any optimization: `fridom.benchmarking` numbers
(runtime, compile, memory) vs the pre-optimization commit **and** vs
the old `framework` kernel at identical sizes; a fusion sanity check
on the lowered HLO; the compile-counter test proving no retraces.
CPU-relative comparisons decide kernel shapes; absolute GPU tuning is
deferred.
