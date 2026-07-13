---
status: done
date: 2026-07-13
---

# Krylov scan — O(1) trace for the mapped pressure solve

ROADMAP 3.6. Shipped in `c022d79e` ("Merge perf/krylov-scan"), which
lands `5f8754f2` — that commit's subject still says "WIP, unverified"
and is stale; the merge carries the tests and the numbers.

## The problem

The mapped pressure CG
([`spatial/operators/krylov.py`](../../../src/fridom/spatial/operators/krylov.py))
ran a fixed, fully unrolled iteration loop, so tracing and XLA
compilation were **O(iterations)** while warm execution was ~0.05% of
the cost — a trace/compile problem, not a runtime one. At 300
iterations: ~245 k HLO lines and ~48 s of compile. The model default
`pressure_iterations = 30` put ~7-14 s of compile into every mapped
model build, and a jitted forced-4 solve never finished compiling.

## What shipped

**The carry is raw arrays, not fields.** A `ScalarField`'s
`_halo_valid` is static aux data that participates in treedef equality
(it drives sync *placement*, so it must key the jit cache — the
`annotation=` exemption stays rejected). Carrying a field through
`lax.scan` would therefore impose treedef stability on a quantity
operators legitimately change. The carry is a flat tuple of
`jax.Array`s plus the 0-d `rz`; the body rebuilds the fields through
`with_data`, which re-declares the **canonical** halo state (zero valid
ghost layers, synced at first consumption).

The design question "is the canonical state actually invariant?"
resolved *yes*: it is the fixed point the unrolled loop already sat in.
Every carried iterate is the output of field arithmetic, which routes
through the storage write path and hence already claims zero ghost
validity. The rebuild is exchange-neutral — it adds no sync, so the
feared per-iteration exchange trade never materialized.

**`lax.scan`, not `lax.fori_loop`**, so reverse-mode keeps flowing
through the solve.

**The first iteration is peeled.** The operator and preconditioner are
opaque closures that may do trace-time bookkeeping on first application
(registry rows, a memoized halo exchange, the mapped solver's per-solve
metric memo). Created *inside* a scan body, such an entry would hold a
body-level tracer and leak out of the loop. Peeling forces those side
effects to the enclosing trace level. Cost: two traced bodies instead
of one — still O(1) in the iteration count.

## Outcome against the gates

| gate | result |
|---|---|
| O(1) trace/compile | **468 HLO lines at 12, 30, 60 and 300 iterations alike** (unrolled: 2102 / 5198 / 10358). `test_hlo_size_is_constant_in_the_iteration_count` |
| operator traced O(1) times | 2 traces (peel + one body) at any budget. `test_operator_is_traced_twice_regardless_of_iterations` |
| bitwise vs. unrolled | **Not met, by design.** XLA fuses/FMA-contracts a scan body differently from straight-line code: ~1 ulp (5.6e-17), far below the 6e-15 mapped-flat identity gate. `test_scanned_solve_matches_the_unrolled_recurrence` (12/30 iterations, atol 1e-14) |
| `jax.grad` through the solve | finite. `test_grad_through_the_solve_is_finite`, `test_grad_flows_through_a_long_scanned_solve` |
| exchanges per iteration | unchanged. `test_exchanges_per_iteration_are_unchanged_by_the_scan` |
| C4 moving geometry / metric memo | still rebuilt per solve; `tests/nonhydro2/test_mapped_pressure.py` green |

One gate consequence in the mapped tests: the memo now decides whether
the metric fields enter the body as hoisted constants or are recomputed
inside it — two different body computations. So
`test_solve_is_bitwise_identical_to_the_unmemoized_operator` became
`test_solve_matches_the_unmemoized_operator_to_rounding` (~1 ulp:
2.8e-17 absolute, 2.3e-16 relative).

Known, pre-existing and *not* caused by the scan: over-iterating an
exact preconditioner drives the residual to ~1e-17 rather than exact
zero, so `_guarded_ratio` divides tiny-by-tiny — finite forward, NaN in
reverse. The unrolled recurrence NaNs at the same iteration counts. It
is a property of the fixed-iteration design.

## The forced-4 gate — closed (measured 2026-07-13)

The last open gate. It is met: a **jitted forced-4 mapped solve now
compiles**, where before it did not finish in 10 minutes. Measured on
cpu with `--xla_force_host_platform_device_count=4`, the 16x16 mapped
solve of `test_mapped_projection_is_device_count_invariant` at 12
iterations:

| | 1 device | forced 4 |
|---|---|---|
| jit compile | 0.7 s | **9.1 s** (was: never finished) |
| eager per call | 426 ms | 16 399 ms |
| jitted per call | 0.28 ms | 194 ms |
| eager penalty | 1525x | **85x** |

The predicted 74-87x eager penalty is confirmed at **84.6x** — that is
what the multi-device gates still pay by solving eagerly. Jitted, the
1- and 4-device solves agree to 6.2e-16 (the test's gate is 1e-11).

The O(1) trace survives sharding: HLO is **52 200 lines at 12, 30, 60
and 120 iterations alike**, with compile flat at ~8-10 s; only warm
runtime scales with the iteration count (191 / 456 / 907 / 1798 ms), as
it should — that is the arithmetic, not the trace.

Consequence, taken: `test_mapped_projection_is_device_count_invariant`
(`tests/validation/test_terrain_following_pressure.py`) now jits the
solve — 21.5 s -> 14.8 s on forced-4. The win is much smaller than the
85x per-call figure because the test makes a single call, so the 9.1 s
compile eats most of it.

**What this exposed.** Even jitted, forced-4 costs 9.1 s to compile and
194 ms per call — **694x** the single-device 0.28 ms, on a 4-way sharded
problem. That is not the unrolled loop (this plan removed it); it is the
SPMD partitioning and the CG's cross-shard reductions. Recorded as
roadmap **3.9**, unscheduled.

Follow-ons (separate work, not this plan):

- restore the library default `pressure_iterations` and the test
  iteration budgets that were trimmed under the unrolled cost (the
  default is still 30 in
  [`nonhydro2/modules/core.py`](../../../src/fridom/nonhydro2/modules/core.py));
- a tolerance-break variant (a scan with an early-exit mask is
  differentiable; a `while_loop` is not) — only worth it if iteration
  budgets get large;
- the `custom_vjp`-via-implicit-function-theorem route for adjoints
  through the solve (CS-D2 records it as deferred), more attractive now
  that the forward solve is a single scanned body.
