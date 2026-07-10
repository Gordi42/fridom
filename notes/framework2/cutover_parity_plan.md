# Cutover-parity work plan (drop old framework/nonhydro/shallowwater)

**Status: in progress, 2026-07-10.** Gap analysis of what still lives
only in the old packages, and the porting waves closing it. Rule of
record (ROADMAP.md Cutover): models reach parity on framework2 →
rename framework2 → framework, retire the old package in one swap,
update imports/examples/docs.

## Owner decisions

- **Hydrostatic is NOT part of the port.** It will be implemented
  later (ROADMAP 3.1), not as a cutover blocker (owner, 2026-07-10).
- NNMD stays descoped (ROADMAP 2.8 sign-off; future rewrite).
- Still open (owner call, low urgency): NetCDF writer (new stack is
  zarr-only), animation/`VideoWriter`/figure saver, RFFT pressure
  solver variant, old mpi4py multi-host runs (jax.distributed is
  ROADMAP 3.2/3.3).

## Wave A — physics-module ports (parallel, running 2026-07-10)

| Package | Content | Status |
|---|---|---|
| Closures | `ClosureBase` (spec: `model/classes/module.md` §fr.closures.ClosureBase) + harmonic/biharmonic diffusion & friction (role-targeted: mixing→TRACER, friction→Velocity family) + nonhydro2 `SmagorinskyLilly` | agent running |
| Forcings | framework2 `Relaxation`, nonhydro2 `GaussianWaveMaker` + `PolarizedWaveMaker` | agent running |
| Advection | nonhydro2 `UpwindAdvection` + `WENOAdvection` as model modules over the existing `grid/operators/{weno,reconstruct,…}` layer | agent running |

## Wave B — after A merges

- Analytic initial conditions, both packages: jets, coherent eddy,
  wave package (nh), barotropic jet (nh), equatorial wave (sw).
  Single/Kelvin waves are covered better by `em.mode`/`eb.mode` —
  port only thin named conveniences where examples need them.
- Examples refresh: all 13 `examples/*.py` currently import the old
  stack; port to nonhydro2/shallowwater2 (doubles as the parity
  shakedown). Include the sw.eigenbasis gallery example
  (β slow-mode filtering; see projection_eigenmode_roadmap §5).
- Docs refresh: 13 `.rst` files (`fridom_api`, getting-started,
  10 tutorials) reference only the old packages.

## Wave C — gate + swap

- §8.8 cutover-parity list sign-off
  (`notes/framework2/model/06_validation.md:272`).
- Full suite + 95% coverage + ruff (the wave-8 gate), merge to dev.
- The swap: rename framework2 → framework; **move
  `fridom.framework.utils` into the new package** (framework2
  imports it today — jaxify/jaxjit/dtypes); delete old packages +
  their 114 test files (tests/{framework,nonhydro,shallowwater,
  hydrostatic}); retire old benchmarks (`bench_nonhydro`,
  `bench_shallowwater`, `bench_fields`, `bench_operators`); update
  CI/coverage config, imports, README.
- Keep the old `hydrostatic` sources parked (not deleted blindly)
  until ROADMAP 3.1 reimplements it — it is the one old package
  with no new-stack successor.

## Not cutover blockers

Waves 10–11 of the 2.9 substrate (linear-term blocks, symbolic
`BlockSymbol`): the probe-based eigen stack covers the
functionality; the substrate refinement proceeds independently.
