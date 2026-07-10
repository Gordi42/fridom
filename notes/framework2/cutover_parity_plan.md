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

## Wave A — physics-module ports (MERGED 2026-07-10)

| Package | Content | Status |
|---|---|---|
| Closures | `ClosureBase` + `Harmonic/BiharmonicDiffusion` (TRACER) + `Harmonic/BiharmonicFriction` (Velocity family) + nonhydro2 `SmagorinskyLilly` | merged 14e7af5 |
| Forcings | framework2 `Relaxation` (rate/target/mask profiles), nonhydro2 `GaussianWaveMaker` + `PolarizedWaveMaker` (eigenmode packet at bind) | merged (013ee2d..90d91c4) |
| Advection | nonhydro2 `UpwindAdvection` + `WENOAdvection` (orders 3/5) over the grid WENO kernels; `_FluxFormAdvection` shared base; old-stack tendency parity ≤1e-15 | merged 5e5a061 |

Wave-A gate: full framework2 suite **3697 passed, 14 skipped**;
ruff clean. Deliberate non-ports (recorded by the agents): old
`nonhydro.BiharmonicClosure` (grid-scaled, mask-BC variant — the
generic biharmonics cover periodic physics), spatially varying
diffusion coefficients (AUXILIARY-field design point, door open),
walled-grid closures/advection (taught bind errors), orders > 5,
implicit closure treatment, PolarizedWaveMaker on walled grids.
Framework note (advection agent): assembly's zero-field dry run
(step 6) runs real kernels under the provisional halo before
negotiation — module-side widening workaround in
`UpwindAdvection.bind`; the designed tracer-based dry run would
retire it.

## Wave A2 — background-flow advection (MERGED 2026-07-10, option 3)

Owner picked option 3 (difference-form split). Landed: nh2
Centered/Upwind/WENO `background=` — linear term `S_lin(U,q)`
(WENO's linear term = its optimal-weight smooth-limit row; upwind
select by static sign(U), exactly linear), nonlinear term
`S_full(U+Ro·u',q) − S_lin(U,q)` (telescopes to the full-velocity
scheme; old-stack parity ~6e-16 at Ro=1); sw2 Sadourny
`background=` — one V-S3 linear term (old semantics were already
split), walled grids fully supported (structural impermeability +
bind validation of wall-normal and discrete solenoidality),
background term conserves plain quadratic energy exactly.
Convention change (documented): background is O(1), NOT Ro-scaled
(old stack scaled it). Doppler verified against the eigen probe in
both packages (shift = discrete symbol, ≤1e-12). Sheared U: linear
model works; eigh-based engines refuse via the Hermiticity guard
(eig backend = future item). A2 gate: full framework2 suite 3775
passed; ruff clean.

## Original A2 design record (for reference)

Gap (owner, 2026-07-10): advection by a prescribed background flow
is unported. Ruling V-S3 (06_validation.md:89) settles the shape:
linearity is a TERM property — the advection module owns a separate
``linear=True`` background term; no separate linear module.
Proposed uniform construction (all schemes, incl. WENO):
``background=`` adds two terms — linear ``L(q) = S_lin(U, q)``
(centered or static-upwind; upwind selection by sign(U) is fixed at
bind, hence exactly linear reusing the existing operators) and
nonlinear ``N = S_full(U + Ro u', q) − S_lin(U, q)``, so the sum is
exactly the full-velocity scheme and ``fr.linearize`` keeps exactly
``L``. A "linear WENO" is a category error (smoothness weights
depend on q) — never needed. Old ``disable_nonlinear`` ≙
``fr.linearize``. Notes: constant U → Doppler eigenmodes work in
the whole eigen stack for free; sheared U is non-normal → the
Hermiticity guard correctly refuses eigh (an ``eig`` backend is a
separate future item). u'·∇Q production terms stay the province of
dedicated linear modules (stratification pattern), as in the old
stack.

## Wave B — after A merges

- Analytic initial conditions — MERGED 2026-07-10: sw2
  `single_wave`/`jet`/`coherent_eddy`/`equatorial_wave` (Hermite
  port with movable `equator=`), nh2 `single_wave`/`kelvin_wave`
  (labeled channel mode, exact — replaces the old approximate
  V=0 ansatz)/`wave_package`/`barotropic_jet`/`jet`/
  `coherent_eddy`. Wave factories return `(omega, state)` on the
  `em.mode` phase convention; `use_discrete` dropped (the
  eigenmodes ARE discrete); L2 normalization → envelope
  convention. 81 tests, 100% branch coverage on both edited
  modules.
- Examples refresh (DEFERRED — owner 2026-07-10, not now): all 13
  `examples/*.py` currently import the old stack; port to
  nonhydro2/shallowwater2 (doubles as the parity shakedown).
  Include the sw.eigenbasis gallery example (β slow-mode filtering;
  see projection_eigenmode_roadmap §5).
- Docs refresh (DEFERRED — owner 2026-07-10, not now): 13 `.rst`
  files (`fridom_api`, getting-started, 10 tutorials) reference
  only the old packages.

## Wave C — gate + swap

- Audit flags RESOLVED 2026-07-11: `p = φ/stage_dt` wired
  (physical pressure stored; Sketch-A regression, also closes
  parity item 1) and `rest` policy wired centrally in
  `transforms/base.py` (tracer-completion law tested end to end).
  Audit state now: 23/23 rows covered, 1 subtle sketch open
  (one-period dispersion, Appendix A Sketch B), 0 flags,
  16-row intentional-deltas table awaiting owner sign-off.
- §8.8 audit DONE 2026-07-10 → [`parity_audit.md`](parity_audit.md):
  20/23 rows covered by verified tests, 1 mechanical regression
  written (eps'd-AB2 jit tolerance), 2 subtle sketches remaining
  (project-state ≡ project-tendency; one-period dispersion — audit
  Appendix A, for a strong pass), 16-row intentional-deltas table
  AWAITING OWNER SIGN-OFF, and 2 owner flags: `p = φ/stage_dt`
  documented but unwired (nonhydro2 stores the raw potential —
  decide: normalize or retract the claim), and `rest="zero"`
  runtime zeroing unimplemented (policy slot only). Residual:
  `nonhydro2/diagnostics.py` epot returns inf at N²=0 (the guard
  lives only in EnergyMetric.from_model).
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
