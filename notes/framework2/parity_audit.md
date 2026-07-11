# framework2 §8.8 cutover-parity audit

Maps every claim in
[`model/06_validation.md`](model/06_validation.md) §8.8 (the
consolidated cutover-parity list) — plus the newer post-list ports —
to the test that pins it. All claims sit under the bitwise-equality
umbrella ([`model/02_rules.md`](model/02_rules.md), "Bitwise-equality
umbrella"): bitwise means identically-compiled paths (eager-vs-eager,
or one executable against itself); anything across compilations (old
vs framework2, eager-vs-jit, 1-vs-N devices) is tolerance-based at
≤ a few ulp/step, accumulation-aware.

Verification was per-row: each COVERED citation was confirmed by
reading the test body and checking the assertion pins the *claim*
(not merely the code path).

## Summary counts

- **Covered**: 23 rows pinned by an existing test.
- **Mechanical hole written**: 1 — `test_order2_eps_jitted_matches_eager_within_tolerance`
  (the jitted-tolerance half of the eps'd order-2 row).
- **Subtle holes sketched**: 1 — the one-period discrete-eigenmode
  dispersion regression (Appendix A, Sketch B). Sketch A
  (project-state ≡ project-tendency) is implemented:
  `nonhydro2/test_pressure.py::test_project_state_equals_project_tendency`.
- **Intentional deltas** (new behavior deliberately ≠ old stack): 15
  rows, signed-off table below; every row has its *new* behavior
  pinned by a test.
- **Flags for owner**: none — both original flags (`p = φ/stage_dt`,
  `rest="zero"` application) are wired and covered (rows 2 and 7).

## Main audit table

| # | Claim | Class | Test / pointer | Notes |
|---|-------|-------|----------------|-------|
| 1 | Project-the-state (exact-equivalent for explicit schemes) | COVERED | mechanism: `nonhydro2/test_pressure.py::test_pressure_solve_drives_divergence_to_zero`, `::test_walled_projection_drives_divergence_to_zero`; idempotence + div-free reproduction `model/test_eigen.py::test_constrain_matvec_is_the_idempotent_leray_projector`, `nonhydro2/test_nonhydro2.py::test_eigenmode_projector_is_idempotent`; exact equivalence: `nonhydro2/test_pressure.py::test_project_state_equals_project_tendency` | The Sketch-A regression: AB1 one-step u₁ matches the hand-built project-the-tendency u₁′ per component (rel L2 < 1e-12; measured ≤ 3e-15), forward and backward (dt < 0). |
| 2 | `p = φ/stage_dt` | COVERED | `nonhydro2/test_pressure.py::test_project_state_equals_project_tendency` — stored `p ≈ ψ` (the projected-tendency potential, rel L2 < 1e-12); the backward leg pins the sign convention (φ flips with stage_dt, ψ does not) | Wired in `nonhydro2/modules/core.py::_project` (`"p": p / ctx.stage_dt`); the velocity update still subtracts the gradient of the RAW potential — dynamics unchanged, only the stored diagnostic is normalized. With a multistep stepper the diagnosed p is the pressure of the stepper's weighted tendency combination (O(dt²) time-filtered, inherent to projection methods). |
| 3 | Continuous Ramp vs piecewise-constant θ=n/N | COVERED | `model/test_time_dependent.py::test_linear_ramp_values` (exact interior value at t=2.5), `::test_backward_window_signed_t0` | New continuous behavior pinned by exact-value equality; the "vs old θ=n/N" is the delta (sign-off table). |
| 4 | `stop_best` (old returned the diverged iterate) | COVERED | `transforms/test_fixed_point.py::test_divergence_stop_best` — `info.returned_iteration == 1`, out == argmin-error iterate, not the diverged last | Intentional delta. |
| 5 | TimeAverage twin drops Smagorinsky (old kept it) | COVERED (two composed tests) | `transforms/test_time_average.py::test_default_filter_drops_the_nonlinear_term` (default `fr.terms.linear` drops the nonlinear term) + `nonhydro2/test_smagorinsky_lilly.py::test_smagorinsky_terms_are_nonlinear_and_linearize_drops_them` and `::test_stress/mixing terms .linear is False` | Caveat: no single end-to-end Smagorinsky-inside-TimeAverage test; covered by proxy (generic nonlinear term) + Smagorinsky-is-nonlinear. Intentional delta. |
| 6 | Sadourny reads the csqr *field* (old used the scalar in h_full — old bug) | COVERED | `shallowwater2/test_sadourny.py::test_csqr_is_a_state_field_not_a_scalar`; `::test_varying_depth_energy_rate_is_machine_zero` uses a y-varying csqr and telescopes to machine zero only if h_full reads the field | Intentional delta (bug fix). |
| 7 | `rest="zero"` | COVERED (default/metadata + application) | metadata: `transforms/test_signature.py::test_of_prognostic_from_state` (default == "zero"), `::test_rest_excluded_from_equality`, `::test_validate_input_allows_extra_components`; application (central wiring in `transforms/base.py::_apply_rest`): `transforms/test_base.py::test_rest_zero_attaches_zero_field_on_own_space`, `::test_rest_pass_passes_the_input_component_through`, `::test_state_minus_projection_carries_the_full_tracer`, `::test_complement_carries_the_full_tracer` (+ the algebra tests in that file); integration: `shallowwater2/test_transforms.py::test_projection_rest_zero_completes_a_passive_tracer`, `nonhydro2/test_transforms.py::test_projection_rest_zero_completes_a_passive_tracer` | The policy is applied once in `StateTransform.call_with_info` (output completion after `_evaluate`): extras the payload did not emit are attached — zero field on their own space (`"zero"`) or passed through (`"pass"`); algebra nodes inherit completion from their children. |
| 8 | Empty-implicit CNAB2 = textbook AB2 (no eps) | COVERED | `model/time_steppers/test_imex.py::test_cnab2_empty_implicit_is_textbook_ab2` — rtol 1e-11 vs an eps-free `1.5 f − 0.5 f_prev` reference | |
| 9 | `epot` N²=0 → hinted error (old silent formula switch) | COVERED | `model/test_energy.py::test_from_model_rejects_zero_stratification` (`match="1/N"`); sw twin `::test_from_model_rejects_zero_phase_speed` | Enforcement lives in `EnergyMetric.from_model` (the 1/N² weight). Residual: the `epot` diagnostic function itself (`nonhydro2/diagnostics.py:65`) does not guard n2=0 — it returns inf. Intentional delta. |
| 10 | Nyquist zeroing relocated into em.q/p | COVERED | `shallowwater2/test_eigenmodes.py::test_eigenmode_biorthonormality_and_structural_zeros` (geostrophic column of `em.q(0)` exactly zero on Nyquist planes); `nonhydro2/test_walled_eigenmodes.py::test_vortical_alive_count_is_n_plus_1` | Caveat: pins that the zeros *are* carried by em.q, not that a duplicate zeroing was removed from the former call-site. |
| 11 | Seeded-random ICs not bitwise (per-DOF fold_in) | COVERED | `grid/test_random_fields.py::test_normal_keyed_by_global_true_dof_index` (each DOF = `fold_in(key, global_flat_index)`), `::test_normal_deterministic_per_seed` | Intentional delta. |
| 12 | float64 clock (vs old float32 forcing phases) | COVERED | `model/test_clock.py::test_leaf_widths_follow_the_global_width` — `clock.start/elapsed.dtype == float64`, `it.dtype == int64` | Intentional delta. |
| 13 | netcdf→zarr + trigger snap-up + `every()` includes step 0 | COVERED (all three) | zarr: `io/test_writer.py::test_roundtrip_opens_in_xarray`, `::test_writer_module_does_not_import_zarr`; snap-up: `io/test_triggers.py::test_lower_seconds_snaps_up_with_ceil`, `::test_lower_at_ceil_snap_and_realized_time`; step 0: `io/test_triggers.py::test_lower_steps_includes_step_zero` | netcdf→zarr is the delta. |
| 14 | NaN abort at chunk boundary | COVERED | `model/test_step_chunk.py::test_panic_aborts_at_the_chunk_boundary_not_midchunk`; `model/test_end_to_end.py::test_nan_abort_at_the_boundary_with_exact_iteration`; facade `model/test_run.py::test_raise_on_nan_false_returns_nan_abort`, `::test_raise_on_nan_true_reraises` | |
| 15 | Backward runs via signed dt (no `run_backward`) | COVERED | `model/test_run.py::test_backward_leg_runs_when_dt_is_flipped`; `model/test_clock.py::test_tick_backward_still_increments_it`; `model/time_steppers/test_adam_bashforth.py::test_eager_integration_bitwise[backward]` | No `run_backward` method exists (grep-confirmed); sign flip is the API. Intentional delta. |
| 16 | Order ≥ 3 AB warm-up uses textbook AB2 (no eps; one-step startup delta) | COVERED (new behavior) | `model/time_steppers/test_adam_bashforth.py::test_order3_table_warms_up_through_textbook_ab2` (row 1 == `[3/2, −1/2]`, no eps), `::test_order4_table`, `::test_eps_rejected_off_order_2` (`match="order-2-only"`); integration crosses warm-up in `::test_jit_scan_compiles_once_and_crosses_warmup` | The startup delta vs old is signed off (§8.6-6); the new behavior is pinned. Intentional delta. |
| 17 | Discrete-eigenmode dispersion regression (one-period wave propagation at discrete ω) | HOLE-SUBTLE | closest: `nonhydro2/test_nonhydro2.py::test_tendency_eigenrelation_lq_equals_i_omega_q` (instantaneous L q = i ω q, not a period integration) | No test integrates one full period T = 2π/ω and asserts return-to-self → Appendix A, Sketch B. |
| 18 | AB2 eps'd row: bitwise on eager / tolerance jitted | COVERED (eager) + **MECHANICAL WRITTEN** (jitted) | eager: `model/time_steppers/test_adam_bashforth.py::test_eager_integration_bitwise[order=2]` (`np.array_equal`); jitted: **new** `::test_order2_eps_jitted_matches_eager_within_tolerance` (allclose rtol 1e-11) | The jit test uses order=3 only, so the eps'd order-2 jit-tolerance half was unpinned — now written. |
| 19 | Background advection NOT Ro-scaled (old scaled it) | COVERED | `nonhydro2/test_advection.py::test_background_split_telescopes_to_the_full_scheme` (total = Ro·u′ + U, background unscaled); `shallowwater2/test_sadourny.py::test_background_pressure_split_is_exact`; old-contrast comments in `::test_old_stack_background_parity*` | Intentional delta. |
| 20 | Kelvin-wave IC = exact labeled channel mode (vs old approximate V=0) | COVERED (exactness) | `nonhydro2/test_initial_conditions.py::test_kelvin_wave_phase_rotates_in_the_channel` (<1e-4 after 20 steps); `nonhydro2/test_channel_eigenmodes.py::test_mode_satisfies_the_strong_eigen_relation[kelvin-plus]` (residual <1e-12) | The "vs old V=0 ansatz" contrast is not asserted (only the new exact mode). Intentional delta. |
| 21 | Single-wave IC normalization (L2=1 → unit envelope) | COVERED (new) | `nonhydro2/test_channel_eigenmodes.py::test_mode_is_leray_compatible_and_normalized` (peak envelope == 1); `::test_single_wave_is_the_mode_accessor` | The "vs old L2=1" transition is not asserted. Intentional delta. |
| 22 | Relaxation rate = 1/tau | COVERED | `modules/test_relaxation.py::test_field_decays_exponentially_toward_the_target` (exp(−rate·t)), `::test_tendency_is_rate_times_target_minus_field` | Module takes `rate` directly (no `tau` kwarg); the identity r = 1/τ lives in the docstring, the physical meaning is pinned by the decay test. Intentional delta (interface). |
| 23 | Wave-maker parameter-name changes | COVERED (new names pinned) | `nonhydro2/test_gaussian_wave_maker.py` (constructs `GaussianWaveMaker({"x":…},{"x":…},freq,amp)`, provided params `wavemaker.u.amplitude/frequency`); `nonhydro2/test_polarized_wave_maker.py::make_maker` (`k, position, width, amplitude, s`) | Old names are not asserted; new signature pinned by construction. Intentional delta. |

## Intentional-delta sign-off table (owner)

Each row: old behavior → new behavior → why the change is right. A
test pinning the *new* behavior (cited above) counts as coverage; the
sign-off is the owner's confirmation the delta is intended.

| Delta | Old stack | framework2 | Rationale |
|-------|-----------|------------|-----------|
| Continuous Ramp (#3) | Piecewise-constant θ = n/N updated once per step | Continuous linear-in-time Ramp evaluated at the stage clock | Removes step-quantized forcing; sub-step-accurate and sign-agnostic for backward legs (`Ramp.reversed()`). |
| `stop_best` (#4) | Divergent fixed-point returned the last (diverged) iterate | Returns the argmin-residual iterate | A diverged final iterate is never the intended output; best-so-far is the only defensible return. |
| TimeAverage drops Smagorinsky (#5) | The "linear" TimeAverage twin kept Smagorinsky (a nonlinear closure) | Default `fr.terms.linear` filter drops all nonlinear terms incl. Smagorinsky | The averaging twin must be the linearized dynamics; keeping a nonlinear closure was an old inconsistency. |
| Sadourny csqr field (#6) | Scalar csqr used in `h_full`, field elsewhere | csqr field everywhere incl. `h_full` | Old mixing silently broke variable-depth runs; the always-field rule is the correct fix (old bug). |
| epot N²=0 hinted error (#9) | Silent formula switch to the unstratified form | Hinted `ValueError` (1/N² weight needs nonzero stratification); unstratified form is user algebra | A silent formula switch hides a modeling choice; an explicit error surfaces it. |
| Nyquist relocation (#10) | Nyquist zeroed at a transform call-site | Zeroed inside the em.q/p eigenmode construction | Single home for the structural zeros; the eigenmode carries its own well-formedness. |
| Seeded-random ICs (#11) | Bitwise-reproducible global draw | Per-DOF `fold_in(key, global_index)` (not bitwise vs old) | Decomposition-invariant randomness (same field at any device count); worth losing old-bitwise reproducibility. |
| float64 clock (#12) | float32 forcing phases | float64 clock leaves | Removes phase drift in long forced runs; matches the global x64 flag. |
| netcdf → zarr (#13) | netCDF writer | tensorstore-zarr sink; `every()` includes step 0; triggers snap up | Chunked, decomposition-friendly, restart-consistent output. |
| Backward via signed dt (#15) | dedicated `run_backward` | negative dt through the same `model.run` | One code path; sign-agnostic step/trigger lowering (§8.7). |
| Order ≥ 3 AB warm-up (#16) | eps applied through warm-up rows | Textbook AB2 `[3/2,−1/2]` warm-up; eps is order-2-only | Signed off §8.6-6; deliberate one-step startup delta, tolerance-based. |
| Background not Ro-scaled (#19) | Scaling factor Ro multiplied the background too | Background advection carries no Ro factor | The prescribed mean flow is O(1), not an O(Ro) perturbation; scaling it was wrong. |
| Kelvin IC exact mode (#20) | Approximate V=0 ansatz | Exact labeled channel eigenmode | The exact mode is a true step-eigenfunction; the approximation seeded spurious transients. |
| Single-wave normalization (#21) | L2-norm = 1 | Unit velocity envelope | Amplitude-1 is the physically legible normalization for a single mode. |
| Relaxation rate=1/tau (#22) | Constructor took the timescale τ | Constructor takes the rate r = 1/τ | Rate composes linearly and is the natural leaf for `update_parameters` sweeps. |
| Wave-maker param names (#23) | old kwargs | `position/width/frequency/amplitude/variable` (Gaussian); `k/position/width/amplitude/s` (Polarized) | Consistent dict-of-coordinate spellings across makers. |

## Flags for owner — all resolved (2026-07-11)

1. **`p = φ/stage_dt` (#2)** — wired in
   `nonhydro2/modules/core.py::_project` (`"p": p / ctx.stage_dt`;
   velocities still subtract the raw-potential gradient, dynamics
   unchanged); pinned by the Sketch-A regression (row 2).
2. **`rest="zero"` runtime application (#7)** — wired centrally in
   `transforms/base.py::_apply_rest` (fill-only completion after the
   payload; leaf-level, algebra-safe) and covered (row 7).

## Appendix A — subtle-hole sketches (for a stronger pass)

### Sketch A — project-state ≡ project-tendency exact-equivalence (items 1 + 2) — IMPLEMENTED

Implemented as
`nonhydro2/test_pressure.py::test_project_state_equals_project_tendency`
(AB1, periodic grid, divergence-free IC, forward + backward dt): the
production one-step u₁ matches the hand-built project-the-tendency
u₁′ per component and the stored `p` matches ψ (rel L2 < 1e-12;
measured ≤ 3e-15). The normalization is wired in
`nonhydro2/modules/core.py::_project` (`"p": p / ctx.stage_dt`).

### Sketch B — one-period discrete-eigenmode dispersion regression (item 17)

- **Model**: nonhydro2 (or shallowwater2) *linear* model on a
  periodic grid; seed a single discrete eigenmode `em.mode(s)` for a
  resolved wavevector k. Use a low-dissipation stepper (RK3) to keep
  the amplitude-error budget small.
- **Discrete frequency**: take the semi-discrete ω from
  `em.omega_at(k)`, then map it through
  `stepper.time_discretization_effect(ω)` to the fully-discrete ω_d
  (the AB/RK dispersion). Period T_d = 2π/Re(ω_d); n_steps =
  `round(T_d/dt)`.
- **Assertion**: after n_steps, `relative_l2(state, z0)` is small —
  the wave returns to itself after one discrete period. Realistic
  tolerance **1e-3 … 1e-4** on the relative L2, driven by (a) the
  period-rounding residual |n_steps·dt − T_d| and (b) the stepper's
  amplitude drift over one period; tighten by choosing dt so T_d/dt
  is near-integer and k well-resolved.
- **Old-stack tie-in**: this is the §8.1 "keep the old 'wave
  propagates at discrete ω' run as the 2.7 regression" obligation;
  the intended reference is the old stack's one-period propagation.
- **Precedent**: `nonhydro2/test_initial_conditions.py::test_kelvin_wave_phase_rotates_in_the_channel`
  does a 20-step phase-rotation check (<1e-4) — the same shape at
  sub-period scale; extend it to a full period with the discrete ω_d.
- **Why subtle**: the semi-discrete-vs-fully-discrete ω choice, the
  period-rounding residual, and the phase-vs-amplitude tolerance
  split all require numerics judgment.
