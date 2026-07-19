---
status: active (S1 in implementation)
date: 2026-07-19
---

# sw2 physical-components flip — campaign plan

Ruling (c) of [`../../decisions/physical_state_components.md`](../../decisions/physical_state_components.md)
(owner-ratified 2026-07-19): move the shallowwater2 spherical
prognostics from the chart convention (`u = dlon/dt`, `v = dphi/dt`,
contravariant) to **physical m/s components** — the NEMO/MITgcm
curvilinear standard — completing invariant (a) ("state components
are physical on every grid") across all three model packages.
Retires `State.u_physical` / `State.v_physical`, brings physical IC
input to the sphere, adds `state.chart` (ruling (d)), and re-proves
the energy-exactness gates.

## 1. Scope census (verified in code, 2026-07-19)

Chart-convention surfaces (all must flip):

| surface | file | chart-path role |
| --- | --- | --- |
| gravity term | `shallowwater2/modules/core.py` (`gravity`, chart branch) | grad → raise_index on p; flux-form div of `c² u^i` |
| Sadourny advection | `shallowwater2/modules/sadourny.py` (`_advect_chart`) | full vector-invariant metric path |
| conserving Coriolis | `shallowwater2/modules/coriolis.py` (`_conserving_chart`) | f-part PV flux, raised |
| linear chart rotation | `model/modules/coriolis.py` (`chart_rotation`) | shared single source of truth; subtracted exactly by `CoriolisEnergyCorrection` |
| state diagnostics | `shallowwater2/state.py` (`rel_vort`, `divergence`) | lower_index → curl; flux div |
| conversion properties | `shallowwater2/state.py` (`u_physical`, `v_physical`) | **retired** |
| bound diagnostics | `shallowwater2/diagnostics.py` (`ekin`, `ekin_full`) | metric quadratics |
| docs-in-source | `shallowwater2/modules/core.py`, `sadourny.py`, `modules/coriolis.py`, `model.py`, `state.py` docstrings | recorded convention text |

**Confirmed untouched** (do not edit; audit only):

- Eigen machinery, transforms, initial_conditions: the analytic
  `Eigenmodes` requires a fully periodic flat 2-D grid, the channel
  engine a walled flat grid — neither runs on a chart, and on flat
  grids physical ≡ chart-native. `energy.py` weights serve only that
  flat/channel machinery.
- Immersed paths (`_advect_immersed`, `_gravity_immersed`,
  `immersed_weighting.py`): flat-only by taught error (chart+immersed
  refused at bind, both bind sites).
- The flat branches of every term: taken verbatim (the binding
  bitwise results-neutrality gate).
- io/writer: no conversion layer anywhere (invariant (a): checkpoints
  hold physical fields verbatim). The `FieldDeclaration` units
  (`"m/s"`) become *true* on the sphere — today they are a recorded
  wart (`dlon/dt` is 1/s).
- Old stack (`fridom/shallowwater`): out of scope (cutover plan).

## 2. Design decisions

### D1 — seam conversion, not operator rewrite (the load-bearing one)

The dispatch kinds (`grad` / `div` / `curl` / `raise_index` /
`lower_index`) remain **variance-native** (contravariant in,
covariant out, etc.) — they are differential-geometric operators and
the registry seam stays mathematically canonical. The *state
convention* flips at the term boundaries:

- **entry**: `u^i = U_i / sqrt(g_ii)` (sealed divide, D2), on each
  component's **own** staggered bare space (the `u_physical`
  precedent placement, D6);
- **exit**: `dU_i = sqrt(g_ii) * du^i_contra` (plain multiply; the
  metric is static, so d/dt commutes with the rescale).

Everything between entry and exit runs **verbatim** — the existing
spellings, placements, and wall-tag machinery are not touched.

Why not fold the factors into the spellings (the "native physical"
NEMO spelling `sqrt(g)/sqrt(g_ii) = e_j` scale factors): the
conversions are pointwise but the spellings are full of
interpolations, and pointwise rescales do **not** commute with
`.to` — every fold is a per-site re-derivation with a fresh
energy-exactness proof obligation. The seam conversion is instead a
similarity transform `T_phys = S ∘ T_con ∘ S⁻¹` with
`S = diag(sqrt(g_ii))` static and pointwise, so every existing proof
conjugates:

- M-skewness: `T_con` skew under `M_con = diag(sqrt_g h̄ g_ii ...)`
  ⇒ `T_phys` skew under `M_phys = S⁻¹ M_con S⁻¹ = diag(sqrt_g h̄ ...)`
  — the conserved functional in physical components,
  `E = Σ sqrt_g h̄ U²/2 + Σ sqrt_g p²/2` (exactly what the flipped
  `ekin_full` computes, D4).
- Telescoping/adjoint identities: internal, untouched.

Exactness cost: one `sqrt(g_ii)·(x/sqrt(g_ii))` round trip per
component per term — the same ~1-ulp-per-node class as the recorded
`g_ii g^ii` round trip of the current raise/lower path. The gates
stay in their current tolerance class (§4 rule).

On a non-orthogonal chart `U_i = sqrt(g_ii) u^i` are the standard
curvilinear "physical components" (unit-tangent frame); the diagonal
conversion stays well-defined, and the energy spellings were already
diagonal-only before this campaign — nothing regresses.

### D2 — sealed conversion divides

On walled charts the materialized metric is an **exact 0** in
never-valid padding, so the entry divide must use the double-`where`
seal (`bad = den == 0; where(bad, 0, num/where(bad, 1, den))`) —
both forward (a bare divide plants pad-inf, the
`mapped_chunk_nonfinite` failure class) and reverse (the masked 0/0
VJP poison). One shared helper; the exit multiply is safe bare.

### D3 — `chart_rotation` flips internally

It stays the single source of truth: it now reads **physical**
`state["u"]`/`state["v"]`, converts on entry, rescales on exit, and
returns physical increments. `RotationCoriolis` and the
energy-correction subtraction keep delegating verbatim, so the two
can never drift. Its docstring contract line ("contravariant
components") flips. This is a shared `fr.model.modules` surface;
invariant (a) makes physical the correct contract for every future
chart consumer.

### D4 — diagnostics simplify

- `ekin`: the chart branch **collapses** — `0.5 (U² + V²)` at
  centre is the flat spelling. (The old chart branch multiplied the
  centre-interpolated contravariant by the centre metric root — a
  different placement from `u_physical`; the flip removes the
  approximation rather than preserving it.)
- `ekin_full`: the chart branch drops the `g_ii` factors and keeps
  the `sqrt_g` Jacobian placement:
  `E = (mean(sqrt_g h̄ U²) + mean(sqrt_g h̄ V²)) / (2 sqrt_g)`.
- `epot`/`epot_full`/`thickness`/mass: no velocity dependence,
  untouched.

### D5 — `state.chart` (ruling (d)) and retirement

New `shallowwater2/chart.py` with the per-package derivation hook
(the `nonhydro2/chart.py` + `spatial.fields.ChartView` pattern,
CS-14-safe): `chart["u"]` / `chart["v"]` are the **coordinate
velocities** `dlon/dt = U/sqrt(g_lonlon)` etc. (sealed divide) on a
chart grid, the identity on flat grids and for `p`;
`State.chart` returns `ChartView(self, hook, ("u", "v"))`.
`u_physical` / `v_physical` are deleted (no deprecation shim — the
decision record retires them, pre-1.0 new stack).

### D6 — metric placement

Conversions evaluate `sqrt(g_ii)` on the converted component's own
bare staggered space via `grid.metric` (derived per call, never
cached) — the `u_physical` precedent placement, and the placement
under which the conjugation argument in D1 is exact per node.

## 3. Touch list (S1)

1. **`src/fridom/shallowwater2/chart.py`** (new): sealed-divide
   helper + `to_contravariant(field, axis)` / entry-exit helpers +
   `chart_component(state, name)` hook. Module docstring records the
   convention.
2. **`state.py`**: delete `u_physical`/`v_physical`; add `chart`
   property; entry conversions in `rel_vort` / `divergence` chart
   branches; module + class docstrings flip.
3. **`modules/core.py`**: `gravity` chart branch — convert `u`, `v`
   at entry (the flux and the raised tendency both consume the
   contravariant), rescale `du`/`dv` at exit (`dp` needs none);
   docstrings ("Velocity convention" section now records the
   physical convention, pointing at the decision record).
4. **`model/modules/coriolis.py`**: `chart_rotation` per D3;
   `RotationCoriolis` docstring.
5. **`modules/sadourny.py`**: `_advect_chart` — entry conversions
   once, all internal uses (fluxes, lower/curl, corner fluxes, ekin
   quadratics with `g_ii` unchanged **on the contravariant
   intermediates**) verbatim, exit rescale after `raise_index`;
   module docstring chart section (conserved functional restated in
   physical components).
6. **`modules/coriolis.py`** (sw2): `_conserving_chart` — same
   entry/exit pattern; module docstring chart paragraph.
7. **`diagnostics.py`**: per D4.
8. **`model.py`**: preset docstring convention text.
9. Sweep: `grep -rn "contravariant\|u_physical\|v_physical\|dlon"
   src/fridom/shallowwater2 src/fridom/model` — every survivor is
   either inside a chart-path *intermediate* comment (fine) or a
   miss.

## 4. Gates (S1, all must pass before merge)

- **Flat results-neutrality (bitwise)**: flat grids take the flat
  branches verbatim — zero diff by construction; the suite pins it.
- **Identity-chart flat limit**:
  `tests/validation/test_spherical_shallowwater.py::test_identity_chart_run_is_bitwise_flat`
  — under `disable_jit` the conversions multiply/divide by exact
  1.0s, which are bitwise identities, so the op-by-op bitwise
  assertion **must stay green** (the jitted allclose tail likewise).
- **Energy exactness re-proven**: the machine-zero semi-discrete
  rate gates (`test_semi_discrete_energy_rate_is_machine_zero_on_the_sphere`,
  the etot_full drift and mass gates, the coriolis route-A≡B gate)
  at their **current tolerance class**. Hard rule: if any gate needs
  loosening beyond ~2× its current bound, STOP — that falsifies the
  conjugation argument (a conversion placed on the wrong space or on
  the wrong side of an interpolation), it is not a tolerance to
  negotiate.
- **Hand-built metric-form tests** (`test_spherical.py`): rebuild
  the hand forms with the same entry/exit conversions.
- **IC flip**: TC2 solid body becomes the *physical*
  `u = U0 cos(lat)` (was: constant contravariant `U0/a`); steady +
  convergence gates unchanged.
- **`state.chart`**: new tests — sphere round trip
  (`chart.u == u/sqrt(g_lonlon)` to machine precision, identity on
  flat), read-only refusal, destructuring order; `u_physical` gone.
- **Autodiff**: `test_spherical_autodiff.py` — grad through a chart
  run now w.r.t. a *physical* IC; finite and FD-matched (the sealed
  entry divides are the hazard, D2).
- **Mirrored tests** for every edited file (AGENTS.md map), incl.
  `tests/model/modules/test_coriolis.py`, plus the model smoke file
  `tests/nonhydro/test_linear_model.py` (shared-machinery edit);
  `uv run ruff check src tests` at zero.

## 5. Stages

- **S1 — the flip** (branch `feat/sw2-physical-flip`, own worktree):
  touch list §3 + test updates §4, merge gate per AGENTS.md.
- **S2 — closure** (design-only, direct to dev): move the roadmap
  entry `open.md → done.md`, flip ruling (c)'s status line in the
  decision record, note the shipped surface here, archive this plan
  as shipped.

## 6. Out of scope (recorded)

- sw2 mapped+immersed (its own open roadmap leaf; both bind-site
  taught errors unchanged by this campaign).
- Non-orthogonal-chart energy spellings (diagonal-only before and
  after; D1 note).
- Old-stack `shallowwater`.
- Any perf work: the chart path gains 4–6 pointwise ops per term;
  sphere runs are not in the step-guard suite and agents never
  submit GPU jobs.
